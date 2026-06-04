"""DeepSeek-V4 hybrid attention: Heavily Compressed and Compressed Sparse.

Both compress the KV cache by pooling every few tokens into one entry with
learned, position-biased softmax weights, then run shared-KV MQA over the
compressed entries plus a short uncompressed sliding window, with a learnable
attention sink. HCA compresses hard and keeps everything. CSA compresses lightly
with overlap and then selects the top-k entries with a DeepSeek lightning
indexer.

Partial RoPE (V4 applies RoPE to the last 64 dims with a negative-position
output correction) is omitted here for clarity.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .utils import neg_value, rms_norm


def _compress(c: Tensor, z: Tensor, bias: Tensor, rate: int) -> Tensor:
    """Pool every rate tokens into one entry, softmax over positions.

    Args:
        c: Value tensor of shape (batch, seq, dim_c).
        z: Pooling weights of shape (batch, seq, dim_c).
        bias: Per-position learned bias of shape (rate, dim_c).
        rate: Tokens per compressed entry.

    Returns:
        Compressed tensor of shape (batch, ceil(seq / rate), dim_c).
    """
    b, n, d = c.shape
    pad = (-n) % rate
    if pad:
        c = F.pad(c, (0, 0, 0, pad))
        z = F.pad(z, (0, 0, 0, pad), value=neg_value(z.dtype))
    nb = (n + pad) // rate
    c = c.view(b, nb, rate, d)
    z = z.view(b, nb, rate, d) + bias
    s = torch.softmax(z.float(), dim=2).to(c.dtype)
    return (s * c).sum(dim=2)


def _compress_overlap(
    ca: Tensor,
    cb: Tensor,
    za: Tensor,
    zb: Tensor,
    ba: Tensor,
    bb: Tensor,
    rate: int,
) -> Tensor:
    """CSA compression. Entry i pools its own block and the previous one.

    The overlap gives each entry a wider receptive field than non-overlapped
    pooling at the same compression rate. Entry 0 has no previous block, so that
    half is masked out.

    Args:
        ca: Value tensor for the current block, shape (batch, seq, dim_c).
        cb: Value tensor for the previous block, shape (batch, seq, dim_c).
        za: Pooling weights for the current block.
        zb: Pooling weights for the previous block.
        ba: Per-position bias for the current block, shape (rate, dim_c).
        bb: Per-position bias for the previous block, shape (rate, dim_c).
        rate: Tokens per block.

    Returns:
        Compressed tensor of shape (batch, ceil(seq / rate), dim_c).
    """
    b, n, d = ca.shape
    pad = (-n) % rate
    if pad:
        ca, cb = F.pad(ca, (0, 0, 0, pad)), F.pad(cb, (0, 0, 0, pad))
        za = F.pad(za, (0, 0, 0, pad), value=neg_value(za.dtype))
        zb = F.pad(zb, (0, 0, 0, pad), value=neg_value(zb.dtype))
    nb = (n + pad) // rate
    a_c = ca.view(b, nb, rate, d)
    a_z = za.view(b, nb, rate, d) + ba
    zeros = ca.new_zeros(b, 1, rate, d)
    fill = za.new_full((b, 1, rate, d), neg_value(za.dtype))
    b_c = torch.cat([zeros, cb.view(b, nb, rate, d)[:, :-1]], dim=1)
    b_z = torch.cat([fill, zb.view(b, nb, rate, d)[:, :-1] + bb], dim=1)
    c = torch.cat([a_c, b_c], dim=2)
    z = torch.cat([a_z, b_z], dim=2)
    s = torch.softmax(z.float(), dim=2).to(c.dtype)
    return (s * c).sum(dim=2)


def _mqa_core(
    q: Tensor, kv: Tensor, keep: Tensor, scale: float, sink: Tensor
) -> Tensor:
    """Shared-KV MQA with a learnable per-head attention sink.

    Every kv entry is both key and value. The sink is an extra logit added
    to the softmax denominator only, so a head can send mass nowhere. It also
    keeps rows with no visible key finite.

    Args:
        q: Queries of shape (batch, heads, seq, dim_c).
        kv: Shared entries of shape (batch, num_entries, dim_c).
        keep: Boolean mask broadcastable to (batch, heads, seq, num_entries).
        scale: Softmax temperature.
        sink: Per-head sink logit of shape (heads,).

    Returns:
        Attention output of shape (batch, heads, seq, dim_c).
    """
    b, h, n, _ = q.shape
    logits = torch.einsum("bhnd,bsd->bhns", q, kv) * scale
    logits = logits.masked_fill(~keep, neg_value(q.dtype))
    sink_col = sink.view(1, h, 1, 1).expand(b, h, n, 1).to(logits.dtype)
    logits = torch.cat([logits, sink_col], dim=-1)
    attn = torch.softmax(logits.float(), dim=-1).to(q.dtype)
    return torch.einsum("bhns,bsd->bhnd", attn[..., :-1], kv)


def _window_keep(n: int, window: int, device: torch.device) -> Tensor:
    """Boolean (n, n), True where key s is in the recent window of t."""
    t = torch.arange(n, device=device).unsqueeze(1)
    s = torch.arange(n, device=device).unsqueeze(0)
    return (s <= t) & (s > t - window)


class GroupedOutputProjection(nn.Module):
    """Project per-head outputs to the model dim, in head groups.

    The combined head width can be large, so heads are split into groups, each
    group is projected to group_dim, and the concatenation is projected to dim.
    Cheaper than one dense projection over all heads at once.

    Args:
        num_heads: Number of heads feeding the projection.
        head_dim: Width of each head.
        num_groups: Number of groups. Must divide num_heads.
        dim: Output model dimension.
        group_dim: Per-group projection width. Defaults to head_dim.
        bias: Whether the output projection carries a bias term.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_groups: int,
        dim: int,
        group_dim: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if num_heads % num_groups != 0:
            raise ValueError("num_heads must be divisible by num_groups")
        self.num_groups = num_groups
        group_dim = group_dim or head_dim
        in_g = (num_heads // num_groups) * head_dim
        self.w = nn.Parameter(torch.empty(num_groups, in_g, group_dim))
        nn.init.xavier_uniform_(self.w)
        self.out = nn.Linear(num_groups * group_dim, dim, bias=bias)

    def forward(self, o: Tensor) -> Tensor:
        """Project head outputs.

        Args:
            o: Head outputs of shape (batch, seq, num_heads, head_dim).

        Returns:
            Tensor of shape (batch, seq, dim).
        """
        b, n, _, _ = o.shape
        o = o.reshape(b, n, self.num_groups, -1)
        o = torch.einsum("bngi,gio->bngo", o, self.w)
        return self.out(o.reshape(b, n, -1))


class HeavilyCompressedAttention(nn.Module):
    """Heavy non-overlapped KV compression with shared-KV MQA, no selection.

    Args:
        dim: Model dimension.
        num_heads: Number of query heads.
        compression_rate: Tokens pooled into one compressed entry.
        kv_dim: Width of a compressed entry. Defaults to the head dim.
        q_lora_rank: Width of the query latent. Defaults to dim // 2.
        num_groups: Groups for the output projection. Defaults to num_heads.
        window: Recent uncompressed tokens each query also attends.
        bias: Whether the linear projections carry a bias term.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        compression_rate: int = 16,
        kv_dim: int | None = None,
        q_lora_rank: int | None = None,
        num_groups: int | None = None,
        window: int = 64,
        bias: bool = False,
    ) -> None:
        super().__init__()
        c = kv_dim or dim // num_heads
        self.num_heads = num_heads
        self.kv_dim = c
        self.rate = compression_rate
        self.window = window
        q_lora_rank = q_lora_rank or dim // 2
        self.kv = nn.Linear(dim, c, bias=bias)
        self.z = nn.Linear(dim, c, bias=bias)
        self.pos_bias = nn.Parameter(torch.zeros(compression_rate, c))
        self.q_down = nn.Linear(dim, q_lora_rank, bias=bias)
        self.q_up = nn.Linear(q_lora_rank, num_heads * c, bias=bias)
        self.q_norm = nn.Parameter(torch.ones(c))
        self.kv_norm = nn.Parameter(torch.ones(c))
        self.sink = nn.Parameter(torch.zeros(num_heads))
        self.proj = GroupedOutputProjection(
            num_heads, c, num_groups or num_heads, dim, bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run compressed attention over the sequence.

        Args:
            x: Input of shape (batch, seq, dim).

        Returns:
            Output of shape (batch, seq, dim).
        """
        b, n, _ = x.shape
        c = self.kv_dim
        comp = _compress(self.kv(x), self.z(x), self.pos_bias, self.rate)
        nb = comp.shape[1]
        kv = torch.cat([comp, self.kv(x)], dim=1)
        kv = rms_norm(kv, self.kv_norm)

        q = self.q_up(self.q_down(x)).view(b, n, self.num_heads, c)
        q = rms_norm(q, self.q_norm).transpose(1, 2)

        t = torch.arange(n, device=x.device).unsqueeze(1)
        block = torch.arange(nb, device=x.device).unsqueeze(0)
        comp_keep = (block + 1) * self.rate <= t
        keep = torch.cat(
            [comp_keep, _window_keep(n, self.window, x.device)], dim=1
        )

        out = _mqa_core(q, kv, keep[None, None], c**-0.5, self.sink)
        return self.proj(out.transpose(1, 2))


class CompressedSparseAttention(nn.Module):
    """Light overlapped compression plus DSA top-k entry selection.

    Args:
        dim: Model dimension.
        num_heads: Number of query heads.
        compression_rate: Tokens pooled into one compressed entry.
        top_k: Compressed entries kept per query.
        kv_dim: Width of a compressed entry. Defaults to the head dim.
        q_lora_rank: Width of the query latent. Defaults to dim // 2.
        num_index_heads: Heads in the lightning indexer.
        index_dim: Width of each indexer head.
        num_groups: Groups for the output projection. Defaults to num_heads.
        window: Recent uncompressed tokens each query also attends.
        bias: Whether the linear projections carry a bias term.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        compression_rate: int = 4,
        top_k: int = 16,
        kv_dim: int | None = None,
        q_lora_rank: int | None = None,
        num_index_heads: int = 4,
        index_dim: int = 64,
        num_groups: int | None = None,
        window: int = 64,
        bias: bool = False,
    ) -> None:
        super().__init__()
        c = kv_dim or dim // num_heads
        self.num_heads = num_heads
        self.kv_dim = c
        self.rate = compression_rate
        self.top_k = top_k
        self.window = window
        self.num_index_heads = num_index_heads
        self.index_dim = index_dim
        q_lora_rank = q_lora_rank or dim // 2

        self.kv_a = nn.Linear(dim, c, bias=bias)
        self.kv_b = nn.Linear(dim, c, bias=bias)
        self.z_a = nn.Linear(dim, c, bias=bias)
        self.z_b = nn.Linear(dim, c, bias=bias)
        self.bias_a = nn.Parameter(torch.zeros(compression_rate, c))
        self.bias_b = nn.Parameter(torch.zeros(compression_rate, c))

        self.q_down = nn.Linear(dim, q_lora_rank, bias=bias)
        self.q_up = nn.Linear(q_lora_rank, num_heads * c, bias=bias)
        self.q_norm = nn.Parameter(torch.ones(c))
        self.kv_norm = nn.Parameter(torch.ones(c))
        self.sink = nn.Parameter(torch.zeros(num_heads))

        self.idx_q = nn.Linear(
            q_lora_rank, num_index_heads * index_dim, bias=bias
        )
        self.idx_k = nn.Linear(c, index_dim, bias=bias)
        self.idx_w = nn.Linear(dim, num_index_heads, bias=bias)
        self.proj = GroupedOutputProjection(
            num_heads, c, num_groups or num_heads, dim, bias=bias
        )

    def forward(
        self, x: Tensor, return_aux: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Run compressed sparse attention over the sequence.

        Args:
            x: Input of shape (batch, seq, dim).
            return_aux: If True, also return the index scores and the selection
                mask over compressed entries.

        Returns:
            Output of shape (batch, seq, dim). If return_aux is True, a pair
            (output, aux) where aux has index_scores (batch, seq, num_entries)
            and selected (batch, seq, num_entries) boolean.
        """
        b, n, _ = x.shape
        c = self.kv_dim
        comp = _compress_overlap(
            self.kv_a(x),
            self.kv_b(x),
            self.z_a(x),
            self.z_b(x),
            self.bias_a,
            self.bias_b,
            self.rate,
        )
        nb = comp.shape[1]
        kv = rms_norm(torch.cat([comp, self.kv_a(x)], dim=1), self.kv_norm)

        c_q = self.q_down(x)
        q = self.q_up(c_q).view(b, n, self.num_heads, c)
        q = rms_norm(q, self.q_norm).transpose(1, 2)

        # Lightning indexer over the compressed entries, then top-k per query.
        qi = self.idx_q(c_q).view(b, n, self.num_index_heads, self.index_dim)
        ki = self.idx_k(comp)
        wi = self.idx_w(x)
        dots = torch.einsum("bnhd,bsd->bnhs", qi, ki).relu()
        scores = torch.einsum("bnh,bnhs->bns", wi, dots)

        t = torch.arange(n, device=x.device).unsqueeze(1)
        block = torch.arange(nb, device=x.device).unsqueeze(0)
        comp_visible = (block + 1) * self.rate <= t
        scores = scores.masked_fill(~comp_visible, neg_value(scores.dtype))
        topi = scores.topk(min(self.top_k, nb), dim=-1).indices
        comp_keep = torch.zeros(b, n, nb, dtype=torch.bool, device=x.device)
        comp_keep.scatter_(-1, topi, True)
        comp_keep &= comp_visible

        win = _window_keep(n, self.window, x.device).expand(b, n, n)
        keep = torch.cat([comp_keep, win], dim=2)
        out = _mqa_core(q, kv, keep[:, None], c**-0.5, self.sink)
        out = self.proj(out.transpose(1, 2))
        if return_aux:
            return out, {"index_scores": scores, "selected": comp_keep}
        return out
