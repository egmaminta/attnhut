"""MiniMax Sparse Attention (MiniMax M2/M3).

A cheap index branch scores key blocks with a single shared index key and one
index query per GQA group, max-pools the token scores into block scores, and
keeps the top-k blocks per group. The main attention then runs over the selected
blocks only. Block-level selection is what separates it from the token-level
DeepSeek indexer.

Masked-dense reference. The selection is exact, the speedup is not (that needs a
block-gather kernel).
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rope import apply_rope, build_rope_cache
from .utils import causal_mask, neg_value, repeat_kv


class MiniMaxSparseAttention(nn.Module):
    """Top-k block selection on a GQA backbone.

    Args:
        dim: Model dimension. Must be divisible by num_heads.
        num_heads: Number of query heads.
        num_kv_groups: Number of key/value groups. Must divide num_heads. Query
            heads in a group share one block selection.
        block_size: Tokens per key block.
        top_k: Blocks kept per query.
        index_dim: Width of the index branch. Defaults to the head dim.
        causal: If True, attention and selection are restricted to the past.
        use_rope: Apply rotary embeddings to the main and index projections.
        rope_theta: RoPE base frequency.
        force_current_block: Always keep a query's own block, so no softmax row
            is ever empty.
        bias: Whether the linear projections carry a bias term.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_groups: int,
        block_size: int = 64,
        top_k: int = 8,
        index_dim: int | None = None,
        causal: bool = True,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        force_current_block: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_groups != 0:
            raise ValueError("num_heads must be divisible by num_kv_groups")
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.index_dim = index_dim or head_dim
        self.block_size = block_size
        self.top_k = top_k
        self.causal = causal
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.force_current_block = force_current_block
        if use_rope and (head_dim % 2 or self.index_dim % 2):
            raise ValueError(
                "head_dim and index_dim must be even when use_rope"
            )
        kv = num_kv_groups * head_dim
        self.q = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, kv, bias=bias)
        self.v = nn.Linear(dim, kv, bias=bias)
        self.idx_q = nn.Linear(dim, num_kv_groups * self.index_dim, bias=bias)
        self.idx_k = nn.Linear(dim, self.index_dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

    def _select_blocks(self, qi: Tensor, ki: Tensor) -> tuple[Tensor, Tensor]:
        """Score key blocks and pick the top-k per group.

        Args:
            qi: Index queries of shape (batch, groups, seq, index_dim).
            ki: Shared index key of shape (batch, 1, seq, index_dim).

        Returns:
            A pair (block_keep, block_scores), both of shape
            (batch, groups, seq, num_blocks). block_keep is boolean. Causal
            masking is applied to the token scores before pooling so a future
            block can never win the max for an earlier query.
        """
        b, h, t, _ = qi.shape
        bs, device = self.block_size, qi.device
        neg = neg_value(torch.float32)
        scores = (
            torch.matmul(qi.float(), ki.float().transpose(-2, -1))
            * self.index_dim**-0.5
        )
        valid = causal_mask(t, t, device) if self.causal else None
        if valid is not None:
            scores = scores.masked_fill(~valid, neg)
        pad = (-t) % bs
        if pad:
            scores = F.pad(scores, (0, pad), value=neg)
        nb = (t + pad) // bs
        block_scores = scores.view(b, h, t, nb, bs).amax(dim=-1)

        topi = block_scores.topk(max(1, min(self.top_k, nb)), dim=-1).indices
        keep = torch.zeros(b, h, t, nb, dtype=torch.bool, device=device)
        keep.scatter_(-1, topi, True)
        if self.causal:
            start = (torch.arange(nb, device=device) * bs).view(1, 1, 1, nb)
            keep &= start <= torch.arange(t, device=device).view(1, 1, t, 1)
        if self.force_current_block:
            cur = (torch.arange(t, device=device) // bs).clamp_max(nb - 1)
            keep |= F.one_hot(cur, nb).bool().view(1, 1, t, nb)
        return keep, block_scores

    def forward(
        self, x: Tensor, return_aux: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Run block-sparse attention over the sequence.

        Args:
            x: Input of shape (batch, seq, dim).
            return_aux: If True, also return the index scores and attention
                weights needed by msa_index_aux_loss.

        Returns:
            Output of shape (batch, seq, dim). If return_aux is True, a pair
            (output, aux) where aux has block_scores (batch, groups, seq,
            num_blocks) and attn_weights (batch, num_heads, seq, seq).
        """
        b, t, _ = x.shape
        h = self.num_kv_groups
        g = self.num_heads // h
        d, di, bs = self.head_dim, self.index_dim, self.block_size
        device = x.device

        q = self.q(x).view(b, t, self.num_heads, d).transpose(1, 2)
        k = self.k(x).view(b, t, h, d).transpose(1, 2)
        v = self.v(x).view(b, t, h, d).transpose(1, 2)
        qi = self.idx_q(x).view(b, t, h, di).transpose(1, 2)
        ki = self.idx_k(x).view(b, t, 1, di).transpose(1, 2)

        if self.use_rope:
            pos = torch.arange(t, device=device)
            cos, sin = build_rope_cache(pos, d, self.rope_theta, x.dtype)
            cos_i, sin_i = build_rope_cache(pos, di, self.rope_theta, x.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            qi, ki = apply_rope(qi, cos_i, sin_i), apply_rope(ki, cos_i, sin_i)

        block_keep, block_scores = self._select_blocks(qi, ki)
        keep = block_keep.repeat_interleave(bs, dim=-1)[..., :t]
        keep = keep.repeat_interleave(g, dim=1)
        if self.causal:
            keep = keep & causal_mask(t, t, device)

        kr, vr = repeat_kv(k, g), repeat_kv(v, g)
        logits = torch.matmul(q, kr.transpose(-2, -1)) * d**-0.5
        logits = logits.masked_fill(~keep, neg_value(q.dtype))
        empty = ~keep.any(dim=-1, keepdim=True)
        logits = logits.masked_fill(empty, 0.0)
        attn = torch.softmax(logits.float(), dim=-1).to(q.dtype)
        attn = attn.masked_fill(empty, 0.0)
        out = torch.matmul(attn, vr).transpose(1, 2).reshape(b, t, -1)
        out = self.proj(out)
        if return_aux:
            return out, {"block_scores": block_scores, "attn_weights": attn}
        return out


def msa_index_aux_loss(
    block_scores: Tensor,
    attn_weights: Tensor,
    block_size: int,
    group_size: int,
) -> Tensor:
    """Train the index branch to rank blocks by realized attention mass.

    Hard top-k is not differentiable, so without this the index projections get
    no gradient. The target pools the main attention weights into per-block
    mass, averages over the heads of each group, and the index scores are fit to
    it with a cross entropy.

    Args:
        block_scores: Index branch scores of shape (batch, groups, seq,
            num_blocks), from the aux dict.
        attn_weights: Main attention weights of shape (batch, num_heads, seq,
            seq), from the aux dict.
        block_size: Tokens per block, must match the forward pass.
        group_size: Query heads per group, num_heads // num_kv_groups.

    Returns:
        A scalar loss.
    """
    b, heads, t, k = attn_weights.shape
    nb = block_scores.shape[-1]
    mass = F.pad(attn_weights.detach(), (0, nb * block_size - k))
    mass = mass.view(b, heads, t, nb, block_size).sum(-1)
    mass = mass.view(b, heads // group_size, group_size, t, nb).mean(2)
    target = mass / mass.sum(-1, keepdim=True).clamp_min(1e-9)
    log_p = torch.log_softmax(block_scores.to(torch.float32), dim=-1)
    return -(target * log_p).sum(-1).mean()
