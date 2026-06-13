"""MiniMax Sparse Attention (Lai et al., 2026).

A blockwise sparse attention on a GQA backbone, the selector behind MiniMax-M3.
A lightweight index branch scores key blocks with a single shared index key and
one index query per GQA group, max pools the token scores into block scores, and
keeps the top-k blocks per group. The local block of each query is always kept.
The main branch then runs exact attention over the selected blocks only. Per
group block-level selection is what separates it from the token-level DeepSeek
indexer.

Both branches apply QK norm to their query and key and use partial RoPE, and the
index branch input is detached from the backbone, so the auxiliary KL loss trains
the index projections alone and never reaches the rest of the model.

Masked-dense reference. The selection is exact, the speedup is not (that needs a
block-gather kernel).
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rope import apply_rope, build_rope_cache
from .utils import causal_mask, neg_value, repeat_kv, rms_norm


def _partial_rope(x: Tensor, cos: Tensor, sin: Tensor, rope_dim: int) -> Tensor:
    """Rotate the first rope_dim channels and pass the rest through.

    Args:
        x: Tensor of shape (batch, heads, seq, channels).
        cos: Cosine cache of shape (seq, rope_dim).
        sin: Sine cache of shape (seq, rope_dim).
        rope_dim: Number of leading channels that get rotated.

    Returns:
        Tensor of the same shape as x.
    """
    if rope_dim == x.shape[-1]:
        return apply_rope(x, cos, sin)
    rotated = apply_rope(x[..., :rope_dim], cos, sin)
    return torch.cat([rotated, x[..., rope_dim:]], dim=-1)


class MiniMaxSparseAttention(nn.Module):
    """Top-k block selection on a GQA backbone.

    Args:
        dim: Model dimension. Must be divisible by num_heads.
        num_heads: Number of query heads.
        num_kv_groups: Number of key/value groups. Must divide num_heads. Query
            heads in a group share one block selection.
        block_size: Tokens per key block. The paper deploys 128.
        top_k: Blocks kept per query. The paper deploys 16.
        index_dim: Width of the index branch. Defaults to the head dim.
        rope_dim: Channels that get RoPE in both branches, the rest pass through.
            Defaults to min(head_dim, index_dim), a full rotation. The paper
            deploys head_dim 128 with rope_dim 64.
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
        block_size: int = 128,
        top_k: int = 16,
        index_dim: int | None = None,
        rope_dim: int | None = None,
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
        self.rope_dim = (
            rope_dim if rope_dim is not None else min(head_dim, self.index_dim)
        )
        if use_rope:
            if self.rope_dim % 2:
                raise ValueError("rope_dim must be even")
            if self.rope_dim > head_dim or self.rope_dim > self.index_dim:
                raise ValueError("rope_dim cannot exceed head_dim or index_dim")
        kv = num_kv_groups * head_dim
        self.q = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, kv, bias=bias)
        self.v = nn.Linear(dim, kv, bias=bias)
        self.idx_q = nn.Linear(dim, num_kv_groups * self.index_dim, bias=bias)
        self.idx_k = nn.Linear(dim, self.index_dim, bias=bias)
        self.q_norm = nn.Parameter(torch.ones(head_dim))
        self.k_norm = nn.Parameter(torch.ones(head_dim))
        self.idx_q_norm = nn.Parameter(torch.ones(self.index_dim))
        self.idx_k_norm = nn.Parameter(torch.ones(self.index_dim))
        self.proj = nn.Linear(dim, dim, bias=bias)

    def _select_blocks(self, qi: Tensor, ki: Tensor) -> tuple[Tensor, Tensor]:
        """Score key blocks and pick the top-k per group.

        Args:
            qi: Index queries of shape (batch, groups, seq, index_dim).
            ki: Shared index key of shape (batch, 1, seq, index_dim).

        Returns:
            A pair (block_keep, scores). block_keep is boolean of shape
            (batch, groups, seq, num_blocks). scores holds the token-level index
            scores of shape (batch, groups, seq, seq), causally masked, used by
            the aux loss. Causal masking is applied to the token scores before
            pooling so a future block can never win the max for an earlier query.
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
        padded = F.pad(scores, (0, pad), value=neg) if pad else scores
        nb = (t + pad) // bs
        block_scores = padded.view(b, h, t, nb, bs).amax(dim=-1)

        topi = block_scores.topk(max(1, min(self.top_k, nb)), dim=-1).indices
        keep = torch.zeros(b, h, t, nb, dtype=torch.bool, device=device)
        keep.scatter_(-1, topi, True)
        if self.causal:
            start = (torch.arange(nb, device=device) * bs).view(1, 1, 1, nb)
            keep &= start <= torch.arange(t, device=device).view(1, 1, t, 1)
        if self.force_current_block:
            cur = (torch.arange(t, device=device) // bs).clamp_max(nb - 1)
            keep |= F.one_hot(cur, nb).bool().view(1, 1, t, nb)
        return keep, scores

    def forward(
        self, x: Tensor, return_aux: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Run block-sparse attention over the sequence.

        Args:
            x: Input of shape (batch, seq, dim).
            return_aux: If True, also return the tensors needed by
                msa_index_aux_loss.

        Returns:
            Output of shape (batch, seq, dim). If return_aux is True, a pair
            (output, aux) where aux has index_scores (batch, groups, seq, seq),
            attn_weights (batch, num_heads, seq, seq), and keep, the boolean
            per-group selected support of shape (batch, groups, seq, seq).
        """
        b, t, _ = x.shape
        h = self.num_kv_groups
        g = self.num_heads // h
        d, di = self.head_dim, self.index_dim
        bs = self.block_size
        device = x.device

        xd = x.detach()  # index branch is trained only by the aux loss
        q = self.q(x).view(b, t, self.num_heads, d).transpose(1, 2)
        k = self.k(x).view(b, t, h, d).transpose(1, 2)
        v = self.v(x).view(b, t, h, d).transpose(1, 2)
        qi = self.idx_q(xd).view(b, t, h, di).transpose(1, 2)
        ki = self.idx_k(xd).view(b, t, 1, di).transpose(1, 2)
        q = rms_norm(q, self.q_norm)  # QK norm on both branches
        k = rms_norm(k, self.k_norm)
        qi = rms_norm(qi, self.idx_q_norm)
        ki = rms_norm(ki, self.idx_k_norm)

        if self.use_rope:
            pos = torch.arange(t, device=device)
            rd = self.rope_dim
            cos, sin = build_rope_cache(pos, rd, self.rope_theta, x.dtype)
            q = _partial_rope(q, cos, sin, rd)
            k = _partial_rope(k, cos, sin, rd)
            qi = _partial_rope(qi, cos, sin, rd)
            ki = _partial_rope(ki, cos, sin, rd)

        block_keep, index_scores = self._select_blocks(qi, ki)
        keep_g = block_keep.repeat_interleave(bs, dim=-1)[..., :t]
        if self.causal:
            keep_g = keep_g & causal_mask(t, t, device)
        keep = keep_g.repeat_interleave(g, dim=1)

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
            aux = {
                "index_scores": index_scores,
                "attn_weights": attn,
                "keep": keep_g,
            }
            return out, aux
        return out


def msa_index_aux_loss(
    index_scores: Tensor,
    attn_weights: Tensor,
    keep: Tensor,
    eps: float = 1e-9,
) -> Tensor:
    """Train the index branch to match the main branch on the selected tokens.

    Hard top-k is not differentiable, so without this the index projections get
    no gradient. This is the paper's KL alignment loss. For each query and group,
    the teacher is the group-averaged main attention distribution over the
    selected tokens, and the student is the index scores softmaxed over the same
    tokens. The teacher is detached, so the loss updates only the index branch.

    Args:
        index_scores: Token-level index scores of shape (batch, groups, seq,
            seq), from the aux dict.
        attn_weights: Main attention weights of shape (batch, num_heads, seq,
            seq), from the aux dict.
        keep: Boolean per-group selected support of shape (batch, groups, seq,
            seq), from the aux dict.
        eps: Floor on the teacher probabilities inside the log.

    Returns:
        A scalar KL loss, averaged over query positions and groups.
    """
    b, heads, t, _ = attn_weights.shape
    groups = index_scores.shape[1]
    g = heads // groups
    teacher = attn_weights.detach().view(b, groups, g, t, t).mean(dim=2)
    scores = index_scores.float().masked_fill(~keep, float("-inf"))
    log_student = torch.log_softmax(scores, dim=-1)
    kl = teacher * (torch.log(teacher.clamp_min(eps)) - log_student)
    kl = kl.masked_fill(teacher <= 0, 0.0)
    return kl.sum(dim=-1).mean()
