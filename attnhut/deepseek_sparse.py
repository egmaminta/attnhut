"""DeepSeek Sparse Attention (DeepSeek-V3.2).

A small lightning indexer scores every query-key pair cheaply, then each query
keeps only its top-k keys for the main attention. The indexer is gated ReLU
rather than softmax, with a single shared key head, so it is far lighter than
the main heads. Selection is token-level, which is what separates DSA from the
block-level pooling in MiniMax Sparse Attention.
"""

import torch
from torch import Tensor, nn

from .utils import causal_mask, masked_softmax_attention, neg_value


class LightningIndexer(nn.Module):
    """Cheap relevance scorer I[t, s] = sum_h w[t, h] * ReLU(q[t, h] . k[s]).

    Args:
        dim: Model dimension.
        num_heads: Number of indexer heads.
        head_dim: Width of each indexer head.
        bias: Whether the linear projections carry a bias term.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        head_dim: int = 64,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.wq = nn.Linear(dim, num_heads * head_dim, bias=bias)
        self.wk = nn.Linear(dim, head_dim, bias=bias)
        self.ww = nn.Linear(dim, num_heads, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Score every query against every key.

        Args:
            x: Input of shape (batch, seq, dim).

        Returns:
            Index scores of shape (batch, seq, seq), unmasked.
        """
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.num_heads, self.head_dim)
        k = self.wk(x)
        w = self.ww(x)
        dots = torch.einsum("bthd,bsd->bths", q, k).relu()
        return torch.einsum("bth,bths->bts", w, dots)


class DeepSeekSparseAttention(nn.Module):
    """Lightning-indexer top-k token selection on a multi-head backbone.

    Args:
        dim: Model dimension. Must be divisible by num_heads.
        num_heads: Number of attention heads.
        top_k: Keys kept per query.
        num_index_heads: Heads in the lightning indexer.
        index_head_dim: Width of each indexer head.
        causal: If True, selection and attention are restricted to the past.
        bias: Whether the linear projections carry a bias term.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        top_k: int = 64,
        num_index_heads: int = 4,
        index_head_dim: int = 64,
        causal: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.top_k = top_k
        self.causal = causal
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.indexer = LightningIndexer(
            dim, num_index_heads, index_head_dim, bias
        )

    def forward(
        self, x: Tensor, return_aux: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Run token-sparse attention over the sequence.

        Args:
            x: Input of shape (batch, seq, dim).
            return_aux: If True, also return the index scores and the selection
                mask.

        Returns:
            Output of shape (batch, seq, dim). If return_aux is True, a pair
            (output, aux) where aux has index_scores (batch, seq, seq) and
            selected (batch, seq, seq) boolean.
        """
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        scores = self.indexer(x)
        if self.causal:
            valid = causal_mask(t, t, x.device)
        else:
            valid = torch.ones(t, t, dtype=torch.bool, device=x.device)
        scores = scores.masked_fill(~valid, neg_value(scores.dtype))

        topi = scores.topk(min(self.top_k, t), dim=-1).indices
        keep = torch.zeros(b, t, t, dtype=torch.bool, device=x.device)
        keep.scatter_(-1, topi, True)
        keep &= valid

        out = masked_softmax_attention(
            q, k, v, keep[:, None], self.head_dim**-0.5
        )
        out = out.transpose(1, 2).reshape(b, t, -1)
        out = self.proj(out)
        if return_aux:
            return out, {"index_scores": scores, "selected": keep}
        return out


def dsa_index_aux_loss(index_scores: Tensor, attn_weights: Tensor) -> Tensor:
    """KL warmup loss that fits the indexer to the main attention spread.

    The target sums the dense attention over heads and normalizes it per query.
    Run it against the index scores so the indexer learns where attention goes
    before the hard top-k turns on.

    Args:
        index_scores: Indexer scores of shape (batch, seq, seq), from the aux
            dict.
        attn_weights: Dense attention weights of shape (batch, num_heads, seq,
            seq).

    Returns:
        A scalar loss.
    """
    target = attn_weights.detach().sum(dim=1)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    log_p = torch.log_softmax(index_scores.to(torch.float32), dim=-1)
    return -(target * log_p).sum(dim=-1).mean()
