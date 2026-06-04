"""Multi-query attention (Shazeer, 2019)."""

import torch.nn.functional as F
from torch import Tensor, nn


class MultiQueryAttention(nn.Module):
    """All query heads share a single key/value head.

    This is the most aggressive KV cache shrink. The query projection stays full
    width while keys and values collapse to one head, broadcast to every query.

    Args:
        dim: Model dimension. Must be divisible by num_heads.
        num_heads: Number of query heads.
        causal: If True, a position only attends to itself and the past.
        dropout: Dropout probability on the attention weights, train time only.
        bias: Whether the linear projections carry a bias term.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        causal: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.dropout = dropout
        self.q = nn.Linear(dim, dim, bias=bias)
        self.kv = nn.Linear(dim, 2 * self.head_dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Run attention over the sequence.

        Args:
            x: Input of shape (batch, seq, dim).
            attn_mask: Optional additive or boolean mask broadcastable to
                (batch, num_heads, seq, seq).

        Returns:
            Output of shape (batch, seq, dim).
        """
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(x).chunk(2, dim=-1)
        k = k.view(b, t, 1, self.head_dim).transpose(1, 2)
        v = v.view(b, t, 1, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal and attn_mask is None,
            enable_gqa=True,
        )
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.proj(out)
