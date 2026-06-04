"""Delta Attention (Willette et al., NeurIPS 2025).

Sliding-window attention shifts the output away from full attention because each
row renormalizes over a different key set. Delta Attention corrects it cheaply.
Run full attention on every stride-th query, take the gap between the dense and
sparse outputs at those anchors, and add it back to every row in the block.
Training free, so it bolts onto a model you already have.

Masked-dense reference. The paper drives the sparse base and anchor pass with
flash kernels.
"""

import torch
from torch import Tensor, nn

from .utils import masked_softmax_attention


class DeltaAttention(nn.Module):
    """Sliding-window plus sink attention with the delta correction.

    Args:
        dim: Model dimension. Must be divisible by num_heads.
        num_heads: Number of attention heads.
        window: Recent keys each query attends in the sparse base.
        sink: Leading keys every query keeps, the attention sink.
        stride: One dense anchor query is computed per stride rows. Smaller is
            more accurate and more expensive. stride=1 recovers full attention.
        bias: Whether the linear projections carry a bias term.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window: int = 256,
        sink: int = 4,
        stride: int = 16,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = window
        self.sink = sink
        self.stride = stride
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Run sparse attention with the delta correction.

        Args:
            x: Input of shape (batch, seq, dim).

        Returns:
            Output of shape (batch, seq, dim).
        """
        b, t, _ = x.shape
        scale = self.head_dim**-0.5
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        i = torch.arange(t, device=x.device).unsqueeze(1)
        j = torch.arange(t, device=x.device).unsqueeze(0)
        causal = j <= i
        sparse_keep = causal & ((j > i - self.window) | (j < self.sink))
        sparse_out = masked_softmax_attention(
            q, k, v, sparse_keep[None, None], scale
        )

        # Full attention on every stride-th query, then broadcast delta.
        idx = torch.arange(0, t, self.stride, device=x.device)
        dense = masked_softmax_attention(
            q[:, :, idx], k, v, causal[idx][None, None], scale
        )
        delta = dense - sparse_out[:, :, idx]
        out = sparse_out + delta.repeat_interleave(self.stride, dim=2)[:, :, :t]

        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.proj(out)
