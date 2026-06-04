"""Differential Transformer attention (Ye et al., 2024).

Each head computes two softmax attention maps and subtracts one from the other,
scaled by a learned lambda. Shared common-mode noise in the two maps cancels,
so the head puts less weight on irrelevant context. A per-head RMSNorm and a
(1 - lambda_init) scale keep the output magnitude close to a normal head.

RoPE from the paper is omitted here for clarity.
"""

import math

import torch
from torch import Tensor, nn

from .utils import causal_mask, neg_value, rms_norm


def lambda_init(depth: int) -> float:
    """Starting lambda for a layer, larger the deeper the layer sits.

    Args:
        depth: Zero-based index of the layer in the stack.

    Returns:
        The lambda_init scalar, between about 0.2 and 0.8.
    """
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class DifferentialAttention(nn.Module):
    """Two-softmax differential attention.

    Args:
        dim: Model dimension. Must be divisible by 2 * num_heads.
        num_heads: Number of differential heads.
        depth: Zero-based layer index, which sets lambda_init.
        causal: If True, a position only attends to itself and the past.
        bias: Whether the linear projections carry a bias term.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        depth: int = 0,
        causal: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if dim % (2 * num_heads) != 0:
            raise ValueError("dim must be divisible by 2 * num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads // 2
        self.causal = causal
        self.lambda_init = lambda_init(depth)
        self.q = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, dim, bias=bias)
        self.v = nn.Linear(dim, dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.head_norm = nn.Parameter(torch.ones(2 * self.head_dim))
        self.lambda_q1 = nn.Parameter(
            torch.empty(self.head_dim).normal_(0, 0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.empty(self.head_dim).normal_(0, 0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.empty(self.head_dim).normal_(0, 0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.empty(self.head_dim).normal_(0, 0.1)
        )

    def _lambda(self) -> Tensor:
        """Scalar lambda = exp(q1.k1) - exp(q2.k2) + lambda_init."""
        l1 = torch.exp((self.lambda_q1 * self.lambda_k1).sum())
        l2 = torch.exp((self.lambda_q2 * self.lambda_k2).sum())
        return l1 - l2 + self.lambda_init

    def forward(self, x: Tensor) -> Tensor:
        """Run differential attention over the sequence.

        Args:
            x: Input of shape (batch, seq, dim).

        Returns:
            Output of shape (batch, seq, dim).
        """
        b, t, _ = x.shape
        hd, h = self.head_dim, self.num_heads
        q = self.q(x).view(b, t, 2 * h, hd).transpose(1, 2)
        k = self.k(x).view(b, t, 2 * h, hd).transpose(1, 2)
        v = self.v(x).view(b, t, h, 2 * hd).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) * hd**-0.5
        if self.causal:
            logits = logits.masked_fill(
                ~causal_mask(t, t, x.device), neg_value(logits.dtype)
            )
        w = torch.softmax(logits.float(), dim=-1).to(x.dtype)
        w = w.view(b, h, 2, t, t)
        attn = w[:, :, 0] - self._lambda() * w[:, :, 1]

        out = torch.matmul(attn, v)  # (b, h, t, 2 * hd)
        out = rms_norm(out, self.head_norm, eps=1e-5) * (1 - self.lambda_init)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.proj(out)
