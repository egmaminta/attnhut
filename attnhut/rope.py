"""Rotary position embeddings, rotate-half (Llama / GPT-NeoX) style.

Kept separate so it is reusable and easy to test. The rotation runs in float32
(MPS safe) and is cast back to the operand dtype.
"""

import torch
from torch import Tensor


def build_rope_cache(
    positions: Tensor,
    dim: int,
    theta: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Precompute the cos and sin tables for a set of positions.

    Args:
        positions: Absolute token positions, shape (seq,) for positions shared
            across the batch or (batch, seq) for per-sample positions.
        dim: Head dimension to rotate. Must be even.
        theta: Base frequency.
        dtype: Output dtype of the tables.

    Returns:
        A pair (cos, sin), each of shape positions.shape + (dim,).
    """
    if dim % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {dim}")
    device = positions.device
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    freqs = positions.to(torch.float32).unsqueeze(-1) * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: Tensor) -> Tensor:
    """Swap and negate the two halves of the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary position embeddings to x.

    Args:
        x: Tensor of shape (batch, heads, seq, dim).
        cos: Cosine table of shape (seq, dim) shared across the batch, or
            (batch, seq, dim) for per-sample positions.
        sin: Sine table of the same shape as cos.

    Returns:
        Rotated tensor with the same shape as x.
    """
    if cos.dim() == 2:
        cos, sin = cos[None, None], sin[None, None]
    else:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)
