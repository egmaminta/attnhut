"""Shared building blocks for attnhut attention modules."""

import torch
import torch.nn.functional as F
from torch import Tensor


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Repeat grouped key/value heads so every query head has one to read.

    Query head q reads kv head q // n_rep, the contiguous grouping GQA uses.

    Args:
        x: Key or value tensor of shape (batch, kv_heads, seq, head_dim).
        n_rep: Number of query heads sharing each kv head.

    Returns:
        Tensor of shape (batch, kv_heads * n_rep, seq, head_dim).
    """
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    x = x[:, :, None, :, :].expand(b, h, n_rep, s, d)
    return x.reshape(b, h * n_rep, s, d)


def neg_value(dtype: torch.dtype) -> float:
    """Masking fill value for a dtype.

    We use a large finite negative rather than -inf so that a fully masked
    softmax row stays finite instead of turning into NaN.

    Args:
        dtype: Dtype of the tensor that will be masked.

    Returns:
        The smallest finite value representable in dtype.
    """
    return torch.finfo(dtype).min


def causal_mask(q_len: int, k_len: int, device: torch.device) -> Tensor:
    """Boolean causal mask, True where a query may attend a key.

    Query i sees key j when j <= i + (k_len - q_len). The offset lets a short
    query attend into a longer cached key sequence, which matches the is_causal
    convention of scaled_dot_product_attention.

    Args:
        q_len: Number of query positions.
        k_len: Number of key positions.
        device: Device for the returned mask.

    Returns:
        Boolean tensor of shape (q_len, k_len).
    """
    off = k_len - q_len
    i = torch.arange(q_len, device=device).unsqueeze(1)
    j = torch.arange(k_len, device=device).unsqueeze(0)
    return j <= i + off


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """RMSNorm over the last dimension.

    Args:
        x: Input of shape (..., dim).
        weight: Learned scale of shape (dim,).
        eps: Numerical floor for the norm.

    Returns:
        Tensor with the same shape as x.
    """
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


def masked_softmax_attention(
    q: Tensor, k: Tensor, v: Tensor, keep: Tensor, scale: float
) -> Tensor:
    """Softmax attention restricted to the keys marked in keep.

    This is the shared core for the sparse modules. Dropped entries go to
    a large negative before the softmax, and any query with no visible key
    returns zeros instead of NaN.

    Args:
        q: Queries of shape (batch, heads, q_seq, head_dim).
        k: Keys of shape (batch, heads, k_seq, head_dim).
        v: Values of shape (batch, heads, k_seq, head_dim).
        keep: Boolean mask broadcastable to (batch, heads, q_seq, k_seq), True
            where a query is allowed to attend a key.
        scale: Softmax temperature, usually head_dim ** -0.5.

    Returns:
        Attention output of shape (batch, heads, q_seq, head_dim).
    """
    logits = torch.matmul(q, k.transpose(-2, -1)) * scale
    logits = logits.masked_fill(~keep, neg_value(q.dtype))
    empty = ~keep.any(dim=-1, keepdim=True)
    logits = logits.masked_fill(empty, 0.0)
    sm_dtype = (
        torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    )
    attn = torch.softmax(logits.to(sm_dtype), dim=-1).to(q.dtype)
    attn = attn.masked_fill(empty, 0.0)
    return torch.matmul(attn, v)
