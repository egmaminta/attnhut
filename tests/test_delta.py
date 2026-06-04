"""Delta Attention."""

import torch
import torch.nn.functional as F

from attnhut import DeltaAttention
from conftest import assert_causal


def test_shape():
    m = DeltaAttention(32, 4, window=8, sink=2, stride=4)
    y = m(torch.randn(2, 20, 32))
    assert y.shape == (2, 20, 32)
    assert torch.isfinite(y).all()


def test_stride_one_recovers_full_attention():
    # With stride 1 every query is corrected densely, so the output is exactly
    # full causal attention regardless of the window.
    m = DeltaAttention(32, 4, window=4, sink=0, stride=1).eval()
    x = torch.randn(1, 12, 32)
    out = m(x)

    q, k, v = m.qkv(x).chunk(3, dim=-1)
    q = q.view(1, 12, 4, 8).transpose(1, 2)
    k = k.view(1, 12, 4, 8).transpose(1, 2)
    v = v.view(1, 12, 4, 8).transpose(1, 2)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = m.proj(ref.transpose(1, 2).reshape(1, 12, 32))
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_causal():
    assert_causal(DeltaAttention(32, 4, window=4, sink=2, stride=4), 32)
