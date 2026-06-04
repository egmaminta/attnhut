"""Multi-head Latent Attention."""

import torch

from attnhut import MultiHeadLatentAttention
from conftest import assert_causal


def test_shape():
    m = MultiHeadLatentAttention(64, 4)
    y = m(torch.randn(2, 10, 64))
    assert y.shape == (2, 10, 64)
    assert torch.isfinite(y).all()


def test_causal():
    assert_causal(MultiHeadLatentAttention(64, 4, causal=True), 64, t=12)


def test_custom_latent_dims():
    m = MultiHeadLatentAttention(
        64,
        4,
        kv_lora_rank=16,
        q_lora_rank=24,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=12,
    )
    y = m(torch.randn(1, 8, 64))
    assert y.shape == (1, 8, 64)
