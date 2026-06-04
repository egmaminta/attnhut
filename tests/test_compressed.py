"""DeepSeek-V4 Heavily Compressed and Compressed Sparse attention."""

import torch

from attnhut import CompressedSparseAttention, HeavilyCompressedAttention
from conftest import assert_causal


def test_hca_shape():
    m = HeavilyCompressedAttention(32, 4, compression_rate=4, window=8)
    y = m(torch.randn(2, 20, 32))
    assert y.shape == (2, 20, 32)
    assert torch.isfinite(y).all()


def test_hca_causal():
    assert_causal(
        HeavilyCompressedAttention(32, 4, compression_rate=4, window=8), 32
    )


def test_csa_shape():
    m = CompressedSparseAttention(32, 4, compression_rate=4, top_k=2, window=8)
    y = m(torch.randn(2, 20, 32))
    assert y.shape == (2, 20, 32)
    assert torch.isfinite(y).all()


def test_csa_causal():
    assert_causal(
        CompressedSparseAttention(32, 4, compression_rate=4, top_k=2, window=8),
        32,
    )


def test_csa_selects_at_most_top_k():
    m = CompressedSparseAttention(32, 4, compression_rate=4, top_k=2, window=8)
    _, aux = m(torch.randn(1, 40, 32), return_aux=True)
    assert aux["selected"].sum(-1).max().item() <= 2
