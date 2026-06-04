"""MiniMax Sparse Attention."""

import torch

from attnhut import MiniMaxSparseAttention, msa_index_aux_loss
from conftest import assert_causal


def test_shape_with_padding():
    m = MiniMaxSparseAttention(32, 4, 2, block_size=4, top_k=2)
    x = torch.randn(2, 18, 32)
    y = m(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_causal():
    assert_causal(MiniMaxSparseAttention(32, 4, 2, block_size=4, top_k=2), 32)


def test_degrade_to_dense_when_all_blocks_kept():
    m = MiniMaxSparseAttention(
        32, 4, 2, block_size=4, top_k=1000, use_rope=False
    )
    _, aux = m(torch.randn(1, 16, 32), return_aux=True)
    nonzero = aux["attn_weights"] > 0
    causal = torch.tril(torch.ones(16, 16, dtype=torch.bool))
    assert torch.equal(nonzero, causal.expand_as(nonzero))


def test_sparser_than_dense():
    m = MiniMaxSparseAttention(32, 4, 2, block_size=4, top_k=1)
    _, aux = m(torch.randn(1, 32, 32), return_aux=True)
    assert (aux["attn_weights"] > 0).float().mean() < 0.5


def test_index_branch_trains_only_with_aux_loss():
    m = MiniMaxSparseAttention(32, 4, 2, block_size=4, top_k=2)
    x = torch.randn(2, 16, 32)
    out, _ = m(x, return_aux=True)
    out.sum().backward()
    assert m.idx_q.weight.grad is None  # hard top-k severs the path
    m.zero_grad()
    _, aux = m(x, return_aux=True)
    msa_index_aux_loss(
        aux["block_scores"], aux["attn_weights"], 4, 2
    ).backward()
    assert m.idx_q.weight.grad is not None
