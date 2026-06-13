"""MiniMax Sparse Attention."""

import pytest
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
    x = torch.randn(2, 16, 32, requires_grad=True)
    out, _ = m(x, return_aux=True)
    out.sum().backward()
    assert m.idx_q.weight.grad is None  # hard top-k severs the path
    m.zero_grad()
    _, aux = m(x, return_aux=True)
    msa_index_aux_loss(
        aux["index_scores"], aux["attn_weights"], aux["keep"]
    ).backward()
    assert m.idx_q.weight.grad is not None


def test_aux_loss_is_nonnegative_kl():
    m = MiniMaxSparseAttention(32, 4, 2, block_size=4, top_k=2)
    _, aux = m(torch.randn(2, 20, 32), return_aux=True)
    loss = msa_index_aux_loss(
        aux["index_scores"], aux["attn_weights"], aux["keep"]
    )
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0  # KL of two distributions on the same support


def test_index_branch_detached_from_backbone():
    m = MiniMaxSparseAttention(32, 4, 2, block_size=4, top_k=2)
    x = torch.randn(2, 16, 32, requires_grad=True)
    _, aux = m(x, return_aux=True)
    msa_index_aux_loss(
        aux["index_scores"], aux["attn_weights"], aux["keep"]
    ).backward()
    assert x.grad is None  # stop gradient on the index input


def test_partial_rope():
    m = MiniMaxSparseAttention(32, 4, 2, block_size=4, top_k=2, rope_dim=4)
    y = m(torch.randn(2, 18, 32))
    assert y.shape == (2, 18, 32)
    assert torch.isfinite(y).all()
    assert_causal(
        MiniMaxSparseAttention(32, 4, 2, block_size=4, top_k=2, rope_dim=4), 32
    )


def test_rope_dim_validation():
    with pytest.raises(ValueError):
        MiniMaxSparseAttention(32, 4, 2, rope_dim=3)  # odd
    with pytest.raises(ValueError):
        MiniMaxSparseAttention(32, 4, 2, rope_dim=16)  # over head_dim of 8


def test_qk_norm_trains_with_aux_loss():
    m = MiniMaxSparseAttention(32, 4, 2, block_size=4, top_k=2)
    _, aux = m(torch.randn(2, 16, 32), return_aux=True)
    msa_index_aux_loss(
        aux["index_scores"], aux["attn_weights"], aux["keep"]
    ).backward()
    assert m.idx_q_norm.grad is not None
    assert m.idx_k_norm.grad is not None


def test_main_qk_norm_trains_with_lm_loss():
    m = MiniMaxSparseAttention(32, 4, 2, block_size=4, top_k=2)
    m(torch.randn(2, 16, 32)).sum().backward()
    assert m.q_norm.grad is not None
    assert m.k_norm.grad is not None
