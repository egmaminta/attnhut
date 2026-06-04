"""DeepSeek Sparse Attention."""

import torch

from attnhut import DeepSeekSparseAttention, dsa_index_aux_loss
from conftest import assert_causal


def test_shape():
    m = DeepSeekSparseAttention(32, 4, top_k=8)
    y = m(torch.randn(2, 20, 32))
    assert y.shape == (2, 20, 32)
    assert torch.isfinite(y).all()


def test_causal():
    assert_causal(DeepSeekSparseAttention(32, 4, top_k=4), 32)


def test_topk_count_per_query():
    m = DeepSeekSparseAttention(32, 4, top_k=3)
    _, aux = m(torch.randn(1, 12, 32), return_aux=True)
    sel = aux["selected"][0]
    for t in range(12):
        assert sel[t].sum().item() == min(3, t + 1)


def test_degrade_to_dense():
    m = DeepSeekSparseAttention(32, 4, top_k=1000)
    _, aux = m(torch.randn(1, 12, 32), return_aux=True)
    causal = torch.tril(torch.ones(12, 12, dtype=torch.bool))
    assert torch.equal(aux["selected"][0], causal)


def test_aux_loss_trains_indexer():
    m = DeepSeekSparseAttention(32, 4, top_k=4)
    _, aux = m(torch.randn(2, 16, 32), return_aux=True)
    target = torch.rand(2, 4, 16, 16)
    dsa_index_aux_loss(aux["index_scores"], target).backward()
    assert m.indexer.wq.weight.grad is not None
