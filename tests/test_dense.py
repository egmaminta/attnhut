"""Standard, multi-query and grouped-query attention."""

import pytest
import torch

from attnhut import (
    GroupedQueryAttention,
    MultiQueryAttention,
    StandardAttention,
)
from conftest import assert_causal


def test_shapes():
    x = torch.randn(2, 10, 32)
    for m in (
        StandardAttention(32, 4),
        MultiQueryAttention(32, 4),
        GroupedQueryAttention(32, 4, 2),
    ):
        y = m(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()


def test_causal():
    assert_causal(StandardAttention(32, 4, causal=True), 32)
    assert_causal(MultiQueryAttention(32, 4, causal=True), 32)
    assert_causal(GroupedQueryAttention(32, 4, 2, causal=True), 32)


def test_gqa_requires_divisible_kv_heads():
    with pytest.raises(ValueError):
        GroupedQueryAttention(32, 4, 3)
