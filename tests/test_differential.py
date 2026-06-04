"""Differential Transformer attention."""

import pytest
import torch

from attnhut import DifferentialAttention
from attnhut.differential import lambda_init
from conftest import assert_causal


def test_shape():
    m = DifferentialAttention(64, 4)
    y = m(torch.randn(2, 10, 64))
    assert y.shape == (2, 10, 64)
    assert torch.isfinite(y).all()


def test_causal():
    assert_causal(DifferentialAttention(64, 4, causal=True), 64)


def test_requires_dim_divisible_by_two_times_heads():
    with pytest.raises(ValueError):
        DifferentialAttention(64, 5)


def test_lambda_init_grows_with_depth():
    assert lambda_init(0) < lambda_init(8) < lambda_init(64)
    assert 0.0 < lambda_init(0) and lambda_init(64) < 0.8
