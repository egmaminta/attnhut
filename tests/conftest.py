"""Shared test fixtures and helpers (pure torch, no numpy)."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def assert_causal(module, dim, t=16, batch=1):
    """Perturbing the last token must not change any earlier output."""
    module.eval()
    x = torch.randn(batch, t, dim)
    with torch.no_grad():
        y1 = module(x)
        x2 = x.clone()
        x2[:, -1] += 5.0
        y2 = module(x2)
    torch.testing.assert_close(y1[:, :-1], y2[:, :-1], atol=1e-4, rtol=1e-4)
