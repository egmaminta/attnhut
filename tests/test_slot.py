"""Slot Attention."""

import torch

from attnhut import SlotAttention


def test_output_shape():
    m = SlotAttention(dim=32, num_slots=5)
    slots = m(torch.randn(2, 49, 32))
    assert slots.shape == (2, 5, 32)
    assert torch.isfinite(slots).all()


def test_gradients_flow():
    m = SlotAttention(dim=32, num_slots=4, num_iters=2)
    x = torch.randn(2, 16, 32, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_num_slots_configurable():
    slots = SlotAttention(dim=16, num_slots=7)(torch.randn(1, 20, 16))
    assert slots.shape == (1, 7, 16)
