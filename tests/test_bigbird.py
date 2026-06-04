"""BigBird block-sparse attention."""

import torch

from attnhut import BigBirdAttention
from conftest import assert_causal


def test_shape_with_padding():
    m = BigBirdAttention(32, 4, block_size=8, num_global_blocks=1)
    x = torch.randn(2, 20, 32)  # not a multiple of block_size
    y = m(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_mask_invariants():
    m = BigBirdAttention(
        32,
        4,
        block_size=4,
        num_window_blocks=3,
        num_random_blocks=1,
        num_global_blocks=1,
    )
    mask = m.block_mask(8, torch.device("cpu"))
    assert mask[0].all() and mask[:, 0].all()  # global block is symmetric
    assert mask.diagonal().all()  # self block always in window
    for j in range(1, 8):
        assert mask[j, j - 1] and mask[j - 1, j]  # window neighbours


def test_random_block_count():
    m = BigBirdAttention(
        32,
        4,
        block_size=4,
        num_window_blocks=3,
        num_random_blocks=2,
        num_global_blocks=1,
    )
    mask = m.block_mask(12, torch.device("cpu"))
    j = 6  # a block far from globals and edges
    window = {j - 1, j, j + 1}
    extra = [k for k in range(12) if mask[j, k] and k not in window and k != 0]
    assert len(extra) == 2


def test_causal():
    assert_causal(
        BigBirdAttention(32, 4, block_size=4, num_global_blocks=1, causal=True),
        32,
    )
