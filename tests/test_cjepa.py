"""Causal-JEPA object-level masked attention."""

import torch

from attnhut import CausalJEPAAttention, cjepa_masked_loss


def _model(**kw):
    args = dict(
        dim=32,
        num_heads=4,
        num_slots=5,
        history_frames=3,
        pred_frames=2,
        num_masked_slots=2,
    )
    args.update(kw)
    return CausalJEPAAttention(**args)


def test_shapes():
    m = _model()
    pred, masked = m(torch.randn(2, 3, 5, 32))
    assert pred.shape == (2, 5, 5, 32)  # total_frames = history + pred
    assert masked.shape == (2,)  # num_masked_slots
    assert torch.isfinite(pred).all()


def test_predict_returns_future_only():
    fut = _model().predict(torch.randn(2, 3, 5, 32))
    assert fut.shape == (2, 2, 5, 32)  # pred_frames
    assert torch.isfinite(fut).all()


def test_gradients_flow():
    m = _model()
    slots = torch.randn(2, 3, 5, 32, requires_grad=True)
    pred, masked = m(slots)
    cjepa_masked_loss(pred, torch.randn_like(pred), masked, 3).backward()
    assert slots.grad is not None
    assert torch.isfinite(slots.grad).all()


def test_loss_scores_only_inferred_positions():
    m = _model(depth=1)
    pred, masked = m(torch.randn(1, 3, 5, 32))
    visible = [i for i in range(5) if i not in masked.tolist()][0]

    assert cjepa_masked_loss(pred, pred.clone(), masked, 3) == 0.0

    only_visible = pred.clone()
    only_visible[:, 1, visible] += 1.0  # visible history slot is never scored
    assert cjepa_masked_loss(pred, only_visible, masked, 3) == 0.0

    a_future = pred.clone()
    a_future[:, -1] += 1.0  # the future is always scored
    assert cjepa_masked_loss(pred, a_future, masked, 3) > 0.0


def test_permutation_equivariant_over_objects():
    m = _model(depth=2)
    m.eval()
    slots = torch.randn(2, 3, 5, 32)
    perm = torch.tensor([2, 0, 4, 1, 3])
    inv = torch.argsort(perm)
    mask = torch.tensor([0, 1])
    with torch.no_grad():
        out, _ = m(slots, mask_slots=mask)
        out_perm, _ = m(slots[:, :, perm], mask_slots=inv[mask])
    torch.testing.assert_close(out_perm, out[:, :, perm], atol=1e-4, rtol=1e-4)
