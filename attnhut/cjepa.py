"""Causal-JEPA object-level masked attention (Nam et al., 2026).

A world model predictor that learns object interactions by hiding objects and
asking attention to put them back. Object slots are arranged as a grid of time
by objects and flattened into one sequence. A random subset of objects is hidden
across the history window, the whole future is hidden, and a bidirectional
transformer rebuilds every hidden slot from the visible ones. Each hidden slot
starts as a learned mask token plus a temporal embedding plus a linear projection
of that object at the first frame, which anchors its identity. Nothing encodes
object order, so the predictor is permutation equivariant over objects.
"""

import torch
from torch import Tensor, nn

from .standard import StandardAttention


class _Block(nn.Module):
    """One bidirectional transformer block, full attention then MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()
        self.attn = StandardAttention(
            dim, num_heads, causal=False, dropout=dropout, bias=bias
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Args: x of shape (batch, seq, dim). Returns: (batch, seq, dim)."""
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class CausalJEPAAttention(nn.Module):
    """Object-level masked bidirectional attention predictor.

    Args:
        dim: Slot dimension. Must be divisible by num_heads.
        num_heads: Number of attention heads.
        num_slots: Number of object slots per frame.
        history_frames: Length of the observed history window.
        pred_frames: Number of future frames to predict.
        num_masked_slots: How many objects to hide across the history window.
        depth: Number of transformer blocks.
        mlp_dim: Hidden width of each block MLP. Defaults to 4 * dim.
        dropout: Dropout probability inside the transformer, train time only.
        bias: Whether the linear projections carry a bias term.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_slots: int,
        history_frames: int,
        pred_frames: int = 1,
        num_masked_slots: int = 2,
        depth: int = 6,
        mlp_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.history_frames = history_frames
        self.pred_frames = pred_frames
        self.total_frames = history_frames + pred_frames
        self.num_masked_slots = num_masked_slots
        mlp_dim = mlp_dim or 4 * dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.time_pos = nn.Parameter(torch.zeros(1, self.total_frames, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        self.id_proj = nn.Linear(dim, dim, bias=bias)
        self.blocks = nn.ModuleList(
            [
                _Block(dim, num_heads, mlp_dim, dropout, bias)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, dim, bias=bias)

    def _build_input(self, slots: Tensor, mask_slots: Tensor) -> Tensor:
        """Lay out the masked time by object grid the predictor sees.

        Args:
            slots: History slots of shape (batch, history_frames, num_slots, dim).
            mask_slots: Object indices hidden over history, shape (num_masked,).

        Returns:
            Grid of shape (batch, total_frames, num_slots, dim) where hidden
            positions hold identity anchored queries and visible positions hold
            the real slots, both with a temporal embedding added.
        """
        b, t_hist, s, d = slots.shape
        anchor = self.id_proj(slots[:, 0]).unsqueeze(1)  # (b, 1, s, d)
        x = (self.mask_token + self.time_pos + anchor).clone()
        # the first frame is real and anchors each object's identity
        x[:, 0] = slots[:, 0] + self.time_pos[:, 0]
        visible = torch.ones(s, dtype=torch.bool, device=slots.device)
        visible[mask_slots] = False
        keep = visible.nonzero(as_tuple=True)[0]
        if t_hist > 1 and keep.numel() > 0:
            x[:, 1:t_hist, keep] = (
                slots[:, 1:, keep] + self.time_pos[:, 1:t_hist]
            )
        return x

    def forward(
        self, slots: Tensor, mask_slots: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Mask objects and rebuild them with bidirectional attention.

        Args:
            slots: History slots of shape (batch, history_frames, num_slots, dim).
            mask_slots: Optional object indices to hide over history. If None,
                num_masked_slots indices are drawn at random and shared across
                the batch.

        Returns:
            A pair (pred, mask_slots) where pred has shape
            (batch, total_frames, num_slots, dim) and mask_slots holds the hidden
            object indices.
        """
        b, _, s, d = slots.shape
        if mask_slots is None:
            order = torch.randperm(self.num_slots, device=slots.device)
            mask_slots = order[: self.num_masked_slots]
        x = self._build_input(slots, mask_slots)
        x = x.reshape(b, self.total_frames * s, d)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).reshape(b, self.total_frames, s, d)
        return self.to_out(x), mask_slots

    def predict(self, slots: Tensor) -> Tensor:
        """Roll out future frames from a fully visible history.

        Args:
            slots: History slots of shape (batch, history_frames, num_slots, dim).

        Returns:
            Predicted future of shape (batch, pred_frames, num_slots, dim).
        """
        b, t_hist, s, d = slots.shape
        anchor = self.id_proj(slots[:, 0]).unsqueeze(1)  # (b, 1, s, d)
        hist = slots + self.time_pos[:, :t_hist]
        future = self.mask_token + self.time_pos[:, t_hist:] + anchor
        x = torch.cat([hist, future], dim=1).reshape(
            b, self.total_frames * s, d
        )
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).reshape(b, self.total_frames, s, d)
        return self.to_out(x)[:, t_hist:]


def cjepa_masked_loss(
    pred: Tensor,
    target: Tensor,
    mask_slots: Tensor,
    history_frames: int,
) -> Tensor:
    """Masked latent prediction loss from Causal-JEPA.

    Scores only the positions the predictor had to infer, which are the hidden
    objects across the history window and every object across the future. The
    first frame is the identity anchor and is never scored.

    Args:
        pred: Predicted slots of shape (batch, total_frames, num_slots, dim).
        target: Ground truth slots of the same shape.
        mask_slots: Object indices hidden over history, shape (num_masked,).
        history_frames: Length of the history window.

    Returns:
        Scalar mean squared error over the inferred positions.
    """
    _, total, s, _ = pred.shape
    keep = torch.zeros(total, s, dtype=torch.bool, device=pred.device)
    keep[history_frames:] = True
    keep[1:history_frames, mask_slots] = True
    err = ((pred - target) ** 2).mean(-1)  # (batch, total, num_slots)
    return err[:, keep].mean()
