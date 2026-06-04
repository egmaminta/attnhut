"""Slot Attention (Locatello et al., 2020).

An object-centric attention. A small set of slots compete to explain the input
features. The softmax runs over the slots rather than the inputs, so each input
location is softly assigned to a slot, and the slots are refined over a few
iterations with a GRU and a residual MLP. The output is a set of slot vectors,
not a sequence the same length as the input.
"""

import torch
from torch import Tensor, nn


class SlotAttention(nn.Module):
    """Iterative slot attention with GRU refinement.

    Args:
        dim: Feature and slot dimension.
        num_slots: Number of slots competing for the input.
        num_iters: Refinement iterations.
        hidden_dim: Hidden width of the residual MLP.
        eps: Small constant added before the weighted-mean normalization.
    """

    def __init__(
        self,
        dim: int,
        num_slots: int,
        num_iters: int = 3,
        hidden_dim: int = 128,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.eps = eps
        self.scale = dim**-0.5
        self.slot_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slot_log_sigma)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        hidden_dim = max(dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim)
        )
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_mlp = nn.LayerNorm(dim)

    def forward(self, inputs: Tensor) -> Tensor:
        """Group the input features into slots.

        Args:
            inputs: Input features of shape (batch, num_inputs, dim).

        Returns:
            Slots of shape (batch, num_slots, dim).
        """
        b, _, d = inputs.shape
        mu = self.slot_mu.expand(b, self.num_slots, -1)
        sigma = self.slot_log_sigma.exp().expand(b, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)
        for _ in range(self.num_iters):
            prev = slots
            q = self.to_q(self.norm_slots(slots)) * self.scale
            dots = torch.einsum("bsd,bnd->bsn", q, k)
            attn = dots.softmax(dim=1) + self.eps  # slots compete per input
            attn = attn / attn.sum(dim=-1, keepdim=True)  # mean over inputs
            updates = torch.einsum("bsn,bnd->bsd", attn, v)
            slots = self.gru(updates.reshape(-1, d), prev.reshape(-1, d))
            slots = slots.reshape(b, self.num_slots, d)
            slots = slots + self.mlp(self.norm_pre_mlp(slots))
        return slots
