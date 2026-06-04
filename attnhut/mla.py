"""Multi-head Latent Attention (DeepSeek-V2/V3).

Keys and values are cached as one low-rank latent per token instead of full
per-head K and V, which shrinks the KV cache by an order of magnitude. Position
rides on a small decoupled RoPE part kept separate from the latent. This is the
readable non-absorbed form.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rope import apply_rope, build_rope_cache
from .utils import rms_norm


class MultiHeadLatentAttention(nn.Module):
    """MLA with a decoupled RoPE key head.

    Args:
        dim: Model dimension.
        num_heads: Number of attention heads.
        q_lora_rank: Width of the query latent. Defaults to dim // 2.
        kv_lora_rank: Width of the shared KV latent. Defaults to dim // 4.
        qk_nope_head_dim: Per-head query/key width without RoPE. Defaults to
            half the head dim.
        qk_rope_head_dim: Per-head query/key width carrying RoPE. Must be even.
            Defaults to half the head dim.
        v_head_dim: Per-head value width. Defaults to the head dim.
        causal: If True, a position only attends to itself and the past.
        rope_theta: RoPE base frequency.
        bias: Whether the linear projections carry a bias term.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        q_lora_rank: int | None = None,
        kv_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        v_head_dim: int | None = None,
        causal: bool = True,
        rope_theta: float = 10000.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank or dim // 2
        self.kv_lora_rank = kv_lora_rank or dim // 4
        self.qk_nope_head_dim = qk_nope_head_dim or head_dim // 2
        self.qk_rope_head_dim = qk_rope_head_dim or head_dim // 2
        self.v_head_dim = v_head_dim or head_dim
        self.causal = causal
        self.rope_theta = rope_theta
        if self.qk_rope_head_dim % 2 != 0:
            raise ValueError("qk_rope_head_dim must be even")
        qk = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.qk_head_dim = qk

        self.q_down = nn.Linear(dim, self.q_lora_rank, bias=bias)
        self.q_up = nn.Linear(self.q_lora_rank, num_heads * qk, bias=bias)
        self.kv_down = nn.Linear(
            dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=bias
        )
        self.kv_up = nn.Linear(
            self.kv_lora_rank,
            num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=bias,
        )
        self.q_norm = nn.Parameter(torch.ones(self.q_lora_rank))
        self.kv_norm = nn.Parameter(torch.ones(self.kv_lora_rank))
        self.proj = nn.Linear(num_heads * self.v_head_dim, dim, bias=bias)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Run latent attention over the sequence.

        Args:
            x: Input of shape (batch, seq, dim).
            attn_mask: Optional additive or boolean mask broadcastable to
                (batch, num_heads, seq, seq).

        Returns:
            Output of shape (batch, seq, dim).
        """
        b, t, _ = x.shape
        h, nope, rope = (
            self.num_heads,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
        )

        q = self.q_up(rms_norm(self.q_down(x), self.q_norm))
        q = q.view(b, t, h, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = q.split([nope, rope], dim=-1)

        c_kv, k_rope = self.kv_down(x).split([self.kv_lora_rank, rope], dim=-1)
        kv = self.kv_up(rms_norm(c_kv, self.kv_norm))
        kv = kv.view(b, t, h, nope + self.v_head_dim).transpose(1, 2)
        k_nope, v = kv.split([nope, self.v_head_dim], dim=-1)
        k_rope = k_rope.view(b, t, 1, rope).transpose(1, 2)

        cos, sin = build_rope_cache(
            torch.arange(t, device=x.device), rope, self.rope_theta, x.dtype
        )
        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope.expand(b, h, t, rope)], dim=-1)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=self.causal and attn_mask is None,
        )
        out = out.transpose(1, 2).reshape(b, t, h * self.v_head_dim)
        return self.proj(out)
