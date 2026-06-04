"""BigBird block-sparse attention (Zaheer et al., 2020).

A good first sparse attention to read. Each query attends to three things: a few
global blocks, a local window of neighbor blocks, and a few random blocks. Their
union is a boolean mask and the rest is ordinary masked softmax, so the sparsity
pattern is the only new idea.

This is the readable masked-dense reference, internal global tokens (ITC). It is
O(n^2) in memory. The paper's gather/roll kernel computes the same numbers in
O(n) and is out of scope here.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .utils import causal_mask, masked_softmax_attention


class BigBirdAttention(nn.Module):
    """Global plus sliding-window plus random block-sparse attention.

    Args:
        dim: Model dimension. Must be divisible by num_heads.
        num_heads: Number of attention heads.
        block_size: Tokens per block. The sequence is padded up to a multiple.
        num_window_blocks: Width of the local window in blocks. Must be odd.
        num_random_blocks: Random blocks each query block also attends.
        num_global_blocks: Leading blocks attended by and attending to all.
        causal: If True, attention is restricted to the past.
        seed: Seed for the random block draw, so the pattern is reproducible.
        bias: Whether the linear projections carry a bias term.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        block_size: int = 64,
        num_window_blocks: int = 3,
        num_random_blocks: int = 3,
        num_global_blocks: int = 2,
        causal: bool = False,
        seed: int = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if num_window_blocks % 2 == 0:
            raise ValueError("num_window_blocks must be odd")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.block_size = block_size
        self.window = num_window_blocks
        self.random = num_random_blocks
        self.globals = num_global_blocks
        self.causal = causal
        self.seed = seed
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

    def block_mask(self, num_blocks: int, device: torch.device) -> Tensor:
        """Build the block connectivity = global | window | random.

        Args:
            num_blocks: Number of blocks in the (padded) sequence.
            device: Device for the returned mask.

        Returns:
            Boolean tensor of shape (num_blocks, num_blocks), True where a query
            block attends a key block.
        """
        m = num_blocks
        half = self.window // 2
        mask = torch.zeros(m, m, dtype=torch.bool, device=device)
        glob = range(min(self.globals, m))

        for g in glob:
            mask[g, :] = True
            mask[:, g] = True

        hi = 0 if self.causal else half
        for j in range(m):
            for o in range(-half, hi + 1):
                k = j + o
                if 0 <= k < m:
                    mask[j, k] = True

        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        for j in range(m):
            if j in glob:
                continue
            taken = set(glob) | {j + o for o in range(-half, half + 1)}
            cand = [
                k
                for k in range(m)
                if k not in taken and (not self.causal or k < j - half)
            ]
            if not cand:
                continue
            cand_t = torch.tensor(cand)
            pick = torch.randperm(len(cand), generator=gen)[: self.random]
            mask[j, cand_t[pick].to(device)] = True
        return mask

    def forward(
        self, x: Tensor, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        """Run block-sparse attention over the sequence.

        Args:
            x: Input of shape (batch, seq, dim).
            key_padding_mask: Optional boolean mask of shape (batch, seq), True
                for real tokens and False for padding.

        Returns:
            Output of shape (batch, seq, dim).
        """
        b, t, _ = x.shape
        bs = self.block_size
        pad = (-t) % bs
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        n = t + pad
        device = x.device

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        blocks = self.block_mask(n // bs, device)
        keep = blocks.repeat_interleave(bs, 0).repeat_interleave(bs, 1)
        if self.causal:
            keep = keep & causal_mask(n, n, device)
        keep = keep[None, None]

        valid = torch.ones(b, n, dtype=torch.bool, device=device)
        if pad:
            valid[:, t:] = False
        if key_padding_mask is not None:
            valid[:, :t] &= key_padding_mask.bool()
        keep = keep & valid[:, None, None, :]

        out = masked_softmax_attention(q, k, v, keep, self.head_dim**-0.5)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.proj(out)[:, :t]
