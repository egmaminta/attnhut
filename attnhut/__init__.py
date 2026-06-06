"""attnhut, attention mechanisms from frontier labs as plug-n-play modules.

Every mechanism is an nn.Module you build with plain ints and call on a
(batch, seq, dim) tensor.

    from attnhut import GroupedQueryAttention
    attn = GroupedQueryAttention(dim=512, num_heads=8, num_kv_heads=2)
    y = attn(x)
"""

from .bigbird import BigBirdAttention
from .cjepa import CausalJEPAAttention, cjepa_masked_loss
from .compressed import CompressedSparseAttention, HeavilyCompressedAttention
from .deepseek_sparse import (
    DeepSeekSparseAttention,
    LightningIndexer,
    dsa_index_aux_loss,
)
from .delta import DeltaAttention
from .differential import DifferentialAttention
from .gqa import GroupedQueryAttention
from .minimax_sparse import MiniMaxSparseAttention, msa_index_aux_loss
from .mla import MultiHeadLatentAttention
from .mqa import MultiQueryAttention
from .slot import SlotAttention
from .standard import StandardAttention

__version__ = "0.4.1"

__all__ = [
    "StandardAttention",
    "MultiQueryAttention",
    "GroupedQueryAttention",
    "BigBirdAttention",
    "MiniMaxSparseAttention",
    "msa_index_aux_loss",
    "DeepSeekSparseAttention",
    "LightningIndexer",
    "dsa_index_aux_loss",
    "MultiHeadLatentAttention",
    "HeavilyCompressedAttention",
    "CompressedSparseAttention",
    "DeltaAttention",
    "DifferentialAttention",
    "SlotAttention",
    "CausalJEPAAttention",
    "cjepa_masked_loss",
]
