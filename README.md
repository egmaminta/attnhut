# attnhut

A collection of Transformer Attention mechanisms in PyTorch, all in one place.

To hut something is to house it or give it shelter, and attnhut *houses* attention
mechanism implementations. Frontier labs and conference papers keep shipping new
ways to do attention (sparse, latent, compressed, corrected), but the code
usually lives buried inside a giant training repo or never gets released at all.
attnhut collects clean, readable implementations of these so a research lab can
import one and try it the same afternoon. The goal is to make it easy to study
what the frontier is doing and to build on it together in the open.

Everything is a plain `nn.Module`. You build it with ints and call it on a
`(batch, seq, dim)` tensor. No config objects, no framework, no ceremony.

```python
from attnhut import GroupedQueryAttention

attn = GroupedQueryAttention(dim=512, num_heads=8, num_kv_heads=2)
y = attn(x)            # x and y are (batch, seq, dim)
```

These are reference implementations. They are written to be read and to be
correct, not to win a kernel benchmark (see [Notes](#notes)). Each mechanism is
one short file you can read in a sitting, so you can see what DeepSeek or MiniMax
actually do and then take it from there.

## Contents

- [Install](#install)
- [Mechanisms](#mechanisms)
- [MiniMax Sparse Attention](#minimax-sparse-attention)
- [Heavily Compressed and Compressed Sparse Attention](#heavily-compressed-and-compressed-sparse-attention)
- [Causal-JEPA Attention](#causal-jepa-attention)
- [DeepSeek Sparse Attention](#deepseek-sparse-attention)
- [Delta Attention](#delta-attention)
- [Differential Attention](#differential-attention)
- [Multi-head Latent Attention](#multi-head-latent-attention)
- [GQA](#gqa)
- [BigBird](#bigbird)
- [Slot Attention](#slot-attention)
- [MQA](#mqa)
- [Standard](#standard)
- [Notes](#notes)
- [Contributing](#contributing)
- [Tests](#tests)

## Install

```bash
pip install attnhut
```

Or with uv.

```bash
uv add attnhut
```

To work on attnhut itself, clone the repo and sync the dev environment.

```bash
git clone https://github.com/egmaminta/attnhut.git
cd attnhut
uv sync
```

## Mechanisms

| Module | Idea | Reference |
|---|---|---|
| `MiniMaxSparseAttention` | top k block selection on GQA | [MiniMax, 2026](https://huggingface.co/blog/AtlasCloud-AI/minimax-goes-sparse) |
| `CompressedSparseAttention` | light compression plus selection | [DeepSeek-AI, 2026](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) |
| `HeavilyCompressedAttention` | heavy KV compression | [DeepSeek-AI, 2026](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) |
| `CausalJEPAAttention` | object level masking with bidirectional attention | [Nam et al., 2026](https://arxiv.org/abs/2602.11389) |
| `DeepSeekSparseAttention` | lightning indexer top k tokens | [DeepSeek-AI, 2025](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) |
| `DeltaAttention` | correction for sliding window | [Willette et al., 2025](https://neurips.cc/virtual/2025/poster/118545) |
| `DifferentialAttention` | two softmax maps subtracted | [Ye et al., 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html) |
| `MultiHeadLatentAttention` | low rank latent KV cache | [DeepSeek-AI, 2024](https://arxiv.org/abs/2405.04434) |
| `GroupedQueryAttention` | key value heads shared in groups | [Ainslie et al., 2023](https://aclanthology.org/2023.emnlp-main.298/) |
| `BigBirdAttention` | global plus window plus random blocks | [Zaheer et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html) |
| `SlotAttention` | iterative slots that compete | [Locatello et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html) |
| `MultiQueryAttention` | one shared key value head | [Shazeer, 2019](https://arxiv.org/abs/1911.02150) |
| `StandardAttention` | full multi head attention | [Vaswani et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) |

## MiniMax Sparse Attention

A cheap index branch scores key blocks with a single shared index key and one
index query per GQA group, max pools the token scores into block scores, and
keeps the top k blocks per group. The main attention then runs over the selected
blocks. Selection is block level.

```python
from attnhut import MiniMaxSparseAttention, msa_index_aux_loss

attn = MiniMaxSparseAttention(dim, num_heads, num_kv_groups, block_size=64, top_k=8)
out, aux = attn(x, return_aux=True)
```

Hard top k is not differentiable, so the index projections get no gradient from
the forward pass. Add `msa_index_aux_loss(aux["block_scores"], aux["attn_weights"],
block_size, group_size)` to the loss to train the selector.

## Heavily Compressed and Compressed Sparse Attention

The DeepSeek V4 hybrid pair. Both pool every few tokens into one KV entry with
learned position biased softmax weights, then run shared key value MQA over the
compressed entries plus a short uncompressed sliding window, with a learnable
attention sink. HCA compresses hard and keeps everything. CSA compresses lightly
with overlap and then selects the top k entries with a lightning indexer.

```python
HeavilyCompressedAttention(dim, num_heads, compression_rate=16, window=64)
CompressedSparseAttention(dim, num_heads, compression_rate=4, top_k=16, window=64)
```

## Causal-JEPA Attention

A world model predictor that learns how objects interact by hiding objects and
asking attention to put them back. Object slots are laid out as a grid of time by
objects and flattened into one sequence. A random set of objects is hidden across
the history and the whole future is hidden, then a bidirectional transformer
rebuilds every hidden slot from the visible ones. Each hidden slot starts as a
learned mask token plus a temporal embedding plus a linear projection of that
object at the first frame, which keeps its identity. Nothing encodes object
order, so the predictor does not care how the slots are arranged.

Input is (batch, history_frames, num_slots, dim) and the forward pass returns the
full (batch, total_frames, num_slots, dim) grid along with the hidden object
indices. Call `predict` to roll out the future from a fully visible history.

```python
from attnhut import CausalJEPAAttention, cjepa_masked_loss

attn = CausalJEPAAttention(dim, num_heads, num_slots, history_frames, pred_frames=1)
pred, masked = attn(slots)
loss = cjepa_masked_loss(pred, target, masked, history_frames)
```

## DeepSeek Sparse Attention

A small lightning indexer scores every query key pair with gated ReLU instead of
softmax, then each query keeps only its top k keys. Selection is token level,
which is the difference from MiniMax block selection.

```python
from attnhut import DeepSeekSparseAttention, dsa_index_aux_loss

attn = DeepSeekSparseAttention(dim, num_heads, top_k=64)
out, aux = attn(x, return_aux=True)
```

Train the indexer with `dsa_index_aux_loss(aux["index_scores"], dense_attn_weights)`,
a KL warmup toward the dense attention spread.

## Delta Attention

Sliding window attention shifts the output away from full attention because each
row renormalizes over a different key set. Delta Attention runs full attention on
every stride th query, takes the gap between dense and sparse at those anchors,
and adds it back to every row in the block. Training free, so you can bolt it
onto a model you already have.

```python
DeltaAttention(dim, num_heads, window=256, sink=4, stride=16)
```

## Differential Attention

Each head computes two softmax attention maps and subtracts one from the other,
scaled by a learned lambda. The common noise in the two maps cancels, so the head
puts less weight on irrelevant context. A per head RMSNorm keeps the magnitude in
line. lambda_init grows with depth, so pass the layer index.

```python
DifferentialAttention(dim, num_heads, depth=0, causal=True)
```

## Multi-head Latent Attention

Keys and values are cached as one low rank latent per token instead of full per
head K and V. Position rides on a small decoupled RoPE part kept out of the
latent. This is the trick behind DeepSeek's tiny KV cache.

```python
MultiHeadLatentAttention(dim, num_heads, kv_lora_rank=None, q_lora_rank=None)
```

## GQA

Query heads share key value heads in groups, sitting between full multi head
attention and MQA. This is what most current open models use.

```python
GroupedQueryAttention(dim, num_heads, num_kv_heads, causal=False)
```

## BigBird

A good first sparse attention to read. A query attends to a few global blocks, a
local window of neighbor blocks, and a few random blocks. The union of those is a
boolean mask and the rest is ordinary masked softmax, so the pattern is the only
new idea.

```python
BigBirdAttention(dim, num_heads, block_size=64, num_window_blocks=3,
                 num_random_blocks=3, num_global_blocks=2, causal=False)
```

## Slot Attention

Object centric attention. A small set of slots compete to explain the input
features, with the softmax taken over the slots rather than the inputs, and the
slots refined over a few iterations with a GRU. Unlike the others this maps a set
of inputs to a set of slots, so the output is (batch, num_slots, dim).

```python
SlotAttention(dim, num_slots, num_iters=3)
```

## MQA

All query heads read a single key value head, which shrinks the KV cache the
most.

```python
MultiQueryAttention(dim, num_heads, causal=False)
```

## Standard

Plain multi head attention where every query head keeps its own key value head.
The reference point the others compress or sparsify.

```python
StandardAttention(dim, num_heads, causal=False, dropout=0.0)
```

## Notes

The sparse modules are masked dense references. The selection logic is exact and
easy to read, but the wall clock speedups reported in the papers need fused
gather kernels that are out of scope here. The same applies to the V4 compressed
modules, where partial RoPE is left out for clarity. In other words, use these to
understand a mechanism and to prototype, not as a drop in fast kernel.

## Contributing

Pull requests are welcome. If a lab or a paper has an attention variant worth
reading, this is a good home for a clean version of it. The bar is one file per
mechanism, a plain `nn.Module` with the same call shape as the rest, a short test
that checks shape and causality, and a pointer to the paper. Keep it readable
over clever.

## Tests

```bash
uv run pytest
```

## License

MIT.

## Built by

Emmanuel G. Maminta (emmanuel.maminta@eee.upd.edu.ph, egmaminta@up.edu.ph),
Ubiquitous Computing Laboratory, Artificial Intelligence Graduate Program,
University of the Philippines, Diliman, Quezon City, Philippines.

## How to cite

If attnhut helped your research or project, please cite it.

```bibtex
@software{maminta2026attnhut,
  author  = {Maminta, Emmanuel G.},
  title   = {attnhut: A collection of Transformer Attention mechanisms in PyTorch},
  year    = {2026},
  url     = {https://github.com/egmaminta/attnhut},
  version = {0.4.0},
}
```
