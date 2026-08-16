# KDA hybrid training experiment

This experiment turns the integrated KDA module from
`attention-gym/examples/kda_training.py` into a TorchTitan decoder stack. Each
four-layer group contains three fused KDA blocks followed by one global causal
varlen-attention block. Both mixers use the same packed-document boundaries.

Two random-initialized recipes are available:

- `kda_hybrid_debugmodel`: 8 layers, physical token capacity `T=256`, and
  sequence capacity `N=32`.
- `kda_hybrid_qwen3_8b`: Qwen3-8B dimensions (36 layers, width 4096, dense FFN,
  and the Qwen3 vocabulary), `T=2048`, and `N=128`. Replacing 27 GQA layers
  with KDA adds parameters, so this is Qwen3-8B-shaped at 8.93B parameters
  rather than an exact 8B model.

## Environment

The experiment intentionally keeps `attention-gym` optional. Install the local
checkout into the environment used to launch TorchTitan:

```bash
uv pip install --python ~/.venvs/nightly/bin/python -e ../attention-gym
```

The fused KDA path currently requires a Blackwell GPU, BF16 projection compute,
head dimension 128, and chunk size 64. Each KDA mixer is a nested FP32 FSDP
unit so its decay parameters remain FP32; its projection kernels still cast
weights and activations to BF16. Global attention, FFNs, embeddings, and the LM
head use the recipe's normal BF16 mixed-precision policy. Keeping the complete
KDA mixer in the nested FP32 unit is a correctness-first compromise: projection
kernels compute in BF16, but KDA parameter materialization/all-gather is still
FP32. A narrower child unit for only the strict FP32 decay state is a future
memory/bandwidth optimization.

## Fixed-capacity CUDA Graph contract

A captured graph owns fixed physical capacities:

- `T`: allocated token rows
- `N`: maximum logical subsequences

A replay may use an active prefix of `L <= T` tokens containing `M <= N`
subsequences. Its `cu_seqlens` tensor always has shape `[N + 1]`; entries after
the real endpoint repeat `L`:

```text
[0, sequence_start_1, ..., L, L, ..., L]
```

The custom trainer constructs this metadata on CPU before capture, masks labels
after `L`, and copies the fixed-shape tensor into graph-owned storage. The debug
recipe cycles through `L = 256, 192, 96` while document packing independently
changes `M`. Set `--min-active-tokens -1` to keep `L=T` on every batch.

Attention-gym KDA masks inactive values and gradient paths. The global varlen
attention path needs equivalent barriers because both its output suffix and its
Q/K/V input-gradient suffix are outside the primitive's contract past `L`.
The wrapper sanitizes the output before its projection and masks before other
parameter-gradient reductions as well as at block boundaries. This also
prevents stale NaNs in inactive token storage from contaminating weight
gradients.

This capacity encoding is explicit in attention-gym KDA, but it is not yet a
public PyTorch `varlen_attn` guarantee. That API currently documents its physical
`T` as the sum of sequence lengths. On the tested nightly FA4 backend, `T > L`
and repeated trailing endpoints work, while inactive output, LSE, and Q/K/V
input-gradient rows remain undefined. An upstream fixed-capacity mode should
formalize the encoding and offer opt-in zeroed suffix semantics. Until then,
`capacity_aware_global_attention` is a tested integration boundary rather than a
portable primitive contract.

The current implementation optimizes for semantic validation. Dense projections,
normalization, and FFN operations still execute over physical capacity `T`; KDA
and global varlen kernels can skip work based on `L` and the padded sequence
schedule. Fixed `N` also launches schedule/attention work for padded empty
sequences, whose kernels must early-exit. Reported TPS counts active tokens
rather than physical rows, while the FLOP estimate omits the KDA scan and still
models the configured physical sequence length, so TPS/MFU are diagnostic
rather than benchmark metrics. Profiling and replacing the remaining dense
operations with sequence-aware variants is the next performance step.

## Run

Run the graph-replay integration on one exclusively reserved GPU:

```bash
gpu-run auto -- ~/.venvs/nightly/bin/torchrun --standalone --nproc-per-node=1 \
  -m torchtitan.train --module kda_hybrid --config kda_hybrid_debugmodel
```

Run the Qwen3-8B-shaped model similarly:

```bash
gpu-run auto -- ~/.venvs/nightly/bin/torchrun --standalone --nproc-per-node=1 \
  -m torchtitan.train --module kda_hybrid --config kda_hybrid_qwen3_8b
```

For an eager reference run, add `--training.disable-cuda-graphs`. Override
`--max-sequences` or `--min-active-tokens` to exercise another capacity bound.
The trainer raises if a batch contains more than `N` active documents.

The initial integration is wired for FSDP data parallelism, although the runs
reported here use one GPU. It does not support TP, CP, PP, checkpoint import,
local batches greater than one, or non-default SPMD backends. The global
capacity wrapper currently mirrors `GQAttention.forward`; capacity-aware suffix
handling should eventually move into `VarlenAttention` so changes to QKV norm,
RoPE, or output-transform ordering cannot make the copy drift.
