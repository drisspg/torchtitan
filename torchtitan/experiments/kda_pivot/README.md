# KDA gate-reference (pivot) A/B

Does a future-dependent gate reference in the KDA intra-chunk kernel change what a
model learns? Two identical training runs differ only in
`ATTN_GYM_KDA_GATE_REFERENCE` (`causal` = production, `midpoint` = row 8 of each
16-token strip). The Attention Gym switch lives on the `experiment/kda-pivot` branch
of the sibling `attention-gym` checkout (`attn_gym/linear/kda/gate_reference.py`).

Layout (parent of this checkout):

```
kda-pivot-experiment/
  attention-gym/   experiment/kda-pivot branch, owns .venv (torchtitan installed into it)
  torchtitan/      this checkout, experiment/kda-pivot branch
  data/c4/en/      allenai/c4 en shards 00000-00005 + validation 00000 (json.gz)
  assets/hf/Qwen3-0.6B/   tokenizer (Kimi's tiktoken tokenizer is not loadable here)
  runs/            local dump folders
```

## Configs

- `kda_pivot_debugmodel`: 4-layer, dim 256, test tokenizer, `c4_test`; plumbing checks.
- `kda_pivot_pilot`: 12-layer text-only Kimi K3 topology (9 KDA + 3 MLA, dense FFN,
  no vision/MoE), dim 1024, 8 heads, KDA head_dim 128, 520M params (208M non-embedding),
  seq 1024, 32K tokens/microbatch/rank, local C4 shards, validation every 250 steps.

Both configs disable Attention Gym autotuning under `--debug.deterministic`
(`torchtitan/models/kimi_k3/kda.py`); without that, repeated runs diverged at step 4.

## Local

```zsh
source ../attention-gym/.venv/bin/activate
export ATTN_GYM_KDA_GATE_REFERENCE=causal   # or midpoint
gpu-run auto -- torchrun --standalone --nproc_per_node=1 -m torchtitan.train \
  --module torchtitan.experiments.kda_pivot --config kda_pivot_pilot \
  --debug.seed 42 --debug.deterministic --metrics.enable_tensorboard \
  --training.steps 4000 --dump_folder ../runs/pilot_causal
```

Evaluate a checkpoint (parallel vs prefix-only NLL, bucketed by strip offset):

```zsh
gpu-run auto -- torchrun --standalone --nproc_per_node=1 \
  -m torchtitan.experiments.kda_pivot.eval_prefix \
  --eval-output ../runs/pilot_causal/eval_prefix.json --eval-seq-len 256 --eval-num-sequences 64 \
  --module torchtitan.experiments.kda_pivot --config kda_pivot_pilot \
  --debug.seed 42 --debug.deterministic --training.steps 4000 \
  --dump_folder ../runs/pilot_causal --eval-steps 500 1000 1500 2000 2500 3000 3500 4000
```

Set `ATTN_GYM_KDA_GATE_REFERENCE` to the arm being evaluated: the question is whether
the trained weights depend on the future signal, so evaluation uses the same kernel.

## MAST

Packages built and preflighted on 2026-09-09 (`mastjob local` on 2x GB200, 3 steps):

```zsh
cd ~/dotfiles/.ai/skills/mast-interactive/mast_play
pins=(--workspace-fbpkg interactive_mast:33420b4a1728d914fd24dcd81b0d3ca0
      --conda-fbpkg torchx_base_conda_env:a9453be15a2b2f87811785dd9395a28a)
script=repos/torchtitan/torchtitan/experiments/kda_pivot/mast_launch.py
mastjob launch --tenant pytorch --h gb300 --nnodes 2 --name kdapivot-causal "${pins[@]}" -- $script --variant causal --seed 42 --steps 4000
mastjob launch --tenant pytorch --h gb300 --nnodes 2 --name kdapivot-midpoint "${pins[@]}" -- $script --variant midpoint --seed 42 --steps 4000
```

2 nodes x 4 GPUs = dp_shard 8 = 262K tokens/step; 4000 steps = 1.05B tokens, roughly
35-45 min per arm at the 64K tok/s/GPU measured on GB200. Both arms must use the same
`--nnodes`, `--seed` and `--steps`. Data (`kda_pivot_data/`) is staged under
`manifold://pytorch_distributed/tree/drisspg/mast_play/` and copied to node-local disk
at startup; W&B runs offline into the dump folder.

After `mastjob fetch <job>`:

```zsh
wandb sync ~/.mast_play/results/<job>/<variant>-seed42/tb/*/wandb/offline-run-*   # -> meta.wandb.io/drisspg/kda-pivot
mastjob launch --tenant pytorch --h gb300_1 --nnodes 1 --name kdapivot-eval-causal "${pins[@]}" -- $script \
  --mode eval --variant causal --seed 42 --steps 4000 --train-job <causal job> --load-steps 500 1000 1500 2000 2500 3000 3500 4000
```

or evaluate the fetched checkpoint locally with `--train-job /abs/path/to/results/<job>`.

The workspace package freezes the code; rebuild with `--repo ../attention-gym --repo
. --conda-fbpkg torchx_base_conda_env:a9453be...` after any edit (env unchanged).

## Reading the result

The prefix eval runs only after training, sweeping the saved checkpoints (every 500 steps)
so the gap is visible as a function of training progress; the in-training `Validator` is
parallel-mode only. It skips the first 20000 validation documents so it never overlaps the
Validator's slice. `eval_prefix_step<N>.json` reports, per arm, `prefix_nll - parallel_nll` in nats for all positions
and for strip offsets 0-7 (rows a midpoint reference can leak into) vs 8-15. The primary
metric is the difference-in-differences `gap(midpoint) - gap(causal)`; the causal arm bounds
ordinary shape-dependent kernel rounding (observed ~1e-3 nats/position at init).
