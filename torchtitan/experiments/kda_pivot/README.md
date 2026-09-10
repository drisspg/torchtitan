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
- `kda_pivot_scaled`: same topology at dim 2048, 16 heads, FFN 8192: 1.45B params (830M
  non-embedding, the MatX study's size), 24 C4 shards (~4B tokens), 7600 steps at 16 ranks x
  32K tokens, lr 4e-4, checkpoints every 1000 steps (8.2 GB each). Measured 38K tok/s and
  49.7 GiB per GB200 -> about 2 h per arm on 4 nodes x 4 GB300.

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

Evaluate checkpoints (parallel vs autoregressive NLL, bucketed by strip offset). The
default `--eval-mode recurrent` runs KDA through Attention Gym's recurrent kernel (one
forward per sequence, seconds per checkpoint); `--eval-mode prefix` is the literal
per-prefix oracle (~200x slower) and agreed with recurrent to within 2e-4 nats on both
trained arms:

```zsh
gpu-run auto -- torchrun --standalone --nproc_per_node=1 \
  -m torchtitan.experiments.kda_pivot.eval_prefix \
  --eval-output ../runs/pilot_causal/eval_recurrent.json --eval-seq-len 256 --eval-num-sequences 64 \
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
exp=pilot-520m-1b   # W&B group + tag shared by both arms; one per model/data scale
mastjob launch --tenant pytorch --h gb300 --nnodes 2 --name kdapivot-causal "${pins[@]}" -- $script --experiment $exp --variant causal --seed 42 --steps 4000
mastjob launch --tenant pytorch --h gb300 --nnodes 2 --name kdapivot-midpoint "${pins[@]}" -- $script --experiment $exp --variant midpoint --seed 42 --steps 4000
```

2 nodes x 4 GPUs = dp_shard 8 = 262K tokens/step; 4000 steps = 1.05B tokens, roughly
35-45 min per arm at the 64K tok/s/GPU measured on GB200. Both arms must use the same
`--nnodes`, `--seed` and `--steps`. Data (`kda_pivot_data/`) is staged under
`manifold://pytorch_distributed/tree/drisspg/mast_play/` and copied to node-local disk
at startup; W&B runs offline into the dump folder.

After `mastjob fetch <job>`:

```zsh
wandb sync ~/.mast_play/results/<job>/<variant>-seed42/tb/*/wandb/offline-run-*   # training run -> meta.wandb.io/drisspg/kda-pivot
# recurrent eval is seconds per checkpoint: fetch the checkpoints and run it locally
MAST_PLAY_OUT=../runs/eval torchrun --standalone --nproc_per_node=1 $script --mode eval --experiment $exp --variant causal --seed 42 --steps 4000 \
  --train-job ~/.mast_play/results/<causal job> --load-steps 500 1000 1500 2000 2500 3000 3500 4000 --local-data ../data_local
python -m torchtitan.experiments.kda_pivot.wandb_log_eval --eval-dir ../runs/eval/causal-seed42 --experiment $exp --arm causal --seed 42
```

(or run the eval on MAST with `mastjob launch --h gb300_1 --nnodes 1 ... -- $script --mode eval ...`
and `--train-job <job name>`, which reads the checkpoint from the results mount).

### W&B layout (project `kda-pivot`)

- One **group per experiment** (`--experiment`, e.g. `pilot-520m-1b`); tags `<arm>`, `<experiment>`, `<config>`.
- Training runs `<arm>-seed<seed>` (`job_type=train`): torchtitan metrics, parallel-mode `validation_metrics/loss`.
- Eval runs `<arm>-seed<seed>-eval` (`job_type=eval`, logged by `wandb_log_eval.py` with x = checkpoint step):
  `eval/<bucket>/{autoregressive_nll,parallel_nll,gap_nats,gap_stderr_by_sequence}` for buckets
  `all`, `leakable_rows`, `causal_rows`, plus `eval/leakable_minus_causal_rows_gap`.
- Arms are `causal`, `midpoint`, or `leak<eps>` (positive control, `--leak-control <eps>`).

### Positive control

`--leak-control 0.5` adds `0.5 * v[t+1]` to the *parallel* KDA output only (chunked path), a
deliberate next-token leak. 300 steps of the pilot config (group `control-300steps`): parallel
train loss 0.025 vs 5.69 clean; eval gap **+9.98 nats** (recurrent) / **+9.72** (per-prefix
oracle) vs +0.00005 for the clean model. The evaluator sees exploitation when it exists.

The workspace package freezes the code; rebuild with `--repo ../attention-gym --repo
. --conda-fbpkg torchx_base_conda_env:a9453be...` after any edit (env unchanged).

## Reading the result

The autoregressive eval runs only after training, sweeping the saved checkpoints (every 500 steps)
so the gap is visible as a function of training progress; the in-training `Validator` is
parallel-mode only. It skips the first 20000 validation documents so it never overlaps the
Validator's slice. `eval_recurrent_step<N>.json` reports, per arm, `autoregressive_nll - parallel_nll` in nats for
`all` positions and for two disjoint halves of every 16-token KDA strip: `leakable_rows` (strip
rows 0-7, i.e. `p % 16 < 8`, the only rows whose gate rebase can involve a future gate under the
midpoint reference) and `causal_rows` (rows 8-15, causal under both references). The primary
metric is the difference-in-differences `gap(midpoint) - gap(causal)`; the causal arm bounds
ordinary shape-dependent kernel rounding (observed ~1e-3 nats/position at init).
