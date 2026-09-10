# Channel-capacity probe: midpoint-minus-causal KDA rounding residual

Question: the midpoint gate reference makes strip rows 0-7 of the KDA intra-chunk kernel
depend on a *future* gate (strip row 8) through BF16 rounding. Training with it showed no
exploitation (autoregressive-minus-parallel NLL difference-in-differences +0.00001 +-
0.00005 nats at 1.45B params / 4B tokens). Is that because the rounding residual carries
essentially no usable information about the future ("informationally empty channel")?

Setup: trained causal-arm checkpoint `kdascaled-causal-xf3jj3jb` step 7600 (`kda_pivot_scaled`,
1.45B params), 1024 held-out C4 validation sequences (documents 20000+, disjoint from the
training validator) x 256 input tokens, B=1 per sequence, first two KDA layers (model
layers 0 and 1). Exact `chunk_kda` inputs were captured from the model's forward under
`ATTN_GYM_KDA_GATE_REFERENCE=causal` and re-run under both references
(`resolve_gate_reference.cache_clear()` between; the causal re-run matched the recorded
model output bitwise). Parallel NLL over the capture: 3.238 nats. Probes are trained on
sequences 0-639, early-stopped / hyper-parameter-selected on 640-767, and every number below
is on held-out sequences 768-1023. Data: `../runs/probe/capture_1024.pt` (15 GB, not
committed), results `../runs/probe/probe_1024_seed{0,1}.json`.

## Commands

```zsh
cd ~/meta/kda-pivot-experiment/torchtitan && source ../attention-gym/.venv/bin/activate
export ATTN_GYM_KDA_GATE_REFERENCE=causal HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
ckpt=~/.mast_play/results/kdascaled-causal-xf3jj3jb/causal-seed42/checkpoint
gpu-run --timeout 1800 auto -- timeout --signal=TERM --kill-after=10s 3000s torchrun --standalone --nproc_per_node=1 -m torchtitan.experiments.kda_pivot.probe.capture --capture-output ../runs/probe/capture_1024.pt --capture-num-sequences 1024 --capture-num-layers 2 --module torchtitan.experiments.kda_pivot --config kda_pivot_scaled --debug.seed 42 --debug.deterministic --training.steps 7600 --dump_folder ../runs/probe/dump --checkpoint.folder $ckpt --checkpoint.load_step 7600
gpu-run --timeout 1800 auto -- timeout --signal=TERM --kill-after=10s 3000s python -m torchtitan.experiments.kda_pivot.probe.probe --capture ../runs/probe/capture_1024.pt --output ../runs/probe/probe_1024_seed0.json --seed 0
```

Capture: ~80 s (1024 forwards + 4096 kernel calls). Probe: ~4.5 min per seed on one GB200.

## 1. Residual statistics (`out_midpoint - out_causal`, float32)

Per layer over 1024 x 256 x 16 heads x 128 = 537M elements. "1 ulp" means |residual| equals the
BF16 spacing at the larger of the two outputs.

| layer | rows | frac nonzero | of nonzero, exactly 1 ulp | mean abs (nonzero) | max abs | mean abs out_causal |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | all | 0.485 | 0.596 | 1.0e-6 | 2.4e-4 | 2.35e-4 |
| 0 | leakable (p%16<8) | 0.484 | 0.595 | 1.0e-6 | 2.4e-4 | 2.34e-4 |
| 0 | causal (p%16>=8) | 0.487 | 0.596 | 1.0e-6 | 2.4e-4 | 2.36e-4 |
| 1 | all | 0.445 | 0.624 | 4.4e-6 | 4.9e-4 | 1.05e-3 |
| 1 | leakable | 0.443 | 0.624 | 4.4e-6 | 4.9e-4 | 1.05e-3 |
| 1 | causal | 0.447 | 0.625 | 4.4e-6 | 4.9e-4 | 1.06e-3 |

Nonzero fraction by strip row (layer 0): 0.42 (row 0), 0.48, 0.49, 0.50 ... 0.50, 0.43 (row 8),
0.49, ..., 0.50. Every token has at least one differing element.

What this shows: changing the reference row perturbs *every* row's rounding (rows 0-7 through a
different rebase constant, rows 8-15 likewise, and everything downstream through the carried
state), so ~45-49% of output elements differ at the ~0.4% relative level (about 1 BF16 ulp).
Rows 0 and 8 differ least because each is the exact reference row under one of the two
variants. Leakable and causal rows are statistically indistinguishable in footprint and
magnitude; the earlier "~9% of prefix elements" figure (Stage A cut test) measured only the
future-*dependent* part of this residual, which is a subset of what is tabulated here. The
probes below see the full residual, i.e. everything the midpoint model would have seen.

## 2. Regression probes: does the residual reveal the future gate?

Leakable rows only. Sample = (sequence, position with p%16<8, head); features and targets are
128-vectors. Targets (natural-log gate units, chunk-local cumulative): `future_sum` = cumulative
gate at strip row 8 minus at the current row (the strictly future quantity the rebase leaks),
`reference` = cumulative gate at row 8 itself, `current` = the row's own cumulative gate (fully
causal control). Held-out R^2 (variance-weighted over the 128 outputs; 1 = perfect, 0 = predict
the train mean). Ridge lambda and MLP (128-256-128, GELU) weight decay chosen on validation.
Train 1.3M samples, held-out 524K. Two probe seeds gave the same values to +-0.0003.

| layer | features | future_sum ridge | future_sum MLP | reference MLP | current (causal) MLP |
| --- | --- | --- | --- | --- | --- |
| 0 | residual (raw) | +0.0000 | +0.0011 | +0.0018 | +0.0014 |
| 0 | residual (ulp units) | -0.0000 | +0.0001 | -0.0000 | +0.0001 |
| 0 | shuffled residual (ulp) | -0.0000 | -0.0001 | -0.0001 | -0.0002 |
| 0 | out_causal | +0.0011 | +0.0040 | +0.0050 | +0.0033 |
| 0 | out_causal + residual (ulp) | +0.0010 | +0.0038 | +0.0040 | +0.0027 |
| 1 | residual (raw) | -0.0000 | +0.0207 | +0.0237 | +0.0177 |
| 1 | residual (ulp units) | -0.0000 | +0.0009 | +0.0013 | +0.0009 |
| 1 | shuffled residual (ulp) | -0.0000 | -0.0002 | -0.0001 | -0.0001 |
| 1 | out_causal | +0.0118 | +0.0601 | +0.0649 | +0.0488 |
| 1 | out_causal + residual (ulp) | +0.0117 | +0.0589 | +0.0636 | +0.0474 |

Reading:

- Linear probes on the residual explain nothing (R^2 = 0 to 5 decimals) for any target.
- The raw residual's MLP R^2 (0.02 in layer 1) is not future information: the raw residual's
  magnitude is the BF16 spacing of `out_causal`, so it encodes |out_causal|, and `out_causal`
  itself predicts the gate with R^2 0.06. Rescaling to ulp units removes the magnitude and
  leaves R^2 ~ 0.001; adding the residual to `out_causal` never improves on `out_causal`
  alone (slightly worse, as extra dimensions with finite data should be).
- The remaining R^2 ~ 0.001 (10x the shuffled control, so probably not zero) is not specific
  to the future: the fully causal `current` target gets the same value. The residual weakly
  reveals the gate *regime* (how much decay is active in the strip), which the past and future
  gates share; it does not resolve the future gate beyond what the past already says.

## 3. Next-token probes: does the residual help predict token p+1?

Leakable rows only, one sample per (sequence, position); features flatten the 16 heads
(2048-d each for residual-in-ulp-units and `out_causal`, 4096-d concatenated). Labels
restricted to the 1024 most frequent next tokens in the training split (covers 64% of held-out
positions): 53,940 train / 21,018 held-out samples. Probes: logistic (linear) and MLP
(hidden 1024, GELU), AdamW, output bias initialised to the train log-marginal, weight decay in
{1e-2, 1e-1, 1} and epoch chosen on validation. Standard errors are by held-out sequence
(n = 256). Cross-entropy in nats, lower is better.

Held-out cross-entropy / top-1 accuracy (seed 0; seed 1 in parentheses where it differs by
more than 0.005):

| features | probe | layer 0 CE | layer 0 acc | layer 1 CE | layer 1 acc |
| --- | --- | --- | --- | --- | --- |
| marginal frequency | -- | 5.434 | 0.055 | 5.434 | 0.055 |
| out_causal | logistic | 5.165 | 0.109 | 4.964 (4.951) | 0.133 |
| out_causal | MLP | 5.010 | 0.122 | 4.715 (4.708) | 0.150 |
| residual (ulp) | logistic | 5.569 | 0.049 | 5.566 | 0.050 |
| residual (ulp) | MLP | 5.477 | 0.055 | 5.478 | 0.054 |
| shuffled residual (ulp) | logistic | 5.570 | 0.049 | 5.567 | 0.050 |
| shuffled residual (ulp) | MLP | 5.478 | 0.053 | 5.483 (5.487) | 0.055 |
| out_causal + residual (ulp) | logistic | 5.208 | 0.105 | 5.019 (5.012) | 0.126 |
| out_causal + residual (ulp) | MLP | 5.056 | 0.115 | 4.818 (4.815) | 0.137 |
| out_causal + shuffled residual | logistic | 5.204 (5.210) | 0.107 | 5.021 (5.017) | 0.127 |
| out_causal + shuffled residual | MLP | 5.055 (5.059) | 0.116 | 4.816 (4.820) | 0.140 |

Paired per-sample CE differences (positive = the first-named probe is worse, i.e. the second
one gained), mean +- stderr by sequence, nats:

| quantity | probe | layer 0 seed 0 | layer 0 seed 1 | layer 1 seed 0 | layer 1 seed 1 |
| --- | --- | --- | --- | --- | --- |
| out_causal -> out_causal + residual | logistic | -0.043 +- 0.005 | -0.050 +- 0.005 | -0.055 +- 0.004 | -0.061 +- 0.004 |
| out_causal -> out_causal + residual | MLP | -0.046 +- 0.003 | -0.049 +- 0.003 | -0.103 +- 0.007 | -0.107 +- 0.007 |
| out_causal -> out_causal + shuffled | logistic | -0.039 +- 0.004 | -0.049 +- 0.004 | -0.058 +- 0.004 | -0.066 +- 0.004 |
| out_causal -> out_causal + shuffled | MLP | -0.045 +- 0.003 | -0.048 +- 0.003 | -0.101 +- 0.007 | -0.112 +- 0.007 |
| (out_causal + shuffled) -> (out_causal + residual) | logistic | -0.004 +- 0.005 | -0.001 +- 0.005 | +0.003 +- 0.005 | +0.004 +- 0.005 |
| (out_causal + shuffled) -> (out_causal + residual) | MLP | -0.001 +- 0.003 | -0.001 +- 0.003 | -0.002 +- 0.004 | +0.005 +- 0.004 |
| shuffled residual -> residual (alone) | logistic | +0.001 +- 0.005 | +0.002 +- 0.005 | +0.001 +- 0.005 | +0.005 +- 0.005 |
| shuffled residual -> residual (alone) | MLP | +0.001 +- 0.003 | +0.000 +- 0.003 | +0.005 +- 0.003 | +0.009 +- 0.004 |

Reading:

- The probe works on honest features: `out_causal` of layer 1 beats the marginal by 0.47
  (logistic) / 0.72 (MLP) nats and nearly triples top-1 accuracy. Layer 0 (a function of the
  current token's embedding and its KDA state only) gives 0.27 / 0.42 nats.
- The residual alone is no better than a shuffled residual and both are slightly *worse* than
  the marginal predictor (+0.04-0.13 nats): 2048 noise dimensions over 54K samples cost a
  little even with weight decay and early stopping.
- Adding the residual to `out_causal` makes every probe worse by 0.04-0.11 nats, and adding a
  shuffled residual costs the same amount. The paired real-vs-shuffled difference, which
  cancels that overfitting cost, is 0.000 +- 0.005 nats: layer 0 in [-0.004, -0.001], layer 1
  in [-0.002, +0.005], never more than 1.4 standard errors from zero, with no consistent sign
  across probes or seeds. The residual-alone vs shuffled-alone comparison is likewise within
  noise except one cell (layer 1, seed 1, MLP: +0.009 +- 0.004, 2.4 standard errors), which
  compares two probes that are both below the marginal predictor and is not reproduced by the
  augmented comparison or the other seed; it is treated as the ceiling, not a detection.

## Conclusion

The channel exists but is, for practical purposes, empty. Under the midpoint reference about
half of the first two KDA layers' output elements change, almost always by one BF16 ulp
(~0.4% relative), on leakable and causal rows alike. Probed on the leakable rows of 1024
held-out sequences, that residual carries no linearly decodable information about the future
gate (R^2 = 0.0000), and an MLP extracts R^2 ~ 0.001 of the future gate's variance -- the same
R^2 it gets for the strictly causal current gate, so even that sliver is "how much decay is
active", not the future per se. For next-token prediction, the best estimate of the channel
capacity is the paired CE improvement of the residual-augmented probe over the same probe with a
shuffled residual: **0.000 +- 0.005 nats per leakable position (0.00 +- 0.007 bits)**, with a
95% upper bound of about **0.01-0.02 nats (0.02-0.03 bits)** from the worst cells (layer 1, seed
1); the naive
"add the residual to the causal features" estimate is *negative* (-0.04 to -0.11 nats) because the
extra dimensions overfit, and identically so for a shuffled residual. For scale, the honest
causal features of the same layers are worth 0.4-0.7 nats to the same probes, and the training
experiment bounded exploitation at ~1e-4 nats. Not shown: information that only a probe with far
more data or a deeper readout could find below ~0.01 nats; the same measurement on later KDA
layers (only layers 0-1 were captured, and the residual compounds through depth); and whether a
model could *amplify* the residual during training (the A/B run says it did not at this scale).
The result is consistent with the training null being a property of the channel rather than
of the optimizer: the future-dependent rounding is real but not readable.
