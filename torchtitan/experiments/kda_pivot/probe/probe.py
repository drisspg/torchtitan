# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Channel-capacity probe on the midpoint-minus-causal KDA rounding residual.

Reads a ``capture.py`` file and, per captured KDA layer:

1. residual statistics (``out_midpoint - out_causal`` in float32), split into leakable
   strip rows (``p % 16 < 8``, whose gate rebase reads the future gate at strip row 8
   under the midpoint reference) and causal rows (8-15);
2. regression probes (ridge, 2-layer MLP) from the residual at a leakable row to the
   future gate it was rebased against; controls: shuffled residual, causal output;
3. next-token probes (logistic, MLP) restricted to the most frequent labels, with
   marginal-frequency, causal-output, shuffled-residual and residual-augmented variants.

The residual is also expressed in units of the BF16 spacing of ``out_causal``
("ulp units", almost always in {-1, 0, +1}), a lossless rescaling that is the most
probe-friendly representation of the channel. Sequences are split by index into
train / validation (early stopping, ridge lambda) / held-out; every metric below is
held-out::

    python -m torchtitan.experiments.kda_pivot.probe.probe \\
        --capture ../runs/probe/capture_1024.pt --output ../runs/probe/probe_1024.json
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

STRIP = 16
CHUNK = 64
REFERENCE_ROW = 8
RIDGE_LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
WEIGHT_DECAYS = (1e-2, 1e-1, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=1024)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def bf16_ulp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """BF16 spacing (8-bit significand) at the larger magnitude of two float32 tensors.

    Taking the coarser of the two grids bounds ``(a - b) / ulp`` when one output is
    near zero, so residuals in "ulp units" stay finite and comparable.
    """
    magnitude = torch.maximum(a.abs(), b.abs()).clamp_min(
        torch.finfo(torch.bfloat16).tiny
    )
    return torch.exp2(torch.floor(torch.log2(magnitude)) - 7)


@dataclass
class Split:
    train: torch.Tensor
    validation: torch.Tensor
    holdout: torch.Tensor

    @classmethod
    def by_sequence(
        cls, num_sequences: int, holdout_fraction: float, validation_fraction: float
    ):
        holdout = int(num_sequences * holdout_fraction)
        validation = int(num_sequences * validation_fraction)
        train_end = num_sequences - holdout - validation
        index = torch.arange(num_sequences)
        return cls(index[:train_end], index[train_end:-holdout], index[-holdout:])


class Standardizer:
    def __init__(self, x: torch.Tensor):
        if not torch.isfinite(x).all():
            raise ValueError("non-finite probe features")
        self.mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        # Near-constant features stay near zero instead of exploding.
        self.std = std.clamp_min(1e-6 * std.max())

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def r2(
    target: torch.Tensor, prediction: torch.Tensor, train_mean: torch.Tensor
) -> float:
    """Variance-weighted R^2 over all output dimensions (1 = perfect, 0 = train mean)."""
    residual = ((target - prediction) ** 2).sum()
    total = ((target - train_mean) ** 2).sum()
    return 1.0 - (residual / total).item()


def ridge_r2(x: dict[str, torch.Tensor], y: dict[str, torch.Tensor]) -> float:
    """Closed-form ridge on standardized features; lambda picked on the validation split."""
    scale = Standardizer(x["train"])
    xtr = scale(x["train"]).double()
    ytr = y["train"].double()
    y_mean = ytr.mean(dim=0, keepdim=True)
    gram = xtr.T @ xtr
    cross = xtr.T @ (ytr - y_mean)
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    best = (-math.inf, None)
    for lam in RIDGE_LAMBDAS:
        weight = torch.linalg.solve(gram + lam * xtr.shape[0] * eye, cross)
        prediction = scale(x["validation"]).double() @ weight + y_mean
        score = r2(y["validation"].double(), prediction, y_mean)
        if score > best[0]:
            best = (score, weight)
    assert best[1] is not None
    prediction = scale(x["holdout"]).double() @ best[1] + y_mean
    return r2(y["holdout"].double(), prediction, y_mean)


def fit(
    x: dict[str, torch.Tensor],
    y: dict[str, torch.Tensor],
    *,
    out_dim: int,
    hidden: int | None,
    loss: str,
    seed: int,
    batch_size: int = 4096,
    max_epochs: int = 40,
    patience: int = 4,
) -> nn.Module:
    """Train a linear or 2-layer MLP probe with AdamW and validation early stopping.

    The weight decay is selected on the validation split too, so a probe on pure-noise
    features can fall back to (roughly) the marginal predictor instead of overfitting.
    """
    best = (math.inf, None)
    for weight_decay in WEIGHT_DECAYS:
        loss_value, model = fit_once(
            x,
            y,
            out_dim=out_dim,
            hidden=hidden,
            loss=loss,
            seed=seed,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
        )
        if loss_value < best[0]:
            best = (loss_value, model)
    assert best[1] is not None
    return best[1]


def fit_once(
    x: dict[str, torch.Tensor],
    y: dict[str, torch.Tensor],
    *,
    out_dim: int,
    hidden: int | None,
    loss: str,
    seed: int,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> tuple[float, nn.Module]:
    """One training run; returns the best validation loss and the model at that epoch."""
    torch.manual_seed(seed)
    in_dim = x["train"].shape[1]
    device = x["train"].device
    if hidden is None:
        model = nn.Linear(in_dim, out_dim)
    else:
        model = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
        )
    model = model.to(device)
    if loss == "ce":
        # Start at the marginal predictor so a probe on uninformative features is
        # measured against it rather than against a slowly learned bias.
        counts = torch.bincount(y["train"], minlength=out_dim).float() + 1.0
        output_layer = model if hidden is None else model[-1]
        output_layer.bias.data = torch.log(counts / counts.sum())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=weight_decay
    )
    criterion = F.mse_loss if loss == "mse" else F.cross_entropy
    best_loss, best_state, stale = math.inf, None, 0
    num_train = x["train"].shape[0]
    for _ in range(max_epochs):
        model.train()
        for index in torch.randperm(num_train, device=device).split(batch_size):
            optimizer.zero_grad()
            criterion(model(x["train"][index]), y["train"][index]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x["validation"]), y["validation"]).item()
        if val_loss < best_loss - 1e-5:
            best_loss, stale = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return best_loss, model.eval()


def mlp_r2(x: dict[str, torch.Tensor], y: dict[str, torch.Tensor], seed: int) -> float:
    scale_x, scale_y = Standardizer(x["train"]), Standardizer(y["train"])
    xs = {k: scale_x(v) for k, v in x.items()}
    ys = {k: scale_y(v) for k, v in y.items()}
    model = fit(xs, ys, out_dim=y["train"].shape[1], hidden=256, loss="mse", seed=seed)
    with torch.no_grad():
        prediction = model(xs["holdout"]) * scale_y.std + scale_y.mean
    return r2(y["holdout"], prediction, y["train"].mean(dim=0, keepdim=True))


def shuffled(features: dict[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    """Permute samples within each split: same marginals, no pairing with the target."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return {
        k: v[torch.randperm(v.shape[0], generator=generator).to(v.device)]
        for k, v in features.items()
    }


def gather(
    tensor: torch.Tensor, split: Split, mask: torch.Tensor, *, per_head: bool = False
) -> dict[str, torch.Tensor]:
    """Select ``tensor[seqs][:, mask]`` per split; samples are (sequence, position[, head])."""
    return {
        name: tensor[getattr(split, name)][:, mask].flatten(0, 2 if per_head else 1)
        for name in ("train", "validation", "holdout")
    }


def residual_stats(
    residual: torch.Tensor, out_causal: torch.Tensor, ulp: torch.Tensor
) -> dict:
    """Nonzero fraction and magnitude of the residual, overall and per strip-row bucket."""
    position = torch.arange(residual.shape[1], device=residual.device)
    row = position % STRIP
    buckets = {
        "all": row >= 0,
        "leakable_rows": row < REFERENCE_ROW,
        "causal_rows": row >= REFERENCE_ROW,
        **{f"strip_row_{r}": row == r for r in range(STRIP)},
    }
    stats = {}
    for name, mask in buckets.items():
        res = residual[:, mask]
        nonzero = res != 0
        stats[name] = {
            "frac_nonzero": nonzero.float().mean().item(),
            "frac_nonzero_is_one_ulp": (
                (res.abs() == ulp[:, mask])[nonzero].float().mean().item()
            ),
            "mean_abs_nonzero": res.abs()[nonzero].mean().item(),
            "max_abs": res.abs().max().item(),
            "mean_abs_out": out_causal[:, mask].abs().mean().item(),
            "frac_tokens_any_nonzero": nonzero.flatten(2)
            .any(dim=-1)
            .float()
            .mean()
            .item(),
        }
    return stats


def gate_targets(gate: torch.Tensor) -> dict[str, torch.Tensor]:
    """Gate regression targets per position ``[N, L, H, K]`` (natural-log units).

    ``reference``: chunk-local cumulative gate at strip row 8 (what the kernel reads as
    ``grf``). ``future_sum``: cumulative gate at row 8 minus at the current row, i.e.
    the strictly-future gate mass the rebase leaks. ``current``: the row's own
    cumulative gate, a fully causal control target -- a residual that predicts it just
    as well carries information about the past, not the future.
    """
    num_positions = gate.shape[1]
    cumulative = gate.unflatten(1, (-1, CHUNK)).cumsum(dim=2).flatten(1, 2)
    position = torch.arange(num_positions, device=gate.device)
    reference_index = position // STRIP * STRIP + REFERENCE_ROW
    reference = cumulative[:, reference_index]
    return {
        "reference": reference,
        "future_sum": reference - cumulative,
        "current": cumulative,
    }


def regression_probes(
    layer: dict[str, torch.Tensor], residual_ulp: torch.Tensor, split: Split, seed: int
) -> dict:
    position = torch.arange(residual_ulp.shape[1], device=residual_ulp.device)
    leakable = position % STRIP < REFERENCE_ROW
    # Sample = (sequence, leakable position, head); features and targets are
    # per-head 128-vectors.
    features = {
        "residual": layer["residual"],
        "residual_ulp": residual_ulp,
        "out_causal": layer["out_causal"].float(),
        "out_causal+residual_ulp": torch.cat(
            [layer["out_causal"].float(), residual_ulp], -1
        ),
    }
    features = {
        name: gather(f, split, leakable, per_head=True) for name, f in features.items()
    }
    features["shuffled_residual_ulp"] = shuffled(features["residual_ulp"], seed)
    results = {}
    for target_name, target in gate_targets(layer["gate"].float()).items():
        y = gather(target, split, leakable, per_head=True)
        results[target_name] = {
            name: {"ridge_r2": ridge_r2(x, y), "mlp_r2": mlp_r2(x, y, seed)}
            for name, x in features.items()
        }
    return results


def classification_probes(
    layer: dict[str, torch.Tensor],
    residual_ulp: torch.Tensor,
    tokens: torch.Tensor,
    split: Split,
    num_classes: int,
    seed: int,
) -> dict:
    device = residual_ulp.device
    position = torch.arange(residual_ulp.shape[1], device=device)
    leakable = position % STRIP < REFERENCE_ROW
    next_token = tokens[:, 1:].to(device)  # label at position p is token p + 1
    labels = gather(next_token, split, leakable)
    vocab = int(tokens.max()) + 1
    counts = torch.bincount(labels["train"], minlength=vocab)
    remap = torch.full((vocab,), -1, device=device)
    remap[counts.topk(num_classes).indices] = torch.arange(num_classes, device=device)
    class_of = {k: remap[v] for k, v in labels.items()}
    keep = {k: v >= 0 for k, v in class_of.items()}
    y = {k: class_of[k][keep[k]] for k in labels}
    # Sequence id of every kept held-out sample, for a by-sequence standard error.
    holdout_sequence = torch.arange(len(split.holdout), device=device)
    holdout_sequence = holdout_sequence.repeat_interleave(int(leakable.sum()))[
        keep["holdout"]
    ]

    def stderr(values: torch.Tensor) -> float:
        sums = torch.zeros(len(split.holdout), device=device).index_add_(
            0, holdout_sequence, values
        )
        n = torch.bincount(holdout_sequence, minlength=len(split.holdout)).clamp_min(1)
        per_sequence = sums / n
        return per_sequence.std().item() / math.sqrt(len(split.holdout))

    features = {
        "out_causal": layer["out_causal"].float().flatten(2),
        "residual_ulp": residual_ulp.flatten(2),
        "out_causal+residual_ulp": torch.cat(
            [layer["out_causal"].float(), residual_ulp], -1
        ).flatten(2),
    }
    features = {
        name: {k: v[keep[k]] for k, v in gather(f, split, leakable).items()}
        for name, f in features.items()
    }
    features["shuffled_residual_ulp"] = shuffled(features["residual_ulp"], seed)
    features["out_causal+shuffled_residual_ulp"] = {
        k: torch.cat(
            [features["out_causal"][k], features["shuffled_residual_ulp"][k]], -1
        )
        for k in y
    }

    log_marginal = torch.log1p(
        torch.bincount(y["train"], minlength=num_classes).float()
    )
    log_marginal -= log_marginal.logsumexp(0)
    marginal_ce = -log_marginal[y["holdout"]]
    results = {
        "num_classes": num_classes,
        "coverage_holdout": keep["holdout"].float().mean().item(),
        "num_train": int(y["train"].shape[0]),
        "num_holdout": int(y["holdout"].shape[0]),
        "marginal": {
            "cross_entropy": marginal_ce.mean().item(),
            "accuracy": (y["holdout"] == log_marginal.argmax()).float().mean().item(),
        },
        "probes": {},
        "per_sample_ce": {},
    }
    for name, x in features.items():
        scale = Standardizer(x["train"])
        xs = {k: scale(v) for k, v in x.items()}
        for probe, hidden in (("logistic", None), ("mlp", 1024)):
            model = fit(
                xs,
                y,
                out_dim=num_classes,
                hidden=hidden,
                loss="ce",
                seed=seed,
                batch_size=1024,
            )
            with torch.no_grad():
                logits = model(xs["holdout"])
            ce = F.cross_entropy(logits, y["holdout"], reduction="none")
            results["probes"][f"{name}/{probe}"] = {
                "cross_entropy": ce.mean().item(),
                "cross_entropy_stderr": stderr(ce),
                "accuracy": (logits.argmax(-1) == y["holdout"]).float().mean().item(),
            }
            results["per_sample_ce"][f"{name}/{probe}"] = ce
    # Channel capacity estimate: paired per-sample CE differences. "gain" is the honest
    # probe minus the residual-augmented probe; "vs_shuffled" is the shuffled-augmented
    # probe minus the real-augmented probe (the overfitting cost of 2048 extra noise
    # features cancels, leaving only what pairing the residual with its own sample buys).
    ce = results.pop("per_sample_ce")
    for probe in ("logistic", "mlp"):
        paired = {
            "ce_gain_from_residual_ulp": ce[f"out_causal/{probe}"]
            - ce[f"out_causal+residual_ulp/{probe}"],
            "ce_gain_from_shuffled_residual_ulp": ce[f"out_causal/{probe}"]
            - ce[f"out_causal+shuffled_residual_ulp/{probe}"],
            "ce_gain_residual_vs_shuffled": ce[
                f"out_causal+shuffled_residual_ulp/{probe}"
            ]
            - ce[f"out_causal+residual_ulp/{probe}"],
            "ce_gain_residual_alone_vs_shuffled": ce[f"shuffled_residual_ulp/{probe}"]
            - ce[f"residual_ulp/{probe}"],
        }
        for name, gain in paired.items():
            results[f"{name}/{probe}"] = {
                "nats": gain.mean().item(),
                "stderr_by_sequence": stderr(gain),
                "bits": gain.mean().item() / math.log(2),
            }
    return results


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    capture = torch.load(args.capture, mmap=True, weights_only=True)
    tokens = capture["tokens"]
    split = Split.by_sequence(
        tokens.shape[0], args.holdout_fraction, args.validation_fraction
    )
    report = {
        "capture": str(args.capture),
        "checkpoint_step": capture["checkpoint_step"],
        "num_sequences": int(tokens.shape[0]),
        "seq_len": int(tokens.shape[1] - 1),
        "parallel_nll": capture["parallel_nll"].mean().item(),
        "split": {
            k: [int(getattr(split, k)[0]), int(getattr(split, k)[-1])]
            for k in ("train", "validation", "holdout")
        },
        "layers": [],
    }
    for index, stored in enumerate(capture["layers"]):
        layer = {
            name: stored[name].to(device)
            for name in ("gate", "out_causal", "out_midpoint")
        }
        out_causal = layer["out_causal"].float()
        layer["residual"] = layer["out_midpoint"].float() - out_causal
        ulp = bf16_ulp(out_causal, layer["out_midpoint"].float())
        residual_ulp = layer["residual"] / ulp
        print(f"layer {index}: residual stats", flush=True)
        stats = residual_stats(layer["residual"], out_causal, ulp)
        print(f"layer {index}: regression probes", flush=True)
        regression = regression_probes(layer, residual_ulp, split, args.seed)
        print(f"layer {index}: classification probes", flush=True)
        classification = classification_probes(
            layer, residual_ulp, tokens, split, args.num_classes, args.seed
        )
        report["layers"].append(
            {
                "kda_layer": index,
                "residual": stats,
                "regression": regression,
                "classification": classification,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report["layers"][-1], indent=1), flush=True)


if __name__ == "__main__":
    main()
