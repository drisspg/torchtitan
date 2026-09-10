# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Paired comparison of two arms evaluated on the same held-out sequences.

Sequence-to-sequence NLL variance (~0.3 nats) dwarfs the arm difference (~0.005), so
an unpaired comparison of means says nothing. Because both arms see identical
sequences, the per-sequence difference cancels the shared variance; this prints
mean, standard error and a t statistic for each checkpoint::

    python -m torchtitan.experiments.kda_pivot.paired_compare \\
        --a ../runs/scaled_eval/midpoint --b ../runs/scaled_eval/causal
"""

import argparse
import json
import math
from pathlib import Path


def load(eval_dir: Path) -> dict[int, dict]:
    return {
        r["checkpoint_step"]: r
        for r in (json.loads(p.read_text()) for p in eval_dir.glob("eval_*.json"))
    }


def paired(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Mean, standard error and t statistic of ``a - b`` over paired samples."""
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    stderr = math.sqrt(var / n)
    return mean, stderr, mean / stderr if stderr else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a", type=Path, required=True, help="eval dir of arm A (e.g. midpoint)"
    )
    parser.add_argument(
        "--b", type=Path, required=True, help="eval dir of arm B (e.g. causal)"
    )
    args = parser.parse_args()
    a_runs, b_runs = load(args.a), load(args.b)

    print(
        f"A = {args.a.name}, B = {args.b.name}; values are A - B, nats, n = paired sequences"
    )
    print(
        "step | parallel NLL: mean +- se (t) | autoregressive NLL: mean +- se (t) | gap DiD: mean +- se (t) | leakable-rows gap DiD: mean +- se (t)"
    )
    for step in sorted(a_runs):
        a, b = a_runs[step]["per_sequence"], b_runs[step]["per_sequence"]
        cols = []
        for key in ("parallel_nll", "autoregressive_nll"):
            m, se, t = paired(a[key], b[key])
            cols.append(f"{m:+.4f} +- {se:.4f} ({t:+.1f})")
        gap_a = [
            x - y
            for x, y in zip(a["autoregressive_nll"], a["parallel_nll"], strict=True)
        ]
        gap_b = [
            x - y
            for x, y in zip(b["autoregressive_nll"], b["parallel_nll"], strict=True)
        ]
        m, se, t = paired(gap_a, gap_b)
        cols.append(f"{m:+.5f} +- {se:.5f} ({t:+.1f})")
        m, se, t = paired(a["leakable_rows_gap"], b["leakable_rows_gap"])
        cols.append(f"{m:+.5f} +- {se:.5f} ({t:+.1f})")
        print(f"{step} | " + " | ".join(cols))


if __name__ == "__main__":
    main()
