# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Log the paired midpoint-minus-causal comparison to W&B as a `comparison` run.

Panels in the experiment view read ``paired/*`` from this run (``job_type=comparison``,
name ``midpoint-minus-causal-seed<seed>``), with ``step`` = checkpoint step::

    python -m torchtitan.experiments.kda_pivot.wandb_log_paired \\
        --midpoint ../runs/scaled_eval_1024/midpoint --causal ../runs/scaled_eval_1024/causal \\
        --experiment scaled-1.45b-4b --seed 42
"""

import argparse
from pathlib import Path

import wandb

from torchtitan.experiments.kda_pivot.paired_compare import load, paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--midpoint", type=Path, required=True)
    parser.add_argument("--causal", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--project", default="kda-pivot")
    args = parser.parse_args()
    mid, cau = load(args.midpoint), load(args.causal)
    steps = sorted(set(mid) & set(cau))
    num_sequences = mid[steps[0]]["num_sequences"]

    run = wandb.init(
        project=args.project,
        group=args.experiment,
        job_type="comparison",
        name=f"midpoint-minus-causal-seed{args.seed}-n{num_sequences}",
        tags=["comparison", args.experiment],
        config={
            "seed": args.seed,
            "num_sequences": num_sequences,
            "experiment": args.experiment,
        },
    )
    for step in steps:
        a, b = mid[step]["per_sequence"], cau[step]["per_sequence"]
        metrics = {}
        for key in ("parallel_nll", "autoregressive_nll"):
            mean, se, t = paired(a[key], b[key])
            metrics |= {
                f"paired/{key}_diff": mean,
                f"paired/{key}_diff_se": se,
                f"paired/{key}_diff_t": t,
            }
        gap_a = [
            x - y
            for x, y in zip(a["autoregressive_nll"], a["parallel_nll"], strict=True)
        ]
        gap_b = [
            x - y
            for x, y in zip(b["autoregressive_nll"], b["parallel_nll"], strict=True)
        ]
        for key, da, db in (
            ("gap_did", gap_a, gap_b),
            ("leakable_rows_gap_did", a["leakable_rows_gap"], b["leakable_rows_gap"]),
        ):
            mean, se, t = paired(da, db)
            metrics |= {
                f"paired/{key}": mean,
                f"paired/{key}_se": se,
                f"paired/{key}_t": t,
            }
        run.log(metrics, step=step)
    run.finish()
    print(run.url)


if __name__ == "__main__":
    main()
