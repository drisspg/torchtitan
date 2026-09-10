# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Log eval_prefix results to W&B as an eval run alongside the training run.

Eval runs on MAST without network access, so this runs on the devgpu after
``mastjob fetch``. Each arm gets its own eval run (``<arm>-seed<seed>-eval``) in the
same group as its training run, logged with ``step = checkpoint step`` so the
default x-axis is right and both arms overlay when the workspace groups by
``variant``::

    python -m torchtitan.experiments.kda_pivot.wandb_log_eval \\
        --eval-dir ~/.mast_play/results/<eval job>/causal-seed42 \\
        --experiment pilot-520m-1b --arm causal --seed 42
"""

import argparse
import json
from pathlib import Path

import wandb

BUCKETS = ("all", "leakable_rows", "causal_rows")
FIELDS = ("gap_nats", "gap_stderr_by_sequence", "autoregressive_nll", "parallel_nll")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment", required=True, help="W&B group shared with the training run"
    )
    parser.add_argument(
        "--arm", required=True, help="causal, midpoint, or leak<eps> (positive control)"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--project", default="kda-pivot")
    args = parser.parse_args()

    results = sorted(
        (json.loads(path.read_text()) for path in args.eval_dir.glob("eval_*.json")),
        key=lambda result: result["checkpoint_step"],
    )
    if not results:
        raise FileNotFoundError(f"no eval_*.json under {args.eval_dir}")

    run = wandb.init(
        project=args.project,
        group=args.experiment,
        job_type="eval",
        name=f"{args.arm}-seed{args.seed}-eval",
        tags=[args.arm, args.experiment, "eval"],
        config={
            "arm": args.arm,
            "seed": args.seed,
            "experiment": args.experiment,
            "eval_mode": results[0]["eval_mode"],
            "seq_len": results[0]["seq_len"],
            "num_sequences": results[0]["num_sequences"],
        },
    )
    for result in results:
        metrics = {
            f"eval/{bucket}/{field}": result[bucket][field]
            for bucket in BUCKETS
            for field in FIELDS
        }
        # Difference of the two halves: the leak-specific signal net of shared rounding.
        metrics["eval/leakable_minus_causal_rows_gap"] = (
            result["leakable_rows"]["gap_nats"] - result["causal_rows"]["gap_nats"]
        )
        run.log(metrics, step=result["checkpoint_step"])
        print(
            f"step {result['checkpoint_step']}: gap {result['all']['gap_nats']:+.5f} "
            f"(leakable {result['leakable_rows']['gap_nats']:+.5f}, "
            f"causal {result['causal_rows']['gap_nats']:+.5f})"
        )
    run.finish()


if __name__ == "__main__":
    main()
