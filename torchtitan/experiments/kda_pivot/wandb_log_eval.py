# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attach eval_prefix results to the W&B run of the training arm they evaluate.

Eval runs on MAST without network access, so this runs on the devgpu after
``mastjob fetch``. Metrics use ``eval/checkpoint_step`` as their x-axis so they
can be appended to a finished run whose history already covers training steps::

    python -m torchtitan.experiments.kda_pivot.wandb_log_eval \\
        --run-id gg0muppq --eval-dir ~/.mast_play/results/<eval job>/causal-seed42
"""

import argparse
import json
from pathlib import Path

import wandb

SUMMARY_KEYS = ("all", "strip_offset_0_7", "strip_offset_8_15")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id", required=True, help="W&B run id of the training arm"
    )
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--project", default="kda-pivot")
    args = parser.parse_args()

    results = sorted(
        (
            json.loads(path.read_text())
            for path in args.eval_dir.glob("eval_*.json")
        ),
        key=lambda result: result["checkpoint_step"],
    )
    if not results:
        raise FileNotFoundError(f"no eval_*.json under {args.eval_dir}")

    run = wandb.init(project=args.project, id=args.run_id, resume="must")
    run.define_metric("eval/checkpoint_step")
    run.define_metric("eval/*", step_metric="eval/checkpoint_step")
    for result in results:
        metrics = {"eval/checkpoint_step": result["checkpoint_step"]}
        for key in SUMMARY_KEYS:
            summary = result[key]
            metrics[f"eval/{key}/gap_nats"] = summary["gap_nats"]
            metrics[f"eval/{key}/gap_stderr"] = summary["gap_stderr_by_sequence"]
            metrics[f"eval/{key}/prefix_nll"] = summary["prefix_nll"]
            metrics[f"eval/{key}/parallel_nll"] = summary["parallel_nll"]
        run.log(metrics)
        print(
            f"step {result['checkpoint_step']}: gap {result['all']['gap_nats']:+.5f} "
            f"(0-7 {result['strip_offset_0_7']['gap_nats']:+.5f}, "
            f"8-15 {result['strip_offset_8_15']['gap_nats']:+.5f})"
        )
    run.finish()


if __name__ == "__main__":
    main()
