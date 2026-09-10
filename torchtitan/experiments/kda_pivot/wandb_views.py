# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Create a minimal saved W&B workspace view for one experiment group.

The default project workspace auto-generates a panel per logged key (dozens of
torchtitan metrics). This builds a view with only the panels the A/B needs, its
runset filtered to one ``--experiment`` group::

    python -m torchtitan.experiments.kda_pivot.wandb_views --experiment pilot-520m-1b

Prints the view URL. Re-running with the same name updates the view in place.
"""

import argparse

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ENTITY = "drisspg"
PROJECT = "kda-pivot"


def line(title: str, x: str, ys: list[str], **kwargs) -> wr.LinePlot:
    return wr.LinePlot(title=title, x=x, y=ys, layout=wr.Layout(w=8, h=6), **kwargs)


def build(experiment: str) -> ws.Workspace:
    train_x = "Step"
    eval_x = "Step"  # eval runs log with step = checkpoint step
    return ws.Workspace(
        entity=ENTITY,
        project=PROJECT,
        name=f"kda-pivot {experiment}",
        settings=ws.WorkspaceSettings(smoothing_type="none"),
        runset_settings=ws.RunsetSettings(
            filters=[ws.Metric("Group") == experiment],
            order=[ws.Ordering(ws.Metric("Name"))],
        ),
        sections=[
            ws.Section(
                name="Leak test: autoregressive - parallel NLL (nats, 0 = causal)",
                is_open=True,
                panels=[
                    line("gap, all positions", eval_x, ["eval/all/gap_nats"]),
                    line(
                        "gap, leakable rows (strip 0-7)",
                        eval_x,
                        ["eval/leakable_rows/gap_nats"],
                    ),
                    line(
                        "gap, causal rows (strip 8-15)",
                        eval_x,
                        ["eval/causal_rows/gap_nats"],
                    ),
                    line(
                        "leakable - causal rows",
                        eval_x,
                        ["eval/leakable_minus_causal_rows_gap"],
                    ),
                ],
            ),
            ws.Section(
                name="Held-out NLL by mode",
                is_open=True,
                panels=[
                    line(
                        "autoregressive vs parallel, all positions",
                        eval_x,
                        ["eval/all/autoregressive_nll", "eval/all/parallel_nll"],
                    ),
                    line(
                        "autoregressive vs parallel, leakable rows",
                        eval_x,
                        [
                            "eval/leakable_rows/autoregressive_nll",
                            "eval/leakable_rows/parallel_nll",
                        ],
                    ),
                ],
            ),
            ws.Section(
                name="Training (parallel mode)",
                is_open=True,
                panels=[
                    line("validation loss", train_x, ["validation_metrics/loss"]),
                    line("train loss", train_x, ["loss_metrics/global_avg_loss"]),
                    line("grad norm", train_x, ["grad_norm"]),
                    line("lr", train_x, ["lr/AdamW"]),
                ],
            ),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="W&B group to show")
    args = parser.parse_args()
    workspace = build(args.experiment)
    workspace.save()
    print(workspace.url)


if __name__ == "__main__":
    main()
