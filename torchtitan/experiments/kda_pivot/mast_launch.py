# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MAST entry point for one arm of the KDA gate-reference A/B.

Runs under ``mastjob launch ... -- repos/torchtitan/torchtitan/experiments/kda_pivot/mast_launch.py``
via ``run_in_out.py`` (torchrun already configured, cwd = ``$MAST_PLAY_OUT``). The
node has no internet and environment variables do not propagate from the launcher,
so everything the run needs is baked in here: the gate-reference switch, offline
Hugging Face/W&B modes, and the staged dataset/tokenizer copied from the Manifold
mount to local disk.

Both arms must be launched with identical ``--seed``, ``--steps`` and node count so
the data order and global batch match; only ``--variant`` differs.

    mast_launch.py --variant causal   --seed 42 --steps 4000
    mast_launch.py --variant midpoint --seed 42 --steps 4000
    mast_launch.py --mode eval --variant midpoint --seed 42 --steps 4000 \\
        --train-job <midpoint job name> --load-steps 500 1000 1500 2000 2500 3000 3500 4000
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

RESULTS_MOUNT = Path("/mnt/pytorch_distributed")
STAGED_DATA = RESULTS_MOUNT / "kda_pivot_data"
LOCAL_DATA = Path("/tmp/kda_pivot_data")
SHARD_NAMES = (
    "c4-train.00000-of-01024.json.gz",
    "c4-train.00001-of-01024.json.gz",
    "c4-train.00002-of-01024.json.gz",
    "c4-train.00003-of-01024.json.gz",
    "c4-train.00004-of-01024.json.gz",
    "c4-train.00005-of-01024.json.gz",
    "c4-validation.00000-of-00008.json.gz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("causal", "midpoint"), required=True)
    parser.add_argument(
        "--experiment",
        required=True,
        help="W&B group and tag shared by both arms of one comparison, e.g. pilot-520m-1b",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--config", default="kda_pivot_pilot")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument(
        "--train-job",
        help="eval: MAST job name (or absolute dump root) whose checkpoint to load",
    )
    parser.add_argument(
        "--load-steps",
        type=int,
        nargs="+",
        default=[-1],
        help="eval: checkpoint steps to evaluate in turn (-1 = latest)",
    )
    parser.add_argument("--eval-seq-len", type=int, default=256)
    parser.add_argument("--eval-num-sequences", type=int, default=64)
    parser.add_argument(
        "--eval-mode", choices=("recurrent", "prefix"), default="recurrent"
    )
    parser.add_argument(
        "--leak-control",
        type=float,
        default=0.0,
        help="positive control: inject this multiple of the next token's value into the"
        " parallel KDA output (see config_registry.LEAK_CONTROL_ENV); never for real arms",
    )
    parser.add_argument(
        "--local-data",
        type=Path,
        default=None,
        help="Use an existing local data dir (with en/ shards and hf/ tokenizer) instead of the mount",
    )
    return parser.parse_args()


def stage_data(local_data: Path | None) -> Path:
    """Copy shards and tokenizer to local disk once per node; return the data root."""
    if local_data is not None:
        return local_data
    if os.environ.get("LOCAL_RANK", "0") == "0":
        (LOCAL_DATA / "c4" / "en").mkdir(parents=True, exist_ok=True)
        for shard in SHARD_NAMES:
            target = LOCAL_DATA / "c4" / "en" / shard
            if not target.exists():
                shutil.copyfile(STAGED_DATA / "c4" / "en" / shard, target)
        if not (LOCAL_DATA / "hf" / "Qwen3-0.6B").exists():
            shutil.copytree(
                STAGED_DATA / "hf" / "Qwen3-0.6B", LOCAL_DATA / "hf" / "Qwen3-0.6B"
            )
        (LOCAL_DATA / "READY").touch()
    else:
        while not (LOCAL_DATA / "READY").exists():
            time.sleep(5)
    return LOCAL_DATA


def main() -> None:
    args = parse_args()
    out = Path(os.environ["MAST_PLAY_OUT"])
    # The arm label names dump folders and W&B runs; a leak control is its own arm.
    arm = f"leak{args.leak_control:g}" if args.leak_control else args.variant
    run_name = f"{arm}-seed{args.seed}"
    dump_folder = out / run_name

    data_root = stage_data(args.local_data)
    os.environ["ATTN_GYM_KDA_GATE_REFERENCE"] = args.variant
    if args.leak_control:
        os.environ["KDA_PIVOT_LEAK_CONTROL"] = str(args.leak_control)
    os.environ["KDA_PIVOT_DATA_DIR"] = str(data_root / "c4" / "en")
    os.environ["KDA_PIVOT_HF_ASSETS"] = str(data_root / "hf" / "Qwen3-0.6B")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", "/tmp/kda_pivot_hf_home")
    # Offline W&B: run files land in the dump folder, get published with the job,
    # and are synced from the devgpu with `wandb sync` after `mastjob fetch`.
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_PROJECT"] = "kda-pivot"
    os.environ["WANDB_RUN_GROUP"] = args.experiment
    os.environ["WANDB_RUN_NAME"] = run_name
    os.environ["WANDB_RUN_JOB_TYPE"] = "train"
    os.environ["WANDB_RUN_TAGS"] = f"{arm},{args.experiment},{args.config}"
    os.environ["WANDB_RUN_NOTES"] = (
        f"gate reference={args.variant}; seed={args.seed}; steps={args.steps}; "
        f"config={args.config}"
        + (f"; LEAK CONTROL eps={args.leak_control}" if args.leak_control else "")
    )

    titan_args = [
        "--module",
        "torchtitan.experiments.kda_pivot",
        "--config",
        args.config,
        "--debug.seed",
        str(args.seed),
        "--debug.deterministic",
        "--training.steps",
        str(args.steps),
        "--metrics.enable_tensorboard",
        "--metrics.enable_wandb",
    ]
    if args.mode == "train":
        sys.argv = ["torchtitan.train", *titan_args, "--dump_folder", str(dump_folder)]
        from torchtitan.train import main as train_main

        train_main()
        return

    if args.train_job is None:
        raise ValueError("--mode eval requires --train-job")
    train_dump = Path(args.train_job) / run_name
    if not train_dump.is_absolute():
        train_dump = RESULTS_MOUNT / train_dump
    # Load weights from the training job's published checkpoint; keep logs local.
    sys.argv = [
        "torchtitan.experiments.kda_pivot.eval_prefix",
        "--eval-output",
        str(dump_folder / f"eval_{args.eval_mode}.json"),
        "--eval-mode",
        args.eval_mode,
        "--eval-steps",
        *(str(step) for step in args.load_steps),
        "--eval-seq-len",
        str(args.eval_seq_len),
        "--eval-num-sequences",
        str(args.eval_num_sequences),
        *titan_args,
        "--dump_folder",
        str(dump_folder),
        "--checkpoint.folder",
        str(train_dump / "checkpoint"),
    ]
    from torchtitan.experiments.kda_pivot.eval_prefix import main as eval_main

    eval_main()


if __name__ == "__main__":
    main()
