# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallel vs prefix-only next-token NLL for a trained checkpoint.

For each held-out sequence of ``L + 1`` tokens this computes, per predicted position
``p`` (predicting token ``p + 1`` from tokens ``0..p``):

* ``parallel``: one teacher-forced forward over all ``L`` input tokens, read logit ``p``.
* ``prefix``: a forward over only the first ``p + 1`` input tokens, read the last logit.

The prefix pass is the numerical oracle for autoregressive use: nothing after ``p``
exists, so any kernel dependence on future tokens is impossible. A model that learned
to exploit a future-dependent rounding channel shows ``prefix`` NLL above ``parallel``
NLL; the causal-reference arm bounds how much of that gap is ordinary shape-dependent
kernel rounding. Results are bucketed by position within the 16-token KDA strip so the
rows a midpoint reference can leak into (offsets 0-7) are visible separately.

Run exactly like training, adding eval flags before the torchtitan config args::

    torchrun --standalone --nproc_per_node=1 -m torchtitan.experiments.kda_pivot.eval_prefix \\
        --eval-output out.json --eval-seq-len 256 --eval-num-sequences 64 \\
        --module torchtitan.experiments.kda_pivot --config kda_pivot_pilot \\
        --dump_folder <run> --checkpoint.load_step 2000
"""

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import cast

import torch

from torchtitan.config import ConfigManager
from torchtitan.experiments.kda_pivot.config_registry import _data_dir, VALIDATION_SHARD
from torchtitan.protocols.model import BaseModel
from torchtitan.tools.logging import init_logger, logger
from torchtitan.trainer import Trainer

STRIP = 16


def parse_eval_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--eval-seq-len", type=int, default=256)
    parser.add_argument("--eval-num-sequences", type=int, default=64)
    parser.add_argument(
        "--eval-skip-documents",
        type=int,
        default=0,
        help="Validation documents to skip first (disjoint from the training validator).",
    )
    return parser.parse_known_args(argv)


def held_out_sequences(
    tokenizer, *, seq_len: int, num_sequences: int, skip_documents: int
) -> list[list[int]]:
    """First ``num_sequences`` validation documents with at least ``seq_len + 1`` tokens."""
    sequences: list[list[int]] = []
    with gzip.open(_data_dir() / VALIDATION_SHARD, "rt") as lines:
        for index, line in enumerate(lines):
            if index < skip_documents:
                continue
            tokens = tokenizer.encode(json.loads(line)["text"], add_bos=True, add_eos=True)
            if len(tokens) >= seq_len + 1:
                sequences.append(tokens[: seq_len + 1])
            if len(sequences) == num_sequences:
                return sequences
    raise ValueError(f"validation shard has fewer than {num_sequences} long-enough documents")


def next_token_nll(trainer: Trainer, tokens: torch.Tensor) -> torch.Tensor:
    """Per-position NLL of ``tokens[1:]`` given ``tokens[:-1]`` in one teacher-forced pass."""
    model = trainer.model_parts[0]
    device = tokens.device
    inputs = tokens[:-1]
    labels = tokens[1:]
    inputs, labels, extra_kwargs = cast(BaseModel, model).preprocess_inputs(
        {
            "input": inputs,
            "labels": labels,
            "positions": torch.arange(inputs.shape[0], device=device),
        },
        parallel_dims=trainer.parallel_dims,
        parallelism=trainer.config.parallelism,
        max_num_documents=None,
        max_context_length=trainer.config.training.max_context_length,
    )
    with trainer.train_context():
        hidden = model(inputs, **extra_kwargs)
        # ChunkedLossWrapper detaches lm_head from the model forward, so apply it here.
        logits = model.lm_head(hidden) if model._skip_lm_head else hidden
    return torch.nn.functional.cross_entropy(logits.float(), labels, reduction="none")


@torch.no_grad()
def evaluate(trainer: Trainer, sequences: list[list[int]]) -> dict:
    device = torch.device(trainer.device)
    seq_len = len(sequences[0]) - 1
    parallel = torch.zeros(len(sequences), seq_len, dtype=torch.float64)
    prefix = torch.zeros_like(parallel)
    for s, sequence in enumerate(sequences):
        tokens = torch.tensor(sequence, device=device)
        parallel[s] = next_token_nll(trainer, tokens).double().cpu()
        for p in range(seq_len):
            prefix[s, p] = next_token_nll(trainer, tokens[: p + 2])[-1].item()
        if s % 8 == 0:
            logger.info(f"sequence {s}/{len(sequences)}")

    gap = prefix - parallel
    position = torch.arange(seq_len)
    strip_offset = position % STRIP

    def summary(mask: torch.Tensor) -> dict[str, float]:
        selected = gap[:, mask]
        per_sequence = selected.mean(dim=1)
        stderr = per_sequence.std().item() / math.sqrt(len(sequences))
        return {
            "parallel_nll": parallel[:, mask].mean().item(),
            "prefix_nll": prefix[:, mask].mean().item(),
            "gap_nats": selected.mean().item(),
            "gap_stderr_by_sequence": stderr,
            "max_abs_gap": selected.abs().max().item(),
            "num_positions": int(mask.sum()),
        }

    return {
        "seq_len": seq_len,
        "num_sequences": len(sequences),
        "all": summary(torch.ones_like(position, dtype=torch.bool)),
        "strip_offset_0_7": summary(strip_offset < STRIP // 2),
        "strip_offset_8_15": summary(strip_offset >= STRIP // 2),
        "by_strip_offset": [summary(strip_offset == o) for o in range(STRIP)],
        "per_position_gap": gap.mean(dim=0).tolist(),
    }


def main() -> None:
    init_logger()
    eval_args, titan_args = parse_eval_args(sys.argv[1:])
    # The lr-scheduler state in the checkpoint is validated against training.steps,
    # so pass the same torchtitan args the training run used.
    config = ConfigManager().parse_args(titan_args)
    if not config.checkpoint.enable:
        raise ValueError("eval_prefix requires checkpoint.enable so weights can be loaded")

    trainer = Trainer(config)
    try:
        if not trainer.checkpointer.load(step=config.checkpoint.load_step):
            raise RuntimeError(f"no checkpoint found in {config.dump_folder}")
        logger.info(f"Evaluating checkpoint step {trainer.step}")
        for model in trainer.model_parts:
            model.eval()
        sequences = held_out_sequences(
            trainer.tokenizer,
            seq_len=eval_args.eval_seq_len,
            num_sequences=eval_args.eval_num_sequences,
            skip_documents=eval_args.eval_skip_documents,
        )
        results = evaluate(trainer, sequences)
        results["checkpoint_step"] = trainer.step
        results["dump_folder"] = config.dump_folder
        eval_args.eval_output.parent.mkdir(parents=True, exist_ok=True)
        eval_args.eval_output.write_text(json.dumps(results, indent=2))
        logger.info(
            "all positions: gap %.5f +- %.5f nats; strip offsets 0-7: %.5f; 8-15: %.5f",
            results["all"]["gap_nats"],
            results["all"]["gap_stderr_by_sequence"],
            results["strip_offset_0_7"]["gap_nats"],
            results["strip_offset_8_15"]["gap_nats"],
        )
    finally:
        trainer.close()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
