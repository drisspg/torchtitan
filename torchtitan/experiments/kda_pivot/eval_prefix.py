# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallel vs prefix-only next-token NLL for a trained checkpoint.

For each held-out sequence of ``L + 1`` tokens this computes, per predicted position
``p`` (predicting token ``p + 1`` from tokens ``0..p``):

* ``parallel``: one teacher-forced forward over all ``L`` input tokens, read logit ``p``.
* ``recurrent`` (default): one forward per sequence with the KDA layers switched to
  Attention Gym's token-recurrent kernel (the decode path; each token sees only its past
  by construction) while MLA stays causal teacher-forced.
* ``prefix`` (``--eval-mode prefix``): a forward over only the first ``p + 1`` input
  tokens, read the last logit. The literal oracle -- nothing after ``p`` exists -- but it
  costs ``L`` forwards per sequence (about 200x slower).

Both modes agreed to within 2e-4 nats overall and per strip bucket on trained step-4000
checkpoints of both arms (2026-09-09), so ``recurrent`` is the default. A model that learned
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
from attn_gym.linear.kda import recurrent_kda

from torchtitan.config import ConfigManager
from torchtitan.experiments.kda_pivot.config_registry import _data_dir, VALIDATION_SHARD
from torchtitan.models.kimi_k3 import kda as kda_module
from torchtitan.protocols.model import BaseModel
from torchtitan.tools.logging import init_logger, logger
from torchtitan.trainer import Trainer

STRIP = 16


def parse_eval_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--eval-output",
        type=Path,
        required=True,
        help="JSON path; with several --eval-steps the step is appended to the stem",
    )
    parser.add_argument("--eval-seq-len", type=int, default=256)
    parser.add_argument("--eval-num-sequences", type=int, default=64)
    parser.add_argument(
        "--eval-skip-documents",
        type=int,
        # The training-time Validator packs the first documents of the validation
        # shard (50 steps x 32K tokens per pass, ~7000 docs); start well past them so
        # the final metric is on text no decision ever touched.
        default=20000,
        help="Validation documents to skip first (disjoint from the training validator).",
    )
    parser.add_argument(
        "--eval-mode",
        choices=("recurrent", "prefix"),
        default="recurrent",
        help="autoregressive oracle: one recurrent-KDA forward, or per-prefix forwards",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        nargs="+",
        default=None,
        help="Checkpoint steps to evaluate in turn (default: --checkpoint.load_step)",
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
            tokens = tokenizer.encode(
                json.loads(line)["text"], add_bos=True, add_eos=True
            )
            if len(tokens) >= seq_len + 1:
                sequences.append(tokens[: seq_len + 1])
            if len(sequences) == num_sequences:
                return sequences
    raise ValueError(
        f"validation shard has fewer than {num_sequences} long-enough documents"
    )


def next_token_nll(trainer: Trainer, tokens: torch.Tensor) -> torch.Tensor:
    """Per-position NLL of ``tokens[1:]`` given ``tokens[:-1]`` in one teacher-forced pass."""
    model = cast(BaseModel, trainer.model_parts[0])
    device = tokens.device
    inputs = tokens[:-1]
    labels = tokens[1:]
    inputs, labels, extra_kwargs = model.preprocess_inputs(
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
        # pyrefly: ignore [missing-attribute, not-callable]
        logits = model.lm_head(hidden) if model._skip_lm_head else hidden
    return torch.nn.functional.cross_entropy(logits.float(), labels, reduction="none")


class recurrent_kda_layers:
    """Route the KDA layer through ``recurrent_kda`` for the duration of the block.

    Eval-only: ``KDAKernel.forward`` looks up ``chunk_kda`` in its module globals, so
    rebinding that name is enough and nothing in the model or config changes.
    """

    def __enter__(self) -> None:
        self.saved = kda_module.chunk_kda

        def recurrent(*args, autotune: bool, **kwargs):
            return recurrent_kda(*args, autotune=autotune, **kwargs)

        kda_module.chunk_kda = recurrent

    def __exit__(self, *exc: object) -> None:
        kda_module.chunk_kda = self.saved


@torch.no_grad()
def evaluate(trainer: Trainer, sequences: list[list[int]], *, mode: str) -> dict:
    device = torch.device(trainer.device)
    seq_len = len(sequences[0]) - 1
    parallel = torch.zeros(len(sequences), seq_len, dtype=torch.float64)
    prefix = torch.zeros_like(parallel)
    for s, sequence in enumerate(sequences):
        tokens = torch.tensor(sequence, device=device)
        parallel[s] = next_token_nll(trainer, tokens).double().cpu()
        if mode == "recurrent":
            with recurrent_kda_layers():
                prefix[s] = next_token_nll(trainer, tokens).double().cpu()
        else:
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
    config = cast(Trainer.Config, ConfigManager().parse_args(titan_args))
    if not config.checkpoint.enable:
        raise ValueError(
            "eval_prefix requires checkpoint.enable so weights can be loaded"
        )
    # Only weights and the step counter matter here. The dataloader state refuses a
    # data-parallel degree different from training, and the lr scheduler asserts on
    # training.steps, so neither is loaded.
    config.checkpoint.exclude_from_loading = ["dataloader", "optimizer", "lr_scheduler"]

    steps = eval_args.eval_steps or [config.checkpoint.load_step]
    trainer = Trainer(config)
    try:
        sequences = held_out_sequences(
            trainer.tokenizer,
            seq_len=eval_args.eval_seq_len,
            num_sequences=eval_args.eval_num_sequences,
            skip_documents=eval_args.eval_skip_documents,
        )
        for step in steps:
            if not trainer.checkpointer.load(step=step):
                raise RuntimeError(
                    f"no checkpoint for step {step} in {config.dump_folder}"
                )
            logger.info(f"Evaluating checkpoint step {trainer.step}")
            for model in trainer.model_parts:
                model.eval()
            results = evaluate(trainer, sequences, mode=eval_args.eval_mode)
            results["checkpoint_step"] = trainer.step
            results["eval_mode"] = eval_args.eval_mode
            results["dump_folder"] = config.dump_folder
            output = eval_args.eval_output
            if len(steps) > 1:
                output = output.with_name(
                    f"{output.stem}_step{trainer.step}{output.suffix}"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(results, indent=2))
            logger.info(
                "step %d: gap %.5f +- %.5f nats; strip offsets 0-7: %.5f; 8-15: %.5f",
                trainer.step,
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
