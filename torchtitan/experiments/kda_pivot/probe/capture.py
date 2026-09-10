# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Capture KDA kernel inputs from a trained checkpoint and evaluate both gate references.

For each held-out sequence (B=1, ``L`` input tokens) the first ``--capture-num-layers``
KDA layers' exact ``chunk_kda`` arguments (L2-normalized q/k, v, per-token gate, beta)
are recorded during one teacher-forced forward. Afterwards ``chunk_kda`` is re-run on
every recorded input under ``ATTN_GYM_KDA_GATE_REFERENCE=causal`` and ``=midpoint``
(``resolve_gate_reference.cache_clear()`` between them; the CuTe cache keys include the
variant) so ``out_midpoint - out_causal`` is the exact rounding residual the midpoint
kernel would have injected into this model's activations. The causal re-run must match
the recorded outputs bitwise.

Run like ``eval_prefix`` with the process env set to ``causal``::

    ATTN_GYM_KDA_GATE_REFERENCE=causal torchrun --standalone --nproc_per_node=1 \\
        -m torchtitan.experiments.kda_pivot.probe.capture \\
        --capture-output ../runs/probe/capture.pt --capture-num-sequences 256 \\
        --module torchtitan.experiments.kda_pivot --config kda_pivot_scaled \\
        --debug.seed 42 --debug.deterministic --training.steps 7600 \\
        --dump_folder ../runs/probe/dump --checkpoint.folder <ckpt dir> --checkpoint.load_step 7600
"""

import argparse
import os
import sys
from pathlib import Path
from typing import cast

import torch
from attn_gym.linear.kda import chunk_kda
from attn_gym.linear.kda.gate_reference import (
    GATE_REFERENCE_ENV,
    GateReference,
    resolve_gate_reference,
)

from torchtitan.config import ConfigManager
from torchtitan.experiments.kda_pivot.eval_prefix import (
    held_out_sequences,
    next_token_nll,
)
from torchtitan.models.kimi_k3 import kda as kda_module
from torchtitan.tools.logging import init_logger, logger
from torchtitan.trainer import Trainer

KERNEL_INPUTS = ("q", "k", "v", "gate", "beta")


def parse_capture_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--capture-output", type=Path, required=True)
    parser.add_argument("--capture-seq-len", type=int, default=256)
    parser.add_argument("--capture-num-sequences", type=int, default=256)
    parser.add_argument("--capture-skip-documents", type=int, default=20000)
    parser.add_argument(
        "--capture-num-layers",
        type=int,
        default=2,
        help="record the first N KDA layers, in forward order",
    )
    return parser.parse_known_args(argv)


class RecordKDAInputs:
    """Record the first ``num_layers`` ``chunk_kda`` calls of each forward, verbatim.

    ``KDAKernel.forward`` looks ``chunk_kda`` up in its module globals (the same hook
    point ``eval_prefix`` uses for the recurrent kernel), so the recorded tensors are
    exactly what the trained model hands the kernel.
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.calls: list[dict[str, torch.Tensor]] = []

    def __enter__(self) -> "RecordKDAInputs":
        self.saved = kda_module.chunk_kda
        kda_module.chunk_kda = self.recording_chunk_kda
        return self

    def __exit__(self, *exc: object) -> None:
        kda_module.chunk_kda = self.saved

    def recording_chunk_kda(self, q, k, v, gate, beta, *args, **kwargs):
        output, state = self.saved(q, k, v, gate, beta, *args, **kwargs)
        if len(self.calls) < self.num_layers:
            if q.shape[0] != 1 or kwargs.get("cu_seqlens") is not None:
                raise RuntimeError(
                    "probe expects one unpacked sequence per forward (B=1, no cu_seqlens)"
                )
            tensors = dict(zip(KERNEL_INPUTS, (q, k, v, gate, beta), strict=True))
            tensors["out"] = output
            self.calls.append(
                {name: t.detach()[0].cpu() for name, t in tensors.items()}
            )
        return output, state


def run_variant(
    layer: dict[str, torch.Tensor], variant: GateReference, device: torch.device
) -> torch.Tensor:
    """Re-run ``chunk_kda`` on recorded inputs ``[N, L, ...]`` under one gate reference."""
    os.environ[GATE_REFERENCE_ENV] = variant.value
    resolve_gate_reference.cache_clear()
    assert resolve_gate_reference() is variant
    outputs = []
    for s in range(layer["q"].shape[0]):
        args = [layer[name][s : s + 1].to(device) for name in KERNEL_INPUTS]
        output, _ = chunk_kda(*args, autotune=False)
        outputs.append(output[0].cpu())
    return torch.stack(outputs)


@torch.no_grad()
def main() -> None:
    init_logger()
    args, titan_args = parse_capture_args(sys.argv[1:])
    if resolve_gate_reference() is not GateReference.CAUSAL:
        raise ValueError(f"run the capture with {GATE_REFERENCE_ENV}=causal")
    config = cast(Trainer.Config, ConfigManager().parse_args(titan_args))
    config.checkpoint.exclude_from_loading = ["dataloader", "optimizer", "lr_scheduler"]

    trainer = Trainer(config)
    try:
        device = torch.device(trainer.device)
        sequences = held_out_sequences(
            trainer.tokenizer,
            seq_len=args.capture_seq_len,
            num_sequences=args.capture_num_sequences,
            skip_documents=args.capture_skip_documents,
        )
        if not trainer.checkpointer.load(step=config.checkpoint.load_step):
            raise RuntimeError(f"no checkpoint for step {config.checkpoint.load_step}")
        for model in trainer.model_parts:
            model.eval()

        per_layer: list[list[dict[str, torch.Tensor]]] = [
            [] for _ in range(args.capture_num_layers)
        ]
        nll = torch.zeros(len(sequences), dtype=torch.float64)
        for s, sequence in enumerate(sequences):
            tokens = torch.tensor(sequence, device=device)
            with RecordKDAInputs(args.capture_num_layers) as recorder:
                nll[s] = next_token_nll(trainer, tokens).double().mean().cpu()
            if len(recorder.calls) != args.capture_num_layers:
                raise RuntimeError(f"recorded {len(recorder.calls)} KDA calls")
            for layer, call in zip(per_layer, recorder.calls, strict=True):
                layer.append(call)
            if s % 32 == 0:
                logger.info(f"sequence {s}/{len(sequences)}")
        logger.info(f"parallel NLL over capture: {nll.mean():.4f}")

        layers = [
            {name: torch.stack([call[name] for call in layer]) for name in layer[0]}
            for layer in per_layer
        ]
        for index, layer in enumerate(layers):
            layer["out_midpoint"] = run_variant(layer, GateReference.MIDPOINT, device)
            out_causal = run_variant(layer, GateReference.CAUSAL, device)
            if not torch.equal(out_causal, layer.pop("out")):
                raise RuntimeError(
                    "causal re-run differs from the recorded model output"
                )
            layer["out_causal"] = out_causal
            residual = layer["out_midpoint"].float() - out_causal.float()
            logger.info(
                "KDA layer %d: %.4f of output elements differ between references",
                index,
                (residual != 0).float().mean().item(),
            )

        args.capture_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "tokens": torch.tensor(sequences),
                "parallel_nll": nll,
                "checkpoint_step": trainer.step,
                "layers": layers,
            },
            args.capture_output,
        )
        logger.info(f"wrote {args.capture_output}")
    finally:
        trainer.close()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
