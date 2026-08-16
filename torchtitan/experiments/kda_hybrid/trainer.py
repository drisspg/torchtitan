# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fixed-capacity packed training support for the KDA hybrid experiment."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.models.common.attention import VarlenMetadata
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer

from .model import create_capacity_cu_seqlens, KDAHybridAttentionMasks


_CAPACITY_ACTIVE_TOKENS_KEY = "_capacity_active_tokens"
_CAPACITY_OFFSETS_KEY = "_capacity_cu_seqlens"


class KDAHybridTrainer(Trainer):
    """Train with fixed physical token and sequence-metadata capacities."""

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        max_sequences: int = 128
        """Maximum logical sequences represented by the captured graph."""

        min_active_tokens: int = -1
        """Minimum active prefix used by the three-batch capacity stress cycle.

        A negative value keeps every physical token active. A positive value
        cycles through ``T``, ``max(3*T/4, min_active_tokens)``, and
        ``min_active_tokens`` while retaining fixed physical capacity ``T``.
        """

        def __post_init__(self) -> None:
            Trainer.Config.__post_init__(self)
            if self.training.local_batch_size != 1:
                raise ValueError(
                    "KDA fixed-capacity packing currently requires local_batch_size=1"
                )
            if self.max_sequences < 1:
                raise ValueError(
                    f"max_sequences must be positive, got {self.max_sequences}"
                )
            if self.min_active_tokens == 0 or (
                self.min_active_tokens > self.training.seq_len
            ):
                raise ValueError(
                    "min_active_tokens must be negative or in "
                    f"[1, {self.training.seq_len}], got {self.min_active_tokens}"
                )

    def _active_tokens_for_batch(self, batch_index: int, capacity: int) -> int:
        minimum = self.config.min_active_tokens
        if minimum < 0:
            return capacity
        three_quarters = max(minimum, 3 * capacity // 4)
        return (capacity, three_quarters, minimum)[batch_index % 3]

    def batch_generator(
        self,
        data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]],
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Mask a varying active prefix and attach fixed-size CPU offsets."""
        for batch_index, (input_dict, labels) in enumerate(
            super().batch_generator(data_iterable)
        ):
            positions = input_dict.get("positions")
            if positions is None:
                raise ValueError("KDA capacity packing requires document positions")
            if positions.ndim != 2 or positions.shape[0] != 1:
                raise ValueError(
                    "KDA capacity packing currently expects positions with shape "
                    f"[1, T], got {tuple(positions.shape)}"
                )
            if labels.numel() != positions.numel():
                raise ValueError(
                    "labels and positions must have the same physical token capacity"
                )

            capacity = positions.numel()
            active_tokens = self._active_tokens_for_batch(batch_index, capacity)
            cu_seqlens = create_capacity_cu_seqlens(
                positions,
                active_tokens=active_tokens,
                max_sequences=self.config.max_sequences,
            )

            labels = labels.clone()
            labels.reshape(-1)[active_tokens:] = IGNORE_INDEX
            self.metrics_processor.ntokens_since_last_log -= capacity - active_tokens
            input_dict = {
                **input_dict,
                _CAPACITY_ACTIVE_TOKENS_KEY: active_tokens,
                _CAPACITY_OFFSETS_KEY: cu_seqlens,
            }

            if batch_index < 10:
                num_sequences = int((cu_seqlens[:-1] < active_tokens).sum())
                logger.info(
                    "KDA capacity batch %d: active_tokens=%d/%d, sequences=%d/%d",
                    batch_index + 1,
                    active_tokens,
                    capacity,
                    num_sequences,
                    self.config.max_sequences,
                )
            yield input_dict, labels

    def post_dataloading_process(
        self,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build graph-stable metadata without dynamic GPU ``nonzero`` output."""
        inputs = input_dict["input"]
        active_tokens = input_dict[_CAPACITY_ACTIVE_TOKENS_KEY]
        if not isinstance(active_tokens, int):
            raise TypeError("capacity active token count must remain a host integer")
        cu_seqlens = input_dict[_CAPACITY_OFFSETS_KEY]
        if cu_seqlens.shape != (self.config.max_sequences + 1,):
            raise ValueError(
                "capacity cu_seqlens must have shape "
                f"[{self.config.max_sequences + 1}], got {tuple(cu_seqlens.shape)}"
            )

        token_capacity = inputs.numel()
        metadata = VarlenMetadata(
            cu_seq_q=cu_seqlens,
            cu_seq_k=cu_seqlens,
            max_q=token_capacity,
            max_k=token_capacity,
        )
        attention_masks: KDAHybridAttentionMasks = {
            "global_attention": metadata,
            "kda": metadata,
        }
        extra_kwargs: dict[str, Any] = {
            key: value
            for key, value in input_dict.items()
            if key
            not in (
                "input",
                _CAPACITY_ACTIVE_TOKENS_KEY,
                _CAPACITY_OFFSETS_KEY,
            )
        }
        extra_kwargs["attention_masks"] = attention_masks

        self.ntokens_seen += active_tokens
        return inputs, labels, extra_kwargs
