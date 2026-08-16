# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training recipes for the attention-gym KDA hybrid experiment."""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from . import model_registry
from .trainer import KDAHybridTrainer


def training_config(
    flavor: str,
    *,
    seq_len: int,
    steps: int,
    max_sequences: int,
    min_active_tokens: int,
    use_full_activation_checkpointing: bool,
) -> Trainer.Config:
    """Build a local C4 recipe that exercises packed varlen training."""
    model_spec = model_registry(flavor, attn_backend="varlen")
    activation_checkpoint = (
        FullAC.Config() if use_full_activation_checkpointing else SelectiveAC.Config()
    )
    return KDAHybridTrainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=default_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=seq_len,
            steps=steps,
            disable_cuda_graphs=False,
            # The experiment's nested FSDP policy keeps complete KDA mixers
            # FP32 while the surrounding Qwen projections and FFNs use BF16.
            mixed_precision_param="bfloat16",
        ),
        parallelism=ParallelismConfig(data_parallel_shard_degree=-1),
        checkpoint=CheckpointManager.Config(enable=False),
        activation_checkpoint=activation_checkpoint,
        debug=DebugConfig(seed=42),
        max_sequences=max_sequences,
        min_active_tokens=min_active_tokens,
    )


def kda_hybrid_debugmodel() -> Trainer.Config:
    """Eight-layer integration recipe for fast end-to-end debugging."""
    return training_config(
        "debugmodel",
        seq_len=256,
        steps=10,
        max_sequences=32,
        min_active_tokens=96,
        use_full_activation_checkpointing=False,
    )


def kda_hybrid_qwen3_8b() -> Trainer.Config:
    """Qwen3-8B-shaped 36-layer hybrid recipe for one or more large GPUs."""
    return training_config(
        "8B",
        seq_len=2048,
        steps=10,
        max_sequences=128,
        min_active_tokens=1024,
        use_full_activation_checkpointing=True,
    )
