# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Self-contained training configuration for the Qwen3 FlexGEMM experiment."""

from torchtitan.config import CompileConfig
from torchtitan.models.qwen3.config_registry import qwen3_0_6b
from torchtitan.trainer import Trainer

from . import model_registry


def configure_local_run(config: Trainer.Config) -> Trainer.Config:
    """Apply the shared deterministic local-data measurement contract."""
    config.hf_assets_path = "./tests/assets/tokenizer"
    config.dataloader.dataset = "c4_test"
    config.training.local_batch_size = 2
    config.training.seq_len = 1024
    config.training.steps = 10
    config.debug.seed = 42
    config.debug.deterministic = True
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.parallelism.context_parallel_degree = 1
    config.parallelism.expert_parallel_degree = 1
    config.activation_checkpoint = None
    config.compile = CompileConfig(enable=True, components=["model"])
    config.metrics.enable_tensorboard = True
    config.metrics.log_freq = 1
    return config


def stock_qwen3_0_6b() -> Trainer.Config:
    """Configure the stock Qwen3-0.6B baseline for local 10-step testing."""
    return configure_local_run(qwen3_0_6b())


def flex_gemm_qwen3_0_6b() -> Trainer.Config:
    """Configure the FlexGEMM Qwen3-0.6B prototype for local 10-step testing."""
    config = qwen3_0_6b()
    config.model_spec = model_registry("0.6B", tuned=True)
    return configure_local_run(config)
