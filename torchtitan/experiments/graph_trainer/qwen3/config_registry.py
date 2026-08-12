# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.qwen3.config_registry import (
    qwen3_0_6b,
    qwen3_14b,
    qwen3_8b,
    qwen3_debugmodel,
    qwen3_moe_debug,
)

from . import model_registry


def graph_trainer_qwen3_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(qwen3_debugmodel(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_qwen3_debugmodel_moe() -> GraphTrainer.Config:
    config = to_graph_trainer_config(qwen3_moe_debug(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_qwen3_0_6b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(qwen3_0_6b(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_qwen3_0_6b_flex_gemm() -> GraphTrainer.Config:
    config = graph_trainer_qwen3_0_6b()
    config.compile.enable_flex_gemm_swiglu = True
    config.compile.enable_flex_gemm_cross_entropy = True
    return config


def graph_trainer_qwen3_8b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(qwen3_8b(), model_registry)
    # This workload leaves enough headroom for full-graph compilation on one B200.
    config.training.local_batch_size = 2
    config.training.seq_len = 1024
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_qwen3_8b_fused_swiglu() -> GraphTrainer.Config:
    config = graph_trainer_qwen3_8b()
    config.override.imports.append("torchtitan.overrides.fused_swiglu.fused_swiglu")
    return config


def graph_trainer_qwen3_8b_optimized() -> GraphTrainer.Config:
    config = graph_trainer_qwen3_8b_fused_swiglu()
    config.compile.enable_packed_w13_wgrad_layout = True
    config.compile.enable_flex_gemm_packed_w13_wgrad_fp32 = True
    config.compile.enable_flex_gemm_packed_swiglu_backward = True
    config.compile.enable_shared_lm_head_weight_cast = True
    return config


def graph_trainer_qwen3_8b_flex_gemm() -> GraphTrainer.Config:
    config = graph_trainer_qwen3_8b_fused_swiglu()
    config.compile.enable_flex_gemm_swiglu = True
    config.compile.enable_flex_gemm_cross_entropy = True
    return config


def graph_trainer_qwen3_14b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(qwen3_14b(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config
