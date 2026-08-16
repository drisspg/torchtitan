# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelization policy for the KDA hybrid experiment."""

import torch
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import (
    apply_fsdp_to_decoder,
    get_fsdp_reshard_after_forward_policy,
)

from .model import KDAAttention, KDAHybridModel


def parallelize_kda_hybrid(
    model: KDAHybridModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> KDAHybridModel:
    """Apply AC, compile, and mixed-precision FSDP with FP32 KDA units.

    Attention-gym's gate kernels require ``A_log`` and ``dt_bias`` to remain
    FP32. Nesting each complete KDA mixer in an FP32 FSDP unit preserves those
    parameters, while the enclosing block's norms/FFN and every global-attention
    block use the recipe's normal mixed-precision dtype. KDA projections still
    compute in BF16 through explicit casts in the attention-gym implementation.
    """
    if parallel_dims.tp_enabled or parallel_dims.cp_enabled or parallel_dims.pp_enabled:
        raise NotImplementedError(
            "KDA hybrid parallelization supports only data parallelism"
        )

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    if compile_config.enable and "model" in compile_config.components:
        apply_compile(
            model,
            compile_config=compile_config,
            parallel_dims=parallel_dims,
        )

    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
    kda_mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        cast_forward_inputs=False,
    )
    kda_fsdp_config = {
        "mesh": dp_mesh,
        "mp_policy": kda_mp_policy,
    }
    if training.enable_cpu_offload:
        kda_fsdp_config["offload_policy"] = CPUOffloadPolicy()
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward,
        pp_enabled=False,
    )
    for module in model.modules():
        if isinstance(module, KDAAttention):
            fully_shard(
                module,
                **kda_fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=False,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )
    return model
