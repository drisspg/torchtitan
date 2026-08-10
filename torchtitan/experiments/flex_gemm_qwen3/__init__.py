# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Experiment-scoped Qwen3 model registry using explicit FlexGEMM autograd."""

from dataclasses import replace

from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.qwen3 import model_registry as qwen3_model_registry
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.parallelize import parallelize_qwen3
from torchtitan.protocols.model_spec import ModelSpec

from .feed_forward import FlexGEMMFeedForward


def parallelize_single_gpu(model, **kwargs):
    """Apply stock Qwen3 setup while rejecting unsupported distributed use."""
    if kwargs["parallel_dims"].world_size != 1:
        raise ValueError("The FlexGEMM Qwen3 prototype only supports one GPU")
    return parallelize_qwen3(model, **kwargs)


def model_registry(flavor: str, *, tuned: bool = True) -> ModelSpec:
    """Build dense stock Qwen3 with only FeedForward configs replaced."""
    model_spec = qwen3_model_registry(flavor)
    model_config = model_spec.model
    assert isinstance(model_config, Qwen3Model.Config)
    if any(layer.feed_forward is None for layer in model_config.layers):
        raise ValueError("The FlexGEMM Qwen3 prototype only supports dense flavors")

    for layer_config in model_config.layers:
        feed_forward = layer_config.feed_forward
        assert isinstance(feed_forward, FeedForward.Config)
        layer_config.feed_forward = FlexGEMMFeedForward.Config(
            w1=feed_forward.w1,
            w2=feed_forward.w2,
            w3=feed_forward.w3,
            param_init=feed_forward.param_init,
            sharding_config=feed_forward.sharding_config,
            tuned=tuned,
        )

    return replace(
        model_spec,
        name="flex_gemm_qwen3",
        parallelize_fn=parallelize_single_gpu,
    )


__all__ = ["model_registry"]
