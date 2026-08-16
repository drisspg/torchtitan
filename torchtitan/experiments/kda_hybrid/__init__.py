# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model registry for the attention-gym KDA hybrid experiment."""

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.models.common import Conv1d, Linear
from torchtitan.models.qwen3 import qwen3_configs
from torchtitan.protocols.model_spec import ModelSpec

from .model import KDAAttention, KDAHybridModel, KDAHybridTransformerBlock
from .parallelize import parallelize_kda_hybrid


__all__ = [
    "KDAAttention",
    "KDAHybridModel",
    "KDAHybridTransformerBlock",
    "model_registry",
]


_LINEAR_INIT: dict[str, Callable] = {
    "weight": partial(nn.init.normal_, std=0.02),
}
_KDA_PARAM_INIT: dict[str, Callable] = {
    "output_norm_weight": nn.init.ones_,
    "A_log": nn.init.zeros_,
    "dt_bias": nn.init.zeros_,
}


def residual_output_init(num_layers: int) -> dict[str, Callable]:
    """Initialize residual output projections with depth-scaled variance."""
    return {
        "weight": partial(nn.init.normal_, std=0.02 / (2 * num_layers) ** 0.5),
    }


def kda_attention_config(
    *,
    hidden_size: int,
    num_heads: int,
    num_layers: int,
    head_dim: int = 128,
    chunk_size: int = 64,
    short_conv_kernel_size: int = 4,
) -> KDAAttention.Config:
    """Build the integrated attention-gym KDA module used by each linear layer."""
    projection_size = num_heads * head_dim
    return KDAAttention.Config(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        chunk_size=chunk_size,
        qkv_proj=Linear.Config(
            in_features=hidden_size,
            out_features=3 * projection_size,
            param_init=_LINEAR_INIT,
        ),
        qkv_conv1d=Conv1d.Config(
            in_channels=3 * projection_size,
            out_channels=3 * projection_size,
            kernel_size=short_conv_kernel_size,
            groups=3 * projection_size,
            bias=False,
        ),
        beta_proj=Linear.Config(
            in_features=hidden_size,
            out_features=num_heads,
            param_init=_LINEAR_INIT,
        ),
        f_a_proj=Linear.Config(
            in_features=hidden_size,
            out_features=head_dim,
            param_init=_LINEAR_INIT,
        ),
        f_b_proj=Linear.Config(
            in_features=head_dim,
            out_features=projection_size,
            param_init=_LINEAR_INIT,
        ),
        g_a_proj=Linear.Config(
            in_features=hidden_size,
            out_features=head_dim,
            param_init=_LINEAR_INIT,
        ),
        g_b_proj=Linear.Config(
            in_features=head_dim,
            out_features=projection_size,
            param_init=_LINEAR_INIT,
        ),
        out_proj=Linear.Config(
            in_features=projection_size,
            out_features=hidden_size,
            param_init=residual_output_init(num_layers),
        ),
        param_init=_KDA_PARAM_INIT,
    )


def hybrid_qwen3_config(
    flavor: str,
    *,
    attn_backend: str,
    full_attention_interval: int = 4,
) -> KDAHybridModel.Config:
    """Replace three out of every four Qwen3 attention layers with KDA."""
    if full_attention_interval < 2:
        raise ValueError("full_attention_interval must be at least 2")

    base = qwen3_configs[flavor](attn_backend=attn_backend)
    if base.dim % 128:
        raise ValueError(f"model dim {base.dim} must be divisible by KDA head dim 128")

    num_layers = len(base.layers)
    layers = []
    for layer_id, layer in enumerate(base.layers):
        full_attention = (layer_id + 1) % full_attention_interval == 0
        layers.append(
            KDAHybridTransformerBlock.Config(
                attention=layer.attention if full_attention else None,
                kda=(
                    None
                    if full_attention
                    else kda_attention_config(
                        hidden_size=base.dim,
                        num_heads=base.dim // 128,
                        num_layers=num_layers,
                    )
                ),
                feed_forward=layer.feed_forward,
                attention_norm=layer.attention_norm,
                ffn_norm=layer.ffn_norm,
            )
        )

    return KDAHybridModel.Config(
        dim=base.dim,
        vocab_size=base.vocab_size,
        lm_head=base.lm_head,
        tok_embeddings=base.tok_embeddings,
        norm=base.norm,
        layers=layers,
        enable_weight_tying=base.enable_weight_tying,
    )


def model_registry(
    flavor: str,
    *,
    attn_backend: str = "varlen",
) -> ModelSpec:
    """Create a random-init Qwen3-shaped hybrid KDA model specification."""
    return ModelSpec(
        name="kda_hybrid",
        flavor=flavor,
        model=hybrid_qwen3_config(flavor, attn_backend=attn_backend),
        parallelize_fn=parallelize_kda_hybrid,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
