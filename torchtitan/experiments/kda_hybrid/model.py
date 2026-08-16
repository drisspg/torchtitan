# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3-style hybrid decoder blocks backed by attention-gym KDA.

This experiment adapts attention-gym's integrated KDA training example to the
TorchTitan module and decoder contracts. KDA layers consume the same packed
sequence boundaries as global attention, so recurrent and convolution state do
not cross document boundaries.
"""

from dataclasses import dataclass

import torch
from attn_gym.linear.kda import active_token_mask
from examples.kda_training import KDAAttention as AttentionGymKDAAttention
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.models.common import Conv1d, Linear
from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    FlexAttention,
    GQAttention,
    VarlenAttention,
    VarlenMetadata,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.protocols.module import Module


# Shape suffix legend:
#   B = batch, L = physical sequence length, D = model dimension,
#   N = attention heads, H = per-head dimension, T = packed token capacity B*L.
KDAHybridAttentionMasks = dict[str, BlockMask | VarlenMetadata | None]


def create_capacity_cu_seqlens(
    positions: torch.Tensor,
    *,
    active_tokens: int,
    max_sequences: int,
) -> torch.Tensor:
    """Create fixed-capacity offsets with inactive entries equal to the endpoint.

    ``positions`` owns storage for the physical token capacity ``T`` while only
    its first ``active_tokens`` rows are logically active. The returned tensor
    always has shape ``[max_sequences + 1]``. If the active prefix contains
    ``M`` documents, entries after the real endpoint are filled with ``L``::

        [0, document_start_1, ..., L, L, ..., L]

    Metadata construction stays outside CUDA Graph capture. Replay only copies
    new values into this fixed-shape tensor.
    """
    if positions.ndim != 2:
        raise ValueError(
            f"positions must have shape [B, T], got {tuple(positions.shape)}"
        )
    token_capacity = positions.numel()
    if active_tokens < 1 or active_tokens > token_capacity:
        raise ValueError(
            f"active_tokens must be in [1, {token_capacity}], got {active_tokens}"
        )
    if max_sequences < 1:
        raise ValueError(f"max_sequences must be positive, got {max_sequences}")

    flat_positions = positions.reshape(-1)
    document_starts = (
        (flat_positions[:active_tokens] == 0).nonzero(as_tuple=True)[0].to(torch.int32)
    )
    num_sequences = document_starts.numel()
    if num_sequences == 0 or int(document_starts[0]) != 0:
        raise ValueError("the active packed prefix must start at position zero")
    if num_sequences > max_sequences:
        raise ValueError(
            f"active prefix contains {num_sequences} sequences, exceeding "
            f"max_sequences={max_sequences}"
        )

    cu_seqlens = torch.full(
        (max_sequences + 1,),
        active_tokens,
        dtype=torch.int32,
        device=positions.device,
    )
    cu_seqlens[:num_sequences] = document_starts
    return cu_seqlens


def _mask_packed_values(
    tensor_BLD: torch.Tensor,
    active_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Apply attention-gym's value-mask semantics to a TorchTitan layout."""
    if active_mask is None:
        return tensor_BLD
    mask = active_mask.reshape(
        tensor_BLD.shape[0],
        tensor_BLD.shape[1],
        *((1,) * (tensor_BLD.ndim - 2)),
    )
    return torch.where(mask, tensor_BLD, 0)


def _mask_packed_gradients(
    tensor_BLD: torch.Tensor,
    active_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Apply attention-gym's gradient-barrier semantics to a TorchTitan layout."""
    if active_mask is None:
        return tensor_BLD
    mask = active_mask.reshape(
        tensor_BLD.shape[0],
        tensor_BLD.shape[1],
        *((1,) * (tensor_BLD.ndim - 2)),
    )
    return torch.where(mask, tensor_BLD, tensor_BLD.detach())


def capacity_aware_global_attention(
    attention: GQAttention,
    x_BLD: torch.Tensor,
    metadata: VarlenMetadata,
    positions: torch.Tensor | None,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Run GQA while containing undefined varlen gradients beyond active ``L``.

    This standalone boundary intentionally repeats the block-level input/output
    masks so focused tests and future callers cannot bypass suffix containment.
    Keep the projection/norm/RoPE sequence aligned with ``GQAttention.forward``.
    """
    x_BLD = _mask_packed_values(x_BLD, active_mask)
    xq_BLNH, xk_BLNH, xv_BLNH = attention.qkv_linear(x_BLD)

    if attention.q_norm is not None or attention.k_norm is not None:
        assert attention.q_norm is not None and attention.k_norm is not None
        xq_BLNH = attention.q_norm(xq_BLNH)
        xk_BLNH = attention.k_norm(xk_BLNH)

    xq_BLNH, xk_BLNH = attention.rope(xq_BLNH, xk_BLNH, positions)

    # Varlen attention ignores the inactive suffix in forward, but its input
    # gradients outside cu_seqlens[-1] are unspecified. These masks make both
    # values and automatic-differentiation paths inert before parameterized
    # Q/K/V producers consume those gradients.
    xq_BLNH = _mask_packed_values(xq_BLNH, active_mask)
    xk_BLNH = _mask_packed_values(xk_BLNH, active_mask)
    xv_BLNH = _mask_packed_values(xv_BLNH, active_mask)
    out_BLNH = attention.inner_attention(
        xq_BLNH,
        xk_BLNH,
        xv_BLNH,
        attention_masks=metadata,
        scale=attention.scaling,
        enable_gqa=attention.enable_gqa,
    ).contiguous()

    # The output suffix is likewise outside the primitive's contract. Sanitize
    # it before the output projection performs a weight reduction.
    out_BLNH = _mask_packed_values(out_BLNH, active_mask)
    out_BLD = attention.wo(out_BLNH.flatten(2))
    return _mask_packed_gradients(out_BLD, active_mask)


class KDAAttention(AttentionGymKDAAttention, Module):
    """TorchTitan-compatible wrapper around attention-gym's fused KDA module."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        num_heads: int
        head_dim: int
        qkv_proj: Linear.Config
        qkv_conv1d: Conv1d.Config
        beta_proj: Linear.Config
        f_a_proj: Linear.Config
        f_b_proj: Linear.Config
        g_a_proj: Linear.Config
        g_b_proj: Linear.Config
        out_proj: Linear.Config
        chunk_size: int = 64
        lower_bound: float = -5.0
        fastmath: bool = False
        rms_norm_eps: float = 1e-5

    def __init__(self, config: Config) -> None:
        # AttentionGymKDAAttention owns the implementation, while TorchTitan
        # owns construction and initialization of every stateful child.
        nn.Module.__init__(self)
        if config.head_dim != 128 or config.chunk_size != 64:
            raise ValueError(
                "attention-gym fused KDA requires head_dim=128 and chunk_size=64"
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.chunk_size = config.chunk_size
        self.lower_bound = config.lower_bound
        self.backend = "fused"
        self.fastmath = config.fastmath
        self.rms_norm_eps = config.rms_norm_eps
        self.profile_ranges = False
        self.mask_inactive_capacity = True
        self.compute_dtype = torch.bfloat16

        self.qkv_proj = config.qkv_proj.build()
        self.qkv_conv1d = config.qkv_conv1d.build()
        self.beta_proj = config.beta_proj.build()
        self.f_a_proj = config.f_a_proj.build()
        self.f_b_proj = config.f_b_proj.build()
        self.g_a_proj = config.g_a_proj.build()
        self.g_b_proj = config.g_b_proj.build()
        self.out_proj = config.out_proj.build()

        self.output_norm_weight = nn.Parameter(
            torch.empty(config.head_dim, dtype=torch.float32)
        )
        self.A_log = nn.Parameter(torch.empty(config.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(
            torch.empty(config.num_heads, config.head_dim, dtype=torch.float32)
        )

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: VarlenMetadata | None = None,
    ) -> torch.Tensor:
        """Run dense or packed KDA and return a transformer residual update."""
        B, L, D = x_BLD.shape
        if attention_masks is None:
            return AttentionGymKDAAttention.forward(self, x_BLD).hidden_states
        if not isinstance(attention_masks, VarlenMetadata):
            raise TypeError(
                "KDA attention requires VarlenMetadata, got "
                f"{type(attention_masks).__name__}"
            )

        # attention-gym's packed ABI uses one physical token row. Folding B and L
        # preserves TorchTitan's flattened cu_seqlens convention.
        x_BLD = x_BLD.reshape(1, B * L, D)
        output = AttentionGymKDAAttention.forward(
            self,
            x_BLD,
            cu_seqlens=attention_masks.cu_seq_q,
        ).hidden_states
        if B == 1:
            return output
        return output.reshape(B, L, D).clone()


class KDAHybridTransformerBlock(Module):
    """Decoder block containing either fused KDA or global causal attention."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention: GQAttention.Config | None
        kda: KDAAttention.Config | None
        feed_forward: FeedForward.Config
        attention_norm: RMSNorm.Config
        ffn_norm: RMSNorm.Config

        def __post_init__(self) -> None:
            if (self.attention is None) == (self.kda is None):
                raise ValueError("exactly one of attention and kda must be configured")

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.full_attention = config.attention is not None
        self.attention_mask_key = "global_attention" if self.full_attention else "kda"
        self.attention = (
            config.attention.build()
            if config.attention is not None
            else config.kda.build()  # pyrefly: ignore [missing-attribute]
        )
        self.feed_forward = config.feed_forward.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: KDAHybridAttentionMasks | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the selected attention mixer and the shared dense FFN."""
        layer_mask = (
            attention_masks[self.attention_mask_key]
            if attention_masks is not None
            else None
        )
        active_mask = None
        if isinstance(layer_mask, VarlenMetadata):
            packed = x_BLD.reshape(1, x_BLD.shape[0] * x_BLD.shape[1], -1)
            active_mask = active_token_mask(packed, layer_mask.cu_seq_q)
            # Mask before the pre-norm and residual path. A zero cotangent does
            # not neutralize stale NaNs in a parameter-gradient reduction.
            x_BLD = _mask_packed_values(x_BLD, active_mask)

        h_BLD = self.attention_norm(x_BLD)
        if self.full_attention:
            if isinstance(layer_mask, VarlenMetadata):
                assert isinstance(self.attention, GQAttention)
                assert active_mask is not None
                h_BLD = capacity_aware_global_attention(
                    self.attention,
                    h_BLD,
                    layer_mask,
                    positions,
                    active_mask,
                )
            else:
                h_BLD = self.attention(h_BLD, layer_mask, positions)
        else:
            h_BLD = self.attention(h_BLD, layer_mask)
        x_BLD = x_BLD + h_BLD
        output = x_BLD + self.feed_forward(self.ffn_norm(x_BLD))
        return _mask_packed_values(output, active_mask)


class KDAHybridModel(Decoder):
    """Dense Qwen3-style decoder alternating three KDA layers with one global layer."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        def update_from_config(self, *, config, **kwargs) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism
            if parallelism.spmd_backend != "default":
                raise NotImplementedError(
                    "the KDA hybrid experiment currently supports only the default "
                    "SPMD backend"
                )
            if parallelism.tensor_parallel_degree > 1:
                raise NotImplementedError(
                    "tensor parallelism is not implemented for attention-gym KDA"
                )
            if parallelism.context_parallel_degree > 1:
                raise NotImplementedError(
                    "context parallelism cannot shard the recurrent KDA sequence"
                )
            if parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError(
                    "pipeline parallelism is not validated for the KDA hybrid experiment"
                )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            attention = self.first_attention
            assert isinstance(attention, GQAttention.Config)
            assert attention.head_dim is not None
            num_full_attention_layers = sum(
                layer.attention is not None for layer in self.layers
            )
            return get_dense_model_nparams_and_flops(
                model,
                num_full_attention_layers,
                attention.n_heads,
                2 * attention.head_dim,
                seq_len,
                enable_weight_tying=self.enable_weight_tying,
            )

    def get_attention_masks(
        self,
        positions: torch.Tensor,
    ) -> KDAHybridAttentionMasks:
        """Build document isolation metadata for both mixer types."""
        kda_metadata = create_varlen_metadata_for_document(positions)
        attention = self.config.first_attention
        assert attention is not None
        if isinstance(attention.inner_attention, VarlenAttention.Config):
            global_attention = kda_metadata
        elif isinstance(attention.inner_attention, FlexAttention.Config):
            global_attention = super().get_attention_masks(positions)
            assert isinstance(global_attention, BlockMask)
        else:
            raise TypeError(
                "global attention must use FlexAttention or VarlenAttention, got "
                f"{type(attention.inner_attention).__name__}"
            )
        return {
            "global_attention": global_attention,
            "kda": kda_metadata,
        }
