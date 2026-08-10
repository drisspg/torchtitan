# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit first-order autograd for a QUACK FlexGEMM Qwen3 SwiGLU block.

The forward fuses the gate projection, FP32 SiLU, and multiplication by the
captured up projection. FlexGEMM has no autograd formula, so the custom Function
owns the activation gradient and all three weight gradients.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch._higher_order_ops import flex_gemm

from torchtitan.models.common.feed_forward import FeedForward


# Tensor dimensions in this file:
# B: batch, L: sequence, D: model width, H: FFN width, M: flattened B * L.


class FlexGEMMSwiGLUFunction(torch.autograd.Function):
    """Own the first-order backward for a captured-up FlexGEMM SwiGLU block."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, x_MD, w1_HD, w2_DH, w3_HD, tuned):
        up_MH = torch.mm(x_MD, w3_HD.t())

        def epilogue(gate_MH):
            gate_fp32_MH = gate_MH.float()
            return (F.silu(gate_fp32_MH) * up_MH.float()).to(gate_MH.dtype), gate_MH

        gated_MH, gate_MH = flex_gemm(
            torch.mm,
            (x_MD, w1_HD.t()),
            epilogue,
            kernel_options={"backend": "QUACK", "tuned": tuned},
        )
        ctx.save_for_backward(x_MD, w1_HD, w2_DH, w3_HD, gate_MH, up_MH)
        return torch.mm(gated_MH, w2_DH.t())

    @staticmethod
    @once_differentiable
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out_MD):
        x_MD, w1_HD, w2_DH, w3_HD, gate_MH, up_MH = ctx.saved_tensors
        needs_x, needs_w1, needs_w2, needs_w3, _ = ctx.needs_input_grad

        grad_w2_DH = None
        if needs_w2:
            gated_MH = (
                F.silu(gate_MH.float()) * up_MH.float()
            ).to(gate_MH.dtype)
            grad_w2_DH = torch.mm(grad_out_MD.t(), gated_MH)

        grad_x_MD = grad_w1_HD = grad_w3_HD = None
        if needs_x or needs_w1 or needs_w3:
            grad_gated_MH = torch.mm(grad_out_MD, w2_DH)
            gate_fp32_MH = gate_MH.float()
            sigmoid_MH = torch.sigmoid(gate_fp32_MH)
            grad_gated_fp32_MH = grad_gated_MH.float()
            grad_gate_MH = (
                grad_gated_fp32_MH
                * up_MH.float()
                * sigmoid_MH
                * (1.0 + gate_fp32_MH * (1.0 - sigmoid_MH))
            ).to(gate_MH.dtype)
            grad_up_MH = (
                grad_gated_fp32_MH * gate_fp32_MH * sigmoid_MH
            ).to(up_MH.dtype)

            if needs_x:
                grad_x_MD = torch.mm(grad_gate_MH, w1_HD) + torch.mm(
                    grad_up_MH, w3_HD
                )
            if needs_w1:
                grad_w1_HD = torch.mm(grad_gate_MH.t(), x_MD)
            if needs_w3:
                grad_w3_HD = torch.mm(grad_up_MH.t(), x_MD)

        return grad_x_MD, grad_w1_HD, grad_w2_DH, grad_w3_HD, None


def flex_gemm_swiglu(
    x_MD: torch.Tensor,
    w1_HD: torch.Tensor,
    w2_DH: torch.Tensor,
    w3_HD: torch.Tensor,
    *,
    tuned: bool,
) -> torch.Tensor:
    """Apply the explicit-autograd FlexGEMM SwiGLU block to flattened input."""
    return FlexGEMMSwiGLUFunction.apply(x_MD, w1_HD, w2_DH, w3_HD, tuned)


class FlexGEMMFeedForward(FeedForward):
    """Qwen3 FeedForward replacement that preserves stock parameter FQNs."""

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        tuned: bool = True

    def __init__(self, config: Config):
        super().__init__(config)
        self.tuned = config.tuned

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        input_shape = x_BLD.shape
        x_MD = x_BLD.reshape(-1, input_shape[-1])
        out_MD = flex_gemm_swiglu(
            x_MD,
            self.w1.weight,
            self.w2.weight,
            self.w3.weight,
            tuned=self.tuned,
        )
        return out_MD.reshape(*input_shape)
