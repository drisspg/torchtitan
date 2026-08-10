# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from torchtitan.experiments.flex_gemm_qwen3 import model_registry
from torchtitan.experiments.flex_gemm_qwen3.feed_forward import (
    FlexGEMMFeedForward,
    flex_gemm_swiglu,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.qwen3.model import Qwen3Model


def reference_swiglu(x_MD, w1_HD, w2_DH, w3_HD):
    """Evaluate the same mixed-precision SwiGLU expression without FlexGEMM."""
    gate_MH = torch.mm(x_MD, w1_HD.t())
    up_MH = torch.mm(x_MD, w3_HD.t())
    gated_MH = (F.silu(gate_MH.float()) * up_MH.float()).to(gate_MH.dtype)
    return torch.mm(gated_MH, w2_DH.t())


def make_inputs(*, m: int = 7, d: int = 11, h: int = 13, device, dtype):
    """Create independent nontrivial inputs for output and gradient checks."""
    torch.manual_seed(11)
    return tuple(
        (0.25 * torch.randn(*shape, device=device, dtype=dtype)).requires_grad_()
        for shape in ((m, d), (h, d), (d, h), (h, d))
    )


def assert_output_and_gradients_close(
    actual_inputs, expected_inputs, grad_out, *, rtol, atol
):
    """Compare the output and every activation/weight gradient."""
    actual = flex_gemm_swiglu(*actual_inputs, tuned=False)
    expected = reference_swiglu(*expected_inputs)
    actual.backward(grad_out)
    expected.backward(grad_out)

    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    for actual_input, expected_input in zip(
        actual_inputs, expected_inputs, strict=True
    ):
        torch.testing.assert_close(
            actual_input.grad, expected_input.grad, rtol=rtol, atol=atol
        )


def test_cpu_forward_and_all_gradients_match_reference():
    actual_inputs = make_inputs(device="cpu", dtype=torch.float32)
    expected_inputs = tuple(
        value.detach().clone().requires_grad_() for value in actual_inputs
    )
    grad_out = torch.randn(7, 11)
    assert_output_and_gradients_close(
        actual_inputs, expected_inputs, grad_out, rtol=1e-5, atol=1e-6
    )


def test_fake_forward_and_backward_cover_all_inputs():
    with FakeTensorMode():
        inputs = make_inputs(device="cpu", dtype=torch.float32)
        out = flex_gemm_swiglu(*inputs, tuned=False)
        out.sum().backward()

    assert isinstance(out, FakeTensor)
    assert out.shape == (7, 11)
    assert all(isinstance(value.grad, FakeTensor) for value in inputs)


def test_model_registry_preserves_stock_parameter_names():
    model_spec = model_registry("0.6B", tuned=False)
    model_config = model_spec.model
    assert isinstance(model_config, Qwen3Model.Config)
    assert len(model_config.layers) == 28
    assert all(
        isinstance(layer.feed_forward, FlexGEMMFeedForward.Config)
        for layer in model_config.layers
    )

    feed_forward = FlexGEMMFeedForward.Config(
        w1=Linear.Config(in_features=11, out_features=13),
        w2=Linear.Config(in_features=13, out_features=11),
        w3=Linear.Config(in_features=11, out_features=13),
        tuned=False,
    ).build()
    assert set(dict(feed_forward.named_parameters())) == {
        "w1.weight",
        "w2.weight",
        "w3.weight",
    }


def test_model_registry_rejects_moe_flavors():
    with pytest.raises(ValueError, match="only supports dense flavors"):
        model_registry("debugmodel_moe", tuned=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_fullgraph_quack_forward_and_all_gradients_match_reference():
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("QUACK captured-tile FlexGEMM test requires SM100+")

    actual_inputs = make_inputs(
        m=128, d=64, h=128, device="cuda", dtype=torch.bfloat16
    )
    expected_inputs = tuple(
        value.detach().clone().requires_grad_() for value in actual_inputs
    )
    grad_out = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)

    def actual_fn(x_MD, w1_HD, w2_DH, w3_HD):
        return flex_gemm_swiglu(x_MD, w1_HD, w2_DH, w3_HD, tuned=False)

    compiled = torch.compile(actual_fn, backend="inductor", fullgraph=True)
    actual, generated_code = run_and_get_code(compiled, *actual_inputs)
    expected = reference_swiglu(*expected_inputs)
    actual.backward(grad_out)
    expected.backward(grad_out)

    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-3)
    for actual_input, expected_input in zip(
        actual_inputs, expected_inputs, strict=True
    ):
        torch.testing.assert_close(
            actual_input.grad, expected_input.grad, rtol=5e-2, atol=5e-3
        )

    code = "\n".join(generated_code)
    assert "gemm_epimod as flex_gemm_runtime" in code
    assert "fast_math: True" in code
    assert "fastmath=True" in code
    assert "epilogue_arg_kinds=('tile',)" in code
    assert "aux_outs=" in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_autocast_returns_float32_leaf_gradients():
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("QUACK captured-tile FlexGEMM test requires SM100+")

    actual_inputs = make_inputs(m=128, d=64, h=128, device="cuda", dtype=torch.float32)
    expected_inputs = tuple(
        value.detach().clone().requires_grad_() for value in actual_inputs
    )
    grad_out = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)

    def actual_fn(x_MD, w1_HD, w2_DH, w3_HD):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return flex_gemm_swiglu(x_MD, w1_HD, w2_DH, w3_HD, tuned=False)

    def expected_fn(x_MD, w1_HD, w2_DH, w3_HD):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return reference_swiglu(x_MD, w1_HD, w2_DH, w3_HD)

    actual = torch.compile(actual_fn, backend="inductor", fullgraph=True)(
        *actual_inputs
    )
    expected = expected_fn(*expected_inputs)
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-3)
    actual.backward(grad_out)
    expected.backward(grad_out)

    for actual_input, expected_input in zip(
        actual_inputs, expected_inputs, strict=True
    ):
        assert actual_input.grad.dtype == torch.float32
        torch.testing.assert_close(
            actual_input.grad, expected_input.grad, rtol=5e-2, atol=5e-3
        )
