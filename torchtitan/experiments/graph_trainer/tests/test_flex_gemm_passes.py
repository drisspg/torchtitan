# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the dense-Qwen3 FlexGEMM graph passes.

The positive cases build a train-step graph shaped like the one
``minimal_fx_tracer`` + ``selective_activation_remat_pass`` produce for a dense
Qwen3 layer: 3-D activations reshaped for 2-D GEMMs, transposed weights, and a
backward that recomputes ``silu`` from the saved gate GEMM output.

Requires CUDA. ``test_compiled_*`` additionally requires SM100+ with CuTeDSL.
"""

import operator
import unittest

import torch
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.graph_trainer.flex_gemm_passes import (
    flex_gemm_cross_entropy_pass,
    flex_gemm_packed_w13_wgrad_fp32_pass,
    flex_gemm_residual_pass,
    flex_gemm_swiglu_pass,
    packed_w13_wgrad_layout_pass,
    QUACK_CROSS_ENTROPY_KERNEL_OPTIONS,
    QUACK_KERNEL_OPTIONS,
    QUACK_SWIGLU_KERNEL_OPTIONS,
)
from torchtitan.experiments.graph_trainer.inductor_passes import (
    full_inductor_compilation_pass,
)
from torchtitan.overrides.fused_swiglu import silu_and_mul_op
from torchtitan.tools.utils import has_cuda_capability

B, S, D, H = 2, 256, 512, 1024
CE_M, CE_K, CE_N = 32, 64, 256
FLEX_GEMM = torch.ops.higher_order.flex_gemm


def swiglu_train_step(x, w1, w3, w2, dout):
    """Dense SwiGLU forward plus the backward shape the traced graph has.

    After the memory policy and activation-remat passes, the backward re-views
    the saved gate/up GEMM outputs and recomputes ``silu`` from them, so the
    forward ``silu`` feeds only the forward ``mul``.
    """
    x_2d = x.reshape(B * S, D)
    gate_2d = torch.mm(x_2d, w1.t())
    up_2d = torch.mm(x_2d, w3.t())
    fused = F.silu(gate_2d.reshape(B, S, H)) * up_2d.reshape(B, S, H)
    out = torch.mm(fused.reshape(B * S, H), w2.t()).reshape(B, S, D)

    d_fused = torch.mm(dout.reshape(B * S, D), w2).reshape(B, S, H)
    gate, up = gate_2d.reshape(B, S, H), up_2d.reshape(B, S, H)
    d_up = d_fused * F.silu(gate)
    d_gate = torch.ops.aten.silu_backward(d_fused * up, gate)
    return out, d_gate, d_up


def packed_swiglu_train_step(x, w13, w2, dout):
    """Fused-override SwiGLU forward plus its explicit custom-op backward."""
    gate, up = torch.einsum("bsd,hgd->bshg", x, w13).unbind(-1)
    gate_2d = gate.reshape(B * S, H)
    up_2d = up.reshape(B * S, H)
    fused_2d = silu_and_mul_op(gate_2d, up_2d)
    out = torch.mm(fused_2d, w2.t()).reshape(B, S, D)

    d_fused = torch.mm(dout.reshape(B * S, D), w2)
    d_gate, d_up = torch.ops.torchtitan.silu_and_mul_backward.default(
        d_fused, gate_2d, up_2d
    )
    return out, d_gate, d_up


def packed_w13_wgrad_step(x, packed_grad):
    """Return the packed W13 gradient through the traced einsum layout."""
    x_batched = x.reshape(1, B * S, D)
    grad_batched = packed_grad.reshape(1, B * S, 2 * H)
    wgrad = torch.bmm(x_batched.transpose(1, 2), grad_batched)
    return wgrad.squeeze(0).reshape(D, H, 2).permute(1, 2, 0).to(torch.float32)


def residual_step(x, weight, residual):
    out = torch.mm(x.reshape(B * S, D), weight.t()).reshape(B, S, D)
    return out + residual


def cross_entropy_joint_step(x, weight, targets, grad_scale):
    """Chunked LM-head CE with the explicit backward shape from AOTAutograd."""
    logits = torch.ops.aten.mm.default(x, weight)
    log_probs = torch.ops.aten._log_softmax.default(logits.float(), 1, False)
    saved_log_probs = torch.ops.aten.alias.default(
        torch.ops.aten.alias.default(log_probs)
    )
    loss, total_weight = torch.ops.aten.nll_loss_forward.default(
        log_probs, targets, None, 2, -100
    )
    nll_grad = torch.ops.aten.nll_loss_backward.default(
        grad_scale, log_probs, targets, None, 2, -100, total_weight
    )
    logits_grad = torch.ops.aten._log_softmax_backward_data.default(
        nll_grad, saved_log_probs, 1, torch.float32
    )
    return loss, logits_grad.to(torch.bfloat16)


def make_inputs(dtype=torch.bfloat16, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(42)

    def normal(*shape):
        # Keep magnitudes near unit scale so BFloat16 rounding stays informative.
        return torch.randn(*shape, generator=generator, device=device, dtype=dtype) / (
            D**0.5
        )

    return (normal(B, S, D), normal(H, D), normal(H, D), normal(D, H), normal(B, S, D))


def make_packed_inputs(dtype=torch.bfloat16, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(42)

    def normal(*shape):
        return torch.randn(*shape, generator=generator, device=device, dtype=dtype) / (
            D**0.5
        )

    return normal(B, S, D), normal(H, 2, D), normal(D, H), normal(B, S, D)


def make_w13_wgrad_inputs(dtype=torch.bfloat16, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(42)
    x = torch.randn(B, S, D, generator=generator, device=device, dtype=dtype) / (
        D**0.5
    )
    packed_grad = torch.randn(
        B * S, 2 * H, generator=generator, device=device, dtype=dtype
    ) / (D**0.5)
    return x, packed_grad


def make_cross_entropy_inputs(device="cuda", grad_scale=1 / 2048):
    generator = torch.Generator(device=device).manual_seed(42)
    x = torch.randn(
        CE_M, CE_K, generator=generator, device=device, dtype=torch.bfloat16
    )
    weight = torch.randn(
        CE_K, CE_N, generator=generator, device=device, dtype=torch.bfloat16
    )
    targets = torch.randint(
        CE_N, (CE_M,), generator=generator, device=device, dtype=torch.int64
    )
    targets[::7] = -100
    return x, weight, targets, torch.tensor(grad_scale, device=device)


def make_residual_inputs(dtype=torch.bfloat16, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(42)

    def normal(*shape):
        return torch.randn(*shape, generator=generator, device=device, dtype=dtype) / (
            D**0.5
        )

    return normal(B, S, D), normal(D, D), normal(B, S, D)


def traced(fn, *args):
    return make_fx(fn, tracing_mode="fake")(*args)


def fake_inputs(gm):
    return tuple(node.meta["val"] for node in gm.graph.find_nodes(op="placeholder"))


def flex_gemm_nodes(gm):
    return list(gm.graph.find_nodes(op="call_function", target=FLEX_GEMM))


def count_target(gm, target):
    return len(list(gm.graph.find_nodes(op="call_function", target=target)))


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestFlexGemmSwiGluPass(TestCase):
    def _rewrite(self, fn, *args):
        gm = traced(fn, *args)
        gm = flex_gemm_swiglu_pass(gm, fake_inputs(gm))
        gm.graph.eliminate_dead_code()
        gm.recompile()
        return gm

    def _rewrite_w13_wgrad_layout(self, *args):
        gm = traced(packed_w13_wgrad_step, *args)
        bmm = next(
            iter(
                gm.graph.find_nodes(
                    op="call_function", target=torch.ops.aten.bmm.default
                )
            )
        )
        bmm.meta.setdefault("custom", {})["module_fqn"] = "layers.0.feed_forward"
        gm = packed_w13_wgrad_layout_pass(gm, fake_inputs(gm))
        gm.graph.eliminate_dead_code()
        gm.recompile()
        return gm

    def _rewrite_w13_wgrad_fp32(self, *args):
        gm = self._rewrite_w13_wgrad_layout(*args)
        gm = flex_gemm_packed_w13_wgrad_fp32_pass(gm, fake_inputs(gm))
        gm.graph.eliminate_dead_code()
        gm.recompile()
        return gm

    def _rewrite_residual(self, module_fqn, *args):
        gm = traced(residual_step, *args)
        output_mm = next(
            iter(
                gm.graph.find_nodes(
                    op="call_function", target=torch.ops.aten.mm.default
                )
            )
        )
        output_mm.meta.setdefault("custom", {})["module_fqn"] = module_fqn
        gm = flex_gemm_residual_pass(gm, fake_inputs(gm))
        gm.graph.eliminate_dead_code()
        gm.recompile()
        return gm

    def test_fuses_dense_swiglu_site(self):
        """The gate GEMM, silu, and mul collapse into one flex_gemm call."""
        args = make_inputs()
        gm = traced(swiglu_train_step, *args)
        gemms_before = count_target(gm, torch.ops.aten.mm.default)
        rewritten = self._rewrite(swiglu_train_step, *args)

        fused = flex_gemm_nodes(rewritten)
        self.assertEqual(len(fused), 1)
        gemm_op, body_attr, gemm_args, gemm_kwargs, kernel_options = fused[0].args
        self.assertIs(gemm_op, torch.ops.aten.mm.default)
        self.assertEqual(body_attr.op, "get_attr")
        self.assertEqual(len(gemm_args), 3)
        self.assertEqual(gemm_kwargs, {})
        self.assertEqual(kernel_options, QUACK_SWIGLU_KERNEL_OPTIONS)

        # The forward silu/mul are gone and the forward gate GEMM was absorbed;
        # the backward recompute GEMM stays.
        self.assertEqual(count_target(rewritten, torch.ops.aten.silu.default), 1)
        self.assertEqual(count_target(rewritten, torch.ops.aten.mul.Tensor), 2)
        self.assertEqual(
            count_target(rewritten, torch.ops.aten.mm.default), gemms_before - 1
        )

    def test_fuses_packed_override_swiglu_site(self):
        """The packed projection and custom activation become one FlexGEMM."""
        args = make_packed_inputs()
        gm = traced(packed_swiglu_train_step, *args)
        bmm_before = count_target(gm, torch.ops.aten.bmm.default)
        rewritten = self._rewrite(packed_swiglu_train_step, *args)

        fused = flex_gemm_nodes(rewritten)
        self.assertEqual(len(fused), 1)
        gemm_op, _, gemm_args, _, kernel_options = fused[0].args
        self.assertIs(gemm_op, torch.ops.aten.mm.default)
        self.assertEqual(len(gemm_args), 2)
        self.assertEqual(kernel_options, QUACK_KERNEL_OPTIONS)
        self.assertEqual(
            count_target(rewritten, torch.ops.aten.bmm.default), bmm_before - 1
        )
        self.assertEqual(
            count_target(rewritten, torch.ops.torchtitan.silu_and_mul.default), 0
        )

        body = getattr(rewritten, fused[0].args[1].target)
        body_output = next(iter(body.graph.find_nodes(op="output"))).args[0]
        self.assertIs(body_output[1].target, torch.ops.aten.mm.default)

        actual = rewritten(*args)
        expected = packed_swiglu_train_step(*args)
        for candidate, reference in zip(actual, expected):
            self.assertTrue(torch.equal(candidate, reference))

    def test_reorients_packed_w13_weight_gradient(self):
        """The W13 wgrad GEMM directly produces contiguous parameter layout."""
        args = make_w13_wgrad_inputs()
        rewritten = self._rewrite_w13_wgrad_layout(*args)

        self.assertEqual(count_target(rewritten, torch.ops.aten.bmm.default), 0)
        self.assertEqual(count_target(rewritten, torch.ops.aten.mm.default), 1)
        actual = rewritten(*args)
        expected = packed_w13_wgrad_step(*args)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(actual.is_contiguous())
        self.assertEqual(actual.stride(), (2 * D, D, 1))

    def test_fuses_packed_w13_weight_gradient_fp32_store(self):
        """The W13 wgrad FlexGEMM directly stores rounded FP32 values."""
        args = make_w13_wgrad_inputs()
        rewritten = self._rewrite_w13_wgrad_fp32(*args)

        self.assertEqual(len(flex_gemm_nodes(rewritten)), 1)
        self.assertEqual(count_target(rewritten, torch.ops.aten.mm.default), 0)
        actual = rewritten(*args)
        expected = packed_w13_wgrad_step(*args)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(actual.is_contiguous())
        self.assertEqual(actual.stride(), (2 * D, D, 1))

    def test_packed_override_preserves_activation_remat(self):
        """Only the forward activation is fused; its remat consumer remains."""

        def repeated_activation(x, w13):
            gate, up = torch.einsum("bsd,hgd->bshg", x, w13).unbind(-1)
            gate = gate.reshape(B * S, H)
            up = up.reshape(B * S, H)
            return silu_and_mul_op(gate, up), silu_and_mul_op(gate, up)

        x, w13, _, _ = make_packed_inputs()
        args = x, w13
        rewritten = self._rewrite(repeated_activation, *args)

        self.assertEqual(len(flex_gemm_nodes(rewritten)), 1)
        self.assertEqual(
            count_target(rewritten, torch.ops.torchtitan.silu_and_mul.default), 1
        )
        for candidate, reference in zip(rewritten(*args), repeated_activation(*args)):
            self.assertTrue(torch.equal(candidate, reference))

    def test_packed_override_rejects_unsupported_layouts(self):
        """Blocked lanes, batched projections, and offsets are safe non-matches."""
        x, w13, _, _ = make_packed_inputs()

        def blocked_lanes(x, blocked_weight):
            gate, up = torch.einsum("bsd,ghd->bsgh", x, blocked_weight).unbind(-2)
            return silu_and_mul_op(gate.reshape(B * S, H), up.reshape(B * S, H))

        blocked_weight = w13.permute(1, 0, 2).contiguous()
        self.assertEqual(
            flex_gemm_nodes(self._rewrite(blocked_lanes, x, blocked_weight)), []
        )

        def batched_projection(x, w13):
            weight = w13.permute(2, 0, 1).reshape(D, 2 * H)
            packed = torch.bmm(x, weight.expand(B, -1, -1))
            lanes = packed.view(B, S, H, 2)
            return silu_and_mul_op(
                lanes.select(-1, 0).reshape(B * S, H),
                lanes.select(-1, 1).reshape(B * S, H),
            )

        self.assertEqual(flex_gemm_nodes(self._rewrite(batched_projection, x, w13)), [])

        def with_offsets(x, w13, offsets):
            gate, up = torch.einsum("bsd,hgd->bshg", x, w13).unbind(-1)
            return silu_and_mul_op(
                gate.reshape(B * S, H), up.reshape(B * S, H), offsets
            )

        offsets = torch.tensor([0, B * S], device="cuda", dtype=torch.int32)
        self.assertEqual(
            flex_gemm_nodes(self._rewrite(with_offsets, x, w13, offsets)), []
        )

    def test_body_graph_is_the_matched_subgraph(self):
        """The traced body reproduces mm -> silu -> mul and returns the gate."""
        rewritten = self._rewrite(swiglu_train_step, *make_inputs())
        body = getattr(rewritten, flex_gemm_nodes(rewritten)[0].args[1].target)

        self.assertEqual(len(list(body.graph.find_nodes(op="placeholder"))), 3)
        targets = [
            node.target for node in body.graph.nodes if node.op == "call_function"
        ]
        self.assertEqual(
            targets,
            [
                torch.ops.aten.mm.default,
                torch.ops.aten.silu.default,
                torch.ops.aten.mul.Tensor,
            ],
        )
        main, gate = next(iter(body.graph.find_nodes(op="output"))).args[0]
        self.assertIs(main.target, torch.ops.aten.mul.Tensor)
        self.assertIs(gate.target, torch.ops.aten.mm.default)

    def test_backward_reads_auxiliary_gate_output(self):
        """Surviving gate readers move to the flex_gemm auxiliary output."""
        rewritten = self._rewrite(swiglu_train_step, *make_inputs())
        fused = flex_gemm_nodes(rewritten)[0]
        getitems = {
            node.args[1]: node
            for node in fused.users
            if node.target is operator.getitem
        }
        self.assertEqual(sorted(getitems), [0, 1])
        gate_readers = list(getitems[1].users)
        self.assertTrue(gate_readers)
        for reader in gate_readers:
            self.assertIn(reader.target, _VIEW_OR_SILU)

    def test_interpreted_output_matches_unfused_graph(self):
        """Interpreting the rewritten graph is bitwise equal to the original."""
        args = make_inputs()
        rewritten = self._rewrite(swiglu_train_step, *args)
        for fused, reference in zip(rewritten(*args), swiglu_train_step(*args)):
            self.assertTrue(torch.equal(fused, reference))

    def test_preserves_node_metadata(self):
        """New nodes carry the fake value plus the replaced nodes' annotations."""
        args = make_inputs()
        gm = traced(swiglu_train_step, *args)
        mul = next(
            node
            for node in gm.graph.find_nodes(
                op="call_function", target=torch.ops.aten.mul.Tensor
            )
            if node.args[0].target is torch.ops.aten.silu.default
        )
        gate_mm = mul.args[0].args[0].args[0]
        mul.meta.setdefault("custom", {})["module_fqn"] = "layers.0.feed_forward"
        gate_mm.meta.setdefault("custom", {})["module_fqn"] = "layers.0.feed_forward.w1"
        mul_val, gate_val = mul.meta["val"], gate_mm.meta["val"]

        gm = flex_gemm_swiglu_pass(gm, fake_inputs(gm))
        gm.graph.eliminate_dead_code()
        gm.recompile()
        fused = flex_gemm_nodes(gm)[0]
        main, gate = (
            next(node for node in fused.users if node.args[1] == index)
            for index in (0, 1)
        )
        # The HOP inherits the GEMM's annotation; the reshaped main output keeps
        # the mul's annotation and logical shape.
        self.assertEqual(fused.meta["custom"]["module_fqn"], "layers.0.feed_forward.w1")
        self.assertEqual(gate.meta["val"].shape, gate_val.shape)
        self.assertEqual(main.meta["val"].shape, (B * S, H))
        reshaped = next(iter(main.users))
        self.assertEqual(reshaped.meta["val"].shape, mul_val.shape)
        self.assertEqual(reshaped.meta["custom"]["module_fqn"], "layers.0.feed_forward")
        self.assertIsNot(fused.meta["custom"], gate_mm.meta.get("custom"))

    def test_fuses_qwen3_output_gemm_with_residual(self):
        """Attention and FFN output projections use the same residual rewrite."""
        args = make_residual_inputs()
        for module_fqn in (
            "layers.0.attention.wo",
            "layers.0.feed_forward.w2",
        ):
            with self.subTest(module_fqn=module_fqn):
                rewritten = self._rewrite_residual(module_fqn, *args)
                fused = flex_gemm_nodes(rewritten)
                self.assertEqual(len(fused), 1)
                self.assertEqual(
                    fused[0].args[-1],
                    QUACK_KERNEL_OPTIONS,
                )
                self.assertEqual(count_target(rewritten, torch.ops.aten.mm.default), 0)
                self.assertEqual(count_target(rewritten, torch.ops.aten.add.Tensor), 0)
                body = getattr(rewritten, fused[0].args[1].target)
                targets = [
                    node.target
                    for node in body.graph.nodes
                    if node.op == "call_function"
                ]
                self.assertEqual(
                    targets,
                    [torch.ops.aten.mm.default, torch.ops.aten.add.Tensor],
                )
                self.assertTrue(torch.equal(rewritten(*args), residual_step(*args)))

    def test_residual_rewrite_handles_chained_residuals(self):
        """Downstream-first rewriting preserves residual dependencies."""

        def chained(x, weight1, weight2, residual):
            first = torch.mm(x.reshape(B * S, D), weight1.t()).reshape(B, S, D)
            hidden = first + residual
            second = torch.mm(hidden.reshape(B * S, D), weight2.t()).reshape(B, S, D)
            return hidden + second

        x, weight1, residual = make_residual_inputs()
        weight2 = weight1.clone()
        args = x, weight1, weight2, residual
        gm = traced(chained, *args)
        for output_mm in gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mm.default
        ):
            output_mm.meta.setdefault("custom", {})[
                "module_fqn"
            ] = "layers.0.feed_forward.w2"
        expected = chained(*args)
        gm = flex_gemm_residual_pass(gm, fake_inputs(gm))
        self.assertEqual(len(flex_gemm_nodes(gm)), 2)
        self.assertTrue(torch.equal(gm(*args), expected))

    def test_residual_rewrite_requires_qwen3_output_fqn(self):
        """Do not turn arbitrary GEMM-plus-add graphs into FlexGEMM calls."""
        rewritten = self._rewrite_residual(
            "layers.0.attention.qkv_linear.wqkv",
            *make_residual_inputs(),
        )
        self.assertEqual(flex_gemm_nodes(rewritten), [])

    def test_residual_rewrite_preserves_later_gemm_reader(self):
        """A later backward-style reader consumes the FlexGEMM auxiliary."""

        def shared_output(x, weight, residual):
            out = torch.mm(x.reshape(B * S, D), weight.t())
            return out.reshape(B, S, D) + residual, out

        args = make_residual_inputs()
        gm = traced(shared_output, *args)
        output_mm = next(
            iter(
                gm.graph.find_nodes(
                    op="call_function", target=torch.ops.aten.mm.default
                )
            )
        )
        output_mm.meta.setdefault("custom", {})[
            "module_fqn"
        ] = "layers.0.feed_forward.w2"
        expected = shared_output(*args)
        gm = flex_gemm_residual_pass(gm, fake_inputs(gm))
        fused = flex_gemm_nodes(gm)
        self.assertEqual(len(fused), 1)
        self.assertEqual(
            sorted(user.args[1] for user in fused[0].users),
            [0, 1],
        )
        for actual, reference in zip(gm(*args), expected):
            self.assertTrue(torch.equal(actual, reference))

    def test_residual_rewrite_rejects_early_gemm_reader(self):
        """A reader before the residual add cannot consume a later auxiliary."""

        def early_reader(x, weight, residual):
            out = torch.mm(x.reshape(B * S, D), weight.t())
            before_add = out + 1.0
            return out.reshape(B, S, D) + residual, before_add

        args = make_residual_inputs()
        gm = traced(early_reader, *args)
        output_mm = next(
            iter(
                gm.graph.find_nodes(
                    op="call_function", target=torch.ops.aten.mm.default
                )
            )
        )
        output_mm.meta.setdefault("custom", {})[
            "module_fqn"
        ] = "layers.0.feed_forward.w2"
        gm = flex_gemm_residual_pass(gm, fake_inputs(gm))
        self.assertEqual(flex_gemm_nodes(gm), [])

    def test_residual_rewrite_rejects_gemm_as_its_own_residual(self):
        """Do not create a cyclic FlexGEMM capture for ``add(mm, mm)``."""

        def self_residual(x, weight):
            out = torch.mm(x.reshape(B * S, D), weight.t())
            return out + out

        x, weight, _ = make_residual_inputs()
        gm = traced(self_residual, x, weight)
        output_mm = next(
            iter(
                gm.graph.find_nodes(
                    op="call_function", target=torch.ops.aten.mm.default
                )
            )
        )
        output_mm.meta.setdefault("custom", {})[
            "module_fqn"
        ] = "layers.0.feed_forward.w2"
        gm = flex_gemm_residual_pass(gm, fake_inputs(gm))
        self.assertEqual(flex_gemm_nodes(gm), [])

    def test_no_match_when_forward_silu_is_saved(self):
        """A joint graph without activation remat keeps silu live; skip it."""

        def saved_silu(x, w1, w3, w2, dout):
            x_2d = x.reshape(B * S, D)
            gate = torch.mm(x_2d, w1.t()).reshape(B, S, H)
            up = torch.mm(x_2d, w3.t()).reshape(B, S, H)
            silu = F.silu(gate)
            out = torch.mm((silu * up).reshape(B * S, H), w2.t()).reshape(B, S, D)
            d_fused = torch.mm(dout.reshape(B * S, D), w2).reshape(B, S, H)
            return out, d_fused * silu

        self.assertEqual(flex_gemm_nodes(self._rewrite(saved_silu, *make_inputs())), [])

    def test_no_match_for_broadcast_gate_multiplier(self):
        """A multiplier that is not the GEMM's output shape is not a capture."""

        def broadcast_mul(x, w1, scale):
            gate = torch.mm(x.reshape(B * S, D), w1.t())
            return F.silu(gate) * scale

        args = make_inputs()
        scale = args[0].new_ones(1, H)
        self.assertEqual(
            flex_gemm_nodes(self._rewrite(broadcast_mul, args[0], args[1], scale)), []
        )

    def test_no_match_for_mismatched_capture_dtype(self):
        """The captured up-projection must share the GEMM's dtype."""

        def mixed_dtype(x, w1, up):
            gate = torch.mm(x.reshape(B * S, D), w1.t())
            return F.silu(gate) * up

        args = make_inputs()
        up = torch.ones(B * S, H, device="cuda", dtype=torch.float32)
        self.assertEqual(
            flex_gemm_nodes(self._rewrite(mixed_dtype, args[0], args[1], up)), []
        )

    def test_no_match_for_addmm_gate(self):
        """Only a plain ``aten.mm`` gate projection is matched."""

        def biased_gate(x, w1, bias, up):
            gate = torch.addmm(bias, x.reshape(B * S, D), w1.t())
            return F.silu(gate) * up

        args = make_inputs()
        bias = torch.zeros(H, device="cuda", dtype=torch.bfloat16)
        up = torch.ones(B * S, H, device="cuda", dtype=torch.bfloat16)
        self.assertEqual(
            flex_gemm_nodes(self._rewrite(biased_gate, args[0], args[1], bias, up)),
            [],
        )

    def test_no_match_for_other_activation(self):
        """``sigmoid`` gating is a near miss, not a SwiGLU site."""

        def sigmoid_gate(x, w1, up):
            gate = torch.mm(x.reshape(B * S, D), w1.t())
            return torch.sigmoid(gate) * up

        args = make_inputs()
        up = torch.ones(B * S, H, device="cuda", dtype=torch.bfloat16)
        self.assertEqual(
            flex_gemm_nodes(self._rewrite(sigmoid_gate, args[0], args[1], up)), []
        )

    def test_no_match_when_gate_is_read_before_the_multiply(self):
        """A gate reader scheduled before the mul cannot use the aux output."""

        def early_gate_reader(x, w1, up):
            gate = torch.mm(x.reshape(B * S, D), w1.t())
            early = gate + 1.0
            return F.silu(gate) * up, early

        args = make_inputs()
        up = torch.ones(B * S, H, device="cuda", dtype=torch.bfloat16)
        self.assertEqual(
            flex_gemm_nodes(self._rewrite(early_gate_reader, args[0], args[1], up)), []
        )

    def test_no_match_on_cpu(self):
        """The QUACK backend is CUDA-only, so CPU sites are left alone."""
        args = make_inputs(device="cpu", dtype=torch.float32)
        self.assertEqual(flex_gemm_nodes(self._rewrite(swiglu_train_step, *args)), [])


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestFlexGemmCrossEntropyPass(TestCase):
    def _rewrite(self, *args):
        gm = traced(cross_entropy_joint_step, *args)
        logits_mm = next(
            iter(
                gm.graph.find_nodes(
                    op="call_function", target=torch.ops.aten.mm.default
                )
            )
        )
        logits_mm.meta.setdefault("custom", {})["module_fqn"] = "lm_head"
        return flex_gemm_cross_entropy_pass(gm, fake_inputs(gm))

    def test_fuses_joint_cross_entropy_forward_and_backward(self):
        """The LM-head GEMM owns indexed CE statistics and explicit backward."""
        rewritten = self._rewrite(*make_cross_entropy_inputs())
        fused = flex_gemm_nodes(rewritten)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].args[-1], QUACK_CROSS_ENTROPY_KERNEL_OPTIONS)
        self.assertEqual(len(fused[0].args[2]), 3)
        self.assertEqual(count_target(rewritten, torch.ops.aten.mm.default), 0)
        self.assertEqual(
            count_target(rewritten, torch.ops.aten._log_softmax.default), 0
        )
        self.assertEqual(
            count_target(rewritten, torch.ops.aten.nll_loss_forward.default), 0
        )
        self.assertEqual(
            count_target(rewritten, torch.ops.aten.nll_loss_backward.default), 0
        )
        self.assertEqual(
            count_target(rewritten, torch.ops.aten._log_softmax_backward_data.default),
            0,
        )

        body = getattr(rewritten, fused[0].args[1].target)
        output = next(iter(body.graph.find_nodes(op="output"))).args[0]
        self.assertEqual(len(output), 3)
        self.assertEqual(output[2].meta["val"].shape, (CE_M, CE_N // 64))

    def test_interpreted_rewrite_matches_original_graph(self):
        """The functional HOP path preserves sum CE and its explicit gradient."""
        args = make_cross_entropy_inputs(grad_scale=1.0)
        reference = cross_entropy_joint_step(*args)
        actual = self._rewrite(*args)(*args)
        torch.testing.assert_close(actual[0], reference[0], atol=0, rtol=0)
        torch.testing.assert_close(actual[1], reference[1], atol=1e-8, rtol=0)

    def test_all_ignored_targets_produce_zero_loss_and_gradient(self):
        """Safe gather indices must not leak ignored rows into CE outputs."""
        args = list(make_cross_entropy_inputs(grad_scale=1.0))
        args[2].fill_(-100)
        loss, grad = self._rewrite(*args)(*args)
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(grad, torch.zeros_like(grad)))

    def test_accepts_real_qwen_output_cast_kwargs(self):
        """Same-device layout kwargs remain a dtype-only output conversion."""
        args = make_cross_entropy_inputs()
        gm = traced(cross_entropy_joint_step, *args)
        logits_mm = next(
            iter(
                gm.graph.find_nodes(
                    op="call_function", target=torch.ops.aten.mm.default
                )
            )
        )
        logits_mm.meta.setdefault("custom", {})["module_fqn"] = "lm_head"
        grad_cast = list(
            gm.graph.find_nodes(
                op="call_function", target=torch.ops.aten._to_copy.default
            )
        )[-1]
        grad_cast.kwargs = {
            "dtype": torch.bfloat16,
            "layout": torch.strided,
            "device": torch.device("cuda", 0),
        }
        gm = flex_gemm_cross_entropy_pass(gm, fake_inputs(gm))
        self.assertEqual(len(flex_gemm_nodes(gm)), 1)

    def test_missing_target_metadata_skips_rewrite(self):
        """Incomplete fake metadata is an unsupported near miss, not a crash."""
        args = make_cross_entropy_inputs()
        gm = traced(cross_entropy_joint_step, *args)
        logits_mm = next(
            iter(
                gm.graph.find_nodes(
                    op="call_function", target=torch.ops.aten.mm.default
                )
            )
        )
        logits_mm.meta.setdefault("custom", {})["module_fqn"] = "lm_head"
        targets = next(
            node
            for node in gm.graph.find_nodes(op="placeholder")
            if node.meta["val"].dtype is torch.int64
        )
        del targets.meta["val"]
        gm = flex_gemm_cross_entropy_pass(gm, ())
        self.assertEqual(flex_gemm_nodes(gm), [])

    def test_requires_lm_head_annotation(self):
        """Do not rewrite an arbitrary GEMM followed by cross entropy."""
        args = make_cross_entropy_inputs()
        gm = traced(cross_entropy_joint_step, *args)
        gm = flex_gemm_cross_entropy_pass(gm, fake_inputs(gm))
        self.assertEqual(flex_gemm_nodes(gm), [])


_VIEW_OR_SILU = (
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.silu.default,
    torch.ops.aten.silu_backward.default,
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(not has_cuda_capability(10, 0), "FlexGEMM QUACK requires SM100+")
class TestFlexGemmCompiled(TestCase):
    def test_compiled_residual_auxiliary_is_consumed(self):
        """Compiled QUACK returns a saved GEMM value for later readers."""

        def shared_output(x, weight, residual):
            out = torch.mm(x.reshape(B * S, D), weight.t())
            return out.reshape(B, S, D) + residual, out

        args = make_residual_inputs()
        rewritten = traced(shared_output, *args)
        output_mm = next(
            iter(
                rewritten.graph.find_nodes(
                    op="call_function", target=torch.ops.aten.mm.default
                )
            )
        )
        output_mm.meta.setdefault("custom", {})[
            "module_fqn"
        ] = "layers.0.feed_forward.w2"
        rewritten = flex_gemm_residual_pass(rewritten, fake_inputs(rewritten))
        rewritten, sources = run_and_get_code(
            full_inductor_compilation_pass, rewritten, fake_inputs(rewritten)
        )
        code = "\n".join(sources)
        self.assertIn("flex_gemm_runtime(", code)
        self.assertIn("tuned=True", code)
        self.assertIn("aux_outs=", code)
        actual_main, actual_saved = rewritten(*args)
        reference_main, reference_saved = shared_output(*args)
        torch.testing.assert_close(actual_main, reference_main, atol=0.002, rtol=0.02)
        self.assertTrue(torch.equal(actual_saved, reference_saved))

    def test_compiled_cross_entropy_selects_tuned_quack(self):
        """Compiled CE uses coherent rounded logits for loss and backward."""
        args = make_cross_entropy_inputs(grad_scale=1.0)
        control = traced(cross_entropy_joint_step, *args)
        control = full_inductor_compilation_pass(control, fake_inputs(control))
        expected = control(*args)

        rewritten = traced(cross_entropy_joint_step, *args)
        logits_mm = next(
            iter(
                rewritten.graph.find_nodes(
                    op="call_function", target=torch.ops.aten.mm.default
                )
            )
        )
        logits_mm.meta.setdefault("custom", {})["module_fqn"] = "lm_head"
        rewritten = flex_gemm_cross_entropy_pass(rewritten, fake_inputs(rewritten))
        rewritten, sources = run_and_get_code(
            full_inductor_compilation_pass, rewritten, fake_inputs(rewritten)
        )
        code = "\n".join(sources)
        self.assertIn("flex_gemm_runtime(", code)
        self.assertIn("tuned=True", code)
        self.assertIn("FlexGemmEpiModIndexedOutputPlan", code)
        self.assertIn("reduce_planes=2", code)
        self.assertIn("cvt.rn.bf16.f32", code)
        self.assertIn("fast_math: True", code)

        actual = rewritten(*args)
        torch.testing.assert_close(actual[0], expected[0], atol=2e-4, rtol=2e-5)
        torch.testing.assert_close(actual[1], expected[1], atol=0.015, rtol=0.05)

        valid = args[2] != -100
        actual_row_sum = actual[1][valid].float().sum(-1).abs().max()
        expected_row_sum = expected[1][valid].float().sum(-1).abs().max()
        self.assertLessEqual(actual_row_sum.item(), expected_row_sum.item() + 0.002)
        self.assertTrue(
            torch.equal(actual[1][~valid], torch.zeros_like(actual[1][~valid]))
        )

    def test_compiled_packed_swiglu_selects_quack(self):
        """Packed override lowering contracts main and saves physical lanes."""
        args = make_packed_inputs()
        control = full_inductor_compilation_pass(
            traced(packed_swiglu_train_step, *args),
            args,
        )
        rewritten = traced(packed_swiglu_train_step, *args)
        rewritten = flex_gemm_swiglu_pass(rewritten, fake_inputs(rewritten))
        rewritten.graph.eliminate_dead_code()
        rewritten.recompile()
        rewritten, sources = run_and_get_code(
            full_inductor_compilation_pass, rewritten, fake_inputs(rewritten)
        )
        code = "\n".join(sources)
        self.assertIn("flex_gemm_runtime(", code)
        self.assertIn("tuned=True", code)
        self.assertIn("cvt.rn.bf16.f32", code)
        self.assertIn("FlexGemmGroupedMainOutputTransform(group=2", code)
        self.assertIn("aux_outs=", code)

        for candidate, reference in zip(rewritten(*args), control(*args)):
            self.assertTrue(torch.equal(candidate, reference))

    def test_compiled_selects_quack_and_matches_reference(self):
        """The compiled rewrite runs QUACK FlexGEMM and stays as accurate as eager."""
        args = make_inputs()
        control = traced(swiglu_train_step, *args)
        control = full_inductor_compilation_pass(control, fake_inputs(control))

        rewritten = traced(swiglu_train_step, *args)
        rewritten = flex_gemm_swiglu_pass(rewritten, fake_inputs(rewritten))
        rewritten.graph.eliminate_dead_code()
        rewritten.recompile()
        rewritten, sources = run_and_get_code(
            full_inductor_compilation_pass, rewritten, fake_inputs(rewritten)
        )
        code = "\n".join(sources)
        self.assertIn("flex_gemm_runtime(", code)
        self.assertIn("tuned=True", code)
        self.assertIn("fast_math: True", code)
        self.assertIn("cute.math.tanh(", code)
        self.assertIn("fastmath=True", code)
        self.assertIn("epilogue_arg_kinds=('tile',)", code)
        self.assertIn("aux_outs=", code)
        # The fused site no longer runs a separate gate GEMM: the control code
        # calls one more extern mm than the rewritten code.
        control_sources = run_and_get_code(
            full_inductor_compilation_pass,
            traced(swiglu_train_step, *args),
            fake_inputs(control),
        )[1]
        self.assertEqual(
            code.count("extern_kernels.mm("),
            "\n".join(control_sources).count("extern_kernels.mm(") - 1,
        )

        reference = swiglu_train_step(*[arg.double() for arg in args])
        for fused, baseline, expected in zip(
            rewritten(*args), control(*args), reference
        ):
            fused_error = (fused.double() - expected).abs().mean()
            baseline_error = (baseline.double() - expected).abs().mean()
            # FlexGEMM applies the epilogue to the Float32 accumulator, so the
            # fused site must not be worse than the unfused BFloat16 graph.
            self.assertLessEqual(
                fused_error.item(),
                baseline_error.item() * 1.05,
            )


if __name__ == "__main__":
    run_tests()
