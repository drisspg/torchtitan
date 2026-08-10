# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the FlexGEMM dense-SwiGLU graph pass.

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
    flex_gemm_residual_pass,
    flex_gemm_swiglu_pass,
    QUACK_KERNEL_OPTIONS,
    QUACK_SWIGLU_KERNEL_OPTIONS,
)
from torchtitan.experiments.graph_trainer.inductor_passes import (
    full_inductor_compilation_pass,
)
from torchtitan.tools.utils import has_cuda_capability

B, S, D, H = 2, 256, 512, 1024
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


def residual_step(x, weight, residual):
    out = torch.mm(x.reshape(B * S, D), weight.t()).reshape(B, S, D)
    return out + residual


def make_inputs(dtype=torch.bfloat16, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(42)

    def normal(*shape):
        # Keep magnitudes near unit scale so BFloat16 rounding stays informative.
        return torch.randn(
            *shape, generator=generator, device=device, dtype=dtype
        ) / (D**0.5)

    return (normal(B, S, D), normal(H, D), normal(H, D), normal(D, H), normal(B, S, D))


def make_residual_inputs(dtype=torch.bfloat16, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(42)

    def normal(*shape):
        return torch.randn(
            *shape, generator=generator, device=device, dtype=dtype
        ) / (D**0.5)

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
        self.assertEqual(
            reshaped.meta["custom"]["module_fqn"], "layers.0.feed_forward"
        )
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
                self.assertEqual(
                    count_target(rewritten, torch.ops.aten.mm.default), 0
                )
                self.assertEqual(
                    count_target(rewritten, torch.ops.aten.add.Tensor), 0
                )
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
                self.assertTrue(
                    torch.equal(rewritten(*args), residual_step(*args))
                )

    def test_residual_rewrite_requires_qwen3_output_fqn(self):
        """Do not turn arbitrary GEMM-plus-add graphs into FlexGEMM calls."""
        rewritten = self._rewrite_residual(
            "layers.0.attention.qkv_linear.wqkv",
            *make_residual_inputs(),
        )
        self.assertEqual(flex_gemm_nodes(rewritten), [])

    def test_residual_rewrite_rejects_shared_gemm_output(self):
        """The residual rewrite cannot erase a GEMM with another reader."""

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
        output_mm.meta.setdefault("custom", {})["module_fqn"] = (
            "layers.0.feed_forward.w2"
        )
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
        self.assertEqual(
            flex_gemm_nodes(self._rewrite(swiglu_train_step, *args)), []
        )


_VIEW_OR_SILU = (
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.silu.default,
    torch.ops.aten.silu_backward.default,
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(not has_cuda_capability(10, 0), "FlexGEMM QUACK requires SM100+")
class TestFlexGemmSwiGluCompiled(TestCase):
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
