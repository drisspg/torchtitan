# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 FlexGEMM rewrites for the full train-step graph.

The passes cover the dense SwiGLU forward epilogue and independently fuse
eligible attention/FFN output GEMMs with their residual adds.

``aot_fx_trace`` captures forward + loss + backward in one FX graph, so the
backward of the SwiGLU block is already explicit aten code. That lets a plain
graph pass fuse the forward side without any custom ``autograd.Function``: the
pass rewrites

    gate = mm(x, w1_t)            # gate projection
    out  = silu(gate) * up        # up = mm(x, w3_t)

into a single ``flex_gemm`` HOP that runs the gate GEMM with a QUACK epilogue.
``up`` becomes a captured ``[M, N]`` epilogue tensor and the gate accumulator is
returned as a full-shape auxiliary output, because the traced backward reads the
saved gate value (``silu_backward``) and the saved up value. Returning the gate
as an auxiliary is what keeps the rewrite work-neutral: without it the standalone
gate GEMM would have to stay live for the backward, leaving two gate GEMMs.

Numerics: FlexGEMM evaluates the epilogue on the Float32 GEMM accumulator and
uses its fast-math tanh identity for SiLU, while the matched graph applies the
standard ``silu``/``mul`` to the rounded BFloat16 gate value. The fused output is
therefore not bitwise equal to the graph it replaces. The auxiliary gate output
keeps the original ``[M, N]`` rounding, so the traced backward still reads the
same kind of value it read before. The pass is opt-in for that reason.

NOTE [FlexGEMM kernels and compile-time autotuning]
``standalone_compile``, which backs both regional and full Inductor compilation,
forces ``triton.autotune_at_compile_time``. That epilogue rebuilds example
tensors for every kernel call flagged as Triton, but FlexGEMM's CuTeDSL kernel
passes ``reinterpret_tensor(...)`` operands (any transposed weight or reshaped
activation) without ``raw_args``, so the generator has nothing to reconstruct
and raises ``V.graph.get_buffer(arg) and raw_arg can't be None at the same
time``. QUACK resolves its own configuration during lowering (``tuned=True``),
so the kernel does not need Inductor's Triton autotune epilogue: recording the
kernel name as already autotuned skips it. Drop this shim once the FlexGEMM
template passes ``raw_args`` or opts out of that epilogue upstream.
"""

from __future__ import annotations

import functools
import operator
from dataclasses import dataclass

import torch
from torch._higher_order_ops.flex_gemm import (
    flex_gemm_hop,
    mark_flex_gemm_body_gemm_node,
)
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.tools.logging import logger

QUACK_KERNEL_OPTIONS = {"backend": "QUACK", "tuned": True}
QUACK_SWIGLU_KERNEL_OPTIONS = {**QUACK_KERNEL_OPTIONS, "fast_math": True}

_CUTEDSL_KERNEL_PREFIX = "cutedsl_"

_VIEW_TARGETS = (
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
)


def install_flex_gemm_codegen_shim() -> None:
    """Skip Inductor's compile-time autotune epilogue for CuTeDSL kernels.

    FlexGEMM template kernels can be renamed to ``cutedsl_fused_*`` after
    scheduler fusion, so the workaround cannot identify them by their original
    template name. Idempotent. See NOTE [FlexGEMM kernels and compile-time
    autotuning].
    """
    original = PythonWrapperCodegen._generate_kernel_call_helper
    if getattr(original, "_flex_gemm_shim", False):
        return

    @functools.wraps(original)
    def generate_kernel_call_helper(self, kernel_name, call_args, **kwargs):
        if kernel_name.startswith(_CUTEDSL_KERNEL_PREFIX):
            self.kernel_autotune_names.add(kernel_name)
        return original(self, kernel_name, call_args, **kwargs)

    generate_kernel_call_helper._flex_gemm_shim = True
    PythonWrapperCodegen._generate_kernel_call_helper = generate_kernel_call_helper


@dataclass(frozen=True, slots=True)
class SwiGluSite:
    """One eligible dense SwiGLU forward site.

    ``gate_views`` is the (possibly empty) view chain between ``gate_mm`` and
    ``silu``; every node in it dies with the rewrite. ``up`` is the 2-D
    up-projection value that becomes the captured epilogue tensor.
    """

    mul: torch.fx.Node
    silu: torch.fx.Node
    gate_mm: torch.fx.Node
    gate_views: tuple[torch.fx.Node, ...]
    up: torch.fx.Node


@dataclass(frozen=True, slots=True)
class ResidualSite:
    """One Qwen3 output GEMM followed by its residual add."""

    add: torch.fx.Node
    output_mm: torch.fx.Node
    output_views: tuple[torch.fx.Node, ...]
    residual: torch.fx.Node


_RESIDUAL_GEMM_FQN_SUFFIXES = (".attention.wo", ".feed_forward.w2")


def _fake_val(node: torch.fx.Node) -> torch.Tensor | None:
    """Return the node's fake tensor metadata, or ``None`` if it has none."""
    val = node.meta.get("val")
    return val if isinstance(val, torch.Tensor) else None


def _static_shape(val: torch.Tensor) -> tuple[int, ...] | None:
    """Return the shape when every dimension is a plain int, else ``None``."""
    if any(not isinstance(dim, int) for dim in val.shape):
        return None
    return tuple(val.shape)


def _has_single_user(node: torch.fx.Node, user: torch.fx.Node) -> bool:
    return len(node.users) == 1 and next(iter(node.users)) is user


def _gate_gemm_chain(
    silu: torch.fx.Node,
) -> tuple[torch.fx.Node, tuple[torch.fx.Node, ...]] | None:
    """Walk view-only producers from ``silu``'s input back to the gate GEMM.

    Every view in the chain must feed only its successor so that the whole chain
    dies with the rewrite. Returns ``(gemm_candidate, views)`` or ``None``.
    """
    views: tuple[torch.fx.Node, ...] = ()
    consumer = silu
    node = silu.args[0]
    while isinstance(node, torch.fx.Node) and node.target in _VIEW_TARGETS:
        if not _has_single_user(node, consumer):
            return None
        views += (node,)
        consumer = node
        node = node.args[0]
    if not isinstance(node, torch.fx.Node):
        return None
    return node, views


def _view_source_with_shape(
    node: torch.fx.Node, shape: tuple[int, ...]
) -> torch.fx.Node | None:
    """Follow view-only producers of ``node`` until one has shape ``shape``."""
    current = node
    while True:
        val = _fake_val(current)
        if val is None:
            return None
        if _static_shape(val) == shape:
            return current
        if current.target not in _VIEW_TARGETS:
            return None
        current = current.args[0]
        if not isinstance(current, torch.fx.Node):
            return None


def _module_fqn(node: torch.fx.Node) -> str:
    custom = node.meta.get("custom")
    if not isinstance(custom, dict):
        return ""
    module_fqn = custom.get("module_fqn")
    return module_fqn if isinstance(module_fqn, str) else ""


def _output_gemm_chain(
    output: torch.fx.Node, add: torch.fx.Node
) -> tuple[torch.fx.Node, tuple[torch.fx.Node, ...]] | None:
    """Walk a single-use view chain from an add input to its output GEMM."""
    views: tuple[torch.fx.Node, ...] = ()
    consumer = add
    node = output
    while node.target in _VIEW_TARGETS:
        if not _has_single_user(node, consumer):
            return None
        views += (node,)
        consumer = node
        producer = node.args[0]
        if not isinstance(producer, torch.fx.Node):
            return None
        node = producer
    if node.target is not torch.ops.aten.mm.default:
        return None
    if not _has_single_user(node, consumer):
        return None
    return node, views


def match_residual_site(add: torch.fx.Node) -> ResidualSite | None:
    """Match a Qwen3 attention/FFN output GEMM followed by a residual add."""
    if add.target is not torch.ops.aten.add.Tensor or len(add.args) != 2:
        return None
    if add.kwargs.get("alpha", 1) != 1:
        return None

    for output, residual in (add.args, add.args[::-1]):
        if not isinstance(output, torch.fx.Node) or not isinstance(
            residual, torch.fx.Node
        ):
            continue
        chain = _output_gemm_chain(output, add)
        if chain is None:
            continue
        output_mm, output_views = chain
        if not _module_fqn(output_mm).endswith(_RESIDUAL_GEMM_FQN_SUFFIXES):
            continue

        gemm_val = _fake_val(output_mm)
        residual_val = _fake_val(residual)
        add_val = _fake_val(add)
        if gemm_val is None or residual_val is None or add_val is None:
            continue
        gemm_shape = _static_shape(gemm_val)
        if (
            gemm_val.device.type != "cuda"
            or gemm_shape is None
            or len(gemm_shape) != 2
            or add_val.numel() != gemm_val.numel()
            or residual_val.numel() != gemm_val.numel()
            or add_val.dtype is not gemm_val.dtype
            or residual_val.dtype is not gemm_val.dtype
            or residual_val.device != gemm_val.device
        ):
            continue
        return ResidualSite(
            add=add,
            output_mm=output_mm,
            output_views=output_views,
            residual=residual,
        )
    return None


def match_swiglu_site(
    mul: torch.fx.Node, node_order: dict[torch.fx.Node, int]
) -> SwiGluSite | None:
    """Match ``silu(mm(a, b)) * up`` at ``mul``, or return ``None``.

    The site is only eligible when the fused rewrite is provably equivalent:
    the gate GEMM is a static 2-D CUDA ``mm``, ``up`` is a same-dtype value of
    the GEMM's output shape, ``silu`` and the gate view chain feed nothing but
    this ``mul`` (so they can be deleted), and every surviving reader of the
    gate value is scheduled after ``mul`` (so it can read the fused auxiliary
    output instead).
    """
    if mul.target is not torch.ops.aten.mul.Tensor or len(mul.args) != 2:
        return None
    silu, up_arg = mul.args
    if not isinstance(silu, torch.fx.Node) or not isinstance(up_arg, torch.fx.Node):
        return None
    if silu.target is not torch.ops.aten.silu.default:
        silu, up_arg = up_arg, silu
    if silu.target is not torch.ops.aten.silu.default:
        return None
    if not _has_single_user(silu, mul):
        return None

    chain = _gate_gemm_chain(silu)
    if chain is None:
        return None
    gate_mm, gate_views = chain
    if gate_mm.target is not torch.ops.aten.mm.default:
        return None

    gemm_val = _fake_val(gate_mm)
    mul_val = _fake_val(mul)
    if gemm_val is None or mul_val is None or gemm_val.device.type != "cuda":
        return None
    gemm_shape = _static_shape(gemm_val)
    if gemm_shape is None or len(gemm_shape) != 2:
        return None
    if mul_val.numel() != gemm_val.numel() or mul_val.dtype is not gemm_val.dtype:
        return None

    up = _view_source_with_shape(up_arg, gemm_shape)
    if up is None or _fake_val(up).dtype is not gemm_val.dtype:
        return None

    matched = {*gate_views, silu, mul}
    if any(
        user not in matched and node_order[user] < node_order[mul]
        for user in gate_mm.users
    ):
        return None
    return SwiGluSite(
        mul=mul, silu=silu, gate_mm=gate_mm, gate_views=gate_views, up=up
    )


def _build_body_graph(
    a_val: torch.Tensor, b_val: torch.Tensor, up_val: torch.Tensor
) -> torch.fx.GraphModule:
    """Trace the FlexGEMM body ``(silu(mm(a, b)) * up, mm(a, b))``.

    The body is the matched subgraph itself, traced from the same fake tensors,
    so the HOP's eager and fake semantics reproduce the replaced nodes exactly.
    """

    def body(a, b, up):
        gate = torch.ops.aten.mm.default(a, b)
        return torch.ops.aten.mul.Tensor(torch.ops.aten.silu.default(gate), up), gate

    with a_val.fake_mode:
        body_gm = make_fx(body)(a_val, b_val, up_val)
    mark_flex_gemm_body_gemm_node(body_gm, torch.ops.aten.mm.default)
    return body_gm


def _build_residual_body_graph(
    a_val: torch.Tensor,
    b_val: torch.Tensor,
    residual_val: torch.Tensor,
) -> torch.fx.GraphModule:
    """Trace the FlexGEMM body ``mm(a, b) + residual``."""

    def body(a, b, residual):
        out = torch.ops.aten.add.Tensor(
            torch.ops.aten.mm.default(a, b), residual
        )
        return (out,)

    with a_val.fake_mode:
        body_gm = make_fx(body)(a_val, b_val, residual_val)
    mark_flex_gemm_body_gemm_node(body_gm, torch.ops.aten.mm.default)
    return body_gm


def _inherit_meta(node: torch.fx.Node, source: torch.fx.Node, val) -> None:
    """Carry the replaced node's provenance and annotations onto its rewrite."""
    node.meta.update({key: v for key, v in source.meta.items() if key != "val"})
    if "custom" in source.meta:
        node.meta["custom"] = dict(source.meta["custom"])
    node.meta["val"] = val


def _fuse_site(gm: torch.fx.GraphModule, site: SwiGluSite, body_name: str) -> None:
    """Replace one matched site with a ``flex_gemm`` call and its two outputs."""
    graph = gm.graph
    a_node, b_node = site.gate_mm.args
    a_val, b_val, up_val = (_fake_val(node) for node in (a_node, b_node, site.up))
    body_gm = _build_body_graph(a_val, b_val, up_val)
    gm.register_module(body_name, body_gm)
    with a_val.fake_mode:
        main_val, gate_val = body_gm(a_val, b_val, up_val)
    mul_val = site.mul.meta["val"]

    with graph.inserting_before(site.mul):
        body_attr = graph.get_attr(body_name)
        fused = graph.call_function(
            flex_gemm_hop,
            (
                torch.ops.aten.mm.default,
                body_attr,
                (a_node, b_node, site.up),
                {},
                dict(QUACK_SWIGLU_KERNEL_OPTIONS),
            ),
        )
        main = graph.call_function(operator.getitem, (fused, 0))
        gate = graph.call_function(operator.getitem, (fused, 1))
        out = main
        if tuple(mul_val.shape) != tuple(main_val.shape):
            out = graph.call_function(
                torch.ops.aten.reshape.default, (main, list(mul_val.shape))
            )

    _inherit_meta(fused, site.gate_mm, (main_val, gate_val))
    _inherit_meta(main, site.mul, main_val)
    _inherit_meta(gate, site.gate_mm, gate_val)
    if out is not main:
        _inherit_meta(out, site.mul, mul_val)

    site.mul.replace_all_uses_with(out)
    graph.erase_node(site.mul)
    graph.erase_node(site.silu)
    for view in site.gate_views:
        graph.erase_node(view)
    # Every surviving gate reader is ordered after the fused node (checked by
    # the matcher), so they can be redirected to the auxiliary output.
    site.gate_mm.replace_all_uses_with(gate)
    graph.erase_node(site.gate_mm)


def _fuse_residual_site(
    gm: torch.fx.GraphModule, site: ResidualSite, body_name: str
) -> None:
    """Replace one output GEMM plus residual add with ``flex_gemm``."""
    graph = gm.graph
    a_node, b_node = site.output_mm.args
    a_val, b_val = (_fake_val(node) for node in (a_node, b_node))
    residual_val = _fake_val(site.residual)
    gemm_shape = tuple(site.output_mm.meta["val"].shape)
    with a_val.fake_mode:
        residual_2d_val = residual_val.reshape(gemm_shape)
    body_gm = _build_residual_body_graph(a_val, b_val, residual_2d_val)
    gm.register_module(body_name, body_gm)
    with a_val.fake_mode:
        (main_val,) = body_gm(a_val, b_val, residual_2d_val)
    add_val = site.add.meta["val"]

    with graph.inserting_before(site.add):
        residual_2d = site.residual
        if tuple(residual_val.shape) != gemm_shape:
            residual_2d = graph.call_function(
                torch.ops.aten.reshape.default,
                (site.residual, list(gemm_shape)),
            )
            _inherit_meta(residual_2d, site.residual, residual_2d_val)
        body_attr = graph.get_attr(body_name)
        fused = graph.call_function(
            flex_gemm_hop,
            (
                torch.ops.aten.mm.default,
                body_attr,
                (a_node, b_node, residual_2d),
                {},
                dict(QUACK_KERNEL_OPTIONS),
            ),
        )
        main = graph.call_function(operator.getitem, (fused, 0))
        out = main
        if tuple(add_val.shape) != tuple(main_val.shape):
            out = graph.call_function(
                torch.ops.aten.reshape.default, (main, list(add_val.shape))
            )

    _inherit_meta(fused, site.add, (main_val,))
    _inherit_meta(main, site.add, main_val)
    if out is not main:
        _inherit_meta(out, site.add, add_val)
    site.add.replace_all_uses_with(out)
    graph.erase_node(site.add)
    for view in site.output_views:
        graph.erase_node(view)
    graph.erase_node(site.output_mm)


def flex_gemm_residual_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Fuse Qwen3 attention/FFN output GEMMs with their residual adds."""
    sites = [
        site
        for node in gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.add.Tensor
        )
        if (site := match_residual_site(node)) is not None
    ]
    if not sites:
        logger.info("flex_gemm_residual_pass: no eligible Qwen3 residual site")
        return gm

    install_flex_gemm_codegen_shim()
    for index, site in enumerate(sites):
        _fuse_residual_site(gm, site, f"flex_gemm_residual_body_{index}")
    gm.graph.lint()
    gm.recompile()
    logger.info(
        f"flex_gemm_residual_pass: fused {len(sites)} Qwen3 output GEMMs "
        f"with residual adds using kernel_options={QUACK_KERNEL_OPTIONS}"
    )
    return gm


def flex_gemm_swiglu_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Fuse eligible dense SwiGLU forward sites into QUACK ``flex_gemm`` calls.

    Must run before the terminal Inductor pass, which owns the graph once it
    collapses into a compiled artifact. Sites that do not match the exact
    contract in :func:`match_swiglu_site` are left untouched.
    """
    node_order = {node: index for index, node in enumerate(gm.graph.nodes)}
    sites = [
        site
        for node in gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mul.Tensor
        )
        if (site := match_swiglu_site(node, node_order)) is not None
    ]
    if not sites:
        logger.info("flex_gemm_swiglu_pass: no eligible dense SwiGLU site")
        return gm

    install_flex_gemm_codegen_shim()
    for index, site in enumerate(sites):
        _fuse_site(gm, site, f"flex_gemm_swiglu_body_{index}")
    gm.graph.lint()
    gm.recompile()
    logger.info(
        f"flex_gemm_swiglu_pass: fused {len(sites)} dense SwiGLU sites into "
        f"flex_gemm with kernel_options={QUACK_SWIGLU_KERNEL_OPTIONS}"
    )
    return gm
