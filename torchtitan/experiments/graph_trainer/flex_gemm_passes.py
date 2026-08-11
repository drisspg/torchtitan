# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 FlexGEMM rewrites for the full train-step graph.

The passes cover dense SwiGLU forward and backward, attention/FFN output
residual adds, and chunked LM-head cross entropy with its explicit backward.

``aot_fx_trace`` captures forward + loss + backward in one FX graph, so the
backward of the SwiGLU block is already explicit aten code. The pass handles
both TorchTitan feed-forward layouts. The stock layout rewrites

    gate = mm(x, w1_t)            # gate projection
    out  = silu(gate) * up        # up = mm(x, w3_t)

into a ``flex_gemm`` HOP that captures ``up`` and returns the saved gate as an
auxiliary. The ``fused_swiglu`` override instead computes interleaved gate/up
lanes with one singleton-batch BMM. That pattern is canonicalized to a 2-D MM;
its FlexGEMM epilogue contracts each lane pair into the logical SwiGLU output
and returns the unchanged packed GEMM result for the explicit backward. Both
forms therefore remove the standalone activation kernel without duplicating a
projection GEMM.

The cross-entropy pass similarly moves target-logit gather and group-64 online
logsumexp onto each LM-head GEMM. A small final reduction computes the summed
loss, while a pointwise expression replaces NLL and log-softmax backward. This
removes the full Float32 log-probability tensor from the joint graph.

Numerics: the stock-layout SwiGLU rewrite evaluates its fast-math SiLU on the
Float32 GEMM accumulator. The packed override rewrite explicitly rounds the
accumulator through BFloat16 and uses exact math, preserving the custom op's
activation boundary while still returning the identical packed backward state.
Cross entropy similarly rounds the accumulator before its target gather, LSE,
and backward so all three use one coherent logits value. Its grouped LSE uses
fast math and can still differ slightly from the original reduction order. All
three passes remain opt-in.

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
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.tools.logging import logger

QUACK_KERNEL_OPTIONS = {"backend": "QUACK", "tuned": True}
QUACK_SWIGLU_KERNEL_OPTIONS = {**QUACK_KERNEL_OPTIONS, "fast_math": True}
QUACK_CROSS_ENTROPY_KERNEL_OPTIONS = {**QUACK_KERNEL_OPTIONS, "fast_math": True}

_CROSS_ENTROPY_GROUP = 64
_IGNORE_INDEX = -100

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
class PackedSwiGluSite:
    """One fused-override forward SwiGLU consumer of a packed BMM."""

    swiglu: torch.fx.Node
    packed_bmm: torch.fx.Node


@dataclass(frozen=True, slots=True)
class PackedW13WgradLayoutSite:
    """One packed W13 weight-gradient BMM with a strided parameter output."""

    cast: torch.fx.Node
    bmm: torch.fx.Node
    x_batched: torch.fx.Node
    grad_batched: torch.fx.Node


@dataclass(frozen=True, slots=True)
class ResidualSite:
    """One Qwen3 output GEMM followed by its residual add."""

    add: torch.fx.Node
    output_mm: torch.fx.Node
    output_views: tuple[torch.fx.Node, ...]
    residual: torch.fx.Node
    preserve_output: bool


@dataclass(frozen=True, slots=True)
class CrossEntropySite:
    """One chunked LM-head GEMM and its explicit CE forward/backward."""

    logits_mm: torch.fx.Node
    log_softmax: torch.fx.Node
    loss: torch.fx.Node
    targets: torch.fx.Node
    grad_scale: torch.fx.Node
    grad_to_bf16: torch.fx.Node


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


def _depends_on(node: torch.fx.Node, ancestor: torch.fx.Node) -> bool:
    """Return whether ``node`` transitively reads ``ancestor``."""
    pending = list(node.all_input_nodes)
    visited: set[torch.fx.Node] = set()
    while pending:
        producer = pending.pop()
        if producer is ancestor:
            return True
        if producer not in visited:
            visited.add(producer)
            pending.extend(producer.all_input_nodes)
    return False


def _is_exact_dtype_cast(node: torch.fx.Node, dtype: torch.dtype) -> bool:
    """Recognize an ``_to_copy`` that changes only dtype."""
    if (
        node.target is not torch.ops.aten._to_copy.default
        or len(node.args) != 1
        or set(node.kwargs) - {"dtype", "layout", "device"}
        or node.kwargs.get("dtype") is not dtype
        or not isinstance(node.args[0], torch.fx.Node)
    ):
        return False
    source_val = _fake_val(node.args[0])
    output_val = _fake_val(node)
    return (
        source_val is not None
        and output_val is not None
        and node.kwargs.get("layout", source_val.layout) is source_val.layout
        and node.kwargs.get("device", source_val.device) == source_val.device
        and output_val.dtype is dtype
        and output_val.device == source_val.device
        and output_val.layout is source_val.layout
        and output_val.shape == source_val.shape
        and output_val.stride() == source_val.stride()
    )


def _unique_user(node: torch.fx.Node, target) -> torch.fx.Node | None:
    """Return the unique user with ``target``, or ``None``."""
    matches = [user for user in node.users if user.target is target]
    return matches[0] if len(matches) == 1 else None


def _getitem_user(node: torch.fx.Node, index: int) -> torch.fx.Node | None:
    """Return the unique ``operator.getitem(node, index)`` user."""
    matches = [
        user
        for user in node.users
        if user.target is operator.getitem and user.args == (node, index)
    ]
    return matches[0] if len(matches) == 1 else None


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
    output: torch.fx.Node,
    add: torch.fx.Node,
    node_order: dict[torch.fx.Node, int],
) -> tuple[torch.fx.Node, tuple[torch.fx.Node, ...], bool] | None:
    """Walk the forward view chain and classify later GEMM readers."""
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
    if node.target is not torch.ops.aten.mm.default or consumer not in node.users:
        return None
    surviving_users = [user for user in node.users if user is not consumer]
    if any(node_order[user] < node_order[add] for user in surviving_users):
        return None
    return node, views, bool(surviving_users)


def match_residual_site(
    add: torch.fx.Node, node_order: dict[torch.fx.Node, int]
) -> ResidualSite | None:
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
        chain = _output_gemm_chain(output, add, node_order)
        if chain is None:
            continue
        output_mm, output_views, preserve_output = chain
        if (
            not _module_fqn(output_mm).endswith(_RESIDUAL_GEMM_FQN_SUFFIXES)
            or residual is output_mm
            or _depends_on(residual, output_mm)
        ):
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
            preserve_output=preserve_output,
        )
    return None


def _lm_head_gemm(log_softmax: torch.fx.Node) -> torch.fx.Node | None:
    """Follow the loss cast/view chain back to a single-use LM-head GEMM."""
    logits_float = log_softmax.args[0]
    if not isinstance(logits_float, torch.fx.Node) or not _is_exact_dtype_cast(
        logits_float, torch.float32
    ):
        return None

    consumer = logits_float
    node = logits_float.args[0]
    while isinstance(node, torch.fx.Node) and node.target in _VIEW_TARGETS:
        if not _has_single_user(node, consumer):
            return None
        consumer = node
        node = node.args[0]
    if (
        not isinstance(node, torch.fx.Node)
        or node.target is not torch.ops.aten.mm.default
        or not _has_single_user(node, consumer)
        or _module_fqn(node) != "lm_head"
    ):
        return None
    return node


def _log_softmax_backward(
    log_softmax: torch.fx.Node, nll_backward: torch.fx.Node
) -> torch.fx.Node | None:
    """Follow the saved-log-probability alias chain to its backward op."""
    alias = _unique_user(log_softmax, torch.ops.aten.alias.default)
    if alias is None:
        return None
    while True:
        next_alias = _unique_user(alias, torch.ops.aten.alias.default)
        if next_alias is None:
            break
        if len(alias.users) != 1:
            return None
        alias = next_alias
    backward = _unique_user(alias, torch.ops.aten._log_softmax_backward_data.default)
    if (
        backward is None
        or len(alias.users) != 1
        or backward.args[:3] != (nll_backward, alias, 1)
        or backward.args[3] is not torch.float32
    ):
        return None
    return backward


def match_cross_entropy_site(log_softmax: torch.fx.Node) -> CrossEntropySite | None:
    """Match one sum-reduced chunked LM-head CE forward and backward."""
    if (
        log_softmax.target is not torch.ops.aten._log_softmax.default
        or log_softmax.args[1:] != (1, False)
    ):
        return None
    logits_mm = _lm_head_gemm(log_softmax)
    if logits_mm is None:
        return None

    a_node, b_node = logits_mm.args
    logits_float = log_softmax.args[0]
    if not all(
        isinstance(node, torch.fx.Node) for node in (a_node, b_node, logits_float)
    ):
        return None
    a_val, b_val, logits_val, logits_float_val, log_softmax_val = (
        _fake_val(node)
        for node in (a_node, b_node, logits_mm, logits_float, log_softmax)
    )
    if any(
        val is None
        for val in (a_val, b_val, logits_val, logits_float_val, log_softmax_val)
    ):
        return None
    a_shape, b_shape, logits_shape = (
        _static_shape(val) for val in (a_val, b_val, logits_val)
    )
    if (
        a_shape is None
        or b_shape is None
        or logits_shape is None
        or len(a_shape) != 2
        or len(b_shape) != 2
        or logits_shape != (a_shape[0], b_shape[1])
        or a_shape[1] != b_shape[0]
        or logits_shape[1] % _CROSS_ENTROPY_GROUP
        or any(val.device.type != "cuda" for val in (a_val, b_val, logits_val))
        or len({a_val.device, b_val.device, logits_val.device}) != 1
        or any(val.dtype is not torch.bfloat16 for val in (a_val, b_val, logits_val))
        or logits_float_val.device != logits_val.device
        or logits_float_val.dtype is not torch.float32
        or _static_shape(logits_float_val) != logits_shape
        or log_softmax_val.device != logits_val.device
        or log_softmax_val.dtype is not torch.float32
        or _static_shape(log_softmax_val) != logits_shape
    ):
        return None

    nll_forward = _unique_user(log_softmax, torch.ops.aten.nll_loss_forward.default)
    nll_backward = _unique_user(log_softmax, torch.ops.aten.nll_loss_backward.default)
    if nll_forward is None or nll_backward is None:
        return None
    if nll_forward.args[0] is not log_softmax or nll_forward.args[2:] != (
        None,
        2,
        _IGNORE_INDEX,
    ):
        return None
    targets = nll_forward.args[1]
    if not isinstance(targets, torch.fx.Node):
        return None
    targets_val = _fake_val(targets)
    if (
        targets_val is None
        or targets_val.device != logits_val.device
        or targets_val.dtype not in (torch.int32, torch.int64)
        or _static_shape(targets_val) != (logits_shape[0],)
        or tuple(targets_val.stride()) != (1,)
    ):
        return None
    loss = _getitem_user(nll_forward, 0)
    total_weight = _getitem_user(nll_forward, 1)
    if (
        loss is None
        or total_weight is None
        or set(nll_forward.users) != {loss, total_weight}
        or set(total_weight.users) != {nll_backward}
    ):
        return None
    if nll_backward.args[1:] != (
        log_softmax,
        targets,
        None,
        2,
        _IGNORE_INDEX,
        total_weight,
    ) or not isinstance(nll_backward.args[0], torch.fx.Node):
        return None

    log_softmax_backward = _log_softmax_backward(log_softmax, nll_backward)
    if (
        log_softmax_backward is None
        or set(nll_backward.users) != {log_softmax_backward}
        or len(log_softmax.users) != 3
    ):
        return None
    grad_to_bf16 = _unique_user(log_softmax_backward, torch.ops.aten._to_copy.default)
    if grad_to_bf16 is None or not _is_exact_dtype_cast(grad_to_bf16, torch.bfloat16):
        return None
    grad_scale = nll_backward.args[0]
    grad_scale_val = _fake_val(grad_scale)
    grad_val = _fake_val(grad_to_bf16)
    if (
        not isinstance(grad_scale, torch.fx.Node)
        or grad_scale_val is None
        or grad_scale_val.device != logits_val.device
        or grad_scale_val.dtype is not torch.float32
        or _static_shape(grad_scale_val) != ()
        or grad_val is None
        or grad_val.device != logits_val.device
        or grad_val.dtype is not torch.bfloat16
        or _static_shape(grad_val) != logits_shape
    ):
        return None
    return CrossEntropySite(
        logits_mm=logits_mm,
        log_softmax=log_softmax,
        loss=loss,
        targets=targets,
        grad_scale=grad_scale,
        grad_to_bf16=grad_to_bf16,
    )


def _packed_lane_source(
    node: torch.fx.Node,
) -> tuple[torch.fx.Node, int] | None:
    """Return the shared unbind and lane index feeding a packed SwiGLU input."""
    while node.target in _VIEW_TARGETS:
        producer = node.args[0]
        if not isinstance(producer, torch.fx.Node):
            return None
        node = producer
    if (
        node.target is not operator.getitem
        or len(node.args) != 2
        or not isinstance(node.args[0], torch.fx.Node)
        or not isinstance(node.args[1], int)
    ):
        return None
    unbind, index = node.args
    if unbind.target is not torch.ops.aten.unbind.int:
        return None
    return unbind, index


def _packed_bmm_source(unbind: torch.fx.Node) -> torch.fx.Node | None:
    """Follow the override's post-contraction views back to its packed BMM."""
    node = unbind.args[0]
    while isinstance(node, torch.fx.Node) and node.target in (
        *_VIEW_TARGETS,
        torch.ops.aten.permute.default,
    ):
        node = node.args[0]
    if not isinstance(node, torch.fx.Node):
        return None
    return node if node.target is torch.ops.aten.bmm.default else None


def match_packed_swiglu_site(swiglu: torch.fx.Node) -> PackedSwiGluSite | None:
    """Match one interleaved packed projection followed by exact SwiGLU.

    The override's lane tensor must be contiguous with trailing shape ``[H, 2]``.
    Together with the singleton BMM batch and equal element counts, that proves
    physical columns are ordered ``gate0, up0, gate1, up1, ...``. The explicit
    backward may keep reading those lanes after the forward activation is fused.
    """
    schema = getattr(swiglu.target, "_schema", None)
    if (
        schema is None
        or schema.name != "torchtitan::silu_and_mul"
        or len(swiglu.args) < 2
        or not all(isinstance(arg, torch.fx.Node) for arg in swiglu.args[:2])
        or (len(swiglu.args) > 2 and swiglu.args[2] is not None)
        or swiglu.kwargs.get("offsets") is not None
    ):
        return None
    gate, up = swiglu.args[:2]
    gate_lane = _packed_lane_source(gate)
    up_lane = _packed_lane_source(up)
    if gate_lane is None or up_lane is None:
        return None
    gate_unbind, gate_index = gate_lane
    up_unbind, up_index = up_lane
    if gate_unbind is not up_unbind or (gate_index, up_index) != (0, 1):
        return None

    packed_bmm = _packed_bmm_source(gate_unbind)
    if packed_bmm is None:
        return None
    packed_val = _fake_val(packed_bmm)
    gate_val = _fake_val(gate)
    up_val = _fake_val(up)
    swiglu_val = _fake_val(swiglu)
    lane_source = gate_unbind.args[0]
    lane_source_val = (
        _fake_val(lane_source) if isinstance(lane_source, torch.fx.Node) else None
    )
    if (
        packed_val is None
        or gate_val is None
        or up_val is None
        or swiglu_val is None
        or lane_source_val is None
    ):
        return None
    packed_shape = _static_shape(packed_val)
    gate_shape = _static_shape(gate_val)
    lane_source_shape = _static_shape(lane_source_val)
    if (
        packed_val.device.type != "cuda"
        or packed_shape is None
        or len(packed_shape) != 3
        or packed_shape[0] != 1
        or gate_shape is None
        or len(gate_shape) != 2
        or packed_shape[1] != gate_shape[0]
        or packed_shape[2] != 2 * gate_shape[1]
        or lane_source_shape is None
        or lane_source_shape[-2:] != (gate_shape[1], 2)
        or lane_source_val.numel() != packed_val.numel()
        or not lane_source_val.is_contiguous()
        or up_val.shape != gate_val.shape
        or swiglu_val.shape != gate_val.shape
        or up_val.dtype is not gate_val.dtype
        or swiglu_val.dtype is not gate_val.dtype
        or packed_val.dtype is not gate_val.dtype
    ):
        return None

    lhs_val, rhs_val = (_fake_val(arg) for arg in packed_bmm.args)
    if lhs_val is None or rhs_val is None:
        return None
    lhs_shape = _static_shape(lhs_val)
    rhs_shape = _static_shape(rhs_val)
    if (
        lhs_shape is None
        or rhs_shape is None
        or lhs_shape[:2] != (1, packed_shape[1])
        or rhs_shape != (1, lhs_shape[2], packed_shape[2])
        or lhs_val.dtype is not packed_val.dtype
        or rhs_val.dtype is not packed_val.dtype
        or lhs_val.device != packed_val.device
        or rhs_val.device != packed_val.device
    ):
        return None
    return PackedSwiGluSite(swiglu=swiglu, packed_bmm=packed_bmm)


def match_packed_w13_wgrad_layout_site(
    cast: torch.fx.Node,
) -> PackedW13WgradLayoutSite | None:
    """Match the packed W13 wgrad chain that returns a strided parameter grad."""
    if not _is_exact_dtype_cast(cast, torch.float32):
        return None
    cast_val = _fake_val(cast)
    if cast_val is None:
        return None
    cast_shape = _static_shape(cast_val)
    if cast_shape is None or len(cast_shape) != 3 or cast_shape[1] != 2:
        return None

    consumer = cast
    node = cast.args[0]
    layout_targets = (
        *_VIEW_TARGETS,
        torch.ops.aten.permute.default,
        torch.ops.aten.squeeze.dim,
    )
    while isinstance(node, torch.fx.Node) and node.target in layout_targets:
        if not _has_single_user(node, consumer):
            return None
        consumer = node
        node = node.args[0]
    if (
        not isinstance(node, torch.fx.Node)
        or node.target is not torch.ops.aten.bmm.default
        or not _has_single_user(node, consumer)
        or not _module_fqn(node).endswith(".feed_forward")
    ):
        return None
    bmm = node
    lhs, grad_batched = bmm.args
    if (
        not isinstance(lhs, torch.fx.Node)
        or not isinstance(grad_batched, torch.fx.Node)
        or lhs.target is not torch.ops.aten.transpose.int
        or lhs.args[1:] != (1, 2)
        or not isinstance(lhs.args[0], torch.fx.Node)
    ):
        return None
    x_batched = lhs.args[0]
    x_val, grad_val, bmm_val = (
        _fake_val(value) for value in (x_batched, grad_batched, bmm)
    )
    if x_val is None or grad_val is None or bmm_val is None:
        return None
    x_shape, grad_shape, bmm_shape = (
        _static_shape(value) for value in (x_val, grad_val, bmm_val)
    )
    h, lanes, d = cast_shape
    if (
        x_shape is None
        or grad_shape is None
        or bmm_shape is None
        or x_shape[0] != 1
        or grad_shape[0] != 1
        or x_shape[1] != grad_shape[1]
        or x_shape[2] != d
        or grad_shape[2] != lanes * h
        or bmm_shape != (1, d, lanes * h)
        or any(value.device.type != "cuda" for value in (x_val, grad_val, bmm_val))
        or len({x_val.device, grad_val.device, bmm_val.device}) != 1
        or any(
            value.dtype is not torch.bfloat16 for value in (x_val, grad_val, bmm_val)
        )
        or cast_val.device != bmm_val.device
        or cast_val.dtype is not torch.float32
    ):
        return None
    return PackedW13WgradLayoutSite(
        cast=cast,
        bmm=bmm,
        x_batched=x_batched,
        grad_batched=grad_batched,
    )


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
    return SwiGluSite(mul=mul, silu=silu, gate_mm=gate_mm, gate_views=gate_views, up=up)


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


def _build_packed_swiglu_body_graph(
    a_val: torch.Tensor,
    b_val: torch.Tensor,
    logical_n: int,
) -> torch.fx.GraphModule:
    """Trace interleaved packed MM -> SwiGLU plus the saved packed result."""
    m = a_val.shape[0]

    def body(a, b):
        packed = torch.ops.aten.mm.default(a, b)
        rounded = _round_bfloat16_to_float(packed.float())
        lanes = rounded.view(m, logical_n, 2)
        gate = lanes.select(-1, 0)
        up = lanes.select(-1, 1)
        main = torch.ops.aten.mul.Tensor(torch.ops.aten.silu.default(gate), up)
        return main.to(packed.dtype), packed

    with a_val.fake_mode:
        body_gm = make_fx(body)(a_val, b_val)
    mark_flex_gemm_body_gemm_node(body_gm, torch.ops.aten.mm.default)
    return body_gm


def _build_residual_body_graph(
    a_val: torch.Tensor,
    b_val: torch.Tensor,
    residual_val: torch.Tensor,
    *,
    preserve_output: bool,
) -> torch.fx.GraphModule:
    """Trace residual fusion, optionally returning the unfused GEMM value."""

    def body(a, b, residual):
        gemm = torch.ops.aten.mm.default(a, b)
        out = torch.ops.aten.add.Tensor(gemm, residual)
        return (out, gemm) if preserve_output else (out,)

    with a_val.fake_mode:
        body_gm = make_fx(body)(a_val, b_val, residual_val)
    mark_flex_gemm_body_gemm_node(body_gm, torch.ops.aten.mm.default)
    return body_gm


def _round_bfloat16_to_float(value: torch.Tensor) -> torch.Tensor:
    """Round a Float32 accumulator through BFloat16 without an FX cast fold."""
    return inline_asm_elementwise(
        value,
        asm_str="{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.f32.bf16 $0, h; }",
        constraints="=f,f",
        dtype=torch.float32,
    )


def _build_cross_entropy_body_graph(
    a_val: torch.Tensor,
    b_val: torch.Tensor,
    targets_val: torch.Tensor,
) -> torch.fx.GraphModule:
    """Trace LM-head GEMM outputs from one coherently rounded logits value."""
    m, n = a_val.shape[0], b_val.shape[1]

    def body(a, b, targets):
        accumulator = torch.ops.aten.mm.default(a, b)
        logits_float = _round_bfloat16_to_float(accumulator.float())
        logits = logits_float.to(accumulator.dtype)
        return (
            logits,
            logits.gather(1, targets[:, None]).squeeze(1),
            logits_float.view(
                m, n // _CROSS_ENTROPY_GROUP, _CROSS_ENTROPY_GROUP
            ).logsumexp(-1),
        )

    with a_val.fake_mode:
        body_gm = make_fx(body)(a_val, b_val, targets_val)
    mark_flex_gemm_body_gemm_node(body_gm, torch.ops.aten.mm.default)
    return body_gm


def _trace_cross_entropy_targets(
    targets_val: torch.Tensor,
) -> torch.fx.GraphModule:
    """Trace the ignore mask and valid gather indices."""

    def preprocess(targets):
        valid = targets != _IGNORE_INDEX
        return valid, torch.where(valid, targets, 0)

    with targets_val.fake_mode:
        return make_fx(preprocess)(targets_val)


def _trace_cross_entropy_loss(
    target_logits_val: torch.Tensor,
    partial_lse_val: torch.Tensor,
    valid_val: torch.Tensor,
) -> torch.fx.GraphModule:
    """Trace the final LSE reduction and sum-reduced token loss."""

    def finish(target_logits, partial_lse, valid):
        lse = partial_lse.logsumexp(-1)
        losses = torch.where(valid, lse - target_logits.float(), 0.0)
        return losses.sum(), lse

    with target_logits_val.fake_mode:
        return make_fx(finish)(target_logits_val, partial_lse_val, valid_val)


def _trace_cross_entropy_backward(
    logits_val: torch.Tensor,
    lse_val: torch.Tensor,
    targets_val: torch.Tensor,
    valid_val: torch.Tensor,
    scale_val: torch.Tensor,
) -> torch.fx.GraphModule:
    """Trace ``(softmax - one_hot) * scale`` without materializing one-hot."""
    n = logits_val.shape[1]

    def backward(logits, lse, targets, valid, scale):
        columns = torch.arange(n, device=logits.device, dtype=targets.dtype)
        probabilities = torch.exp(logits.float() - lse[:, None])
        target_columns = columns[None, :] == targets[:, None]
        grad = torch.where(target_columns, probabilities - 1.0, probabilities)
        return torch.where(valid[:, None], grad * scale, 0.0).to(logits.dtype)

    with logits_val.fake_mode:
        return make_fx(backward)(logits_val, lse_val, targets_val, valid_val, scale_val)


def _inline_graph(
    graph: torch.fx.Graph,
    graph_module: torch.fx.GraphModule,
    args: tuple[torch.fx.Node, ...],
    before: torch.fx.Node,
):
    """Copy a traced tensor expression into ``graph`` before ``before``."""
    placeholders = list(graph_module.graph.find_nodes(op="placeholder"))
    if len(placeholders) != len(args):
        raise AssertionError("inlined graph argument count must match placeholders")
    with graph.inserting_before(before):
        return graph.graph_copy(graph_module.graph, dict(zip(placeholders, args)))


def _fake_outputs(graph_module: torch.fx.GraphModule):
    """Return output metadata from a graph traced with fake tensors."""
    output = next(iter(graph_module.graph.find_nodes(op="output")))
    return torch.fx.node.map_arg(output.args[0], lambda node: node.meta["val"])


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


def _fuse_packed_swiglu_site(
    gm: torch.fx.GraphModule,
    site: PackedSwiGluSite,
    body_name: str,
) -> None:
    """Replace one packed BMM plus custom SwiGLU with grouped-main FlexGEMM."""
    graph = gm.graph
    lhs, rhs = site.packed_bmm.args
    lhs_val, rhs_val = (_fake_val(node) for node in (lhs, rhs))
    packed_bmm_val = _fake_val(site.packed_bmm)
    swiglu_val = _fake_val(site.swiglu)
    m, k = lhs_val.shape[-2:]
    physical_n = rhs_val.shape[-1]
    logical_n = physical_n // 2
    with lhs_val.fake_mode:
        lhs_2d_val = lhs_val.reshape(m, k)
        rhs_2d_val = rhs_val.reshape(k, physical_n)
    body_gm = _build_packed_swiglu_body_graph(
        lhs_2d_val,
        rhs_2d_val,
        logical_n,
    )
    gm.register_module(body_name, body_gm)
    with lhs_val.fake_mode:
        main_val, packed_val = body_gm(lhs_2d_val, rhs_2d_val)

    with graph.inserting_before(site.packed_bmm):
        lhs_2d = graph.call_function(torch.ops.aten.reshape.default, (lhs, [m, k]))
        rhs_2d = graph.call_function(
            torch.ops.aten.reshape.default, (rhs, [k, physical_n])
        )
        body_attr = graph.get_attr(body_name)
        fused = graph.call_function(
            flex_gemm_hop,
            (
                torch.ops.aten.mm.default,
                body_attr,
                (lhs_2d, rhs_2d),
                {},
                # Exact math plus the explicit BF16 round preserves the
                # fused_swiglu override's activation boundary.
                dict(QUACK_KERNEL_OPTIONS),
            ),
        )
        main = graph.call_function(operator.getitem, (fused, 0))
        packed = graph.call_function(operator.getitem, (fused, 1))
        packed_bmm = graph.call_function(
            torch.ops.aten.reshape.default,
            (packed, list(packed_bmm_val.shape)),
        )

    _inherit_meta(lhs_2d, lhs, lhs_2d_val)
    _inherit_meta(rhs_2d, rhs, rhs_2d_val)
    _inherit_meta(fused, site.packed_bmm, (main_val, packed_val))
    _inherit_meta(main, site.swiglu, swiglu_val)
    _inherit_meta(packed, site.packed_bmm, packed_val)
    _inherit_meta(packed_bmm, site.packed_bmm, packed_bmm_val)

    site.swiglu.replace_all_uses_with(main)
    graph.erase_node(site.swiglu)
    site.packed_bmm.replace_all_uses_with(packed_bmm)
    graph.erase_node(site.packed_bmm)


def _rewrite_packed_w13_wgrad_layout(
    gm: torch.fx.GraphModule,
    site: PackedW13WgradLayoutSite,
) -> None:
    """Produce the packed W13 gradient directly in parameter layout."""
    graph = gm.graph
    x_val = _fake_val(site.x_batched)
    grad_val = _fake_val(site.grad_batched)
    cast_val = _fake_val(site.cast)
    _, m, d = x_val.shape
    h, lanes, _ = cast_val.shape
    physical_n = h * lanes
    with x_val.fake_mode:
        x_2d_val = x_val.reshape(m, d)
        grad_2d_val = grad_val.reshape(m, physical_n)
        grad_t_val = grad_2d_val.t()
        wgrad_val = torch.mm(grad_t_val, x_2d_val)
        packed_val = wgrad_val.reshape(h, lanes, d)
        output_val = packed_val.to(torch.float32)

    with graph.inserting_before(site.cast):
        x_2d = graph.call_function(
            torch.ops.aten.reshape.default,
            (site.x_batched, [m, d]),
        )
        grad_2d = graph.call_function(
            torch.ops.aten.reshape.default,
            (site.grad_batched, [m, physical_n]),
        )
        grad_t = graph.call_function(torch.ops.aten.t.default, (grad_2d,))
        wgrad = graph.call_function(torch.ops.aten.mm.default, (grad_t, x_2d))
        packed = graph.call_function(
            torch.ops.aten.reshape.default,
            (wgrad, [h, lanes, d]),
        )
        output = graph.call_function(
            torch.ops.aten._to_copy.default,
            (packed,),
            {"dtype": torch.float32},
        )

    _inherit_meta(x_2d, site.x_batched, x_2d_val)
    _inherit_meta(grad_2d, site.grad_batched, grad_2d_val)
    _inherit_meta(grad_t, site.grad_batched, grad_t_val)
    _inherit_meta(wgrad, site.bmm, wgrad_val)
    _inherit_meta(packed, site.cast, packed_val)
    _inherit_meta(output, site.cast, output_val)
    site.cast.replace_all_uses_with(output)


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
    body_gm = _build_residual_body_graph(
        a_val,
        b_val,
        residual_2d_val,
        preserve_output=site.preserve_output,
    )
    gm.register_module(body_name, body_gm)
    body_vals = _fake_outputs(body_gm)
    main_val = body_vals[0]
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
        preserved = (
            graph.call_function(operator.getitem, (fused, 1))
            if site.preserve_output
            else None
        )
        out = main
        if tuple(add_val.shape) != tuple(main_val.shape):
            out = graph.call_function(
                torch.ops.aten.reshape.default, (main, list(add_val.shape))
            )

    _inherit_meta(fused, site.add, body_vals)
    _inherit_meta(main, site.add, main_val)
    if preserved is not None:
        _inherit_meta(preserved, site.output_mm, body_vals[1])
    if out is not main:
        _inherit_meta(out, site.add, add_val)
    site.add.replace_all_uses_with(out)
    graph.erase_node(site.add)
    for view in site.output_views:
        graph.erase_node(view)
    if preserved is not None:
        site.output_mm.replace_all_uses_with(preserved)
    graph.erase_node(site.output_mm)


def _fuse_cross_entropy_site(
    gm: torch.fx.GraphModule,
    site: CrossEntropySite,
    body_name: str,
) -> None:
    """Replace one joint LM-head CE forward/backward with FlexGEMM."""
    graph = gm.graph
    a_node, b_node = site.logits_mm.args
    a_val, b_val = (_fake_val(node) for node in (a_node, b_node))
    targets_val = _fake_val(site.targets)
    scale_val = _fake_val(site.grad_scale)

    target_gm = _trace_cross_entropy_targets(targets_val)
    valid, safe_targets = _inline_graph(
        graph, target_gm, (site.targets,), site.log_softmax
    )
    valid_val, safe_targets_val = _fake_outputs(target_gm)

    body_gm = _build_cross_entropy_body_graph(a_val, b_val, safe_targets_val)
    gm.register_module(body_name, body_gm)
    logits_val, target_logits_val, partial_lse_val = _fake_outputs(body_gm)
    with graph.inserting_before(site.log_softmax):
        body_attr = graph.get_attr(body_name)
        fused = graph.call_function(
            flex_gemm_hop,
            (
                torch.ops.aten.mm.default,
                body_attr,
                (a_node, b_node, safe_targets),
                {},
                dict(QUACK_CROSS_ENTROPY_KERNEL_OPTIONS),
            ),
        )
        logits = graph.call_function(operator.getitem, (fused, 0))
        target_logits = graph.call_function(operator.getitem, (fused, 1))
        partial_lse = graph.call_function(operator.getitem, (fused, 2))

    _inherit_meta(
        fused, site.logits_mm, (logits_val, target_logits_val, partial_lse_val)
    )
    _inherit_meta(logits, site.logits_mm, logits_val)
    _inherit_meta(target_logits, site.loss, target_logits_val)
    _inherit_meta(partial_lse, site.log_softmax, partial_lse_val)

    loss_gm = _trace_cross_entropy_loss(target_logits_val, partial_lse_val, valid_val)
    loss_sum, lse = _inline_graph(
        graph,
        loss_gm,
        (target_logits, partial_lse, valid),
        site.log_softmax,
    )
    _, lse_val = _fake_outputs(loss_gm)
    _inherit_meta(loss_sum, site.loss, loss_sum.meta["val"])
    site.loss.replace_all_uses_with(loss_sum)

    backward_gm = _trace_cross_entropy_backward(
        logits_val, lse_val, safe_targets_val, valid_val, scale_val
    )
    grad = _inline_graph(
        graph,
        backward_gm,
        (logits, lse, safe_targets, valid, site.grad_scale),
        site.grad_to_bf16,
    )
    _inherit_meta(grad, site.grad_to_bf16, grad.meta["val"])
    site.grad_to_bf16.replace_all_uses_with(grad)


def flex_gemm_cross_entropy_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Fuse chunked LM-head CE forward/backward into tuned FlexGEMM sites."""
    sites = [
        site
        for node in gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten._log_softmax.default
        )
        if (site := match_cross_entropy_site(node)) is not None
    ]
    if not sites:
        logger.warning(
            "flex_gemm_cross_entropy_pass was enabled but found no eligible "
            "LM-head CE site"
        )
        return gm

    install_flex_gemm_codegen_shim()
    for index, site in enumerate(sites):
        _fuse_cross_entropy_site(gm, site, f"flex_gemm_cross_entropy_body_{index}")
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    logger.info(
        f"flex_gemm_cross_entropy_pass: fused {len(sites)} chunked LM-head CE "
        f"forward/backward sites with group={_CROSS_ENTROPY_GROUP} and "
        f"kernel_options={QUACK_CROSS_ENTROPY_KERNEL_OPTIONS}"
    )
    return gm


def flex_gemm_residual_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Fuse Qwen3 attention/FFN output GEMMs with their residual adds."""
    node_order = {node: index for index, node in enumerate(gm.graph.nodes)}
    sites = [
        site
        for node in gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.add.Tensor
        )
        if (site := match_residual_site(node, node_order)) is not None
    ]
    if not sites:
        logger.info("flex_gemm_residual_pass: no eligible Qwen3 residual site")
        return gm

    install_flex_gemm_codegen_shim()
    # Later residuals can capture earlier residual outputs. Rewrite downstream
    # sites first so replacing an earlier add updates already-inserted captures.
    for index, site in enumerate(reversed(sites)):
        _fuse_residual_site(gm, site, f"flex_gemm_residual_body_{index}")
    gm.graph.lint()
    gm.recompile()
    logger.info(
        f"flex_gemm_residual_pass: fused {len(sites)} Qwen3 output GEMMs "
        f"with residual adds using kernel_options={QUACK_KERNEL_OPTIONS}"
    )
    return gm


def packed_w13_wgrad_layout_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Orient packed W13 wgrad GEMMs to return parameter-layout gradients."""
    sites = [
        site
        for node in gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten._to_copy.default
        )
        if (site := match_packed_w13_wgrad_layout_site(node)) is not None
    ]
    if not sites:
        logger.warning(
            "packed_w13_wgrad_layout_pass was enabled but found no eligible "
            "packed W13 weight-gradient site"
        )
        return gm

    for site in sites:
        _rewrite_packed_w13_wgrad_layout(gm, site)
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    logger.info(
        f"packed_w13_wgrad_layout_pass: rewrote {len(sites)} packed W13 "
        "weight-gradient sites to parameter layout"
    )
    return gm


def flex_gemm_swiglu_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Fuse eligible dense SwiGLU forward sites into QUACK ``flex_gemm`` calls.

    Must run before the terminal Inductor pass, which owns the graph once it
    collapses into a compiled artifact. Sites that do not match either exact
    feed-forward layout contract are left untouched.
    """
    node_order = {node: index for index, node in enumerate(gm.graph.nodes)}
    packed_sites_by_bmm: dict[torch.fx.Node, PackedSwiGluSite] = {}
    for node in gm.graph.nodes:
        site = match_packed_swiglu_site(node)
        if site is not None:
            # Fuse only the first activation consumer. Later consumers are
            # activation-remat recomputations that should stay short-lived.
            packed_sites_by_bmm.setdefault(site.packed_bmm, site)
    packed_sites = list(packed_sites_by_bmm.values())
    sites = [
        site
        for node in gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mul.Tensor
        )
        if (site := match_swiglu_site(node, node_order)) is not None
    ]
    if not packed_sites and not sites:
        logger.info("flex_gemm_swiglu_pass: no eligible dense SwiGLU site")
        return gm

    install_flex_gemm_codegen_shim()
    for index, site in enumerate(packed_sites):
        _fuse_packed_swiglu_site(
            gm,
            site,
            f"flex_gemm_packed_swiglu_body_{index}",
        )
    for index, site in enumerate(sites):
        _fuse_site(gm, site, f"flex_gemm_swiglu_body_{index}")
    gm.graph.lint()
    gm.recompile()
    logger.info(
        "flex_gemm_swiglu_pass: fused "
        f"{len(packed_sites)} packed and {len(sites)} separate dense SwiGLU "
        "sites with tuned QUACK kernels"
    )
    return gm
