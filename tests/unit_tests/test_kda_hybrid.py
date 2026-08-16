# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Focused coverage for the optional attention-gym KDA experiment."""

import pytest
import torch

pytest.importorskip("attn_gym")
pytest.importorskip("examples.kda_training")

from examples import kda_training  # noqa: E402
from torchtitan.experiments.kda_hybrid import (  # noqa: E402
    kda_attention_config,
    model_registry,
)
from torchtitan.experiments.kda_hybrid.config_registry import (  # noqa: E402
    kda_hybrid_debugmodel,
)
from torchtitan.experiments.kda_hybrid.model import (  # noqa: E402
    capacity_aware_global_attention,
    create_capacity_cu_seqlens,
    KDAHybridTransformerBlock,
)
from torchtitan.experiments.kda_hybrid.trainer import KDAHybridTrainer  # noqa: E402
from torchtitan.models.common.attention import VarlenMetadata  # noqa: E402


def test_hybrid_config_uses_three_kda_layers_per_global_layer() -> None:
    """Keep the intended 3:1 layer pattern and TorchTitan module contract."""
    model_config = model_registry("debugmodel").model
    assert len(model_config.layers) == 8
    assert all(
        isinstance(layer, KDAHybridTransformerBlock.Config)
        for layer in model_config.layers
    )
    assert [layer.attention is not None for layer in model_config.layers] == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]

    with torch.device("meta"):
        model = model_config.build()
    model.verify_module_protocol()


def test_capacity_offsets_have_fixed_shape_and_repeated_endpoint() -> None:
    """Pad M document starts to an N-sequence graph input with repeated L."""
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 0, 1, 2]])
    offsets = create_capacity_cu_seqlens(
        positions,
        active_tokens=7,
        max_sequences=4,
    )
    torch.testing.assert_close(
        offsets,
        torch.tensor([0, 3, 7, 7, 7], dtype=torch.int32),
    )

    with pytest.raises(ValueError, match="exceeding max_sequences"):
        create_capacity_cu_seqlens(
            positions,
            active_tokens=10,
            max_sequences=2,
        )


def test_debug_recipe_enables_capacity_cuda_graph_replay() -> None:
    """Keep the integration recipe on its graph-replay stress path."""
    config = kda_hybrid_debugmodel()
    assert isinstance(config, KDAHybridTrainer.Config)
    assert not config.training.disable_cuda_graphs
    assert config.training.seq_len == 256
    assert config.training.mixed_precision_param == "bfloat16"
    assert config.max_sequences == 32
    assert config.min_active_tokens == 96

    trainer = object.__new__(KDAHybridTrainer)
    trainer.config = config
    assert [trainer._active_tokens_for_batch(i, 256) for i in range(6)] == [
        256,
        192,
        96,
        256,
        192,
        96,
    ]


def test_hybrid_varlen_layers_share_document_boundaries() -> None:
    """Route one offsets object to both global attention and KDA."""
    with torch.device("meta"):
        model = model_registry("debugmodel").model.build()
    positions = torch.tensor(
        [
            [0, 1, 2, 0, 1],
            [0, 1, 2, 3, 4],
        ],
        dtype=torch.int32,
    )
    masks = model.get_attention_masks(positions)

    assert set(masks) == {"global_attention", "kda"}
    assert isinstance(masks["kda"], VarlenMetadata)
    assert masks["global_attention"] is masks["kda"]
    torch.testing.assert_close(
        masks["kda"].cu_seq_q,
        torch.tensor([0, 3, 5, 10], dtype=torch.int32),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="attention-gym fused KDA requires a Blackwell GPU",
)
def test_kda_wrapper_varlen_forward_and_backward(monkeypatch) -> None:
    """Use every fused stage, match documents, and propagate finite gradients."""
    calls = {
        "short_conv": 0,
        "l2norm": 0,
        "gate": 0,
        "core": 0,
    }

    def record(name, operation):
        def wrapped(*args, **kwargs):
            calls[name] += 1
            return operation(*args, **kwargs)

        return wrapped

    for name, attribute in (
        ("short_conv", "cute_causal_conv1d_silu"),
        ("l2norm", "l2norm"),
        ("gate", "_bounded_gate_cumsum"),
        ("core", "_chunk_kda"),
    ):
        monkeypatch.setattr(
            kda_training,
            attribute,
            record(name, getattr(kda_training, attribute)),
        )

    torch.manual_seed(42)
    with torch.device("cuda"):
        model = kda_attention_config(
            hidden_size=128,
            num_heads=1,
            num_layers=4,
        ).build()
    with torch.no_grad():
        model.init_states()
    assert model.backend == "fused"

    positions = (
        torch.cat(
            (
                torch.cat((torch.arange(17), torch.arange(47))),
                torch.cat((torch.arange(31), torch.arange(33))),
            )
        )
        .reshape(2, 64)
        .to(device="cuda", dtype=torch.int32)
    )
    capacity_offsets = create_capacity_cu_seqlens(
        positions,
        active_tokens=positions.numel(),
        max_sequences=8,
    ).to("cuda")
    metadata = VarlenMetadata(
        capacity_offsets,
        capacity_offsets,
        positions.numel(),
        positions.numel(),
    )
    hidden = torch.randn(
        2,
        64,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    actual = model(hidden, metadata)
    assert calls == {
        "short_conv": 1,
        "l2norm": 2,
        "gate": 1,
        "core": 1,
    }

    expected = torch.empty_like(actual)
    document_ranges = ((0, 0, 17), (0, 17, 64), (1, 0, 31), (1, 31, 64))
    for batch_idx, start, end in document_ranges:
        expected[batch_idx : batch_idx + 1, start:end] = model(
            hidden[batch_idx : batch_idx + 1, start:end]
        )
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)

    actual.float().square().mean().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="capacity-aware varlen attention requires a Blackwell GPU",
)
def test_global_attention_masks_inactive_values_and_gradients() -> None:
    """Match an exact run despite stale values and cotangents beyond active L."""
    torch.manual_seed(43)
    attention_config = model_registry("debugmodel").model.layers[3].attention
    assert attention_config is not None
    with torch.device("cuda"):
        attention = attention_config.build()
    with torch.no_grad():
        attention.init_states()

    capacity, active_tokens, dim = 128, 96, 256
    positions = torch.arange(capacity, device="cuda").reshape(1, -1)
    capacity_offsets = create_capacity_cu_seqlens(
        positions,
        active_tokens=active_tokens,
        max_sequences=4,
    )
    capacity_metadata = VarlenMetadata(
        capacity_offsets,
        capacity_offsets,
        capacity,
        capacity,
    )
    exact_offsets = torch.tensor(
        [0, active_tokens],
        dtype=torch.int32,
        device="cuda",
    )
    exact_metadata = VarlenMetadata(
        exact_offsets,
        exact_offsets,
        active_tokens,
        active_tokens,
    )
    active_mask = torch.arange(capacity, device="cuda") < active_tokens

    hidden = torch.randn(1, capacity, dim, device="cuda")
    hidden[:, active_tokens:] = float("nan")
    hidden.requires_grad_()
    exact_hidden = hidden[:, :active_tokens].detach().clone().requires_grad_()
    actual = capacity_aware_global_attention(
        attention,
        hidden,
        capacity_metadata,
        positions,
        active_mask,
    )
    expected = attention(
        exact_hidden,
        exact_metadata,
        positions[:, :active_tokens],
    )
    torch.testing.assert_close(
        actual[:, :active_tokens],
        expected,
        rtol=3e-2,
        atol=3e-2,
    )
    assert not actual[:, active_tokens:].any()

    cotangent = torch.randn_like(actual)
    cotangent[:, active_tokens:] = float("nan")
    names, parameters = zip(*attention.named_parameters(), strict=True)
    actual_gradients = torch.autograd.grad(
        actual,
        (hidden, *parameters),
        cotangent,
    )
    expected_gradients = torch.autograd.grad(
        expected,
        (exact_hidden, *parameters),
        cotangent[:, :active_tokens],
    )
    torch.testing.assert_close(
        actual_gradients[0][:, :active_tokens],
        expected_gradients[0],
        rtol=3e-2,
        atol=3e-2,
    )
    assert not actual_gradients[0][:, active_tokens:].any()
    for name, actual_gradient, expected_gradient in zip(
        names,
        actual_gradients[1:],
        expected_gradients[1:],
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=3e-2,
            atol=3e-2,
            msg=lambda message, parameter=name: f"{parameter}: {message}",
        )
