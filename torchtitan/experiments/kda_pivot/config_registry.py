# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training configs for the KDA gate-reference A/B.

Both arms of the A/B run the same config function; the only difference is the
``ATTN_GYM_KDA_GATE_REFERENCE`` environment variable read by Attention Gym. Data
order is fixed by the random-access local C4 shards plus the dataloader seed, and
``--debug.seed``/``--debug.deterministic`` pin initialization and kernels.
"""

import os
from dataclasses import replace
from pathlib import Path

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.text_datasets import DATASETS, TextProcessor
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.kimi_k3 import (
    _feed_forward_config,
    _kimi_k3_config,
    KimiK3StateDictAdapter,
    parallelize_kimi_k3,
)
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

# Directory holding the downloaded ``allenai/c4`` ``en`` json.gz shards. The
# default matches the experiment checkout layout (``<root>/data/c4/en``); the
# MAST launcher points it at the staged copy.
DATA_DIR_ENV = "KDA_PIVOT_DATA_DIR"
# Tokenizer directory (``tokenizer.json`` + config). Qwen3's tokenizer is used
# because Kimi's ships as tiktoken files torchtitan cannot load.
HF_ASSETS_ENV = "KDA_PIVOT_HF_ASSETS"
# Padded Qwen3 vocabulary (151669 real tokens).
QWEN3_VOCAB_SIZE = 151936

# Six shards is roughly 1B Qwen3 tokens: enough for the 4000-step, 262K-token/step
# pilot without repeating data.
TRAIN_SHARDS = tuple(f"c4-train.{i:05d}-of-01024.json.gz" for i in range(6))
VALIDATION_SHARD = "c4-validation.00000-of-00008.json.gz"


def _experiment_root() -> Path:
    """Parent of the torchtitan checkout: holds ``attention-gym``, ``data`` and ``assets``."""
    return Path(__file__).resolve().parents[4]


def _data_dir() -> Path:
    return Path(os.environ.get(DATA_DIR_ENV, _experiment_root() / "data" / "c4" / "en"))


def _hf_assets_path() -> str:
    return os.environ.get(
        HF_ASSETS_ENV, str(_experiment_root() / "assets" / "hf" / "Qwen3-0.6B")
    )


def _local_c4(shards: tuple[str, ...]) -> SingleDatasetConfig:
    """Random-access C4 from fixed local shards, so both arms see identical data order."""
    data_dir = _data_dir()
    data_files = [str(data_dir / shard) for shard in shards]
    missing = [path for path in data_files if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            f"KDA pivot C4 shards missing under {data_dir} (set {DATA_DIR_ENV}): {missing}"
        )
    return SingleDatasetConfig(
        source=HuggingFaceRandomAccessSource.Config(
            path="json",
            split="train",
            load_dataset_kwargs={"data_files": data_files},
        ),
        processor=TextProcessor.Config(),
        post_filters=(lambda sample: sample is not None,),
    )


def text_only_kimi_k3(
    *,
    dim: int,
    num_layers: int,
    full_attention_layers: set[int],
    num_heads: int,
    vocab_size: int,
    dense_hidden_dim: int,
) -> KimiK3Model.Config:
    """Kimi K3 decoder topology with dense FFNs and no vision encoder.

    Keeps the KDA layer exactly as the production model builds it (head_dim 128,
    conv width 4, Attention Gym kernel) so the A/B exercises the real training path.
    MoE and vision are dropped to remove routing noise and unrelated cost.
    """
    config = _kimi_k3_config(
        dim=dim,
        vocab_size=vocab_size,
        num_layers=num_layers,
        full_attention_layers=full_attention_layers,
        attn_res_block_size=num_layers // 2,
        num_heads=num_heads,
        q_lora_rank=dim // 2,
        kv_lora_rank=dim // 4,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        kda_head_dim=128,
        conv_kernel_size=4,
        dense_hidden_dim=dense_hidden_dim,
        latent_dim=dim // 2,
        expert_hidden_dim=dense_hidden_dim,
        num_experts=2,
        top_k=1,
        num_shared_experts=1,
        vision_encoder=None,  # pyrefly: ignore [bad-argument-type]
        attn_backend="flex",
    )
    dense_layers = [
        replace(
            layer,
            moe=None,
            feed_forward=_feed_forward_config(dim=dim, hidden_dim=dense_hidden_dim),
        )
        for layer in config.layers
    ]
    return replace(config, layers=dense_layers)


def _model_spec(model: KimiK3Model.Config, *, max_context_length: int) -> ModelSpec:
    return ModelSpec(
        name="kimi_k3",
        flavor="kda_pivot",
        model=model,
        max_context_length=max_context_length,
        parallelize_fn=parallelize_kimi_k3,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=KimiK3StateDictAdapter,
    )


def _trainer_config(
    model_spec: ModelSpec,
    *,
    hf_assets_path: str,
    train_dataset: SingleDatasetConfig,
    validation_dataset: SingleDatasetConfig,
    tokens_per_microbatch: int,
    steps: int,
    warmup_steps: int,
    checkpoint_interval: int,
    validation_freq: int,
    validation_steps: int,
) -> Trainer.Config:
    seq_len = model_spec.max_context_length
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path=hf_assets_path,
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=train_dataset),
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=warmup_steps,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=tokens_per_microbatch,
            max_context_length=seq_len,
            steps=steps,
            dtype="bfloat16",
            disable_cuda_graphs=True,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=checkpoint_interval,
            last_save_model_only=False,
            keep_latest_k=0,
        ),
        activation_checkpoint=SelectiveAC.Config(),
        validator=Validator.Config(
            enable=True,
            freq=validation_freq,
            steps=validation_steps,
            dataloader=GrainDataLoader.Config(
                dataset=ConcatThenSplitPackingConfig(dataset=validation_dataset),
                repeat=False,
            ),
        ),
    )


def kda_pivot_debugmodel() -> Trainer.Config:
    """Tiny text-only config for local plumbing checks (test tokenizer, c4_test)."""
    model = text_only_kimi_k3(
        dim=256,
        num_layers=4,
        full_attention_layers={3},
        num_heads=2,
        vocab_size=2048,
        dense_hidden_dim=512,
    )
    return _trainer_config(
        _model_spec(model, max_context_length=256),
        hf_assets_path="./tests/assets/tokenizer",
        train_dataset=DATASETS["c4_test"],
        validation_dataset=DATASETS["c4_test"],
        tokens_per_microbatch=512,
        steps=10,
        warmup_steps=2,
        checkpoint_interval=10,
        validation_freq=5,
        validation_steps=2,
    )


def kda_pivot_pilot() -> Trainer.Config:
    """Text-only pilot: ~12-layer Kimi K3 topology (9 KDA + 3 MLA layers) on local C4."""
    model = text_only_kimi_k3(
        dim=1024,
        num_layers=12,
        full_attention_layers={3, 7, 11},
        num_heads=8,
        vocab_size=QWEN3_VOCAB_SIZE,
        dense_hidden_dim=4096,
    )
    return _trainer_config(
        _model_spec(model, max_context_length=1024),
        hf_assets_path=_hf_assets_path(),
        train_dataset=_local_c4(TRAIN_SHARDS),
        validation_dataset=_local_c4((VALIDATION_SHARD,)),
        tokens_per_microbatch=32 * 1024,
        steps=2000,
        warmup_steps=100,
        checkpoint_interval=500,
        validation_freq=250,
        validation_steps=50,
    )
