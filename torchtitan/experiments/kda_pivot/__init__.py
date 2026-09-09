# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KDA gate-reference (pivot) A/B experiment.

Trains a text-only Kimi K3 topology twice on identical data, differing only in the
Attention Gym intra-chunk gate reference row (``ATTN_GYM_KDA_GATE_REFERENCE`` set to
``causal`` or ``midpoint``), then compares parallel (teacher-forced full sequence)
NLL against prefix-only NLL to detect training that exploits future-gate rounding.
See ``README.md`` in this directory.
"""
