# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "apply_fsdp_to_wav2vec2_llama",
    "Wav2Vec2LlamaBeamSearchSeq2SeqGenerator",
    "WAV2VEC2_LLAMA_FAMILY",
    "Wav2Vec2LlamaBeamSearchConfig",
    "Wav2Vec2LlamaConfig",
    "register_wav2vec2_llama_configs",
    "Wav2Vec2LlamaFactory",
    "create_wav2vec2_llama_model",
    "get_wav2vec2_llama_model_hub",
    "convert_wav2vec2_llama_state_dict",
    "Wav2Vec2LlamaModel",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple dispatcher
    module = importlib.import_module("omnilingual_asr.models.wav2vec2_llama.module_map")
    if hasattr(module, name):
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
