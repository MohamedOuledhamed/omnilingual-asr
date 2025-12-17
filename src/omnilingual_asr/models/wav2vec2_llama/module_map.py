from __future__ import annotations

import importlib
from typing import Any

__all__ = [
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


def _load_module(module_name: str) -> Any:
    return importlib.import_module(f"omnilingual_asr.models.wav2vec2_llama.{module_name}")


def __getattr__(name: str) -> Any:  # pragma: no cover - simple dispatch
    if name in {
        "Wav2Vec2LlamaBeamSearchSeq2SeqGenerator",
    }:
        module = _load_module("beamsearch")
        return getattr(module, name)

    if name in {
        "WAV2VEC2_LLAMA_FAMILY",
        "Wav2Vec2LlamaBeamSearchConfig",
        "Wav2Vec2LlamaConfig",
        "register_wav2vec2_llama_configs",
    }:
        module = _load_module("config")
        return getattr(module, name)

    if name in {"Wav2Vec2LlamaFactory", "create_wav2vec2_llama_model"}:
        module = _load_module("factory")
        return getattr(module, name)

    if name in {"get_wav2vec2_llama_model_hub"}:
        module = _load_module("hub")
        return getattr(module, name)

    if name in {"convert_wav2vec2_llama_state_dict"}:
        module = _load_module("interop")
        return getattr(module, name)

    if name in {"Wav2Vec2LlamaModel"}:
        module = _load_module("model")
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
