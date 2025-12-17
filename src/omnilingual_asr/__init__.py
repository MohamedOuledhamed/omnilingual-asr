# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

__version__ = "0.1.0"


def setup_fairseq2_extension(container) -> None:
    """Register omnilingual ASR assets and models with fairseq2."""
    from fairseq2.composition.assets import register_package_assets
    from fairseq2.composition.models import register_model_family
    from fairseq2.runtime.dependency import DependencyContainer

    from omnilingual_asr.models.wav2vec2_asr.config import (
        register_omnilingual_asr_wav2vec2_asr_configs,
    )
    from omnilingual_asr.models.wav2vec2_llama import (
        WAV2VEC2_LLAMA_FAMILY,
        Wav2Vec2LlamaConfig,
        Wav2Vec2LlamaModel,
        convert_wav2vec2_llama_state_dict,
        create_wav2vec2_llama_model,
        register_wav2vec2_llama_configs,
    )
    from omnilingual_asr.models.wav2vec2_ssl.config import (
        register_omnilingual_asr_wav2vec2_ssl_configs,
    )

    if not isinstance(container, DependencyContainer):
        raise TypeError("container must be a DependencyContainer")

    # Make sure that the default fairseq2 asset store can resolve cards under
    # the directory <omnilingual_asr>/cards.
    register_package_assets(container, "omnilingual_asr.cards")

    _register_models(
        container,
        register_model_family,
        WAV2VEC2_LLAMA_FAMILY,
        Wav2Vec2LlamaModel,
        Wav2Vec2LlamaConfig,
        create_wav2vec2_llama_model,
        convert_wav2vec2_llama_state_dict,
        register_wav2vec2_llama_configs,
        register_omnilingual_asr_wav2vec2_ssl_configs,
        register_omnilingual_asr_wav2vec2_asr_configs,
    )


def _register_models(
    container,
    register_model_family,
    wav2vec2_llama_family,
    wav2vec2_llama_model,
    wav2vec2_llama_config,
    create_wav2vec2_llama_model,
    convert_wav2vec2_llama_state_dict,
    register_wav2vec2_llama_configs,
    register_omnilingual_asr_wav2vec2_ssl_configs,
    register_omnilingual_asr_wav2vec2_asr_configs,
) -> None:
    register_omnilingual_asr_wav2vec2_ssl_configs(container)
    register_omnilingual_asr_wav2vec2_asr_configs(container)

    register_model_family(
        container,
        wav2vec2_llama_family,
        kls=wav2vec2_llama_model,
        config_kls=wav2vec2_llama_config,
        factory=create_wav2vec2_llama_model,
        fsdp_applier=apply_fsdp_to_wav2vec2_llama,
        state_dict_converter=convert_wav2vec2_llama_state_dict,
    )

    register_wav2vec2_llama_configs(container)
