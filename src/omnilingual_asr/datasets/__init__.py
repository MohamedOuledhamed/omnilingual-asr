# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "ManifestAsrDataset",
    "ManifestAsrDatasetConfig",
    "MixtureParquetAsrDataset",
    "MixtureParquetAsrDatasetConfig",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple dispatch
    if name in {"ManifestAsrDataset", "ManifestAsrDatasetConfig"}:
        module = importlib.import_module(
            "omnilingual_asr.datasets.impl.manifest_asr_dataset"
        )
        return getattr(module, name)

    if name in {"MixtureParquetAsrDataset", "MixtureParquetAsrDatasetConfig"}:
        module = importlib.import_module(
            "omnilingual_asr.datasets.impl.mixture_parquet_asr_dataset"
        )
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
