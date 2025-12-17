# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

__all__ = ["inference"]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple dispatch
    if name == "inference":
        return importlib.import_module("omnilingual_asr.models.inference")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Allow attribute style access for static type checkers
inference: ModuleType
