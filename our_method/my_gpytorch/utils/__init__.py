#!/usr/bin/env python3

from __future__ import annotations

import warnings as _warnings
from typing import Any

import linear_operator

from . import deprecation, errors, generic, grid, interpolation, quadrature, transforms, warnings
from .memoize import cached
from .nearest_neighbors import NNUtil
#NP ADDING
from .utils import otdd_dist, otdd_dist_1hot, slice_wasserstein, possible_coalitions, from_coalition_to_list, from_list_to_coalitions, My_Single_Dataset, mse, msll

__all__ = [
    "cached",
    "deprecation",
    "errors",
    "generic",
    "grid",
    "interpolation",
    "otdd_dist", #NP adding
    "otdd_dist_1hot",
    "possible_coalitions",
    "from_coalition_to_list",
    "from_list_to_coalitions",
    "quadrature",
    "transforms",
    "warnings",
    "NNUtil",
    "My_Single_Dataset",
    "mse",
    "msll",
]


def __getattr__(name: str) -> Any:
    if hasattr(linear_operator.utils, name):
        _warnings.warn(
            f"gpytorch.utils.{name} is deprecated. Use linear_operator.utils.{name} instead.",
            DeprecationWarning,
        )
        return getattr(linear_operator.utils, name)
    raise AttributeError(f"module gpytorch.utils has no attribute {name}")
