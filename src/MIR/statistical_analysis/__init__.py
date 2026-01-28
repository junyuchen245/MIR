"""Statistical analysis utilities for MIR."""

from .GLM import GLM, winsorize, janmahasatian_lbm, robust_bp_ref

__all__ = [
    'GLM',
    'winsorize',
    'janmahasatian_lbm',
    'robust_bp_ref',
]