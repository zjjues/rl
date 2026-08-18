"""Geometry diagnostics for intent-representation studies.

The original experiment runner exposed this calculation as a top-level helper.
Keeping it in ``src`` makes the diagnostic reusable after old experiment runners
and generated artifacts are archived.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def geometry_distortion(reference: np.ndarray, projected: np.ndarray) -> Dict[str, float]:
    """Compare pairwise cosine geometry before and after projection.

    Both inputs must contain the same number of row-normalized representations.
    The function deliberately validates normalization because a dot product is a
    cosine similarity only under that invariant.
    """

    reference = np.asarray(reference, dtype=np.float64)
    projected = np.asarray(projected, dtype=np.float64)
    if reference.ndim != 2 or projected.ndim != 2:
        raise ValueError("reference and projected must be two-dimensional")
    if reference.shape[0] != projected.shape[0]:
        raise ValueError("reference and projected must have the same row count")
    if reference.shape[0] < 3:
        raise ValueError("at least three representations are required")
    if reference.shape[1] == 0 or projected.shape[1] == 0:
        raise ValueError("representation dimensions must be non-zero")
    if not np.isfinite(reference).all() or not np.isfinite(projected).all():
        raise ValueError("representations must be finite")

    reference_norms = np.linalg.norm(reference, axis=1)
    projected_norms = np.linalg.norm(projected, axis=1)
    if not np.allclose(reference_norms, 1.0, atol=1e-5):
        raise ValueError("reference rows must be unit normalized")
    if not np.allclose(projected_norms, 1.0, atol=1e-5):
        raise ValueError("projected rows must be unit normalized")

    upper = np.triu_indices(reference.shape[0], k=1)
    reference_values = (reference @ reference.T)[upper]
    projected_values = (projected @ projected.T)[upper]
    reference_scale = float(np.std(reference_values))
    projected_scale = float(np.std(projected_values))
    if reference_scale <= 1e-12 or projected_scale <= 1e-12:
        correlation = 1.0 if np.allclose(reference_values, projected_values) else 0.0
    else:
        correlation = float(np.corrcoef(reference_values, projected_values)[0, 1])
    errors = np.abs(reference_values - projected_values)
    return {
        "pairwise_cosine_correlation": correlation,
        "mean_absolute_cosine_error": float(np.mean(errors)),
        "max_absolute_cosine_error": float(np.max(errors)),
    }
