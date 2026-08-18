from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intent_geometry import geometry_distortion  # noqa: E402


class IntentGeometryStudyTests(unittest.TestCase):
    def test_identical_geometry_has_unit_correlation_and_zero_error(self):
        vectors = np.random.default_rng(7).normal(size=(6, 4)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        metrics = geometry_distortion(vectors, vectors.copy())
        self.assertAlmostEqual(metrics["pairwise_cosine_correlation"], 1.0)
        self.assertAlmostEqual(metrics["mean_absolute_cosine_error"], 0.0)
        self.assertAlmostEqual(metrics["max_absolute_cosine_error"], 0.0)

    def test_rejects_non_normalized_rows(self):
        reference = np.ones((3, 2), dtype=np.float32)
        projected = np.eye(3, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "unit normalized"):
            geometry_distortion(reference, projected)

    def test_rejects_mismatched_row_counts(self):
        with self.assertRaisesRegex(ValueError, "same row count"):
            geometry_distortion(np.eye(3), np.eye(4))


if __name__ == "__main__":
    unittest.main()
