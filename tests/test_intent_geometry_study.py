from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_intent_geometry_study import geometry_distortion  # noqa: E402


class IntentGeometryStudyTests(unittest.TestCase):
    def test_identical_geometry_has_unit_correlation_and_zero_error(self):
        vectors = np.random.default_rng(7).normal(size=(6, 4)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        metrics = geometry_distortion(vectors, vectors.copy())
        self.assertAlmostEqual(metrics["pairwise_cosine_correlation"], 1.0)
        self.assertAlmostEqual(metrics["mean_absolute_cosine_error"], 0.0)
        self.assertAlmostEqual(metrics["max_absolute_cosine_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
