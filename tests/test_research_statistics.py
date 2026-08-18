from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_statistics import (  # noqa: E402
    bootstrap_interval,
    holm_adjust,
    interquartile_mean,
    paired_difference_summary,
    paired_randomization_test,
    performance_profile,
    summarize_sample,
)


class ResearchStatisticsTests(unittest.TestCase):
    def test_interquartile_mean_reduces_outlier_influence(self):
        values = [0.0, 1.0, 2.0, 3.0, 100.0]
        self.assertLess(interquartile_mean(values), np.mean(values))

    def test_bootstrap_is_reproducible(self):
        first = bootstrap_interval([1, 2, 3, 4, 5], n_resamples=500, seed=11)
        second = bootstrap_interval([1, 2, 3, 4, 5], n_resamples=500, seed=11)
        self.assertEqual(first, second)

    def test_singleton_bootstrap_is_exact(self):
        interval = bootstrap_interval([3.25], n_resamples=500, seed=17)
        self.assertEqual(interval["low"], 3.25)
        self.assertEqual(interval["high"], 3.25)

    def test_vectorized_iqm_bootstrap_is_reproducible(self):
        first = bootstrap_interval(
            [0, 1, 2, 3, 100], statistic="iqm", n_resamples=500, seed=19
        )
        second = bootstrap_interval(
            [0, 1, 2, 3, 100], statistic="iqm", n_resamples=500, seed=19
        )
        self.assertEqual(first, second)

    def test_summary_keeps_raw_values(self):
        summary = summarize_sample([0.1, 0.2, 0.3], n_resamples=500, seed=5)
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["raw"], [0.1, 0.2, 0.3])
        self.assertIn("mean_ci", summary)

    def test_paired_direction_for_lower_is_better(self):
        result = paired_difference_summary(
            treatment=[0.1, 0.2, 0.3],
            baseline=[0.2, 0.3, 0.4],
            lower_is_better=True,
            n_resamples=500,
            seed=13,
        )
        self.assertAlmostEqual(result["win_rate"], 1.0)
        self.assertLess(result["mean_difference"], 0.0)

    def test_exact_paired_randomization_detects_consistent_shift(self):
        result = paired_randomization_test(
            treatment=[0.0] * 8,
            baseline=[1.0] * 8,
        )
        self.assertEqual(result["method"], "exact_paired_sign_flip")
        self.assertAlmostEqual(result["p_value"], 2.0 / 256.0)

    def test_holm_adjustment_is_monotone_in_rank_order(self):
        result = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
        adjusted = result["adjusted_p_values"]
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertAlmostEqual(adjusted["c"], 0.20)

    def test_paired_summary_drops_nonfinite_values_pairwise(self):
        result = paired_difference_summary(
            treatment=[1.0, np.nan, 3.0],
            baseline=[0.0, 2.0, np.nan],
            lower_is_better=False,
            n_resamples=500,
        )
        self.assertEqual(result["raw_differences"], [1.0])

    def test_performance_profile_validates_equal_lengths(self):
        with self.assertRaises(ValueError):
            performance_profile({"a": [1, 2], "b": [1]})


if __name__ == "__main__":
    unittest.main()
