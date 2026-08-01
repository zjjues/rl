from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imappo import detect_switch_response_latency, env_reset  # noqa: E402


class _SeedRecordingEnv:
    def __init__(self):
        self.seeds = []

    def reset(self, seed=None):
        self.seeds.append(seed)
        return ("obs", {"seed": seed})


class EvaluationSeedingTests(unittest.TestCase):
    def test_env_reset_forwards_explicit_seed(self):
        env = _SeedRecordingEnv()
        obs, info = env_reset(env, seed=7000123)
        self.assertEqual(obs, "obs")
        self.assertEqual(info["seed"], 7000123)
        self.assertEqual(env.seeds, [7000123])

    def test_dynamic_response_latency_is_first_material_counterfactual_change(self):
        self.assertEqual(
            detect_switch_response_latency([0.01, 0.049, 0.051, 0.2], 0.05),
            2,
        )
        self.assertIsNone(detect_switch_response_latency([0.01, 0.02], 0.05))
        with self.assertRaises(ValueError):
            detect_switch_response_latency([0.1], 0.0)


if __name__ == "__main__":
    unittest.main()
