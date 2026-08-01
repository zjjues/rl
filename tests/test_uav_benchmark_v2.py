from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from envs.uav_scheduling_env import (  # noqa: E402
    UAVSchedulingEnv,
    infer_obs_dim,
    infer_obs_dim_v2,
)


class UAVBenchmarkV2Tests(unittest.TestCase):
    def test_v2_adds_nearest_threat_vector_without_changing_v1_shape(self):
        legacy = UAVSchedulingEnv(n_agents=4, benchmark_version="v1")
        revised = UAVSchedulingEnv(n_agents=4, benchmark_version="v2")
        legacy_obs, _ = legacy.reset(seed=17)
        revised_obs, _ = revised.reset(seed=17)
        self.assertEqual(legacy_obs[0].shape, (infer_obs_dim(4),))
        self.assertEqual(revised_obs[0].shape, (infer_obs_dim_v2(4),))
        expected = revised.threat_zone_centers[
            np.argmin(np.linalg.norm(
                revised.threat_zone_centers - revised.positions[0], axis=1
            ))
        ] - revised.positions[0]
        np.testing.assert_allclose(revised_obs[0][12:15], expected, atol=1e-6)

    def test_v2_task_progress_rewards_target_aligned_control(self):
        idle = UAVSchedulingEnv(n_agents=2, n_targets=2, benchmark_version="v2")
        aligned = UAVSchedulingEnv(n_agents=2, n_targets=2, benchmark_version="v2")
        idle.reset(seed=23)
        aligned.reset(seed=23)
        target_delta = aligned.targets - aligned.positions
        actions = target_delta / np.maximum(
            np.linalg.norm(target_delta, axis=1, keepdims=True), 1e-6
        )
        idle.step(np.zeros((2, 3), dtype=np.float32))
        _, _, _, _, info = aligned.step(actions.astype(np.float32))
        aligned_completion = np.mean([
            value["task_completion"] for value in info.values()
        ])
        idle_completion = float(1.0 - idle.pending_tasks.mean())
        self.assertGreater(aligned_completion, idle_completion)
        self.assertTrue(all("distance_to_threat" in value for value in info.values()))

    def test_invalid_benchmark_version_is_rejected(self):
        with self.assertRaises(ValueError):
            UAVSchedulingEnv(benchmark_version="future")


if __name__ == "__main__":
    unittest.main()
