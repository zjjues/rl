from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from envs.uav_scheduling_env import UAVSchedulingEnv  # noqa: E402


class UAVDisturbanceTests(unittest.TestCase):
    def test_explicit_zero_disturbances_preserve_default_trajectory(self):
        default = UAVSchedulingEnv(n_agents=4, n_targets=4)
        explicit = UAVSchedulingEnv(
            n_agents=4,
            n_targets=4,
            wind_std=0.0,
            observation_noise_std=0.0,
            action_delay_steps=0,
            communication_dropout_prob=0.0,
        )
        obs_default, _ = default.reset(seed=123)
        obs_explicit, _ = explicit.reset(seed=123)
        self.assertTrue(np.allclose(np.stack(obs_default), np.stack(obs_explicit)))
        actions = np.full((4, 3), 0.25, dtype=np.float32)
        for _ in range(3):
            out_default = default.step(actions)
            out_explicit = explicit.step(actions)
            self.assertTrue(
                np.allclose(np.stack(out_default[0]), np.stack(out_explicit[0]))
            )
            self.assertTrue(np.allclose(out_default[1], out_explicit[1]))

    def test_one_step_action_delay_holds_first_command(self):
        delayed = UAVSchedulingEnv(n_agents=2, n_targets=2, action_delay_steps=1)
        delayed.reset(seed=7)
        initial_positions = delayed.positions.copy()
        delayed.step(np.ones((2, 3), dtype=np.float32))
        self.assertTrue(np.allclose(delayed.positions, initial_positions))
        delayed.step(np.zeros((2, 3), dtype=np.float32))
        self.assertFalse(np.allclose(delayed.positions, initial_positions))

    def test_full_communication_dropout_zeros_neighbor_features(self):
        env = UAVSchedulingEnv(
            n_agents=4, n_targets=4, communication_dropout_prob=1.0
        )
        observations, _ = env.reset(seed=9)
        self.assertTrue(np.allclose(np.stack(observations)[:, 12:], 0.0))

    def test_combined_disturbances_are_seed_deterministic(self):
        kwargs = dict(
            n_agents=4,
            n_targets=4,
            wind_std=0.15,
            observation_noise_std=0.03,
            action_delay_steps=2,
            communication_dropout_prob=0.25,
        )
        first = UAVSchedulingEnv(**kwargs)
        second = UAVSchedulingEnv(**kwargs)
        obs_first, _ = first.reset(seed=77)
        obs_second, _ = second.reset(seed=77)
        self.assertTrue(np.allclose(np.stack(obs_first), np.stack(obs_second)))
        actions = np.full((4, 3), 0.4, dtype=np.float32)
        for _ in range(4):
            out_first = first.step(actions)
            out_second = second.step(actions)
            self.assertTrue(
                np.allclose(np.stack(out_first[0]), np.stack(out_second[0]))
            )
            self.assertTrue(np.allclose(out_first[1], out_second[1]))

    def test_resource_and_safety_metrics_are_exposed(self):
        env = UAVSchedulingEnv(n_agents=4, n_targets=4)
        env.reset(seed=5)
        _, _, _, _, info = env.step(np.full((4, 3), 0.25, dtype=np.float32))
        expected = {
            "energy_remaining",
            "action_magnitude",
            "speed",
            "distance_to_target",
            "min_neighbor_distance",
            "threat_zone_violation",
        }
        self.assertTrue(expected.issubset(info["uav_0"]))
        self.assertGreaterEqual(info["uav_0"]["energy_remaining"], 0.0)
        self.assertGreater(info["uav_0"]["min_neighbor_distance"], 0.0)


if __name__ == "__main__":
    unittest.main()
