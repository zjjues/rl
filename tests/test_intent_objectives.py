from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from envs.uav_scheduling_env import UAVSchedulingEnv  # noqa: E402
from intent_objectives import UAV_INTENT_REWARD_PROFILES  # noqa: E402
from intent_semantic_encoder import DEFAULT_INTENT_DESCRIPTIONS  # noqa: E402


class IntentObjectiveTests(unittest.TestCase):
    def test_every_uav_catalog_intent_has_an_explicit_reward_profile(self):
        catalog_labels = {label for label, _ in DEFAULT_INTENT_DESCRIPTIONS}
        self.assertEqual(catalog_labels, set(UAV_INTENT_REWARD_PROFILES))

    def _reward_for(self, env: UAVSchedulingEnv, label: str) -> np.ndarray:
        env.set_intent(np.zeros(8, dtype=np.float32), label)
        return env._compute_rewards(
            dist_to_target=np.full(env.n_agents, 2.0, dtype=np.float32),
            action_norm=np.full(env.n_agents, 0.5, dtype=np.float32),
            collision_penalty=np.zeros(env.n_agents, dtype=np.float32),
            proximity_penalty=np.full(env.n_agents, 0.2, dtype=np.float32),
            task_completion=env.prev_task_completion + 0.05,
            task_cost=float(env.prev_task_cost) - 0.5,
        )

    def test_enabled_profiles_make_labels_behaviorally_distinct(self):
        env = UAVSchedulingEnv(n_agents=4, intent_reward_profiles_enabled=True)
        env.reset(seed=7)
        safety_reward = self._reward_for(env, "safety_first")
        efficiency_reward = self._reward_for(env, "efficiency_first")
        self.assertFalse(np.allclose(safety_reward, efficiency_reward))

    def test_disabled_profiles_preserve_legacy_reward(self):
        env = UAVSchedulingEnv(n_agents=4, intent_reward_profiles_enabled=False)
        env.reset(seed=7)
        safety_reward = self._reward_for(env, "safety_first")
        efficiency_reward = self._reward_for(env, "efficiency_first")
        np.testing.assert_allclose(safety_reward, efficiency_reward)

    def test_neutral_posture_has_no_attack_or_stealth_threat_term(self):
        env = UAVSchedulingEnv(n_agents=4, intent_reward_profiles_enabled=True)
        env.reset(seed=7)
        env.positions[:] = env.threat_zone_centers[0]
        env.set_tactical_posture("neutral")
        self._reward_for(env, "balanced")
        np.testing.assert_allclose(env.last_reward_terms["threat"], 0.0)
        self.assertEqual(env.current_tactical_posture, 0.5)


if __name__ == "__main__":
    unittest.main()
