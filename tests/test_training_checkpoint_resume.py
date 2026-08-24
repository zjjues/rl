from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from imappo import IMAPPOConfig, RolloutBuffer, train_imappo  # noqa: E402
from matd3_baseline import train_matd3  # noqa: E402
from run_research_study import (  # noqa: E402
    load_training_checkpoint,
    write_training_checkpoint,
)


class EpisodeBoundaryInterrupt(RuntimeError):
    pass


class SeededTinyEnv:
    """Small deterministic environment whose reset seed owns all environment RNG."""

    def __init__(self, config: IMAPPOConfig):
        self.config = config
        self.agent_names = [f"uav_{index}" for index in range(config.n_agents)]
        self.unwrapped = self
        self.rng = None
        self.observations = None
        self.steps = 0

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.observations = self.rng.normal(
            size=(self.config.n_agents, self.config.obs_dim)
        ).astype(np.float32)
        return self._obs(), {name: {} for name in self.agent_names}

    def _obs(self):
        return {
            name: self.observations[index].copy()
            for index, name in enumerate(self.agent_names)
        }

    def step(self, actions):
        action_array = np.stack(
            [np.asarray(actions[name], dtype=np.float32) for name in self.agent_names]
        )
        noise = self.rng.normal(
            scale=0.01, size=self.observations.shape
        ).astype(np.float32)
        action_effect = np.pad(
            action_array,
            ((0, 0), (0, self.config.obs_dim - self.config.action_dim)),
        )
        self.observations = 0.9 * self.observations + 0.05 * action_effect + noise
        self.steps += 1
        terminal = self.steps >= self.config.max_steps
        rewards = {
            name: float(1.0 - np.square(action_array[index]).mean())
            for index, name in enumerate(self.agent_names)
        }
        dones = {name: terminal for name in self.agent_names}
        truncated = {name: False for name in self.agent_names}
        infos = {
            name: {
                "collision": False,
                "task_completion": float(self.steps) / self.config.max_steps,
            }
            for name in self.agent_names
        }
        return self._obs(), rewards, dones, truncated, infos

    def set_intent(self, intent, label=""):
        del intent, label

    def set_tactical_posture(self, posture):
        del posture

    def close(self):
        return None


def on_policy_config(algorithm: str) -> IMAPPOConfig:
    return IMAPPOConfig(
        algorithm=algorithm,
        critic_mode="mlp",
        intent_source="none",
        intent_profile_decoder="none",
        use_action_mask=False,
        policy_mode="direct",
        safety_filter_mode="none",
        n_agents=2,
        n_targets=2,
        obs_dim=6,
        state_dim=12,
        action_dim=3,
        intent_dim=25,
        ppo_epochs=1,
        minibatch_size=2,
        rollout_length=3,
        max_episodes=4,
        max_steps=2,
        eta=0.0,
        eta_end=0.0,
        potential_update_mode="frozen",
        device="cpu",
        seed=7,
    )


def matd3_config() -> IMAPPOConfig:
    config = on_policy_config("matd3")
    config.minibatch_size = 2
    config.replay_capacity = 32
    config.matd3_warmup_steps = 1
    config.matd3_policy_delay = 2
    return config


def assert_module_equal(test: unittest.TestCase, expected, actual) -> None:
    expected_state = expected.state_dict()
    actual_state = actual.state_dict()
    test.assertEqual(set(expected_state), set(actual_state))
    for key in expected_state:
        test.assertTrue(
            torch.equal(expected_state[key], actual_state[key]),
            f"module tensor differs after resume: {key}",
        )


class TrainingCheckpointResumeTests(unittest.TestCase):
    identity = {"test": "episode-boundary-exact-resume-v1"}

    def _interrupted_callback(self, path: Path):
        def callback(algo, state):
            write_training_checkpoint(path, algo, state, self.identity)
            if state["next_episode"] == 2:
                raise EpisodeBoundaryInterrupt("simulated process interruption")

        return callback

    def test_on_policy_resume_is_bitwise_equivalent_for_imappo_and_happo(self):
        for algorithm in ("imappo", "happo"):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as temp_dir:
                config = on_policy_config(algorithm)
                factory = lambda: SeededTinyEnv(config)
                expected_algo, expected_logs = train_imappo(
                    env_factory=factory, config=config
                )
                checkpoint = Path(temp_dir) / f"{algorithm}.pt"
                with self.assertRaises(EpisodeBoundaryInterrupt):
                    train_imappo(
                        env_factory=factory,
                        config=config,
                        training_state_callback=self._interrupted_callback(checkpoint),
                    )
                restored_algo, state = load_training_checkpoint(
                    checkpoint, config, self.identity
                )
                restored_buffer = RolloutBuffer()
                restored_buffer.load_state_dict(state["rollout_buffer"])
                actual_algo, actual_logs = train_imappo(
                    env_factory=factory,
                    config=config,
                    initial_algo=restored_algo,
                    initial_buffer=restored_buffer,
                    initial_logs=state["logs"],
                    start_episode=state["next_episode"],
                )
                self.assertEqual(expected_logs, actual_logs)
                if algorithm == "happo":
                    for expected_actor, actual_actor in zip(
                        expected_algo.actor, actual_algo.actor
                    ):
                        assert_module_equal(self, expected_actor, actual_actor)
                else:
                    assert_module_equal(self, expected_algo.actor, actual_algo.actor)
                    assert_module_equal(self, expected_algo.potential, actual_algo.potential)
                assert_module_equal(self, expected_algo.critic, actual_algo.critic)

    def test_matd3_resume_restores_targets_replay_and_delayed_update_cursor(self):
        config = matd3_config()
        factory = lambda: SeededTinyEnv(config)
        expected_algo, expected_logs = train_matd3(factory, config)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "matd3.pt"
            with self.assertRaises(EpisodeBoundaryInterrupt):
                train_matd3(
                    factory,
                    config,
                    training_state_callback=self._interrupted_callback(checkpoint),
                )
            self.assertTrue(checkpoint.is_file())
            self.assertFalse(checkpoint.with_suffix(".pt.tmp").exists())
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_training_checkpoint(checkpoint, config, {"test": "wrong"})
            restored_algo, state = load_training_checkpoint(
                checkpoint, config, self.identity
            )
            actual_algo, actual_logs = train_matd3(
                factory,
                config,
                initial_algo=restored_algo,
                initial_logs=state["logs"],
                start_episode=state["next_episode"],
                total_steps=state["total_steps"],
            )
        self.assertEqual(expected_logs, actual_logs)
        for name in ("actor", "actor_target", "critic", "critic_target"):
            assert_module_equal(self, getattr(expected_algo, name), getattr(actual_algo, name))
        self.assertEqual(expected_algo.update_steps, actual_algo.update_steps)
        self.assertEqual(expected_algo.actor_updates, actual_algo.actor_updates)
        self.assertEqual(len(expected_algo.replay), len(actual_algo.replay))
        for expected, actual in zip(expected_algo.replay.data, actual_algo.replay.data):
            for expected_value, actual_value in zip(expected, actual):
                if isinstance(expected_value, np.ndarray):
                    np.testing.assert_array_equal(expected_value, actual_value)
                else:
                    self.assertEqual(expected_value, actual_value)


if __name__ == "__main__":
    unittest.main()
