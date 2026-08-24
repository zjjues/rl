"""VMAS adapter for I-MAPPO — continuous-action multi-agent benchmark environments.

Provides a thin wrapper around vmas.make_env() that conforms to the interface
expected by imappo.py adapter functions (env_reset, env_step, normalise_obs, etc.).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
import vmas


# ── Dimension inference (environment-agnostic, callable before env creation) ──

def infer_vmas_dims(scenario: str, n_agents: int) -> Tuple[int, int, int]:
    """Return (obs_dim, state_dim, action_dim) for a VMAS scenario.

    Creates a temporary env to probe dimensions, then closes it.
    """
    env = vmas.make_env(
        scenario,
        num_envs=1,
        n_agents=n_agents,
        continuous_actions=True,
        dict_spaces=False,
        terminated_truncated=True,
        wrapper="gymnasium",
    )
    try:
        obs_space = env.observation_space
        action_space = env.action_space
        obs_dim = int(obs_space[0].shape[0]) if hasattr(obs_space, "__getitem__") else int(obs_space.shape[0])
        action_dim = int(action_space[0].shape[0]) if hasattr(action_space, "__getitem__") else int(action_space.shape[0])
        state_dim = int(n_agents * obs_dim)
    finally:
        env.close()
    return obs_dim, state_dim, action_dim


# ── VMAS Scenarios ────────────────────────────────────────────────────────────

VMAS_SCENARIOS = [
    "dispersion",      # cooperative: agents spread to cover landmarks
    "discovery",       # cooperative: discover and cover targets
    "flocking",        # cooperative: maintain formation while moving
    "give_way",        # collision avoidance
    "dropout",         # competitive: agents push each other out
    "football",        # competitive: simple football-like game
    "navigation",      # cooperative navigation around obstacles
]


# ── Adapter ───────────────────────────────────────────────────────────────────

class VMASAdapter(gym.Env):
    """Wraps a VMAS scenario for use with I-MAPPO training loops.

    Conforms to the interface expected by imappo.py helper functions:
    - reset() → (obs_tuple, info_dict)
    - step(actions_list) → (obs_tuple, reward_list, done, truncated, info_dict)
    - unwrapped.set_intent() — no-op hook for metadata logging

    Key differences from VMASWrapper in vmas_wrapper.py:
    - Continuous actions (not discrete)
    - No dict_spaces (list/tuple observations)
    - set_intent() and set_tactical_posture() no-op methods
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        scenario: str,
        n_agents: int = 3,
        max_steps: int = 100,
        seed: Optional[int] = None,
        **scenario_kwargs,
    ):
        self.scenario = scenario
        self.n_agents = n_agents
        self.max_steps_val = max_steps

        self._env = vmas.make_env(
            scenario,
            num_envs=1,
            n_agents=n_agents,
            continuous_actions=True,
            dict_spaces=False,
            terminated_truncated=True,
            wrapper="gymnasium",
            **scenario_kwargs,
        )

        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

        # Dimension attributes used by imappo.py adapter functions
        self.obs_dim = int(self.observation_space[0].shape[0])
        self.action_dim = int(self.action_space[0].shape[0])
        self.state_dim = int(n_agents * self.obs_dim)
        self.episode_limit = max_steps

        # Intent metadata (set by training loop, never used by env logic)
        self.current_intent: np.ndarray = np.zeros((1,), dtype=np.float32)
        self.current_intent_label: str = ""
        self.current_tactical_posture: float = 1.0

        self._seed = seed
        self._step_count = 0

    # ── Gymnasium Env API ─────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed = seed
        obs, info = self._env.reset(seed=self._seed, options=options)
        self._step_count = 0
        # VMAS returns list/tuple of arrays — this is what the adapter expects
        return tuple(obs), self._normalise_info(info)

    def step(self, actions):
        """Accept list of arrays (one per agent), return (obs, rewards, done, truncated, info).

        actions: list of np.ndarray, each shape (action_dim,)
        """
        obs, rews, done, truncated, info = self._env.step(actions)
        self._step_count += 1
        truncated = bool(truncated) or self._step_count >= self.max_steps_val
        return (
            tuple(obs), list(rews), bool(done), truncated,
            self._normalise_info(info),
        )

    def render(self, mode="human"):
        return self._env.render(mode=mode)

    def close(self):
        return self._env.close()

    # ── I-MAPPO hooks (no-ops — VMAS does not consume intent) ─────────────

    def set_intent(self, intent, label: str = "") -> None:
        self.current_intent = np.asarray(intent, dtype=np.float32).reshape(-1)
        self.current_intent_label = str(label) if label else ""

    def set_tactical_posture(self, posture) -> None:
        if isinstance(posture, str):
            self.current_tactical_posture = 1.0 if posture.lower() == "attack" else 0.0
        else:
            self.current_tactical_posture = float(posture)

    # ── Helpers ───────────────────────────────────────────────────────────

    @property
    def unwrapped(self):
        return self

    @staticmethod
    def _normalise_info(info: dict) -> dict:
        """Map VMAS per-agent info to the training loop's neutral schema.

        Only environment reward and explicit collision diagnostics are mapped.
        VMAS position rewards are deliberately not relabelled as UAV task or
        preference objectives.
        """
        if not info:
            return {}
        normalised = {}
        any_collision = False
        for index, (_, value) in enumerate(sorted(info.items())):
            if not isinstance(value, dict):
                continue
            agent_info = {}
            if "final_rew" in value:
                agent_info["reward_env"] = float(np.asarray(value["final_rew"]).item())
            if "agent_collisions" in value:
                collision_count = float(
                    np.asarray(value["agent_collisions"]).item()
                )
                agent_info["collision"] = float(collision_count > 0.0)
                any_collision = any_collision or collision_count > 0.0
            normalised[f"uav_{index}"] = agent_info
        normalised["collision"] = any_collision
        return normalised


# ── Helpers for ipappo.py adapter functions ────────────────────────────────────

def extract_vmas_metrics(infos: dict, agent_order: List[str]) -> Dict[str, float]:
    """Extract MPE/VMAS-style metrics from step info dicts.

    Returns keys that train_imappo's summarise_step_info expects,
    so the training loop logs without crashing. All UAV-specific
    reward terms default to 0.0.
    """
    result = {
        "task_completion": 0.0,
        "reward_env": 0.0,
        "reward_dist": 0.0,
        "reward_energy": 0.0,
        "reward_collision": 0.0,
        "reward_safety": 0.0,
        "reward_task": 0.0,
        "reward_time": 0.0,
        "reward_threat": 0.0,
    }
    # Try to extract VMAS reward components from flattened info
    for agent_id in agent_order:
        for suffix in ["reward", "task_completion", "coverage"]:
            key = f"{agent_id}_{suffix}" if isinstance(agent_id, str) else f"agent_{agent_id}_{suffix}"
            if key in infos:
                result["reward_env"] = float(infos.get(key, 0.0))
                break
    return result
