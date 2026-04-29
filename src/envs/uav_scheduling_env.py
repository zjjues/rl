from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional


def infer_neighbor_slots(n_agents: int, neighbor_slots: Optional[int] = None) -> int:
    if neighbor_slots is not None:
        return int(max(1, min(neighbor_slots, max(n_agents - 1, 1))))
    return int(min(3, max(n_agents - 1, 1)))


def infer_obs_dim(n_agents: int, neighbor_slots: Optional[int] = None) -> int:
    # self position(3) + self velocity(3) + assigned target delta(3) + energy(1)
    # + pending task(2) + K nearest neighbours * (relative position(3) + velocity(3))
    return 12 + 6 * infer_neighbor_slots(n_agents, neighbor_slots)


def infer_state_dim(
    n_agents: int,
    obs_dim: Optional[int] = None,
    neighbor_slots: Optional[int] = None,
) -> int:
    resolved_obs_dim = int(obs_dim if obs_dim is not None else infer_obs_dim(n_agents, neighbor_slots))
    return int(n_agents * resolved_obs_dim)


class UAVSchedulingEnv(gym.Env):
    """
    Lightweight continuous multi-agent UAV scheduling environment.
    Observation and global state sizes scale automatically with the swarm size.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        n_agents=4,
        obs_dim=None,
        action_dim=3,
        max_episode_steps=100,
        world_size=5.0,
        dt=0.2,
        n_targets=None,
        neighbor_slots=None,
        d_safe=0.1,
        reward_clip=2.0,
        reward_clip_min=-3.0,
        reward_clip_max=10.0,
        spawn_region_scale=0.35,
        spawn_separation_scale=1.05,
        task_progress_rate=0.03,
        task_reward_coef=1.20,
        safety_reward_coef=1.0,
        threat_zone_radius=0.75,
        threat_zone_target_count=2,
        stealth_threat_penalty=6.0,
        attack_threat_bonus=8.0,
        active_time_penalty=0.2,
        target_tracking_radius=1.25,
        intent_threshold=0.5,
        threat_penalty_respects_clip=True,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.neighbor_slots = infer_neighbor_slots(n_agents, neighbor_slots)
        self.obs_dim = int(obs_dim if obs_dim is not None else infer_obs_dim(n_agents, self.neighbor_slots))
        self.action_dim = action_dim
        self.state_dim = infer_state_dim(n_agents, self.obs_dim, self.neighbor_slots)
        self.max_episode_steps = max_episode_steps
        self.world_size = world_size
        self.dt = dt
        self.n_targets = n_targets or n_agents
        self.spawn_region_scale = spawn_region_scale
        self.task_progress_rate = task_progress_rate
        self.task_reward_coef = task_reward_coef
        self.safety_reward_coef = safety_reward_coef
        self.threat_zone_radius = float(threat_zone_radius)
        self.threat_zone_target_count = int(max(1, threat_zone_target_count))
        self.stealth_threat_penalty = float(stealth_threat_penalty)
        self.intent_threshold = float(intent_threshold)
        self.threat_penalty_respects_clip = bool(threat_penalty_respects_clip)

        self.agent_names = [f"uav_{i}" for i in range(self.n_agents)]
        self.action_space = spaces.Tuple(
            tuple(
                spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.action_dim,),
                    dtype=np.float32,
                )
                for _ in range(self.n_agents)
            )
        )
        self.observation_space = spaces.Tuple(
            tuple(
                spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.obs_dim,),
                    dtype=np.float32,
                )
                for _ in range(self.n_agents)
            )
        )

        self.positions = None
        self.velocities = None
        self.targets = None
        self.energy = None
        self.pending_tasks = None
        self.prev_dist_to_target = None
        self.prev_task_completion = None
        self.step_count = 0
        self.d_safe = d_safe
        self.reward_clip = reward_clip
        self.reward_clip_min = float(reward_clip_min)
        self.reward_clip_max = float(reward_clip_max)
        self.spawn_separation_scale = spawn_separation_scale
        self.spawn_separation = self.spawn_separation_scale * self.d_safe * 2.0 * self.world_size
        self.np_random = np.random.default_rng()
        self.collision_count = 0
        self.last_reward_terms = {}
        self.current_intent = np.zeros((1,), dtype=np.float32)
        self.current_tactical_posture = 1.0
        self.threat_zone_centers = np.zeros((0, 3), dtype=np.float32)
        self.attack_threat_bonus = float(attack_threat_bonus)
        self.active_time_penalty = float(active_time_penalty)
        self.target_tracking_radius = float(target_tracking_radius)

    def set_intent(self, intent) -> None:
        intent_array = np.asarray(intent, dtype=np.float32).reshape(-1)
        self.current_intent = intent_array

    def set_tactical_posture(self, posture) -> None:
        if isinstance(posture, str):
            posture_value = 1.0 if posture.lower() == "attack" else 0.0
        else:
            posture_value = float(posture)
        self.current_tactical_posture = posture_value

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.step_count = 0
        self.positions = self._sample_initial_positions().astype(np.float32)
        self.velocities = np.zeros((self.n_agents, 3), dtype=np.float32)
        target_pool = self.np_random.uniform(
            low=-self.world_size,
            high=self.world_size,
            size=(self.n_targets, 3),
        ).astype(np.float32)
        target_indices = np.arange(self.n_agents) % self.n_targets
        self.targets = target_pool[target_indices].astype(np.float32)
        threat_zone_count = min(self.threat_zone_target_count, len(target_pool))
        # In Stage 6 v2, radar zones sit directly on the high-value targets.
        # Attack-mode success therefore requires entering the threat bubble,
        # while stealth-mode pursuit remains explicitly hazardous.
        self.threat_zone_centers = target_pool[:threat_zone_count].astype(np.float32)
        self.energy = np.ones((self.n_agents, 1), dtype=np.float32)
        self.pending_tasks = self.np_random.uniform(
            low=0.0,
            high=1.0,
            size=(self.n_agents, 2),
        ).astype(np.float32)
        self.prev_dist_to_target = np.linalg.norm(
            self.targets - self.positions, axis=1
        ).astype(np.float32)
        self.prev_task_completion = (
            1.0 - self.pending_tasks.mean(axis=1)
        ).astype(np.float32)
        self.collision_count = 0
        self.last_reward_terms = {}
        self.current_intent = np.zeros((1,), dtype=np.float32)
        self.current_tactical_posture = 1.0
        obs = self._get_obs()
        info = {agent: {} for agent in self.agent_names}
        return obs, info

    def step(self, actions):
        self.step_count += 1
        actions = np.asarray(actions, dtype=np.float32)
        actions = np.clip(actions, -1.0, 1.0)

        self.velocities = 0.7 * self.velocities + 0.3 * actions
        self.positions = self.positions + self.dt * self.velocities
        self.positions = np.clip(self.positions, -self.world_size, self.world_size)
        self.energy = np.clip(
            self.energy - 0.02 * np.linalg.norm(actions, axis=1, keepdims=True),
            0.0,
            1.0,
        )
        self.pending_tasks = np.clip(
            self.pending_tasks - self.task_progress_rate * np.abs(actions[:, :2]),
            0.0,
            1.0,
        )

        dist_to_target = np.linalg.norm(self.targets - self.positions, axis=1)
        pairwise_dist = self._pairwise_distances(self.positions)
        normalised_pairwise_dist = pairwise_dist / max(2.0 * self.world_size, 1e-6)
        collision_penalty = (normalised_pairwise_dist < self.d_safe).astype(np.float32)
        np.fill_diagonal(collision_penalty, 0.0)
        collision_penalty = collision_penalty.sum(axis=1)
        safety_margin = 2.0 * self.d_safe
        proximity_penalty = np.clip(
            safety_margin - normalised_pairwise_dist,
            0.0,
            None,
        ).astype(np.float32)
        np.fill_diagonal(proximity_penalty, 0.0)
        proximity_penalty = proximity_penalty.sum(axis=1)
        collision_occurred = bool(np.any(collision_penalty > 0.0))
        if collision_occurred:
            self.collision_count += 1

        task_completion = 1.0 - self.pending_tasks.mean(axis=1)
        rewards = self._compute_rewards(
            dist_to_target=dist_to_target,
            action_norm=np.linalg.norm(actions, axis=1),
            collision_penalty=collision_penalty,
            proximity_penalty=proximity_penalty,
            task_completion=task_completion,
        )
        self.prev_dist_to_target = dist_to_target.astype(np.float32)
        self.prev_task_completion = task_completion.astype(np.float32)

        done = self.step_count >= self.max_episode_steps
        obs = self._get_obs()
        rewards = tuple(float(r) for r in rewards)
        info = {
            agent: {
                "task_completion": float(task_completion[i]),
                "collision": collision_occurred,
                "collision_count": int(self.collision_count),
                "reward_env": float(rewards[i]),
                "reward_env_raw": float(self.last_reward_terms["raw_total"][i]),
                "reward_dist": float(self.last_reward_terms["dist"][i]),
                "reward_energy": float(self.last_reward_terms["energy"][i]),
                "reward_collision": float(self.last_reward_terms["collision"][i]),
                "reward_safety": float(self.last_reward_terms["safety"][i]),
                "reward_task": float(self.last_reward_terms["task"][i]),
                "reward_time": float(self.last_reward_terms["time"][i]),
                "reward_threat": float(self.last_reward_terms["threat"][i]),
                "threat_zone_violation": bool(self.last_reward_terms["threat_violation"][i] > 0.0),
                "tactical_posture_flag": float(self.current_tactical_posture),
            }
            for i, agent in enumerate(self.agent_names)
        }
        return obs, rewards, done, done, info

    def _compute_rewards(
        self,
        dist_to_target,
        action_norm,
        collision_penalty,
        proximity_penalty,
        task_completion,
    ):
        # Keep the extrinsic reward in a narrow range so PPO targets remain stable.
        distance_progress = np.clip(
            (self.prev_dist_to_target - dist_to_target) / max(self.world_size, 1e-6),
            -1.0,
            1.0,
        )
        task_progress = np.clip(
            task_completion - self.prev_task_completion,
            -1.0,
            1.0,
        )
        dist_term = 0.30 * distance_progress
        energy_term = -0.015 * action_norm
        collision_term = -0.35 * np.clip(collision_penalty, 0.0, 1.0)
        safety_term = -self.safety_reward_coef * proximity_penalty
        task_term = self.task_reward_coef * task_progress
        distance_over_radius = np.clip(
            (dist_to_target - self.target_tracking_radius) / max(self.target_tracking_radius, 1e-6),
            0.0,
            3.0,
        )
        time_term = (-self.active_time_penalty * distance_over_radius).astype(np.float32)
        posture_flag = float(self.current_tactical_posture)
        threat_violation = np.zeros((self.n_agents,), dtype=np.float32)
        threat_term = np.zeros((self.n_agents,), dtype=np.float32)
        if self.threat_zone_centers.size > 0:
            threat_distances = np.linalg.norm(
                self.positions[:, None, :] - self.threat_zone_centers[None, :, :],
                axis=-1,
            )
            threat_violation = (threat_distances <= self.threat_zone_radius).any(axis=1).astype(np.float32)
            if posture_flag >= self.intent_threshold:
                task_term = task_term + self.attack_threat_bonus * threat_violation
            else:
                threat_term = -self.stealth_threat_penalty * threat_violation

        base_rewards = (
            dist_term
            + energy_term
            + collision_term
            + safety_term
            + task_term
            + time_term
        )
        total_rewards = base_rewards + threat_term
        if self.threat_penalty_respects_clip:
            clipped_rewards = np.clip(
                total_rewards,
                self.reward_clip_min,
                self.reward_clip_max,
            )
        else:
            clipped_rewards = (
                np.clip(base_rewards, self.reward_clip_min, self.reward_clip_max) + threat_term
            )
        clipped_rewards = clipped_rewards.astype(np.float32)
        self.last_reward_terms = {
            "dist": dist_term.astype(np.float32),
            "energy": energy_term.astype(np.float32),
            "collision": collision_term.astype(np.float32),
            "safety": safety_term.astype(np.float32),
            "task": task_term.astype(np.float32),
            "time": time_term.astype(np.float32),
            "threat": threat_term.astype(np.float32),
            "threat_violation": threat_violation.astype(np.float32),
            "raw_total": total_rewards.astype(np.float32),
        }
        return clipped_rewards

    def _sample_initial_positions(self):
        low = -self.spawn_region_scale * self.world_size
        high = self.spawn_region_scale * self.world_size
        positions = []
        for _ in range(self.n_agents):
            accepted = False
            for _ in range(128):
                candidate = self.np_random.uniform(low=low, high=high, size=(3,))
                if not positions:
                    positions.append(candidate)
                    accepted = True
                    break
                distances = np.linalg.norm(np.asarray(positions) - candidate, axis=1)
                if np.all(distances >= self.spawn_separation):
                    positions.append(candidate)
                    accepted = True
                    break
            if not accepted:
                positions.append(self.np_random.uniform(low=low, high=high, size=(3,)))
        return np.asarray(positions, dtype=np.float32)

    def _pairwise_distances(self, positions):
        diff = positions[:, None, :] - positions[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    def _get_obs(self):
        obs = []
        pairwise_dist = self._pairwise_distances(self.positions)
        for i in range(self.n_agents):
            rel_target = self.targets[i] - self.positions[i]
            nearest_indices = [j for j in np.argsort(pairwise_dist[i]) if j != i][: self.neighbor_slots]
            nearest_feats = []
            for j in nearest_indices:
                nearest_feats.extend((self.positions[j] - self.positions[i]).tolist())
                nearest_feats.extend(self.velocities[j].tolist())
            while len(nearest_feats) < 6 * self.neighbor_slots:
                nearest_feats.append(0.0)

            obs_i = np.concatenate(
                [
                    self.positions[i],
                    self.velocities[i],
                    rel_target,
                    self.energy[i],
                    self.pending_tasks[i],
                    np.asarray(nearest_feats, dtype=np.float32),
                ]
            ).astype(np.float32)
            obs.append(obs_i[: self.obs_dim])
        return tuple(obs)

    def render(self):
        return None

    def close(self):
        return None


gym.register(
    id="uav-scheduling-v0",
    entry_point="envs.uav_scheduling_env:UAVSchedulingEnv",
)
