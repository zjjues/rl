import gymnasium as gym
import numpy as np
from gymnasium import spaces


class UAVSchedulingEnv(gym.Env):
    """
    Lightweight continuous multi-agent UAV scheduling environment matching rl.md:
    - 4 agents
    - per-agent observation dim 18
    - global state dim 72
    - per-agent continuous action dim 3
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        n_agents=4,
        obs_dim=18,
        action_dim=3,
        max_episode_steps=100,
        world_size=5.0,
        dt=0.2,
        d_safe=0.1,
        reward_clip=1.0,
        spawn_region_scale=0.35,
        spawn_separation_scale=1.05,
        task_progress_rate=0.03,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = n_agents * obs_dim
        self.max_episode_steps = max_episode_steps
        self.world_size = world_size
        self.dt = dt
        self.spawn_region_scale = spawn_region_scale
        self.task_progress_rate = task_progress_rate

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
        self.spawn_separation_scale = spawn_separation_scale
        self.spawn_separation = self.spawn_separation_scale * self.d_safe * 2.0 * self.world_size
        self.np_random = np.random.default_rng()
        self.collision_count = 0
        self.last_reward_terms = {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.step_count = 0
        self.positions = self._sample_initial_positions().astype(np.float32)
        self.velocities = np.zeros((self.n_agents, 3), dtype=np.float32)
        self.targets = self.np_random.uniform(
            low=-self.world_size,
            high=self.world_size,
            size=(self.n_agents, 3),
        ).astype(np.float32)
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
        safety_margin = 1.75 * self.d_safe
        proximity_penalty = np.clip(
            (safety_margin - normalised_pairwise_dist) / max(safety_margin, 1e-6),
            0.0,
            1.0,
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
                "reward_dist": float(self.last_reward_terms["dist"][i]),
                "reward_energy": float(self.last_reward_terms["energy"][i]),
                "reward_collision": float(self.last_reward_terms["collision"][i]),
                "reward_safety": float(self.last_reward_terms["safety"][i]),
                "reward_task": float(self.last_reward_terms["task"][i]),
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
        safety_term = -0.12 * np.clip(proximity_penalty, 0.0, 1.5)
        task_term = 0.60 * task_progress

        rewards = (
            dist_term
            + energy_term
            + collision_term
            + safety_term
            + task_term
        )
        clipped_rewards = np.clip(rewards, -self.reward_clip, self.reward_clip).astype(np.float32)
        self.last_reward_terms = {
            "dist": dist_term.astype(np.float32),
            "energy": energy_term.astype(np.float32),
            "collision": collision_term.astype(np.float32),
            "safety": safety_term.astype(np.float32),
            "task": task_term.astype(np.float32),
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
            nearest_indices = [j for j in np.argsort(pairwise_dist[i]) if j != i][:2]
            nearest_feats = []
            for j in nearest_indices:
                nearest_feats.extend((self.positions[j] - self.positions[i]).tolist())
                nearest_feats.extend(self.velocities[j].tolist())
            while len(nearest_feats) < 12:
                nearest_feats.append(0.0)

            obs_i = np.concatenate(
                [
                    self.positions[i],
                    self.velocities[i],
                    rel_target,
                    self.energy[i],
                    self.pending_tasks[i],
                    np.asarray(nearest_feats[:6], dtype=np.float32),
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
