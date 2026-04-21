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
    ):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = n_agents * obs_dim
        self.max_episode_steps = max_episode_steps
        self.world_size = world_size
        self.dt = dt

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
        self.step_count = 0
        self.np_random = np.random.default_rng()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.step_count = 0
        self.positions = self.np_random.uniform(
            low=-self.world_size,
            high=self.world_size,
            size=(self.n_agents, 3),
        ).astype(np.float32)
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
            self.pending_tasks - 0.03 * np.abs(actions[:, :2]),
            0.0,
            1.0,
        )

        dist_to_target = np.linalg.norm(self.targets - self.positions, axis=1)
        pairwise_dist = self._pairwise_distances(self.positions)
        safe_sep = 0.75
        collision_penalty = (pairwise_dist < safe_sep).astype(np.float32)
        np.fill_diagonal(collision_penalty, 0.0)
        collision_penalty = collision_penalty.sum(axis=1)

        task_completion = 1.0 - self.pending_tasks.mean(axis=1)
        rewards = (
            -0.3 * dist_to_target
            -0.1 * np.linalg.norm(actions, axis=1)
            -0.25 * collision_penalty
            +0.5 * task_completion
            +0.1 * self.energy.squeeze(-1)
        ).astype(np.float32)

        done = self.step_count >= self.max_episode_steps
        obs = self._get_obs()
        rewards = tuple(float(r) for r in rewards)
        info = {agent: {"task_completion": float(task_completion[i])} for i, agent in enumerate(self.agent_names)}
        return obs, rewards, done, done, info

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
