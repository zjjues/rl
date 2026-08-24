"""True continuous-action multi-agent TD3 baseline for the UAV study."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import fields
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from intent_semantic_encoder import IntentLibrary


def mlp(input_dim: int, hidden: Tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    previous = input_dim
    for width in hidden:
        layers.extend((nn.Linear(previous, width), nn.ReLU()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class SharedDeterministicActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.network = mlp(obs_dim, (256, 256), action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(observations))


class CentralTwinCritic(nn.Module):
    def __init__(self, state_dim: int, n_agents: int, action_dim: int) -> None:
        super().__init__()
        input_dim = state_dim + n_agents * action_dim
        self.q1 = mlp(input_dim, (256, 256), 1)
        self.q2 = mlp(input_dim, (256, 256), 1)

    def forward(
        self, state: torch.Tensor, joint_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat((state, joint_action.flatten(start_dim=1)), dim=-1)
        return self.q1(features).squeeze(-1), self.q2(features).squeeze(-1)

    def first(self, state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        features = torch.cat((state, joint_action.flatten(start_dim=1)), dim=-1)
        return self.q1(features).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.data = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self.data)

    def add(
        self,
        state: np.ndarray,
        observations: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        next_observations: np.ndarray,
        done: bool,
    ) -> None:
        self.data.append(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(observations, dtype=np.float32),
                np.asarray(actions, dtype=np.float32),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                np.asarray(next_observations, dtype=np.float32),
                float(done),
            )
        )

    def sample(self, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, ...]:
        indices = rng.choice(len(self.data), size=int(batch_size), replace=False)
        fields = list(zip(*(self.data[int(index)] for index in indices)))
        return tuple(np.stack(field) for field in fields)

    def state_dict(self) -> Dict[str, object]:
        return {
            "capacity": int(self.data.maxlen),
            "data": [
                tuple(
                    value.copy() if isinstance(value, np.ndarray) else value
                    for value in transition
                )
                for transition in self.data
            ],
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        capacity = int(state["capacity"])
        if capacity != int(self.data.maxlen):
            raise ValueError(
                "replay-buffer capacity mismatch: "
                f"checkpoint={capacity}, config={self.data.maxlen}"
            )
        restored = deque(maxlen=capacity)
        for transition in state["data"]:
            if len(transition) != 7:
                raise ValueError("invalid MATD3 replay transition")
            restored.append(
                tuple(
                    value.copy() if isinstance(value, np.ndarray) else value
                    for value in transition
                )
            )
        self.data = restored


class MATD3Baseline:
    """Shared actor with centralized twin critics and no intent input."""

    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.actor = SharedDeterministicActor(config.obs_dim, config.action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = CentralTwinCritic(
            config.state_dim, config.n_agents, config.action_dim
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.replay = ReplayBuffer(config.replay_capacity)
        self.update_steps = 0
        self.actor_updates = 0
        self.last_actor_loss = 0.0
        self.task_intent_library = IntentLibrary.create_onehot()
        if config.intent_train_labels:
            self.task_intent_library = self.task_intent_library.subset_by_labels(
                config.intent_train_labels
            )

    def intent_representation_metadata(self) -> Dict[str, object]:
        return {
            "representation_type": "none",
            "semantic_geometry": False,
            "intent_conditioning": False,
            "task_labels_hidden_from_policy": True,
            "algorithm": "matd3",
            "centralized_twin_critics": True,
            "target_policy_smoothing": True,
            "delayed_policy_updates": int(self.config.matd3_policy_delay),
        }

    def _mask(self) -> torch.Tensor:
        return torch.ones(
            self.config.n_agents,
            self.config.action_dim,
            dtype=torch.float32,
            device=self.device,
        )

    def _action_mask_for_posture(self, posture=None) -> torch.Tensor:
        del posture
        return self._mask()

    def sample_episode_intent_and_mask(self, tactical_posture=None):
        _, labels, _ = self.task_intent_library.sample_with_info(
            1, posture=tactical_posture, rng=self.rng
        )
        return torch.zeros(self.config.intent_dim, device=self.device), self._mask(), labels[0]

    def evaluation_intent_and_mask(self, mode: str = "standard"):
        label = "safety_first" if mode == "dense" else "balanced"
        if self.task_intent_library.get_by_label(label) is None:
            label = self.task_intent_library.labels[0]
        return torch.zeros(self.config.intent_dim, device=self.device), self._mask(), label

    def encode_intent_queries(self, entries):
        return torch.zeros(len(entries), self.config.intent_dim, device=self.device)

    def set_evaluation_context(self, label: str, posture: str) -> None:
        del label, posture

    def select_actions(
        self,
        observations: torch.Tensor,
        intent: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ):
        del intent
        with torch.no_grad():
            actions = self.actor(observations.to(self.device))
            if not deterministic:
                actions = actions + torch.randn_like(actions) * self.config.matd3_exploration_noise
            actions = torch.clamp(actions, -1.0, 1.0) * action_mask
        return actions, torch.zeros(actions.size(0), device=self.device)

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for target_parameter, parameter in zip(target.parameters(), source.parameters()):
                target_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)

    def update(self) -> Dict[str, float]:
        batch = self.replay.sample(self.config.minibatch_size, self.rng)
        state, observations, actions, reward, next_state, next_observations, done = [
            torch.as_tensor(value, dtype=torch.float32, device=self.device) for value in batch
        ]
        batch_size = observations.size(0)
        with torch.no_grad():
            next_actions = self.actor_target(
                next_observations.reshape(-1, self.config.obs_dim)
            ).reshape(batch_size, self.config.n_agents, self.config.action_dim)
            noise = torch.clamp(
                torch.randn_like(next_actions) * self.config.matd3_policy_noise,
                -self.config.matd3_noise_clip,
                self.config.matd3_noise_clip,
            )
            next_actions = torch.clamp(next_actions + noise, -1.0, 1.0)
            target_q1, target_q2 = self.critic_target(next_state, next_actions)
            target = reward + self.config.gamma * (1.0 - done) * torch.minimum(
                target_q1, target_q2
            )

        q1, q2 = self.critic(state, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optim.step()

        self.update_steps += 1
        actor_updated = False
        if self.update_steps % self.config.matd3_policy_delay == 0:
            policy_actions = self.actor(
                observations.reshape(-1, self.config.obs_dim)
            ).reshape(batch_size, self.config.n_agents, self.config.action_dim)
            actor_loss = -self.critic.first(state, policy_actions).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_optim.step()
            self._soft_update(self.actor_target, self.actor, self.config.matd3_tau)
            self._soft_update(self.critic_target, self.critic, self.config.matd3_tau)
            self.last_actor_loss = float(actor_loss.item())
            self.actor_updates += 1
            actor_updated = True
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": self.last_actor_loss,
            "actor_updated": float(actor_updated),
            "actor_updates": float(self.actor_updates),
            "replay_size": float(len(self.replay)),
        }

    def save_checkpoint(
        self, path: str, extra: Optional[Dict[str, object]] = None
    ) -> None:
        config_fields = {field.name for field in fields(self.config)}
        torch.save(
            {
                "checkpoint_schema": "matd3_training_v1",
                "config": {
                    key: value
                    for key, value in self.config.__dict__.items()
                    if key in config_fields
                },
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "replay": self.replay.state_dict(),
                "update_steps": int(self.update_steps),
                "actor_updates": int(self.actor_updates),
                "last_actor_loss": float(self.last_actor_loss),
                "rng_state": self.rng.bit_generator.state,
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls, path: str, device: Optional[str] = None
    ) -> "MATD3Baseline":
        from imappo import IMAPPOConfig

        checkpoint = torch.load(
            path, map_location=device or "cpu", weights_only=False
        )
        if checkpoint.get("checkpoint_schema") != "matd3_training_v1":
            raise ValueError("checkpoint is not a MATD3 training checkpoint")
        config_dict = dict(checkpoint["config"])
        if device is not None:
            config_dict["device"] = device
        algo = cls(IMAPPOConfig(**config_dict))
        algo.actor.load_state_dict(checkpoint["actor"])
        algo.actor_target.load_state_dict(checkpoint["actor_target"])
        algo.critic.load_state_dict(checkpoint["critic"])
        algo.critic_target.load_state_dict(checkpoint["critic_target"])
        algo.actor_optim.load_state_dict(checkpoint["actor_optim"])
        algo.critic_optim.load_state_dict(checkpoint["critic_optim"])
        algo.replay.load_state_dict(checkpoint["replay"])
        algo.update_steps = int(checkpoint.get("update_steps", 0))
        algo.actor_updates = int(checkpoint.get("actor_updates", 0))
        algo.last_actor_loss = float(checkpoint.get("last_actor_loss", 0.0))
        if "rng_state" in checkpoint:
            algo.rng.bit_generator.state = checkpoint["rng_state"]
        return algo


def train_matd3(
    env_factory,
    config,
    *,
    initial_algo: Optional[MATD3Baseline] = None,
    initial_logs: Optional[List[Dict[str, object]]] = None,
    start_episode: int = 0,
    total_steps: int = 0,
    training_state_callback: Optional[
        Callable[[object, Dict[str, object]], None]
    ] = None,
):
    """Train MATD3 under the same deterministic episode seed protocol."""
    from imappo import (
        build_global_state,
        env_reset,
        env_step,
        extract_collision_count,
        infer_agent_order,
        normalise_obs,
        set_env_intent,
        set_env_tactical_posture,
        summarise_step_info,
        training_tactical_posture,
    )

    if not 0 <= int(start_episode) <= int(config.max_episodes):
        raise ValueError(
            f"start_episode must be in [0, {config.max_episodes}], got {start_episode}"
        )
    algo = initial_algo if initial_algo is not None else MATD3Baseline(config)
    if algo.config.algorithm != config.algorithm:
        raise ValueError("restored algorithm does not match training config")
    logs = list(initial_logs or [])
    total_steps = int(total_steps)
    for episode in range(int(start_episode), config.max_episodes):
        config._env_progress = episode / max(config.max_episodes - 1, 1)
        config._use_hard_train_env = (
            config.hard_train_interval > 0
            and (episode + 1) % config.hard_train_interval == 0
        )
        env = env_factory()
        try:
            obs_data, _ = env_reset(env, seed=int(config.seed) * 1_000_000 + episode)
            agent_order = infer_agent_order(env, obs_data, config)
            observations = normalise_obs(agent_order, obs_data)
            state = build_global_state(observations, config)
            posture = training_tactical_posture(episode)
            intent, mask, label = algo.sample_episode_intent_and_mask(posture)
            set_env_intent(env, intent, label)
            set_env_tactical_posture(env, posture)
            episode_return = 0.0
            episode_collisions = 0.0
            episode_completion = 0.0
            episode_steps = 0
            last_update = {}
            for _ in range(config.max_steps):
                obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=algo.device)
                if total_steps < config.matd3_warmup_steps:
                    actions = algo.rng.uniform(
                        -1.0, 1.0, size=(config.n_agents, config.action_dim)
                    ).astype(np.float32)
                else:
                    actions, _ = algo.select_actions(obs_tensor, intent, mask, deterministic=False)
                    actions = actions.cpu().numpy()
                next_obs_data, rewards, done, truncated, info = env_step(
                    env, agent_order, actions
                )
                next_observations = normalise_obs(agent_order, next_obs_data)
                next_state = build_global_state(next_observations, config)
                terminal = bool(done or truncated)
                algo.replay.add(
                    state,
                    observations,
                    actions,
                    float(np.mean(rewards)),
                    next_state,
                    next_observations,
                    terminal,
                )
                if len(algo.replay) >= config.minibatch_size:
                    last_update = algo.update()
                step_info = summarise_step_info(info, agent_order)
                episode_return += float(np.mean(rewards))
                episode_collisions += float(extract_collision_count(info, agent_order))
                episode_completion += step_info["task_completion"]
                episode_steps += 1
                total_steps += 1
                observations, state = next_observations, next_state
                if terminal:
                    break
            logs.append(
                {
                    "episode": float(episode),
                    "episode_return": episode_return,
                    "episode_collisions": episode_collisions,
                    "episode_collision_rate": episode_collisions / max(episode_steps, 1),
                    "episode_task_completion": episode_completion / max(episode_steps, 1),
                    "algorithm": "matd3",
                    "intent_label": label,
                    "tactical_posture": posture,
                    **last_update,
                }
            )
        finally:
            if hasattr(env, "close"):
                env.close()
        if training_state_callback is not None:
            training_state_callback(
                algo,
                {
                    "next_episode": int(episode + 1),
                    "logs": logs,
                    "total_steps": int(total_steps),
                },
            )
    return algo, logs
