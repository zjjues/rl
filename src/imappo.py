from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym


Tensor = torch.Tensor


@dataclass
class IMAPPOConfig:
    n_agents: int = 4
    obs_dim: int = 18
    state_dim: int = 72
    action_dim: int = 3
    intent_dim: int = 8

    gamma: float = 0.99
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    eta: float = 0.5
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    max_grad_norm: float = 10.0

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    potential_lr: float = 3e-4
    potential_update_mode: str = "normal"
    potential_update_interval: int = 4

    ppo_epochs: int = 4
    minibatch_size: int = 128
    rollout_length: int = 512
    max_episodes: int = 100
    max_steps: int = 200

    actor_hidden_dims: Tuple[int, int, int] = (256, 256, 128)
    critic_hidden_dims: Tuple[int, int, int] = (256, 256, 128)
    feature_hidden_dim: int = 128
    attention_dim: int = 128

    action_low: float = -1.0
    action_high: float = 1.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    device: str = "cpu"
    seed: int = 42


def build_mlp(input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class IntentConditionedActor(nn.Module):
    def __init__(self, config: IMAPPOConfig):
        super().__init__()
        self.config = config
        input_dim = config.obs_dim + config.intent_dim
        h1, h2, h3 = config.actor_hidden_dims
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(h3, config.action_dim)
        self.log_std_head = nn.Linear(h3, config.action_dim)

    def forward(
        self,
        obs: Tensor,
        intent: Tensor,
        action_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x = torch.cat([obs, intent], dim=-1)
        hidden = self.backbone(x)
        action_mean = torch.tanh(self.mean_head(hidden))
        action_mean = action_mean * self.config.action_high
        if action_mask is not None:
            action_mean = action_mean * action_mask

        action_log_std = self.log_std_head(hidden)
        action_log_std = torch.clamp(
            action_log_std, self.config.log_std_min, self.config.log_std_max
        )
        if action_mask is not None:
            # Masked dimensions collapse to a near-deterministic safe action around zero.
            action_log_std = torch.where(
                action_mask > 0,
                action_log_std,
                torch.full_like(action_log_std, self.config.log_std_min),
            )
        return action_mean, action_log_std, hidden

    def distribution(
        self,
        obs: Tensor,
        intent: Tensor,
        action_mask: Optional[Tensor] = None,
    ) -> Tuple[Normal, Tensor]:
        action_mean, action_log_std, hidden = self.forward(obs, intent, action_mask)
        action_std = action_log_std.exp()
        return Normal(action_mean, action_std), hidden

    def sample_action(
        self,
        obs: Tensor,
        intent: Tensor,
        action_mask: Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        dist, hidden = self.distribution(obs, intent, action_mask)
        if deterministic:
            raw_action = dist.mean
        else:
            raw_action = dist.rsample()
        action = torch.clamp(raw_action, self.config.action_low, self.config.action_high)
        if action_mask is not None:
            action = action * action_mask
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, hidden


class CrossAttentionCritic(nn.Module):
    def __init__(self, config: IMAPPOConfig):
        super().__init__()
        self.config = config
        self.agent_feature_extractor = nn.Sequential(
            nn.Linear(config.obs_dim, config.feature_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.feature_hidden_dim, config.feature_hidden_dim),
            nn.ReLU(),
        )
        self.query = nn.Linear(config.intent_dim, config.attention_dim)
        self.key = nn.Linear(config.feature_hidden_dim, config.attention_dim)
        self.value = nn.Linear(config.feature_hidden_dim, config.feature_hidden_dim)
        h1, h2, h3 = config.critic_hidden_dims
        self.value_head = nn.Sequential(
            nn.Linear(config.state_dim + config.feature_hidden_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, 1),
        )

    def encode_agent_features(self, obs_all_agents: Tensor) -> Tensor:
        batch_size, n_agents, obs_dim = obs_all_agents.shape
        features = self.agent_feature_extractor(obs_all_agents.reshape(-1, obs_dim))
        return features.reshape(batch_size, n_agents, -1)

    def forward(
        self,
        state: Tensor,
        intent: Tensor,
        obs_all_agents: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        agent_features = self.encode_agent_features(obs_all_agents)
        query = self.query(intent).unsqueeze(1)
        key = self.key(agent_features)
        value = self.value(agent_features)
        scale = float(key.size(-1)) ** 0.5
        attention_logits = torch.matmul(query, key.transpose(1, 2)) / scale
        attention_weights = F.softmax(attention_logits, dim=-1)
        context = torch.matmul(attention_weights, value).squeeze(1)
        critic_input = torch.cat([state, context], dim=-1)
        return self.value_head(critic_input).squeeze(-1), attention_weights.squeeze(1)


class StateIntentPotential(nn.Module):
    def __init__(self, config: IMAPPOConfig):
        super().__init__()
        self.config = config
        self.state_encoder = build_mlp(
            config.state_dim,
            (128, 128),
            config.intent_dim,
        )

    def forward(self, state: Tensor, intent: Tensor) -> Tensor:
        state_embedding = self.state_encoder(state)
        mse = F.mse_loss(state_embedding, intent, reduction="none").mean(dim=-1)
        return -mse


class RolloutBuffer:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.storage: Dict[str, List[Tensor]] = {
            "states": [],
            "obs": [],
            "actions": [],
            "action_masks": [],
            "intents": [],
            "rewards": [],
            "dones": [],
            "log_probs": [],
            "next_states": [],
            "next_obs": [],
        }

    def add(
        self,
        *,
        state: Tensor,
        obs: Tensor,
        action: Tensor,
        action_mask: Tensor,
        intent: Tensor,
        reward: Tensor,
        done: Tensor,
        log_prob: Tensor,
        next_state: Tensor,
        next_obs: Tensor,
    ) -> None:
        self.storage["states"].append(state.detach().cpu())
        self.storage["obs"].append(obs.detach().cpu())
        self.storage["actions"].append(action.detach().cpu())
        self.storage["action_masks"].append(action_mask.detach().cpu())
        self.storage["intents"].append(intent.detach().cpu())
        self.storage["rewards"].append(reward.detach().cpu())
        self.storage["dones"].append(done.detach().cpu())
        self.storage["log_probs"].append(log_prob.detach().cpu())
        self.storage["next_states"].append(next_state.detach().cpu())
        self.storage["next_obs"].append(next_obs.detach().cpu())

    def is_ready(self, rollout_length: int) -> bool:
        return len(self.storage["states"]) >= rollout_length

    def tensorize(self, device: torch.device) -> Dict[str, Tensor]:
        return {
            key: torch.stack(value, dim=0).to(device)
            for key, value in self.storage.items()
        }


class IMAPPO:
    def __init__(self, config: IMAPPOConfig):
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.actor = IntentConditionedActor(config).to(self.device)
        self.critic = CrossAttentionCritic(config).to(self.device)
        self.potential = StateIntentPotential(config).to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr
        )
        self.potential_optim = torch.optim.Adam(
            self.potential.parameters(), lr=config.potential_lr
        )
        self.potential_update_step = 0

        if config.potential_update_mode == "frozen":
            for param in self.potential.parameters():
                param.requires_grad = False

    def sample_episode_intent_and_mask(self) -> Tuple[Tensor, Tensor]:
        intent = torch.randn(self.config.intent_dim, device=self.device)
        mask = (torch.rand(self.config.n_agents, self.config.action_dim, device=self.device) > 0.2).float()
        mask[:, 0] = 1.0
        return intent, mask

    def compute_shaped_rewards(
        self,
        env_reward: Tensor,
        state: Tensor,
        next_state: Tensor,
        intent: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        phi_t = self.potential(state.unsqueeze(0), intent.unsqueeze(0)).squeeze(0)
        phi_tp1 = self.potential(next_state.unsqueeze(0), intent.unsqueeze(0)).squeeze(0)
        intrinsic_reward = self.config.gamma * phi_tp1 - phi_t
        total_reward = env_reward + self.config.eta * intrinsic_reward
        return total_reward, intrinsic_reward

    def compute_gae(
        self,
        rewards: Tensor,
        dones: Tensor,
        values: Tensor,
        next_values: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(1, device=rewards.device)
        for t in reversed(range(rewards.size(0))):
            delta = rewards[t] + self.config.gamma * next_values[t] * (1.0 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        batch = buffer.tensorize(self.device)
        states = batch["states"]
        obs = batch["obs"]
        actions = batch["actions"]
        action_masks = batch["action_masks"]
        intents = batch["intents"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        old_log_probs = batch["log_probs"]
        next_states = batch["next_states"]
        next_obs = batch["next_obs"]

        values, _ = self.critic(states, intents, obs)
        with torch.no_grad():
            next_values, _ = self.critic(next_states, intents, next_obs)

        team_rewards = rewards.mean(dim=-1)
        advantages, returns = self.compute_gae(team_rewards, dones, values.detach(), next_values.detach())
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        agent_advantages = advantages.unsqueeze(1).expand(-1, self.config.n_agents)

        batch_size = states.size(0)
        flat_indices = np.arange(batch_size)
        last_actor_loss = 0.0
        last_critic_loss = 0.0
        last_entropy = 0.0
        last_potential_loss = 0.0

        for _ in range(self.config.ppo_epochs):
            np.random.shuffle(flat_indices)
            for start in range(0, batch_size, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                idx = flat_indices[start:end]
                mb_states = states[idx]
                mb_obs = obs[idx]
                mb_actions = actions[idx]
                mb_action_masks = action_masks[idx]
                mb_intents = intents[idx]
                mb_returns = returns[idx]
                mb_advantages = agent_advantages[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_next_states = next_states[idx]
                mb_dones = dones[idx]

                critic_values, _ = self.critic(mb_states, mb_intents, mb_obs)
                critic_loss = F.mse_loss(critic_values, mb_returns)

                flat_obs = mb_obs.reshape(-1, self.config.obs_dim)
                flat_intents = (
                    mb_intents.unsqueeze(1)
                    .expand(-1, self.config.n_agents, -1)
                    .reshape(-1, self.config.intent_dim)
                )
                flat_masks = mb_action_masks.reshape(-1, self.config.action_dim)
                flat_actions = mb_actions.reshape(-1, self.config.action_dim)
                flat_advantages = mb_advantages.reshape(-1)
                flat_old_log_probs = mb_old_log_probs.reshape(-1)

                dist, _ = self.actor.distribution(flat_obs, flat_intents, flat_masks)
                new_log_probs = dist.log_prob(flat_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                ratios = torch.exp(new_log_probs - flat_old_log_probs)
                surr1 = ratios * flat_advantages
                surr2 = torch.clamp(
                    ratios,
                    1.0 - self.config.eps_clip,
                    1.0 + self.config.eps_clip,
                ) * flat_advantages
                actor_loss = -(torch.min(surr1, surr2).mean() + self.config.entropy_coef * entropy)

                state_embed = self.potential.state_encoder(mb_states)
                next_state_embed = self.potential.state_encoder(mb_next_states)
                potential_loss = (
                    F.mse_loss(state_embed, mb_intents)
                    + F.mse_loss(next_state_embed, mb_intents)
                )

                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                (self.config.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optim.step()

                should_update_potential = False
                if self.config.potential_update_mode == "normal":
                    should_update_potential = True
                elif self.config.potential_update_mode == "slow":
                    should_update_potential = (
                        self.potential_update_step % max(self.config.potential_update_interval, 1) == 0
                    )
                elif self.config.potential_update_mode == "frozen":
                    should_update_potential = False
                else:
                    raise ValueError(
                        f"Unsupported potential_update_mode: {self.config.potential_update_mode}"
                    )

                if should_update_potential:
                    self.potential_optim.zero_grad()
                    potential_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.potential.parameters(), self.config.max_grad_norm
                    )
                    self.potential_optim.step()
                self.potential_update_step += 1

                last_actor_loss = float(actor_loss.item())
                last_critic_loss = float(critic_loss.item())
                last_entropy = float(entropy.item())
                last_potential_loss = float(potential_loss.item())

        buffer.clear()
        return {
            "actor_loss": last_actor_loss,
            "critic_loss": last_critic_loss,
            "entropy": last_entropy,
            "potential_loss": last_potential_loss,
            "return_mean": float(team_rewards.mean().item()),
        }

    def select_actions(
        self,
        obs: Tensor,
        intent: Tensor,
        action_mask: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        repeated_intent = intent.unsqueeze(0).expand(self.config.n_agents, -1)
        actions, log_probs, _ = self.actor.sample_action(
            obs,
            repeated_intent,
            action_mask,
            deterministic=deterministic,
        )
        return actions, log_probs


class MockContinuousUAVEnv:
    """
    Minimal smoke-test environment matching the tensor shapes from rl.md.
    This is only a fallback when a real PettingZoo continuous UAV task is not available.
    """

    def __init__(self, config: IMAPPOConfig):
        self.config = config
        self.agent_names = [f"uav_{i}" for i in range(config.n_agents)]
        self.steps = 0
        self.max_steps = config.max_steps
        self.state = None
        self.obs = None

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        self.steps = 0
        self.state = np.random.randn(self.config.state_dim).astype(np.float32)
        self.obs = np.random.randn(self.config.n_agents, self.config.obs_dim).astype(np.float32)
        obs_dict = {
            name: self.obs[i] for i, name in enumerate(self.agent_names)
        }
        infos = {name: {} for name in self.agent_names}
        return obs_dict, infos

    def step(
        self, action_dict: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        self.steps += 1
        actions = np.stack([action_dict[name] for name in self.agent_names], axis=0)
        action_penalty = np.square(actions).mean()
        self.state = (
            0.9 * self.state + 0.1 * np.random.randn(self.config.state_dim).astype(np.float32)
        )
        self.obs = (
            0.9 * self.obs + 0.1 * np.random.randn(self.config.n_agents, self.config.obs_dim).astype(np.float32)
        )
        reward = float(1.0 - action_penalty)
        done = self.steps >= self.max_steps
        obs_dict = {
            name: self.obs[i] for i, name in enumerate(self.agent_names)
        }
        rewards = {name: reward for name in self.agent_names}
        dones = {name: done for name in self.agent_names}
        truncated = {name: done for name in self.agent_names}
        infos = {name: {} for name in self.agent_names}
        return obs_dict, rewards, dones, truncated, infos

    def close(self) -> None:
        return None


def stack_agent_obs(agent_order: List[str], obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([obs_dict[agent_id] for agent_id in agent_order], axis=0).astype(np.float32)


def build_global_state(obs_array: np.ndarray, config: IMAPPOConfig) -> np.ndarray:
    flat_state = obs_array.reshape(-1)
    if flat_state.shape[0] == config.state_dim:
        return flat_state.astype(np.float32)
    if flat_state.shape[0] > config.state_dim:
        return flat_state[: config.state_dim].astype(np.float32)
    padded = np.zeros(config.state_dim, dtype=np.float32)
    padded[: flat_state.shape[0]] = flat_state
    return padded


def infer_agent_order(env, obs_data, config: IMAPPOConfig) -> List[str]:
    if hasattr(env, "possible_agents"):
        return list(env.possible_agents)
    if hasattr(env, "agents") and getattr(env, "agents"):
        return list(env.agents)
    if hasattr(env, "_env") and hasattr(env._env, "agents") and env._env.agents:
        return list(env._env.agents)
    if isinstance(obs_data, dict):
        return list(obs_data.keys())
    if isinstance(obs_data, (list, tuple)):
        return [f"uav_{i}" for i in range(len(obs_data))]
    return [f"uav_{i}" for i in range(config.n_agents)]


def normalise_obs(agent_order: List[str], obs_data) -> np.ndarray:
    if isinstance(obs_data, dict):
        return stack_agent_obs(agent_order, obs_data)
    if isinstance(obs_data, (list, tuple)):
        return np.stack(obs_data, axis=0).astype(np.float32)
    raise TypeError(f"Unsupported observation type: {type(obs_data)}")


def env_reset(env):
    reset_out = env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        return reset_out
    return reset_out, {}


def env_step(env, agent_order: List[str], actions: np.ndarray):
    try:
        step_out = env.step({agent_id: actions[i] for i, agent_id in enumerate(agent_order)})
        used_dict_actions = True
    except Exception:
        step_out = env.step(list(actions))
        used_dict_actions = False

    obs, rewards, dones, truncated, infos = step_out

    if isinstance(rewards, dict):
        reward_vec = np.array([rewards[agent_id] for agent_id in agent_order], dtype=np.float32)
    else:
        reward_vec = np.array(rewards, dtype=np.float32)

    if isinstance(dones, dict):
        done = all(dones[agent_id] for agent_id in agent_order)
    else:
        done = bool(dones)

    if isinstance(truncated, dict):
        truncated_done = all(truncated[agent_id] for agent_id in agent_order)
    else:
        truncated_done = bool(truncated)

    if not used_dict_actions and isinstance(infos, dict):
        infos = infos

    return obs, reward_vec, done, truncated_done, infos


def train_imappo(
    env_factory: Optional[Callable[[], object]] = None,
    config: Optional[IMAPPOConfig] = None,
    logger=None,
) -> Tuple[IMAPPO, List[Dict[str, float]]]:
    cfg = config or IMAPPOConfig()
    algo = IMAPPO(cfg)
    buffer = RolloutBuffer()
    logs: List[Dict[str, float]] = []

    env = env_factory() if env_factory is not None else MockContinuousUAVEnv(cfg)

    for episode in range(cfg.max_episodes):
        obs_data, _ = env_reset(env)
        agent_order = infer_agent_order(env, obs_data, cfg)
        obs_array = normalise_obs(agent_order, obs_data)
        state_array = build_global_state(obs_array, cfg)
        intent, episode_mask = algo.sample_episode_intent_and_mask()

        episode_return = 0.0
        for _ in range(cfg.max_steps):
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=algo.device)
            state_tensor = torch.tensor(state_array, dtype=torch.float32, device=algo.device)
            actions, log_probs = algo.select_actions(obs_tensor, intent, episode_mask)

            action_np = actions.detach().cpu().numpy()
            next_obs_data, reward_vec, done_flag, truncated_flag, _ = env_step(
                env, agent_order, action_np
            )
            next_obs_array = normalise_obs(agent_order, next_obs_data)
            next_state_array = build_global_state(next_obs_array, cfg)

            extrinsic_rewards = torch.tensor(reward_vec, dtype=torch.float32, device=algo.device)
            next_state_tensor = torch.tensor(
                next_state_array, dtype=torch.float32, device=algo.device
            )
            total_reward, _ = algo.compute_shaped_rewards(
                extrinsic_rewards, state_tensor, next_state_tensor, intent
            )

            done = done_flag or truncated_flag
            done_tensor = torch.tensor(float(done), dtype=torch.float32, device=algo.device)

            buffer.add(
                state=state_tensor,
                obs=obs_tensor,
                action=actions,
                action_mask=episode_mask,
                intent=intent,
                reward=total_reward,
                done=done_tensor,
                log_prob=log_probs,
                next_state=next_state_tensor,
                next_obs=torch.tensor(next_obs_array, dtype=torch.float32, device=algo.device),
            )

            episode_return += float(total_reward.mean().item())
            obs_array = next_obs_array
            state_array = next_state_array

            if buffer.is_ready(cfg.rollout_length):
                update_log = algo.update(buffer)
                update_log["episode"] = float(episode)
                logs.append(update_log)
                if logger is not None:
                    for key, value in update_log.items():
                        if key != "episode":
                            logger.log_stat(key, value, int(episode))

            if done:
                break

        episode_log = {"episode": float(episode), "episode_return": episode_return}
        logs.append(episode_log)
        if logger is not None:
            logger.log_stat("episode", episode, episode)
            logger.log_stat("episode_return", episode_return, episode)

    env.close()
    return algo, logs


def build_imappo_config_from_args(args) -> IMAPPOConfig:
    config = IMAPPOConfig(
        n_agents=getattr(args, "imappo_n_agents", 4),
        obs_dim=getattr(args, "imappo_obs_dim", 18),
        state_dim=getattr(args, "imappo_state_dim", 72),
        action_dim=getattr(args, "imappo_action_dim", 3),
        intent_dim=getattr(args, "intent_dim", 8),
        gamma=args.gamma,
        gae_lambda=getattr(args, "gae_lambda", 0.95),
        eps_clip=getattr(args, "eps_clip", 0.2),
        eta=getattr(args, "eta", 0.5),
        entropy_coef=getattr(args, "entropy_coef", 1e-3),
        value_coef=getattr(args, "value_coef", 0.5),
        max_grad_norm=args.grad_norm_clip,
        actor_lr=args.lr,
        critic_lr=getattr(args, "critic_lr", args.lr),
        potential_lr=getattr(args, "potential_lr", args.lr),
        potential_update_mode=getattr(args, "potential_update_mode", "normal"),
        potential_update_interval=getattr(args, "potential_update_interval", 4),
        ppo_epochs=getattr(args, "epochs", 4),
        minibatch_size=args.batch_size,
        rollout_length=getattr(args, "rollout_length", args.batch_size),
        max_episodes=getattr(args, "max_episodes", 100),
        max_steps=args.env_args.get("time_limit", getattr(args, "max_steps", 200)),
        action_low=getattr(args, "action_low", -1.0),
        action_high=getattr(args, "action_high", 1.0),
        device=args.device,
        seed=args.seed,
    )
    return config


def make_pettingzoo_env_from_args(args):
    # Importing envs.gymma triggers PettingZoo registration side effects used by gymnasium.
    from envs import gymma as _gymma  # noqa: F401

    env_args = dict(args.env_args)
    key = env_args.pop("key")
    env_args.pop("time_limit", None)
    env_args.pop("pretrained_wrapper", None)
    env_args.pop("seed", None)
    if not str(key).startswith("pz-"):
        env_args.pop("continuous_actions", None)
    return gym.make(key, **env_args)


def run_imappo_experiment(args, logger):
    config = build_imappo_config_from_args(args)
    env_factory = None
    if args.env == "gymma" and args.env_args.get("key"):
        env_factory = lambda: make_pettingzoo_env_from_args(args)

    _, logs = train_imappo(env_factory=env_factory, config=config, logger=logger)
    if logger is not None and logger.stats.get("episode"):
        logger.print_recent_stats()
    return logs


if __name__ == "__main__":
    trainer, training_logs = train_imappo()
    print("Completed I-MAPPO smoke run.")
    print(training_logs[-5:])
