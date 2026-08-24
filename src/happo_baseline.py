"""HAPPO baseline with independent actors and sequential factor updates.

The update follows the sequential scheme in Kuba et al. (ICLR 2022) and the
official PKU-MARL/HARL runner: agents never share actor parameters; each agent
finishes its PPO epochs before the next begins; the likelihood ratio of every
updated predecessor multiplies the next agent's surrogate advantage.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from imappo import (
    IMAPPO,
    IMAPPOConfig,
    IntentConditionedActor,
    RolloutBuffer,
    Tensor,
)


class HAPPOBaseline(IMAPPO):
    """Continuous-action HAPPO with a centralized value function."""

    def __init__(self, config: IMAPPOConfig):
        if config.algorithm != "happo":
            raise ValueError("HAPPOBaseline requires algorithm='happo'")
        if config.critic_mode != "mlp":
            raise ValueError("HAPPO requires a centralized MLP critic")
        if config.intent_source != "none":
            raise ValueError("HAPPO baseline must not receive intent conditioning")
        if config.use_action_mask:
            raise ValueError("HAPPO baseline must use a full action mask")
        if config.policy_mode != "direct":
            raise ValueError("HAPPO baseline supports direct policies only")
        if config.safety_filter_mode != "none":
            raise ValueError("HAPPO baseline must not receive a controller safety filter")
        super().__init__(config)
        first_actor = self.actor
        first_optimizer = self.actor_optim
        actors: List[IntentConditionedActor] = [first_actor]
        optimizers = [first_optimizer]
        for _ in range(1, config.n_agents):
            actor = IntentConditionedActor(config).to(self.device)
            actors.append(actor)
            optimizers.append(
                torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
            )
        self.actor = nn.ModuleList(actors)
        self.actor_optims = optimizers
        # Prevent generic code from silently treating HAPPO as one shared actor.
        self.actor_optim = None
        self.last_agent_order: List[int] = list(range(config.n_agents))

    def algorithm_metadata(self) -> Dict[str, object]:
        return {
            "algorithm": "happo",
            "actor_parameter_sharing": "independent",
            "actor_count": len(self.actor),
            "update_scheme": "random_sequential_likelihood_factor",
            "critic": "centralized_mlp",
            "intent_conditioning": False,
            "action_masking": False,
            "safety_filter": "none",
        }

    def select_actions(
        self,
        obs: Tensor,
        intent: Tensor,
        action_mask: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        if int(obs.size(0)) != self.config.n_agents:
            raise ValueError(
                "HAPPO independent actors require the registered training agent count"
            )
        actions = []
        log_probs = []
        policy_latents = []
        for agent_id, actor in enumerate(self.actor):
            action, log_prob, latent = actor.sample_action(
                obs[agent_id : agent_id + 1],
                intent.unsqueeze(0),
                action_mask[agent_id : agent_id + 1],
                deterministic=deterministic,
            )
            actions.append(action)
            log_probs.append(log_prob)
            policy_latents.append(latent)
        joint_actions = torch.cat(actions, dim=0)
        self._last_base_actions = torch.zeros_like(joint_actions)
        self._last_policy_latents = torch.cat(policy_latents, dim=0).detach()
        self._last_residual_actions = joint_actions.detach()
        self._last_safety_filter_correction = torch.zeros_like(joint_actions)
        self._last_safety_filter_profile = None
        self._last_safety_solver_diagnostics = {}
        return joint_actions, torch.cat(log_probs, dim=0)

    def _agent_log_prob(
        self,
        agent_id: int,
        obs: Tensor,
        intents: Tensor,
        masks: Tensor,
        actions: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        actor = self.actor[agent_id]
        dist, _ = actor.distribution(obs, intents, masks)
        return actor.log_prob(dist, actions, masks), dist.entropy().sum(dim=-1)

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
        old_values = values.detach()
        team_rewards = rewards.mean(dim=-1)
        advantages, returns = self.compute_gae(
            team_rewards, dones, old_values, next_values.detach()
        )
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        batch_size = int(states.size(0))
        factor = torch.ones(batch_size, dtype=states.dtype, device=self.device)
        self.last_agent_order = self.rng.permutation(
            self.config.n_agents
        ).astype(int).tolist()
        actor_losses = []
        entropies = []

        for agent_id in self.last_agent_order:
            actor = self.actor[agent_id]
            optimizer = self.actor_optims[agent_id]
            registered_old_log_prob = old_log_probs[:, agent_id].detach()
            agent_factor = factor.detach().clone()
            flat_indices = np.arange(batch_size)
            for _ in range(self.config.ppo_epochs):
                self.rng.shuffle(flat_indices)
                for start in range(0, batch_size, self.config.minibatch_size):
                    idx = flat_indices[start : start + self.config.minibatch_size]
                    new_log_prob, entropy_by_sample = self._agent_log_prob(
                        agent_id,
                        obs[idx, agent_id],
                        intents[idx],
                        action_masks[idx, agent_id],
                        actions[idx, agent_id],
                    )
                    ratio = torch.exp(new_log_prob - registered_old_log_prob[idx])
                    surrogate_unclipped = ratio * advantages[idx]
                    surrogate_clipped = torch.clamp(
                        ratio,
                        1.0 - self.config.eps_clip,
                        1.0 + self.config.eps_clip,
                    ) * advantages[idx]
                    entropy = entropy_by_sample.mean()
                    policy_loss = -(
                        agent_factor[idx]
                        * torch.minimum(surrogate_unclipped, surrogate_clipped)
                    ).mean()
                    actor_loss = policy_loss - self.current_entropy_coef * entropy
                    optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(
                        actor.parameters(), self.config.max_grad_norm
                    )
                    optimizer.step()
                    actor_losses.append(float(policy_loss.item()))
                    entropies.append(float(entropy.item()))

            with torch.no_grad():
                updated_log_prob, _ = self._agent_log_prob(
                    agent_id,
                    obs[:, agent_id],
                    intents,
                    action_masks[:, agent_id],
                    actions[:, agent_id],
                )
                factor = factor * torch.exp(
                    updated_log_prob - registered_old_log_prob
                )
            if not bool(torch.isfinite(factor).all().item()):
                raise FloatingPointError("HAPPO sequential likelihood factor is non-finite")

        last_critic_loss = 0.0
        flat_indices = np.arange(batch_size)
        for _ in range(self.config.ppo_epochs):
            self.rng.shuffle(flat_indices)
            for start in range(0, batch_size, self.config.minibatch_size):
                idx = flat_indices[start : start + self.config.minibatch_size]
                critic_values, _ = self.critic(states[idx], intents[idx], obs[idx])
                clipped_values = old_values[idx] + torch.clamp(
                    critic_values - old_values[idx],
                    -self.config.value_clip,
                    self.config.value_clip,
                )
                critic_loss = 0.5 * torch.maximum(
                    (critic_values - returns[idx]).pow(2),
                    (clipped_values - returns[idx]).pow(2),
                ).mean()
                self.critic_optim.zero_grad()
                (self.config.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm
                )
                self.critic_optim.step()
                last_critic_loss = float(critic_loss.item())

        factor_mean = float(factor.mean().item())
        factor_max = float(factor.abs().max().item())
        buffer.clear()
        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": last_critic_loss,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "potential_loss": 0.0,
            "return_mean": float(team_rewards.mean().item()),
            "eta": 0.0,
            "entropy_coef": float(self.current_entropy_coef),
            "happo_factor_mean": factor_mean,
            "happo_factor_abs_max": factor_max,
            "happo_first_agent": float(self.last_agent_order[0]),
        }

    def save_checkpoint(
        self, path: str, extra: Optional[Dict[str, object]] = None
    ) -> None:
        config_fields = {field.name for field in fields(IMAPPOConfig)}
        torch.save(
            {
                "checkpoint_schema": "happo_independent_actors_v1",
                "config": {
                    key: value
                    for key, value in self.config.__dict__.items()
                    if key in config_fields
                },
                "actors": [actor.state_dict() for actor in self.actor],
                "critic": self.critic.state_dict(),
                "actor_optims": [optimizer.state_dict() for optimizer in self.actor_optims],
                "critic_optim": self.critic_optim.state_dict(),
                "current_entropy_coef": self.current_entropy_coef,
                "rng_state": self.rng.bit_generator.state,
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls, path: str, device: Optional[str] = None
    ) -> "HAPPOBaseline":
        checkpoint = torch.load(
            path, map_location=device or "cpu", weights_only=False
        )
        if checkpoint.get("checkpoint_schema") != "happo_independent_actors_v1":
            raise ValueError("checkpoint is not an independent-actor HAPPO checkpoint")
        config_dict = dict(checkpoint["config"])
        if device is not None:
            config_dict["device"] = device
        algo = cls(IMAPPOConfig(**config_dict))
        if len(checkpoint["actors"]) != len(algo.actor):
            raise ValueError("HAPPO checkpoint actor count mismatch")
        for actor, state in zip(algo.actor, checkpoint["actors"]):
            actor.load_state_dict(state)
        algo.critic.load_state_dict(checkpoint["critic"])
        for optimizer, state in zip(algo.actor_optims, checkpoint["actor_optims"]):
            optimizer.load_state_dict(state)
        algo.critic_optim.load_state_dict(checkpoint["critic_optim"])
        algo.current_entropy_coef = checkpoint.get(
            "current_entropy_coef", algo.config.entropy_coef
        )
        if "rng_state" in checkpoint:
            algo.rng.bit_generator.state = checkpoint["rng_state"]
        return algo
