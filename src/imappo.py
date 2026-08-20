from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import time
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym

from envs.uav_scheduling_env import infer_obs_dim, infer_state_dim
from intent_semantic_encoder import (
    DEFAULT_INTENT_DESCRIPTIONS,
    IntentLibrary,
    infer_intent_posture,
)
from objective_semantic_adapter import ObjectiveSemanticAdapter


Tensor = torch.Tensor


@dataclass
class IMAPPOConfig:
    algorithm: str = "imappo"
    critic_mode: str = "attention"
    use_action_mask: bool = True
    intent_source: str = "onehot"
    policy_mode: str = "direct"
    residual_action_scale: float = 0.25
    residual_initial_log_std: float = -2.0
    rule_prior_context: str = "neutral"
    safety_filter_mode: str = "none"
    cbf_base_min_distance: float = 1.0
    cbf_iterations: int = 4
    cbf_solver_tolerance: float = 1e-7
    cbf_solver_max_iterations: int = 100
    replay_capacity: int = 100_000
    matd3_warmup_steps: int = 200
    matd3_exploration_noise: float = 0.10
    matd3_policy_noise: float = 0.20
    matd3_noise_clip: float = 0.50
    matd3_policy_delay: int = 2
    matd3_tau: float = 0.005

    n_agents: int = 8
    n_targets: int = 6
    obs_dim: int = 30
    state_dim: int = 240
    action_dim: int = 3
    intent_dim: int = 64
    intent_library_path: str = ""
    intent_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    intent_encoder_revision: str = ""
    intent_encoder_batch_size: int = 32
    intent_projection_seed: int = 0
    intent_code_seed: int = 0
    intent_encoder_device: str = ""
    intent_train_labels: Tuple[str, ...] = ()
    intent_adapter_ridge: float = 0.01
    intent_profile_decoder: str = "dual_ridge"
    intent_nli_model: str = "cross-encoder/nli-deberta-v3-small"
    intent_nli_model_revision: str = ""
    intent_nli_batch_size: int = 32
    intent_semantic_weight: float = 1.0
    intent_objective_weight: float = 1.0
    align_intent_posture: bool = True

    gamma: float = 0.99
    gae_lambda: float = 0.95
    eps_clip: float = 0.1
    eta: float = 0.5
    eta_end: float = 0.1
    entropy_coef: float = 1e-3
    entropy_coef_end: float = 1e-4
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    value_clip: float = 0.2

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
    eval_interval: int = 10
    eval_episodes: int = 3
    curriculum_spawn_scale_start: float = 0.45
    curriculum_spawn_scale_end: float = 0.30
    curriculum_separation_start: float = 0.95
    curriculum_separation_end: float = 1.20
    eval_spawn_scale: float = 0.34
    eval_separation_scale: float = 0.95
    collision_probe_spawn_scale: float = 0.29
    collision_probe_separation_scale: float = 0.82
    hard_train_interval: int = 6
    hard_train_spawn_scale: float = 0.31
    hard_train_separation_scale: float = 0.86
    safety_reward_coef: float = 1.0
    intent_reward_profiles_enabled: bool = False
    wind_std: float = 0.0
    observation_noise_std: float = 0.0
    action_delay_steps: int = 0
    communication_dropout_prob: float = 0.0

    actor_hidden_dims: Tuple[int, int, int] = (256, 256, 128)
    critic_hidden_dims: Tuple[int, int, int] = (256, 256, 128)
    feature_hidden_dim: int = 128
    attention_dim: int = 128

    action_low: float = -1.0
    action_high: float = 1.0
    log_std_min: float = -5.0
    log_std_max: float = 0.5
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
        if config.policy_mode == "residual_rule":
            nn.init.zeros_(self.mean_head.weight)
            nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(
            self.log_std_head.bias,
            config.residual_initial_log_std if config.policy_mode == "residual_rule" else -1.0,
        )

    def forward(
        self,
        obs: Tensor,
        intent: Tensor,
        action_mask: Optional[Tensor] = None,
        base_action: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x = torch.cat([obs, intent], dim=-1)
        hidden = self.backbone(x)
        action_mean = self.mean_head(hidden)
        if base_action is not None and self.config.policy_mode != "residual_rule":
            raise ValueError("base_action is only valid for residual_rule policies")
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
        base_action: Optional[Tensor] = None,
    ) -> Tuple[Normal, Tensor]:
        action_mean, action_log_std, hidden = self.forward(
            obs, intent, action_mask, base_action
        )
        action_std = action_log_std.exp()
        return Normal(action_mean, action_std), hidden

    def sample_action(
        self,
        obs: Tensor,
        intent: Tensor,
        action_mask: Optional[Tensor] = None,
        deterministic: bool = False,
        base_action: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        dist, hidden = self.distribution(obs, intent, action_mask, base_action)
        if deterministic:
            raw_action = dist.mean
        else:
            raw_action = dist.rsample()
        squashed_action = torch.tanh(raw_action)
        if base_action is None:
            action = squashed_action * self.config.action_high
            log_prob = self.log_prob_from_raw_action(
                dist, raw_action, squashed_action, action_mask
            )
        else:
            bounded_base = torch.clamp(
                base_action / max(self.config.action_high, 1e-6), -1.0, 1.0
            )
            # A signed headroom map preserves the exact rule action at zero while
            # retaining an inward gradient even when the rule action is saturated.
            headroom = torch.where(
                squashed_action >= 0.0,
                1.0 - bounded_base,
                1.0 + bounded_base,
            )
            action = (
                bounded_base
                + self.config.residual_action_scale * squashed_action * headroom
            ) * self.config.action_high
            log_prob = self.latent_log_prob(dist, raw_action, action_mask)
        if action_mask is not None:
            action = action * action_mask
        return action, log_prob, raw_action

    @staticmethod
    def latent_log_prob(
        dist: Normal,
        latent_action: Tensor,
        action_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Log density of the residual latent variable used for PPO ratios."""
        log_prob = dist.log_prob(latent_action)
        if action_mask is not None:
            log_prob = log_prob * action_mask
        return log_prob.sum(dim=-1)

    def log_prob(
        self,
        dist: Normal,
        actions: Tensor,
        action_mask: Optional[Tensor] = None,
    ) -> Tensor:
        scaled_actions = torch.clamp(
            actions / max(self.config.action_high, 1e-6),
            -0.999999,
            0.999999,
        )
        raw_action = torch.atanh(scaled_actions)
        return self.log_prob_from_raw_action(dist, raw_action, scaled_actions, action_mask)

    def log_prob_from_raw_action(
        self,
        dist: Normal,
        raw_action: Tensor,
        squashed_action: Tensor,
        action_mask: Optional[Tensor] = None,
    ) -> Tensor:
        correction = torch.log1p(-squashed_action.pow(2) + 1e-6)
        log_prob = dist.log_prob(raw_action) - correction
        if action_mask is not None:
            log_prob = log_prob * action_mask
        return log_prob.sum(dim=-1)


class CrossAttentionCritic(nn.Module):
    def __init__(self, config: IMAPPOConfig):
        super().__init__()
        self.config = config
        h1, h2, h3 = config.critic_hidden_dims
        if config.critic_mode == "local":
            self.local_value_head = nn.Sequential(
                nn.Linear(config.obs_dim + config.intent_dim, h1),
                nn.ReLU(),
                nn.Linear(h1, h2),
                nn.ReLU(),
                nn.Linear(h2, h3),
                nn.ReLU(),
                nn.Linear(h3, 1),
            )
            self.mlp_value_head = None
            self.agent_feature_extractor = None
            self.query = None
            self.key = None
            self.value = None
            self.value_head = None
            return
        if config.critic_mode == "mlp":
            self.mlp_value_head = nn.Sequential(
                nn.Linear(config.state_dim + config.intent_dim, h1),
                nn.ReLU(),
                nn.Linear(h1, h2),
                nn.ReLU(),
                nn.Linear(h2, h3),
                nn.ReLU(),
                nn.Linear(h3, 1),
            )
            self.agent_feature_extractor = None
            self.query = None
            self.key = None
            self.value = None
            self.value_head = None
            return

        self.agent_feature_extractor = nn.Sequential(
            nn.Linear(config.obs_dim, config.feature_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.feature_hidden_dim, config.feature_hidden_dim),
            nn.ReLU(),
        )
        self.query = nn.Linear(config.intent_dim, config.attention_dim)
        self.key = nn.Linear(config.feature_hidden_dim, config.attention_dim)
        self.value = nn.Linear(config.feature_hidden_dim, config.feature_hidden_dim)
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
        if self.config.critic_mode == "local":
            batch_size, n_agents, _ = obs_all_agents.shape
            expanded_intent = intent.unsqueeze(1).expand(-1, n_agents, -1)
            local_input = torch.cat([obs_all_agents, expanded_intent], dim=-1)
            values = self.local_value_head(
                local_input.reshape(-1, local_input.size(-1))
            ).reshape(batch_size, n_agents)
            attention_weights = torch.eye(
                n_agents, dtype=obs_all_agents.dtype, device=obs_all_agents.device
            ).unsqueeze(0).expand(batch_size, -1, -1)
            return values, attention_weights
        if self.config.critic_mode == "mlp":
            critic_input = torch.cat([state, intent], dim=-1)
            batch_size = state.size(0)
            attention_weights = torch.full(
                (batch_size, self.config.n_agents),
                1.0 / self.config.n_agents,
                dtype=state.dtype,
                device=state.device,
            )
            return self.mlp_value_head(critic_input).squeeze(-1), attention_weights

        agent_features = self.encode_agent_features(obs_all_agents)
        value = self.value(agent_features)
        if self.config.critic_mode == "uniform":
            batch_size = obs_all_agents.size(0)
            attention_weights = torch.full(
                (batch_size, 1, self.config.n_agents),
                1.0 / self.config.n_agents,
                dtype=obs_all_agents.dtype,
                device=obs_all_agents.device,
            )
        else:
            query = self.query(intent).unsqueeze(1)
            key = self.key(agent_features)
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
            "base_actions": [],
            "policy_latents": [],
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
        base_action: Tensor,
        policy_latent: Tensor,
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
        self.storage["base_actions"].append(base_action.detach().cpu())
        self.storage["policy_latents"].append(policy_latent.detach().cpu())
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
        if config.policy_mode not in {"direct", "residual_rule"}:
            raise ValueError(f"unsupported policy_mode: {config.policy_mode}")
        if config.critic_mode not in {"attention", "uniform", "mlp", "local"}:
            raise ValueError(f"unsupported critic_mode: {config.critic_mode}")
        if config.rule_prior_context not in {
            "neutral", "oracle_posture", "intent_retrieval", "objective_profile",
            "oracle_profile",
        }:
            raise ValueError(f"unsupported rule_prior_context: {config.rule_prior_context}")
        if config.intent_profile_decoder not in {
            "none",
            "dual_ridge", "concept_anchor", "contrastive_anchor", "prototype_ridge",
            "augmented_prototype_ridge",
            "nli_entailment",
            "nli_relevance_gated",
            "nli_similarity_gated",
            "nli_prototype_gated",
            "polarity_prototype",
        }:
            raise ValueError(
                f"unsupported intent_profile_decoder: {config.intent_profile_decoder}"
            )
        if config.safety_filter_mode not in {"none", "pairwise_cbf", "pairwise_qp"}:
            raise ValueError(
                f"unsupported safety_filter_mode: {config.safety_filter_mode}"
            )
        if config.residual_action_scale < 0.0:
            raise ValueError("residual_action_scale must be non-negative")
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.intent_library: Optional[IntentLibrary] = None
        self.task_intent_library: Optional[IntentLibrary] = None
        self.objective_semantic_adapter: Optional[ObjectiveSemanticAdapter] = None
        self._profile_cache_vectors: List[np.ndarray] = []
        self._profile_cache_values: List[np.ndarray] = []
        self._eval_intent_cache: Optional[Dict[str, Tensor]] = None
        self.current_objective_profile: Optional[Dict[str, float]] = None
        if config.algorithm not in {"mappo", "ippo"} and config.intent_source in {
            "onehot",
            "semantic_library",
            "legacy_hash",
            "random_dense",
            "pretrained_semantic",
            "objective_grounded_semantic",
        }:
            self._init_intent_library()
        elif config.algorithm in {"mappo", "ippo"} or config.intent_source == "none":
            self.task_intent_library = IntentLibrary.create_onehot()
            if config.intent_train_labels:
                self.task_intent_library = self.task_intent_library.subset_by_labels(
                    config.intent_train_labels
                )

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
        self.current_eta = config.eta
        self.current_entropy_coef = config.entropy_coef
        self.current_posture = "neutral"
        self._last_base_actions = torch.zeros(
            config.n_agents,
            config.action_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self._last_policy_latents = torch.zeros_like(self._last_base_actions)

        if config.potential_update_mode == "frozen":
            for param in self.potential.parameters():
                param.requires_grad = False

    def _init_intent_library(self) -> None:
        """Load or create an explicitly typed intent representation library."""
        lib_path = self.config.intent_library_path
        already_subset = False
        if lib_path and Path(lib_path).with_suffix(".npz").exists():
            self.intent_library = IntentLibrary.load(lib_path)
        elif self.config.intent_source in {"semantic_library", "legacy_hash"}:
            if self.config.intent_source == "semantic_library":
                import warnings

                warnings.warn(
                    "intent_source='semantic_library' is a legacy alias for non-semantic hash vectors; "
                    "use 'legacy_hash' for reproduction or 'pretrained_semantic' for paper experiments.",
                    FutureWarning,
                    stacklevel=2,
                )
            self.intent_library = IntentLibrary.create_legacy_hash(
                intent_dim=self.config.intent_dim,
            )
        elif self.config.intent_source == "random_dense":
            self.intent_library = IntentLibrary.create_random_dense(
                intent_dim=self.config.intent_dim,
                seed=self.config.intent_code_seed,
            )
        elif self.config.intent_source == "onehot":
            self.intent_library = IntentLibrary.create_onehot(
                intent_dim=self.config.intent_dim,
            )
        elif self.config.intent_source == "objective_grounded_semantic":
            catalog = dict(DEFAULT_INTENT_DESCRIPTIONS)
            train_labels = list(self.config.intent_train_labels) or list(catalog)
            missing = [label for label in train_labels if label not in catalog]
            if missing:
                raise ValueError(f"unknown objective adapter training labels: {missing}")
            entries = [(label, catalog[label]) for label in train_labels]
            self.objective_semantic_adapter = ObjectiveSemanticAdapter.fit(
                entries,
                intent_dim=self.config.intent_dim,
                model_name=self.config.intent_encoder_model,
                model_revision=self.config.intent_encoder_revision or None,
                projection_seed=self.config.intent_projection_seed,
                ridge=self.config.intent_adapter_ridge,
                semantic_weight=self.config.intent_semantic_weight,
                objective_weight=self.config.intent_objective_weight,
                batch_size=self.config.intent_encoder_batch_size,
                device=self.config.intent_encoder_device or None,
                profile_decoder=self.config.intent_profile_decoder,
                nli_model_name=self.config.intent_nli_model,
                nli_model_revision=self.config.intent_nli_model_revision,
                nli_batch_size=self.config.intent_nli_batch_size,
            )
            metadata = self.objective_semantic_adapter.metadata(train_labels)
            metadata.update(
                {
                    "source": "prebuilt",
                    "domain": "uav",
                    "postures": [
                        infer_intent_posture(label, description)
                        for label, description in entries
                    ],
                }
            )
            adapter_vectors = self.objective_semantic_adapter.encode_entries(entries)
            adapter_profiles = self.objective_semantic_adapter.predict_profiles(entries)
            self._profile_cache_vectors.extend(adapter_vectors)
            self._profile_cache_values.extend(adapter_profiles)
            self.intent_library = IntentLibrary(
                vectors=adapter_vectors,
                labels=train_labels,
                descriptions=[description for _, description in entries],
                metadata=metadata,
            )
            already_subset = bool(self.config.intent_train_labels)
        elif self.config.intent_source == "pretrained_semantic":
            self.intent_library = IntentLibrary.create_pretrained(
                intent_dim=self.config.intent_dim,
                model_name=self.config.intent_encoder_model,
                model_revision=self.config.intent_encoder_revision or None,
                batch_size=self.config.intent_encoder_batch_size,
                projection_seed=self.config.intent_projection_seed,
                device=self.config.intent_encoder_device or None,
            )
        else:
            raise ValueError(f"unsupported intent_source for library: {self.config.intent_source}")
        if self.intent_library.intent_dim != self.config.intent_dim:
            raise ValueError(
                f"Intent library dim ({self.intent_library.intent_dim}) "
                f"does not match config intent_dim ({self.config.intent_dim})"
            )
        if self.config.intent_train_labels and not already_subset:
            self.intent_library = self.intent_library.subset_by_labels(
                self.config.intent_train_labels
            )
        expected_type = {
            "onehot": "onehot",
            "semantic_library": "legacy_hash",
            "legacy_hash": "legacy_hash",
            "random_dense": "random_dense",
            "pretrained_semantic": "pretrained_semantic",
            "objective_grounded_semantic": "objective_grounded_semantic",
        }[self.config.intent_source]
        actual_type = self.intent_library.metadata.get("representation_type")
        if actual_type is None:
            legacy_model = self.intent_library.metadata.get("embed_model")
            if expected_type == "legacy_hash" and legacy_model in {"static_hash", "sha256_prng"}:
                actual_type = "legacy_hash"
                self.intent_library.metadata["representation_type"] = actual_type
                self.intent_library.metadata["semantic_geometry"] = False
            else:
                raise ValueError(
                    "loaded intent library has no representation_type metadata; "
                    "regenerate it with the current research pipeline"
                )
        if actual_type is not None and actual_type != expected_type:
            raise ValueError(
                f"loaded intent library type ({actual_type}) does not match "
                f"intent_source ({self.config.intent_source})"
            )

    def encode_intent_queries(
        self,
        entries: List[Tuple[str, str]],
    ) -> Tensor:
        """Encode evaluation-only intent texts without adding them to training.

        Text-based representations consume the query wording. Identity controls use
        the canonical label and therefore act as an oracle for paraphrase identity;
        held-out labels activate coordinates/codes never observed during training.
        """
        if not entries:
            return torch.empty((0, self.config.intent_dim), device=self.device)
        if self.config.algorithm in {"mappo", "ippo"}:
            return torch.zeros((len(entries), self.config.intent_dim), device=self.device)
        if self.config.intent_source == "none":
            return torch.zeros((len(entries), self.config.intent_dim), device=self.device)
        if self.config.intent_source == "objective_grounded_semantic":
            if self.objective_semantic_adapter is None:
                raise RuntimeError("objective semantic adapter is not initialized")
            vectors = self.objective_semantic_adapter.encode_entries(entries)
            profiles = self.objective_semantic_adapter.predict_profiles(entries)
            self._profile_cache_vectors.extend(vectors)
            self._profile_cache_values.extend(profiles)
        elif self.config.intent_source == "pretrained_semantic":
            query_library = IntentLibrary.create_pretrained(
                intent_dim=self.config.intent_dim,
                descriptions=entries,
                model_name=self.config.intent_encoder_model,
                model_revision=self.config.intent_encoder_revision or None,
                batch_size=self.config.intent_encoder_batch_size,
                projection_seed=self.config.intent_projection_seed,
                device=self.config.intent_encoder_device or None,
            )
            vectors = query_library.vectors
        elif self.config.intent_source in {"semantic_library", "legacy_hash"}:
            vectors = IntentLibrary.create_legacy_hash(
                intent_dim=self.config.intent_dim,
                descriptions=entries,
            ).vectors
        elif self.config.intent_source == "random_dense":
            full_library = IntentLibrary.create_random_dense(
                intent_dim=self.config.intent_dim,
                seed=self.config.intent_code_seed,
            )
            resolved = [
                full_library.get_by_label(label) for label, _ in entries
            ]
            if any(vector is None for vector in resolved):
                raise ValueError("intent query contains a label outside the configured catalog")
            vectors = np.stack(resolved)
        elif self.config.intent_source == "onehot":
            full_library = IntentLibrary.create_onehot(intent_dim=self.config.intent_dim)
            resolved = [
                full_library.get_by_label(label) for label, _ in entries
            ]
            if any(vector is None for vector in resolved):
                raise ValueError("intent query contains a label outside the configured catalog")
            vectors = np.stack(resolved)
        else:
            raise ValueError(f"unsupported intent query source: {self.config.intent_source}")
        return torch.as_tensor(vectors, dtype=torch.float32, device=self.device)

    def intent_representation_metadata(self) -> Dict[str, object]:
        """Return metadata that must accompany every experiment result."""
        if self.config.algorithm in {"mappo", "ippo"} or self.config.intent_source == "none":
            return {
                "representation_type": "none",
                "semantic_geometry": False,
                "intent_conditioning": False,
                "task_labels_hidden_from_actor_and_critic": True,
                "task_posture_exposed_via_action_mask": bool(
                    self.config.intent_source == "none" and self.config.use_action_mask
                ),
                "policy_mode": self.config.policy_mode,
                "navigation_prior": (
                    "target_tracking_plus_neighbor_potential_field"
                    if self.config.policy_mode == "residual_rule" else "none"
                ),
                "rule_prior_context": self.config.rule_prior_context,
                "safety_filter_mode": self.config.safety_filter_mode,
                "cbf_base_min_distance": self.config.cbf_base_min_distance,
                "cbf_iterations": self.config.cbf_iterations,
            }
        if self.intent_library is None:
            return {
                "representation_type": self.config.intent_source,
                "semantic_geometry": False,
            }
        metadata = dict(self.intent_library.metadata)
        metadata.update(
            {
                "policy_mode": self.config.policy_mode,
                "navigation_prior": (
                    "target_tracking_plus_neighbor_potential_field"
                    if self.config.policy_mode == "residual_rule" else "none"
                ),
                "residual_action_scale": (
                    self.config.residual_action_scale
                    if self.config.policy_mode == "residual_rule" else None
                ),
                "rule_prior_context": self.config.rule_prior_context,
            }
        )
        return metadata

    def set_evaluation_context(self, label: str, posture: str) -> None:
        """Set episode context used by the shared navigation prior."""
        del label
        self.current_posture = str(posture)

    def set_evaluation_objective_profile(
        self, profile: Optional[Mapping[str, float]]
    ) -> None:
        self.current_objective_profile = (
            None if profile is None else {str(key): float(value) for key, value in profile.items()}
        )

    def _rule_prior_posture(self, intent: Tensor) -> str:
        if self.config.rule_prior_context == "neutral":
            return "neutral"
        if self.config.rule_prior_context == "oracle_posture":
            return self.current_posture
        if self.intent_library is None:
            return "neutral"
        query = intent.detach().cpu().numpy().reshape(-1)
        query_norm = max(float(np.linalg.norm(query)), 1e-8)
        library = self.intent_library.vectors
        similarities = library @ query / (
            np.maximum(np.linalg.norm(library, axis=1), 1e-8) * query_norm
        )
        label = self.intent_library.labels[int(np.argmax(similarities))]
        posture = self.intent_library.posture_for_label(label)
        return posture if posture in {"attack", "stealth"} else "neutral"

    def _rule_prior_objective_profile(self, intent: Tensor) -> Optional[Dict[str, float]]:
        if self.config.rule_prior_context == "oracle_profile":
            return self.current_objective_profile
        if self.config.rule_prior_context != "objective_profile":
            return None
        if not self._profile_cache_vectors:
            return None
        from intent_objectives import OBJECTIVE_KEYS

        query = intent.detach().cpu().numpy().reshape(-1)
        vectors = np.asarray(self._profile_cache_vectors, dtype=np.float32)
        similarities = vectors @ query / (
            np.maximum(np.linalg.norm(vectors, axis=1), 1e-8)
            * max(float(np.linalg.norm(query)), 1e-8)
        )
        profile = self._profile_cache_values[int(np.argmax(similarities))]
        return {key: float(profile[index]) for index, key in enumerate(OBJECTIVE_KEYS)}

    def _action_mask_for_posture(self, posture: Optional[str]) -> Tensor:
        mask = torch.ones(
            self.config.n_agents,
            self.config.action_dim,
            device=self.device,
        )
        if not self.config.use_action_mask:
            return mask
        if str(posture).lower() == "stealth" and self.config.action_dim >= 3:
            # Preserve the historical stealth constraint for a controlled ablation.
            mask[:, 2] = 0.0
        return mask

    def _sample_intent_vector(self, rng: Optional[np.random.Generator] = None) -> Tensor:
        """Sample a single intent vector from the library as a torch Tensor."""
        vec = self.intent_library.sample_single(rng=rng)
        return torch.from_numpy(vec).to(self.device)

    def _build_eval_intent_cache(self) -> Dict[str, Tensor]:
        """Pre-compute intent vectors for evaluation modes."""
        lib = self.intent_library
        standard = lib.get_by_label("balanced")
        if standard is None:
            standard = lib.sample_single()
        dense = lib.get_by_label("safety_first")
        if dense is None:
            dense = lib.sample_single()
        attack = lib.get_by_label("aggressive_pursuit")
        if attack is None:
            attack = lib.sample_single()
        stealth = lib.get_by_label("stealth_approach")
        if stealth is None:
            stealth = lib.sample_single()
        return {
            "standard": torch.from_numpy(standard).to(self.device),
            "dense": torch.from_numpy(dense).to(self.device),
            "attack_probe": torch.from_numpy(attack).to(self.device),
            "stealth_probe": torch.from_numpy(stealth).to(self.device),
        }

    def set_training_progress(self, progress: float) -> None:
        progress = float(np.clip(progress, 0.0, 1.0))
        self.current_eta = self.config.eta + progress * (
            self.config.eta_end - self.config.eta
        )
        self.current_entropy_coef = self.config.entropy_coef + progress * (
            self.config.entropy_coef_end - self.config.entropy_coef
        )

    def sample_episode_intent_and_mask(
        self,
        tactical_posture: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor, str]:
        self.current_posture = str(tactical_posture or "neutral")
        uniform_mask = self._action_mask_for_posture(None)
        if self.config.algorithm in {"mappo", "ippo"}:
            _, labels, _ = self.task_intent_library.sample_with_info(
                1, posture=tactical_posture
            )
            return (
                torch.zeros(self.config.intent_dim, device=self.device),
                uniform_mask,
                labels[0],
            )
        if self.config.intent_source == "none":
            _, labels, _ = self.task_intent_library.sample_with_info(
                1, posture=tactical_posture
            )
            label = labels[0]
            posture = self.task_intent_library.posture_for_label(label)
            if posture == "neutral":
                posture = tactical_posture
            return (
                torch.zeros(self.config.intent_dim, device=self.device),
                self._action_mask_for_posture(posture),
                label,
            )

        # Representation-library modes use the same posture taxonomy as the environment.
        if self.intent_library is not None:
            vecs, labels, _ = self.intent_library.sample_with_info(
                1,
                posture=tactical_posture if self.config.align_intent_posture else None,
            )
            intent = torch.from_numpy(vecs[0]).to(self.device)
            label = labels[0] if labels else ""
            posture = self.intent_library.posture_for_label(label) if label else tactical_posture
            if posture == "neutral":
                posture = tactical_posture
            return intent, self._action_mask_for_posture(posture), label

        # Original one-hot mode
        intent = torch.zeros(self.config.intent_dim, device=self.device)
        if tactical_posture == "attack":
            intent_mode = 0
            label = "attack"
        elif tactical_posture == "stealth":
            intent_mode = min(1, self.config.intent_dim - 1)
            label = "stealth"
        else:
            intent_mode = int(np.random.randint(0, min(3, self.config.intent_dim)))
            label = ["attack", "stealth", "frozen"][intent_mode]
        intent[intent_mode] = 1.0
        if self.config.use_action_mask:
            mask = uniform_mask.clone()
            if intent_mode == 1:
                mask[:, 2] = 0.0
            elif intent_mode == 2:
                frozen_agents = max(1, self.config.n_agents // 4)
                selected = torch.as_tensor(
                    np.random.choice(self.config.n_agents, size=frozen_agents, replace=False),
                    device=self.device,
                    dtype=torch.long,
                )
                mask[selected, 1] = 0.0
        else:
            mask = uniform_mask
        return intent, mask, label

    def evaluation_intent_and_mask(self, mode: str = "standard") -> Tuple[Tensor, Tensor, str]:
        uniform_mask = self._action_mask_for_posture(None)
        if self.config.algorithm in {"mappo", "ippo"}:
            label = "safety_first" if mode == "dense" else "balanced"
            if self.task_intent_library.get_by_label(label) is None:
                candidates = self.task_intent_library.indices_for_posture(
                    evaluation_tactical_posture(mode)
                )
                label = self.task_intent_library.labels[int(candidates[0])]
            return torch.zeros(self.config.intent_dim, device=self.device), uniform_mask, label
        if self.config.intent_source == "none":
            label = "safety_first" if mode == "dense" else "balanced"
            if self.task_intent_library.get_by_label(label) is None:
                candidates = self.task_intent_library.indices_for_posture(
                    evaluation_tactical_posture(mode)
                )
                label = self.task_intent_library.labels[int(candidates[0])]
            posture = self.task_intent_library.posture_for_label(label)
            return (
                torch.zeros(self.config.intent_dim, device=self.device),
                self._action_mask_for_posture(posture),
                label,
            )

        # Representation-library modes use pre-cached, named evaluation intents.
        if self.intent_library is not None:
            if self._eval_intent_cache is None:
                self._eval_intent_cache = self._build_eval_intent_cache()
            cache_key = mode if mode in self._eval_intent_cache else "standard"
            intent = self._eval_intent_cache[cache_key]
            # Find the label for this cached intent
            label = ""
            for lab in ["balanced", "safety_first", "aggressive_pursuit", "stealth_approach"]:
                cached = self.intent_library.get_by_label(lab)
                if cached is not None and np.allclose(intent.cpu().numpy(), cached):
                    label = lab
                    break
            posture = self.intent_library.posture_for_label(label) if label else "neutral"
            return intent, self._action_mask_for_posture(posture), label

        # Original one-hot mode
        intent = torch.zeros(self.config.intent_dim, device=self.device)
        mask = uniform_mask.clone()
        if mode == "dense":
            intent[min(1, self.config.intent_dim - 1)] = 1.0
            if self.config.use_action_mask and self.config.action_dim >= 3:
                mask[:, 2] = 0.0
            label = "stealth"
        else:
            intent[0] = 1.0
            label = "attack"
        return intent, mask, label

    def compute_shaped_rewards(
        self,
        env_reward: Tensor,
        state: Tensor,
        next_state: Tensor,
        intent: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if self.current_eta == 0.0:
            return env_reward, torch.zeros((), dtype=env_reward.dtype, device=env_reward.device)
        phi_t = self.potential(state.unsqueeze(0), intent.unsqueeze(0)).squeeze(0)
        phi_tp1 = self.potential(next_state.unsqueeze(0), intent.unsqueeze(0)).squeeze(0)
        intrinsic_reward = self.config.gamma * phi_tp1 - phi_t
        total_reward = env_reward + self.current_eta * intrinsic_reward
        return total_reward, intrinsic_reward

    def save_checkpoint(self, path: str, extra: Optional[Dict[str, object]] = None) -> None:
        config_fields = {field.name for field in fields(IMAPPOConfig)}
        checkpoint = {
            "config": {
                key: value
                for key, value in self.config.__dict__.items()
                if key in config_fields
            },
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "potential": self.potential.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "potential_optim": self.potential_optim.state_dict(),
            "potential_update_step": self.potential_update_step,
            "current_eta": self.current_eta,
            "current_entropy_coef": self.current_entropy_coef,
            "extra": extra or {},
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[str] = None) -> "IMAPPO":
        checkpoint = torch.load(path, map_location=device or "cpu")
        config_dict = dict(checkpoint["config"])
        if device is not None:
            config_dict["device"] = device
        config = IMAPPOConfig(**config_dict)
        algo = cls(config)
        algo.actor.load_state_dict(checkpoint["actor"])
        algo.critic.load_state_dict(checkpoint["critic"])
        algo.potential.load_state_dict(checkpoint["potential"])
        if "actor_optim" in checkpoint:
            algo.actor_optim.load_state_dict(checkpoint["actor_optim"])
        if "critic_optim" in checkpoint:
            algo.critic_optim.load_state_dict(checkpoint["critic_optim"])
        if "potential_optim" in checkpoint:
            algo.potential_optim.load_state_dict(checkpoint["potential_optim"])
        algo.potential_update_step = checkpoint.get("potential_update_step", 0)
        algo.current_eta = checkpoint.get("current_eta", config.eta)
        algo.current_entropy_coef = checkpoint.get(
            "current_entropy_coef", config.entropy_coef
        )
        return algo

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
        base_actions = batch["base_actions"]
        policy_latents = batch["policy_latents"]
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
        critic_rewards = rewards if self.config.algorithm == "ippo" else team_rewards
        advantages, returns = self.compute_gae(
            critic_rewards, dones, values.detach(), next_values.detach()
        )
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        agent_advantages = (
            advantages
            if self.config.algorithm == "ippo"
            else advantages.unsqueeze(1).expand(-1, self.config.n_agents)
        )

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
                mb_base_actions = base_actions[idx]
                mb_policy_latents = policy_latents[idx]
                mb_action_masks = action_masks[idx]
                mb_intents = intents[idx]
                mb_returns = returns[idx]
                mb_advantages = agent_advantages[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_old_values = old_values[idx]
                mb_next_states = next_states[idx]

                critic_values, _ = self.critic(mb_states, mb_intents, mb_obs)
                clipped_values = mb_old_values + torch.clamp(
                    critic_values - mb_old_values,
                    -self.config.value_clip,
                    self.config.value_clip,
                )
                critic_loss_unclipped = (critic_values - mb_returns).pow(2)
                critic_loss_clipped = (clipped_values - mb_returns).pow(2)
                critic_loss = 0.5 * torch.max(
                    critic_loss_unclipped, critic_loss_clipped
                ).mean()

                flat_obs = mb_obs.reshape(-1, self.config.obs_dim)
                flat_intents = (
                    mb_intents.unsqueeze(1)
                    .expand(-1, self.config.n_agents, -1)
                    .reshape(-1, self.config.intent_dim)
                )
                flat_masks = mb_action_masks.reshape(-1, self.config.action_dim)
                flat_actions = mb_actions.reshape(-1, self.config.action_dim)
                flat_base_actions = mb_base_actions.reshape(-1, self.config.action_dim)
                flat_policy_latents = mb_policy_latents.reshape(
                    -1, self.config.action_dim
                )
                flat_advantages = mb_advantages.reshape(-1)
                flat_old_log_probs = mb_old_log_probs.reshape(-1)

                dist, _ = self.actor.distribution(
                    flat_obs,
                    flat_intents,
                    flat_masks,
                    flat_base_actions if self.config.policy_mode == "residual_rule" else None,
                )
                if self.config.policy_mode == "residual_rule":
                    new_log_probs = self.actor.latent_log_prob(
                        dist, flat_policy_latents, flat_masks
                    )
                else:
                    new_log_probs = self.actor.log_prob(dist, flat_actions, flat_masks)
                entropy = dist.entropy().sum(dim=-1).mean()
                ratios = torch.exp(new_log_probs - flat_old_log_probs)
                surr1 = ratios * flat_advantages
                surr2 = torch.clamp(
                    ratios,
                    1.0 - self.config.eps_clip,
                    1.0 + self.config.eps_clip,
                ) * flat_advantages
                actor_loss = -(
                    torch.min(surr1, surr2).mean()
                    + self.current_entropy_coef * entropy
                )

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
            "eta": float(self.current_eta),
            "entropy_coef": float(self.current_entropy_coef),
        }

    def select_actions(
        self,
        obs: Tensor,
        intent: Tensor,
        action_mask: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        execution_agents = int(obs.size(0))
        repeated_intent = intent.unsqueeze(0).expand(execution_agents, -1)
        profile = None
        if self.config.policy_mode == "residual_rule":
            from rule_based_baseline import compute_rule_actions

            profile = self._rule_prior_objective_profile(intent)
            base_actions = compute_rule_actions(
                obs,
                self._rule_prior_posture(intent),
                action_mask,
                objective_profile=profile,
            )
        elif self.config.policy_mode == "direct":
            base_actions = torch.zeros_like(action_mask)
        else:
            raise ValueError(f"unsupported policy_mode: {self.config.policy_mode}")
        self._last_base_actions = base_actions.detach()
        actions, log_probs, policy_latents = self.actor.sample_action(
            obs,
            repeated_intent,
            action_mask,
            deterministic=deterministic,
            base_action=(
                base_actions if self.config.policy_mode == "residual_rule" else None
            ),
        )
        unfiltered_actions = actions
        self._last_safety_solver_diagnostics = {}
        if self.config.safety_filter_mode == "pairwise_cbf":
            from rule_based_baseline import (
                apply_pairwise_cbf_filter,
                pairwise_cbf_constraint_diagnostics,
            )

            filter_start = time.perf_counter()
            actions = apply_pairwise_cbf_filter(
                obs,
                unfiltered_actions,
                profile,
                base_min_distance=self.config.cbf_base_min_distance,
                iterations=self.config.cbf_iterations,
            )
            cyclic_audit = pairwise_cbf_constraint_diagnostics(
                obs,
                actions,
                profile,
                base_min_distance=self.config.cbf_base_min_distance,
            )
            self._last_safety_solver_diagnostics = {
                "safety_filter_solver_success": float(
                    cyclic_audit["cbf_constraint_max_violation"]
                    <= self.config.cbf_solver_tolerance
                ),
                "safety_filter_solver_reported_success": float(
                    cyclic_audit["cbf_constraint_max_violation"]
                    <= self.config.cbf_solver_tolerance
                ),
                "safety_filter_solver_iterations": float(self.config.cbf_iterations),
                "safety_filter_solver_time_ms": float(
                    1000.0 * (time.perf_counter() - filter_start)
                ),
                "safety_filter_used_fallback": 0.0,
                **cyclic_audit,
            }
        elif self.config.safety_filter_mode == "pairwise_qp":
            from rule_based_baseline import apply_pairwise_qp_filter

            actions, self._last_safety_solver_diagnostics = apply_pairwise_qp_filter(
                obs,
                unfiltered_actions,
                profile,
                base_min_distance=self.config.cbf_base_min_distance,
                tolerance=self.config.cbf_solver_tolerance,
                max_iterations=self.config.cbf_solver_max_iterations,
            )
        self._last_policy_latents = policy_latents.detach()
        self._last_residual_actions = (unfiltered_actions - base_actions).detach()
        self._last_safety_filter_correction = (
            actions - unfiltered_actions
        ).detach()
        self._last_safety_filter_profile = profile
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


def env_reset(env, seed: Optional[int] = None):
    try:
        reset_out = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        reset_out = env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        return reset_out
    return reset_out, {}


def set_env_intent(env, intent: Tensor | np.ndarray, label: str = "") -> None:
    base_env = getattr(env, "unwrapped", env)
    if not hasattr(base_env, "set_intent"):
        return
    if isinstance(intent, torch.Tensor):
        intent_array = intent.detach().cpu().numpy()
    else:
        intent_array = np.asarray(intent, dtype=np.float32)
    try:
        base_env.set_intent(intent_array, label)
    except TypeError:
        base_env.set_intent(intent_array)


def set_env_tactical_posture(env, posture: str | float) -> None:
    base_env = getattr(env, "unwrapped", env)
    if hasattr(base_env, "set_tactical_posture"):
        base_env.set_tactical_posture(posture)


def set_env_objective_profile(
    env, profile: Optional[Mapping[str, float]]
) -> None:
    base_env = getattr(env, "unwrapped", env)
    if profile is not None and hasattr(base_env, "set_objective_profile"):
        base_env.set_objective_profile(profile)


def training_tactical_posture(episode: int) -> str:
    # Alternate externally between attack and stealth so both algorithms
    # experience the same scenario schedule during training.
    return "attack" if episode % 2 == 0 else "stealth"


def evaluation_tactical_posture(mode: str) -> str:
    return "stealth" if mode == "dense" else "attack"


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


def extract_collision_count(infos, agent_order: List[str]) -> int:
    if isinstance(infos, dict):
        for agent_id in agent_order:
            agent_info = infos.get(agent_id, {})
            if isinstance(agent_info, dict) and "collision" in agent_info:
                return int(bool(agent_info["collision"]))
        if "collision" in infos:
            return int(bool(infos["collision"]))
    return 0


def summarise_step_info(infos, agent_order: List[str]) -> Dict[str, float]:
    summary = {
        "task_completion": 0.0,
        "reward_env": 0.0,
        "reward_dist": 0.0,
        "reward_energy": 0.0,
        "reward_collision": 0.0,
        "reward_safety": 0.0,
        "reward_task": 0.0,
        "task_cost": 0.0,
        "reward_time": 0.0,
        "reward_threat": 0.0,
        "energy_remaining": 0.0,
        "action_magnitude": 0.0,
        "speed": 0.0,
        "distance_to_target": 0.0,
        "min_neighbor_distance": 0.0,
        "threat_zone_violation": 0.0,
        "distance_to_threat": 0.0,
    }
    if not isinstance(infos, dict):
        return summary

    collected = []
    for agent_id in agent_order:
        agent_info = infos.get(agent_id, {})
        if isinstance(agent_info, dict):
            collected.append(agent_info)
    if not collected:
        return summary

    denom = float(len(collected))
    for key in summary:
        summary[key] = float(
            sum(float(agent_info.get(key, 0.0)) for agent_info in collected) / denom
        )
    return summary


def build_uav_env_factory(config: IMAPPOConfig, mode: str = "train") -> Callable[[], object]:
    import gymnasium as gym

    def make_env():
        if mode == "train":
            progress = float(np.clip(getattr(config, "_env_progress", 0.0), 0.0, 1.0))
            hard_episode = bool(getattr(config, "_use_hard_train_env", False))
            if hard_episode:
                spawn_scale = config.hard_train_spawn_scale
                separation_scale = config.hard_train_separation_scale
            else:
                spawn_scale = config.curriculum_spawn_scale_start + progress * (
                    config.curriculum_spawn_scale_end - config.curriculum_spawn_scale_start
                )
                separation_scale = config.curriculum_separation_start + progress * (
                    config.curriculum_separation_end - config.curriculum_separation_start
                )
        elif mode == "collision_probe":
            spawn_scale = config.collision_probe_spawn_scale
            separation_scale = config.collision_probe_separation_scale
        else:
            spawn_scale = config.eval_spawn_scale
            separation_scale = config.eval_separation_scale

        return gym.make(
            str(getattr(config, "environment_name", "uav-scheduling-v0")),
            n_agents=config.n_agents,
            n_targets=config.n_targets,
            obs_dim=config.obs_dim,
            spawn_region_scale=spawn_scale,
            spawn_separation_scale=separation_scale,
            safety_reward_coef=config.safety_reward_coef,
            task_reward_coef=getattr(config, "task_reward_coef", 1.20),
            intent_reward_profiles_enabled=config.intent_reward_profiles_enabled,
            wind_std=config.wind_std,
            observation_noise_std=config.observation_noise_std,
            action_delay_steps=config.action_delay_steps,
            communication_dropout_prob=config.communication_dropout_prob,
        )

    return make_env


def evaluate_imappo(
    algo: IMAPPO,
    env_factory: Callable[[], object],
    config: IMAPPOConfig,
    prefix: str = "eval",
    evaluation_mode: str = "standard",
    intent_override: Optional[Tensor] = None,
    intent_label_override: str = "",
    tactical_posture_override: Optional[str] = None,
    objective_profile_override: Optional[Mapping[str, float]] = None,
    evaluation_seed_offset: int = 0,
) -> Dict[str, float]:
    env = env_factory()
    episode_returns = []
    episode_collisions = []
    episode_collision_rates = []
    episode_task_completion = []
    resource_metric_names = (
        "energy_remaining",
        "action_magnitude",
        "speed",
        "distance_to_target",
        "min_neighbor_distance",
        "threat_zone_violation",
        "distance_to_threat",
        "policy_residual_magnitude",
        "safety_filter_correction_magnitude",
        "cbf_constraint_max_violation",
        "cbf_constraint_mean_violation",
        "cbf_constraint_violation_fraction",
        "cbf_predicted_min_pairwise_distance",
        "safety_filter_solver_success",
        "safety_filter_solver_reported_success",
        "safety_filter_solver_iterations",
        "safety_filter_solver_time_ms",
        "safety_filter_used_fallback",
    )
    collect_resources = not str(
        getattr(config, "environment_name", "uav-scheduling-v0")
    ).startswith("vmas:")
    episode_resources = {name: [] for name in resource_metric_names}

    for episode_index in range(config.eval_episodes):
        evaluation_seed = int(config.seed) * 1_000_000 + int(evaluation_seed_offset) + episode_index
        obs_data, _ = env_reset(env, seed=evaluation_seed)
        agent_order = infer_agent_order(env, obs_data, config)
        obs_array = normalise_obs(agent_order, obs_data)
        if intent_override is None:
            intent, episode_mask, intent_label = algo.evaluation_intent_and_mask(mode=evaluation_mode)
            tactical_posture = evaluation_tactical_posture(evaluation_mode)
        else:
            intent = intent_override.to(algo.device)
            intent_label = str(intent_label_override)
            tactical_posture = tactical_posture_override or (
                algo.intent_library.posture_for_label(intent_label)
                if algo.intent_library is not None
                else evaluation_tactical_posture(evaluation_mode)
            )
            episode_mask = algo._action_mask_for_posture(tactical_posture)
        if hasattr(algo, "set_evaluation_context"):
            algo.set_evaluation_context(intent_label, tactical_posture)
        if hasattr(algo, "set_evaluation_objective_profile"):
            algo.set_evaluation_objective_profile(objective_profile_override)
        if episode_mask.size(0) != len(agent_order):
            episode_mask = episode_mask[:1].expand(len(agent_order), -1).clone()
        set_env_intent(env, intent, intent_label)
        set_env_objective_profile(env, objective_profile_override)
        set_env_tactical_posture(env, tactical_posture)

        ep_return = 0.0
        ep_collisions = 0.0
        ep_task_completion = 0.0
        ep_resources = {name: 0.0 for name in resource_metric_names}
        ep_steps = 0
        for _ in range(config.max_steps):
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=algo.device)
            actions, _ = algo.select_actions(
                obs_tensor, intent, episode_mask, deterministic=True
            )
            cbf_diagnostics = {}
            if collect_resources:
                from rule_based_baseline import pairwise_cbf_constraint_diagnostics

                cbf_diagnostics = pairwise_cbf_constraint_diagnostics(
                    obs_tensor,
                    actions,
                    getattr(algo, "_last_safety_filter_profile", None),
                    base_min_distance=config.cbf_base_min_distance,
                )
            next_obs_data, reward_vec, done_flag, truncated_flag, info = env_step(
                env, agent_order, actions.detach().cpu().numpy()
            )
            next_obs_array = normalise_obs(agent_order, next_obs_data)
            step_info = summarise_step_info(info, agent_order)
            residual_actions = getattr(algo, "_last_residual_actions", None)
            step_info["policy_residual_magnitude"] = (
                float(torch.linalg.vector_norm(
                    residual_actions, dim=-1
                ).mean().item())
                if residual_actions is not None else 0.0
            )
            filter_correction = getattr(
                algo, "_last_safety_filter_correction", None
            )
            step_info["safety_filter_correction_magnitude"] = (
                float(torch.linalg.vector_norm(
                    filter_correction, dim=-1
                ).mean().item())
                if filter_correction is not None else 0.0
            )
            step_info.update(cbf_diagnostics)
            step_info.update(
                getattr(algo, "_last_safety_solver_diagnostics", {})
            )
            for resource_name in resource_metric_names:
                step_info.setdefault(resource_name, 0.0)

            ep_return += float(np.mean(reward_vec))
            ep_collisions += float(extract_collision_count(info, agent_order))
            ep_task_completion += step_info["task_completion"]
            if collect_resources:
                for name in resource_metric_names:
                    ep_resources[name] += step_info[name]
            ep_steps += 1

            obs_array = next_obs_array
            done = done_flag or truncated_flag
            if done:
                break

        episode_returns.append(ep_return)
        episode_collisions.append(ep_collisions)
        episode_collision_rates.append(ep_collisions / max(ep_steps, 1))
        episode_task_completion.append(ep_task_completion / max(ep_steps, 1))
        if collect_resources:
            for name in resource_metric_names:
                episode_resources[name].append(ep_resources[name] / max(ep_steps, 1))

    env.close()
    metrics = {
        f"{prefix}_episode_return": float(np.mean(episode_returns)),
        f"{prefix}_episode_collisions": float(np.mean(episode_collisions)),
        f"{prefix}_collision_rate": float(np.mean(episode_collision_rates)),
        f"{prefix}_task_completion": float(np.mean(episode_task_completion)),
    }
    if collect_resources:
        metrics.update(
            {
                f"{prefix}_{name}": float(np.mean(values))
                for name, values in episode_resources.items()
            }
        )
    return metrics


def detect_switch_response_latency(
    action_delta_series: List[float],
    threshold: float,
) -> Optional[int]:
    """Return the first zero-based post-switch step with a material action change."""
    if threshold <= 0.0:
        raise ValueError("response threshold must be positive")
    for index, value in enumerate(action_delta_series):
        if float(value) >= threshold:
            return int(index)
    return None


def evaluate_dynamic_intent_switch(
    algo: IMAPPO,
    env_factory: Callable[[], object],
    config: IMAPPOConfig,
    *,
    pre_intent: Tensor,
    post_intent: Tensor,
    pre_label: str,
    post_label: str,
    pre_posture: str,
    post_posture: str,
    pre_objective_profile: Optional[Mapping[str, float]],
    post_objective_profile: Optional[Mapping[str, float]],
    switch_step: int,
    total_steps: int,
    response_threshold: float = 0.05,
    evaluation_seed_offset: int = 0,
) -> Dict[str, object]:
    """Evaluate a within-episode language-intent intervention.

    At every post-switch state, the deterministic post-intent action is compared
    with a counterfactual pre-intent action on the *same observation*. This makes
    latency a controller-response diagnostic rather than a confounded trajectory
    comparison. Environment metrics remain phase-specific trajectory evidence.
    """
    if switch_step < 1 or total_steps <= switch_step:
        raise ValueError("dynamic evaluation requires 1 <= switch_step < total_steps")
    if response_threshold <= 0.0:
        raise ValueError("response_threshold must be positive")
    resource_names = (
        "task_completion", "energy_remaining", "action_magnitude", "speed",
        "distance_to_target", "min_neighbor_distance", "threat_zone_violation",
        "distance_to_threat",
    )
    episode_records: List[Dict[str, object]] = []
    for episode_index in range(config.eval_episodes):
        env = env_factory()
        evaluation_seed = (
            int(config.seed) * 1_000_000 + int(evaluation_seed_offset) + episode_index
        )
        obs_data, _ = env_reset(env, seed=evaluation_seed)
        agent_order = infer_agent_order(env, obs_data, config)
        obs_array = normalise_obs(agent_order, obs_data)
        phase_sums = {
            "pre": {name: 0.0 for name in resource_names},
            "post": {name: 0.0 for name in resource_names},
        }
        phase_counts = {"pre": 0, "post": 0}
        action_delta_series: List[float] = []
        for step in range(total_steps):
            post_phase = step >= switch_step
            phase = "post" if post_phase else "pre"
            intent = post_intent if post_phase else pre_intent
            label = post_label if post_phase else pre_label
            posture = post_posture if post_phase else pre_posture
            profile = post_objective_profile if post_phase else pre_objective_profile
            mask = algo._action_mask_for_posture(posture)
            if mask.size(0) != len(agent_order):
                mask = mask[:1].expand(len(agent_order), -1).clone()
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=algo.device)

            if post_phase:
                pre_mask = algo._action_mask_for_posture(pre_posture)
                if pre_mask.size(0) != len(agent_order):
                    pre_mask = pre_mask[:1].expand(len(agent_order), -1).clone()
                if hasattr(algo, "set_evaluation_context"):
                    algo.set_evaluation_context(pre_label, pre_posture)
                if hasattr(algo, "set_evaluation_objective_profile"):
                    algo.set_evaluation_objective_profile(pre_objective_profile)
                counterfactual_pre_actions, _ = algo.select_actions(
                    obs_tensor, pre_intent, pre_mask, deterministic=True
                )
            else:
                counterfactual_pre_actions = None

            if hasattr(algo, "set_evaluation_context"):
                algo.set_evaluation_context(label, posture)
            if hasattr(algo, "set_evaluation_objective_profile"):
                algo.set_evaluation_objective_profile(profile)
            actions, _ = algo.select_actions(obs_tensor, intent, mask, deterministic=True)
            if counterfactual_pre_actions is not None:
                action_delta_series.append(float(torch.linalg.vector_norm(
                    actions - counterfactual_pre_actions, dim=-1
                ).mean().item()))

            set_env_intent(env, intent, label)
            set_env_objective_profile(env, profile)
            set_env_tactical_posture(env, posture)
            next_obs_data, _, done_flag, truncated_flag, info = env_step(
                env, agent_order, actions.detach().cpu().numpy()
            )
            step_info = summarise_step_info(info, agent_order)
            for name in resource_names:
                phase_sums[phase][name] += step_info[name]
            phase_counts[phase] += 1
            obs_array = normalise_obs(agent_order, next_obs_data)
            if done_flag or truncated_flag:
                break
        env.close()
        latency = detect_switch_response_latency(action_delta_series, response_threshold)
        phase_means = {
            phase: {
                name: value / max(phase_counts[phase], 1)
                for name, value in sums.items()
            }
            for phase, sums in phase_sums.items()
        }
        episode_records.append({
            "seed": evaluation_seed,
            "response_latency_steps": latency,
            "response_detected": latency is not None,
            "switch_action_delta": (
                action_delta_series[0] if action_delta_series else 0.0
            ),
            "mean_post_switch_action_delta": (
                float(np.mean(action_delta_series)) if action_delta_series else 0.0
            ),
            "action_delta_series": action_delta_series,
            "phase_metrics": phase_means,
        })

    post_horizon = max(total_steps - switch_step, 1)
    detected_latencies = [
        int(record["response_latency_steps"])
        for record in episode_records if record["response_detected"]
    ]
    phase_metrics = {}
    for phase in ("pre", "post"):
        phase_metrics[phase] = {
            name: float(np.mean([
                record["phase_metrics"][phase][name] for record in episode_records
            ]))
            for name in resource_names
        }
    return {
        "episodes": episode_records,
        "response_rate": float(np.mean([
            float(record["response_detected"]) for record in episode_records
        ])),
        "response_latency_steps": (
            float(np.mean(detected_latencies)) if detected_latencies else None
        ),
        "censored_response_latency_steps": float(np.mean([
            int(record["response_latency_steps"])
            if record["response_detected"] else post_horizon
            for record in episode_records
        ])),
        "switch_action_delta": float(np.mean([
            record["switch_action_delta"] for record in episode_records
        ])),
        "mean_post_switch_action_delta": float(np.mean([
            record["mean_post_switch_action_delta"] for record in episode_records
        ])),
        "phase_metrics": phase_metrics,
        "post_minus_pre": {
            name: phase_metrics["post"][name] - phase_metrics["pre"][name]
            for name in resource_names
        },
    }


def train_imappo(
    env_factory: Optional[Callable[[], object]] = None,
    eval_env_factory: Optional[Callable[[], object]] = None,
    collision_probe_env_factory: Optional[Callable[[], object]] = None,
    config: Optional[IMAPPOConfig] = None,
    logger=None,
    log_callback: Optional[Callable[[Dict[str, float]], None]] = None,
    checkpoint_callback: Optional[Callable[["IMAPPO", Dict[str, float]], None]] = None,
) -> Tuple[IMAPPO, List[Dict[str, float]]]:
    cfg = config or IMAPPOConfig()
    algo = IMAPPO(cfg)
    buffer = RolloutBuffer()
    logs: List[Dict[str, float]] = []

    for episode in range(cfg.max_episodes):
        cfg._env_progress = episode / max(cfg.max_episodes - 1, 1)
        cfg._use_hard_train_env = (
            cfg.hard_train_interval > 0 and (episode + 1) % cfg.hard_train_interval == 0
        )
        algo.set_training_progress(episode / max(cfg.max_episodes - 1, 1))
        env = env_factory() if env_factory is not None else MockContinuousUAVEnv(cfg)
        training_seed = int(cfg.seed) * 1_000_000 + int(episode)
        obs_data, _ = env_reset(env, seed=training_seed)
        agent_order = infer_agent_order(env, obs_data, cfg)
        obs_array = normalise_obs(agent_order, obs_data)
        state_array = build_global_state(obs_array, cfg)
        tactical_posture = training_tactical_posture(episode)
        intent, episode_mask, intent_label = algo.sample_episode_intent_and_mask(
            tactical_posture=tactical_posture
        )
        set_env_intent(env, intent, intent_label)
        set_env_tactical_posture(env, tactical_posture)

        episode_return = 0.0
        episode_collisions = 0.0
        episode_reward_env = 0.0
        episode_reward_intent = 0.0
        episode_task_completion = 0.0
        episode_reward_dist = 0.0
        episode_reward_energy = 0.0
        episode_reward_collision = 0.0
        episode_reward_safety = 0.0
        episode_reward_task = 0.0
        episode_reward_time = 0.0
        episode_reward_threat = 0.0
        episode_steps = 0
        for _ in range(cfg.max_steps):
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=algo.device)
            state_tensor = torch.tensor(state_array, dtype=torch.float32, device=algo.device)
            actions, log_probs = algo.select_actions(obs_tensor, intent, episode_mask)

            action_np = actions.detach().cpu().numpy()
            next_obs_data, reward_vec, done_flag, truncated_flag, info = env_step(
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
            step_info = summarise_step_info(info, agent_order)

            done = done_flag or truncated_flag
            done_tensor = torch.tensor(float(done), dtype=torch.float32, device=algo.device)

            buffer.add(
                state=state_tensor,
                obs=obs_tensor,
                action=actions,
                base_action=algo._last_base_actions,
                policy_latent=algo._last_policy_latents,
                action_mask=episode_mask,
                intent=intent,
                reward=total_reward,
                done=done_tensor,
                log_prob=log_probs,
                next_state=next_state_tensor,
                next_obs=torch.tensor(next_obs_array, dtype=torch.float32, device=algo.device),
            )

            episode_return += float(total_reward.mean().item())
            episode_reward_env += float(extrinsic_rewards.mean().item())
            episode_reward_intent += float((total_reward - extrinsic_rewards).mean().item())
            episode_collisions += float(extract_collision_count(info, agent_order))
            episode_task_completion += step_info["task_completion"]
            episode_reward_dist += step_info["reward_dist"]
            episode_reward_energy += step_info["reward_energy"]
            episode_reward_collision += step_info["reward_collision"]
            episode_reward_safety += step_info["reward_safety"]
            episode_reward_task += step_info["reward_task"]
            episode_reward_time += step_info["reward_time"]
            episode_reward_threat += step_info["reward_threat"]
            episode_steps += 1
            obs_array = next_obs_array
            state_array = next_state_array

            if buffer.is_ready(cfg.rollout_length):
                update_log = algo.update(buffer)
                update_log["episode"] = float(episode)
                update_log["algorithm"] = cfg.algorithm
                logs.append(update_log)
                if log_callback is not None:
                    log_callback(update_log)
                if logger is not None:
                    for key, value in update_log.items():
                        if key != "episode":
                            logger.log_stat(key, value, int(episode))

            if done:
                break

        if episode == cfg.max_episodes - 1 and buffer.storage["states"]:
            update_log = algo.update(buffer)
            update_log["episode"] = float(episode)
            update_log["algorithm"] = cfg.algorithm
            logs.append(update_log)
            if log_callback is not None:
                log_callback(update_log)
            if logger is not None:
                for key, value in update_log.items():
                    if key != "episode":
                        logger.log_stat(key, value, int(episode))

        mean_task_completion = (
            episode_task_completion / max(episode_steps, 1)
        )
        collision_rate = episode_collisions / max(episode_steps, 1)
        episode_log = {
            "episode": float(episode),
            "episode_return": episode_return,
            "episode_collisions": episode_collisions,
            "episode_collision_rate": collision_rate,
            "episode_reward_env": episode_reward_env,
            "episode_reward_intent": episode_reward_intent,
            "episode_task_completion": mean_task_completion,
            "episode_reward_dist": episode_reward_dist,
            "episode_reward_energy": episode_reward_energy,
            "episode_reward_collision": episode_reward_collision,
            "episode_reward_safety": episode_reward_safety,
            "episode_reward_task": episode_reward_task,
            "episode_reward_time": episode_reward_time,
            "episode_reward_threat": episode_reward_threat,
            "algorithm": cfg.algorithm,
            "intent_label": intent_label,
            "tactical_posture": tactical_posture,
            "intent_source": cfg.intent_source,
        }
        logs.append(episode_log)
        if log_callback is not None:
            log_callback(episode_log)
        if checkpoint_callback is not None:
            checkpoint_callback(algo, episode_log)
        if logger is not None:
            logger.log_stat("episode", episode, episode)
            logger.log_stat("episode_return", episode_return, episode)
            logger.log_stat("episode_collisions", episode_collisions, episode)
            logger.log_stat("episode_collision_rate", collision_rate, episode)
            logger.log_stat("episode_reward_env", episode_reward_env, episode)
            logger.log_stat("episode_reward_intent", episode_reward_intent, episode)
            logger.log_stat("episode_task_completion", mean_task_completion, episode)
            logger.log_stat("episode_reward_dist", episode_reward_dist, episode)
            logger.log_stat("episode_reward_energy", episode_reward_energy, episode)
            logger.log_stat("episode_reward_collision", episode_reward_collision, episode)
            logger.log_stat("episode_reward_safety", episode_reward_safety, episode)
            logger.log_stat("episode_reward_task", episode_reward_task, episode)
            logger.log_stat("episode_reward_time", episode_reward_time, episode)
            logger.log_stat("episode_reward_threat", episode_reward_threat, episode)
            logger.log_stat("curriculum_spawn_scale", getattr(env.unwrapped, "spawn_region_scale", 0.0), episode)
            logger.log_stat("curriculum_separation_scale", getattr(env.unwrapped, "spawn_separation_scale", 0.0), episode)
            logger.log_stat("hard_train_episode", float(cfg._use_hard_train_env), episode)

        if eval_env_factory is not None and (
            (episode + 1) % max(cfg.eval_interval, 1) == 0
            or episode == cfg.max_episodes - 1
        ):
            eval_log = evaluate_imappo(
                algo,
                eval_env_factory,
                cfg,
                prefix="eval",
                evaluation_mode="standard",
                evaluation_seed_offset=500_000,
            )
            eval_log["episode"] = float(episode)
            eval_log["algorithm"] = cfg.algorithm
            logs.append(eval_log)
            if log_callback is not None:
                log_callback(eval_log)
            if checkpoint_callback is not None:
                checkpoint_callback(algo, eval_log)
            if logger is not None:
                for key, value in eval_log.items():
                    if key != "episode":
                        logger.log_stat(key, value, episode)
            if collision_probe_env_factory is not None:
                probe_log = evaluate_imappo(
                    algo,
                    collision_probe_env_factory,
                    cfg,
                    prefix="probe",
                    evaluation_mode="dense",
                    evaluation_seed_offset=600_000,
                )
                probe_log["episode"] = float(episode)
                probe_log["algorithm"] = cfg.algorithm
                logs.append(probe_log)
                if log_callback is not None:
                    log_callback(probe_log)
                if checkpoint_callback is not None:
                    checkpoint_callback(algo, probe_log)
                if logger is not None:
                    for key, value in probe_log.items():
                        if key != "episode":
                            logger.log_stat(key, value, episode)

        env.close()
    return algo, logs


def build_imappo_config_from_args(args) -> IMAPPOConfig:
    n_agents = getattr(args, "imappo_n_agents", 8)
    n_targets = getattr(args, "imappo_n_targets", n_agents)
    obs_dim = getattr(args, "imappo_obs_dim", None)
    state_dim = getattr(args, "imappo_state_dim", None)
    resolved_obs_dim = infer_obs_dim(n_agents) if obs_dim is None else obs_dim
    resolved_state_dim = infer_state_dim(n_agents, resolved_obs_dim) if state_dim is None else state_dim
    config = IMAPPOConfig(
        algorithm=getattr(args, "algorithm", "imappo"),
        critic_mode=getattr(args, "critic_mode", "attention"),
        use_action_mask=getattr(args, "use_action_mask", True),
        intent_source=getattr(args, "intent_source", "onehot"),
        intent_library_path=getattr(args, "intent_library_path", ""),
        intent_encoder_model=getattr(
            args,
            "intent_encoder_model",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        intent_encoder_revision=getattr(args, "intent_encoder_revision", ""),
        intent_encoder_batch_size=getattr(args, "intent_encoder_batch_size", 32),
        intent_projection_seed=getattr(args, "intent_projection_seed", 0),
        intent_code_seed=getattr(args, "intent_code_seed", 0),
        intent_encoder_device=getattr(args, "intent_encoder_device", ""),
        n_agents=n_agents,
        n_targets=n_targets,
        obs_dim=resolved_obs_dim,
        state_dim=resolved_state_dim,
        action_dim=getattr(args, "imappo_action_dim", 3),
        intent_dim=getattr(args, "intent_dim", 8),
        gamma=args.gamma,
        gae_lambda=getattr(args, "gae_lambda", 0.95),
        eps_clip=getattr(args, "eps_clip", 0.1),
        eta=getattr(args, "eta", 0.5),
        eta_end=getattr(args, "eta_end", 0.1),
        entropy_coef=getattr(args, "entropy_coef", 1e-3),
        entropy_coef_end=getattr(args, "entropy_coef_end", 1e-4),
        value_coef=getattr(args, "value_coef", 0.5),
        max_grad_norm=getattr(args, "grad_norm_clip", 0.5),
        value_clip=getattr(args, "value_clip", 0.2),
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
        eval_interval=getattr(args, "eval_interval", 10),
        eval_episodes=getattr(args, "eval_episodes", 3),
        safety_reward_coef=getattr(args, "safety_reward_coef", 1.0),
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
