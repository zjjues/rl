"""Run a pre-registered, non-overwriting UAV intent-representation study."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if TYPE_CHECKING:
    from imappo import IMAPPOConfig


PAPER_MIN_SEEDS = 10
PAPER_MIN_EVAL_EPISODES = 100

EVALUATION_METRIC_DIRECTIONS = (
    ("collision_rate", True),
    ("task_completion", False),
    ("episode_return", False),
    ("episode_collisions", True),
    ("energy_remaining", False),
    ("action_magnitude", True),
    ("speed", False),
    ("distance_to_target", True),
    ("min_neighbor_distance", False),
    ("threat_zone_violation", True),
    ("distance_to_threat", False),
    ("policy_residual_magnitude", False),
    ("safety_filter_correction_magnitude", False),
    ("cbf_constraint_max_violation", True),
    ("cbf_constraint_mean_violation", True),
    ("cbf_constraint_violation_fraction", True),
    ("cbf_predicted_min_pairwise_distance", False),
    ("safety_filter_solver_success", False),
    ("safety_filter_solver_reported_success", False),
    ("safety_filter_solver_iterations", True),
    ("safety_filter_solver_time_ms", True),
    ("safety_filter_used_fallback", True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip()


def validate_spec(spec: Dict[str, object], allow_dirty: bool) -> None:
    required = {"schema_version", "study_id", "level", "seeds", "environment", "training", "intent", "variants", "evaluation"}
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"study config is missing required keys: {missing}")
    level = str(spec["level"])
    if level not in {"smoke", "pilot", "paper"}:
        raise ValueError("level must be smoke, pilot, or paper")
    seeds = list(spec["seeds"])
    eval_episodes = int(spec["evaluation"]["episodes"])
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    if level == "paper":
        if len(seeds) < PAPER_MIN_SEEDS:
            raise ValueError(f"paper studies require at least {PAPER_MIN_SEEDS} seeds")
        if eval_episodes < PAPER_MIN_EVAL_EPISODES:
            raise ValueError(
                f"paper studies require at least {PAPER_MIN_EVAL_EPISODES} evaluation episodes per seed/tier"
            )
        dirty = git_output("status", "--porcelain=v1")
        if dirty and not allow_dirty:
            raise RuntimeError("paper studies require a clean Git worktree; commit the research snapshot first")
        uses_pretrained = any(
            str(item.get("intent_source", "")) in {
                "pretrained_semantic",
                "objective_grounded_semantic",
            }
            for item in spec["variants"]
        )
        if uses_pretrained and not str(spec["intent"].get("encoder_revision", "")):
            raise ValueError("paper studies using pretrained_semantic must pin encoder_revision")
    variant_keys = [str(item["key"]) for item in spec["variants"]]
    if len(variant_keys) != len(set(variant_keys)):
        raise ValueError("variant keys must be unique")
    forbidden = {"qmix", "vdn", "qmix_vdn"}
    if any(key.lower() in forbidden for key in variant_keys):
        raise ValueError("discrete QMIX/VDN cannot be registered in this continuous-action study")
    if "generalization" in spec:
        from intent_generalization import load_generalization_suite

        suite_path = ROOT / str(spec["generalization"].get("suite", ""))
        if not suite_path.is_file():
            raise ValueError(f"generalization suite does not exist: {suite_path}")
        suite = load_generalization_suite(suite_path)
        observed_splits = {str(query["split"]) for query in suite["queries"]}
        if not {"seen", "paraphrase", "unseen"}.issubset(observed_splits):
            raise ValueError("generalization suite must contain seen, paraphrase, and unseen queries")
    if any(
        str(item.get("intent_profile_decoder", "")) in {
            "nli_entailment",
            "nli_relevance_gated",
            "nli_similarity_gated",
            "nli_prototype_gated",
        }
        for item in spec["variants"]
    ) and not str(spec["intent"].get("nli_model_revision", "")):
        raise ValueError("NLI decoder variants must pin intent.nli_model_revision")


def output_dir_for(spec: Dict[str, object]) -> Path:
    return ROOT / "experiments" / str(spec["level"]) / str(spec["study_id"])


def dependency_versions(names: Iterable[str]) -> Dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def build_manifest(spec: Dict[str, object], command: List[str]) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "study_id": spec["study_id"],
        "level": spec["level"],
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "command": command,
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": dependency_versions(
            ["numpy", "torch", "gymnasium", "sentence-transformers", "scipy"]
        ),
        "seed_protocol": {
            "training_reset": "seed * 1000000 + training_episode",
            "periodic_eval_reset": "seed * 1000000 + 500000 + evaluation_episode",
            "collision_probe_reset": "seed * 1000000 + 600000 + evaluation_episode",
            "final_tier_reset": "seed * 1000000 + 10000 + tier_index * 1000 + evaluation_episode",
            "query_reset": "seed * 1000000 + 100000 + query_index * 10000 + tier_index * 1000 + evaluation_episode",
        },
        "config": spec,
        "status": "running",
    }


def write_result_card(
    output_dir: Path,
    spec: Dict[str, object],
    summaries: Dict[str, object],
) -> None:
    lines = [
        f"# Result Card: {spec['study_id']}",
        "",
        f"- Evidence level: `{spec['level']}`",
        f"- Seeds: `{', '.join(str(seed) for seed in spec['seeds'])}`",
        f"- Evaluation episodes per seed/tier: `{spec['evaluation']['episodes']}`",
        f"- Primary objective: {spec.get('objective', 'Not specified')}",
        "",
        "## Variants",
        "",
    ]
    for variant in spec["variants"]:
        key = str(variant["key"])
        algorithm = str(variant.get("algorithm", "imappo"))
        representation = (
            "none" if algorithm in {"mappo", "ippo", "matd3"}
            else "structured_rule_context" if algorithm == "rule_planner"
            else str(variant.get("intent_source", "onehot"))
        )
        lines.append(f"- `{key}`: representation=`{representation}`, algorithm=`{algorithm}`")
    if "generalization" in spec:
        from intent_generalization import load_generalization_suite

        suite = load_generalization_suite(ROOT / str(spec["generalization"]["suite"]))
        split_counts = {
            split: sum(1 for query in suite["queries"] if query["split"] == split)
            for split in ("seen", "paraphrase", "unseen")
        }
        lines.extend(
            [
                "",
                "## Intent generalization protocol",
                "",
                f"- Suite: `{suite['suite_id']}`",
                f"- Training intents: `{len(suite['train_labels'])}`",
                f"- Queries: seen=`{split_counts['seen']}`, paraphrase=`{split_counts['paraphrase']}`, unseen=`{split_counts['unseen']}`",
                "- Query texts are averaged within each seed before cross-seed uncertainty is computed.",
                "- Random-dense and one-hot paraphrase queries receive canonical-label identity as an oracle control.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- `legacy_hash` and `random_dense` are representation controls and must not be described as semantic embeddings.",
            "- Paired confidence intervals that include zero do not support a stable directional advantage.",
            "- Safety improvements must be reported together with task completion and resource costs.",
            "- Representation retrieval metrics diagnose geometry and are not behavioral performance evidence.",
            "- Paired variants use the same deterministic environment-reset seed schedule.",
            "- This automatically generated card records protocol facts; paper claims require researcher review.",
            "",
            "## Artifact status",
            "",
            f"- Variant summaries: `{len(summaries)}`",
            "- Raw per-seed results: retained under each variant directory",
            "- Checksums: `checksums.sha256`",
            "",
        ]
    )
    (output_dir / "RESULT_CARD.md").write_text("\n".join(lines), encoding="utf-8")


def build_risk_factory(cfg: "IMAPPOConfig", tier: Dict[str, object]):
    environment_name = str(getattr(cfg, "environment_name", "uav-scheduling-v0"))
    if environment_name.startswith("vmas:"):
        from envs.vmas_adapter import VMASAdapter

        scenario = environment_name.split(":", 1)[1]

        def make_vmas_env():
            return VMASAdapter(
                scenario=scenario,
                n_agents=int(tier.get("n_agents", cfg.n_agents)),
                max_steps=cfg.max_steps,
            )

        return make_vmas_env

    import gymnasium as gym

    def make_env():
        return gym.make(
            environment_name,
            n_agents=int(tier.get("n_agents", cfg.n_agents)),
            n_targets=int(tier.get("n_targets", tier.get("n_agents", cfg.n_targets))),
            obs_dim=cfg.obs_dim,
            max_episode_steps=cfg.max_steps,
            spawn_region_scale=float(tier["spawn_region_scale"]),
            spawn_separation_scale=float(tier["spawn_separation_scale"]),
            safety_reward_coef=cfg.safety_reward_coef,
            task_reward_coef=getattr(cfg, "task_reward_coef", 1.20),
            intent_reward_profiles_enabled=cfg.intent_reward_profiles_enabled,
            wind_std=float(tier.get("wind_std", cfg.wind_std)),
            observation_noise_std=float(
                tier.get("observation_noise_std", cfg.observation_noise_std)
            ),
            action_delay_steps=int(tier.get("action_delay_steps", cfg.action_delay_steps)),
            communication_dropout_prob=float(
                tier.get("communication_dropout_prob", cfg.communication_dropout_prob)
            ),
        )

    return make_env


def build_config(spec: Dict[str, object], variant: Dict[str, object], seed: int) -> "IMAPPOConfig":
    from envs.uav_scheduling_env import infer_obs_dim, infer_obs_dim_v2, infer_state_dim
    from imappo import IMAPPOConfig

    env = spec["environment"]
    training = spec["training"]
    intent = spec["intent"]
    train_labels = ()
    if "generalization" in spec:
        from intent_generalization import load_generalization_suite

        suite = load_generalization_suite(ROOT / str(spec["generalization"]["suite"]))
        train_labels = tuple(str(label) for label in suite["train_labels"])
    n_agents = int(env["n_agents"])
    environment_name = str(env["name"])
    if environment_name.startswith("vmas:"):
        from envs.vmas_adapter import infer_vmas_dims

        scenario = environment_name.split(":", 1)[1]
        obs_dim, state_dim, action_dim = infer_vmas_dims(scenario, n_agents)
    else:
        obs_dim = (
            infer_obs_dim_v2(n_agents)
            if environment_name == "uav-scheduling-v2"
            else infer_obs_dim(n_agents)
        )
        state_dim = infer_state_dim(n_agents, obs_dim)
        action_dim = 3
    algorithm = str(variant.get("algorithm", "imappo"))
    policy_mode = str(variant.get("policy_mode", "direct"))
    if environment_name.startswith("vmas:") and policy_mode == "residual_rule":
        raise ValueError("UAV rule-residual policies cannot be registered on VMAS scenarios")
    if environment_name.startswith("vmas:") and algorithm == "rule_planner":
        raise ValueError("the UAV rule planner cannot be registered on VMAS scenarios")
    eta = float(training.get("eta", 0.5))
    eta_end = float(training.get("eta_end", 0.1))
    if variant.get("disable_intent_reward", False) or algorithm in {"mappo", "ippo", "matd3", "rule_planner"}:
        eta = 0.0
        eta_end = 0.0
    critic_mode = str(variant.get("critic_mode", "attention"))
    if algorithm == "ippo":
        critic_mode = "local"
    cfg = IMAPPOConfig(
        algorithm=algorithm,
        critic_mode=critic_mode,
        use_action_mask=bool(variant.get("use_action_mask", algorithm not in {"mappo", "ippo"})),
        intent_source=str(variant.get("intent_source", "pretrained_semantic")),
        policy_mode=policy_mode,
        residual_action_scale=float(variant.get("residual_action_scale", 0.25)),
        residual_initial_log_std=float(variant.get("residual_initial_log_std", -2.0)),
        rule_prior_context=str(variant.get("rule_prior_context", "neutral")),
        safety_filter_mode=str(variant.get("safety_filter_mode", "none")),
        cbf_base_min_distance=float(variant.get("cbf_base_min_distance", 1.0)),
        cbf_iterations=int(variant.get("cbf_iterations", 4)),
        cbf_solver_tolerance=float(variant.get("cbf_solver_tolerance", 1e-7)),
        cbf_solver_max_iterations=int(
            variant.get("cbf_solver_max_iterations", 100)
        ),
        replay_capacity=int(training.get("replay_capacity", 100000)),
        matd3_warmup_steps=int(training.get("matd3_warmup_steps", 200)),
        matd3_exploration_noise=float(training.get("matd3_exploration_noise", 0.10)),
        matd3_policy_noise=float(training.get("matd3_policy_noise", 0.20)),
        matd3_noise_clip=float(training.get("matd3_noise_clip", 0.50)),
        matd3_policy_delay=int(training.get("matd3_policy_delay", 2)),
        matd3_tau=float(training.get("matd3_tau", 0.005)),
        intent_dim=int(variant.get("intent_dim", intent["dim"])),
        intent_encoder_model=str(intent["encoder_model"]),
        intent_encoder_revision=str(intent.get("encoder_revision", "")),
        intent_projection_seed=int(intent.get("projection_seed", 0)),
        intent_code_seed=int(intent.get("code_seed", 0)),
        intent_train_labels=train_labels,
        intent_adapter_ridge=float(intent.get("adapter_ridge", 0.01)),
        intent_profile_decoder=str(variant.get(
            "intent_profile_decoder", intent.get("profile_decoder", "dual_ridge")
        )),
        intent_nli_model=str(intent.get(
            "nli_model", "cross-encoder/nli-deberta-v3-small"
        )),
        intent_nli_model_revision=str(intent.get("nli_model_revision", "")),
        intent_nli_batch_size=int(intent.get("nli_batch_size", 32)),
        intent_semantic_weight=float(intent.get("semantic_weight", 1.0)),
        intent_objective_weight=float(intent.get("objective_weight", 1.0)),
        align_intent_posture=bool(variant.get("align_intent_posture", True)),
        n_agents=n_agents,
        n_targets=int(env.get("n_targets", n_agents)),
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        max_episodes=int(training["episodes"]),
        max_steps=int(training["steps"]),
        rollout_length=int(training["rollout_length"]),
        minibatch_size=int(training["minibatch_size"]),
        eval_interval=int(training["eval_interval"]),
        eval_episodes=int(spec["evaluation"]["episodes"]),
        actor_lr=float(training.get("actor_lr", 3e-4)),
        critic_lr=float(training.get("critic_lr", 3e-4)),
        eta=eta,
        eta_end=eta_end,
        potential_update_mode=(
            "frozen" if algorithm in {"mappo", "ippo", "matd3", "rule_planner"}
            else str(training.get("potential_update_mode", "normal"))
        ),
        safety_reward_coef=float(env.get("safety_reward_coef", 1.0)),
        intent_reward_profiles_enabled=bool(
            variant.get(
                "intent_reward_profiles_enabled",
                env.get("intent_reward_profiles_enabled", False),
            )
        ),
        wind_std=float(env.get("wind_std", 0.0)),
        observation_noise_std=float(env.get("observation_noise_std", 0.0)),
        action_delay_steps=int(env.get("action_delay_steps", 0)),
        communication_dropout_prob=float(env.get("communication_dropout_prob", 0.0)),
        device=str(training.get("device", "cpu")),
        seed=int(seed),
    )
    setattr(cfg, "task_reward_coef", float(env.get("task_reward_coef", 1.20)))
    setattr(cfg, "environment_name", environment_name)
    return cfg


def run_seed(spec: Dict[str, object], variant: Dict[str, object], seed: int) -> Dict[str, object]:
    import envs.uav_scheduling_env  # noqa: F401
    from imappo import (
        build_uav_env_factory,
        evaluate_dynamic_intent_switch,
        evaluate_imappo,
        train_imappo,
    )

    cfg = build_config(spec, variant, seed)
    environment_name = str(spec["environment"]["name"])
    if environment_name.startswith("vmas:"):
        from envs.vmas_adapter import VMASAdapter

        scenario = environment_name.split(":", 1)[1]

        def train_factory():
            return VMASAdapter(
                scenario=scenario,
                n_agents=cfg.n_agents,
                max_steps=cfg.max_steps,
            )
    else:
        train_factory = build_uav_env_factory(cfg, "train")
    if cfg.algorithm == "rule_planner":
        from rule_based_baseline import RuleBasedUAVPolicy

        algo = RuleBasedUAVPolicy(cfg)
        logs = []
    elif cfg.algorithm == "matd3":
        from matd3_baseline import train_matd3

        algo, logs = train_matd3(train_factory, cfg)
    else:
        algo, logs = train_imappo(
            env_factory=train_factory,
            eval_env_factory=build_risk_factory(
                cfg, next(iter(spec["evaluation"]["risk_tiers"].values()))
            ),
            collision_probe_env_factory=(
                None if environment_name.startswith("vmas:")
                else build_uav_env_factory(cfg, "collision_probe")
            ),
            config=cfg,
        )
    tier_results = {}
    for tier_index, (tier_name, tier) in enumerate(spec["evaluation"]["risk_tiers"].items()):
        mode = str(tier.get("intent_mode", "dense"))
        tier_results[tier_name] = evaluate_imappo(
            algo,
            build_risk_factory(cfg, tier),
            cfg,
            prefix=tier_name,
            evaluation_mode=mode,
            evaluation_seed_offset=10_000 + tier_index * 1_000,
        )
    result = {
        "seed": int(seed),
        "variant": variant,
        "config": cfg.__dict__,
        "intent_representation": algo.intent_representation_metadata(),
        "tier_metrics": tier_results,
        "logs": logs,
    }
    if "generalization" in spec:
        from intent_generalization import (
            intent_behavior_controllability,
            load_generalization_suite,
            objective_profile_prediction_diagnostics,
            representation_retrieval_diagnostics,
            resolve_query_objective_profile,
        )

        suite = load_generalization_suite(ROOT / str(spec["generalization"]["suite"]))
        queries = list(suite["queries"])
        entries = [
            (str(query["canonical_label"]), str(query["description"]))
            for query in queries
        ]
        query_vectors = algo.encode_intent_queries(entries)
        configured_behavior_keys = spec["generalization"].get("behavior_query_keys")
        if configured_behavior_keys is None:
            behavior_queries = queries
        else:
            requested = {str(key) for key in configured_behavior_keys}
            available = {str(query["key"]) for query in queries}
            missing = sorted(requested - available)
            if missing:
                raise ValueError(f"unknown generalization behavior_query_keys: {missing}")
            behavior_queries = [
                query for query in queries if str(query["key"]) in requested
            ]
        geometry = {}
        if getattr(algo, "intent_library", None) is not None:
            geometry = representation_retrieval_diagnostics(
                algo.intent_library,
                query_vectors.detach().cpu().numpy(),
                queries,
            )
        if getattr(algo, "objective_semantic_adapter", None) is not None:
            geometry["profile_prediction"] = objective_profile_prediction_diagnostics(
                queries,
                algo.objective_semantic_adapter.predict_profiles(entries),
            )
        behavior = {}
        for query in behavior_queries:
            query_index = next(
                index for index, candidate in enumerate(queries)
                if str(candidate["key"]) == str(query["key"])
            )
            query_tiers = {}
            for tier_index, (tier_name, tier) in enumerate(spec["evaluation"]["risk_tiers"].items()):
                raw_metrics = evaluate_imappo(
                    algo,
                    build_risk_factory(cfg, tier),
                    cfg,
                    prefix="query",
                    evaluation_mode=str(tier.get("intent_mode", "standard")),
                    intent_override=query_vectors[query_index],
                    intent_label_override=str(query["canonical_label"]),
                    tactical_posture_override=str(query["posture"]),
                    objective_profile_override=resolve_query_objective_profile(query),
                    evaluation_seed_offset=(
                        100_000 + tier_index * 1_000
                    ),
                )
                query_tiers[str(tier_name)] = {
                    key.removeprefix("query_"): value for key, value in raw_metrics.items()
                }
            behavior[str(query["key"])] = {
                "split": str(query["split"]),
                "canonical_label": str(query["canonical_label"]),
                "posture": str(query["posture"]),
                "contrast_group": query.get("contrast_group"),
                "objective_profile": query.get("objective_profile"),
                "risk_tiers": query_tiers,
            }
        result["intent_generalization"] = {
            "suite_id": suite["suite_id"],
            "train_labels": suite["train_labels"],
            "representation_diagnostics": geometry,
            "behavior": behavior,
            "controllability": intent_behavior_controllability(
                behavior_queries, behavior
            ),
        }
        if "dynamic_intent" in spec:
            dynamic_spec = spec["dynamic_intent"]
            query_indices = {
                str(query["key"]): index for index, query in enumerate(queries)
            }
            configured_tiers = dynamic_spec.get(
                "risk_tiers", list(spec["evaluation"]["risk_tiers"])
            )
            dynamic_results = {}
            for transition_index, transition in enumerate(dynamic_spec["transitions"]):
                from_key = str(transition["from_query"])
                to_key = str(transition["to_query"])
                if from_key not in query_indices or to_key not in query_indices:
                    raise ValueError(
                        f"dynamic transition references unknown query: {from_key}->{to_key}"
                    )
                pre_query = queries[query_indices[from_key]]
                post_query = queries[query_indices[to_key]]
                tier_results_dynamic = {}
                for tier_index, tier_name in enumerate(configured_tiers):
                    tier = spec["evaluation"]["risk_tiers"][str(tier_name)]
                    tier_results_dynamic[str(tier_name)] = evaluate_dynamic_intent_switch(
                        algo,
                        build_risk_factory(cfg, tier),
                        cfg,
                        pre_intent=query_vectors[query_indices[from_key]],
                        post_intent=query_vectors[query_indices[to_key]],
                        pre_label=str(pre_query["canonical_label"]),
                        post_label=str(post_query["canonical_label"]),
                        pre_posture=str(pre_query["posture"]),
                        post_posture=str(post_query["posture"]),
                        pre_objective_profile=resolve_query_objective_profile(pre_query),
                        post_objective_profile=resolve_query_objective_profile(post_query),
                        switch_step=int(dynamic_spec.get("switch_step", 10)),
                        total_steps=int(dynamic_spec.get("total_steps", 30)),
                        response_threshold=float(
                            dynamic_spec.get("response_threshold", 0.05)
                        ),
                        # Transitions share reset seeds within a tier. This is a
                        # deliberate common-random-number intervention design.
                        evaluation_seed_offset=500_000 + tier_index * 10_000,
                    )
                dynamic_results[str(transition["key"])] = {
                    "from_query": from_key,
                    "to_query": to_key,
                    "expected_objective": str(transition["expected_objective"]),
                    "risk_tiers": tier_results_dynamic,
                }
            result["dynamic_intent"] = {
                "switch_step": int(dynamic_spec.get("switch_step", 10)),
                "total_steps": int(dynamic_spec.get("total_steps", 30)),
                "response_threshold": float(
                    dynamic_spec.get("response_threshold", 0.05)
                ),
                "transitions": dynamic_results,
            }
    return result


def summarize_variant(results: List[Dict[str, object]], bootstrap_seed: int) -> Dict[str, object]:
    from research_statistics import summarize_sample

    summary = {"seed_count": len(results), "risk_tiers": {}}
    tier_names = results[0]["tier_metrics"].keys()
    for tier in tier_names:
        tier_summary = {}
        for metric, _ in EVALUATION_METRIC_DIRECTIONS:
            key = f"{tier}_{metric}"
            if not all(key in result["tier_metrics"][tier] for result in results):
                continue
            values = [float(result["tier_metrics"][tier][key]) for result in results]
            tier_summary[metric] = summarize_sample(values, seed=bootstrap_seed)
            bootstrap_seed += 2
        summary["risk_tiers"][tier] = tier_summary
    generalization_results = [
        result.get("intent_generalization") for result in results
    ]
    if all(generalization_results):
        first = generalization_results[0]
        behavior = first["behavior"]
        split_names = sorted({str(item["split"]) for item in behavior.values()})
        generalization_summary = {
            "suite_id": first["suite_id"],
            "representation_diagnostics": first["representation_diagnostics"],
            "behavior_by_split": {},
        }
        for split in split_names:
            split_summary = {}
            query_keys = [
                key for key, item in behavior.items() if str(item["split"]) == split
            ]
            for tier in behavior[query_keys[0]]["risk_tiers"]:
                tier_summary = {}
                for metric, _ in EVALUATION_METRIC_DIRECTIONS:
                    if not all(
                        metric in generalization["behavior"][key]["risk_tiers"][tier]
                        for generalization in generalization_results
                        for key in query_keys
                    ):
                        continue
                    per_seed = []
                    for generalization in generalization_results:
                        per_seed.append(
                            float(np.mean([
                                generalization["behavior"][key]["risk_tiers"][tier][metric]
                                for key in query_keys
                            ]))
                        )
                    tier_summary[metric] = summarize_sample(per_seed, seed=bootstrap_seed)
                    bootstrap_seed += 2
                split_summary[tier] = tier_summary
            generalization_summary["behavior_by_split"][split] = split_summary
        if all(item.get("controllability") for item in generalization_results):
            controllability_summary = {}
            first_controllability = generalization_results[0]["controllability"]
            for tier, scopes in first_controllability.items():
                tier_summary = {}
                for scope, metrics in scopes.items():
                    scope_summary = {"n_queries": int(metrics["n_queries"])}
                    for metric in (
                        "safety_tradeoff_spearman",
                        "collision_preference_spearman",
                        "task_preference_spearman",
                        "energy_preference_spearman",
                        "distance_preference_spearman",
                        "time_preference_spearman",
                        "safety_distance_spearman",
                        "collision_distance_spearman",
                        "threat_preference_spearman",
                        "collision_rate_range",
                        "task_completion_range",
                    ):
                        if metric not in metrics:
                            continue
                        values = [
                            item["controllability"][tier][scope][metric]
                            for item in generalization_results
                        ]
                        if all(value is not None for value in values):
                            scope_summary[metric] = summarize_sample(
                                values, seed=bootstrap_seed
                            )
                            bootstrap_seed += 2
                    tier_summary[scope] = scope_summary
                controllability_summary[tier] = tier_summary
            generalization_summary["controllability"] = controllability_summary
        summary["intent_generalization"] = generalization_summary
    dynamic_results = [result.get("dynamic_intent") for result in results]
    if all(dynamic_results):
        dynamic_summary = {
            "switch_step": dynamic_results[0]["switch_step"],
            "total_steps": dynamic_results[0]["total_steps"],
            "response_threshold": dynamic_results[0]["response_threshold"],
            "transitions": {},
        }
        for transition_key, first_transition in dynamic_results[0]["transitions"].items():
            transition_summary = {
                "from_query": first_transition["from_query"],
                "to_query": first_transition["to_query"],
                "expected_objective": first_transition["expected_objective"],
                "risk_tiers": {},
            }
            for tier in first_transition["risk_tiers"]:
                tier_summary = {}
                for metric in (
                    "response_rate", "censored_response_latency_steps",
                    "switch_action_delta", "mean_post_switch_action_delta",
                ):
                    tier_summary[metric] = summarize_sample(
                        [item["transitions"][transition_key]["risk_tiers"][tier][metric]
                         for item in dynamic_results],
                        seed=bootstrap_seed,
                    )
                    bootstrap_seed += 2
                tier_summary["post_minus_pre"] = {}
                for metric in first_transition["risk_tiers"][tier]["post_minus_pre"]:
                    tier_summary["post_minus_pre"][metric] = summarize_sample(
                        [item["transitions"][transition_key]["risk_tiers"][tier]
                         ["post_minus_pre"][metric] for item in dynamic_results],
                        seed=bootstrap_seed,
                    )
                    bootstrap_seed += 2
                transition_summary["risk_tiers"][tier] = tier_summary
            dynamic_summary["transitions"][transition_key] = transition_summary
        summary["dynamic_intent"] = dynamic_summary
    return summary


def summarize_comparisons(
    results_by_variant: Dict[str, List[Dict[str, object]]],
    treatment_key: str,
    bootstrap_seed: int,
) -> Dict[str, object]:
    from research_statistics import paired_difference_summary

    if treatment_key not in results_by_variant:
        return {}
    treatment = sorted(results_by_variant[treatment_key], key=lambda item: int(item["seed"]))
    comparisons = {}
    for baseline_key, baseline_results in results_by_variant.items():
        if baseline_key == treatment_key:
            continue
        baseline = sorted(baseline_results, key=lambda item: int(item["seed"]))
        if [item["seed"] for item in treatment] != [item["seed"] for item in baseline]:
            raise ValueError(f"paired seeds do not match for {treatment_key} vs {baseline_key}")
        comparison = {"risk_tiers": {}}
        for tier in treatment[0]["tier_metrics"]:
            tier_comparison = {}
            for metric, lower_is_better in EVALUATION_METRIC_DIRECTIONS:
                key = f"{tier}_{metric}"
                if not all(
                    key in item["tier_metrics"][tier]
                    for item in treatment + baseline
                ):
                    continue
                treatment_values = [item["tier_metrics"][tier][key] for item in treatment]
                baseline_values = [item["tier_metrics"][tier][key] for item in baseline]
                tier_comparison[metric] = paired_difference_summary(
                    treatment_values,
                    baseline_values,
                    lower_is_better=lower_is_better,
                    seed=bootstrap_seed,
                )
                bootstrap_seed += 1
            comparison["risk_tiers"][tier] = tier_comparison
        treatment_generalization = treatment[0].get("intent_generalization")
        baseline_generalization = baseline[0].get("intent_generalization")
        if treatment_generalization and baseline_generalization:
            comparison["intent_generalization"] = {}
            treatment_behavior = treatment_generalization["behavior"]
            split_names = sorted({
                str(item["split"]) for item in treatment_behavior.values()
            })
            for split in split_names:
                treatment_query_keys = [
                    key for key, item in treatment_behavior.items()
                    if str(item["split"]) == split
                ]
                baseline_query_keys = [
                    key for key, item in baseline_generalization["behavior"].items()
                    if str(item["split"]) == split
                ]
                if treatment_query_keys != baseline_query_keys:
                    raise ValueError(
                        f"generalization queries do not match for {treatment_key} vs {baseline_key}"
                    )
                split_comparison = {}
                for tier in treatment_behavior[treatment_query_keys[0]]["risk_tiers"]:
                    tier_comparison = {}
                    for metric, lower_is_better in EVALUATION_METRIC_DIRECTIONS:
                        if not all(
                            metric in item["intent_generalization"]["behavior"][key]
                            ["risk_tiers"][tier]
                            for item in treatment + baseline
                            for key in treatment_query_keys
                        ):
                            continue
                        treatment_values = [
                            float(np.mean([
                                item["intent_generalization"]["behavior"][key]
                                ["risk_tiers"][tier][metric]
                                for key in treatment_query_keys
                            ]))
                            for item in treatment
                        ]
                        baseline_values = [
                            float(np.mean([
                                item["intent_generalization"]["behavior"][key]
                                ["risk_tiers"][tier][metric]
                                for key in baseline_query_keys
                            ]))
                            for item in baseline
                        ]
                        tier_comparison[metric] = paired_difference_summary(
                            treatment_values,
                            baseline_values,
                            lower_is_better=lower_is_better,
                            seed=bootstrap_seed,
                        )
                        bootstrap_seed += 1
                    split_comparison[tier] = tier_comparison
                comparison["intent_generalization"][split] = split_comparison
            if (
                treatment_generalization.get("controllability")
                and baseline_generalization.get("controllability")
            ):
                controllability_comparison = {}
                first_controllability = treatment_generalization["controllability"]
                for tier, scopes in first_controllability.items():
                    tier_comparison = {}
                    for scope in scopes:
                        scope_comparison = {}
                        for metric in (
                            "safety_tradeoff_spearman",
                            "collision_preference_spearman",
                            "task_preference_spearman",
                            "energy_preference_spearman",
                            "distance_preference_spearman",
                            "time_preference_spearman",
                            "safety_distance_spearman",
                            "collision_distance_spearman",
                            "threat_preference_spearman",
                        ):
                            if metric not in scopes[scope]:
                                continue
                            treatment_values = [
                                item["intent_generalization"]["controllability"]
                                [tier][scope][metric]
                                for item in treatment
                            ]
                            baseline_values = [
                                item["intent_generalization"]["controllability"]
                                [tier][scope][metric]
                                for item in baseline
                            ]
                            if all(value is not None for value in treatment_values + baseline_values):
                                scope_comparison[metric] = paired_difference_summary(
                                    treatment_values,
                                    baseline_values,
                                    lower_is_better=False,
                                    seed=bootstrap_seed,
                                )
                                bootstrap_seed += 1
                        tier_comparison[scope] = scope_comparison
                    controllability_comparison[tier] = tier_comparison
                comparison["intent_generalization"]["controllability"] = (
                    controllability_comparison
                )
        comparisons[baseline_key] = comparison
    return comparisons


def write_checksums(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    spec = read_json(args.config)
    validate_spec(spec, args.allow_dirty)
    output_dir = output_dir_for(spec)
    plan = {
        "output_dir": str(output_dir),
        "level": spec["level"],
        "seeds": spec["seeds"],
        "variants": [item["key"] for item in spec["variants"]],
        "total_training_runs": len(spec["seeds"]) * len(spec["variants"]),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite existing study: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(spec, sys.argv)
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "config.json", spec)

    summaries = {}
    results_by_variant = {}
    for variant in spec["variants"]:
        variant_results = []
        for seed in spec["seeds"]:
            result_path = output_dir / str(variant["key"]) / f"seed_{seed}" / "result.json"
            if args.resume and result_path.exists():
                result = read_json(result_path)
            else:
                result = run_seed(spec, variant, int(seed))
                write_json(result_path, result)
            variant_results.append(result)
        summaries[str(variant["key"])] = summarize_variant(
            variant_results,
            bootstrap_seed=int(spec.get("bootstrap_seed", 20260801)),
        )
        results_by_variant[str(variant["key"])] = variant_results
    summary_payload = {
        "variants": summaries,
        "paired_comparisons": summarize_comparisons(
            results_by_variant,
            treatment_key=str(spec.get("treatment_key", "pretrained_semantic")),
            bootstrap_seed=int(spec.get("bootstrap_seed", 20260801)) + 1000,
        ),
    }
    write_json(output_dir / "summary.json", summary_payload)
    write_result_card(output_dir, spec, summaries)
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(output_dir / "manifest.json", manifest)
    write_checksums(output_dir)
    print(json.dumps({**plan, "status": "complete"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
