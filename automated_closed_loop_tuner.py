from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envs.uav_scheduling_env import infer_obs_dim, infer_state_dim
import envs.uav_scheduling_env  # noqa: F401
from imappo import (
    IMAPPO,
    IMAPPOConfig,
    build_global_state,
    build_uav_env_factory,
    env_reset,
    env_step,
    evaluate_imappo,
    infer_agent_order,
    normalise_obs,
    set_env_intent,
    set_env_tactical_posture,
    train_imappo,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    safety_reward_coef: float
    eta: float
    eta_end: float
    hard_train_interval: int
    collision_probe_spawn_scale: float
    collision_probe_separation_scale: float
    hard_train_spawn_scale: float
    hard_train_separation_scale: float


DEFAULT_CANDIDATES = [
    Candidate("baseline_semantic", 1.0, 0.5, 0.1, 6, 0.29, 0.82, 0.31, 0.86),
    Candidate("safer_dense", 1.4, 0.45, 0.08, 4, 0.31, 0.86, 0.30, 0.90),
    Candidate("high_safety", 1.8, 0.35, 0.05, 3, 0.33, 0.90, 0.29, 0.94),
]


def build_custom_uav_factory(
    n_agents: int,
    n_targets: int,
    obs_dim: int,
    spawn_region_scale: float,
    spawn_separation_scale: float,
    safety_reward_coef: float,
):
    import gymnasium as gym

    def make_env():
        return gym.make(
            "uav-scheduling-v0",
            n_agents=n_agents,
            n_targets=n_targets,
            obs_dim=obs_dim,
            spawn_region_scale=spawn_region_scale,
            spawn_separation_scale=spawn_separation_scale,
            safety_reward_coef=safety_reward_coef,
        )

    return make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Closed-loop multi-round tuner for semantic-library I-MAPPO."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/closed_loop_tuning"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23, 42, 100])
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--n-agents", type=int, default=8)
    parser.add_argument("--n-targets", type=int, default=6)
    parser.add_argument("--intent-dim", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true", help="Run one short candidate for pipeline validation.")
    return parser.parse_args()


def build_config(args: argparse.Namespace, candidate: Candidate, seed: int) -> IMAPPOConfig:
    obs_dim = infer_obs_dim(args.n_agents)
    state_dim = infer_state_dim(args.n_agents, obs_dim)
    return IMAPPOConfig(
        algorithm="imappo",
        critic_mode="attention",
        use_action_mask=True,
        intent_source="semantic_library",
        n_agents=args.n_agents,
        n_targets=args.n_targets,
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=3,
        intent_dim=args.intent_dim,
        max_episodes=args.episodes,
        max_steps=args.steps,
        rollout_length=args.rollout,
        minibatch_size=args.batch_size,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        safety_reward_coef=candidate.safety_reward_coef,
        eta=candidate.eta,
        eta_end=candidate.eta_end,
        hard_train_interval=candidate.hard_train_interval,
        collision_probe_spawn_scale=candidate.collision_probe_spawn_scale,
        collision_probe_separation_scale=candidate.collision_probe_separation_scale,
        hard_train_spawn_scale=candidate.hard_train_spawn_scale,
        hard_train_separation_scale=candidate.hard_train_separation_scale,
        device=args.device,
        seed=seed,
    )


def evaluate_tiers(algo: IMAPPO, cfg: IMAPPOConfig) -> Dict[str, Dict[str, float]]:
    obs_dim = cfg.obs_dim
    factories = {
        "mid": build_custom_uav_factory(
            cfg.n_agents, cfg.n_targets, obs_dim, 0.37, 0.86, cfg.safety_reward_coef
        ),
        "hard": build_custom_uav_factory(
            cfg.n_agents, cfg.n_targets, obs_dim, 0.33, 0.80, cfg.safety_reward_coef
        ),
    }
    return {
        name: evaluate_imappo(algo, factory, cfg, prefix=name, evaluation_mode="dense")
        for name, factory in factories.items()
    }


def measure_replanning_latency(algo: IMAPPO, cfg: IMAPPOConfig, seed: int) -> float:
    env = build_custom_uav_factory(
        cfg.n_agents, cfg.n_targets, cfg.obs_dim, 0.32, 0.90, cfg.safety_reward_coef
    )()
    obs_data, _ = env.reset(seed=seed)
    agent_order = infer_agent_order(env, obs_data, cfg)
    obs_array = normalise_obs(agent_order, obs_data)
    mutation_step = min(30, max(1, cfg.max_steps // 2))
    distances: List[float] = []

    latency = float(cfg.max_steps)
    for step in range(cfg.max_steps):
        mode = "standard" if step < mutation_step else "dense"
        posture = "attack" if step < mutation_step else "stealth"
        intent, mask, label = algo.evaluation_intent_and_mask(mode=mode)
        set_env_intent(env, intent, label)
        set_env_tactical_posture(env, posture)

        obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=algo.device)
        actions, _ = algo.select_actions(obs_tensor, intent, mask, deterministic=True)
        next_obs_data, _, done, truncated, _ = env_step(
            env, agent_order, actions.detach().cpu().numpy()
        )
        base_env = getattr(env, "unwrapped", env)
        state = build_global_state(normalise_obs(agent_order, next_obs_data), cfg)
        _ = state  # Keep the same state-building path used during training.
        avg_distance = float(np.mean(np.linalg.norm(base_env.targets - base_env.positions, axis=1)))
        distances.append(avg_distance)

        if step >= mutation_step + 3 and latency == float(cfg.max_steps):
            recent = distances[-4:]
            if recent[1] > recent[0] and recent[2] > recent[1] and recent[3] > recent[2]:
                latency = float(step - mutation_step)
        obs_array = normalise_obs(agent_order, next_obs_data)
        if done or truncated:
            break

    env.close()
    return latency


def run_candidate(
    args: argparse.Namespace,
    candidate: Candidate,
    round_idx: int,
) -> Dict[str, object]:
    candidate_dir = args.output_dir / f"round_{round_idx:02d}" / candidate.name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    seed_results = []

    for seed in args.seeds:
        cfg = build_config(args, candidate, seed)
        train_factory = build_uav_env_factory(cfg, mode="train")
        eval_factory = build_uav_env_factory(cfg, mode="eval")
        probe_factory = build_uav_env_factory(cfg, mode="collision_probe")
        algo, logs = train_imappo(
            env_factory=train_factory,
            eval_env_factory=eval_factory,
            collision_probe_env_factory=probe_factory,
            config=cfg,
        )
        tiers = evaluate_tiers(algo, cfg)
        latency = measure_replanning_latency(algo, cfg, seed)
        seed_dir = candidate_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        algo.save_checkpoint(str(seed_dir / "checkpoint_latest.pt"), extra={"seed": seed})
        result = {
            "seed": seed,
            "candidate": asdict(candidate),
            "config": cfg.__dict__,
            "tier_metrics": tiers,
            "replanning_latency": latency,
            "logs": logs,
        }
        with open(seed_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        seed_results.append(result)

    summary = summarise_candidate(candidate, seed_results)
    with open(candidate_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def metric_values(seed_results: List[Dict[str, object]], tier: str, key: str) -> List[float]:
    return [
        float(seed_result["tier_metrics"][tier][f"{tier}_{key}"])
        for seed_result in seed_results
    ]


def summarise_candidate(candidate: Candidate, seed_results: List[Dict[str, object]]) -> Dict[str, object]:
    mid_collision = metric_values(seed_results, "mid", "collision_rate")
    hard_collision = metric_values(seed_results, "hard", "collision_rate")
    mid_task = metric_values(seed_results, "mid", "task_completion")
    hard_task = metric_values(seed_results, "hard", "task_completion")
    latencies = [float(seed_result["replanning_latency"]) for seed_result in seed_results]
    summary = {
        "candidate": asdict(candidate),
        "seed_count": len(seed_results),
        "mid_collision_mean": float(np.mean(mid_collision)),
        "hard_collision_mean": float(np.mean(hard_collision)),
        "mid_task_completion_mean": float(np.mean(mid_task)),
        "hard_task_completion_mean": float(np.mean(hard_task)),
        "replanning_latency_mean": float(np.mean(latencies)),
    }
    summary["success"] = bool(
        summary["mid_collision_mean"] < 0.30
        and summary["hard_collision_mean"] < 0.30
        and summary["mid_task_completion_mean"] >= 0.75
        and summary["hard_task_completion_mean"] >= 0.75
        and summary["replanning_latency_mean"] <= 3.5
    )
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = DEFAULT_CANDIDATES[:1] if args.smoke else DEFAULT_CANDIDATES
    all_summaries: List[Dict[str, object]] = []

    for round_idx in range(1, args.max_rounds + 1):
        for candidate in candidates:
            summary = run_candidate(args, candidate, round_idx)
            all_summaries.append(summary)
            if summary["success"]:
                final = {"status": "success", "round": round_idx, "summary": summary}
                with open(args.output_dir / "closed_loop_summary.json", "w", encoding="utf-8") as f:
                    json.dump({"final": final, "all_summaries": all_summaries}, f, indent=2, ensure_ascii=False)
                print(json.dumps(final, indent=2, ensure_ascii=False))
                return
        if args.smoke:
            break

    final = {"status": "not_met", "summary": all_summaries[-1] if all_summaries else None}
    with open(args.output_dir / "closed_loop_summary.json", "w", encoding="utf-8") as f:
        json.dump({"final": final, "all_summaries": all_summaries}, f, indent=2, ensure_ascii=False)
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
