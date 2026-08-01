from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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


SEEDS = [7, 11, 23, 42, 100]
RISK_TIERS = {
    "easy": (0.42, 0.92, "standard"),
    "medium": (0.37, 0.86, "dense"),
    "hard": (0.33, 0.80, "dense"),
}
SUCCESS_THRESHOLDS = {
    "collision_rate_medium_hard": 0.30,
    "task_completion": 0.75,
    "replanning_latency": 3.5,
}


@dataclass
class TuningState:
    lambda_3: float = 1.0
    lambda_1: float = 1.20
    eta: float = 0.5
    eta_end: float = 0.1
    eps_clip: float = 0.1
    critic_lr: float = 3e-4
    attention_dim: int = 128
    hard_train_interval: int = 6
    hard_train_spawn_scale: float = 0.31
    hard_train_separation_scale: float = 0.86
    collision_probe_spawn_scale: float = 0.29
    collision_probe_separation_scale: float = 0.82


@dataclass(frozen=True)
class VariantSpec:
    key: str
    display_name: str
    family: str
    intent_source: str = "pretrained_semantic"
    algorithm: str = "imappo"
    critic_mode: str = "attention"
    use_action_mask: bool = True
    eta_override: Optional[float] = None
    eta_end_override: Optional[float] = None
    runnable_note: str = ""


BASELINE_VARIANTS = [
    VariantSpec(
        "imappo_pretrained_semantic",
        "I-MAPPO (Pretrained Semantic)",
        "baseline",
    ),
    VariantSpec("imappo_onehot", "I-MAPPO (One-hot)", "baseline", intent_source="onehot"),
    VariantSpec(
        "imappo_random_dense",
        "I-MAPPO (Random Dense Code)",
        "representation_control",
        intent_source="random_dense",
    ),
    VariantSpec(
        "imappo_legacy_hash",
        "I-MAPPO (Legacy Hash)",
        "representation_control",
        intent_source="legacy_hash",
    ),
    VariantSpec(
        "mappo",
        "MAPPO",
        "baseline",
        intent_source="onehot",
        algorithm="mappo",
        critic_mode="uniform",
        use_action_mask=False,
        eta_override=0.0,
        eta_end_override=0.0,
    ),
]

ABLATION_VARIANTS = [
    VariantSpec("ablation_full", "I-MAPPO (Full)", "ablation"),
    VariantSpec(
        "ablation_no_masking",
        "I-MAPPO (w/o Masking)",
        "ablation",
        use_action_mask=False,
    ),
    VariantSpec(
        "ablation_no_reward",
        "I-MAPPO (w/o Reward)",
        "ablation",
        eta_override=0.0,
        eta_end_override=0.0,
    ),
    VariantSpec(
        "ablation_no_attn",
        "I-MAPPO (w/o Attn)",
        "ablation",
        critic_mode="mlp",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pilot-only closed-loop tuner for intent-conditioned I-MAPPO."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/pilot/semantic_intent_uav"),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("experiments/pilot/closed_loop_tuning"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--n-agents", type=int, default=8)
    parser.add_argument("--n-targets", type=int, default=6)
    parser.add_argument("--intent-dim", type=int, default=64)
    parser.add_argument(
        "--intent-encoder-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--intent-encoder-revision",
        default="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    )
    parser.add_argument("--intent-projection-seed", type=int, default=0)
    parser.add_argument("--intent-code-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Short pipeline validation only.")
    return parser.parse_args()


def make_uav_factory(
    cfg: IMAPPOConfig,
    spawn_region_scale: float,
    spawn_separation_scale: float,
):
    import gymnasium as gym

    def make_env():
        return gym.make(
            "uav-scheduling-v0",
            n_agents=cfg.n_agents,
            n_targets=cfg.n_targets,
            obs_dim=cfg.obs_dim,
            spawn_region_scale=spawn_region_scale,
            spawn_separation_scale=spawn_separation_scale,
            safety_reward_coef=cfg.safety_reward_coef,
            task_reward_coef=getattr(cfg, "task_reward_coef", 1.20),
        )

    return make_env


def build_config(args: argparse.Namespace, state: TuningState, variant: VariantSpec, seed: int) -> IMAPPOConfig:
    obs_dim = infer_obs_dim(args.n_agents)
    cfg = IMAPPOConfig(
        algorithm=variant.algorithm,
        critic_mode=variant.critic_mode,
        use_action_mask=variant.use_action_mask,
        intent_source=variant.intent_source,
        n_agents=args.n_agents,
        n_targets=args.n_targets,
        obs_dim=obs_dim,
        state_dim=infer_state_dim(args.n_agents, obs_dim),
        action_dim=3,
        intent_dim=args.intent_dim,
        intent_encoder_model=args.intent_encoder_model,
        intent_encoder_revision=args.intent_encoder_revision,
        intent_projection_seed=args.intent_projection_seed,
        intent_code_seed=args.intent_code_seed,
        max_episodes=args.episodes,
        max_steps=args.steps,
        rollout_length=args.rollout,
        minibatch_size=args.batch_size,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        eps_clip=state.eps_clip,
        critic_lr=state.critic_lr,
        attention_dim=state.attention_dim,
        safety_reward_coef=state.lambda_3,
        eta=state.eta if variant.eta_override is None else variant.eta_override,
        eta_end=state.eta_end if variant.eta_end_override is None else variant.eta_end_override,
        hard_train_interval=state.hard_train_interval,
        collision_probe_spawn_scale=state.collision_probe_spawn_scale,
        collision_probe_separation_scale=state.collision_probe_separation_scale,
        hard_train_spawn_scale=state.hard_train_spawn_scale,
        hard_train_separation_scale=state.hard_train_separation_scale,
        device=args.device,
        seed=seed,
    )
    setattr(cfg, "task_reward_coef", state.lambda_1)
    return cfg


def evaluate_risk_tiers(algo: IMAPPO, cfg: IMAPPOConfig) -> Dict[str, Dict[str, float]]:
    results = {}
    for tier, (spawn, separation, mode) in RISK_TIERS.items():
        results[tier] = evaluate_imappo(
            algo,
            make_uav_factory(cfg, spawn, separation),
            cfg,
            prefix=tier,
            evaluation_mode=mode,
        )
    return results


def measure_replanning_latency(algo: IMAPPO, cfg: IMAPPOConfig, seed: int) -> float:
    env = make_uav_factory(cfg, 0.32, 0.90)()
    obs_data, _ = env_reset(env)
    agent_order = infer_agent_order(env, obs_data, cfg)
    obs_array = normalise_obs(agent_order, obs_data)
    mutation_step = 30
    distance_series: List[float] = []
    latency = float(cfg.max_steps)

    for step in range(cfg.max_steps):
        mode = "attack_probe" if step < mutation_step else "stealth_probe"
        posture = "attack" if step < mutation_step else "stealth"
        intent, mask, label = algo.evaluation_intent_and_mask(mode=mode)
        set_env_intent(env, intent, label)
        set_env_tactical_posture(env, posture)
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=algo.device)
        actions, _ = algo.select_actions(obs_tensor, intent, mask, deterministic=True)
        next_obs_data, _, done, truncated, _ = env_step(env, agent_order, actions.detach().cpu().numpy())
        base_env = getattr(env, "unwrapped", env)
        distance_series.append(float(np.mean(np.linalg.norm(base_env.targets - base_env.positions, axis=1))))
        if step >= mutation_step + 3 and latency == float(cfg.max_steps):
            recent = distance_series[-4:]
            if recent[1] > recent[0] and recent[2] > recent[1] and recent[3] > recent[2]:
                latency = float(step - mutation_step)
        obs_array = normalise_obs(agent_order, next_obs_data)
        if done or truncated:
            break

    env.close()
    return latency


def early_collision_rate(logs: Iterable[Dict[str, object]]) -> float:
    values = [
        float(item["episode_collision_rate"])
        for item in logs
        if "episode_collision_rate" in item and int(float(item.get("episode", 999999))) < 500
    ]
    return float(np.mean(values)) if values else float("nan")


def run_seed(
    args: argparse.Namespace,
    state: TuningState,
    round_idx: int,
    variant: VariantSpec,
    seed: int,
) -> Dict[str, object]:
    torch.set_num_threads(max(1, int(getattr(args, "torch_threads", 1))))
    cfg = build_config(args, state, variant, seed)
    algo, logs = train_imappo(
        env_factory=build_uav_env_factory(cfg, mode="train"),
        eval_env_factory=build_uav_env_factory(cfg, mode="eval"),
        collision_probe_env_factory=build_uav_env_factory(cfg, mode="collision_probe"),
        config=cfg,
    )
    tiers = evaluate_risk_tiers(algo, cfg)
    latency = measure_replanning_latency(algo, cfg, seed)
    return {
        "round": round_idx,
        "seed": seed,
        "variant": asdict(variant),
        "config": cfg.__dict__,
        "intent_representation": algo.intent_representation_metadata(),
        "early_exploration_collision_rate": early_collision_rate(logs),
        "tier_metrics": tiers,
        "replanning_latency": latency,
        "logs": logs,
    }


def mean_metric(seed_results: List[Dict[str, object]], tier: str, suffix: str) -> float:
    return float(np.mean([
        float(result["tier_metrics"][tier][f"{tier}_{suffix}"])
        for result in seed_results
    ]))


def std_metric(seed_results: List[Dict[str, object]], tier: str, suffix: str) -> float:
    return float(np.std([
        float(result["tier_metrics"][tier][f"{tier}_{suffix}"])
        for result in seed_results
    ]))


def summarize_variant(variant: VariantSpec, seed_results: List[Dict[str, object]]) -> Dict[str, object]:
    summary = {
        "variant": asdict(variant),
        "seed_count": len(seed_results),
        "early_exploration_collision_rate_mean": float(np.mean([
            float(result["early_exploration_collision_rate"]) for result in seed_results
        ])),
        "replanning_latency_mean": float(np.mean([
            float(result["replanning_latency"]) for result in seed_results
        ])),
        "replanning_latency_std": float(np.std([
            float(result["replanning_latency"]) for result in seed_results
        ])),
        "risk_tiers": {},
    }
    for tier in RISK_TIERS:
        summary["risk_tiers"][tier] = {
            "collision_rate_mean": mean_metric(seed_results, tier, "collision_rate"),
            "collision_rate_std": std_metric(seed_results, tier, "collision_rate"),
            "task_completion_mean": mean_metric(seed_results, tier, "task_completion"),
            "task_completion_std": std_metric(seed_results, tier, "task_completion"),
        }
    return summary


def success_metrics(summary: Dict[str, object]) -> Dict[str, float]:
    tiers = summary["risk_tiers"]
    return {
        "medium_collision": float(tiers["medium"]["collision_rate_mean"]),
        "hard_collision": float(tiers["hard"]["collision_rate_mean"]),
        "medium_task_completion": float(tiers["medium"]["task_completion_mean"]),
        "hard_task_completion": float(tiers["hard"]["task_completion_mean"]),
        "replanning_latency": float(summary["replanning_latency_mean"]),
    }


def is_success(summary: Dict[str, object]) -> bool:
    m = success_metrics(summary)
    return (
        m["medium_collision"] < SUCCESS_THRESHOLDS["collision_rate_medium_hard"]
        and m["hard_collision"] < SUCCESS_THRESHOLDS["collision_rate_medium_hard"]
        and m["medium_task_completion"] >= SUCCESS_THRESHOLDS["task_completion"]
        and m["hard_task_completion"] >= SUCCESS_THRESHOLDS["task_completion"]
        and m["replanning_latency"] <= SUCCESS_THRESHOLDS["replanning_latency"]
    )


def tune_next(state: TuningState, summary: Dict[str, object]) -> Tuple[TuningState, List[str]]:
    m = success_metrics(summary)
    next_state = TuningState(**asdict(state))
    actions: List[str] = []
    if max(m["medium_collision"], m["hard_collision"]) >= SUCCESS_THRESHOLDS["collision_rate_medium_hard"]:
        next_state.lambda_3 *= 1.5
        next_state.eps_clip = max(0.06, next_state.eps_clip * 0.8)
        next_state.hard_train_interval = max(2, next_state.hard_train_interval - 1)
        actions.append("collision high: lambda_3 *= 1.5, eps_clip *= 0.8, hard_train_interval reduced")
    if min(m["medium_task_completion"], m["hard_task_completion"]) < SUCCESS_THRESHOLDS["task_completion"]:
        next_state.eta *= 1.2
        next_state.eta_end *= 1.2
        next_state.lambda_1 *= 1.1
        actions.append("task completion low: eta *= 1.2, lambda_1 *= 1.1")
    if m["replanning_latency"] > SUCCESS_THRESHOLDS["replanning_latency"]:
        next_state.critic_lr *= 1.25
        next_state.attention_dim = min(256, int(next_state.attention_dim * 2))
        actions.append("latency high: critic_lr *= 1.25, attention_dim increased")
    return next_state, actions


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def write_study_results(args: argparse.Namespace, final_payload: Dict[str, object]) -> Path:
    result_path = args.output_dir / "pilot_results.json"
    write_json(result_path, final_payload)
    return result_path


def write_study_report(args: argparse.Namespace, final_payload: Dict[str, object]) -> None:
    final = final_payload["final"]
    lines = [
        "# Pilot Closed-Loop MARL Validation Report",
        "",
        f"- Status: **{final['status']}**",
        f"- Tuning rounds executed: **{final['round']}**",
        f"- Episodes per training run: **{args.episodes}**",
        f"- Seeds: **{', '.join(str(seed) for seed in args.seeds)}**",
        f"- Output directory: `{args.output_dir.as_posix()}`",
        "",
        "## Success Criteria",
        "",
        "| Metric | Threshold | Final I-MAPPO (Pretrained Semantic) |",
        "| --- | ---: | ---: |",
    ]
    semantic = final_payload["rounds"][-1]["summaries"]["imappo_pretrained_semantic"]
    metrics = success_metrics(semantic)
    lines.extend([
        f"| Medium collision rate | < 0.30 | {metrics['medium_collision']:.4f} |",
        f"| Hard collision rate | < 0.30 | {metrics['hard_collision']:.4f} |",
        f"| Medium task completion | >= 0.75 | {metrics['medium_task_completion']:.4f} |",
        f"| Hard task completion | >= 0.75 | {metrics['hard_task_completion']:.4f} |",
        f"| Step-30 re-planning latency | <= 3.5 | {metrics['replanning_latency']:.4f} |",
        "",
        "## Optimal Hyperparameters",
        "",
        "```json",
        json.dumps(final["hyperparameters"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Notes",
        "",
        "- This is a pilot tuning report and is not eligible for the paper's frozen result tables.",
        "- Legacy hash and random dense codes are representation controls, not semantic methods.",
        "- Discrete QMIX/VDN are excluded from this continuous-action matrix rather than replaced by a mislabeled surrogate.",
    ])
    report_path = args.output_dir / "PILOT_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cleanup_workspace(paths: Iterable[Path]) -> None:
    for root in paths:
        if not root.exists():
            continue
        for pattern in ("*.pt", "*.ckpt", "events.out.tfevents*", "*.csv", "*.jsonl"):
            for path in root.rglob(pattern):
                path.unlink(missing_ok=True)
        for name in ("tensorboard", "tb_logs", "runs"):
            for path in root.rglob(name):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)


def run_plot_results(args: argparse.Namespace) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "plot_results.py"),
            "--stage7-json",
            str(args.output_dir / "pilot_results.json"),
            "--save_dir",
            str(args.output_dir),
        ],
        check=True,
    )


def run_round(args: argparse.Namespace, state: TuningState, round_idx: int) -> Dict[str, object]:
    variants = BASELINE_VARIANTS + ABLATION_VARIANTS
    if args.smoke:
        variants = [BASELINE_VARIANTS[0]]
        args.episodes = min(args.episodes, 3)
        args.eval_episodes = min(args.eval_episodes, 1)
        args.seeds = args.seeds[:1]

    round_dir = args.work_dir / f"round_{round_idx:02d}"
    seed_results_by_variant: Dict[str, List[Dict[str, object]]] = {}
    summaries: Dict[str, Dict[str, object]] = {}
    for variant in variants:
        print(f"[round {round_idx}] running {variant.display_name}", flush=True)
        variant_results = []
        max_workers = max(1, min(int(args.workers), len(args.seeds)))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_seed = {}
            for seed in args.seeds:
                print(f"[round {round_idx}] {variant.key} seed={seed} episodes={args.episodes}", flush=True)
                future = executor.submit(run_seed, args, state, round_idx, variant, seed)
                future_to_seed[future] = seed
            for future in concurrent.futures.as_completed(future_to_seed):
                seed = future_to_seed[future]
                result = future.result()
                variant_results.append(result)
                write_json(round_dir / variant.key / f"seed_{seed}" / "result.json", result)
                print(f"[round {round_idx}] {variant.key} seed={seed} complete", flush=True)
        variant_results.sort(key=lambda item: int(item["seed"]))
        seed_results_by_variant[variant.key] = variant_results
        summaries[variant.key] = summarize_variant(variant, variant_results)
        write_json(round_dir / variant.key / "summary.json", summaries[variant.key])

    semantic_summary = summaries["imappo_pretrained_semantic"]
    next_state, tuning_actions = tune_next(state, semantic_summary)
    round_payload = {
        "round": round_idx,
        "hyperparameters": asdict(state),
        "summaries": summaries,
        "success": is_success(semantic_summary),
        "tuning_actions": tuning_actions,
        "next_hyperparameters": asdict(next_state),
    }
    write_json(round_dir / "round_summary.json", round_payload)
    return round_payload


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.output_dir = ROOT / "experiments" / "smoke" / "semantic_intent_uav"
        args.work_dir = ROOT / "experiments" / "smoke" / "closed_loop_tuning"
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, args.torch_threads)))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(1, args.torch_threads)))
    torch.set_num_threads(max(1, args.torch_threads))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    state = TuningState()
    rounds: List[Dict[str, object]] = []
    final: Optional[Dict[str, object]] = None

    for round_idx in range(1, args.max_rounds + 1):
        round_payload = run_round(args, state, round_idx)
        rounds.append(round_payload)
        semantic = round_payload["summaries"]["imappo_pretrained_semantic"]
        metrics = success_metrics(semantic)
        print(
            "[round {round_idx}] semantic metrics: "
            "medium_collision={medium_collision:.4f}, hard_collision={hard_collision:.4f}, "
            "medium_task={medium_task_completion:.4f}, hard_task={hard_task_completion:.4f}, "
            "latency={replanning_latency:.4f}, success={success}".format(
                round_idx=round_idx,
                success=round_payload["success"],
                **metrics,
            ),
            flush=True,
        )
        if round_payload["success"]:
            final = {
                "status": "success",
                "round": round_idx,
                "hyperparameters": round_payload["hyperparameters"],
                "summary": semantic,
            }
            break
        state = TuningState(**round_payload["next_hyperparameters"])

    if final is None:
        last_round = rounds[-1] if rounds else {"hyperparameters": asdict(state), "summaries": {}}
        final = {
            "status": "not_met",
            "round": len(rounds),
            "hyperparameters": last_round["hyperparameters"],
            "summary": last_round["summaries"].get("imappo_pretrained_semantic") if rounds else None,
        }

    payload = {
        "study_level": "smoke" if args.smoke else "pilot",
        "description": "Pilot closed-loop pretrained-semantic I-MAPPO validation matrix",
        "seeds": args.seeds,
        "episodes": args.episodes,
        "work_dir": str(args.work_dir),
        "output_dir": str(args.output_dir),
        "thresholds": SUCCESS_THRESHOLDS,
        "final": final,
        "rounds": rounds,
    }
    write_json(args.work_dir / "closed_loop_summary.json", payload)
    write_study_results(args, payload)
    write_study_report(args, payload)
    if not args.skip_plots:
        run_plot_results(args)
    cleanup_workspace([args.output_dir, args.work_dir])
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
