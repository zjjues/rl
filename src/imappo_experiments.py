from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from imappo import IMAPPOConfig, build_uav_env_factory, evaluate_imappo, train_imappo
import envs.uav_scheduling_env  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Run focused I-MAPPO UAV experiments")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=6)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/imappo_stage2"),
    )
    return parser.parse_args()


def build_custom_uav_factory(
    spawn_region_scale: float,
    spawn_separation_scale: float,
) -> Callable[[], object]:
    def make_env():
        return gym.make(
            "uav-scheduling-v0",
            spawn_region_scale=spawn_region_scale,
            spawn_separation_scale=spawn_separation_scale,
        )

    return make_env


def collect_episode_series(logs: List[Dict[str, float]], key: str) -> Tuple[np.ndarray, np.ndarray]:
    episodes = []
    values = []
    for item in logs:
        if key in item and "episode" in item:
            episodes.append(int(item["episode"]))
            values.append(float(item[key]))
    return np.asarray(episodes, dtype=np.int32), np.asarray(values, dtype=np.float32)


def aggregate_seed_curves(curves: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not curves:
        return np.asarray([]), np.asarray([]), np.asarray([])

    common_episodes = sorted(set.intersection(*(set(ep.tolist()) for ep, _ in curves)))
    common_episodes = np.asarray(common_episodes, dtype=np.int32)
    stacked = []
    for episodes, values in curves:
        mapping = {int(ep): float(val) for ep, val in zip(episodes, values)}
        stacked.append([mapping[int(ep)] for ep in common_episodes])
    data = np.asarray(stacked, dtype=np.float32)
    return common_episodes, data.mean(axis=0), data.std(axis=0)


def save_plot(
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
):
    plt.figure(figsize=(8.5, 5.2))
    for label, episodes, mean, std in series:
        if episodes.size == 0:
            continue
        plt.plot(episodes, mean, linewidth=2.0, label=label)
        plt.fill_between(episodes, mean - std, mean + std, alpha=0.18)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def run_single_seed(seed: int, args, output_dir: Path) -> Dict[str, object]:
    cfg = IMAPPOConfig(
        seed=seed,
        max_episodes=args.episodes,
        max_steps=args.steps,
        rollout_length=args.rollout,
        minibatch_size=args.batch_size,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
    )

    train_factory = build_uav_env_factory(cfg, mode="train")
    eval_factory = build_uav_env_factory(cfg, mode="eval")
    probe_easy_factory = build_custom_uav_factory(0.34, 0.95)
    probe_mid_factory = build_custom_uav_factory(0.31, 0.88)
    probe_hard_factory = build_custom_uav_factory(0.29, 0.82)

    algo, logs = train_imappo(
        env_factory=train_factory,
        eval_env_factory=eval_factory,
        collision_probe_env_factory=probe_hard_factory,
        config=cfg,
    )

    tier_metrics = {
        "easy": evaluate_imappo(algo, probe_easy_factory, cfg, prefix="easy_probe"),
        "mid": evaluate_imappo(algo, probe_mid_factory, cfg, prefix="mid_probe"),
        "hard": evaluate_imappo(algo, probe_hard_factory, cfg, prefix="hard_probe"),
    }
    result = {
        "seed": seed,
        "config": {
            "episodes": args.episodes,
            "steps": args.steps,
            "rollout": args.rollout,
            "batch_size": args.batch_size,
            "eval_interval": args.eval_interval,
            "eval_episodes": args.eval_episodes,
        },
        "logs": logs,
        "tier_metrics": tier_metrics,
    }
    with open(output_dir / f"seed_{seed}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_results = [run_single_seed(seed, args, output_dir) for seed in args.seeds]

    plot_specs = {
        "train_return_curve.png": ("episode_return", "Training Episode Return", "Episode", "Episode Return"),
        "train_collision_curve.png": ("episode_collision_rate", "Training Collision Rate", "Episode", "Collision Rate"),
        "eval_collision_curve.png": ("eval_collision_rate", "Standard Evaluation Collision Rate", "Episode", "Collision Rate"),
        "probe_collision_curve.png": ("probe_collision_rate", "Crowded Evaluation Collision Rate", "Episode", "Collision Rate"),
        "reward_decomposition_curve.png": ("episode_reward_task", "Task Progress Reward Trend", "Episode", "Task Reward"),
    }

    for filename, (metric, title, xlabel, ylabel) in plot_specs.items():
        curves = []
        episodes, mean, std = aggregate_seed_curves(
            [collect_episode_series(seed_result["logs"], metric) for seed_result in seed_results]
        )
        curves.append((metric, episodes, mean, std))
        save_plot(output_dir / filename, title, xlabel, ylabel, curves)

    risk_levels = ["easy_probe", "mid_probe", "hard_probe"]
    risk_names = ["Loose", "Medium", "Dense"]
    risk_collision = []
    risk_task = []
    for risk_key, risk_name in zip(risk_levels, risk_names):
        collision_values = []
        task_values = []
        for seed_result in seed_results:
            tier_values = seed_result["tier_metrics"][risk_key.split("_")[0]]
            collision_values.append(float(tier_values[f"{risk_key}_collision_rate"]))
            task_values.append(float(tier_values[f"{risk_key}_task_completion"]))
        risk_collision.append((risk_name, float(np.mean(collision_values)), float(np.std(collision_values))))
        risk_task.append((risk_name, float(np.mean(task_values)), float(np.std(task_values))))

    plt.figure(figsize=(7.8, 5.0))
    x = np.arange(len(risk_names))
    means = np.asarray([item[1] for item in risk_collision], dtype=np.float32)
    stds = np.asarray([item[2] for item in risk_collision], dtype=np.float32)
    plt.bar(x, means, yerr=stds, width=0.58, color=["#4C78A8", "#F58518", "#E45756"], alpha=0.88)
    plt.xticks(x, risk_names)
    plt.ylabel("Collision Rate")
    plt.title("Collision Rate Across Risk Tiers")
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_dir / "risk_tier_collision_bar.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.8, 5.0))
    means = np.asarray([item[1] for item in risk_task], dtype=np.float32)
    stds = np.asarray([item[2] for item in risk_task], dtype=np.float32)
    plt.bar(x, means, yerr=stds, width=0.58, color=["#72B7B2", "#54A24B", "#EECA3B"], alpha=0.88)
    plt.xticks(x, risk_names)
    plt.ylabel("Task Completion")
    plt.title("Task Completion Across Risk Tiers")
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_dir / "risk_tier_task_bar.png", dpi=180)
    plt.close()

    summary = {
        "seeds": args.seeds,
        "output_dir": str(output_dir),
        "final_eval_collision_rate_mean": float(
            np.mean(
                [
                    seed_result["tier_metrics"]["easy"]["easy_probe_collision_rate"]
                    for seed_result in seed_results
                ]
            )
        ),
        "final_probe_collision_rate_mean": float(
            np.mean(
                [
                    seed_result["tier_metrics"]["hard"]["hard_probe_collision_rate"]
                    for seed_result in seed_results
                ]
            )
        ),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
