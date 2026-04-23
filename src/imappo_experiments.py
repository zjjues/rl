from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from imappo import IMAPPO, IMAPPOConfig, build_uav_env_factory, evaluate_imappo, train_imappo
import envs.uav_scheduling_env  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Run focused I-MAPPO UAV experiments")
    parser.add_argument("--algorithm", choices=["imappo", "mappo", "both"], default="imappo")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/imappo_stage3"),
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


class JsonlMetricWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "w", encoding="utf-8")

    def write(self, item: Dict[str, float]) -> None:
        self.file.write(json.dumps(item, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class CheckpointManager:
    def __init__(self, directory: Path, save_every: int):
        self.directory = directory
        self.save_every = save_every
        self.best_eval_collision_rate = float("inf")
        self.best_probe_collision_rate = float("inf")

    def __call__(self, algo: IMAPPO, item: Dict[str, float]) -> None:
        episode = int(item.get("episode", -1))
        if episode < 0:
            return
        if "episode_return" in item and self.save_every > 0 and (episode + 1) % self.save_every == 0:
            algo.save_checkpoint(
                str(self.directory / f"checkpoint_ep{episode + 1}.pt"),
                extra={"episode": episode, "kind": "periodic"},
            )
            algo.save_checkpoint(
                str(self.directory / "checkpoint_latest.pt"),
                extra={"episode": episode, "kind": "latest"},
            )
        if "eval_collision_rate" in item and item["eval_collision_rate"] <= self.best_eval_collision_rate:
            self.best_eval_collision_rate = float(item["eval_collision_rate"])
            algo.save_checkpoint(
                str(self.directory / "checkpoint_best_eval.pt"),
                extra={"episode": episode, "kind": "best_eval_collision_rate"},
            )
        if "probe_collision_rate" in item and item["probe_collision_rate"] <= self.best_probe_collision_rate:
            self.best_probe_collision_rate = float(item["probe_collision_rate"])
            algo.save_checkpoint(
                str(self.directory / "checkpoint_best_probe.pt"),
                extra={"episode": episode, "kind": "best_probe_collision_rate"},
            )


def write_metrics_csv(path: Path, logs: List[Dict[str, float]]) -> None:
    if not logs:
        return
    fieldnames = sorted({key for item in logs for key in item.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in logs:
            writer.writerow(item)


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
    algorithm_dir = output_dir / args.algorithm / f"seed_{seed}"
    algorithm_dir.mkdir(parents=True, exist_ok=True)
    is_mappo = args.algorithm == "mappo"
    cfg = IMAPPOConfig(
        algorithm=args.algorithm,
        critic_mode="uniform" if is_mappo else "attention",
        use_action_mask=not is_mappo,
        seed=seed,
        max_episodes=args.episodes,
        max_steps=args.steps,
        rollout_length=args.rollout,
        minibatch_size=args.batch_size,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        eta=0.0 if is_mappo else IMAPPOConfig.eta,
        eta_end=0.0 if is_mappo else IMAPPOConfig.eta_end,
        potential_update_mode="frozen" if is_mappo else IMAPPOConfig.potential_update_mode,
    )

    train_factory = build_uav_env_factory(cfg, mode="train")
    eval_factory = build_uav_env_factory(cfg, mode="eval")
    probe_easy_factory = build_custom_uav_factory(0.34, 0.95)
    probe_mid_factory = build_custom_uav_factory(0.31, 0.88)
    probe_hard_factory = build_custom_uav_factory(0.29, 0.82)

    jsonl_writer = JsonlMetricWriter(algorithm_dir / "metrics.jsonl")
    checkpoint_manager = CheckpointManager(algorithm_dir, args.save_every)
    try:
        algo, logs = train_imappo(
            env_factory=train_factory,
            eval_env_factory=eval_factory,
            collision_probe_env_factory=probe_hard_factory,
            config=cfg,
            log_callback=jsonl_writer.write,
            checkpoint_callback=checkpoint_manager,
        )
    finally:
        jsonl_writer.close()

    write_metrics_csv(algorithm_dir / "metrics.csv", logs)
    algo.save_checkpoint(
        str(algorithm_dir / "checkpoint_latest.pt"),
        extra={"seed": seed, "algorithm": args.algorithm},
    )

    tier_metrics = {
        "easy": evaluate_imappo(algo, probe_easy_factory, cfg, prefix="easy_probe"),
        "mid": evaluate_imappo(algo, probe_mid_factory, cfg, prefix="mid_probe"),
        "hard": evaluate_imappo(algo, probe_hard_factory, cfg, prefix="hard_probe"),
    }
    result = {
        "seed": seed,
        "algorithm": args.algorithm,
        "config": {
            "algorithm": args.algorithm,
            "episodes": args.episodes,
            "steps": args.steps,
            "rollout": args.rollout,
            "batch_size": args.batch_size,
            "eval_interval": args.eval_interval,
            "eval_episodes": args.eval_episodes,
            "save_every": args.save_every,
        },
        "logs": logs,
        "tier_metrics": tier_metrics,
    }
    with open(algorithm_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def save_algorithm_plots(output_dir: Path, algorithm: str, seed_results: List[Dict[str, object]]) -> None:
    plot_specs = {
        "train_return_curve.png": ("episode_return", "Training Episode Return", "Episode", "Episode Return"),
        "train_collision_curve.png": ("episode_collision_rate", "Training Collision Rate", "Episode", "Collision Rate"),
        "eval_collision_curve.png": ("eval_collision_rate", "Standard Evaluation Collision Rate", "Episode", "Collision Rate"),
        "probe_collision_curve.png": ("probe_collision_rate", "Crowded Evaluation Collision Rate", "Episode", "Collision Rate"),
        "reward_decomposition_curve.png": ("episode_reward_task", "Task Progress Reward Trend", "Episode", "Task Reward"),
    }

    for filename, (metric, title, xlabel, ylabel) in plot_specs.items():
        episodes, mean, std = aggregate_seed_curves(
            [collect_episode_series(seed_result["logs"], metric) for seed_result in seed_results]
        )
        save_plot(output_dir / algorithm / filename, title, xlabel, ylabel, [(metric, episodes, mean, std)])


def save_risk_tier_plots(output_dir: Path, algorithm: str, seed_results: List[Dict[str, object]]) -> None:
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
    plt.savefig(output_dir / algorithm / "risk_tier_collision_bar.png", dpi=180)
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
    plt.savefig(output_dir / algorithm / "risk_tier_task_bar.png", dpi=180)
    plt.close()


def save_comparison_plots(output_dir: Path, all_results: Dict[str, List[Dict[str, object]]]) -> None:
    compare_dir = output_dir / "comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "compare_return.png": ("episode_return", "I-MAPPO vs MAPPO Training Return", "Episode Return"),
        "compare_eval_collision.png": ("eval_collision_rate", "I-MAPPO vs MAPPO Eval Collision", "Collision Rate"),
        "compare_probe_collision.png": ("probe_collision_rate", "I-MAPPO vs MAPPO Dense Collision", "Collision Rate"),
        "compare_task_completion.png": ("episode_task_completion", "I-MAPPO vs MAPPO Task Completion", "Task Completion"),
    }
    for filename, (metric, title, ylabel) in specs.items():
        series = []
        for algorithm, seed_results in all_results.items():
            episodes, mean, std = aggregate_seed_curves(
                [collect_episode_series(seed_result["logs"], metric) for seed_result in seed_results]
            )
            series.append((algorithm, episodes, mean, std))
        save_plot(compare_dir / filename, title, "Episode", ylabel, series)


def write_summary(output_dir: Path, algorithm: str, args, seed_results: List[Dict[str, object]]) -> Dict[str, object]:
    summary = {
        "seeds": args.seeds,
        "algorithm": algorithm,
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
    with open(output_dir / algorithm / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    algorithms = ["imappo", "mappo"] if args.algorithm == "both" else [args.algorithm]
    all_results: Dict[str, List[Dict[str, object]]] = {}
    summaries = []
    for algorithm in algorithms:
        args.algorithm = algorithm
        seed_results = [run_single_seed(seed, args, output_dir) for seed in args.seeds]
        all_results[algorithm] = seed_results
        save_algorithm_plots(output_dir, algorithm, seed_results)
        save_risk_tier_plots(output_dir, algorithm, seed_results)
        summaries.append(write_summary(output_dir, algorithm, args, seed_results))

    if len(all_results) > 1:
        save_comparison_plots(output_dir, all_results)

    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
