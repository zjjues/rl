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
from envs.uav_scheduling_env import infer_obs_dim, infer_state_dim
import envs.uav_scheduling_env  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Run focused I-MAPPO UAV experiments")
    parser.add_argument("--algorithm", choices=["imappo", "mappo", "both"], default="imappo")
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--n-agents", type=int, default=8)
    parser.add_argument("--n-targets", type=int, default=6)
    parser.add_argument("--safety-reward-coef", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/imappo_stage4"),
    )
    parser.add_argument(
        "--mutation-results",
        type=Path,
        nargs="*",
        default=[],
        help="Optional mutation result JSON files from test_intent_mutation.py for Figure 4.",
    )
    return parser.parse_args()


def build_custom_uav_factory(
    n_agents: int,
    n_targets: int,
    obs_dim: int,
    spawn_region_scale: float,
    spawn_separation_scale: float,
    safety_reward_coef: float,
) -> Callable[[], object]:
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


def aggregate_seed_curves(
    curves: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if not curves:
        return np.asarray([]), np.asarray([]), np.asarray([]), 0

    common_episodes = sorted(set.intersection(*(set(ep.tolist()) for ep, _ in curves)))
    common_episodes = np.asarray(common_episodes, dtype=np.int32)
    stacked = []
    for episodes, values in curves:
        mapping = {int(ep): float(val) for ep, val in zip(episodes, values)}
        stacked.append([mapping[int(ep)] for ep in common_episodes])
    data = np.asarray(stacked, dtype=np.float32)
    return common_episodes, data.mean(axis=0), data.std(axis=0), int(data.shape[0])


def apply_publication_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.titleweight": "semibold",
            "figure.dpi": 180,
            "savefig.dpi": 240,
        }
    )


def algorithm_display_name(algorithm: str) -> str:
    return "I-MAPPO" if algorithm == "imappo" else "MAPPO"


def algorithm_style(algorithm: str) -> Dict[str, object]:
    if algorithm == "imappo":
        return {"color": "#003366", "linestyle": "-", "label": "I-MAPPO"}
    return {"color": "#990000", "linestyle": "--", "label": "MAPPO"}


def save_publication_line_plot(
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]],
    max_episode: int | None = None,
) -> None:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for algorithm, episodes, mean, std, seed_count in series:
        if episodes.size == 0:
            continue
        style = algorithm_style(algorithm)
        x = episodes
        y = mean
        spread = std
        if max_episode is not None:
            mask = episodes <= max_episode
            x = episodes[mask]
            y = mean[mask]
            spread = std[mask]
        if x.size == 0:
            continue
        ax.plot(
            x,
            y,
            linewidth=2.3,
            color=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
        )
        if seed_count > 1:
            ax.fill_between(x, y - spread, y + spread, color=style["color"], alpha=0.16)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_publication_grouped_bar(
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    categories: List[str],
    series: List[Tuple[str, np.ndarray, np.ndarray]],
) -> None:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    x = np.arange(len(categories), dtype=np.float32)
    width = 0.34
    offsets = np.linspace(-(len(series) - 1) * width / 2.0, (len(series) - 1) * width / 2.0, len(series))

    for idx, (algorithm, means, stds) in enumerate(series):
        style = algorithm_style(algorithm)
        ax.bar(
            x + offsets[idx],
            means,
            yerr=stds,
            width=width,
            color=style["color"],
            alpha=0.9,
            capsize=4,
            label=style["label"],
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_publication_latency_bar(output_path: Path, mutation_results: List[Dict[str, object]]) -> None:
    if not mutation_results:
        return

    apply_publication_style()
    ordered = sorted(mutation_results, key=lambda item: int(item["seed"]))
    labels = [f"Seed {int(item['seed'])}" for item in ordered]
    numeric_latencies = []
    for item in ordered:
        latency = item.get("response_latency")
        numeric_latencies.append(np.nan if latency is None else float(latency))
    latencies = np.asarray(
        numeric_latencies,
        dtype=np.float32,
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    bars = ax.bar(labels, latencies, width=0.62, color="#003366", alpha=0.92)
    ax.set_title("Figure 4. Zero-Shot Intent Mutation Latency")
    ax.set_xlabel("Seed Number")
    ax.set_ylabel("Response Latency (Steps)")
    finite_latencies = latencies[np.isfinite(latencies)]
    ymax = float(finite_latencies.max()) if finite_latencies.size > 0 else 1.0
    ax.set_ylim(0.0, max(ymax + 1.0, 1.0))
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    for bar, latency in zip(bars, latencies):
        if np.isnan(latency):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.05,
            f"{int(latency)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def load_mutation_results(paths: List[Path]) -> List[Dict[str, object]]:
    results = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)
        results.append(
            {
                "seed": int(item["seed"]),
                "response_latency": item.get("response_latency"),
                "path": str(path),
            }
        )
    return results


def run_single_seed(seed: int, args, output_dir: Path) -> Dict[str, object]:
    algorithm_dir = output_dir / args.algorithm / f"seed_{seed}"
    algorithm_dir.mkdir(parents=True, exist_ok=True)
    is_mappo = args.algorithm == "mappo"
    obs_dim = infer_obs_dim(args.n_agents)
    state_dim = infer_state_dim(args.n_agents, obs_dim)
    cfg = IMAPPOConfig(
        algorithm=args.algorithm,
        critic_mode="uniform" if is_mappo else "attention",
        use_action_mask=not is_mappo,
        seed=seed,
        n_agents=args.n_agents,
        n_targets=args.n_targets,
        obs_dim=obs_dim,
        state_dim=state_dim,
        max_episodes=args.episodes,
        max_steps=args.steps,
        rollout_length=args.rollout,
        minibatch_size=args.batch_size,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        safety_reward_coef=args.safety_reward_coef,
        eta=0.0 if is_mappo else IMAPPOConfig.eta,
        eta_end=0.0 if is_mappo else IMAPPOConfig.eta_end,
        potential_update_mode="frozen" if is_mappo else IMAPPOConfig.potential_update_mode,
    )

    train_factory = build_uav_env_factory(cfg, mode="train")
    eval_factory = build_uav_env_factory(cfg, mode="eval")
    probe_easy_factory = build_custom_uav_factory(
        args.n_agents, args.n_targets, obs_dim, 0.42, 0.92, args.safety_reward_coef
    )
    probe_mid_factory = build_custom_uav_factory(
        args.n_agents, args.n_targets, obs_dim, 0.37, 0.86, args.safety_reward_coef
    )
    probe_hard_factory = build_custom_uav_factory(
        args.n_agents, args.n_targets, obs_dim, 0.33, 0.80, args.safety_reward_coef
    )

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
        "easy": evaluate_imappo(algo, probe_easy_factory, cfg, prefix="easy_probe", evaluation_mode="dense"),
        "mid": evaluate_imappo(algo, probe_mid_factory, cfg, prefix="mid_probe", evaluation_mode="dense"),
        "hard": evaluate_imappo(algo, probe_hard_factory, cfg, prefix="hard_probe", evaluation_mode="dense"),
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
            "n_agents": args.n_agents,
            "n_targets": args.n_targets,
            "obs_dim": obs_dim,
            "state_dim": state_dim,
            "safety_reward_coef": args.safety_reward_coef,
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
        episodes, mean, std, seed_count = aggregate_seed_curves(
            [collect_episode_series(seed_result["logs"], metric) for seed_result in seed_results]
        )
        save_publication_line_plot(
            output_dir / algorithm / filename,
            title,
            xlabel,
            ylabel,
            [(algorithm, episodes, mean, std, seed_count)],
        )


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

    means = np.asarray([item[1] for item in risk_collision], dtype=np.float32)
    stds = np.asarray([item[2] for item in risk_collision], dtype=np.float32)
    save_publication_grouped_bar(
        output_dir / algorithm / "risk_tier_collision_bar.png",
        f"{algorithm_display_name(algorithm)} Risk-Tier Collision Robustness",
        "Risk Tier",
        "Collision Rate",
        risk_names,
        [(algorithm, means, stds)],
    )

    means = np.asarray([item[1] for item in risk_task], dtype=np.float32)
    stds = np.asarray([item[2] for item in risk_task], dtype=np.float32)
    save_publication_grouped_bar(
        output_dir / algorithm / "risk_tier_task_bar.png",
        f"{algorithm_display_name(algorithm)} Risk-Tier Task Completion",
        "Risk Tier",
        "Task Completion",
        risk_names,
        [(algorithm, means, stds)],
    )


def save_comparison_plots(output_dir: Path, all_results: Dict[str, List[Dict[str, object]]]) -> None:
    compare_dir = output_dir / "comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)
    figure_specs = [
        (
            "figure1_training_convergence.png",
            "episode_return",
            "Figure 1. Training Convergence Curve",
            "Training Episodes",
            "Episode Return",
            None,
        ),
        (
            "figure2_early_stage_safety.png",
            "episode_collision_rate",
            "Figure 2. Early-Stage Exploration Safety",
            "Training Episodes",
            "Collision & Boundary Violation Rate",
            200,
        ),
    ]

    for filename, metric, title, xlabel, ylabel, max_episode in figure_specs:
        series = []
        for algorithm, seed_results in all_results.items():
            episodes, mean, std, seed_count = aggregate_seed_curves(
                [collect_episode_series(seed_result["logs"], metric) for seed_result in seed_results]
            )
            series.append((algorithm, episodes, mean, std, seed_count))
        save_publication_line_plot(compare_dir / filename, title, xlabel, ylabel, series, max_episode=max_episode)

    risk_names = ["Loose", "Medium", "Dense"]
    grouped_series = []
    for algorithm, seed_results in all_results.items():
        tier_means = []
        tier_stds = []
        for tier_name in ["easy", "mid", "hard"]:
            collision_values = [
                float(seed_result["tier_metrics"][tier_name][f"{tier_name}_probe_collision_rate"])
                for seed_result in seed_results
            ]
            tier_means.append(float(np.mean(collision_values)))
            tier_stds.append(float(np.std(collision_values)))
        grouped_series.append(
            (
                algorithm,
                np.asarray(tier_means, dtype=np.float32),
                np.asarray(tier_stds, dtype=np.float32),
            )
        )
    save_publication_grouped_bar(
        compare_dir / "figure3_risk_robustness.png",
        "Figure 3. Multi-Tier Risk Robustness",
        "Risk Tiers",
        "Final Probe Collision Rate",
        risk_names,
        grouped_series,
    )


def write_summary(output_dir: Path, algorithm: str, args, seed_results: List[Dict[str, object]]) -> Dict[str, object]:
    def tier_mean(tier_name: str, metric_suffix: str) -> float:
        return float(
            np.mean(
                [
                    seed_result["tier_metrics"][tier_name][f"{tier_name}_probe_{metric_suffix}"]
                    for seed_result in seed_results
                ]
            )
        )

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
        "final_easy_probe_collision_rate_mean": tier_mean("easy", "collision_rate"),
        "final_mid_probe_collision_rate_mean": tier_mean("mid", "collision_rate"),
        "final_hard_probe_collision_rate_mean": tier_mean("hard", "collision_rate"),
        "final_easy_probe_task_completion_mean": tier_mean("easy", "task_completion"),
        "final_mid_probe_task_completion_mean": tier_mean("mid", "task_completion"),
        "final_hard_probe_task_completion_mean": tier_mean("hard", "task_completion"),
    }
    summary["final_probe_collision_rate_mean"] = summary["final_hard_probe_collision_rate_mean"]
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
    if args.mutation_results:
        compare_dir = output_dir / "comparison"
        compare_dir.mkdir(parents=True, exist_ok=True)
        save_publication_latency_bar(
            compare_dir / "figure4_intent_mutation_latency.png",
            load_mutation_results(args.mutation_results),
        )

    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
