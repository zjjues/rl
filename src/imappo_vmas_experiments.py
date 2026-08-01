"""I-MAPPO VMAS benchmark experiments.

Runs I-MAPPO vs MAPPO comparison on VMAS (Vectorized Multi-Agent Simulator)
scenarios — standard continuous-control MARL benchmarks.

Usage:
    # Smoke test
    python src/imappo_vmas_experiments.py --scenario dispersion --episodes 10 --seeds 42

    # Full comparison
    python src/imappo_vmas_experiments.py --algorithm both --scenario dispersion \\
        --episodes 3000 --seeds 7 11 23 --intent-source pretrained_semantic
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Tuple

MATPLOTLIB_CACHE = Path(tempfile.gettempdir()) / "rl-matplotlib-cache"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))

import matplotlib.pyplot as plt
import numpy as np

from imappo import (
    IMAPPO,
    IMAPPOConfig,
    evaluate_imappo,
    train_imappo,
)
from envs.vmas_adapter import (
    VMASAdapter,
    VMAS_SCENARIOS,
    infer_vmas_dims,
)
from intent_semantic_encoder import IntentLibrary


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run I-MAPPO on VMAS benchmarks")
    parser.add_argument("--algorithm", choices=["imappo", "mappo", "both"], default="both")
    parser.add_argument("--scenario", choices=VMAS_SCENARIOS, default="dispersion")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument(
        "--intent-source",
        choices=["onehot", "legacy_hash", "random_dense", "pretrained_semantic"],
        default="pretrained_semantic",
    )
    parser.add_argument("--intent-dim", type=int, default=64)
    parser.add_argument("--intent-library-path", type=str, default="")
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
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/vmas_stage1"))
    return parser.parse_args()


# ── Environment factory ────────────────────────────────────────────────────────

def build_vmas_env_factory(
    scenario: str,
    n_agents: int,
    max_steps: int,
    seed: int | None = None,
) -> Callable[[], object]:
    def make_env():
        env = VMASAdapter(
            scenario=scenario,
            n_agents=n_agents,
            max_steps=max_steps,
            seed=seed,
        )
        return env
    return make_env


# ── Metric helpers ─────────────────────────────────────────────────────────────

def vmas_episode_metrics(logs: List[Dict[str, float]]) -> Dict[str, np.ndarray]:
    """Extract episode-level training curves from training logs."""
    episodes = []
    returns = []
    env_rewards = []
    intent_rewards = []
    for item in logs:
        if "episode_return" in item and "episode" in item:
            episodes.append(int(item["episode"]))
            returns.append(float(item["episode_return"]))
            env_rewards.append(float(item.get("episode_reward_env", 0.0)))
            intent_rewards.append(float(item.get("episode_reward_intent", 0.0)))
    return {
        "episodes": np.asarray(episodes, dtype=np.int32),
        "returns": np.asarray(returns, dtype=np.float32),
        "env_rewards": np.asarray(env_rewards, dtype=np.float32),
        "intent_rewards": np.asarray(intent_rewards, dtype=np.float32),
    }


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


# ── Plotting ───────────────────────────────────────────────────────────────────

def apply_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 180,
        "savefig.dpi": 240,
    })


def algorithm_style(algorithm: str) -> Dict[str, object]:
    if algorithm == "imappo":
        return {"color": "#003366", "linestyle": "-", "label": "I-MAPPO"}
    return {"color": "#990000", "linestyle": "--", "label": "MAPPO"}


def save_comparison_plot(output_path: Path, title: str, xlabel: str, ylabel: str,
                         series: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]]):
    apply_style()
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for algorithm, episodes, mean, std, seed_count in series:
        if episodes.size == 0:
            continue
        style = algorithm_style(algorithm)
        ax.plot(episodes, mean, linewidth=2.3, **{k: v for k, v in style.items() if k != "label"})
        ax.plot(episodes, mean, linewidth=2.3, color=style["color"],
                linestyle=style["linestyle"], label=style["label"])
        if seed_count > 1:
            ax.fill_between(episodes, mean - std, mean + std, color=style["color"], alpha=0.16)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ── JSONL writer ───────────────────────────────────────────────────────────────

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


# ── Checkpoint manager ─────────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, directory: Path, save_every: int):
        self.directory = directory
        self.save_every = save_every
        self.best_return = -float("inf")

    def __call__(self, algo: IMAPPO, item: Dict[str, float]) -> None:
        episode = int(item.get("episode", -1))
        if episode < 0:
            return
        if self.save_every > 0 and (episode + 1) % self.save_every == 0:
            algo.save_checkpoint(
                str(self.directory / f"checkpoint_ep{episode + 1}.pt"),
                extra={"episode": episode, "kind": "periodic"},
            )
        if "episode_return" in item and item["episode_return"] >= self.best_return:
            self.best_return = float(item["episode_return"])
            algo.save_checkpoint(
                str(self.directory / "checkpoint_best.pt"),
                extra={"episode": episode, "kind": "best_return"},
            )


# ── Single-seed runner ─────────────────────────────────────────────────────────

def run_single_seed(seed: int, args, output_dir: Path, intent_library: IntentLibrary | None = None) -> Dict[str, object]:
    algorithm_dir = output_dir / args.algorithm / f"seed_{seed}"
    algorithm_dir.mkdir(parents=True, exist_ok=True)

    is_mappo = args.algorithm == "mappo"
    obs_dim, state_dim, action_dim = infer_vmas_dims(args.scenario, args.n_agents)

    cfg = IMAPPOConfig(
        algorithm=args.algorithm,
        critic_mode="uniform" if is_mappo else "attention",
        use_action_mask=not is_mappo,
        intent_source=args.intent_source,
        intent_library_path=args.intent_library_path,
        intent_encoder_model=args.intent_encoder_model,
        intent_encoder_revision=args.intent_encoder_revision,
        intent_projection_seed=args.intent_projection_seed,
        intent_code_seed=args.intent_code_seed,
        intent_dim=args.intent_dim,
        seed=seed,
        n_agents=args.n_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
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

    train_factory = build_vmas_env_factory(args.scenario, args.n_agents, args.steps, seed)
    eval_factory = build_vmas_env_factory(args.scenario, args.n_agents, args.steps, seed)

    jsonl_writer = JsonlMetricWriter(algorithm_dir / "metrics.jsonl")
    checkpoint_manager = CheckpointManager(algorithm_dir, args.save_every)

    try:
        algo, logs = train_imappo(
            env_factory=train_factory,
            eval_env_factory=eval_factory,
            collision_probe_env_factory=None,  # no collision probes for VMAS
            config=cfg,
            log_callback=jsonl_writer.write,
            checkpoint_callback=checkpoint_manager,
        )
    finally:
        jsonl_writer.close()

    # Save CSV
    fieldnames = sorted({key for item in logs for key in item.keys()})
    with open(algorithm_dir / "metrics.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in logs:
            writer.writerow(item)

    algo.save_checkpoint(
        str(algorithm_dir / "checkpoint_latest.pt"),
        extra={"seed": seed, "algorithm": args.algorithm},
    )

    # Run final evaluation
    eval_metrics = evaluate_imappo(algo, eval_factory, cfg, prefix="eval", evaluation_mode="standard")

    result = {
        "seed": seed,
        "algorithm": args.algorithm,
        "scenario": args.scenario,
        "config": {
            "algorithm": args.algorithm,
            "episodes": args.episodes,
            "steps": args.steps,
            "n_agents": args.n_agents,
            "obs_dim": obs_dim,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "intent_source": args.intent_source,
            "intent_dim": args.intent_dim,
        },
        "eval_metrics": eval_metrics,
        "logs": logs,
    }
    with open(algorithm_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist the exact representation library used by non-one-hot studies.
    intent_library = None
    if args.intent_source == "legacy_hash":
        intent_library = IntentLibrary.create_legacy_hash(args.intent_dim, domain="vmas")
    elif args.intent_source == "random_dense":
        intent_library = IntentLibrary.create_random_dense(
            args.intent_dim,
            domain="vmas",
            seed=args.intent_code_seed,
        )
    elif args.intent_source == "pretrained_semantic":
        intent_library = IntentLibrary.create_pretrained(
            args.intent_dim,
            domain="vmas",
            model_name=args.intent_encoder_model,
            model_revision=args.intent_encoder_revision,
            projection_seed=args.intent_projection_seed,
        )
    if intent_library is not None:
        lib_path = output_dir / "intent_library"
        intent_library.save(str(lib_path))

    algorithms = ["imappo", "mappo"] if args.algorithm == "both" else [args.algorithm]
    all_results: Dict[str, List[Dict[str, object]]] = {}
    summaries = []

    for algorithm in algorithms:
        args.algorithm = algorithm
        print(f"\n{'='*60}")
        print(f"Running {algorithm.upper()} on {args.scenario}")
        print(f"Seeds: {args.seeds}, Episodes: {args.episodes}")
        print(f"{'='*60}\n")

        seed_results = [run_single_seed(seed, args, output_dir, intent_library) for seed in args.seeds]
        all_results[algorithm] = seed_results

        # Compute summary stats
        eval_returns = [
            seed_result["eval_metrics"]["eval_episode_return"]
            for seed_result in seed_results
        ]
        summary = {
            "seeds": args.seeds,
            "algorithm": algorithm,
            "scenario": args.scenario,
            "output_dir": str(output_dir),
            "eval_return_mean": float(np.mean(eval_returns)),
            "eval_return_std": float(np.std(eval_returns)),
            "final_eval_return": float(eval_returns[-1]) if eval_returns else 0.0,
        }
        summaries.append(summary)

        # Save per-algorithm summary
        with open(output_dir / algorithm / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Save training return curve per algorithm
        return_curves = []
        for seed_result in seed_results:
            episodes = []
            values = []
            for item in seed_result["logs"]:
                if "episode_return" in item and "episode" in item:
                    episodes.append(int(item["episode"]))
                    values.append(float(item["episode_return"]))
            return_curves.append((np.asarray(episodes, dtype=np.int32), np.asarray(values, dtype=np.float32)))

        ep, mean, std, n = aggregate_seed_curves(return_curves)
        save_comparison_plot(
            output_dir / algorithm / "train_return_curve.png",
            f"{algorithm.upper()} — Training Return ({args.scenario})",
            "Episode", "Episode Return",
            [(algorithm, ep, mean, std, n)],
        )

    # Save comparison if running both
    if len(all_results) > 1:
        compare_dir = output_dir / "comparison"
        compare_dir.mkdir(parents=True, exist_ok=True)

        series = []
        for algorithm, seed_results in all_results.items():
            return_curves = []
            for seed_result in seed_results:
                episodes = []
                values = []
                for item in seed_result["logs"]:
                    if "episode_return" in item and "episode" in item:
                        episodes.append(int(item["episode"]))
                        values.append(float(item["episode_return"]))
                return_curves.append((np.asarray(episodes, dtype=np.int32), np.asarray(values, dtype=np.float32)))
            ep, mean, std, n = aggregate_seed_curves(return_curves)
            series.append((algorithm, ep, mean, std, n))

        save_comparison_plot(
            compare_dir / "training_convergence.png",
            f"I-MAPPO vs MAPPO on VMAS {args.scenario}",
            "Training Episodes", "Episode Return",
            series,
        )

    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
