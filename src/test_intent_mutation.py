from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from imappo import (
    IMAPPO,
    env_reset,
    env_step,
    infer_agent_order,
    normalise_obs,
    set_env_intent,
    set_env_tactical_posture,
)
import envs.uav_scheduling_env  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate I-MAPPO intent mutation response")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/intent_mutation/mutation_trajectory.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--approach-steps", type=int, default=30)
    parser.add_argument("--total-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def make_intent(intent_dim: int, active_index: int, device: torch.device) -> torch.Tensor:
    intent = torch.zeros(intent_dim, device=device)
    intent[active_index] = 1.0
    return intent


def get_env_arrays(env) -> Dict[str, np.ndarray]:
    base_env = env.unwrapped
    return {
        "positions": np.asarray(base_env.positions, dtype=np.float32).copy(),
        "velocities": np.asarray(base_env.velocities, dtype=np.float32).copy(),
        "targets": np.asarray(base_env.targets, dtype=np.float32).copy(),
    }


def average_distance_to_targets(positions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(targets - positions, axis=1)))


def maybe_compute_response_latency(
    mutation_step: int,
    distance_series: List[float],
    response_latency: Optional[int],
) -> Optional[int]:
    if response_latency is not None or len(distance_series) < mutation_step + 4:
        return response_latency

    # A robust response is declared when the average swarm-to-target distance
    # increases for three consecutive post-mutation transitions.
    for current_idx in range(mutation_step + 3, len(distance_series)):
        inc_1 = distance_series[current_idx - 2] > distance_series[current_idx - 3]
        inc_2 = distance_series[current_idx - 1] > distance_series[current_idx - 2]
        inc_3 = distance_series[current_idx] > distance_series[current_idx - 1]
        if inc_1 and inc_2 and inc_3:
            return current_idx - mutation_step
    return response_latency


def save_centroid_trajectory_plot(output_path: Path, trajectory: List[Dict[str, object]], mutation_step: int) -> None:
    if not trajectory:
        return

    centroid_xy = np.asarray([item["swarm_centroid_xy"] for item in trajectory], dtype=np.float32)
    plt.figure(figsize=(6.4, 6.0))
    plt.plot(
        centroid_xy[: mutation_step + 1, 0],
        centroid_xy[: mutation_step + 1, 1],
        color="#4C78A8",
        linewidth=2.2,
        label="Gather phase",
    )
    plt.plot(
        centroid_xy[mutation_step:, 0],
        centroid_xy[mutation_step:, 1],
        color="#E45756",
        linewidth=2.2,
        label="Evade phase",
    )
    plt.scatter(centroid_xy[0, 0], centroid_xy[0, 1], color="#4C78A8", s=40, marker="o")
    plt.scatter(centroid_xy[-1, 0], centroid_xy[-1, 1], color="#E45756", s=40, marker="s")
    plt.axvline(x=0.0, alpha=0.0)
    plt.xlabel("Centroid X")
    plt.ylabel("Centroid Y")
    plt.title("Intent Mutation Swarm Centroid Trajectory")
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_latency_bar_plot(output_path: Path, mutation_results: List[Dict[str, object]]) -> None:
    if not mutation_results:
        return

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
        }
    )

    labels = [f"Seed {int(item['seed'])}" for item in mutation_results]
    numeric_latencies = []
    for item in mutation_results:
        latency = item.get("response_latency")
        numeric_latencies.append(np.nan if latency is None else float(latency))
    latencies = np.asarray(
        numeric_latencies,
        dtype=np.float32,
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    bars = ax.bar(labels, latencies, width=0.62, color="#003366", alpha=0.9)
    ax.set_title("Zero-Shot Intent Mutation Latency")
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
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def run_mutation(args) -> Dict[str, object]:
    algo = IMAPPO.load_checkpoint(str(args.checkpoint), device=args.device)
    device = algo.device
    env = gym.make(
        "uav-scheduling-v0",
        n_agents=algo.config.n_agents,
        n_targets=getattr(algo.config, "n_targets", algo.config.n_agents),
        obs_dim=algo.config.obs_dim,
        max_episode_steps=args.total_steps,
        spawn_region_scale=0.32,
        spawn_separation_scale=0.90,
    )
    obs_data, _ = env.reset(seed=args.seed)
    agent_order = infer_agent_order(env, obs_data, algo.config)
    obs_array = normalise_obs(agent_order, obs_data)

    approach_intent = make_intent(algo.config.intent_dim, 0, device)
    evasion_intent = make_intent(algo.config.intent_dim, 1, device)
    approach_mask = torch.ones(algo.config.n_agents, algo.config.action_dim, device=device)
    evasion_mask = torch.ones(algo.config.n_agents, algo.config.action_dim, device=device)

    trajectory: List[Dict[str, object]] = []
    response_latency = None
    mutation_step = args.approach_steps
    average_distance_series: List[float] = []

    for step in range(args.total_steps):
        phase = "approach" if step < mutation_step else "evasion"
        intent = approach_intent if phase == "approach" else evasion_intent
        mask = approach_mask if phase == "approach" else evasion_mask
        set_env_intent(env, intent)
        set_env_tactical_posture(env, "attack" if phase == "approach" else "stealth")

        obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device)
        actions, _ = algo.select_actions(obs_tensor, intent, mask, deterministic=True)
        next_obs_data, _, done, truncated, info = env_step(
            env, agent_order, actions.detach().cpu().numpy()
        )

        arrays = get_env_arrays(env)
        avg_distance = average_distance_to_targets(arrays["positions"], arrays["targets"])
        average_distance_series.append(avg_distance)
        response_latency = maybe_compute_response_latency(
            mutation_step, average_distance_series, response_latency
        )
        centroid_xy = arrays["positions"][:, :2].mean(axis=0)
        positions_xy = arrays["positions"][:, :2]

        trajectory.append(
            {
                "step": step,
                "phase": phase,
                "intent": intent.detach().cpu().tolist(),
                "positions": arrays["positions"].tolist(),
                "positions_xy": positions_xy.tolist(),
                "swarm_centroid_xy": centroid_xy.tolist(),
                "velocities": arrays["velocities"].tolist(),
                "targets": arrays["targets"].tolist(),
                "average_distance_to_targets": avg_distance,
                "info": info,
            }
        )

        obs_array = normalise_obs(agent_order, next_obs_data)
        if done or truncated:
            break

    env.close()
    result = {
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "mutation_step": mutation_step,
        "response_latency": response_latency,
        "figure4_record": {
            "seed": args.seed,
            "response_latency": response_latency,
        },
        "total_recorded_steps": len(trajectory),
        "average_distance_series": average_distance_series,
        "trajectory": trajectory,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    save_centroid_trajectory_plot(args.output.with_suffix(".png"), trajectory, mutation_step)
    return result


def main():
    args = parse_args()
    result = run_mutation(args)
    print(
        json.dumps(
            {
                "checkpoint": result["checkpoint"],
                "mutation_step": result["mutation_step"],
                "response_latency": result["response_latency"],
                "trajectory_plot": str(args.output.with_suffix(".png")),
                "output": str(args.output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
