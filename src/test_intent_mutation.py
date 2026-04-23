from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import numpy as np
import torch

from imappo import IMAPPO, env_reset, env_step, infer_agent_order, normalise_obs
import envs.uav_scheduling_env  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate I-MAPPO intent mutation response")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/intent_mutation/mutation_trajectory.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--approach-steps", type=int, default=21)
    parser.add_argument("--total-steps", type=int, default=50)
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


def all_uavs_moving_away(positions: np.ndarray, velocities: np.ndarray, targets: np.ndarray) -> bool:
    target_vectors = targets - positions
    dots = np.sum(velocities * target_vectors, axis=1)
    return bool(np.all(dots < 0.0))


def run_mutation(args) -> Dict[str, object]:
    algo = IMAPPO.load_checkpoint(str(args.checkpoint), device=args.device)
    device = algo.device
    env = gym.make(
        "uav-scheduling-v0",
        n_agents=4,
        n_targets=3,
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

    for step in range(args.total_steps):
        phase = "approach" if step < mutation_step else "evasion"
        intent = approach_intent if phase == "approach" else evasion_intent
        mask = approach_mask if phase == "approach" else evasion_mask

        obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device)
        actions, _ = algo.select_actions(obs_tensor, intent, mask, deterministic=True)
        next_obs_data, _, done, truncated, info = env_step(
            env, agent_order, actions.detach().cpu().numpy()
        )

        arrays = get_env_arrays(env)
        moving_away = all_uavs_moving_away(
            arrays["positions"], arrays["velocities"], arrays["targets"]
        )
        if step >= mutation_step and response_latency is None and moving_away:
            response_latency = step - mutation_step

        trajectory.append(
            {
                "step": step,
                "phase": phase,
                "intent": intent.detach().cpu().tolist(),
                "positions": arrays["positions"].tolist(),
                "velocities": arrays["velocities"].tolist(),
                "targets": arrays["targets"].tolist(),
                "moving_away_from_targets": moving_away,
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
        "total_recorded_steps": len(trajectory),
        "trajectory": trajectory,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
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
                "output": str(args.output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
