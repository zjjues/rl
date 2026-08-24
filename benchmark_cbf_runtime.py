"""Benchmark the synchronized legacy CBF path against the optimized equivalent."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rule_based_baseline import (  # noqa: E402
    apply_pairwise_cbf_filter,
    apply_pairwise_cbf_filter_with_diagnostics,
    pairwise_cbf_constraint_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def _minimum_distance(profile, base_min_distance: float) -> float:
    profile = profile or {}
    safety = float(np.clip(max(profile.get("safety", 1.0), 1.0), 1.0, 2.2))
    collision = float(np.clip(max(profile.get("collision", 1.0), 1.0), 1.0, 2.2))
    return float(
        np.clip(
            base_min_distance
            * (1.0 + 0.25 * (safety - 1.0) + 0.25 * (collision - 1.0)),
            0.70 * base_min_distance,
            1.60 * base_min_distance,
        )
    )


def legacy_filter(
    observations: torch.Tensor,
    actions: torch.Tensor,
    profile=None,
    *,
    dt: float = 0.2,
    velocity_retention: float = 0.7,
    action_gain: float = 0.3,
    base_min_distance: float = 1.0,
    iterations: int = 4,
) -> torch.Tensor:
    """Frozen pre-2026-08-20 implementation used only as benchmark oracle."""
    min_distance = _minimum_distance(profile, base_min_distance)
    positions = observations[:, 0:3]
    velocities = observations[:, 3:6]
    filtered = actions.clone()
    for _ in range(iterations):
        for i in range(len(filtered)):
            for j in range(i + 1, len(filtered)):
                relative_position = positions[i] - positions[j]
                distance = torch.linalg.vector_norm(relative_position)
                if float(distance.item()) <= 1e-6:
                    direction = torch.zeros_like(relative_position)
                    direction[0] = 1.0
                else:
                    direction = relative_position / distance
                required = (
                    min_distance
                    - float(distance.item())
                    - dt
                    * velocity_retention
                    * torch.dot(direction, velocities[i] - velocities[j])
                ) / (dt * action_gain)
                violation = required - torch.dot(
                    direction, filtered[i] - filtered[j]
                )
                if float(violation.item()) > 0.0:
                    correction = 0.5 * violation * direction
                    filtered[i] = filtered[i] + correction
                    filtered[j] = filtered[j] - correction
                    filtered = torch.clamp(filtered, -1.0, 1.0)
    return filtered


def legacy_diagnostics(
    observations: torch.Tensor,
    actions: torch.Tensor,
    profile=None,
    *,
    dt: float = 0.2,
    velocity_retention: float = 0.7,
    action_gain: float = 0.3,
    base_min_distance: float = 1.0,
) -> dict[str, float]:
    min_distance = _minimum_distance(profile, base_min_distance)
    positions = observations[:, 0:3]
    velocities = observations[:, 3:6]
    violations = []
    predicted_distances = []
    for i in range(len(actions)):
        for j in range(i + 1, len(actions)):
            relative_position = positions[i] - positions[j]
            distance = torch.linalg.vector_norm(relative_position)
            if float(distance.item()) <= 1e-6:
                direction = torch.zeros_like(relative_position)
                direction[0] = 1.0
            else:
                direction = relative_position / distance
            predicted_distance = (
                float(distance.item())
                + dt
                * velocity_retention
                * float(torch.dot(direction, velocities[i] - velocities[j]).item())
                + dt
                * action_gain
                * float(torch.dot(direction, actions[i] - actions[j]).item())
            )
            predicted_distances.append(predicted_distance)
            violations.append(max(0.0, min_distance - predicted_distance))
    violation_array = np.asarray(violations, dtype=np.float64)
    return {
        "cbf_constraint_max_violation": float(violation_array.max()),
        "cbf_constraint_mean_violation": float(violation_array.mean()),
        "cbf_constraint_violation_fraction": float(np.mean(violation_array > 1e-6)),
        "cbf_predicted_min_pairwise_distance": float(min(predicted_distances)),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_workload(callback, repeats: int, device: torch.device) -> float:
    _sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        callback()
    _sync(device)
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    if args.agents < 2 or args.iterations < 1 or args.repeats < 2 or args.warmup < 0:
        raise ValueError("invalid benchmark dimensions")
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    observations = torch.randn(args.agents, 18, generator=generator).to(device)
    actions = torch.empty(args.agents, 3).uniform_(
        -1.0, 1.0, generator=generator
    ).to(device)
    profile = {"safety": 1.7, "collision": 1.4}

    def run_legacy():
        filtered = legacy_filter(
            observations, actions, profile, iterations=args.iterations
        )
        legacy_diagnostics(observations, filtered, profile)

    def run_optimized():
        apply_pairwise_cbf_filter_with_diagnostics(
            observations, actions, profile, iterations=args.iterations
        )

    for _ in range(args.warmup):
        run_legacy()
        run_optimized()
    legacy_seconds = _time_workload(run_legacy, args.repeats, device)
    optimized_seconds = _time_workload(run_optimized, args.repeats, device)

    legacy_actions = legacy_filter(
        observations, actions, profile, iterations=args.iterations
    )
    optimized_actions, optimized_audit = apply_pairwise_cbf_filter_with_diagnostics(
        observations, actions, profile, iterations=args.iterations
    )
    legacy_audit = legacy_diagnostics(observations, legacy_actions, profile)
    audit_error = max(
        abs(legacy_audit[key] - optimized_audit[key]) for key in legacy_audit
    )
    payload = {
        "schema_version": 1,
        "benchmark": "pairwise_cbf_filter_plus_diagnostics",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "agents": args.agents,
        "pairs": args.agents * (args.agents - 1) // 2,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "seed": args.seed,
        "legacy_total_seconds": legacy_seconds,
        "optimized_total_seconds": optimized_seconds,
        "legacy_mean_milliseconds": 1000.0 * legacy_seconds / args.repeats,
        "optimized_mean_milliseconds": 1000.0 * optimized_seconds / args.repeats,
        "speedup": legacy_seconds / optimized_seconds,
        "max_action_absolute_error": float(
            (legacy_actions - optimized_actions).abs().max().item()
        ),
        "max_diagnostic_absolute_error": float(audit_error),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
