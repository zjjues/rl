"""Portable high-level controller and PyBullet transfer evaluation utilities.

The functions in this module intentionally depend only on NumPy/SciPy.  This
keeps the rigid-body simulator isolated from the PyTorch research environment
while preserving an auditable velocity-command interface between simulators.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np


PROFILE_KEYS = ("distance", "energy", "collision", "safety", "task", "time", "threat")


def resolved_profile(profile: Mapping[str, float] | None) -> Dict[str, float]:
    source = profile or {}
    return {key: float(np.clip(source.get(key, 1.0), 0.3, 2.2)) for key in PROFILE_KEYS}


def safety_distance(profile: Mapping[str, float] | None, base_distance: float) -> float:
    """Return a language-tightenable, never language-relaxable safety distance."""
    values = resolved_profile(profile)
    safety = max(values["safety"], 1.0)
    collision = max(values["collision"], 1.0)
    return float(base_distance * (1.0 + 0.25 * (safety - 1.0) + 0.25 * (collision - 1.0)))


def latency_robust_margin(max_speed: float, tracking_latency_budget: float) -> float:
    """Worst-case relative closing distance for two speed-limited vehicles."""
    if max_speed <= 0.0 or tracking_latency_budget < 0.0:
        raise ValueError("speed must be positive and latency budget non-negative")
    return float(2.0 * max_speed * tracking_latency_budget)


def pairwise_min_distance(positions: np.ndarray) -> float:
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) < 2:
        return float("inf")
    return float(min(
        np.linalg.norm(positions[i] - positions[j])
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
    ))


def crossing_scenario(seed: int, n_agents: int = 4, radius: float = 0.70) -> Tuple[np.ndarray, np.ndarray]:
    """Create a deterministic, jittered antipodal target-swap scenario."""
    if n_agents < 2:
        raise ValueError("crossing scenarios require at least two agents")
    rng = np.random.default_rng(seed)
    angles = 2.0 * np.pi * np.arange(n_agents, dtype=np.float64) / n_agents
    initial = np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.ones(n_agents)))
    initial[:, :2] += rng.normal(0.0, 0.012, size=(n_agents, 2))
    targets = initial[np.roll(np.arange(n_agents), n_agents // 2)].copy()
    return initial, targets


def corridor_scenario(seed: int, n_agents: int = 4, radius: float = 0.72) -> Tuple[np.ndarray, np.ndarray]:
    """Create two opposing traffic streams through a narrow shared corridor."""
    if n_agents != 4:
        return crossing_scenario(seed, n_agents, radius)
    rng = np.random.default_rng(seed)
    initial = np.asarray([
        [-radius, -0.10, 1.00], [-radius, 0.10, 1.04],
        [radius, -0.10, 1.04], [radius, 0.10, 1.00],
    ], dtype=np.float64)
    initial += rng.normal(0.0, 0.008, size=initial.shape)
    targets = initial[[3, 2, 1, 0]].copy()
    return initial, targets


def nominal_goal_velocity(
    positions: np.ndarray,
    velocities: np.ndarray,
    targets: np.ndarray,
    max_speed: float,
    profile: Mapping[str, float] | None = None,
    swirl_gain: float = 0.06,
    transit_lane_span: float = 0.18,
) -> np.ndarray:
    """Compute target tracking with deterministic altitude lanes and circulation.

    The altitude convention depends only on agent index, not on the randomized
    scenario.  It supplies a minimal liveness mechanism for reciprocal target
    swaps; the safety projection remains responsible for separation.
    """
    values = resolved_profile(profile)
    speed_scale = float(np.clip(
        1.0 - 0.35 * (values["energy"] - 1.0) + 0.20 * (values["time"] - 1.0),
        0.30,
        1.0,
    ))
    positions = np.asarray(positions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    transit_targets = targets.copy()
    lane_offsets = np.linspace(-transit_lane_span, transit_lane_span, len(positions))
    horizontal_error = np.linalg.norm(targets[:, :2] - positions[:, :2], axis=1)
    in_transit = horizontal_error > 0.22
    transit_targets[in_transit, 2] += lane_offsets[in_transit]
    delta = transit_targets - positions
    distance = np.linalg.norm(delta, axis=1, keepdims=True)
    direction = np.divide(delta, distance, out=np.zeros_like(delta), where=distance > 1e-9)
    command = max_speed * speed_scale * direction - 0.08 * np.asarray(velocities, dtype=np.float64)

    # A shared clockwise convention avoids exact reciprocal deadlock without
    # using target or seed-specific privileged information.
    for i in range(len(command)):
        for j in range(len(command)):
            if i == j:
                continue
            relative = positions[i] - positions[j]
            separation = float(np.linalg.norm(relative))
            if 1e-6 < separation < 0.55:
                tangent = np.asarray([-relative[1], relative[0], 0.0]) / separation
                command[i] += swirl_gain * max_speed * tangent / max(separation, 0.15)
    norms = np.linalg.norm(command, axis=1, keepdims=True)
    return command * np.minimum(1.0, max_speed * speed_scale / np.maximum(norms, 1e-12))


def velocity_constraint_diagnostics(
    positions: np.ndarray,
    velocities: np.ndarray,
    commands: np.ndarray,
    *,
    min_distance: float,
    horizon: float,
    response: float = 0.85,
) -> Dict[str, float]:
    violations = []
    predictions = []
    for i in range(len(commands)):
        for j in range(i + 1, len(commands)):
            relative = positions[i] - positions[j]
            distance = float(np.linalg.norm(relative))
            direction = relative / distance if distance > 1e-9 else np.asarray([1.0, 0.0, 0.0])
            blended_relative_velocity = (
                response * (commands[i] - commands[j])
                + (1.0 - response) * (velocities[i] - velocities[j])
            )
            predicted = distance + horizon * float(direction @ blended_relative_velocity)
            predictions.append(predicted)
            violations.append(max(0.0, min_distance - predicted))
    array = np.asarray(violations, dtype=np.float64)
    return {
        "constraint_max_violation": float(array.max(initial=0.0)),
        "constraint_mean_violation": float(array.mean()) if array.size else 0.0,
        "constraint_violation_fraction": float(np.mean(array > 1e-6)) if array.size else 0.0,
        "predicted_min_distance": float(min(predictions, default=float("inf"))),
    }


def project_velocity_commands(
    positions: np.ndarray,
    velocities: np.ndarray,
    nominal: np.ndarray,
    *,
    min_distance: float,
    max_speed: float,
    horizon: float,
    mode: str = "qp",
    response: float = 0.85,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Project velocity commands onto linearized pairwise separation rows."""
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    nominal = np.clip(np.asarray(nominal, dtype=np.float64), -max_speed, max_speed)
    if mode == "none":
        diagnostics = velocity_constraint_diagnostics(
            positions, velocities, nominal, min_distance=min_distance, horizon=horizon, response=response
        )
        return nominal, {"solver_success": float(diagnostics["constraint_max_violation"] <= tolerance),
                         "solver_reported_success": 1.0, "solver_iterations": 0.0,
                         "solver_time_ms": 0.0, **diagnostics}
    if mode not in {"cyclic", "qp"}:
        raise ValueError("mode must be one of: none, cyclic, qp")

    rows, lower = [], []
    n_agents = len(nominal)
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            relative = positions[i] - positions[j]
            distance = float(np.linalg.norm(relative))
            direction = relative / distance if distance > 1e-9 else np.asarray([1.0, 0.0, 0.0])
            retained = (1.0 - response) * float(direction @ (velocities[i] - velocities[j]))
            required = ((min_distance - distance) / horizon - retained) / response
            row = np.zeros(3 * n_agents, dtype=np.float64)
            row[3 * i:3 * i + 3] = direction
            row[3 * j:3 * j + 3] = -direction
            rows.append(row)
            lower.append(required)
    matrix = np.asarray(rows, dtype=np.float64)
    lower_array = np.asarray(lower, dtype=np.float64)
    target = nominal.reshape(-1)
    started = time.perf_counter()

    def cyclic_projection(iterations: int) -> np.ndarray:
        candidate = target.copy()
        for _ in range(iterations):
            for row, bound in zip(matrix, lower_array):
                residual = bound - float(row @ candidate)
                if residual > 0.0:
                    candidate += residual * row / max(float(row @ row), 1e-12)
                    candidate = np.clip(candidate, -max_speed, max_speed)
        return candidate

    reported_success = True
    iterations = max_iterations
    if mode == "cyclic":
        candidate = cyclic_projection(max_iterations)
    else:
        from scipy.optimize import minimize

        result = minimize(
            lambda x: 0.5 * float((x - target) @ (x - target)),
            target,
            jac=lambda x: x - target,
            method="SLSQP",
            bounds=[(-max_speed, max_speed)] * len(target),
            constraints=[{"type": "ineq", "fun": lambda x: matrix @ x - lower_array,
                          "jac": lambda x: matrix}],
            options={"ftol": tolerance, "maxiter": max_iterations, "disp": False},
        )
        candidate = np.clip(np.asarray(result.x, dtype=np.float64), -max_speed, max_speed)
        reported_success = bool(result.success)
        iterations = int(getattr(result, "nit", 0))
    commands = candidate.reshape(nominal.shape)
    diagnostics = velocity_constraint_diagnostics(
        positions, velocities, commands, min_distance=min_distance, horizon=horizon, response=response
    )
    return commands, {
        "solver_success": float(diagnostics["constraint_max_violation"] <= tolerance),
        "solver_reported_success": float(reported_success),
        "solver_iterations": float(iterations),
        "solver_time_ms": 1000.0 * (time.perf_counter() - started),
        **diagnostics,
    }


def velocity_to_aviary_action(commands: np.ndarray, speed_limit: float) -> np.ndarray:
    """Convert metric velocity commands to VelocityAviary's direction/fraction action."""
    commands = np.asarray(commands, dtype=np.float64)
    norms = np.linalg.norm(commands, axis=1, keepdims=True)
    direction = np.divide(commands, norms, out=np.zeros_like(commands), where=norms > 1e-12)
    fraction = np.clip(norms / speed_limit, 0.0, 1.0)
    return np.concatenate((direction, fraction), axis=1).astype(np.float32)


@dataclass(frozen=True)
class TransferEpisodeConfig:
    steps: int = 240
    ctrl_freq: int = 30
    pyb_freq: int = 240
    base_min_distance: float = 0.22
    collision_distance: float = 0.10
    goal_tolerance: float = 0.16
    constraint_horizon: float = 0.35
    filter_mode: str = "qp"
    tracking_latency_budget: float = 0.08
    physics: str = "pyb"


def evaluate_transfer_episode(
    seed: int,
    scenario: str,
    profile: Mapping[str, float] | None,
    config: TransferEpisodeConfig,
) -> Dict[str, float]:
    """Run one headless Crazyflie velocity-control episode in PyBullet."""
    try:
        from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
        from gym_pybullet_drones.utils.enums import Physics
    except ImportError as exc:
        raise RuntimeError("run this evaluator inside the rl-pybullet environment") from exc

    initial, targets = (
        crossing_scenario(seed) if scenario == "crossing" else corridor_scenario(seed)
    )
    physics_by_name = {member.value: member for member in Physics}
    if config.physics not in physics_by_name:
        raise ValueError(f"unknown PyBullet physics mode: {config.physics}")
    env = VelocityAviary(
        num_drones=len(initial), initial_xyzs=initial, pyb_freq=config.pyb_freq,
        ctrl_freq=config.ctrl_freq, gui=False, record=False, user_debug_gui=False,
        physics=physics_by_name[config.physics],
    )
    values = resolved_profile(profile)
    min_distance = safety_distance(values, config.base_min_distance)
    speed_limit = float(env.SPEED_LIMIT)
    robust_margin = (
        latency_robust_margin(speed_limit, config.tracking_latency_budget)
        if config.filter_mode == "robust_qp" else 0.0
    )
    constraint_distance = min_distance + robust_margin
    projection_mode = "qp" if config.filter_mode == "robust_qp" else config.filter_mode
    minimum_observed = float("inf")
    collisions = 0
    safety_violations = 0
    corrections, solver_times, violations, successes = [], [], [], []
    command_energy = 0.0
    obs, _ = env.reset(seed=seed)
    try:
        for _ in range(config.steps):
            positions, velocities = obs[:, 0:3], obs[:, 10:13]
            nominal = nominal_goal_velocity(positions, velocities, targets, speed_limit, values)
            commands, audit = project_velocity_commands(
                positions, velocities, nominal, min_distance=constraint_distance,
                max_speed=speed_limit, horizon=config.constraint_horizon,
                mode=projection_mode,
            )
            corrections.append(float(np.linalg.norm(commands - nominal)))
            solver_times.append(audit["solver_time_ms"])
            violations.append(audit["constraint_max_violation"])
            successes.append(audit["solver_success"])
            command_energy += float(np.mean(np.square(commands / speed_limit)))
            obs, _, _, _, _ = env.step(velocity_to_aviary_action(commands, speed_limit))
            separation = pairwise_min_distance(obs[:, 0:3])
            minimum_observed = min(minimum_observed, separation)
            collisions += int(separation < config.collision_distance)
            safety_violations += int(separation < min_distance)
    finally:
        env.close()
    final_errors = np.linalg.norm(obs[:, 0:3] - targets, axis=1)
    return {
        "minimum_pairwise_distance": float(minimum_observed),
        "collision_step_fraction": float(collisions / config.steps),
        "safety_violation_step_fraction": float(safety_violations / config.steps),
        "goal_success_fraction": float(np.mean(final_errors < config.goal_tolerance)),
        "final_goal_rmse": float(np.sqrt(np.mean(np.square(final_errors)))),
        "normalized_command_energy": float(command_energy / config.steps),
        "mean_filter_correction": float(np.mean(corrections)),
        "solver_success_fraction": float(np.mean(successes)),
        "constraint_max_violation": float(max(violations, default=0.0)),
        "mean_solver_time_ms": float(np.mean(solver_times)),
        "safety_distance": float(min_distance),
        "constraint_distance": float(constraint_distance),
        "robust_margin": float(robust_margin),
        "speed_limit_mps": float(speed_limit),
    }
