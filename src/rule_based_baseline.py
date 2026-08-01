"""Deterministic target-tracking and potential-field UAV baseline."""

from __future__ import annotations

import time
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch


def _cbf_min_distance(
    objective_profile: Mapping[str, float] | None,
    base_min_distance: float,
) -> float:
    profile = objective_profile or {}
    # Language may request additional conservatism, but it cannot relax the
    # base collision contract. Collision is a constraint, not a negotiable
    # preference; low safety only affects proactive spacing outside this layer.
    safety = float(np.clip(max(profile.get("safety", 1.0), 1.0), 1.0, 2.2))
    collision = float(np.clip(max(profile.get("collision", 1.0), 1.0), 1.0, 2.2))
    return float(np.clip(
        base_min_distance
        * (1.0 + 0.25 * (safety - 1.0) + 0.25 * (collision - 1.0)),
        0.70 * base_min_distance,
        1.60 * base_min_distance,
    ))


def pairwise_cbf_constraint_diagnostics(
    observations: torch.Tensor,
    actions: torch.Tensor,
    objective_profile: Mapping[str, float] | None = None,
    *,
    dt: float = 0.2,
    velocity_retention: float = 0.7,
    action_gain: float = 0.3,
    base_min_distance: float = 1.0,
) -> Dict[str, float]:
    """Measure residual one-step separation violations for a joint action.

    These diagnostics audit the exact linear half-spaces used by
    :func:`apply_pairwise_cbf_filter`; they are not a claim of continuous-time
    forward invariance.
    """
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("observations and actions must be 2D")
    if len(observations) != len(actions) or actions.size(1) != 3:
        raise ValueError("CBF diagnostics require aligned 3D multi-agent actions")
    if dt <= 0.0 or action_gain <= 0.0:
        raise ValueError("invalid CBF dynamics parameters")
    min_distance = _cbf_min_distance(objective_profile, base_min_distance)
    positions = observations[:, 0:3]
    velocities = observations[:, 3:6]
    violations: List[float] = []
    predicted_distances: List[float] = []
    for i in range(len(actions)):
        for j in range(i + 1, len(actions)):
            relative_position = positions[i] - positions[j]
            distance = torch.linalg.vector_norm(relative_position)
            if float(distance.item()) <= 1e-6:
                direction = torch.zeros_like(relative_position)
                direction[0] = 1.0
            else:
                direction = relative_position / distance
            radial_velocity = torch.dot(direction, velocities[i] - velocities[j])
            radial_action = torch.dot(direction, actions[i] - actions[j])
            predicted_distance = (
                float(distance.item())
                + dt * velocity_retention * float(radial_velocity.item())
                + dt * action_gain * float(radial_action.item())
            )
            predicted_distances.append(predicted_distance)
            violations.append(max(0.0, min_distance - predicted_distance))
    if not violations:
        return {
            "cbf_constraint_max_violation": 0.0,
            "cbf_constraint_mean_violation": 0.0,
            "cbf_constraint_violation_fraction": 0.0,
            "cbf_predicted_min_pairwise_distance": float("inf"),
        }
    violation_array = np.asarray(violations, dtype=np.float64)
    return {
        "cbf_constraint_max_violation": float(violation_array.max()),
        "cbf_constraint_mean_violation": float(violation_array.mean()),
        "cbf_constraint_violation_fraction": float(np.mean(violation_array > 1e-6)),
        "cbf_predicted_min_pairwise_distance": float(min(predicted_distances)),
    }


def apply_pairwise_cbf_filter(
    observations: torch.Tensor,
    actions: torch.Tensor,
    objective_profile: Mapping[str, float] | None = None,
    *,
    dt: float = 0.2,
    velocity_retention: float = 0.7,
    action_gain: float = 0.3,
    base_min_distance: float = 1.0,
    iterations: int = 4,
) -> torch.Tensor:
    """Project joint actions onto linearized one-step pairwise separation constraints.

    The deterministic cyclic projection is dependency-free and preserves the closest
    feasible joint action for each visited half-space before action-box clipping.
    """
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("observations and actions must be 2D")
    if len(observations) != len(actions) or actions.size(1) != 3:
        raise ValueError("CBF filter requires aligned 3D multi-agent actions")
    if dt <= 0.0 or action_gain <= 0.0 or iterations < 1:
        raise ValueError("invalid CBF dynamics/projection parameters")
    min_distance = _cbf_min_distance(objective_profile, base_min_distance)
    positions = observations[:, 0:3]
    velocities = observations[:, 3:6]
    filtered = actions.clone()
    n_agents = int(len(filtered))
    for _ in range(iterations):
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                relative_position = positions[i] - positions[j]
                distance = torch.linalg.vector_norm(relative_position)
                if float(distance.item()) <= 1e-6:
                    direction = torch.zeros_like(relative_position)
                    direction[0] = 1.0
                else:
                    direction = relative_position / distance
                relative_velocity = velocities[i] - velocities[j]
                required_radial_action = (
                    min_distance
                    - float(distance.item())
                    - dt * velocity_retention * torch.dot(direction, relative_velocity)
                ) / (dt * action_gain)
                actual_radial_action = torch.dot(
                    direction, filtered[i] - filtered[j]
                )
                violation = required_radial_action - actual_radial_action
                if float(violation.item()) > 0.0:
                    correction = 0.5 * violation * direction
                    filtered[i] = filtered[i] + correction
                    filtered[j] = filtered[j] - correction
                    filtered = torch.clamp(filtered, -1.0, 1.0)
    return filtered


def apply_pairwise_qp_filter(
    observations: torch.Tensor,
    actions: torch.Tensor,
    objective_profile: Mapping[str, float] | None = None,
    *,
    dt: float = 0.2,
    velocity_retention: float = 0.7,
    action_gain: float = 0.3,
    base_min_distance: float = 1.0,
    tolerance: float = 1e-7,
    max_iterations: int = 100,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Solve the closest box-constrained action satisfying pairwise CBF rows.

    The problem is a small convex quadratic program. SciPy SLSQP is used as a
    dependency already frozen in the research environment. When numerical
    termination fails, a longer cyclic projection is evaluated as a fallback and
    the candidate with the smaller audited violation is returned. ``success`` is
    true only when the returned action satisfies the registered tolerance.
    """
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("observations and actions must be 2D")
    if len(observations) != len(actions) or actions.size(1) != 3:
        raise ValueError("QP-CBF filter requires aligned 3D multi-agent actions")
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("invalid QP-CBF solver parameters")
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise RuntimeError("pairwise_qp safety filtering requires scipy") from exc

    start_time = time.perf_counter()
    obs_np = observations.detach().cpu().numpy().astype(np.float64, copy=False)
    action_np = actions.detach().cpu().numpy().astype(np.float64, copy=False)
    n_agents = len(action_np)
    rows = []
    lower_bounds = []
    min_distance = _cbf_min_distance(objective_profile, base_min_distance)
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            relative_position = obs_np[i, 0:3] - obs_np[j, 0:3]
            distance = float(np.linalg.norm(relative_position))
            if distance <= 1e-6:
                direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                direction = relative_position / distance
            relative_velocity = obs_np[i, 3:6] - obs_np[j, 3:6]
            required = (
                min_distance
                - distance
                - dt * velocity_retention * float(direction @ relative_velocity)
            ) / (dt * action_gain)
            row = np.zeros(3 * n_agents, dtype=np.float64)
            row[3 * i:3 * i + 3] = direction
            row[3 * j:3 * j + 3] = -direction
            rows.append(row)
            lower_bounds.append(required)
    matrix = np.asarray(rows, dtype=np.float64)
    lower = np.asarray(lower_bounds, dtype=np.float64)
    target = np.clip(action_np.reshape(-1), -1.0, 1.0)

    def objective(x: np.ndarray) -> float:
        delta = x - target
        return 0.5 * float(delta @ delta)

    def objective_jacobian(x: np.ndarray) -> np.ndarray:
        return x - target

    constraints = []
    if len(matrix):
        constraints.append({
            "type": "ineq",
            "fun": lambda x: matrix @ x - lower,
            "jac": lambda x: matrix,
        })
    result = minimize(
        objective,
        target,
        jac=objective_jacobian,
        method="SLSQP",
        bounds=[(-1.0, 1.0)] * len(target),
        constraints=constraints,
        options={"ftol": tolerance, "maxiter": max_iterations, "disp": False},
    )
    candidate_np = (
        np.clip(np.asarray(result.x, dtype=np.float64), -1.0, 1.0)
        if np.asarray(result.x).shape == target.shape and np.isfinite(result.x).all()
        else target
    )
    candidate = torch.as_tensor(
        candidate_np.reshape(action_np.shape), dtype=actions.dtype, device=actions.device
    )
    candidate_diagnostics = pairwise_cbf_constraint_diagnostics(
        observations,
        candidate,
        objective_profile,
        dt=dt,
        velocity_retention=velocity_retention,
        action_gain=action_gain,
        base_min_distance=base_min_distance,
    )
    used_fallback = False
    filtered = candidate
    diagnostics = candidate_diagnostics
    if (
        not bool(result.success)
        or candidate_diagnostics["cbf_constraint_max_violation"] > tolerance
    ):
        fallback = apply_pairwise_cbf_filter(
            observations,
            actions,
            objective_profile,
            dt=dt,
            velocity_retention=velocity_retention,
            action_gain=action_gain,
            base_min_distance=base_min_distance,
            iterations=max(16, min(max_iterations, 64)),
        )
        fallback_diagnostics = pairwise_cbf_constraint_diagnostics(
            observations,
            fallback,
            objective_profile,
            dt=dt,
            velocity_retention=velocity_retention,
            action_gain=action_gain,
            base_min_distance=base_min_distance,
        )
        used_fallback = (
            fallback_diagnostics["cbf_constraint_max_violation"]
            + tolerance
            < candidate_diagnostics["cbf_constraint_max_violation"]
        )
        if used_fallback:
            filtered = fallback
            diagnostics = fallback_diagnostics
    elapsed_ms = 1000.0 * (time.perf_counter() - start_time)
    feasible = diagnostics["cbf_constraint_max_violation"] <= tolerance
    return filtered, {
        "safety_filter_solver_success": float(feasible),
        "safety_filter_solver_reported_success": float(bool(result.success)),
        "safety_filter_solver_iterations": float(getattr(result, "nit", 0)),
        "safety_filter_solver_time_ms": float(elapsed_ms),
        "safety_filter_used_fallback": float(used_fallback),
        "safety_filter_solver_status": float(getattr(result, "status", -1)),
        **diagnostics,
    }


def compute_rule_actions(
    observations: torch.Tensor,
    posture: str = "neutral",
    action_mask: torch.Tensor | None = None,
    objective_profile: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Compute the shared target-tracking and collision-avoidance prior."""
    obs = observations
    velocity = obs[:, 3:6]
    target_delta = obs[:, 6:9]
    target_direction = target_delta / torch.clamp(
        torch.linalg.vector_norm(target_delta, dim=-1, keepdim=True), min=1e-6
    )

    profile = None
    proactive_safety_radius = 2.0
    if objective_profile is not None:
        profile = {
            key: float(np.clip(objective_profile.get(key, 1.0), 0.3, 2.2))
            for key in (
                "distance", "energy", "collision", "safety", "task", "time", "threat"
            )
        }
        # Safety is a proactive spacing objective, whereas collision controls the
        # short-range reactive barrier. Separating the radii makes both concepts
        # behaviorally identifiable in counterfactual evaluation.
        proactive_safety_radius = float(np.clip(
            2.0 + 1.15 * (profile["safety"] - 1.0), 1.15, 3.35
        ))

    repulsion = torch.zeros_like(target_direction)
    has_threat_features = obs.size(1) >= 15 and (obs.size(1) - 15) % 6 == 0
    threat_delta = obs[:, 12:15] if has_threat_features else None
    neighbor_features = obs[:, 15:] if has_threat_features else obs[:, 12:]
    for start in range(0, neighbor_features.size(1), 6):
        relative_position = neighbor_features[:, start:start + 3]
        if relative_position.size(1) < 3:
            continue
        distance = torch.linalg.vector_norm(relative_position, dim=-1, keepdim=True)
        active = (distance > 1e-6) & (distance < proactive_safety_radius)
        repulsion = repulsion - torch.where(
            active,
            relative_position / torch.clamp(distance.pow(3), min=1e-3),
            torch.zeros_like(relative_position),
        )

    action_ceiling = 1.0
    threat_response = torch.zeros_like(target_direction)
    if profile is not None:
        target_gain = float(np.clip(
            1.0
            + 0.35 * (profile["distance"] - 1.0)
            + 0.25 * (profile["task"] - 1.0)
            + 0.30 * (profile["time"] - 1.0)
            - 0.20 * (profile["energy"] - 1.0)
            - 0.20 * (profile["safety"] - 1.0)
            - 0.35 * (profile["threat"] - 1.0),
            0.4,
            1.6,
        ))
        safety_gain = float(np.clip(
            1.0
            + 1.00 * (profile["safety"] - 1.0)
            + 0.50 * (max(profile["collision"], 1.0) - 1.0),
            0.2,
            2.4,
        ))
        damping_gain = float(np.clip(
            0.20
            + 0.35 * (profile["energy"] - 1.0)
            - 0.15 * (profile["time"] - 1.0)
            + 0.10 * (profile["safety"] - 1.0),
            0.05,
            0.65,
        ))
        action_ceiling = float(np.clip(
            1.0
            - 0.35 * (profile["energy"] - 1.0)
            + 0.20 * (profile["time"] - 1.0),
            0.30,
            1.0,
        ))
        if threat_delta is not None:
            threat_distance = torch.linalg.vector_norm(
                threat_delta, dim=-1, keepdim=True
            )
            threat_direction = threat_delta / torch.clamp(
                threat_distance, min=1e-6
            )
            active_threat = (threat_distance > 1e-6) & (threat_distance < 2.5)
            threat_response = torch.where(
                active_threat,
                -1.20 * (profile["threat"] - 1.0) * threat_direction
                / torch.clamp(threat_distance, min=0.5),
                torch.zeros_like(threat_direction),
            )
    else:
        gains = {
            "attack": (1.35, 0.45, 0.10),
            "stealth": (0.75, 1.50, 0.30),
            "neutral": (1.00, 1.00, 0.20),
        }
        target_gain, safety_gain, damping_gain = gains.get(
            str(posture), gains["neutral"]
        )
    actions = (
        target_gain * target_direction
        + safety_gain * repulsion
        + threat_response
        - damping_gain * velocity
    )
    actions = torch.clamp(actions, -action_ceiling, action_ceiling)
    return actions if action_mask is None else actions * action_mask


class RuleBasedUAVPolicy:
    """A non-learning controller with task attraction and neighbor repulsion."""

    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.current_posture = "neutral"
        self.current_label = ""
        self.current_objective_profile: Mapping[str, float] | None = None
        self._last_safety_filter_correction = torch.zeros(
            self.config.n_agents, self.config.action_dim, device=self.device
        )
        self._last_safety_solver_diagnostics: Dict[str, float] = {}

    def intent_representation_metadata(self) -> Dict[str, object]:
        return {
            "representation_type": "structured_rule_context",
            "semantic_geometry": False,
            "learning": False,
            "controller": "target_tracking_plus_neighbor_potential_field",
        }

    def _action_mask_for_posture(self, posture: str | None) -> torch.Tensor:
        del posture
        return torch.ones(
            self.config.n_agents,
            self.config.action_dim,
            dtype=torch.float32,
            device=self.device,
        )

    def set_evaluation_context(self, label: str, posture: str) -> None:
        self.current_label = str(label)
        self.current_posture = str(posture)

    def set_evaluation_objective_profile(
        self, profile: Mapping[str, float] | None
    ) -> None:
        self.current_objective_profile = (
            None if profile is None else {str(key): float(value) for key, value in profile.items()}
        )

    def evaluation_intent_and_mask(
        self, mode: str = "standard"
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        posture = "stealth" if mode == "dense" else "neutral"
        label = "safety_first" if mode == "dense" else "balanced"
        self.set_evaluation_context(label, posture)
        return (
            torch.zeros(self.config.intent_dim, device=self.device),
            self._action_mask_for_posture(posture),
            label,
        )

    def encode_intent_queries(self, entries: List[Tuple[str, str]]) -> torch.Tensor:
        return torch.zeros(
            len(entries), self.config.intent_dim, dtype=torch.float32, device=self.device
        )

    def select_actions(
        self,
        observations: torch.Tensor,
        intent: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del intent, deterministic
        actions = compute_rule_actions(
            observations.to(self.device),
            self.current_posture,
            action_mask,
            objective_profile=(
                self.current_objective_profile
                if self.config.rule_prior_context == "oracle_profile" else None
            ),
        )
        self._last_safety_solver_diagnostics = {}
        if self.config.safety_filter_mode == "pairwise_cbf":
            unfiltered_actions = actions
            profile = (
                self.current_objective_profile
                if self.config.rule_prior_context == "oracle_profile" else None
            )
            filter_start = time.perf_counter()
            actions = apply_pairwise_cbf_filter(
                observations.to(self.device),
                unfiltered_actions,
                profile,
                base_min_distance=self.config.cbf_base_min_distance,
                iterations=self.config.cbf_iterations,
            )
            cyclic_audit = pairwise_cbf_constraint_diagnostics(
                observations.to(self.device),
                actions,
                profile,
                base_min_distance=self.config.cbf_base_min_distance,
            )
            self._last_safety_solver_diagnostics = {
                "safety_filter_solver_success": float(
                    cyclic_audit["cbf_constraint_max_violation"]
                    <= self.config.cbf_solver_tolerance
                ),
                "safety_filter_solver_reported_success": float(
                    cyclic_audit["cbf_constraint_max_violation"]
                    <= self.config.cbf_solver_tolerance
                ),
                "safety_filter_solver_iterations": float(self.config.cbf_iterations),
                "safety_filter_solver_time_ms": float(
                    1000.0 * (time.perf_counter() - filter_start)
                ),
                "safety_filter_used_fallback": 0.0,
                **cyclic_audit,
            }
            self._last_safety_filter_correction = (
                actions - unfiltered_actions
            ).detach()
            self._last_safety_filter_profile = profile
        elif self.config.safety_filter_mode == "pairwise_qp":
            unfiltered_actions = actions
            profile = (
                self.current_objective_profile
                if self.config.rule_prior_context == "oracle_profile" else None
            )
            actions, self._last_safety_solver_diagnostics = apply_pairwise_qp_filter(
                observations.to(self.device),
                unfiltered_actions,
                profile,
                base_min_distance=self.config.cbf_base_min_distance,
                tolerance=self.config.cbf_solver_tolerance,
                max_iterations=self.config.cbf_solver_max_iterations,
            )
            self._last_safety_filter_correction = (
                actions - unfiltered_actions
            ).detach()
            self._last_safety_filter_profile = profile
        else:
            self._last_safety_filter_profile = (
                self.current_objective_profile
                if self.config.rule_prior_context == "oracle_profile" else None
            )
        log_probs = torch.zeros(
            actions.size(0), dtype=torch.float32, device=self.device
        )
        return actions, log_probs
