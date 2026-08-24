"""Registered metric contracts shared by runners, audits, and paper artifacts."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple


METRIC_DIRECTIONS: Tuple[tuple[str, bool], ...] = (
    ("collision_rate", True),
    ("task_completion", False),
    ("episode_return", False),
    ("episode_collisions", True),
    ("energy_remaining", False),
    ("action_magnitude", True),
    ("speed", False),
    ("distance_to_target", True),
    ("min_neighbor_distance", False),
    ("threat_zone_violation", True),
    ("distance_to_threat", False),
    ("policy_residual_magnitude", False),
    ("safety_filter_correction_magnitude", False),
    ("cbf_constraint_max_violation", True),
    ("cbf_constraint_mean_violation", True),
    ("cbf_constraint_violation_fraction", True),
    ("cbf_predicted_min_pairwise_distance", False),
    ("safety_filter_solver_success", False),
    ("safety_filter_solver_reported_success", False),
    ("safety_filter_solver_iterations", True),
    ("safety_filter_solver_time_ms", True),
    ("safety_filter_used_fallback", True),
)
METRIC_DIRECTION_MAP = dict(METRIC_DIRECTIONS)
DEFAULT_PRIMARY_METRICS = ("collision_rate", "task_completion")
DEFAULT_ARTIFACT_METRICS = (
    "collision_rate", "task_completion", "episode_return"
)


def resolve_metric_contract(spec: Mapping[str, object]) -> Dict[str, object]:
    """Resolve and validate the predeclared metrics for one study."""
    reporting = spec.get("reporting", {})
    if reporting is None:
        reporting = {}
    if not isinstance(reporting, Mapping):
        raise ValueError("reporting must be an object")
    scope = str(reporting.get("valid_scope", "uav"))
    primary = tuple(map(str, reporting.get(
        "primary_metrics", DEFAULT_PRIMARY_METRICS
    )))
    summary = tuple(map(str, reporting.get(
        "summary_metrics", tuple(METRIC_DIRECTION_MAP)
    )))
    artifact = tuple(map(str, reporting.get(
        "artifact_metrics", DEFAULT_ARTIFACT_METRICS
    )))
    for name, values in (
        ("primary_metrics", primary),
        ("summary_metrics", summary),
        ("artifact_metrics", artifact),
    ):
        if not values or len(values) != len(set(values)):
            raise ValueError(f"reporting.{name} must be non-empty and unique")
        unknown = sorted(set(values) - set(METRIC_DIRECTION_MAP))
        if unknown:
            raise ValueError(f"reporting.{name} contains unknown metrics: {unknown}")
    if not set(primary).issubset(summary):
        raise ValueError("reporting.primary_metrics must be included in summary_metrics")
    if not set(primary).issubset(artifact):
        raise ValueError("reporting.primary_metrics must be included in artifact_metrics")
    if scope == "architecture_only":
        expected = ("episode_return",)
        if primary != expected or summary != expected or artifact != expected:
            raise ValueError(
                "architecture-only studies may aggregate only native episode_return"
            )
    return {
        "valid_scope": scope,
        "primary_metrics": primary,
        "summary_metrics": summary,
        "artifact_metrics": artifact,
        "summary_metric_directions": tuple(
            (metric, METRIC_DIRECTION_MAP[metric]) for metric in summary
        ),
    }
