"""Structured UAV intent objectives used to operationalize text semantics."""

from __future__ import annotations

from typing import Dict


OBJECTIVE_KEYS = ("distance", "energy", "collision", "safety", "task", "time", "threat")
DEFAULT_INTENT_REWARD_PROFILE: Dict[str, float] = {key: 1.0 for key in OBJECTIVE_KEYS}


def _profile(**overrides: float) -> Dict[str, float]:
    unknown = sorted(set(overrides) - set(OBJECTIVE_KEYS))
    if unknown:
        raise ValueError(f"unknown intent objective keys: {unknown}")
    result = dict(DEFAULT_INTENT_REWARD_PROFILE)
    result.update({key: float(value) for key, value in overrides.items()})
    return result


# These profiles are task definitions, not learned outcomes. Values are bounded to
# moderate multipliers so the reward decomposition remains comparable across intents.
UAV_INTENT_REWARD_PROFILES: Dict[str, Dict[str, float]] = {
    "safety_first": _profile(collision=1.6, safety=1.6, task=0.8, time=0.7, threat=1.4),
    "efficiency_first": _profile(distance=1.3, safety=0.7, task=1.4, time=1.4),
    "balanced": _profile(),
    "energy_saving": _profile(distance=0.7, energy=2.0, task=0.8, time=0.5),
    "aggressive_pursuit": _profile(distance=1.5, collision=0.7, safety=0.6, task=1.3, time=1.5),
    "cautious_exploration": _profile(distance=0.8, collision=1.4, safety=1.5, time=0.6),
    "load_balancing": _profile(task=1.2, safety=1.1),
    "formation_keeping": _profile(collision=1.3, safety=1.3, distance=0.9),
    "threat_avoidance": _profile(safety=1.3, threat=1.8, task=0.8, time=0.7),
    "threat_engagement": _profile(safety=0.7, task=1.3, time=1.2, threat=1.8),
    "perimeter_patrol": _profile(distance=1.1, safety=1.1, task=1.1),
    "center_convergence": _profile(distance=1.4, collision=1.2, task=1.2, time=1.3),
    "decentralized_sweep": _profile(distance=1.1, task=1.2, safety=0.9),
    "relay_coordination": _profile(distance=0.9, safety=1.3, task=1.1, time=0.8),
    "minimal_communication": _profile(safety=1.2, task=0.9, time=0.9),
    "full_coordination": _profile(collision=1.3, safety=1.2, task=1.2),
    "altitude_separation": _profile(collision=1.6, safety=1.4, task=0.9),
    "speed_modulation": _profile(energy=1.2, collision=1.2, time=1.1),
    "reactive_avoidance": _profile(collision=1.0, safety=0.8, task=1.1, time=1.2),
    "predictive_planning": _profile(collision=1.5, safety=1.4, task=1.1, time=0.9),
    "target_priority": _profile(distance=1.3, task=1.4, time=1.2),
    "coverage_maximization": _profile(distance=1.2, safety=1.1, task=1.3),
    "stealth_approach": _profile(energy=1.2, safety=1.4, task=0.9, time=0.7, threat=1.7),
    "rapid_response": _profile(distance=1.5, energy=0.6, safety=0.7, task=1.3, time=1.7),
    "hover_and_observe": _profile(distance=0.5, energy=2.0, safety=1.3, task=0.7, time=0.4),
}


def resolve_intent_reward_profile(label: str) -> Dict[str, float]:
    """Return a copy so environments cannot mutate the registered task profile."""
    return dict(UAV_INTENT_REWARD_PROFILES.get(str(label), DEFAULT_INTENT_REWARD_PROFILE))
