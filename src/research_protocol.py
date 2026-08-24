"""Semantic validation for research variants beyond JSON shape checks."""

from __future__ import annotations

from typing import Dict, Mapping


ALGORITHMS = {"imappo", "mappo", "ippo", "happo", "matd3", "rule_planner"}
CRITIC_MODES = {"attention", "uniform", "mlp", "local"}
RESERVED_BASELINE_KEYS = ALGORITHMS


def validate_variant_protocol(spec: Mapping[str, object]) -> Dict[str, object]:
    """Reject labels or modes that do not match the computation actually run."""

    audit = []
    environment = spec.get("environment", {})
    environment_name = (
        str(environment.get("name", ""))
        if isinstance(environment, Mapping) else ""
    )
    is_vmas = environment_name.startswith("vmas:")
    for index, item in enumerate(spec.get("variants", [])):
        if not isinstance(item, Mapping):
            raise ValueError(f"variant {index} must be an object")
        key = str(item.get("key", ""))
        algorithm = str(item.get("algorithm", "imappo"))
        critic_mode = str(item.get("critic_mode", "attention"))
        relevance_gate_path = str(item.get("preference_relevance_gate_path", ""))
        if algorithm not in ALGORITHMS:
            raise ValueError(f"variant {key!r} uses unsupported algorithm {algorithm!r}")
        if critic_mode not in CRITIC_MODES:
            raise ValueError(
                f"variant {key!r} uses unsupported critic_mode {critic_mode!r}; "
                "legacy 'concat' was not implemented and executed as attention"
            )
        if key.lower() in RESERVED_BASELINE_KEYS and algorithm != key.lower():
            raise ValueError(
                f"reserved baseline key {key!r} must use algorithm={key.lower()!r}, "
                f"not {algorithm!r}"
            )
        if relevance_gate_path and (
            algorithm != "imappo"
            or str(item.get("intent_source", "")) != "objective_grounded_semantic"
            or str(item.get("intent_profile_decoder", "none")) == "none"
        ):
            raise ValueError(
                f"variant {key!r} may use a preference relevance gate only with "
                "objective-grounded I-MAPPO and an active profile decoder"
            )
        if is_vmas:
            required_neutral = {
                "intent_source": "none",
                "intent_profile_decoder": "none",
                "disable_intent_reward": True,
                "use_action_mask": False,
                "policy_mode": "direct",
                "safety_filter_mode": "none",
            }
            mismatches = {
                field: (item.get(field), expected)
                for field, expected in required_neutral.items()
                if item.get(field) != expected
            }
            if mismatches:
                raise ValueError(
                    f"VMAS variant {key!r} must be architecture-only and may not "
                    f"inherit UAV language/safety semantics: {mismatches}"
                )
        if algorithm == "ippo" and critic_mode != "local":
            raise ValueError(
                f"IPPO variant {key!r} must explicitly declare critic_mode='local'"
            )
        if algorithm == "mappo" and critic_mode not in {"mlp", "uniform"}:
            raise ValueError(
                f"MAPPO variant {key!r} must use a centralized mlp or uniform critic"
            )
        if algorithm == "happo":
            required = {
                "critic_mode": "mlp",
                "intent_source": "none",
                "use_action_mask": False,
                "policy_mode": "direct",
                "safety_filter_mode": "none",
                "actor_parameter_sharing": "independent",
                "update_scheme": "random_sequential_likelihood_factor",
            }
            mismatches = {
                field: (item.get(field), expected)
                for field, expected in required.items()
                if item.get(field) != expected
            }
            if mismatches:
                raise ValueError(
                    f"HAPPO variant {key!r} violates independent sequential "
                    f"implementation contract: {mismatches}"
                )
        audit.append(
            {
                "key": key,
                "algorithm": algorithm,
                "critic_mode": critic_mode,
                "critic_mode_effective": (
                    "centralized_twin_critics"
                    if algorithm == "matd3"
                    else "centralized_mlp_sequential_independent_actors"
                    if algorithm == "happo"
                    else "not_applicable"
                    if algorithm == "rule_planner"
                    else critic_mode
                ),
                "preference_relevance_gate_enabled": bool(relevance_gate_path),
                "evaluation_scope": (
                    "architecture_only_no_language_claim" if is_vmas else "uav"
                ),
            }
        )
    return {"status": "valid", "variant_count": len(audit), "variants": audit}
