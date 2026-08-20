"""Semantic validation for research variants beyond JSON shape checks."""

from __future__ import annotations

from typing import Dict, Mapping


ALGORITHMS = {"imappo", "mappo", "ippo", "matd3", "rule_planner"}
CRITIC_MODES = {"attention", "uniform", "mlp", "local"}
RESERVED_BASELINE_KEYS = ALGORITHMS


def validate_variant_protocol(spec: Mapping[str, object]) -> Dict[str, object]:
    """Reject labels or modes that do not match the computation actually run."""

    audit = []
    for index, item in enumerate(spec.get("variants", [])):
        if not isinstance(item, Mapping):
            raise ValueError(f"variant {index} must be an object")
        key = str(item.get("key", ""))
        algorithm = str(item.get("algorithm", "imappo"))
        critic_mode = str(item.get("critic_mode", "attention"))
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
        if algorithm == "ippo" and critic_mode != "local":
            raise ValueError(
                f"IPPO variant {key!r} must explicitly declare critic_mode='local'"
            )
        if algorithm == "mappo" and critic_mode not in {"mlp", "uniform"}:
            raise ValueError(
                f"MAPPO variant {key!r} must use a centralized mlp or uniform critic"
            )
        audit.append(
            {
                "key": key,
                "algorithm": algorithm,
                "critic_mode": critic_mode,
                "critic_mode_effective": (
                    "centralized_twin_critics"
                    if algorithm == "matd3"
                    else "not_applicable"
                    if algorithm == "rule_planner"
                    else critic_mode
                ),
            }
        )
    return {"status": "valid", "variant_count": len(audit), "variants": audit}
