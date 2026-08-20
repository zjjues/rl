"""Machine-checkable contracts for causal ablation studies."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Dict, Mapping


PRIMARY_METRICS = {"collision_rate", "task_completion", "episode_return"}


def _variant_map(spec: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    variants: Dict[str, Dict[str, object]] = {}
    for item in spec.get("variants", []):
        if not isinstance(item, Mapping) or "key" not in item:
            raise ValueError("every variant must be an object with a key")
        key = str(item["key"])
        if key in variants:
            raise ValueError(f"duplicate variant key in ablation contract: {key!r}")
        variants[key] = dict(item)
    return variants


def variant_differences(
    reference: Mapping[str, object], candidate: Mapping[str, object]
) -> list[str]:
    """Return the explicit fields that differ, excluding the identity key."""

    fields = (set(reference) | set(candidate)) - {"key"}
    return sorted(
        field for field in fields if reference.get(field) != candidate.get(field)
    )


def validate_ablation_contract(spec: Mapping[str, object]) -> Dict[str, object]:
    """Validate a rooted, fully covered graph of predeclared ablation contrasts.

    Each non-treatment variant must have exactly one incoming comparison.  The
    declared changed fields must exactly match the variant dictionaries, which
    prevents accidental multi-factor drift after an experiment is registered.
    """

    contract = spec.get("ablation_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("ablation_contract must be a JSON object")
    if int(contract.get("version", -1)) != 1:
        raise ValueError("ablation_contract.version must be 1")

    variants = _variant_map(spec)
    if len(variants) < 2:
        raise ValueError("an ablation contract requires at least two variants")
    treatment_key = str(contract.get("treatment_key", ""))
    if treatment_key not in variants:
        raise ValueError("ablation treatment_key must name a configured variant")
    if str(spec.get("treatment_key", "")) != treatment_key:
        raise ValueError("study treatment_key must match ablation_contract.treatment_key")

    comparisons = contract.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("ablation_contract.comparisons must be a non-empty list")

    outcomes: list[str] = []
    graph: dict[str, list[str]] = defaultdict(list)
    audit_rows = []
    for index, item in enumerate(comparisons):
        if not isinstance(item, Mapping):
            raise ValueError(f"ablation comparison {index} must be an object")
        reference = str(item.get("reference", ""))
        variant = str(item.get("variant", ""))
        factor = str(item.get("factor", "")).strip()
        if reference not in variants or variant not in variants:
            raise ValueError(
                f"ablation comparison {index} references an unknown variant"
            )
        if reference == variant:
            raise ValueError(f"ablation comparison {index} compares a variant to itself")
        if not factor:
            raise ValueError(f"ablation comparison {index} has no factor name")
        changed_fields = item.get("changed_fields")
        if not isinstance(changed_fields, list) or not changed_fields:
            raise ValueError(
                f"ablation comparison {index} must declare changed_fields"
            )
        declared = [str(field) for field in changed_fields]
        if len(declared) != len(set(declared)) or "key" in declared:
            raise ValueError(
                f"ablation comparison {index} has invalid changed_fields"
            )
        missing_explicit = [
            field
            for field in declared
            if field not in variants[reference] or field not in variants[variant]
        ]
        if missing_explicit:
            raise ValueError(
                f"ablation comparison {reference}->{variant} does not explicitly "
                f"set fields {missing_explicit} on both variants"
            )
        observed = variant_differences(variants[reference], variants[variant])
        if sorted(declared) != observed:
            raise ValueError(
                f"ablation comparison {reference}->{variant} declares changes "
                f"{sorted(declared)} but observed {observed}"
            )
        hypothesis = str(item.get("hypothesis", "")).strip()
        if not hypothesis:
            raise ValueError(
                f"ablation comparison {reference}->{variant} has no hypothesis"
            )
        metrics = item.get("primary_metrics")
        tiers = item.get("primary_tiers")
        if (
            not isinstance(metrics, list)
            or not metrics
            or any(str(metric) not in PRIMARY_METRICS for metric in metrics)
        ):
            raise ValueError(
                f"ablation comparison {reference}->{variant} has invalid primary_metrics"
            )
        configured_tiers = set(
            str(key)
            for key in spec.get("evaluation", {}).get("risk_tiers", {})
        )
        if (
            not isinstance(tiers, list)
            or not tiers
            or any(str(tier) not in configured_tiers for tier in tiers)
        ):
            raise ValueError(
                f"ablation comparison {reference}->{variant} has invalid primary_tiers"
            )
        outcomes.append(variant)
        graph[reference].append(variant)
        audit_rows.append(
            {
                "reference": reference,
                "variant": variant,
                "factor": factor,
                "changed_fields": observed,
                "primary_metrics": [str(metric) for metric in metrics],
                "primary_tiers": [str(tier) for tier in tiers],
                "hypothesis": hypothesis,
            }
        )

    duplicate_outcomes = sorted(
        key for key, count in Counter(outcomes).items() if count != 1
    )
    if duplicate_outcomes:
        raise ValueError(
            f"ablation variants must have exactly one reference: {duplicate_outcomes}"
        )
    expected_outcomes = set(variants) - {treatment_key}
    if set(outcomes) != expected_outcomes:
        missing = sorted(expected_outcomes - set(outcomes))
        extra = sorted(set(outcomes) - expected_outcomes)
        raise ValueError(
            f"ablation contract does not cover every non-treatment variant; "
            f"missing={missing}, extra={extra}"
        )

    visited = {treatment_key}
    queue = deque([treatment_key])
    while queue:
        reference = queue.popleft()
        for variant in graph.get(reference, []):
            if variant in visited:
                raise ValueError("ablation comparison graph contains a cycle")
            visited.add(variant)
            queue.append(variant)
    if visited != set(variants):
        raise ValueError("ablation comparison graph is not rooted at treatment_key")

    return {
        "status": "valid",
        "version": 1,
        "treatment_key": treatment_key,
        "variant_count": len(variants),
        "comparison_count": len(audit_rows),
        "comparisons": audit_rows,
    }
