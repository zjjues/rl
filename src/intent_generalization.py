"""Leakage-resistant intent generalization protocols and diagnostics."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from intent_semantic_encoder import (
    DEFAULT_INTENT_DESCRIPTIONS,
    IntentLibrary,
    infer_intent_posture,
)
from intent_objectives import OBJECTIVE_KEYS, resolve_intent_reward_profile


VALID_SPLITS = {"seen", "paraphrase", "unseen", "counterfactual"}


def resolve_query_objective_profile(query: Mapping[str, object]) -> Dict[str, float]:
    """Resolve an evaluation target, allowing explicit minimally contrastive profiles."""
    profile = resolve_intent_reward_profile(str(query["canonical_label"]))
    overrides = query.get("objective_profile")
    if overrides is None:
        return profile
    if not isinstance(overrides, Mapping):
        raise ValueError("query objective_profile must be a mapping")
    unknown = sorted(set(overrides) - set(OBJECTIVE_KEYS))
    if unknown:
        raise ValueError(f"query objective_profile has unknown keys: {unknown}")
    for key, value in overrides.items():
        number = float(value)
        if not 0.3 <= number <= 2.2:
            raise ValueError(f"query objective_profile[{key!r}] must be in [0.3, 2.2]")
        profile[str(key)] = number
    return profile


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def validate_generalization_suite(
    suite: Mapping[str, object],
    catalog: Sequence[Tuple[str, str]] = DEFAULT_INTENT_DESCRIPTIONS,
) -> None:
    """Reject label leakage, text leakage, and ambiguous query definitions."""
    required = {"schema_version", "suite_id", "train_labels", "queries"}
    missing = sorted(required - set(suite))
    if missing:
        raise ValueError(f"generalization suite is missing required keys: {missing}")
    catalog_by_label = dict(catalog)
    train_labels = [str(label) for label in suite["train_labels"]]
    if not train_labels or len(train_labels) != len(set(train_labels)):
        raise ValueError("train_labels must be non-empty and unique")
    unknown_train = sorted(set(train_labels) - set(catalog_by_label))
    if unknown_train:
        raise ValueError(f"unknown train labels: {unknown_train}")

    query_keys: List[str] = []
    training_texts = {
        _normalise_text(catalog_by_label[label]) for label in train_labels
    }
    for query in suite["queries"]:
        key = str(query.get("key", ""))
        label = str(query.get("canonical_label", ""))
        description = str(query.get("description", ""))
        split = str(query.get("split", ""))
        if not key or not label or not description:
            raise ValueError("each generalization query needs key, canonical_label, and description")
        if split not in VALID_SPLITS:
            raise ValueError(f"query {key!r} has invalid split {split!r}")
        if label not in catalog_by_label:
            raise ValueError(f"query {key!r} uses unknown canonical label {label!r}")
        if split == "unseen" and label in train_labels:
            raise ValueError(f"unseen query {key!r} leaks training label {label!r}")
        if split in {"seen", "paraphrase"} and label not in train_labels:
            raise ValueError(f"{split} query {key!r} must reference a training label")
        if split == "counterfactual" and "objective_profile" not in query:
            raise ValueError(f"counterfactual query {key!r} needs objective_profile")
        resolve_query_objective_profile(query)
        if split in {"paraphrase", "unseen", "counterfactual"} and _normalise_text(description) in training_texts:
            raise ValueError(f"query {key!r} duplicates an exact training description")
        query_keys.append(key)
    if len(query_keys) != len(set(query_keys)):
        raise ValueError("generalization query keys must be unique")


def load_generalization_suite(
    path: str | Path,
    _seen: set[Path] | None = None,
) -> Dict[str, object]:
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f"generalization suite inheritance cycle at {path}")
    seen.add(path)
    suite = json.loads(path.read_text(encoding="utf-8"))
    if "extends" in suite:
        parent_path = (path.parent / str(suite["extends"])).resolve()
        parent = load_generalization_suite(parent_path, seen)
        merged = deepcopy(parent)
        overrides = suite.get("query_overrides", {})
        excluded = suite.get("exclude_query_keys", [])
        if not isinstance(overrides, Mapping):
            raise ValueError("query_overrides must be a mapping")
        if not isinstance(excluded, list) or len(excluded) != len(set(excluded)):
            raise ValueError("exclude_query_keys must be a unique list")
        by_key = {str(query["key"]): query for query in merged["queries"]}
        unknown = sorted(set(overrides) - set(by_key))
        if unknown:
            raise ValueError(f"query_overrides contains unknown keys: {unknown}")
        unknown_excluded = sorted(set(excluded) - set(by_key))
        if unknown_excluded:
            raise ValueError(
                f"exclude_query_keys contains unknown keys: {unknown_excluded}"
            )
        for key, values in overrides.items():
            if not isinstance(values, Mapping):
                raise ValueError(f"query override {key!r} must be a mapping")
            by_key[str(key)].update(deepcopy(dict(values)))
        merged["queries"] = [
            query for query in merged["queries"]
            if str(query["key"]) not in set(excluded)
        ]
        for key, value in suite.items():
            if key not in {
                "extends", "query_overrides", "exclude_query_keys", "queries"
            }:
                merged[key] = deepcopy(value)
        merged["parent_suite_id"] = parent["suite_id"]
        merged["query_overrides"] = deepcopy(overrides)
        merged["exclude_query_keys"] = deepcopy(excluded)
        suite = merged
    validate_generalization_suite(suite)
    return suite


def _normalise_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Return a tie-aware Spearman correlation, or None for degenerate data."""
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.size != y_array.size or x_array.size < 3:
        return None
    x_rank = _average_ranks(x_array)
    y_rank = _average_ranks(y_array)
    if np.std(x_rank) <= 1e-12 or np.std(y_rank) <= 1e-12:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def objective_profile_prediction_diagnostics(
    queries: Sequence[Mapping[str, object]],
    predicted_profiles: np.ndarray,
) -> Dict[str, object]:
    """Audit text-to-objective predictions against held-out task definitions.

    Canonical labels are used only after inference to construct evaluation targets;
    they are never passed to the profile decoder.
    """
    predicted = np.asarray(predicted_profiles, dtype=np.float64)
    expected_shape = (len(queries), len(OBJECTIVE_KEYS))
    if predicted.shape != expected_shape:
        raise ValueError(
            f"predicted objective profile shape {predicted.shape} != {expected_shape}"
        )
    targets = np.asarray([
        [resolve_query_objective_profile(query)[key] for key in OBJECTIVE_KEYS]
        for query in queries
    ], dtype=np.float64)
    records = []
    for index, query in enumerate(queries):
        target = targets[index]
        prediction = predicted[index]
        correlation = (
            float(np.corrcoef(target, prediction)[0, 1])
            if np.std(target) > 1e-8 and np.std(prediction) > 1e-8 else None
        )
        records.append({
            "key": str(query["key"]),
            "split": str(query["split"]),
            "canonical_label": str(query["canonical_label"]),
            "contrast_group": query.get("contrast_group"),
            "target": {key: float(target[i]) for i, key in enumerate(OBJECTIVE_KEYS)},
            "prediction": {
                key: float(prediction[i]) for i, key in enumerate(OBJECTIVE_KEYS)
            },
            "mean_absolute_error": float(np.mean(np.abs(prediction - target))),
            "profile_correlation": correlation,
            "off_target_mean_absolute_error": (
                float(np.mean(np.abs(
                    np.delete(prediction - target, OBJECTIVE_KEYS.index(
                        str(query["contrast_group"])
                    ))
                )))
                if query.get("contrast_group") in OBJECTIVE_KEYS else None
            ),
        })
    by_split = {}
    for split in sorted({str(query["split"]) for query in queries}):
        indices = [
            index for index, query in enumerate(queries)
            if str(query["split"]) == split
        ]
        correlations = [
            records[index]["profile_correlation"] for index in indices
            if records[index]["profile_correlation"] is not None
        ]
        by_split[split] = {
            "n_queries": len(indices),
            "mean_absolute_error": float(
                np.mean(np.abs(predicted[indices] - targets[indices]))
            ),
            "mean_profile_correlation": (
                float(np.mean(correlations)) if correlations else None
            ),
        }
    objective_alignment = {
        key: spearman_correlation(targets[:, index], predicted[:, index])
        for index, key in enumerate(OBJECTIVE_KEYS)
    }
    contrast_groups = {}
    for group in sorted({
        str(query["contrast_group"]) for query in queries
        if query.get("contrast_group") in OBJECTIVE_KEYS
    }):
        indices = [
            index for index, query in enumerate(queries)
            if str(query.get("contrast_group", "")) == group
        ]
        objective_index = OBJECTIVE_KEYS.index(group)
        off_target_indices = [
            index for index in range(len(OBJECTIVE_KEYS))
            if index != objective_index
        ]
        contrast_groups[group] = {
            "n_queries": len(indices),
            "target_spearman": spearman_correlation(
                targets[indices, objective_index],
                predicted[indices, objective_index],
            ),
            "target_predictions": [
                float(value) for value in predicted[indices, objective_index]
            ],
            "off_target_mean_absolute_error": float(np.mean(np.abs(
                predicted[np.ix_(indices, off_target_indices)]
                - targets[np.ix_(indices, off_target_indices)]
            ))),
            "max_off_target_range": float(np.max(np.ptp(
                predicted[np.ix_(indices, off_target_indices)], axis=0
            ))),
        }
    return {
        "by_split": by_split,
        "objective_spearman": objective_alignment,
        "overall_mean_absolute_error": float(np.mean(np.abs(predicted - targets))),
        "contrast_groups": contrast_groups,
        "queries": records,
    }


def intent_behavior_controllability(
    queries: Sequence[Mapping[str, object]],
    behavior: Mapping[str, object],
) -> Dict[str, object]:
    """Measure whether text objectives move the safety-efficiency operating point.

    Query-level correlations are diagnostics within one trained seed. Paper-level
    uncertainty must aggregate these per-seed correlations, not treat queries as
    independent samples.
    """
    query_by_key = {str(query["key"]): query for query in queries}
    if set(query_by_key) != set(behavior):
        raise ValueError("behavior keys must exactly match generalization queries")
    tier_names = list(next(iter(behavior.values()))["risk_tiers"])
    scopes = {"all": list(query_by_key)}
    for split in sorted({str(query["split"]) for query in queries}):
        scopes[split] = [
            str(query["key"]) for query in queries if str(query["split"]) == split
        ]
    for group in sorted({
        str(query["contrast_group"]) for query in queries if query.get("contrast_group")
    }):
        scopes[f"contrast:{group}"] = [
            str(query["key"]) for query in queries
            if str(query.get("contrast_group", "")) == group
        ]

    result: Dict[str, object] = {}
    for tier in tier_names:
        tier_result = {}
        for scope, keys in scopes.items():
            if len(keys) < 3:
                continue
            profiles = [
                resolve_query_objective_profile(query_by_key[key])
                for key in keys
            ]
            task_preference = [profile["task"] for profile in profiles]
            safety_tradeoff_preference = [
                profile["safety"]
                - 0.5 * (profile["task"] + profile["time"])
                for profile in profiles
            ]
            collisions = [
                float(behavior[key]["risk_tiers"][tier]["collision_rate"])
                for key in keys
            ]
            completion = [
                float(behavior[key]["risk_tiers"][tier]["task_completion"])
                for key in keys
            ]
            collision_array = np.asarray(collisions, dtype=np.float64)
            completion_array = np.asarray(completion, dtype=np.float64)
            safe_z = -(collision_array - collision_array.mean()) / max(
                float(collision_array.std()), 1e-8
            )
            completion_z = (completion_array - completion_array.mean()) / max(
                float(completion_array.std()), 1e-8
            )
            observed_safety_tradeoff = safe_z - completion_z
            tier_result[scope] = {
                "n_queries": len(keys),
                "safety_tradeoff_spearman": spearman_correlation(
                    safety_tradeoff_preference, observed_safety_tradeoff
                ),
                "task_preference_spearman": spearman_correlation(
                    task_preference, completion_array
                ),
                "collision_rate_range": float(np.ptp(collision_array)),
                "task_completion_range": float(np.ptp(completion_array)),
            }
            optional_alignment = {
                "energy_preference_spearman": (
                    "energy", "energy_remaining", False
                ),
                "distance_preference_spearman": (
                    "distance", "distance_to_target", True
                ),
                "time_preference_spearman": ("time", "speed", False),
                "safety_distance_spearman": (
                    "safety", "min_neighbor_distance", False
                ),
                "threat_preference_spearman": (
                    "threat", "distance_to_threat", False
                ),
            }
            for metric_name, (profile_key, behavior_key, negate) in optional_alignment.items():
                if all(behavior_key in behavior[key]["risk_tiers"][tier] for key in keys):
                    observed = np.asarray(
                        [
                            float(behavior[key]["risk_tiers"][tier][behavior_key])
                            for key in keys
                        ],
                        dtype=np.float64,
                    )
                    if negate:
                        observed = -observed
                    tier_result[scope][metric_name] = spearman_correlation(
                        [profile[profile_key] for profile in profiles], observed
                    )
        result[str(tier)] = tier_result
    return result


def representation_retrieval_diagnostics(
    train_library: IntentLibrary,
    query_vectors: np.ndarray,
    queries: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Measure whether query vectors retrieve the intended training intent.

    Query texts are never treated as independent statistical seeds. These metrics
    diagnose the representation; behavioral result aggregation must average queries
    within each seed before computing cross-seed uncertainty.
    """
    query_vectors = np.asarray(query_vectors, dtype=np.float32)
    if query_vectors.shape != (len(queries), train_library.intent_dim):
        raise ValueError("query vector shape does not match query count/train dimension")
    train_vectors = _normalise_rows(train_library.vectors)
    query_vectors = _normalise_rows(query_vectors)
    similarities = query_vectors @ train_vectors.T
    train_profiles = np.asarray([
        [resolve_intent_reward_profile(label)[key] for key in OBJECTIVE_KEYS]
        for label in train_library.labels
    ], dtype=np.float32)
    records: List[Dict[str, object]] = []

    for idx, query in enumerate(queries):
        label = str(query["canonical_label"])
        split = str(query["split"])
        posture = str(query.get("posture") or infer_intent_posture(label, str(query["description"])))
        nearest_idx = int(np.argmax(similarities[idx]))
        nearest_label = train_library.labels[nearest_idx]
        nearest_posture = train_library.posture_for_index(nearest_idx)
        query_profile = np.asarray([
            resolve_query_objective_profile(query)[key] for key in OBJECTIVE_KEYS
        ], dtype=np.float32)
        objective_distances = np.linalg.norm(train_profiles - query_profile[None, :], axis=1)
        objective_similarities = np.exp(-objective_distances)
        best_objective_similarity = float(np.max(objective_similarities))
        selected_objective_similarity = float(objective_similarities[nearest_idx])
        best_objective_distance = float(np.min(objective_distances))
        selected_objective_distance = float(objective_distances[nearest_idx])
        objective_profile_regret = selected_objective_distance - best_objective_distance
        if np.std(similarities[idx]) > 1e-8 and np.std(objective_distances) > 1e-8:
            objective_alignment_correlation = float(
                np.corrcoef(similarities[idx], -objective_distances)[0, 1]
            )
        else:
            objective_alignment_correlation = None
        correct_idx = train_library.labels.index(label) if label in train_library.labels else None
        matched_cosine = None
        best_incorrect_cosine = None
        margin = None
        retrieval_correct = None
        if correct_idx is not None:
            matched_cosine = float(similarities[idx, correct_idx])
            incorrect = np.delete(similarities[idx], correct_idx)
            best_incorrect_cosine = float(np.max(incorrect)) if incorrect.size else -1.0
            margin = matched_cosine - best_incorrect_cosine
            retrieval_correct = bool(nearest_idx == correct_idx)
        records.append(
            {
                "key": str(query["key"]),
                "split": split,
                "canonical_label": label,
                "posture": posture,
                "nearest_train_label": nearest_label,
                "nearest_train_posture": nearest_posture,
                "nearest_cosine": float(similarities[idx, nearest_idx]),
                "matched_cosine": matched_cosine,
                "best_incorrect_cosine": best_incorrect_cosine,
                "semantic_margin": margin,
                "retrieval_correct": retrieval_correct,
                "posture_retrieval_correct": bool(nearest_posture in {posture, "neutral"}),
                "selected_objective_similarity": selected_objective_similarity,
                "best_objective_similarity": best_objective_similarity,
                "objective_profile_regret": objective_profile_regret,
                "objective_retrieval_correct": bool(objective_profile_regret <= 0.25),
                "objective_alignment_correlation": objective_alignment_correlation,
            }
        )

    by_split: Dict[str, object] = {}
    for split in sorted({str(query["split"]) for query in queries}):
        selected = [record for record in records if record["split"] == split]
        retrievable = [record for record in selected if record["retrieval_correct"] is not None]
        margins = [float(record["semantic_margin"]) for record in selected if record["semantic_margin"] is not None]
        matched = [float(record["matched_cosine"]) for record in selected if record["matched_cosine"] is not None]
        correlations = [
            float(record["objective_alignment_correlation"])
            for record in selected
            if record["objective_alignment_correlation"] is not None
        ]
        by_split[split] = {
            "n_queries": len(selected),
            "n_retrievable": len(retrievable),
            "top1_retrieval_accuracy": (
                float(np.mean([record["retrieval_correct"] for record in retrievable]))
                if retrievable else None
            ),
            "posture_retrieval_accuracy": float(
                np.mean([record["posture_retrieval_correct"] for record in selected])
            ),
            "mean_matched_cosine": float(np.mean(matched)) if matched else None,
            "mean_semantic_margin": float(np.mean(margins)) if margins else None,
            "objective_retrieval_accuracy": float(
                np.mean([record["objective_retrieval_correct"] for record in selected])
            ),
            "mean_selected_objective_similarity": float(
                np.mean([record["selected_objective_similarity"] for record in selected])
            ),
            "mean_objective_profile_regret": float(
                np.mean([record["objective_profile_regret"] for record in selected])
            ),
            "mean_objective_alignment_correlation": (
                float(np.mean(correlations)) if correlations else None
            ),
        }
    return {"by_split": by_split, "queries": records}
