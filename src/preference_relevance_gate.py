"""Frozen-embedding relevance gate for abstaining on non-preference language."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def calibrate_false_accept_threshold(
    negative_scores: Sequence[float], max_false_accept_rate: float
) -> Dict[str, object]:
    """Choose a strict empirical threshold without using positive outcomes.

    With acceptance defined as score >= threshold, the next representable float
    above the (allowed+1)-th largest negative score guarantees no more than
    floor(alpha*n) accepted calibration negatives, including tied scores.
    """
    scores = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("negative calibration scores must be finite and non-empty")
    alpha = float(max_false_accept_rate)
    if not 0.0 <= alpha < 1.0:
        raise ValueError("max_false_accept_rate must be in [0, 1)")
    allowed = int(math.floor(alpha * scores.size))
    descending = np.sort(scores)[::-1]
    threshold = float(np.nextafter(descending[allowed], np.inf))
    accepted = int(np.sum(scores >= threshold))
    if accepted > allowed:
        raise AssertionError("strict threshold calibration exceeded its contract")
    return {
        "threshold": threshold,
        "calibration_negative_count": int(scores.size),
        "max_false_accept_rate": alpha,
        "allowed_false_accept_count": allowed,
        "observed_false_accept_count": accepted,
        "observed_false_accept_rate": float(accepted / scores.size),
        "rule": "strict empirical upper-tail threshold; ties at cutoff rejected",
    }


@dataclass(frozen=True)
class PreferenceRelevanceGate:
    coefficients: np.ndarray
    intercept: float
    threshold: float
    encoder_model: str
    encoder_revision: str
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.float64).reshape(-1)
        if coefficients.size == 0 or not np.isfinite(coefficients).all():
            raise ValueError("gate coefficients must be finite and non-empty")
        if not np.isfinite(float(self.intercept)) or not np.isfinite(float(self.threshold)):
            raise ValueError("gate intercept and threshold must be finite")
        if not self.encoder_model or not self.encoder_revision:
            raise ValueError("gate requires pinned encoder model and revision")
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def embedding_dim(self) -> int:
        return int(self.coefficients.size)

    def decision_function(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = _matrix(embeddings, "gate embeddings")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError("gate embedding dimension mismatch")
        return embeddings @ self.coefficients + float(self.intercept)

    def probabilities(self, embeddings: np.ndarray) -> np.ndarray:
        logits = np.clip(self.decision_function(embeddings), -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def accepts(self, embeddings: np.ndarray) -> np.ndarray:
        return self.probabilities(embeddings) >= float(self.threshold)

    def apply(self, embeddings: np.ndarray, profiles: np.ndarray) -> np.ndarray:
        profiles = np.asarray(profiles, dtype=np.float32)
        accepted = self.accepts(embeddings)
        if profiles.ndim != 2 or profiles.shape[0] != accepted.size:
            raise ValueError("profile rows must match relevance-gate embeddings")
        result = profiles.copy()
        result[~accepted] = 1.0
        return result

    def to_payload(self) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "gate_type": "frozen_embedding_logistic_relevance",
            "encoder_model": self.encoder_model,
            "encoder_revision": self.encoder_revision,
            "embedding_dim": self.embedding_dim,
            "coefficients": self.coefficients.tolist(),
            "intercept": float(self.intercept),
            "threshold": float(self.threshold),
            "metadata": dict(self.metadata),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_payload(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "PreferenceRelevanceGate":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("gate_type") != "frozen_embedding_logistic_relevance":
            raise ValueError("unsupported preference relevance gate type")
        if int(payload.get("embedding_dim", -1)) != len(payload.get("coefficients", [])):
            raise ValueError("serialized gate embedding dimension mismatch")
        return cls(
            coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
            intercept=float(payload["intercept"]),
            threshold=float(payload["threshold"]),
            encoder_model=str(payload["encoder_model"]),
            encoder_revision=str(payload["encoder_revision"]),
            metadata=dict(payload.get("metadata", {})),
        )


def fit_logistic_relevance_gate(
    positive_embeddings: np.ndarray,
    negative_embeddings: np.ndarray,
    *,
    negative_calibration_embeddings: np.ndarray,
    max_false_accept_rate: float,
    encoder_model: str,
    encoder_revision: str,
    c: float = 1.0,
    random_state: int = 20260820,
    metadata: Mapping[str, object] | None = None,
) -> PreferenceRelevanceGate:
    positive = _matrix(positive_embeddings, "positive embeddings")
    negative = _matrix(negative_embeddings, "negative embeddings")
    calibration = _matrix(
        negative_calibration_embeddings, "negative calibration embeddings"
    )
    if positive.shape[1] != negative.shape[1] or positive.shape[1] != calibration.shape[1]:
        raise ValueError("all relevance-gate embeddings must share a dimension")
    if c <= 0:
        raise ValueError("logistic regularization C must be positive")
    from sklearn.linear_model import LogisticRegression

    x = np.concatenate([positive, negative], axis=0)
    y = np.concatenate([
        np.ones(positive.shape[0], dtype=np.int64),
        np.zeros(negative.shape[0], dtype=np.int64),
    ])
    classifier = LogisticRegression(
        C=float(c),
        class_weight="balanced",
        max_iter=2000,
        random_state=int(random_state),
        solver="liblinear",
    )
    classifier.fit(x, y)
    provisional = PreferenceRelevanceGate(
        coefficients=classifier.coef_[0],
        intercept=float(classifier.intercept_[0]),
        threshold=0.5,
        encoder_model=encoder_model,
        encoder_revision=encoder_revision,
        metadata={},
    )
    calibration_contract = calibrate_false_accept_threshold(
        provisional.probabilities(calibration), max_false_accept_rate
    )
    resolved_metadata = dict(metadata or {})
    resolved_metadata.update({
        "training_positive_count": int(positive.shape[0]),
        "training_negative_count": int(negative.shape[0]),
        "logistic_c": float(c),
        "random_state": int(random_state),
        "calibration": calibration_contract,
    })
    return PreferenceRelevanceGate(
        coefficients=classifier.coef_[0],
        intercept=float(classifier.intercept_[0]),
        threshold=float(calibration_contract["threshold"]),
        encoder_model=encoder_model,
        encoder_revision=encoder_revision,
        metadata=resolved_metadata,
    )


def relevance_metrics(
    gate: PreferenceRelevanceGate,
    *,
    positive_embeddings: np.ndarray | None = None,
    negative_embeddings: np.ndarray | None = None,
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    scores = []
    labels = []
    if positive_embeddings is not None:
        positive_scores = gate.probabilities(positive_embeddings)
        result["positive_count"] = int(positive_scores.size)
        result["true_accept_rate"] = float(np.mean(positive_scores >= gate.threshold))
        result["positive_score_quantiles"] = {
            key: float(value) for key, value in zip(
                ("p05", "median", "p95"),
                np.quantile(positive_scores, (0.05, 0.5, 0.95)),
            )
        }
        scores.extend(positive_scores.tolist())
        labels.extend([1] * positive_scores.size)
    if negative_embeddings is not None:
        negative_scores = gate.probabilities(negative_embeddings)
        result["negative_count"] = int(negative_scores.size)
        result["false_accept_rate"] = float(np.mean(negative_scores >= gate.threshold))
        result["negative_score_quantiles"] = {
            key: float(value) for key, value in zip(
                ("p05", "median", "p95"),
                np.quantile(negative_scores, (0.05, 0.5, 0.95)),
            )
        }
        scores.extend(negative_scores.tolist())
        labels.extend([0] * negative_scores.size)
    if len(set(labels)) == 2:
        from sklearn.metrics import average_precision_score, roc_auc_score

        result["roc_auc"] = float(roc_auc_score(labels, scores))
        result["average_precision"] = float(average_precision_score(labels, scores))
    return result
