from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preference_relevance_gate import (  # noqa: E402
    PreferenceRelevanceGate,
    calibrate_false_accept_threshold,
    fit_logistic_relevance_gate,
    relevance_metrics,
)


def test_strict_calibration_respects_false_accept_count_with_ties():
    contract = calibrate_false_accept_threshold([0.9, 0.8, 0.8, 0.1], 0.25)
    assert contract["allowed_false_accept_count"] == 1
    assert contract["observed_false_accept_count"] == 1


def test_gate_fit_apply_and_roundtrip(tmp_path: Path):
    positive = np.asarray([[2.0, 0.0], [1.5, 0.2], [1.8, -0.1]])
    negative = np.asarray([[-2.0, 0.0], [-1.5, 0.1], [-1.8, -0.2]])
    calibration = np.asarray([[-1.4, 0.0], [-1.2, 0.1], [-1.0, -0.1]])
    gate = fit_logistic_relevance_gate(
        positive,
        negative,
        negative_calibration_embeddings=calibration,
        max_false_accept_rate=0.0,
        encoder_model="fake",
        encoder_revision="pinned",
    )
    assert gate.accepts(positive).all()
    assert not gate.accepts(calibration).any()
    profiles = np.full((2, 3), 1.7, dtype=np.float32)
    applied = gate.apply(np.asarray([positive[0], calibration[0]]), profiles)
    np.testing.assert_allclose(applied[0], 1.7)
    np.testing.assert_allclose(applied[1], 1.0)
    path = tmp_path / "gate.json"
    gate.save(path)
    loaded = PreferenceRelevanceGate.load(path)
    np.testing.assert_allclose(loaded.probabilities(positive), gate.probabilities(positive))


def test_metrics_report_positive_and_negative_rates():
    gate = PreferenceRelevanceGate(
        coefficients=np.asarray([1.0]),
        intercept=0.0,
        threshold=0.5,
        encoder_model="fake",
        encoder_revision="pinned",
        metadata={},
    )
    metrics = relevance_metrics(
        gate,
        positive_embeddings=np.asarray([[1.0], [2.0]]),
        negative_embeddings=np.asarray([[-1.0], [1.0]]),
    )
    assert metrics["true_accept_rate"] == 1.0
    assert metrics["false_accept_rate"] == 0.5
    assert metrics["roc_auc"] > 0.5


def test_gate_rejects_embedding_dimension_mismatch():
    gate = PreferenceRelevanceGate(
        coefficients=np.ones(2), intercept=0.0, threshold=0.5,
        encoder_model="fake", encoder_revision="pinned", metadata={},
    )
    with pytest.raises(ValueError, match="dimension mismatch"):
        gate.probabilities(np.ones((1, 3)))
