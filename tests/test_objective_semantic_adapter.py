from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from objective_semantic_adapter import (  # noqa: E402
    ObjectiveSemanticAdapter,
    _cached_frozen_model,
    clear_frozen_model_cache,
    fit_dual_ridge,
    frozen_model_cache_info,
)
from intent_objectives import OBJECTIVE_KEYS  # noqa: E402


class ObjectiveSemanticAdapterTests(unittest.TestCase):
    def tearDown(self):
        clear_frozen_model_cache()

    def test_frozen_model_cache_reuses_exact_revision_and_device(self):
        class Parameter:
            def __init__(self):
                self.requires_grad = True

            def requires_grad_(self, value):
                self.requires_grad = value

        class FakeModel:
            def __init__(self):
                self.training = True
                self.parameter = Parameter()

            def eval(self):
                self.training = False

            def parameters(self):
                return [self.parameter]

        calls = []

        def factory():
            calls.append(1)
            return FakeModel()

        first = _cached_frozen_model("encoder", "model", "rev", "cuda", factory)
        second = _cached_frozen_model("encoder", "model", "rev", "cuda", factory)
        third = _cached_frozen_model("encoder", "model", "rev2", "cuda", factory)
        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(len(calls), 2)
        self.assertFalse(first.training)
        self.assertFalse(first.parameter.requires_grad)
        self.assertEqual(frozen_model_cache_info()["entry_count"], 2)

    def test_dual_ridge_recovers_training_targets(self):
        rng = np.random.default_rng(7)
        embeddings = rng.normal(size=(8, 12)).astype(np.float32)
        true_map = rng.normal(size=(12, 3)).astype(np.float32)
        targets = embeddings @ true_map
        fitted = fit_dual_ridge(embeddings, targets, ridge=1e-6)
        predictions = embeddings @ fitted
        np.testing.assert_allclose(predictions, targets, atol=1e-4, rtol=1e-4)

    def test_dual_ridge_rejects_nonpositive_regularization(self):
        with self.assertRaises(ValueError):
            fit_dual_ridge(np.eye(2), np.eye(2), ridge=0.0)

    def test_zero_semantic_weight_preserves_configured_dimension(self):
        class FakeModel:
            def encode(self, descriptions, **kwargs):
                values = np.arange(len(descriptions) * 4, dtype=np.float32).reshape(len(descriptions), 4) + 1
                return values / np.linalg.norm(values, axis=1, keepdims=True)

        adapter = ObjectiveSemanticAdapter(
            FakeModel(),
            coefficients=np.ones((4, 7), dtype=np.float32),
            target_mean=np.zeros(7, dtype=np.float32),
            target_scale=np.ones(7, dtype=np.float32),
            intent_dim=16,
            projection_seed=17,
            semantic_weight=0.0,
            objective_weight=1.0,
            model_name="fake",
            model_revision="test",
            ridge=0.01,
        )
        vectors = adapter.encode_entries([("a", "first"), ("b", "second")])
        self.assertEqual(vectors.shape, (2, 16))
        np.testing.assert_allclose(vectors[:, :9], 0.0)

    def test_concept_anchor_decoder_is_monotonic_and_bounded(self):
        class FakeModel:
            def encode(self, descriptions, **kwargs):
                del kwargs
                values = []
                for description in descriptions:
                    text = str(description).lower()
                    values.append([
                        float("conserve" in text or "energy" in text),
                        float("unimportant" in text or "maximum acceleration" in text),
                    ])
                values = np.asarray(values, dtype=np.float32)
                norms = np.linalg.norm(values, axis=1, keepdims=True)
                return values / np.maximum(norms, 1.0)

        adapter = ObjectiveSemanticAdapter(
            FakeModel(),
            coefficients=np.zeros((2, 7), dtype=np.float32),
            target_mean=np.ones(7, dtype=np.float32),
            target_scale=np.ones(7, dtype=np.float32),
            intent_dim=16,
            projection_seed=17,
            semantic_weight=0.0,
            objective_weight=1.0,
            model_name="fake",
            model_revision="test",
            ridge=0.01,
            profile_decoder="concept_anchor",
            anchor_directions=np.tile(np.asarray([[1.0, -1.0]], dtype=np.float32), (7, 1)),
            anchor_slopes=np.ones(7, dtype=np.float32),
            anchor_intercepts=np.ones(7, dtype=np.float32),
        )
        profiles = adapter.predict_profiles([("high", "conserve energy"), ("low", "energy unimportant")])
        self.assertTrue(np.all(profiles[0] > profiles[1]))
        self.assertTrue(np.all(profiles >= 0.3))
        self.assertTrue(np.all(profiles <= 2.2))

    def test_contrastive_anchor_maps_polar_endpoints_to_declared_range(self):
        class FakeModel:
            def encode(self, descriptions, **kwargs):
                del kwargs
                return np.asarray([
                    [1.0, 0.0] if "high" in str(text) else [-1.0, 0.0]
                    for text in descriptions
                ], dtype=np.float32)

        adapter = ObjectiveSemanticAdapter(
            FakeModel(),
            coefficients=np.zeros((2, 7), dtype=np.float32),
            target_mean=np.ones(7, dtype=np.float32),
            target_scale=np.ones(7, dtype=np.float32),
            intent_dim=16,
            projection_seed=17,
            semantic_weight=0.0,
            objective_weight=1.0,
            model_name="fake",
            model_revision="test",
            ridge=0.01,
            profile_decoder="contrastive_anchor",
            anchor_directions=np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (7, 1)),
            anchor_midpoints=np.zeros(7, dtype=np.float32),
            anchor_half_ranges=np.ones(7, dtype=np.float32),
        )
        profiles = adapter.predict_profiles([("high", "high"), ("low", "low")])
        np.testing.assert_allclose(profiles[0], 1.7, atol=1e-6)
        np.testing.assert_allclose(profiles[1], 0.5, atol=1e-6)

    def test_nli_decoder_uses_entailment_difference_and_caches_text(self):
        class Config:
            id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

        class Model:
            config = Config()

        class FakeCrossEncoder:
            model = Model()

            def __init__(self):
                self.calls = 0

            def predict(self, pairs, **kwargs):
                del kwargs
                self.calls += 1
                logits = []
                for premise, hypothesis in pairs:
                    energy = "battery charge" in hypothesis
                    high = "dominant concern" in hypothesis
                    if "conserve battery" in premise and energy:
                        logits.append([0.0, 5.0, 0.0] if high else [5.0, 0.0, 0.0])
                    else:
                        logits.append([0.0, 0.0, 5.0])
                return np.asarray(logits, dtype=np.float32)

        nli = FakeCrossEncoder()
        adapter = ObjectiveSemanticAdapter(
            model=None,
            coefficients=np.zeros((2, 7), dtype=np.float32),
            target_mean=np.ones(7, dtype=np.float32),
            target_scale=np.ones(7, dtype=np.float32),
            intent_dim=16,
            projection_seed=17,
            semantic_weight=0.0,
            objective_weight=1.0,
            model_name="fake",
            model_revision="test",
            ridge=0.01,
            profile_decoder="nli_entailment",
            nli_model=nli,
            nli_model_name="fake-nli",
            nli_model_revision="test",
        )
        entries = [("energy", "conserve battery")]
        first = adapter.predict_profiles(entries)
        second = adapter.predict_profiles(entries)
        self.assertGreater(first[0, 1], 1.6)
        np.testing.assert_allclose(first, second)
        self.assertEqual(nli.calls, 1)

    def test_nli_prototype_gate_changes_only_nearest_objective_class(self):
        keywords = {
            "distance": ("distance", "route", "target"),
            "energy": ("battery", "energy", "power", "charge"),
            "collision": ("collision", "contact", "encounter"),
            "safety": ("safety", "separation", "spacing", "proximity"),
            "task": ("task", "mission objective", "assigned"),
            "time": ("time", "rapid", "delay", "quick"),
            "threat": ("threat", "radar", "monitored"),
        }

        class FakeEmbeddingModel:
            def encode(self, texts, **kwargs):
                del kwargs
                rows = []
                for text in texts:
                    lower = text.lower()
                    row = np.zeros(15, dtype=np.float32)
                    matched = False
                    for index, key in enumerate(OBJECTIVE_KEYS):
                        if any(token in lower for token in keywords[key]):
                            low_polarity = any(token in lower for token in (
                                "unimportant", "low importance", "secondary",
                                "may sacrifice", "acceptable",
                            ))
                            row[2 * index + (0 if low_polarity else 1)] = 1.0
                            matched = True
                            break
                    if not matched:
                        row[14] = 1.0
                    rows.append(row)
                return np.asarray(rows)

        class Config:
            id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

        class Model:
            config = Config()

        class FakeCrossEncoder:
            model = Model()

            def predict(self, pairs, **kwargs):
                del pairs, kwargs
                return np.tile(
                    np.asarray([[0.0, 5.0, 0.0]], dtype=np.float32),
                    (7, 1),
                )

        adapter = ObjectiveSemanticAdapter(
            model=FakeEmbeddingModel(),
            coefficients=np.zeros((15, 7), dtype=np.float32),
            target_mean=np.ones(7, dtype=np.float32),
            target_scale=np.ones(7, dtype=np.float32),
            intent_dim=16,
            projection_seed=17,
            semantic_weight=0.0,
            objective_weight=1.0,
            model_name="fake",
            model_revision="test",
            ridge=0.01,
            profile_decoder="nli_prototype_gated",
            nli_model=FakeCrossEncoder(),
            nli_model_name="fake-nli",
            nli_model_revision="test",
        )
        profile = adapter.predict_profiles([
            ("energy", "Preserve battery charge and power reserves.")
        ])[0]
        self.assertGreater(profile[OBJECTIVE_KEYS.index("energy")], 1.6)
        unchanged = np.delete(profile, OBJECTIVE_KEYS.index("energy"))
        np.testing.assert_allclose(unchanged, 1.0, atol=1e-6)

        polarity_adapter = ObjectiveSemanticAdapter(
            model=FakeEmbeddingModel(),
            coefficients=np.zeros((15, 7), dtype=np.float32),
            target_mean=np.ones(7, dtype=np.float32),
            target_scale=np.ones(7, dtype=np.float32),
            intent_dim=16,
            projection_seed=17,
            semantic_weight=0.0,
            objective_weight=1.0,
            model_name="fake",
            model_revision="test",
            ridge=0.01,
            profile_decoder="polarity_prototype",
        )
        polarity_profile = polarity_adapter.predict_profiles([
            ("energy", "Preserve battery charge and power reserves.")
        ])[0]
        self.assertAlmostEqual(
            float(polarity_profile[OBJECTIVE_KEYS.index("energy")]), 1.7, places=6
        )
        np.testing.assert_allclose(
            np.delete(polarity_profile, OBJECTIVE_KEYS.index("energy")),
            1.0,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
