from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intent_semantic_encoder import (  # noqa: E402
    IntentLibrary,
    _project_embeddings,
    infer_intent_posture,
)


class IntentRepresentationTests(unittest.TestCase):
    def test_legacy_hash_is_deterministic_and_not_semantic(self):
        first = IntentLibrary.create_legacy_hash(intent_dim=16)
        second = IntentLibrary.create_legacy_hash(intent_dim=16)
        np.testing.assert_allclose(first.vectors, second.vectors)
        self.assertEqual(first.metadata["representation_type"], "legacy_hash")
        self.assertFalse(first.metadata["semantic_geometry"])

    def test_random_dense_is_seeded_control(self):
        first = IntentLibrary.create_random_dense(intent_dim=16, seed=1)
        repeat = IntentLibrary.create_random_dense(intent_dim=16, seed=1)
        other = IntentLibrary.create_random_dense(intent_dim=16, seed=2)
        np.testing.assert_allclose(first.vectors, repeat.vectors)
        self.assertFalse(np.allclose(first.vectors, other.vectors))
        np.testing.assert_allclose(np.linalg.norm(first.vectors, axis=1), 1.0, atol=1e-6)

    def test_onehot_is_true_catalog_identity_and_rejects_small_dimension(self):
        library = IntentLibrary.create_onehot()
        self.assertEqual(library.intent_dim, len(library))
        np.testing.assert_allclose(library.vectors @ library.vectors.T, np.eye(len(library)))
        self.assertEqual(library.metadata["representation_type"], "onehot")
        with self.assertRaises(ValueError):
            IntentLibrary.create_onehot(intent_dim=len(library) - 1)

    def test_label_subset_preserves_parent_identity_coordinates(self):
        full = IntentLibrary.create_onehot(intent_dim=32)
        subset = full.subset_by_labels(["balanced", "safety_first"])
        self.assertEqual(subset.intent_dim, 32)
        self.assertEqual(subset.labels, ["balanced", "safety_first"])
        np.testing.assert_allclose(subset.get_by_label("balanced"), full.get_by_label("balanced"))
        self.assertEqual(subset.metadata["parent_library_size"], len(full))

    def test_posture_filtered_sampling_never_contradicts_requested_posture(self):
        library = IntentLibrary.create_legacy_hash(intent_dim=8)
        rng = np.random.default_rng(7)
        _, labels, _ = library.sample_with_info(200, rng=rng, posture="stealth")
        observed = {library.posture_for_label(label) for label in labels}
        self.assertTrue(observed <= {"stealth", "neutral"})
        self.assertNotIn("attack", observed)

    def test_split_preserves_aligned_posture_metadata(self):
        library = IntentLibrary.create_legacy_hash(intent_dim=8)
        train, test = library.split(train_frac=0.6, seed=42)
        self.assertEqual(len(train.metadata["postures"]), len(train))
        self.assertEqual(len(test.metadata["postures"]), len(test))
        for idx, label in enumerate(train.labels):
            self.assertEqual(train.posture_for_index(idx), infer_intent_posture(label, train.descriptions[idx]))

    def test_random_projection_has_requested_shape_and_unit_norm(self):
        rng = np.random.default_rng(3)
        source = rng.normal(size=(12, 64)).astype(np.float32)
        projected = _project_embeddings(source, target_dim=16, seed=9)
        self.assertEqual(projected.shape, (12, 16))
        np.testing.assert_allclose(np.linalg.norm(projected, axis=1), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
