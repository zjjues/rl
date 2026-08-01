"""Training-intent-only adapter that grounds text embeddings in control objectives."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from intent_objectives import OBJECTIVE_KEYS, resolve_intent_reward_profile
from intent_semantic_encoder import _normalise_rows, _project_embeddings


OBJECTIVE_TEXT_ANCHORS: Dict[str, Tuple[str, str]] = {
    "distance": (
        "Minimize remaining distance and take a direct route toward the mission target.",
        "Remaining distance to the target is unimportant and indirect routes are acceptable.",
    ),
    "energy": (
        "Conserve battery energy by limiting acceleration, speed, and unnecessary movement.",
        "Energy consumption is unimportant and maximum acceleration is acceptable.",
    ),
    "collision": (
        "Avoid every collision and reject trajectories with collision risk.",
        "Collision risk and close encounters are acceptable when pursuing the mission.",
    ),
    "safety": (
        "Prioritize safety and maintain generous separation between all aircraft.",
        "Safety margins and separation are unimportant; close proximity is acceptable.",
    ),
    "task": (
        "Maximize successful task completion and finish all assigned mission objectives.",
        "Task completion is unimportant and assigned mission objectives may remain unfinished.",
    ),
    "time": (
        "Minimize response time and reach the objective as quickly as possible.",
        "Completion time is unimportant and a much slower mission is acceptable.",
    ),
    "threat": (
        "Avoid threat zones, radar coverage, and monitored dangerous regions.",
        "Entering threat zones and radar coverage is acceptable.",
    ),
}

OBJECTIVE_CONCEPT_TERMS: Dict[str, str] = {
    "distance": "direct progress and a small remaining distance to the assigned target",
    "energy": "battery conservation and low power consumption",
    "collision": "collision avoidance and rejection of collision-risk trajectories",
    "safety": "large aircraft separation and generous safety margins",
    "task": "successful completion of every assigned mission task",
    "time": "short response time and rapid arrival at the objective",
    "threat": "avoidance of threat zones, radar coverage, and monitored regions",
}

# NLI models are highly sensitive to whether an imperative is compared against
# another meta-level statement ("assigns high priority") or a direct declarative
# consequence. Objective-specific declarative hypotheses keep the inference task
# close to standard premise--hypothesis NLI and avoid lexical ambiguity such as
# interpreting "large aircraft separation" as physically large aircraft.
OBJECTIVE_NLI_HYPOTHESES: Dict[str, str] = {
    "distance": "Closing the remaining range to the assigned objective is the dominant concern.",
    "energy": "Preservation of battery charge is the dominant concern.",
    "collision": "Prevention of aircraft contact and risky close encounters is the dominant concern.",
    "safety": "Wide inter-vehicle spacing and generous safety margins are the dominant concern.",
    "task": "Finishing every assigned mission job is the dominant concern.",
    "time": "Responding with minimum delay is the dominant concern.",
    "threat": "Remaining outside radar and threat regions is the dominant concern.",
}


def augmented_objective_prototypes() -> List[Tuple[str, str, float]]:
    """Return independently templated concept supervision, not evaluation queries."""
    entries: List[Tuple[str, str, float]] = []
    for key in OBJECTIVE_KEYS:
        term = OBJECTIVE_CONCEPT_TERMS[key]
        positive, negative = OBJECTIVE_TEXT_ANCHORS[key]
        entries.extend([
            (key, positive, 1.7),
            (key, f"Assign high importance to {term}.", 1.7),
            (key, f"Make {term} a primary mission criterion.", 1.7),
            (key, f"The controller should emphasize {term} above the balanced default.", 1.7),
            (key, negative, 0.5),
            (key, f"Assign low importance to {term}.", 0.5),
            (key, f"Make {term} a secondary mission criterion.", 0.5),
            (key, f"The controller may sacrifice {term} relative to the balanced default.", 0.5),
        ])
    return entries


NEUTRAL_OBJECTIVE_PROTOTYPES = (
    "Keep all mission criteria at their balanced default importance.",
    "Do not raise or lower any objective weight from the default control profile.",
    "Follow a balanced flight policy without prioritizing a specific criterion.",
    "Use equal default importance for safety, mission progress, time, and resources.",
)


def fit_dual_ridge(
    embeddings: np.ndarray,
    targets: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Fit a linear map with the dual ridge solution, efficient for few intents."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if embeddings.ndim != 2 or targets.ndim != 2 or len(embeddings) != len(targets):
        raise ValueError("embeddings and targets must be aligned 2D matrices")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    kernel = embeddings @ embeddings.T
    dual = np.linalg.solve(kernel + float(ridge) * np.eye(len(embeddings)), targets)
    return (embeddings.T @ dual).astype(np.float32)


class ObjectiveSemanticAdapter:
    """Frozen text encoder plus a ridge map fitted only on training intents."""

    def __init__(
        self,
        model,
        coefficients: np.ndarray,
        target_mean: np.ndarray,
        target_scale: np.ndarray,
        *,
        intent_dim: int,
        projection_seed: int,
        semantic_weight: float,
        objective_weight: float,
        model_name: str,
        model_revision: str,
        ridge: float,
        profile_decoder: str = "dual_ridge",
        anchor_directions: Optional[np.ndarray] = None,
        anchor_slopes: Optional[np.ndarray] = None,
        anchor_intercepts: Optional[np.ndarray] = None,
        anchor_midpoints: Optional[np.ndarray] = None,
        anchor_half_ranges: Optional[np.ndarray] = None,
        nli_model=None,
        nli_model_name: str = "",
        nli_model_revision: str = "",
        nli_batch_size: int = 32,
    ) -> None:
        self.model = model
        self.coefficients = np.asarray(coefficients, dtype=np.float32)
        self.target_mean = np.asarray(target_mean, dtype=np.float32)
        self.target_scale = np.asarray(target_scale, dtype=np.float32)
        self.intent_dim = int(intent_dim)
        self.projection_seed = int(projection_seed)
        self.semantic_weight = float(semantic_weight)
        self.objective_weight = float(objective_weight)
        self.model_name = str(model_name)
        self.model_revision = str(model_revision)
        self.ridge = float(ridge)
        self.profile_decoder = str(profile_decoder)
        self.anchor_directions = (
            None if anchor_directions is None
            else np.asarray(anchor_directions, dtype=np.float32)
        )
        self.anchor_slopes = (
            None if anchor_slopes is None else np.asarray(anchor_slopes, dtype=np.float32)
        )
        self.anchor_intercepts = (
            None if anchor_intercepts is None
            else np.asarray(anchor_intercepts, dtype=np.float32)
        )
        self.anchor_midpoints = (
            None if anchor_midpoints is None
            else np.asarray(anchor_midpoints, dtype=np.float32)
        )
        self.anchor_half_ranges = (
            None if anchor_half_ranges is None
            else np.asarray(anchor_half_ranges, dtype=np.float32)
        )
        self.nli_model = nli_model
        self.nli_model_name = str(nli_model_name)
        self.nli_model_revision = str(nli_model_revision)
        self.nli_batch_size = int(nli_batch_size)
        self._nli_profile_cache: Dict[str, np.ndarray] = {}
        self._nli_concept_embeddings: Optional[np.ndarray] = None
        self._nli_neutral_similarities: Optional[np.ndarray] = None
        self._nli_relevance_centroids: Optional[np.ndarray] = None
        self._polarity_prototype_centroids: Optional[np.ndarray] = None
        if self.intent_dim < len(OBJECTIVE_KEYS):
            raise ValueError(
                f"objective-grounded intent_dim must be at least {len(OBJECTIVE_KEYS)}"
            )
        if self.semantic_weight < 0 or self.objective_weight <= 0:
            raise ValueError("semantic_weight must be non-negative and objective_weight positive")
        if self.profile_decoder not in {
            "dual_ridge", "concept_anchor", "contrastive_anchor", "prototype_ridge",
            "augmented_prototype_ridge",
            "nli_entailment",
            "nli_relevance_gated",
            "nli_similarity_gated",
            "nli_prototype_gated",
            "polarity_prototype",
        }:
            raise ValueError(f"unsupported profile decoder: {self.profile_decoder}")
        if self.profile_decoder == "concept_anchor" and any(
            value is None
            for value in (
                self.anchor_directions,
                self.anchor_slopes,
                self.anchor_intercepts,
            )
        ):
            raise ValueError("concept_anchor decoder requires fitted anchor parameters")
        if self.profile_decoder == "contrastive_anchor" and any(
            value is None
            for value in (
                self.anchor_directions,
                self.anchor_midpoints,
                self.anchor_half_ranges,
            )
        ):
            raise ValueError("contrastive_anchor decoder requires fitted anchor parameters")
        if self.profile_decoder in {
            "nli_entailment", "nli_relevance_gated", "nli_similarity_gated",
            "nli_prototype_gated",
        } and self.nli_model is None:
            raise ValueError("nli_entailment decoder requires a frozen NLI cross-encoder")

    @classmethod
    def fit(
        cls,
        entries: Sequence[Tuple[str, str]],
        *,
        intent_dim: int,
        model_name: str,
        model_revision: Optional[str],
        projection_seed: int,
        ridge: float = 0.01,
        semantic_weight: float = 1.0,
        objective_weight: float = 1.0,
        batch_size: int = 32,
        device: Optional[str] = None,
        profile_decoder: str = "dual_ridge",
        nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
        nli_model_revision: str = "",
        nli_batch_size: int = 32,
    ) -> "ObjectiveSemanticAdapter":
        if len(entries) < 2:
            raise ValueError("objective adapter requires at least two training intents")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("objective-grounded semantics requires sentence-transformers") from exc
        model_kwargs = {"device": device}
        if model_revision:
            model_kwargs["revision"] = model_revision
        model = SentenceTransformer(model_name, **model_kwargs)
        descriptions = [description for _, description in entries]
        embeddings = model.encode(
            descriptions,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        targets = np.asarray([
            [resolve_intent_reward_profile(label)[key] for key in OBJECTIVE_KEYS]
            for label, _ in entries
        ], dtype=np.float32)
        target_mean = targets.mean(axis=0)
        target_scale = targets.std(axis=0)
        target_scale = np.where(target_scale < 1e-6, 1.0, target_scale)
        standardised_targets = (targets - target_mean) / target_scale
        coefficients = fit_dual_ridge(embeddings, standardised_targets, ridge)

        nli_model = None
        if profile_decoder in {
            "nli_entailment", "nli_relevance_gated", "nli_similarity_gated",
            "nli_prototype_gated",
        }:
            if not nli_model_revision:
                raise ValueError("nli_entailment requires a pinned nli_model_revision")
            from sentence_transformers import CrossEncoder

            nli_model = CrossEncoder(
                nli_model_name,
                revision=nli_model_revision,
                device=device,
            )
            label_names = {
                str(value).lower() for value in nli_model.model.config.id2label.values()
            }
            if not {"contradiction", "entailment", "neutral"}.issubset(label_names):
                raise ValueError("NLI model must expose contradiction/entailment/neutral labels")

        anchor_directions = None
        anchor_slopes = None
        anchor_intercepts = None
        anchor_midpoints = None
        anchor_half_ranges = None
        if profile_decoder in {
            "concept_anchor", "contrastive_anchor", "prototype_ridge",
            "augmented_prototype_ridge",
        }:
            anchor_texts = []
            for key in OBJECTIVE_KEYS:
                anchor_texts.extend(OBJECTIVE_TEXT_ANCHORS[key])
            anchor_embeddings = model.encode(
                anchor_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
            positive = anchor_embeddings[0::2]
            negative = anchor_embeddings[1::2]
            anchor_directions = positive - negative
            anchor_scores = embeddings @ anchor_directions.T
            positive_scores = np.diag(positive @ anchor_directions.T)
            negative_scores = np.diag(negative @ anchor_directions.T)
            anchor_midpoints = 0.5 * (positive_scores + negative_scores)
            anchor_half_ranges = np.maximum(
                0.5 * (positive_scores - negative_scores), 1e-6
            )
            anchor_slopes = np.zeros(len(OBJECTIVE_KEYS), dtype=np.float32)
            anchor_intercepts = np.zeros(len(OBJECTIVE_KEYS), dtype=np.float32)
            for index in range(len(OBJECTIVE_KEYS)):
                score = anchor_scores[:, index].astype(np.float64)
                target = targets[:, index].astype(np.float64)
                centered_score = score - score.mean()
                centered_target = target - target.mean()
                denominator = float(centered_score @ centered_score + ridge)
                slope = max(0.0, float(centered_score @ centered_target) / denominator)
                anchor_slopes[index] = slope
                anchor_intercepts[index] = float(target.mean() - slope * score.mean())
            if profile_decoder == "prototype_ridge":
                prototype_targets = np.ones(
                    (2 * len(OBJECTIVE_KEYS), len(OBJECTIVE_KEYS)),
                    dtype=np.float32,
                )
                for index in range(len(OBJECTIVE_KEYS)):
                    prototype_targets[2 * index, index] = 1.7
                    prototype_targets[2 * index + 1, index] = 0.5
                fitted_embeddings = np.concatenate(
                    [embeddings, anchor_embeddings], axis=0
                )
                fitted_targets = np.concatenate([targets, prototype_targets], axis=0)
                target_mean = fitted_targets.mean(axis=0)
                target_scale = fitted_targets.std(axis=0)
                target_scale = np.where(target_scale < 1e-6, 1.0, target_scale)
                standardised_targets = (
                    fitted_targets - target_mean
                ) / target_scale
                coefficients = fit_dual_ridge(
                    fitted_embeddings, standardised_targets, ridge
                )
            elif profile_decoder == "augmented_prototype_ridge":
                augmented = augmented_objective_prototypes()
                prototype_texts = [text for _, text, _ in augmented]
                prototype_targets = np.ones(
                    (len(augmented), len(OBJECTIVE_KEYS)), dtype=np.float32
                )
                for row, (key, _, value) in enumerate(augmented):
                    prototype_targets[row, OBJECTIVE_KEYS.index(key)] = value
                neutral_targets = np.ones(
                    (len(NEUTRAL_OBJECTIVE_PROTOTYPES), len(OBJECTIVE_KEYS)),
                    dtype=np.float32,
                )
                prototype_embeddings = model.encode(
                    prototype_texts + list(NEUTRAL_OBJECTIVE_PROTOTYPES),
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).astype(np.float32)
                fitted_embeddings = np.concatenate(
                    [embeddings, prototype_embeddings], axis=0
                )
                fitted_targets = np.concatenate(
                    [targets, prototype_targets, neutral_targets], axis=0
                )
                target_mean = fitted_targets.mean(axis=0)
                target_scale = fitted_targets.std(axis=0)
                target_scale = np.where(target_scale < 1e-6, 1.0, target_scale)
                coefficients = fit_dual_ridge(
                    fitted_embeddings,
                    (fitted_targets - target_mean) / target_scale,
                    ridge,
                )
        return cls(
            model,
            coefficients,
            target_mean,
            target_scale,
            intent_dim=intent_dim,
            projection_seed=projection_seed,
            semantic_weight=semantic_weight,
            objective_weight=objective_weight,
            model_name=model_name,
            model_revision=model_revision or "unversioned",
            ridge=ridge,
            profile_decoder=profile_decoder,
            anchor_directions=anchor_directions,
            anchor_slopes=anchor_slopes,
            anchor_intercepts=anchor_intercepts,
            anchor_midpoints=anchor_midpoints,
            anchor_half_ranges=anchor_half_ranges,
            nli_model=nli_model,
            nli_model_name=nli_model_name,
            nli_model_revision=nli_model_revision,
            nli_batch_size=nli_batch_size,
        )

    def _embed(self, descriptions: List[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(
            descriptions,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def predict_profiles_from_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if self.profile_decoder == "contrastive_anchor":
            scores = np.asarray(embeddings, dtype=np.float32) @ self.anchor_directions.T
            normalized = np.clip(
                (scores - self.anchor_midpoints) / self.anchor_half_ranges,
                -1.0,
                1.0,
            )
            predictions = 1.0 + np.where(
                normalized >= 0.0, 0.7 * normalized, 0.5 * normalized
            )
            return predictions.astype(np.float32)
        if self.profile_decoder == "concept_anchor":
            scores = np.asarray(embeddings, dtype=np.float32) @ self.anchor_directions.T
            predictions = scores * self.anchor_slopes + self.anchor_intercepts
            return np.clip(predictions, 0.3, 2.2).astype(np.float32)
        standardised = np.asarray(embeddings, dtype=np.float32) @ self.coefficients
        return standardised * self.target_scale + self.target_mean

    def predict_profiles(self, entries: Sequence[Tuple[str, str]]) -> np.ndarray:
        if self.profile_decoder == "polarity_prototype":
            if self._polarity_prototype_centroids is None:
                prototypes = augmented_objective_prototypes()
                embeddings = self._embed([text for _, text, _ in prototypes])
                centroids = []
                for key in OBJECTIVE_KEYS:
                    for high_polarity in (False, True):
                        indices = [
                            index
                            for index, (prototype_key, _, value) in enumerate(prototypes)
                            if prototype_key == key and ((value > 1.0) == high_polarity)
                        ]
                        centroid = embeddings[indices].mean(axis=0)
                        centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
                        centroids.append(centroid)
                neutral_centroid = self._embed(
                    list(NEUTRAL_OBJECTIVE_PROTOTYPES)
                ).mean(axis=0)
                neutral_centroid /= max(float(np.linalg.norm(neutral_centroid)), 1e-8)
                centroids.append(neutral_centroid)
                self._polarity_prototype_centroids = np.asarray(
                    centroids, dtype=np.float32
                )
            query_embeddings = self._embed([
                str(description) for _, description in entries
            ])
            winners = np.argmax(
                query_embeddings @ self._polarity_prototype_centroids.T, axis=1
            )
            profiles = np.ones((len(entries), len(OBJECTIVE_KEYS)), dtype=np.float32)
            for row, winner in enumerate(winners):
                if int(winner) < 2 * len(OBJECTIVE_KEYS):
                    objective_index = int(winner) // 2
                    high_polarity = bool(int(winner) % 2)
                    profiles[row, objective_index] = 1.7 if high_polarity else 0.5
            return profiles
        if self.profile_decoder in {
            "nli_entailment", "nli_relevance_gated", "nli_similarity_gated",
            "nli_prototype_gated",
        }:
            descriptions = [str(description) for _, description in entries]
            missing = [
                description for description in descriptions
                if description not in self._nli_profile_cache
            ]
            if missing:
                pairs = []
                for description in missing:
                    for key in OBJECTIVE_KEYS:
                        pairs.append((
                            description,
                            OBJECTIVE_NLI_HYPOTHESES[key],
                        ))
                logits = np.asarray(self.nli_model.predict(
                    pairs,
                    batch_size=self.nli_batch_size,
                    show_progress_bar=False,
                ), dtype=np.float64)
                logits = logits - logits.max(axis=1, keepdims=True)
                probabilities = np.exp(logits)
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                label_map = {
                    str(label).lower(): int(index)
                    for index, label in self.nli_model.model.config.id2label.items()
                }
                entailment_index = label_map["entailment"]
                contradiction_index = label_map["contradiction"]
                entailment = probabilities[:, entailment_index].reshape(
                    len(missing), len(OBJECTIVE_KEYS)
                )
                contradiction = probabilities[:, contradiction_index].reshape(
                    len(missing), len(OBJECTIVE_KEYS)
                )
                score = entailment - contradiction
                if self.profile_decoder in {
                    "nli_relevance_gated", "nli_similarity_gated",
                    "nli_prototype_gated",
                }:
                    if self.profile_decoder == "nli_prototype_gated":
                        if self._nli_relevance_centroids is None:
                            prototypes = augmented_objective_prototypes()
                            prototype_embeddings = self._embed([
                                text for _, text, _ in prototypes
                            ])
                            centroids = []
                            for key in OBJECTIVE_KEYS:
                                indices = [
                                    index for index, (prototype_key, _, _) in enumerate(prototypes)
                                    if prototype_key == key
                                ]
                                centroid = prototype_embeddings[indices].mean(axis=0)
                                centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
                                centroids.append(centroid)
                            neutral_centroid = self._embed(
                                list(NEUTRAL_OBJECTIVE_PROTOTYPES)
                            ).mean(axis=0)
                            neutral_centroid /= max(
                                float(np.linalg.norm(neutral_centroid)), 1e-8
                            )
                            centroids.append(neutral_centroid)
                            self._nli_relevance_centroids = np.asarray(
                                centroids, dtype=np.float32
                            )
                        query_embeddings = self._embed(missing)
                        class_similarities = (
                            query_embeddings @ self._nli_relevance_centroids.T
                        )
                        winners = np.argmax(class_similarities, axis=1)
                        gate = np.zeros(
                            (len(missing), len(OBJECTIVE_KEYS)), dtype=np.float64
                        )
                        for row, winner in enumerate(winners):
                            if int(winner) < len(OBJECTIVE_KEYS):
                                gate[row, int(winner)] = 1.0
                    else:
                        if self._nli_concept_embeddings is None:
                            self._nli_concept_embeddings = self._embed([
                                OBJECTIVE_CONCEPT_TERMS[key] for key in OBJECTIVE_KEYS
                            ])
                            neutral_embedding = self._embed([
                                NEUTRAL_OBJECTIVE_PROTOTYPES[0]
                            ])
                            self._nli_neutral_similarities = (
                                neutral_embedding @ self._nli_concept_embeddings.T
                            )[0]
                        query_embeddings = self._embed(missing)
                        similarities = query_embeddings @ self._nli_concept_embeddings.T
                        if self.profile_decoder == "nli_relevance_gated":
                            relevance = np.maximum(
                                similarities - self._nli_neutral_similarities[None, :],
                                0.0,
                            )
                            max_relevance = relevance.max(axis=1, keepdims=True)
                            gate = (
                                (max_relevance >= 0.05)
                                & (relevance >= 0.75 * np.maximum(max_relevance, 1e-8))
                            ).astype(np.float64)
                        else:
                            max_similarity = similarities.max(axis=1, keepdims=True)
                            gate = (
                                similarities >= 0.90 * max_similarity
                            ).astype(np.float64)
                    score = score * gate
                profiles = 1.0 + np.where(
                    score >= 0.0, 0.7 * score, 0.5 * score
                )
                for description, profile in zip(missing, profiles):
                    self._nli_profile_cache[description] = profile.astype(np.float32)
            return np.stack([
                self._nli_profile_cache[description] for description in descriptions
            ]).astype(np.float32)
        return self.predict_profiles_from_embeddings(
            self._embed([description for _, description in entries])
        )

    def encode_entries(self, entries: Sequence[Tuple[str, str]]) -> np.ndarray:
        embeddings = self._embed([description for _, description in entries])
        objective_features = embeddings @ self.coefficients
        objective_features = _normalise_rows(objective_features)
        semantic_dim = self.intent_dim - len(OBJECTIVE_KEYS)
        parts = []
        if semantic_dim > 0:
            if self.semantic_weight > 0:
                semantic_features = _project_embeddings(
                    embeddings, semantic_dim, self.projection_seed
                )
                parts.append(np.sqrt(self.semantic_weight) * semantic_features)
            else:
                parts.append(np.zeros((len(embeddings), semantic_dim), dtype=np.float32))
        parts.append(np.sqrt(self.objective_weight) * objective_features)
        return _normalise_rows(np.concatenate(parts, axis=1)).astype(np.float32)

    def metadata(self, train_labels: Sequence[str]) -> Dict[str, object]:
        return {
            "representation_type": "objective_grounded_semantic",
            "semantic_geometry": True,
            "objective_grounded": True,
            "embed_model": self.model_name,
            "model_revision": self.model_revision,
            "projection_dim": self.intent_dim,
            "projection_seed": self.projection_seed,
            "adapter": "dual_ridge",
            "adapter_ridge": self.ridge,
            "profile_decoder": self.profile_decoder,
            "profile_decoder_monotonic": self.profile_decoder in {
                "concept_anchor", "contrastive_anchor"
            },
            "anchor_calibration_slopes": (
                self.anchor_slopes.tolist() if self.anchor_slopes is not None else None
            ),
            "objective_text_anchors": (
                {key: list(OBJECTIVE_TEXT_ANCHORS[key]) for key in OBJECTIVE_KEYS}
                if self.profile_decoder in {
                    "concept_anchor", "contrastive_anchor", "prototype_ridge",
                    "augmented_prototype_ridge",
                } else None
            ),
            "augmented_prototype_count": (
                len(augmented_objective_prototypes())
                + len(NEUTRAL_OBJECTIVE_PROTOTYPES)
                if self.profile_decoder in {
                    "augmented_prototype_ridge", "polarity_prototype"
                } else 0
            ),
            "nli_model": (
                self.nli_model_name if self.profile_decoder.startswith("nli_") else None
            ),
            "nli_model_revision": (
                self.nli_model_revision if self.profile_decoder.startswith("nli_") else None
            ),
            "nli_decision_rule": (
                "P(entailment|objective-specific declarative hypothesis)-"
                "P(contradiction|objective-specific declarative hypothesis)"
                if self.profile_decoder in {
                    "nli_entailment", "nli_relevance_gated", "nli_similarity_gated",
                    "nli_prototype_gated",
                } else None
            ),
            "nli_relevance_gate": (
                (
                    "positive cosine delta vs neutral; retain >=75% of query max when max>=0.05"
                    if self.profile_decoder == "nli_relevance_gated"
                    else (
                        "nearest centroid among seven independently templated objective classes and a neutral class"
                        if self.profile_decoder == "nli_prototype_gated"
                        else "retain concepts with cosine similarity >=90% of query maximum"
                    )
                )
                if self.profile_decoder in {
                    "nli_relevance_gated", "nli_similarity_gated",
                    "nli_prototype_gated",
                } else None
            ),
            "polarity_prototype_rule": (
                "nearest centroid among seven low, seven high, and one neutral class"
                if self.profile_decoder == "polarity_prototype" else None
            ),
            "semantic_weight": self.semantic_weight,
            "objective_weight": self.objective_weight,
            "objective_keys": list(OBJECTIVE_KEYS),
            "adapter_train_labels": list(train_labels),
        }
