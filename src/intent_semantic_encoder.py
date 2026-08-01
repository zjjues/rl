"""Intent description generation and representation library for I-MAPPO.

Provides:
- IntentLibrary: manage, persist, and sample intent representations
- Offline semantic intent description generation (Anthropic / OpenAI)
- Explicit representation families: legacy hash, random dense, and pretrained text
- Pre-built intent descriptions for UAV scheduling domain

Important: deterministic whole-text hashes do not preserve semantic geometry. They are
kept only for reproducing legacy experiments and must not be reported as semantic
embeddings.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── Pre-built intent descriptions ──────────────────────────────────────────

DEFAULT_INTENT_DESCRIPTIONS = [
    ("safety_first", "Prioritize safety: maximize separation between UAVs, avoid all collision risks even at the cost of slower task completion"),
    ("efficiency_first", "Maximize efficiency: complete all assigned tasks as quickly as possible, accept moderate collision risk"),
    ("balanced", "Balanced approach: equal weight on task completion speed and collision avoidance"),
    ("energy_saving", "Energy conservation: minimize movement and acceleration, prefer slow steady trajectories"),
    ("aggressive_pursuit", "Aggressive task pursuit: charge directly at targets, minimal detour for obstacle avoidance"),
    ("cautious_exploration", "Cautious exploration: move slowly, maintain safe distances, prioritize gathering situational awareness"),
    ("load_balancing", "Load balancing: distribute work evenly across all UAVs, avoid overloading any single unit"),
    ("formation_keeping", "Formation keeping: maintain geometric formation while approaching targets as a group"),
    ("threat_avoidance", "Threat avoidance: actively steer clear of threat zones and radar coverage areas"),
    ("threat_engagement", "Threat engagement: deliberately enter threat zones to engage high-value targets"),
    ("perimeter_patrol", "Perimeter patrol: circle the operational boundary, intercept targets at the edge"),
    ("center_convergence", "Center rush: all UAVs converge rapidly to the center of the operational area"),
    ("decentralized_sweep", "Decentralized sweep: each UAV covers its own sector independently"),
    ("relay_coordination", "Relay coordination: UAVs take turns approaching targets while others provide overwatch"),
    ("minimal_communication", "Minimum communication: operate with implicit coordination, minimal mutual awareness"),
    ("full_coordination", "Full coordination: maintain tight awareness of all other UAV positions and intentions"),
    ("altitude_separation", "Altitude separation: use vertical spacing as primary collision avoidance strategy"),
    ("speed_modulation", "Speed modulation: vary individual speeds to create temporal separation at crossing points"),
    ("reactive_avoidance", "Reactive avoidance: only take evasive action when collision is imminent"),
    ("predictive_planning", "Predictive planning: anticipate future conflicts several steps ahead and adjust early"),
    ("target_priority", "Target prioritization: focus on the nearest target first, then move to the next"),
    ("coverage_maximization", "Coverage maximization: spread out to maximize spatial coverage of the operational area"),
    ("stealth_approach", "Stealth approach: minimize detectability by reducing speed and avoiding threat zones"),
    ("rapid_response", "Rapid response: maximum acceleration to reach targets, accept energy cost"),
    ("hover_and_observe", "Hover and observe: station-keep at safe altitude, only move when target is clearly identified"),
]


# The posture taxonomy is research metadata, not an outcome inferred from a trained
# policy. It ensures the description presented to the policy agrees with the reward
# semantics configured in the UAV environment.
UAV_INTENT_POSTURES: Dict[str, str] = {
    "safety_first": "stealth",
    "efficiency_first": "attack",
    "balanced": "neutral",
    "energy_saving": "stealth",
    "aggressive_pursuit": "attack",
    "cautious_exploration": "stealth",
    "load_balancing": "neutral",
    "formation_keeping": "neutral",
    "threat_avoidance": "stealth",
    "threat_engagement": "attack",
    "perimeter_patrol": "neutral",
    "center_convergence": "attack",
    "decentralized_sweep": "neutral",
    "relay_coordination": "neutral",
    "minimal_communication": "neutral",
    "full_coordination": "neutral",
    "altitude_separation": "stealth",
    "speed_modulation": "neutral",
    "reactive_avoidance": "attack",
    "predictive_planning": "stealth",
    "target_priority": "attack",
    "coverage_maximization": "neutral",
    "stealth_approach": "stealth",
    "rapid_response": "attack",
    "hover_and_observe": "stealth",
}

# ── VMAS / Multi-Agent Particle intent descriptions ────────────────────────────

VMAS_INTENT_DESCRIPTIONS = [
    ("cautious_navigation", "Maintain safe distance from all other agents, prioritize collision-free paths even at the cost of slower coverage"),
    ("aggressive_intercept", "Rush directly to the nearest uncovered landmark, accept close proximity to other agents"),
    ("balanced_coverage", "Equal weight on covering all landmarks quickly and maintaining agent separation"),
    ("formation_spread", "Distribute agents evenly across the spatial area, maintaining a geometric formation while approaching landmarks"),
    ("cooperative_signaling", "Agents take turns approaching landmarks, coordinating to maximize overall coverage"),
    ("perimeter_first", "Cover landmarks on the outer boundary first, then move inward"),
    ("center_priority", "Prioritize landmarks near the center of the environment"),
    ("greedy_nearest", "Each agent independently moves to its nearest uncovered landmark"),
    ("energy_efficient", "Minimize acceleration and movement, prefer slow steady trajectories"),
    ("speed_optimized", "Maximize speed to reach landmarks as quickly as possible"),
    ("reactive_avoidance", "Only avoid other agents when collision is imminent"),
    ("predictive_planning", "Anticipate other agents' movements several steps ahead"),
    ("territorial_division", "Each agent takes responsibility for a fixed spatial sector"),
    ("load_balanced", "Dynamically reassign landmark targets to balance workload across agents"),
    ("communication_minimal", "Operate with minimal awareness of other agents' positions"),
    ("full_awareness", "Maintain tight awareness of all agent positions and landmark assignments"),
    ("sequential_sweep", "Agents sweep the environment in a coordinated sequential pattern"),
    ("random_exploration", "Explore the environment with high entropy, accept temporary inefficiency"),
    ("boundary_patrol", "Circle the environment boundary, intercept landmarks at the edge"),
    ("cluster_then_spread", "Initially cluster together, then spread out once landmark positions are known"),
    ("landmark_priority", "Prioritize landmarks that are currently uncovered over maintaining agent separation"),
    ("safe_conservative", "Move only when a clear path to the next landmark is available"),
    ("hierarchical_leader", "One agent acts as coordinator, others follow its assignments"),
    ("emergent_swarm", "Use simple local rules to achieve emergent coverage behavior"),
    ("minimal_movement", "Stay as still as possible, only move to cover a landmark when absolutely necessary"),
]


# ── semantic intent prompt templates ───────────────────────────────────────────────────

def _build_intent_generation_prompt(n: int, domain: str) -> str:
    return f"""You are designing strategic intent vectors for a multi-UAV reinforcement learning system.
Each intent is a textual description of a high-level strategy that UAVs should follow during a mission.

Domain: {domain}
Number of intents needed: {n}

Requirements:
- Cover diverse strategic dimensions: safety, efficiency, coordination, energy, threat handling, coverage
- Each description should be 1-2 sentences, concrete and actionable
- Include both cooperative and independent strategies
- Include both conservative and aggressive strategies
- Label each with a short snake_case identifier

Return a JSON array of objects with "label" and "description" fields:
```json
[
  {{"label": "safety_first", "description": "Prioritize safety: ..."}},
  ...
]
```

Generate exactly {n} diverse intents. Return ONLY the JSON array, no other text."""


def _build_intent_generation_prompt_zh(n: int, domain: str) -> str:
    return f"""你正在为一个多无人机强化学习系统设计策略意图向量。
每个意图是一段文本描述，表示无人机在任务中应遵循的高层策略。

领域: {domain}
需要的意图数量: {n}

要求:
- 覆盖多样的策略维度：安全、效率、协调、节能、威胁处理、覆盖范围
- 每个描述1-2句话，具体可执行
- 同时包含合作策略和独立策略
- 同时包含保守策略和激进策略
- 每个意图标注简短的英文 snake_case 标签

返回 JSON 数组，每个元素包含 "label" 和 "description" 字段：
```json
[
  {{"label": "safety_first", "description": "安全优先：..."}},
  ...
]
```

生成恰好 {n} 个多样化的意图。只返回 JSON 数组，不要其他文字。"""


# ── optional description-generation helpers ────────────────────────────────────────────────────

def _call_anthropic(client, model: str, prompt: str) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _call_openai(client, model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _parse_intent_response(text: str) -> Tuple[List[str], List[str]]:
    """Parse semantic intent JSON response into (descriptions, labels)."""
    # Try to extract JSON array from the response
    text = text.strip()
    # Remove markdown code fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    data = json.loads(text)
    descriptions = [item["description"] for item in data]
    labels = [item["label"] for item in data]
    return descriptions, labels


# ── IntentLibrary ──────────────────────────────────────────────────────────

class IntentLibrary:
    """Manages a library of intent vectors for I-MAPPO training.

    Only libraries with ``metadata["semantic_geometry"] == True`` should be
    described as semantic embeddings in a paper. Other representations are controls.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        descriptions: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
    ):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
            raise ValueError("intent vectors must have shape (n_intents, intent_dim)")
        if descriptions is not None and len(descriptions) not in {0, vectors.shape[0]}:
            raise ValueError("descriptions and vectors must contain the same number of intents")
        if labels is not None and len(labels) not in {0, vectors.shape[0]}:
            raise ValueError("labels and vectors must contain the same number of intents")
        self.vectors = vectors
        self.descriptions = list(descriptions or [])
        self.labels = list(labels or [])
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.vectors)

    def __repr__(self) -> str:
        return f"IntentLibrary(n={len(self)}, dim={self.intent_dim}, labels={self.labels[:5]}...)"

    @property
    def intent_dim(self) -> int:
        return int(self.vectors.shape[1])

    # ── sampling ───────────────────────────────────────────────────────

    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Randomly sample n intent vectors."""
        rng = rng or np.random.default_rng()
        indices = rng.integers(0, len(self.vectors), size=n)
        return self.vectors[indices].copy()

    def sample_single(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample a single intent vector (returns 1D array)."""
        return self.sample(1, rng=rng)[0]

    def sample_with_info(
        self,
        n: int = 1,
        rng: Optional[np.random.Generator] = None,
        posture: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Sample intents returning (vectors, labels, indices)."""
        rng = rng or np.random.default_rng()
        candidates = self.indices_for_posture(posture) if posture else np.arange(len(self.vectors))
        if len(candidates) == 0:
            raise ValueError(f"intent library has no candidates for posture={posture!r}")
        indices = rng.choice(candidates, size=n, replace=True)
        vectors = self.vectors[indices].copy()
        labels = [self.labels[i] for i in indices] if self.labels else []
        return vectors, labels, indices

    def posture_for_index(self, idx: int) -> str:
        """Return the structured tactical posture associated with an intent."""
        postures = self.metadata.get("postures", [])
        if idx < len(postures):
            return str(postures[idx])
        if self.labels and idx < len(self.labels):
            return infer_intent_posture(self.labels[idx], self.descriptions[idx] if self.descriptions else "")
        return "neutral"

    def posture_for_label(self, label: str) -> str:
        for idx, current in enumerate(self.labels):
            if current == label:
                return self.posture_for_index(idx)
        return "neutral"

    def indices_for_posture(self, posture: Optional[str]) -> np.ndarray:
        """Return indices compatible with a posture, including neutral intents.

        Neutral intents are admitted for either attack or stealth because they do not
        contradict the environment objective. Exact ``neutral`` sampling remains
        neutral-only.
        """
        if posture is None:
            return np.arange(len(self.vectors), dtype=np.int64)
        posture = str(posture).lower()
        accepted = {posture} if posture == "neutral" else {posture, "neutral"}
        return np.asarray(
            [idx for idx in range(len(self.vectors)) if self.posture_for_index(idx) in accepted],
            dtype=np.int64,
        )

    def get_by_label(self, label: str) -> Optional[np.ndarray]:
        """Get intent vector by label name."""
        for i, lab in enumerate(self.labels):
            if lab == label:
                return self.vectors[i].copy()
        return None

    def get_by_index(self, idx: int) -> np.ndarray:
        """Get intent vector by index."""
        return self.vectors[idx].copy()

    def subset_by_labels(
        self,
        labels: Sequence[str],
        *,
        strict: bool = True,
    ) -> "IntentLibrary":
        """Return an ordered label subset without changing the vector space.

        Keeping the original vector dimensionality is important for identity-code
        controls: a held-out one-hot coordinate remains representable at evaluation
        time but is never activated during training.
        """
        if not self.labels:
            raise ValueError("cannot select labels from an unlabeled intent library")
        requested = list(labels)
        if len(requested) != len(set(requested)):
            raise ValueError("intent subset labels must be unique")
        index_by_label = {label: idx for idx, label in enumerate(self.labels)}
        missing = [label for label in requested if label not in index_by_label]
        if missing and strict:
            raise ValueError(f"unknown intent labels in subset: {missing}")
        indices = [index_by_label[label] for label in requested if label in index_by_label]
        if not indices:
            raise ValueError("intent label subset is empty")
        subset_metadata = {**self.metadata}
        if "postures" in subset_metadata:
            postures = list(subset_metadata["postures"])
            subset_metadata["postures"] = [postures[idx] for idx in indices]
        subset_metadata["parent_library_size"] = len(self)
        subset_metadata["train_labels"] = [self.labels[idx] for idx in indices]
        return IntentLibrary(
            vectors=self.vectors[indices].copy(),
            descriptions=[self.descriptions[idx] for idx in indices] if self.descriptions else [],
            labels=[self.labels[idx] for idx in indices],
            metadata=subset_metadata,
        )

    # ── train/test split ────────────────────────────────────────────────

    def split(
        self, train_frac: float = 0.8, seed: int = 42
    ) -> Tuple["IntentLibrary", "IntentLibrary"]:
        """Split into train and test subsets (for intent generalization testing)."""
        rng = np.random.default_rng(seed)
        n = len(self.vectors)
        n_train = int(n * train_frac)
        indices = rng.permutation(n)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        def subset(idx):
            subset_metadata = {**self.metadata}
            if "postures" in subset_metadata:
                postures = list(subset_metadata["postures"])
                subset_metadata["postures"] = [postures[i] for i in idx]
            return IntentLibrary(
                vectors=self.vectors[idx].copy(),
                descriptions=[self.descriptions[i] for i in idx] if self.descriptions else [],
                labels=[self.labels[i] for i in idx] if self.labels else [],
                metadata=subset_metadata,
            )

        return subset(train_idx), subset(test_idx)

    # ── persistence ─────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist to disk as .npz + .json pair."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), vectors=self.vectors)
        meta = {
            "descriptions": self.descriptions,
            "labels": self.labels,
            "metadata": self.metadata,
            "intent_dim": self.intent_dim,
            "size": len(self),
        }
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "IntentLibrary":
        """Load from .npz + .json pair."""
        path = Path(path)
        data = np.load(path.with_suffix(".npz"))
        vectors = data["vectors"]
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {}
        return cls(
            vectors=vectors,
            descriptions=meta.get("descriptions", []),
            labels=meta.get("labels", []),
            metadata=meta.get("metadata", {}),
        )

    # ── factory methods ─────────────────────────────────────────────────

    @classmethod
    def create_legacy_hash(
        cls,
        intent_dim: int = 64,
        descriptions: Optional[List[Tuple[str, str]]] = None,
        domain: str = "uav",
    ) -> "IntentLibrary":
        """Reproduce the historical whole-text hash representation.

        This representation is deterministic but has no semantic geometry. It exists
        only to reproduce Stage7 legacy experiments.
        """
        entries = _resolve_entries(descriptions, domain)
        labels = [e[0] for e in entries]
        texts = [e[1] for e in entries]

        vectors = np.zeros((len(texts), intent_dim), dtype=np.float32)
        for i, text in enumerate(texts):
            h = hashlib.sha256(text.encode()).digest()
            seed = int.from_bytes(h[:8], "big")
            rng = np.random.default_rng(seed)
            vectors[i] = rng.normal(0, 1, intent_dim).astype(np.float32)

        # Normalize to unit vectors for stable dot-product use in Phi
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-8)

        return cls(
            vectors=vectors,
            descriptions=texts,
            labels=labels,
            metadata={
                "representation_type": "legacy_hash",
                "embed_model": "sha256_prng",
                "semantic_geometry": False,
                "source": "prebuilt",
                "domain": domain,
                "postures": [infer_intent_posture(label, text) for label, text in entries],
            },
        )

    @classmethod
    def create_static(
        cls,
        intent_dim: int = 64,
        descriptions: Optional[List[Tuple[str, str]]] = None,
        domain: str = "uav",
    ) -> "IntentLibrary":
        """Deprecated compatibility alias for :meth:`create_legacy_hash`."""
        warnings.warn(
            "IntentLibrary.create_static() creates a legacy hash control, not a semantic embedding; "
            "use create_legacy_hash(), create_random_dense(), or create_pretrained().",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.create_legacy_hash(intent_dim, descriptions, domain)

    @classmethod
    def create_random_dense(
        cls,
        intent_dim: int = 64,
        descriptions: Optional[List[Tuple[str, str]]] = None,
        domain: str = "uav",
        seed: int = 0,
    ) -> "IntentLibrary":
        """Create the random dense-code control independent of description text."""
        entries = _resolve_entries(descriptions, domain)
        rng = np.random.default_rng(seed)
        vectors = rng.normal(size=(len(entries), intent_dim)).astype(np.float32)
        vectors = _normalise_rows(vectors)
        return cls(
            vectors=vectors,
            descriptions=[text for _, text in entries],
            labels=[label for label, _ in entries],
            metadata={
                "representation_type": "random_dense",
                "embed_model": "gaussian_prng",
                "semantic_geometry": False,
                "source": "prebuilt",
                "domain": domain,
                "seed": int(seed),
                "postures": [infer_intent_posture(label, text) for label, text in entries],
            },
        )

    @classmethod
    def create_onehot(
        cls,
        intent_dim: Optional[int] = None,
        descriptions: Optional[List[Tuple[str, str]]] = None,
        domain: str = "uav",
    ) -> "IntentLibrary":
        """Create a true intent-identity one-hot control.

        ``intent_dim`` may pad the identity vectors but cannot be smaller than the
        catalog. This prevents the historical three-mode posture code from being
        mislabeled as a one-hot baseline over natural-language intents.
        """
        entries = _resolve_entries(descriptions, domain)
        required_dim = len(entries)
        resolved_dim = required_dim if intent_dim is None else int(intent_dim)
        if resolved_dim < required_dim:
            raise ValueError(
                f"onehot intent_dim must be at least the catalog size "
                f"({required_dim}), got {resolved_dim}"
            )
        vectors = np.zeros((required_dim, resolved_dim), dtype=np.float32)
        vectors[np.arange(required_dim), np.arange(required_dim)] = 1.0
        return cls(
            vectors=vectors,
            descriptions=[text for _, text in entries],
            labels=[label for label, _ in entries],
            metadata={
                "representation_type": "onehot",
                "embed_model": "identity",
                "semantic_geometry": False,
                "source": "prebuilt",
                "domain": domain,
                "catalog_size": required_dim,
                "postures": [infer_intent_posture(label, text) for label, text in entries],
            },
        )

    @classmethod
    def create_pretrained(
        cls,
        intent_dim: int = 64,
        descriptions: Optional[List[Tuple[str, str]]] = None,
        domain: str = "uav",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_revision: Optional[str] = None,
        batch_size: int = 32,
        projection_seed: int = 0,
        device: Optional[str] = None,
    ) -> "IntentLibrary":
        """Encode descriptions using a frozen sentence-transformers model.

        The method never falls back to a random/hash representation. A missing model
        dependency or unavailable checkpoint is an explicit experiment setup error.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "pretrained_semantic requires sentence-transformers; install requirements-research.txt"
            ) from exc

        entries = _resolve_entries(descriptions, domain)
        labels = [label for label, _ in entries]
        texts = [text for _, text in entries]
        model_kwargs = {"device": device}
        if model_revision:
            model_kwargs["revision"] = model_revision
        model = SentenceTransformer(model_name, **model_kwargs)
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        original_dim = int(vectors.shape[1])
        vectors = _project_embeddings(vectors, intent_dim, projection_seed)
        return cls(
            vectors=vectors,
            descriptions=texts,
            labels=labels,
            metadata={
                "representation_type": "pretrained_semantic",
                "embed_model": model_name,
                "model_revision": model_revision or "unversioned",
                "semantic_geometry": True,
                "source": "prebuilt",
                "domain": domain,
                "original_dim": original_dim,
                "projection_dim": int(intent_dim),
                "projection_seed": int(projection_seed),
                "postures": [infer_intent_posture(label, text) for label, text in entries],
            },
        )

    @classmethod
    def from_descriptions(
        cls,
        descriptions: List[str],
        labels: Optional[List[str]] = None,
        embed_model: str = "text-embedding-3-small",
        embed_client: Optional[object] = None,
        intent_dim: Optional[int] = None,
        api_key: Optional[str] = None,
        projection_seed: int = 0,
    ) -> "IntentLibrary":
        """Embed text descriptions into vectors via an embedding model.

        Args:
            descriptions: List of intent descriptions.
            labels: Optional short labels.
            embed_model: One of:
                - "text-embedding-3-small" / "text-embedding-3-large" (OpenAI)
                - "static" (hash-based, no API)
            embed_client: Pre-configured OpenAI client.
            intent_dim: Target dimension (only used with "static" mode).
            api_key: OpenAI API key.
        """
        if embed_model in {"static", "legacy_hash"}:
            entries = [(lab or f"intent_{i}", desc) for i, (desc, lab) in
                       enumerate(zip(descriptions, labels or [None] * len(descriptions)))]
            return cls.create_legacy_hash(
                intent_dim=intent_dim or 64,
                descriptions=entries,
            )

        from openai import OpenAI
        client = embed_client or OpenAI(api_key=api_key)

        all_embeddings = []
        batch_size = 20
        for i in range(0, len(descriptions), batch_size):
            batch = descriptions[i:i + batch_size]
            resp = client.embeddings.create(model=embed_model, input=batch)
            all_embeddings.extend([item.embedding for item in resp.data])

        embeddings = np.array(all_embeddings, dtype=np.float32)

        if intent_dim is not None and embeddings.shape[1] != intent_dim:
            embeddings = _project_embeddings(embeddings, intent_dim, projection_seed)

        return cls(
            vectors=embeddings,
            descriptions=descriptions,
            labels=labels or [],
            metadata={
                "representation_type": "pretrained_semantic",
                "embed_model": embed_model,
                "semantic_geometry": True,
                "source": "generated_descriptions",
                "projection_seed": int(projection_seed),
                "postures": [
                    infer_intent_posture(label, description)
                    for label, description in zip(labels or [f"intent_{i}" for i in range(len(descriptions))], descriptions)
                ],
            },
        )

    @classmethod
    def generate_from_semantic_library(
        cls,
        n: int = 50,
        semantic_library_client: Optional[object] = None,
        semantic_library_model: str = "claude-sonnet-4-20250514",
        embed_model: str = "static",
        embed_client: Optional[object] = None,
        intent_dim: int = 64,
        api_key: Optional[str] = None,
        domain: str = "uav_scheduling",
        language: str = "en",
    ) -> "IntentLibrary":
        """Full pipeline: offline semantic-intent workflow prepares descriptions → embed → IntentLibrary.

        Args:
            n: Number of intent descriptions to generate.
            semantic_library_client: Anthropic or OpenAI client instance.
            semantic_library_model: Model name for text generation.
            embed_model: Embedding model ("static", "text-embedding-3-small", etc.).
            embed_client: Client for embedding API.
            intent_dim: Target intent vector dimension.
            api_key: API key (used if clients are not provided).
            domain: Domain description for the generation prompt.
            language: "en" or "zh".
        """
        # 1. Prepare semantic descriptions
        descriptions, labels = _generate_intents_from_semantic_library(
            n=n, client=semantic_library_client, model=semantic_library_model,
            api_key=api_key, domain=domain, language=language,
        )
        # 2. Embed
        return cls.from_descriptions(
            descriptions=descriptions,
            labels=labels,
            embed_model=embed_model,
            embed_client=embed_client,
            intent_dim=intent_dim,
            api_key=api_key,
        )


def _generate_intents_from_semantic_library(
    n: int,
    client: Optional[object],
    model: str,
    api_key: Optional[str],
    domain: str,
    language: str,
) -> Tuple[List[str], List[str]]:
    """Generate candidate text descriptions to generate intent descriptions. Returns (descriptions, labels)."""
    if language == "zh":
        prompt = _build_intent_generation_prompt_zh(n, domain)
    else:
        prompt = _build_intent_generation_prompt(n, domain)

    response_text = None

    # Try provided client first
    if client is not None:
        try:
            response_text = _call_anthropic(client, model, prompt)
        except Exception:
            try:
                response_text = _call_openai(client, model, prompt)
            except Exception as e:
                raise RuntimeError(f"description generation failed with provided client: {e}")

    # Auto-detect and try Anthropic SDK
    if response_text is None:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response_text = _call_anthropic(client, model, prompt)
        except ImportError:
            pass
        except Exception:
            pass

    # Fall back to OpenAI SDK
    if response_text is None:
        try:
            from openai import OpenAI
            oai_model = model if model.startswith("gpt") else "gpt-4o"
            client = OpenAI(api_key=api_key)
            response_text = _call_openai(client, oai_model, prompt)
        except ImportError:
            pass
        except Exception:
            pass

    if response_text is None:
        raise RuntimeError(
            "Failed to generate intents. Ensure you have either:\n"
            "  - anthropic SDK installed and ANTHROPIC_API_KEY set, or\n"
            "  - openai SDK installed and OPENAI_API_KEY set, or\n"
            "  - Pass a pre-configured client via semantic_library_client parameter.\n"
            "Alternatively, use IntentLibrary.create_static() for offline mode."
        )

    return _parse_intent_response(response_text)


def _resize_embeddings(embeddings: np.ndarray, target_dim: int) -> np.ndarray:
    """Project embeddings to target_dim via truncated SVD."""
    current_dim = embeddings.shape[1]
    if current_dim == target_dim:
        return embeddings
    U, S, Vt = np.linalg.svd(embeddings, full_matrices=False)
    if target_dim < current_dim:
        return (U[:, :target_dim] * S[:target_dim]).astype(np.float32)
    else:
        padded = np.zeros((embeddings.shape[0], target_dim), dtype=np.float32)
        padded[:, :current_dim] = embeddings
        return padded


def _resolve_entries(
    descriptions: Optional[Sequence[Tuple[str, str]]],
    domain: str,
) -> List[Tuple[str, str]]:
    if descriptions is not None:
        return [(str(label), str(text)) for label, text in descriptions]
    if domain == "vmas":
        return list(VMAS_INTENT_DESCRIPTIONS)
    return list(DEFAULT_INTENT_DESCRIPTIONS)


def _normalise_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def _project_embeddings(
    embeddings: np.ndarray,
    target_dim: int,
    seed: int,
) -> np.ndarray:
    """Project text embeddings while approximately preserving pairwise geometry."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    current_dim = int(embeddings.shape[1])
    if target_dim <= 0:
        raise ValueError("target_dim must be positive")
    if current_dim == target_dim:
        return _normalise_rows(embeddings)
    if target_dim < current_dim:
        rng = np.random.default_rng(seed)
        projection = rng.normal(
            0.0,
            1.0 / np.sqrt(target_dim),
            size=(current_dim, target_dim),
        ).astype(np.float32)
        return _normalise_rows(embeddings @ projection)
    padded = np.zeros((embeddings.shape[0], target_dim), dtype=np.float32)
    padded[:, :current_dim] = embeddings
    return _normalise_rows(padded)


def infer_intent_posture(label: str, description: str = "") -> str:
    """Map intent metadata to the environment's attack/stealth/neutral taxonomy."""
    label = str(label).lower()
    if label in UAV_INTENT_POSTURES:
        return UAV_INTENT_POSTURES[label]
    text = f"{label} {description}".lower()
    stealth_terms = (
        "safe", "safety", "cautious", "avoid", "stealth", "conserve",
        "slow", "predictive", "observe", "separation",
    )
    attack_terms = (
        "attack", "aggressive", "engage", "rapid", "rush", "speed",
        "pursuit", "intercept", "priority",
    )
    stealth_score = sum(term in text for term in stealth_terms)
    attack_score = sum(term in text for term in attack_terms)
    if stealth_score > attack_score:
        return "stealth"
    if attack_score > stealth_score:
        return "attack"
    return "neutral"
