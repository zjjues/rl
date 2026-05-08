"""LLM-powered intent generation and vector library for I-MAPPO.

Provides:
- IntentLibrary: manage, persist, and sample semantic intent vectors
- LLM-powered description generation (Anthropic / OpenAI)
- Embedding via OpenAI API or deterministic hash (offline mode)
- Pre-built intent descriptions for UAV scheduling domain
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

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


# ── LLM prompt templates ───────────────────────────────────────────────────

def _build_intent_generation_prompt(n: int, domain: str) -> str:
    return f"""You are designing strategic intent vectors for a multi-UAV reinforcement learning system.
Each intent is a natural-language description of a high-level strategy that UAVs should follow during a mission.

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
每个意图是一段自然语言描述，表示无人机在任务中应遵循的高层策略。

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


# ── LLM calling helpers ────────────────────────────────────────────────────

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
    """Parse LLM JSON response into (descriptions, labels)."""
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
    """Manages a library of semantic intent vectors for I-MAPPO training.

    Intents are embedded from natural language descriptions and serve as
    conditioning signals for the actor, critic, and potential-based reward shaping.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        descriptions: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
    ):
        self.vectors = vectors  # (N, intent_dim) float32
        self.descriptions = descriptions or []
        self.labels = labels or []
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
        self, n: int = 1, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Sample intents returning (vectors, labels, indices)."""
        rng = rng or np.random.default_rng()
        indices = rng.integers(0, len(self.vectors), size=n)
        vectors = self.vectors[indices].copy()
        labels = [self.labels[i] for i in indices] if self.labels else []
        return vectors, labels, indices

    def get_by_label(self, label: str) -> Optional[np.ndarray]:
        """Get intent vector by label name."""
        for i, lab in enumerate(self.labels):
            if lab == label:
                return self.vectors[i].copy()
        return None

    def get_by_index(self, idx: int) -> np.ndarray:
        """Get intent vector by index."""
        return self.vectors[idx].copy()

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
            return IntentLibrary(
                vectors=self.vectors[idx].copy(),
                descriptions=[self.descriptions[i] for i in idx] if self.descriptions else [],
                labels=[self.labels[i] for i in idx] if self.labels else [],
                metadata={**self.metadata},
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
    def create_static(
        cls,
        intent_dim: int = 64,
        descriptions: Optional[List[Tuple[str, str]]] = None,
        domain: str = "uav",
    ) -> "IntentLibrary":
        """Create library from pre-built descriptions using deterministic hash embedding.

        This requires NO API calls and is the fastest way to get started.
        Each description is hashed to a deterministic unit vector via SHA256→PRNG.

        Args:
            intent_dim: Dimensionality of intent vectors.
            descriptions: Optional custom list of (label, description) pairs.
            domain: "uav" (default) uses UAV intent descriptions,
                    "vmas" uses VMAS/Multi-Agent Particle descriptions.
        """
        if descriptions is not None:
            entries = descriptions
        elif domain == "vmas":
            entries = VMAS_INTENT_DESCRIPTIONS
        else:
            entries = DEFAULT_INTENT_DESCRIPTIONS
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
            metadata={"embed_model": "static_hash", "source": "prebuilt"},
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
        if embed_model == "static":
            entries = [(lab or f"intent_{i}", desc) for i, (desc, lab) in
                       enumerate(zip(descriptions, labels or [None] * len(descriptions)))]
            return cls.create_static(
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
            embeddings = _resize_embeddings(embeddings, intent_dim)

        return cls(
            vectors=embeddings,
            descriptions=descriptions,
            labels=labels or [],
            metadata={"embed_model": embed_model, "source": "llm_embedding"},
        )

    @classmethod
    def generate_from_llm(
        cls,
        n: int = 50,
        llm_client: Optional[object] = None,
        llm_model: str = "claude-sonnet-4-20250514",
        embed_model: str = "static",
        embed_client: Optional[object] = None,
        intent_dim: int = 64,
        api_key: Optional[str] = None,
        domain: str = "uav_scheduling",
        language: str = "en",
    ) -> "IntentLibrary":
        """Full pipeline: LLM generates descriptions → embed → IntentLibrary.

        Args:
            n: Number of intent descriptions to generate.
            llm_client: Anthropic or OpenAI client instance.
            llm_model: Model name for text generation.
            embed_model: Embedding model ("static", "text-embedding-3-small", etc.).
            embed_client: Client for embedding API.
            intent_dim: Target intent vector dimension.
            api_key: API key (used if clients are not provided).
            domain: Domain description for the generation prompt.
            language: "en" or "zh".
        """
        # 1. Generate descriptions via LLM
        descriptions, labels = _generate_intents_from_llm(
            n=n, client=llm_client, model=llm_model,
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


def _generate_intents_from_llm(
    n: int,
    client: Optional[object],
    model: str,
    api_key: Optional[str],
    domain: str,
    language: str,
) -> Tuple[List[str], List[str]]:
    """Call an LLM to generate intent descriptions. Returns (descriptions, labels)."""
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
                raise RuntimeError(f"LLM call failed with provided client: {e}")

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
            "Failed to generate intents via LLM. Ensure you have either:\n"
            "  - anthropic SDK installed and ANTHROPIC_API_KEY set, or\n"
            "  - openai SDK installed and OPENAI_API_KEY set, or\n"
            "  - Pass a pre-configured client via llm_client parameter.\n"
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
