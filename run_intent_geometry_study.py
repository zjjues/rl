"""Run a non-overwriting intent-representation geometry pilot."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intent_generalization import load_generalization_suite, representation_retrieval_diagnostics  # noqa: E402
from intent_semantic_encoder import (  # noqa: E402
    DEFAULT_INTENT_DESCRIPTIONS,
    IntentLibrary,
    _project_embeddings,
    infer_intent_posture,
    _normalise_rows,
)
from intent_objectives import OBJECTIVE_KEYS, resolve_intent_reward_profile  # noqa: E402
from objective_semantic_adapter import fit_dual_ridge  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
    ).stdout.strip()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def geometry_distortion(reference: np.ndarray, projected: np.ndarray) -> Dict[str, float]:
    reference_cosine = reference @ reference.T
    projected_cosine = projected @ projected.T
    upper = np.triu_indices(reference.shape[0], k=1)
    reference_values = reference_cosine[upper]
    projected_values = projected_cosine[upper]
    correlation = float(np.corrcoef(reference_values, projected_values)[0, 1])
    errors = np.abs(reference_values - projected_values)
    return {
        "pairwise_cosine_correlation": correlation,
        "mean_absolute_cosine_error": float(np.mean(errors)),
        "max_absolute_cosine_error": float(np.max(errors)),
    }


def make_train_library(
    vectors: np.ndarray,
    labels: List[str],
    descriptions: List[str],
    dimension: int,
    spec: Dict[str, object],
) -> IntentLibrary:
    return IntentLibrary(
        vectors=vectors,
        labels=labels,
        descriptions=descriptions,
        metadata={
            "representation_type": "pretrained_semantic",
            "semantic_geometry": True,
            "embed_model": spec["encoder_model"],
            "model_revision": spec["encoder_revision"],
            "projection_dim": dimension,
            "projection_seed": int(spec["projection_seed"]),
            "postures": [
                infer_intent_posture(label, description)
                for label, description in zip(labels, descriptions)
            ],
        },
    )


def objective_targets(labels: List[str]) -> np.ndarray:
    return np.asarray([
        [resolve_intent_reward_profile(label)[key] for key in OBJECTIVE_KEYS]
        for label in labels
    ], dtype=np.float32)


def adapter_profile_errors(
    predictions: np.ndarray,
    targets: np.ndarray,
    queries: List[Dict[str, object]],
) -> Dict[str, object]:
    by_split = {}
    per_query_l2 = np.linalg.norm(predictions - targets, axis=1)
    per_query_mae = np.mean(np.abs(predictions - targets), axis=1)
    for split in sorted({str(query["split"]) for query in queries}):
        indices = [idx for idx, query in enumerate(queries) if query["split"] == split]
        by_split[split] = {
            "n_queries": len(indices),
            "mean_profile_l2_error": float(np.mean(per_query_l2[indices])),
            "mean_profile_mae": float(np.mean(per_query_mae[indices])),
        }
    return {"by_split": by_split}


def write_checksums(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    spec = json.loads(args.config.read_text(encoding="utf-8"))
    dimensions = [int(value) for value in spec["dimensions"]]
    if not dimensions or len(dimensions) != len(set(dimensions)) or min(dimensions) <= 0:
        raise ValueError("dimensions must be unique positive integers")
    if max(dimensions) > 384:
        raise ValueError("this frozen MiniLM geometry study supports dimensions up to 384")
    suite = load_generalization_suite(ROOT / str(spec["suite"]))
    output_dir = ROOT / "experiments" / str(spec["level"]) / str(spec["study_id"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing geometry study: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = dict(DEFAULT_INTENT_DESCRIPTIONS)
    train_labels = [str(label) for label in suite["train_labels"]]
    train_descriptions = [catalog[label] for label in train_labels]
    queries = list(suite["queries"])
    query_entries = [
        (str(query["canonical_label"]), str(query["description"])) for query in queries
    ]
    combined_entries = list(zip(train_labels, train_descriptions)) + query_entries
    original_dimension = 384
    base_library = IntentLibrary.create_pretrained(
        intent_dim=original_dimension,
        descriptions=combined_entries,
        model_name=str(spec["encoder_model"]),
        model_revision=str(spec["encoder_revision"]),
        projection_seed=int(spec["projection_seed"]),
    )
    reference = base_library.vectors
    n_train = len(train_labels)
    results = {}
    for dimension in dimensions:
        projected = _project_embeddings(
            reference,
            target_dim=dimension,
            seed=int(spec["projection_seed"]),
        )
        train_library = make_train_library(
            projected[:n_train],
            train_labels,
            train_descriptions,
            dimension,
            spec,
        )
        diagnostics = representation_retrieval_diagnostics(
            train_library,
            projected[n_train:],
            queries,
        )
        results[str(dimension)] = {
            "geometry_distortion": geometry_distortion(reference, projected),
            "retrieval": diagnostics,
        }

    train_embeddings = reference[:n_train]
    query_embeddings = reference[n_train:]
    train_targets = objective_targets(train_labels)
    query_targets = objective_targets([
        str(query["canonical_label"]) for query in queries
    ])
    target_mean = train_targets.mean(axis=0)
    target_scale = train_targets.std(axis=0)
    target_scale = np.where(target_scale < 1e-6, 1.0, target_scale)
    train_standardised = (train_targets - target_mean) / target_scale
    adapter_dim = int(spec.get("adapter_intent_dim", 64))
    objective_dim = len(OBJECTIVE_KEYS)
    if adapter_dim < objective_dim:
        raise ValueError("adapter_intent_dim is smaller than the objective feature count")
    semantic_dim = adapter_dim - objective_dim
    adapter_results = {}
    for ridge in [float(value) for value in spec.get("adapter_ridges", [0.01])]:
        coefficients = fit_dual_ridge(train_embeddings, train_standardised, ridge)
        train_objective_features = train_embeddings @ coefficients
        query_objective_features = query_embeddings @ coefficients
        query_predictions = query_objective_features * target_scale + target_mean
        ridge_result = {
            "profile_prediction": adapter_profile_errors(
                query_predictions, query_targets, queries
            ),
            "fusion": {},
        }
        semantic_all = (
            _project_embeddings(reference, semantic_dim, int(spec["projection_seed"]))
            if semantic_dim > 0 else np.empty((len(reference), 0), dtype=np.float32)
        )
        objective_all = _normalise_rows(np.concatenate(
            [train_objective_features, query_objective_features], axis=0
        ))
        for semantic_weight in [
            float(value) for value in spec.get("adapter_semantic_weights", [1.0])
        ]:
            parts = []
            if semantic_dim > 0 and semantic_weight > 0:
                parts.append(np.sqrt(semantic_weight) * semantic_all)
            parts.append(objective_all)
            fused = _normalise_rows(np.concatenate(parts, axis=1))
            fused_train = make_train_library(
                fused[:n_train], train_labels, train_descriptions, fused.shape[1], spec
            )
            ridge_result["fusion"][str(semantic_weight)] = (
                representation_retrieval_diagnostics(
                    fused_train, fused[n_train:], queries
                )
            )
        adapter_results[str(ridge)] = ridge_result
    results["objective_adapter_grid"] = {
        "intent_dim": adapter_dim,
        "ridges": adapter_results,
    }

    manifest = {
        "schema_version": 1,
        "study_id": spec["study_id"],
        "level": spec["level"],
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            name: importlib_metadata.version(name)
            for name in ("numpy", "sentence-transformers", "torch")
        },
        "config": spec,
        "status": "complete",
    }
    write_json(output_dir / "config.json", spec)
    write_json(output_dir / "results.json", results)
    write_json(output_dir / "manifest.json", manifest)
    lines = [
        f"# Result Card: {spec['study_id']}",
        "",
        "- Evidence level: `pilot` representation diagnostic",
        f"- Encoder: `{spec['encoder_model']}`",
        f"- Revision: `{spec['encoder_revision']}`",
        f"- Dimensions: `{', '.join(map(str, dimensions))}`",
        "- This study selects a representation dimension; it is not policy-performance evidence.",
        "",
    ]
    (output_dir / "RESULT_CARD.md").write_text("\n".join(lines), encoding="utf-8")
    write_checksums(output_dir)
    print(json.dumps({"output_dir": str(output_dir), "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
