"""Train a development-only preference relevance gate with disjoint OOD splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from external_language_corpus import load_jsonl, sha256_file  # noqa: E402
from intent_generalization import load_generalization_suite  # noqa: E402
from intent_semantic_encoder import DEFAULT_INTENT_DESCRIPTIONS  # noqa: E402
from objective_semantic_adapter import (  # noqa: E402
    NEUTRAL_OBJECTIVE_PROTOTYPES,
    _load_sentence_transformer,
    augmented_objective_prototypes,
)
from preference_relevance_gate import (  # noqa: E402
    fit_logistic_relevance_gate,
    relevance_metrics,
)


def normalise(text: str) -> str:
    return " ".join(str(text).lower().split())


def deterministic_records(records, maximum: int, seed: int):
    if maximum <= 0:
        raise ValueError("sample maximum must be positive")
    if len(records) <= maximum:
        return list(records)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(records), size=maximum, replace=False))
    return [records[int(index)] for index in indices]


def id_hash(records) -> str:
    value = "\n".join(str(record["id"]) for record in records).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def encode(model, texts, batch_size):
    return model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-train", type=Path, required=True)
    parser.add_argument("--negative-dev", type=Path, required=True)
    parser.add_argument("--negative-development-eval", type=Path, required=True)
    parser.add_argument("--generalization-suite", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--negative-train-max", type=int, default=4096)
    parser.add_argument("--sample-seed", type=int, default=20260820)
    parser.add_argument("--max-dev-false-accept", type=float, default=0.05)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    negative_train_all = load_jsonl(args.negative_train)
    negative_dev = load_jsonl(args.negative_dev)
    negative_eval = load_jsonl(args.negative_development_eval)
    negative_train = deterministic_records(
        negative_train_all, args.negative_train_max, args.sample_seed
    )

    positive_train = [description for _, description in DEFAULT_INTENT_DESCRIPTIONS]
    positive_train.extend(text for _, text, _ in augmented_objective_prototypes())
    positive_train.extend(NEUTRAL_OBJECTIVE_PROTOTYPES)
    train_norm = {normalise(text) for text in positive_train}
    suite = load_generalization_suite(args.generalization_suite)
    positive_dev = [
        str(query["description"]) for query in suite["queries"]
        if str(query["split"]) != "seen"
        and normalise(str(query["description"])) not in train_norm
    ]
    if len(positive_dev) < 10:
        raise ValueError("too few non-overlapping positive development queries")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    intent = config["intent"]
    model_name = str(intent["encoder_model"])
    model_revision = str(intent["encoder_revision"])
    if not model_revision:
        raise ValueError("relevance gate requires a pinned encoder revision")
    model = _load_sentence_transformer(model_name, model_revision, args.device)
    positive_train_embeddings = encode(model, positive_train, args.batch_size)
    negative_train_embeddings = encode(
        model, [record["text"] for record in negative_train], args.batch_size
    )
    negative_dev_embeddings = encode(
        model, [record["text"] for record in negative_dev], args.batch_size
    )
    positive_dev_embeddings = encode(model, positive_dev, args.batch_size)
    negative_eval_embeddings = encode(
        model, [record["text"] for record in negative_eval], args.batch_size
    )

    gate = fit_logistic_relevance_gate(
        positive_train_embeddings,
        negative_train_embeddings,
        negative_calibration_embeddings=negative_dev_embeddings,
        max_false_accept_rate=args.max_dev_false_accept,
        encoder_model=model_name,
        encoder_revision=model_revision,
        c=args.logistic_c,
        random_state=args.sample_seed,
        metadata={
            "evidence_level": "development",
            "positive_training_source": "canonical catalog + templated prototypes",
            "positive_development_source": str(args.generalization_suite),
            "negative_training_source": str(args.negative_train),
            "negative_calibration_source": str(args.negative_dev),
            "negative_development_eval_source": str(args.negative_development_eval),
            "final_blind_test": False,
        },
    )
    gate.save(args.gate_output)
    audit = {
        "schema_version": 1,
        "audit_id": "preference_relevance_gate_aerialvln_development_v1",
        "evidence_level": "development",
        "claim_boundary": (
            "Positive language is developer-authored and val_unseen was inspected "
            "before gate development; this is not a final blind language result."
        ),
        "inputs": {
            "negative_train_sha256": sha256_file(args.negative_train),
            "negative_dev_sha256": sha256_file(args.negative_dev),
            "negative_development_eval_sha256": sha256_file(
                args.negative_development_eval
            ),
            "negative_train_sample_ids_sha256": id_hash(negative_train),
            "negative_dev_ids_sha256": id_hash(negative_dev),
            "negative_development_eval_ids_sha256": id_hash(negative_eval),
            "positive_train_texts_sha256": hashlib.sha256(
                "\n".join(positive_train).encode("utf-8")
            ).hexdigest(),
            "positive_dev_texts_sha256": hashlib.sha256(
                "\n".join(positive_dev).encode("utf-8")
            ).hexdigest(),
        },
        "counts": {
            "negative_train_available": len(negative_train_all),
            "negative_train_used": len(negative_train),
            "negative_dev": len(negative_dev),
            "negative_development_eval": len(negative_eval),
            "positive_train": len(positive_train),
            "positive_dev": len(positive_dev),
        },
        "gate": gate.to_payload(),
        "development_metrics": relevance_metrics(
            gate,
            positive_embeddings=positive_dev_embeddings,
            negative_embeddings=negative_dev_embeddings,
        ),
        "inspected_val_unseen_metrics": relevance_metrics(
            gate, negative_embeddings=negative_eval_embeddings
        ),
        "gate_sha256": sha256_file(args.gate_output),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "gate_output": str(args.gate_output),
        "audit_output": str(args.audit_output),
        "threshold": gate.threshold,
        "development_metrics": audit["development_metrics"],
        "inspected_val_unseen_metrics": audit["inspected_val_unseen_metrics"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
