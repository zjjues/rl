"""Train and audit a frozen-encoder preference classifier from human JSONL labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from preference_dataset import (  # noqa: E402
    load_preference_jsonl,
    preference_class,
    validate_preference_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--encoder", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def main() -> None:
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    args = parse_args()
    records = load_preference_jsonl(args.dataset)
    audit = validate_preference_records(
        records, require_all_classes=True, require_annotator_disjoint=True
    )
    model = SentenceTransformer(args.encoder, revision=args.revision, device="cpu")
    embeddings = model.encode(
        [str(record["text"]) for record in records],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    labels = np.asarray([preference_class(record) for record in records])
    splits = np.asarray([str(record["split"]) for record in records])
    train_mask = np.isin(splits, ["train", "dev"])
    test_mask = splits == "test"
    if not test_mask.any():
        raise ValueError("dataset requires a non-empty test split")
    classifier = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=args.seed,
        solver="lbfgs",
    )
    classifier.fit(embeddings[train_mask], labels[train_mask])
    predictions = classifier.predict(embeddings[test_mask])
    classes = classifier.classes_.tolist()
    metrics = {
        "test_count": int(test_mask.sum()),
        "accuracy": float(accuracy_score(labels[test_mask], predictions)),
        "macro_f1": float(f1_score(
            labels[test_mask], predictions, labels=classes, average="macro",
            zero_division=0,
        )),
        "classes": classes,
        "confusion_matrix": confusion_matrix(
            labels[test_mask], predictions, labels=classes
        ).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        coefficients=classifier.coef_.astype(np.float32),
        intercept=classifier.intercept_.astype(np.float32),
        classes=np.asarray(classes),
        encoder=np.asarray([args.encoder]),
        revision=np.asarray([args.revision]),
    )
    report = {
        "schema_version": 1,
        "dataset": str(args.dataset.resolve()),
        "model": str(args.output.resolve()),
        "encoder": args.encoder,
        "revision": args.revision,
        "seed": args.seed,
        "audit": audit,
        "metrics": metrics,
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
