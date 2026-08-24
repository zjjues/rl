"""Validate and hash-freeze a paper-scale independently reviewed preference JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from preference_dataset import (  # noqa: E402
    audit_formal_preference_dataset,
    load_preference_jsonl,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--consent-version", required=True)
    parser.add_argument("--min-records-per-class", type=int, default=50)
    parser.add_argument("--min-writers-per-split", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("frozen preference manifest already exists")
    records = load_preference_jsonl(args.records)
    audit = audit_formal_preference_dataset(
        records,
        min_records_per_class=args.min_records_per_class,
        min_writers_per_split=args.min_writers_per_split,
    )
    consent_versions = sorted({str(record["consent_version"]) for record in records})
    if consent_versions != [args.consent_version]:
        raise ValueError(
            f"record consent versions {consent_versions} do not match registered "
            f"version {args.consent_version!r}"
        )
    payload = {
        "schema_version": 1,
        "dataset_id": args.dataset_id,
        "frozen_at_utc": datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat().replace("+00:00", "Z"),
        "records_path": str(args.records),
        "records_sha256": sha256_file(args.records),
        "consent_version": args.consent_version,
        "validation_code_git_head": git_head(),
        "test_access_contract": (
            "Test records and labels must remain unavailable to model/threshold "
            "development after this freeze; access must be logged separately."
        ),
        "audit": audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
