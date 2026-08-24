"""Audit all pre-registered evidence gates required for a submission-ready claim."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from submission_readiness import audit_submission_readiness  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="exit with status 2 unless every critical gate is met",
    )
    args = parser.parse_args()
    spec = json.loads(args.config.read_text(encoding="utf-8"))
    report = audit_submission_readiness(ROOT, spec)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, args.output)
    print(payload, end="")
    return 2 if args.require_ready and report["status"] != "ready" else 0


if __name__ == "__main__":
    raise SystemExit(main())
