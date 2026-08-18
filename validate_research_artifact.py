"""Validate a study directory against its registered configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_artifact import validate_study_artifact  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args()
    expected = (
        json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
    )
    report = validate_study_artifact(
        args.study_dir,
        expected,
        verify_checksums=not args.skip_checksums,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if report["status"] != "valid":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
