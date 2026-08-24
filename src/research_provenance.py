"""Canonical fingerprints for registered studies and per-seed executions."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def registered_study_protocol_fingerprint(spec: Mapping[str, object]) -> str:
    """Hash the complete registered study protocol."""

    return canonical_json_sha256(spec)


def registered_result_protocol_fingerprint(
    spec: Mapping[str, object], variant: Mapping[str, object], seed: int
) -> str:
    """Hash the study, exact variant definition, and seed for one result."""

    return canonical_json_sha256(
        {"spec": spec, "variant": variant, "seed": int(seed)}
    )
