from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_research_study import (  # noqa: E402
    build_resume_manifest,
    merge_resume_specs,
    validate_result_identity,
)


def study_spec(*variants: dict) -> dict:
    return {
        "schema_version": 1,
        "study_id": "resume_test",
        "level": "pilot",
        "seeds": [7, 11],
        "bootstrap_seed": 23,
        "environment": {"name": "test"},
        "training": {"episodes": 2},
        "intent": {"dim": 3},
        "evaluation": {"episodes": 2, "risk_tiers": {"easy": {}}},
        "variants": list(variants),
    }


class ResearchResumeTests(unittest.TestCase):
    def test_resume_adds_variants_without_changing_protocol(self):
        first = study_spec({"key": "a", "algorithm": "imappo"})
        second = study_spec(
            {"key": "a", "algorithm": "imappo"},
            {"key": "b", "algorithm": "ippo"},
        )
        second["treatment_key"] = "a"
        merged = merge_resume_specs(first, second)
        self.assertEqual([item["key"] for item in merged["variants"]], ["a", "b"])
        self.assertEqual(merged["treatment_key"], "a")

    def test_resume_rejects_protocol_change(self):
        first = study_spec({"key": "a", "algorithm": "imappo"})
        second = deepcopy(first)
        second["seeds"] = [7, 12]
        with self.assertRaisesRegex(ValueError, "'seeds'"):
            merge_resume_specs(first, second)

    def test_resume_rejects_variant_redefinition(self):
        first = study_spec({"key": "a", "algorithm": "imappo"})
        second = study_spec({"key": "a", "algorithm": "ippo"})
        with self.assertRaisesRegex(ValueError, "redefines"):
            merge_resume_specs(first, second)

    def test_resume_manifest_keeps_prior_invocation(self):
        spec = study_spec({"key": "a", "algorithm": "imappo"})
        existing = {
            "started_at_utc": "2026-01-01T00:00:00+00:00",
            "completed_at_utc": "2026-01-01T00:01:00+00:00",
            "git_commit": "abc",
            "command": ["first"],
            "config": spec,
            "status": "complete",
        }
        manifest = build_resume_manifest(existing, spec, spec, ["second"])
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(len(manifest["run_history"]), 2)
        self.assertEqual(manifest["run_history"][0]["command"], ["first"])
        self.assertEqual(manifest["run_history"][1]["command"], ["second"])

    def test_cached_result_identity_is_checked(self):
        variant = {"key": "a", "algorithm": "imappo"}
        validate_result_identity({"seed": 7, "variant": variant}, variant, 7)
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_result_identity({"seed": 11, "variant": variant}, variant, 7)


if __name__ == "__main__":
    unittest.main()
