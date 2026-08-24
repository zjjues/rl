from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_research_study import (  # noqa: E402
    build_resume_manifest,
    expected_result_path,
    merge_resume_specs,
    resolve_run_selection,
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
    def test_run_selection_supports_safe_chunks(self):
        spec = study_spec(
            {"key": "full", "algorithm": "imappo"},
            {"key": "baseline", "algorithm": "ippo"},
        )
        variants, seeds = resolve_run_selection(spec, "baseline", "11")
        self.assertEqual(variants, {"baseline"})
        self.assertEqual(seeds, {11})
        with self.assertRaisesRegex(ValueError, "unknown selected variants"):
            resolve_run_selection(spec, "missing", None)
        with self.assertRaisesRegex(ValueError, "unknown selected seeds"):
            resolve_run_selection(spec, None, "23")

    def test_expected_result_path_is_stable(self):
        path = expected_result_path(Path("study"), "full", 7)
        self.assertEqual(path.as_posix(), "study/full/seed_7/result.json")

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
