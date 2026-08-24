from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preference_dataset import (  # noqa: E402
    PREFERENCE_CLASSES,
    audit_formal_preference_dataset,
)
from submission_readiness import audit_submission_readiness  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_checksums(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n")


class SubmissionReadinessTests(unittest.TestCase):
    def test_json_contract_requires_every_registered_condition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(
                root / "evidence.json",
                {"decision": {"outcome": "pass"}, "count": 10, "error": 0.01},
            )
            spec = {
                "schema_version": 1,
                "audit_id": "test",
                "gates": [{
                    "key": "external",
                    "title": "External evidence",
                    "kind": "json_contract",
                    "path": "evidence.json",
                    "equals": {"decision.outcome": "pass"},
                    "at_least": {"count": 10},
                    "at_most": {"error": 0.02},
                }],
            }
            report = audit_submission_readiness(root, spec)
            self.assertEqual(report["status"], "ready")
            payload = json.loads((root / "evidence.json").read_text())
            payload["error"] = 0.2
            write_json(root / "evidence.json", payload)
            failed = audit_submission_readiness(root, spec)
            self.assertEqual(failed["status"], "not_ready")
            self.assertIn("external", failed["blocking_gates"])

    def test_missing_noncritical_gate_does_not_create_false_blocker(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = {
                "schema_version": 1,
                "gates": [{
                    "key": "optional",
                    "title": "Optional evidence",
                    "kind": "json_contract",
                    "critical": False,
                    "path": "missing.json",
                    "required_fields": ["status"],
                }],
            }
            report = audit_submission_readiness(directory, spec)
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["gates"][0]["status"], "unmet")

    def test_formal_preference_gate_recomputes_audit_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for split in ("train", "dev", "test"):
                for index, label in enumerate(PREFERENCE_CLASSES):
                    objective, polarity = (
                        ("neutral", "neutral") if label == "neutral" else label.split(":")
                    )
                    records.append({
                        "id": f"{split}-{index}",
                        "text": f"Preference statement {split} number {index}",
                        "objective": objective,
                        "polarity": polarity,
                        "annotator_id": f"writer-{split}",
                        "source": "independent-human-collection",
                        "split": split,
                        "elicited_objective": objective,
                        "elicited_polarity": polarity,
                        "reviewer_id": f"reviewer-{split}",
                        "reviewer_objective": objective,
                        "reviewer_polarity": polarity,
                        "decision": "agreed",
                        "collection_batch": "batch-1",
                        "prompt_id": f"prompt-{index}",
                        "language": "en",
                        "consent_version": "v1",
                    })
            records_path = root / "records.jsonl"
            records_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            audit = audit_formal_preference_dataset(
                records, min_records_per_class=3, min_writers_per_split=1
            )
            write_json(root / "manifest.json", {
                "dataset_id": "formal-v1",
                "records_path": "records.jsonl",
                "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
                "consent_version": "v1",
                "validation_code_git_head": "a" * 40,
                "test_access_contract": "blind test access is logged once",
                "audit": audit,
            })
            spec = {
                "schema_version": 1,
                "gates": [{
                    "key": "preferences",
                    "title": "Formal preferences",
                    "kind": "formal_preference_dataset",
                    "manifest": "manifest.json",
                    "min_records_per_class": 3,
                    "min_writers_per_split": 1,
                }],
            }
            self.assertEqual(audit_submission_readiness(root, spec)["status"], "ready")
            records_path.write_text(records_path.read_text() + "{}\n", encoding="utf-8")
            failed = audit_submission_readiness(root, spec)
            self.assertEqual(failed["status"], "not_ready")
            self.assertTrue(any("SHA-256" in item for item in failed["gates"][0]["reasons"]))

    def test_valid_study_artifact_is_not_replaced_by_file_existence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = root / "study"
            variant = {"key": "a", "algorithm": "imappo"}
            config = {
                "schema_version": 1,
                "study_id": "study",
                "level": "pilot",
                "seeds": [7],
                "bootstrap_seed": 3,
                "environment": {"name": "test"},
                "training": {"episodes": 2},
                "intent": {"dim": 3},
                "variants": [variant],
                "evaluation": {"episodes": 2, "risk_tiers": {"easy": {}}},
            }
            write_json(root / "config.json", config)
            write_json(study / "config.json", config)
            values = {
                "easy_collision_rate": 0.1,
                "easy_task_completion": 0.8,
                "easy_episode_return": 1.0,
            }
            write_json(study / "a" / "seed_7" / "result.json", {
                "seed": 7,
                "variant": variant,
                "tier_metrics": {"easy": values},
            })
            write_json(study / "summary.json", {
                "variants": {"a": {"risk_tiers": {"easy": {
                    "collision_rate": {"raw": [0.1]},
                    "task_completion": {"raw": [0.8]},
                    "episode_return": {"raw": [1.0]},
                }}}},
            })
            write_json(study / "manifest.json", {
                "config": config,
                "status": "complete",
                "git_status_short": "",
                "command": ["runner"],
            })
            write_checksums(study)
            spec = {
                "schema_version": 1,
                "gates": [{
                    "key": "study",
                    "title": "Study",
                    "kind": "study_artifact",
                    "config": "config.json",
                    "study_dir": "study",
                    "required_level": "pilot",
                    "minimum_result_count": 1,
                }],
            }
            self.assertEqual(audit_submission_readiness(root, spec)["status"], "ready")
            hash_locked = json.loads(json.dumps(spec))
            hash_locked["gates"][0]["config_sha256"] = "0" * 64
            hash_failed = audit_submission_readiness(root, hash_locked)
            self.assertEqual(hash_failed["status"], "not_ready")
            self.assertTrue(any(
                "config SHA-256 mismatch" in reason
                for reason in hash_failed["gates"][0]["reasons"]
            ))
            (study / "a" / "seed_7" / "result.json").write_text("{}", encoding="utf-8")
            failed = audit_submission_readiness(root, spec)
            self.assertEqual(failed["status"], "not_ready")
            self.assertEqual(failed["gates"][0]["artifact_status"], "invalid")

    def test_spec_rejects_duplicate_gate_keys(self):
        spec = {
            "schema_version": 1,
            "gates": [
                {
                    "key": "same", "kind": "json_contract", "path": "a.json",
                    "required_fields": ["status"],
                },
                {
                    "key": "same", "kind": "json_contract", "path": "b.json",
                    "required_fields": ["status"],
                },
            ],
        }
        with self.assertRaisesRegex(ValueError, "unique"):
            audit_submission_readiness(".", spec)

    def test_spec_rejects_missing_kind_specific_path(self):
        spec = {
            "schema_version": 1,
            "gates": [{"key": "broken", "kind": "json_contract"}],
        }
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            audit_submission_readiness(".", spec)

    def test_evidence_paths_cannot_escape_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = {
                "schema_version": 1,
                "gates": [{
                    "key": "escape",
                    "kind": "json_contract",
                    "path": "../outside.json",
                    "required_fields": ["status"],
                }],
            }
            with self.assertRaisesRegex(ValueError, "escapes repository"):
                audit_submission_readiness(directory, spec)

    def test_json_contract_cannot_pass_on_file_existence_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            write_json(Path(directory) / "empty.json", {})
            spec = {
                "schema_version": 1,
                "gates": [{
                    "key": "existence-only",
                    "kind": "json_contract",
                    "path": "empty.json",
                }],
            }
            with self.assertRaisesRegex(ValueError, "not file existence alone"):
                audit_submission_readiness(directory, spec)


if __name__ == "__main__":
    unittest.main()
