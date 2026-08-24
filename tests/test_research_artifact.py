from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_artifact import validate_study_artifact  # noqa: E402


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


class ResearchArtifactTests(unittest.TestCase):
    def make_artifact(self, root: Path) -> dict:
        variant = {"key": "a", "algorithm": "imappo"}
        config = {
            "schema_version": 1,
            "study_id": "artifact_test",
            "level": "pilot",
            "seeds": [7, 11],
            "bootstrap_seed": 23,
            "environment": {"name": "test"},
            "training": {"episodes": 2},
            "intent": {"dim": 3},
            "variants": [variant],
            "evaluation": {"episodes": 2, "risk_tiers": {"easy": {}}},
        }
        write_json(root / "config.json", config)
        raw = {metric: [] for metric in (
            "collision_rate", "task_completion", "episode_return"
        )}
        for index, seed in enumerate(config["seeds"]):
            values = {
                "easy_collision_rate": 0.1 + index,
                "easy_task_completion": 0.8 - index * 0.1,
                "easy_episode_return": 1.0 + index,
            }
            write_json(
                root / "a" / f"seed_{seed}" / "result.json",
                {
                    "seed": seed,
                    "variant": variant,
                    "tier_metrics": {"easy": values},
                },
            )
            for metric in raw:
                raw[metric].append(values[f"easy_{metric}"])
        write_json(
            root / "summary.json",
            {
                "variants": {
                    "a": {
                        "risk_tiers": {
                            "easy": {
                                metric: {"raw": values} for metric, values in raw.items()
                            }
                        }
                    }
                }
            },
        )
        write_json(
            root / "manifest.json",
            {
                "config": config,
                "status": "complete",
                "git_status_short": "",
                "command": ["runner"],
            },
        )
        write_checksums(root)
        return config

    def test_valid_artifact_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_artifact(root)
            report = validate_study_artifact(root, config)
            self.assertEqual(report["status"], "valid")
            self.assertEqual(report["expected_result_count"], 2)

    def test_summary_tampering_is_detected_even_without_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_artifact(root)
            summary = json.loads((root / "summary.json").read_text())
            summary["variants"]["a"]["risk_tiers"]["easy"][
                "collision_rate"
            ]["raw"][0] = 999.0
            write_json(root / "summary.json", summary)
            report = validate_study_artifact(root, config, verify_checksums=False)
            self.assertEqual(report["status"], "invalid")
            self.assertTrue(any("summary raw values" in item for item in report["errors"]))

    def test_checksum_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_artifact(root)
            (root / "a" / "seed_7" / "result.json").write_text("{}")
            report = validate_study_artifact(root, config)
            self.assertEqual(report["status"], "invalid")
            self.assertTrue(any("checksum mismatch" in item for item in report["errors"]))

    def test_paper_artifact_requires_per_seed_resource_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_artifact(root)
            config["level"] = "paper"
            config["evaluation"]["episodes"] = 100
            write_json(root / "config.json", config)
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["config"] = config
            write_json(root / "manifest.json", manifest)
            write_checksums(root)
            report = validate_study_artifact(root, config)
            self.assertEqual(report["status"], "invalid")
            self.assertTrue(
                any("lacks resource_audit" in item for item in report["errors"])
            )

    def test_happo_artifact_requires_and_accepts_independent_actor_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_artifact(root)
            happo_variant = {
                "key": "a",
                "algorithm": "happo",
                "critic_mode": "mlp",
                "intent_source": "none",
                "use_action_mask": False,
                "policy_mode": "direct",
                "safety_filter_mode": "none",
                "actor_parameter_sharing": "independent",
                "update_scheme": "random_sequential_likelihood_factor",
            }
            config["environment"]["n_agents"] = 8
            config["variants"] = [happo_variant]
            write_json(root / "config.json", config)
            for seed in config["seeds"]:
                result_path = root / "a" / f"seed_{seed}" / "result.json"
                result = json.loads(result_path.read_text())
                result["variant"] = happo_variant
                result["algorithm_implementation"] = {
                    "algorithm": "happo",
                    "actor_parameter_sharing": "independent",
                    "actor_count": 8,
                    "update_scheme": "random_sequential_likelihood_factor",
                    "critic": "centralized_mlp",
                }
                write_json(result_path, result)
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["config"] = config
            write_json(root / "manifest.json", manifest)
            write_checksums(root)
            report = validate_study_artifact(root, config)
            self.assertEqual(report["status"], "valid", report["errors"])


if __name__ == "__main__":
    unittest.main()
