from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_metrics import resolve_metric_contract  # noqa: E402


def test_default_uav_contract_preserves_safety_task_primary_family():
    contract = resolve_metric_contract({})
    assert contract["primary_metrics"] == ("collision_rate", "task_completion")
    assert "episode_return" in contract["artifact_metrics"]


def test_architecture_only_contract_accepts_native_return_only():
    contract = resolve_metric_contract({
        "reporting": {
            "valid_scope": "architecture_only",
            "primary_metrics": ["episode_return"],
            "summary_metrics": ["episode_return"],
            "artifact_metrics": ["episode_return"],
        }
    })
    assert contract["summary_metric_directions"] == (("episode_return", False),)


def test_architecture_only_contract_rejects_invented_uav_metrics():
    with pytest.raises(ValueError, match="only native episode_return"):
        resolve_metric_contract({
            "reporting": {
                "valid_scope": "architecture_only",
                "primary_metrics": ["episode_return"],
                "summary_metrics": ["episode_return", "task_completion"],
                "artifact_metrics": ["episode_return"],
            }
        })


def test_primary_metrics_must_be_audited_and_summarized():
    with pytest.raises(ValueError, match="included in artifact_metrics"):
        resolve_metric_contract({
            "reporting": {
                "primary_metrics": ["episode_return"],
                "summary_metrics": ["episode_return"],
                "artifact_metrics": ["collision_rate"],
            }
        })
