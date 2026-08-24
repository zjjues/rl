from __future__ import annotations

import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_paper_artifacts import write_report  # noqa: E402


def test_smoke_report_never_emits_pilot_effect_claims():
    config = {
        "level": "smoke",
        "seeds": [7],
        "variants": [{"key": "imappo"}, {"key": "happo"}],
        "evaluation": {"episodes": 3, "risk_tiers": {"hard": {}}},
    }
    audit = {
        "status": "valid",
        "variant_count": 2,
        "seed_count": 1,
        "expected_result_count": 2,
        "checksum_entry_count": 6,
        "warnings": [],
    }
    main = []
    for variant in ("imappo", "happo"):
        for metric, value in (("collision_rate", 0.5), ("task_completion", 0.6)):
            main.append(
                {
                    "variant": variant,
                    "tier": "hard",
                    "metric": metric,
                    "mean": value,
                    "ci_low": value,
                    "ci_high": value,
                }
            )
    paired = [
        {
            "baseline": "happo",
            "tier": "hard",
            "metric": "collision_rate",
            "mean_difference": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "exact_p": 1.0,
            "holm_p": 1.0,
            "holm_reject_0_05": False,
        }
    ]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "report.md"
        write_report(path, audit, config, main, paired)
        report = path.read_text(encoding="utf-8")
    assert "证据等级为 smoke" in report
    assert "禁止效果推断" in report
    assert "每 seed/tier 3 个评估回合" in report
    assert "安全—任务权衡" not in report
    assert "证据等级为 pilot" not in report
