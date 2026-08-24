from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_cross_scene_artifacts import write_report  # noqa: E402


def test_cross_scene_smoke_report_enforces_native_return_claim_boundary(tmp_path: Path):
    config = {
        "level": "smoke", "seeds": [7],
        "evaluation": {"episodes": 3},
    }
    audit = {"status": "valid"}
    main = [{
        "variant": "attention", "tier": "canonical", "mean": 1.0,
        "ci_low": 1.0, "ci_high": 1.0, "iqm": 1.0,
    }]
    paired = [{
        "baseline": "mappo", "tier": "canonical", "mean_difference": 0.0,
        "ci_low": 0.0, "ci_high": 0.0, "exact_p": 1.0, "holm_p": 1.0,
        "holm_reject_0_05": False,
    }]
    path = tmp_path / "REPORT.md"
    write_report(path, audit, config, main, paired)
    report = path.read_text(encoding="utf-8")
    assert "仅聚合 VMAS 场景原生 `episode_return`" in report
    assert "不能证明语言泛化" in report
    assert "禁止算法排序" in report
