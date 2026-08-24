"""Generate architecture-only VMAS tables, plot, report, and checksums."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rl-matplotlib-cache")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from research_artifact import validate_study_artifact  # noqa: E402
from research_metrics import resolve_metric_contract  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main_rows(summary: dict, variants: list[str], tiers: list[str]) -> list[dict]:
    rows = []
    for variant in variants:
        for tier in tiers:
            record = summary["variants"][variant]["risk_tiers"][tier][
                "episode_return"
            ]
            rows.append({
                "variant": variant,
                "tier": tier,
                "metric": "episode_return",
                "n": record["n"],
                "mean": record["mean"],
                "std": record["std"],
                "iqm": record["iqm"],
                "ci_low": record["mean_ci"]["low"],
                "ci_high": record["mean_ci"]["high"],
            })
    return rows


def paired_rows(summary: dict, baselines: list[str], tiers: list[str]) -> list[dict]:
    rows = []
    for baseline in baselines:
        for tier in tiers:
            record = summary["paired_comparisons"][baseline]["risk_tiers"][tier][
                "episode_return"
            ]
            rows.append({
                "baseline": baseline,
                "tier": tier,
                "metric": "episode_return",
                "mean_difference": record["mean_difference"],
                "ci_low": record["difference_ci"]["low"],
                "ci_high": record["difference_ci"]["high"],
                "effect_dz": record["standardized_effect_dz"],
                "exact_p": record["randomization_test"]["p_value"],
                "holm_p": record["holm_adjusted_p_value"],
                "holm_reject_0_05": record["holm_reject_0_05"],
            })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty cross-scene table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_episode_return(
    path: Path, rows: list[dict], variants: list[str], tiers: list[str]
) -> None:
    width = min(0.8 / max(len(variants), 1), 0.18)
    x = np.arange(len(tiers), dtype=float)
    figure, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    colors = dict(zip(variants, plt.cm.tab10(np.linspace(0.0, 0.8, len(variants)))))
    for index, variant in enumerate(variants):
        selected = [
            next(row for row in rows if row["variant"] == variant and row["tier"] == tier)
            for tier in tiers
        ]
        means = np.asarray([row["mean"] for row in selected])
        low = np.asarray([row["ci_low"] for row in selected])
        high = np.asarray([row["ci_high"] for row in selected])
        positions = x + (index - (len(variants) - 1) / 2.0) * width
        axis.bar(positions, means, width=width, color=colors[variant], label=variant)
        axis.errorbar(
            positions, means, yerr=np.vstack((means - low, high - means)),
            fmt="none", ecolor="black", capsize=2.5,
        )
    axis.set_xticks(x, tiers)
    axis.set_ylabel("Environment-native episode return (bootstrap 95% CI)")
    axis.set_title("Architecture-only cross-scene evaluation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def write_report(
    path: Path, audit: dict, config: dict, main: list[dict], paired: list[dict]
) -> None:
    level = str(config["level"])
    seed_count = len(config["seeds"])
    eval_episodes = int(config["evaluation"]["episodes"])
    if level == "smoke":
        notice = "> Smoke only：只验证执行、统计和产物管线，禁止算法排序。"
    elif level == "pilot":
        notice = "> Pilot calibration：用于估计成本与方差，不是冻结论文结论。"
    else:
        notice = "> Paper protocol：结论仍以 clean artifact、预注册比较和完整审计为准。"
    lines = [
        "# VMAS architecture-only 结果报告", "", notice, "",
        "## 机器合同", "",
        "- 仅聚合 VMAS 场景原生 `episode_return`。",
        "- 语言输入、偏好解码、UAV reward profile、动作掩码和 safety filter 全部关闭。",
        "- 本结果不能证明语言泛化、偏好准确率、UAV 安全迁移或 UAV task completion。",
        f"- Artifact：`{audit['status']}`；seeds={seed_count}；eval episodes/seed/tier={eval_episodes}。",
        "", "## Episode return", "",
        "| Variant | Tier | Mean [95% CI] | IQM |", "|---|---|---:|---:|",
    ]
    for row in main:
        lines.append(
            f"| {row['variant']} | {row['tier']} | {row['mean']:.6f} "
            f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}] | {row['iqm']:.6f} |"
        )
    lines.extend([
        "", "## 配对比较（treatment minus baseline）", "",
        "| Baseline | Tier | Δ mean [95% CI] | exact p | Holm p | Reject |",
        "|---|---|---:|---:|---:|:---:|",
    ])
    for row in paired:
        lines.append(
            f"| {row['baseline']} | {row['tier']} | {row['mean_difference']:.6f} "
            f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}] | "
            f"{row['exact_p']:.6f} | {row['holm_p']:.6f} | "
            f"{'yes' if row['holm_reject_0_05'] else 'no'} |"
        )
    lines.extend(["", "## 解释", ""])
    if level == "smoke":
        lines.append("- 单 seed 区间退化且随机化检验无充分信息，任何点值都不能解释为优势或等效。")
    else:
        rejected = sum(bool(row["holm_reject_0_05"]) for row in paired)
        lines.append(f"- Holm FWER 0.05 下拒绝 {rejected}/{len(paired)} 条主比较。")
        lines.append("- 未拒绝不能解释为等效；方向必须结合区间逐项报告。")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(output_dir: Path, study_dir: Path, config_path: Path) -> None:
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    (output_dir / "manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "source_study": str(study_dir.resolve()),
            "source_config": str(config_path.resolve()),
            "files": files,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    contract = resolve_metric_contract(config)
    if contract["valid_scope"] != "architecture_only":
        raise ValueError("cross-scene generator requires architecture_only scope")
    audit = validate_study_artifact(args.study_dir, config)
    if audit["status"] != "valid":
        raise ValueError(f"invalid source artifact: {audit['errors']}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite cross-scene paper artifacts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = read_json(args.study_dir / "summary.json")
    variants = [str(item["key"]) for item in config["variants"]]
    tiers = list(config["evaluation"]["risk_tiers"])
    baselines = [variant for variant in variants if variant != config["treatment_key"]]
    main = main_rows(summary, variants, tiers)
    paired = paired_rows(summary, baselines, tiers)
    write_csv(args.output_dir / "episode_return.csv", main)
    write_csv(args.output_dir / "paired_comparisons.csv", paired)
    plot_episode_return(
        args.output_dir / "episode_return.png", main, variants, tiers
    )
    write_report(args.output_dir / "REPORT.md", audit, config, main, paired)
    write_manifest(args.output_dir, args.study_dir, args.config)
    print(json.dumps({
        "status": "complete", "output_dir": str(args.output_dir),
        "rows": len(main), "paired_rows": len(paired),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
