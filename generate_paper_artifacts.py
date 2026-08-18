"""Generate audited pilot tables and figures from a research study artifact."""

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
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_artifact import validate_study_artifact  # noqa: E402


PRIMARY_METRICS = ("collision_rate", "task_completion")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main_rows(summary: dict, variants: list[str], tiers: list[str]) -> list[dict]:
    rows = []
    for variant in variants:
        for tier in tiers:
            for metric in PRIMARY_METRICS:
                record = summary["variants"][variant]["risk_tiers"][tier][metric]
                rows.append({
                    "variant": variant,
                    "tier": tier,
                    "metric": metric,
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
            for metric in PRIMARY_METRICS:
                record = summary["paired_comparisons"][baseline]["risk_tiers"][tier][metric]
                rows.append({
                    "baseline": baseline,
                    "tier": tier,
                    "metric": metric,
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_main_metrics(
    path: Path, rows: list[dict], variants: list[str], tiers: list[str]
) -> None:
    colors = dict(zip(variants, plt.cm.tab10(np.linspace(0.0, 0.75, len(variants)))))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    width = 0.18
    x = np.arange(len(tiers), dtype=float)
    for axis, metric, title in zip(
        axes,
        PRIMARY_METRICS,
        ("Collision rate (lower is better)", "Task completion (higher is better)"),
    ):
        for index, variant in enumerate(variants):
            selected = [
                row for tier in tiers for row in rows
                if row["variant"] == variant
                and row["tier"] == tier
                and row["metric"] == metric
            ]
            means = np.asarray([row["mean"] for row in selected])
            low = np.asarray([row["ci_low"] for row in selected])
            high = np.asarray([row["ci_high"] for row in selected])
            positions = x + (index - (len(variants) - 1) / 2.0) * width
            axis.bar(
                positions,
                means,
                width=width,
                color=colors[variant],
                label=variant.upper(),
                alpha=0.88,
            )
            axis.errorbar(
                positions,
                means,
                yerr=np.vstack((means - low, high - means)),
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=2.5,
            )
        axis.set_xticks(x, [tier.capitalize() for tier in tiers])
        axis.set_title(title)
        axis.set_ylabel("Mean with bootstrap 95% CI")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=2, frameon=False)
    figure.suptitle("UAV MARL architecture pilot: 10 paired seeds, 50 eval episodes/tier/seed")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_paired_effects(path: Path, rows: list[dict], tiers: list[str]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 6.2), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        PRIMARY_METRICS,
        ("Collision-rate difference", "Task-completion difference"),
    ):
        selected = [row for row in rows if row["metric"] == metric]
        labels = [f"{row['baseline'].upper()} / {row['tier']}" for row in selected]
        y = np.arange(len(selected))
        means = np.asarray([row["mean_difference"] for row in selected])
        low = np.asarray([row["ci_low"] for row in selected])
        high = np.asarray([row["ci_high"] for row in selected])
        colors = ["#1b9e77" if row["holm_reject_0_05"] else "#7570b3" for row in selected]
        axis.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
        for index, row in enumerate(selected):
            axis.errorbar(
                means[index],
                y[index],
                xerr=[[means[index] - low[index]], [high[index] - means[index]]],
                fmt="o",
                color=colors[index],
                capsize=3,
            )
            axis.text(
                0.98,
                y[index],
                f"Holm p={row['holm_p']:.3g}",
                fontsize=7,
                ha="right",
                va="center",
                transform=axis.get_yaxis_transform(),
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65},
            )
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_xlabel("I-MAPPO minus baseline (bootstrap 95% CI)")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle("Paired primary effects; green denotes Holm FWER rejection at 0.05")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def write_report(
    path: Path,
    audit: dict,
    config: dict,
    main: list[dict],
    paired: list[dict],
) -> None:
    lines = [
        "# UAV I-MAPPO 架构先导实验统计报告",
        "",
        "> 自动生成；证据等级为 pilot，不是 frozen paper result。",
        "",
        "## Artifact 审计",
        "",
        f"- 状态：`{audit['status']}`",
        f"- 变体/种子/结果：{audit['variant_count']} / {audit['seed_count']} / {audit['expected_result_count']}",
        f"- checksum 条目：{audit['checksum_entry_count']}",
        f"- 警告：{'; '.join(audit['warnings']) if audit['warnings'] else '无'}",
        "",
        "## 均值与 95% bootstrap CI",
        "",
        "| Variant | Tier | Collision | Task completion |",
        "|---|---|---:|---:|",
    ]
    for variant in [item["key"] for item in config["variants"]]:
        for tier in config["evaluation"]["risk_tiers"]:
            collision = next(
                row for row in main
                if row["variant"] == variant and row["tier"] == tier
                and row["metric"] == "collision_rate"
            )
            task = next(
                row for row in main
                if row["variant"] == variant and row["tier"] == tier
                and row["metric"] == "task_completion"
            )
            lines.append(
                f"| {variant} | {tier} | {collision['mean']:.4f} "
                f"[{collision['ci_low']:.4f}, {collision['ci_high']:.4f}] | "
                f"{task['mean']:.4f} [{task['ci_low']:.4f}, {task['ci_high']:.4f}] |"
            )
    lines.extend([
        "",
        "## I-MAPPO 配对主比较",
        "",
        "| Baseline | Tier | Metric | Δ mean | 95% CI | exact p | Holm p | Reject |",
        "|---|---|---|---:|---:|---:|---:|:---:|",
    ])
    for row in paired:
        lines.append(
            f"| {row['baseline']} | {row['tier']} | {row['metric']} | "
            f"{row['mean_difference']:.4f} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
            f"{row['exact_p']:.6f} | {row['holm_p']:.6f} | "
            f"{'yes' if row['holm_reject_0_05'] else 'no'} |"
        )
    lines.extend([
        "",
        "## 可支持的结论",
        "",
        "- I-MAPPO 相对 MAPPO 的 collision/task 主比较在 Holm 校正后均不显著，不能主张 attention critic + action mask 优于 MAPPO。",
        "- I-MAPPO 相对 IPPO 在 easy/medium 碰撞率更低，但任务完成率也稳定更低，属于安全—任务权衡。",
        "- I-MAPPO 相对 MATD3 在 easy/medium 碰撞率更低，但任务完成率更低；hard 碰撞差异不稳定。",
        "- 本实验所有方法使用 one-hot intent 且关闭 intent reward，只能回答架构问题，不能证明自然语言/语义意图带来优势。",
        "",
        "## 投稿限制",
        "",
        "- 历史训练运行来自 dirty Git worktree；当前 artifact 可验证但不能升级为 frozen paper evidence。",
        "- 每 seed/tier 仅 50 个评估回合，低于工程规定的 paper 门槛 100。",
        "- 尚缺语义方法的因果消融、独立语言数据、跨场景部署和 HIL/实机证据。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(output_dir: Path, source_dir: Path, source_config: Path) -> None:
    files = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "source_study": str(source_dir.resolve()),
        "source_config": str(source_config.resolve()),
        "files": files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    audit = validate_study_artifact(args.study_dir, config)
    if audit["status"] != "valid":
        raise RuntimeError(f"refusing to plot invalid artifact: {audit['errors']}")
    summary = read_json(args.study_dir / "summary.json")
    variants = [str(item["key"]) for item in config["variants"]]
    tiers = list(config["evaluation"]["risk_tiers"])
    baselines = [key for key in summary["paired_comparisons"]]
    main_table = main_rows(summary, variants, tiers)
    paired_table = paired_rows(summary, baselines, tiers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "main_results.csv", main_table)
    write_csv(args.output_dir / "paired_primary_results.csv", paired_table)
    plot_main_metrics(args.output_dir / "figure_main_metrics.png", main_table, variants, tiers)
    plot_paired_effects(args.output_dir / "figure_paired_primary.png", paired_table, tiers)
    write_report(
        args.output_dir / "PILOT_STATISTICAL_REPORT.md",
        audit,
        config,
        main_table,
        paired_table,
    )
    write_manifest(args.output_dir, args.study_dir, args.config)
    print(json.dumps({
        "status": "complete",
        "output_dir": str(args.output_dir),
        "files": sorted(path.name for path in args.output_dir.iterdir()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
