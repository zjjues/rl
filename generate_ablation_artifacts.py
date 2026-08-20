"""Generate audited tables, figures, and a report for a contracted ablation study."""

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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_mean_rows(summary: dict, config: dict) -> list[dict]:
    rows = []
    for variant in config["variants"]:
        key = str(variant["key"])
        for tier in config["evaluation"]["risk_tiers"]:
            for metric in ("collision_rate", "task_completion", "episode_return"):
                record = summary["variants"][key]["risk_tiers"][tier][metric]
                rows.append(
                    {
                        "variant": key,
                        "tier": tier,
                        "metric": metric,
                        "n": record["n"],
                        "mean": record["mean"],
                        "std": record["std"],
                        "iqm": record["iqm"],
                        "ci_low": record["mean_ci"]["low"],
                        "ci_high": record["mean_ci"]["high"],
                    }
                )
    return rows


def build_comparison_rows(summary: dict) -> list[dict]:
    rows = []
    for variant, comparison in summary["paired_comparisons"].items():
        for tier in comparison["primary_tiers"]:
            for metric in comparison["primary_metrics"]:
                record = comparison["risk_tiers"][tier][metric]
                rows.append(
                    {
                        "reference": comparison["reference_key"],
                        "variant": variant,
                        "factor": comparison["factor"],
                        "changed_fields": ";".join(comparison["changed_fields"]),
                        "tier": tier,
                        "metric": metric,
                        "direction": "variant_minus_reference",
                        "mean_difference": record["mean_difference"],
                        "ci_low": record["difference_ci"]["low"],
                        "ci_high": record["difference_ci"]["high"],
                        "effect_dz": record["standardized_effect_dz"],
                        "exact_p": record["randomization_test"]["p_value"],
                        "holm_p": record["holm_adjusted_p_value"],
                        "holm_reject_0_05": record["holm_reject_0_05"],
                    }
                )
    return rows


def build_resource_rows(study_dir: Path, config: dict) -> list[dict]:
    rows = []
    for variant in config["variants"]:
        key = str(variant["key"])
        for seed in config["seeds"]:
            result = read_json(study_dir / key / f"seed_{seed}" / "result.json")
            resource = result.get("resource_audit", {})
            models = resource.get("model_parameters", {})
            rows.append(
                {
                    "variant": key,
                    "seed": seed,
                    "wall_time_seconds": resource.get("wall_time_seconds"),
                    "cuda_peak_allocated_mb": resource.get("cuda_peak_allocated_mb"),
                    "text_model_cache_entries": resource.get(
                        "frozen_text_model_cache", {}
                    ).get("entry_count"),
                    "actor_trainable_parameters": models.get("actor", {}).get(
                        "trainable"
                    ),
                    "critic_trainable_parameters": models.get("critic", {}).get(
                        "trainable"
                    ),
                    "potential_trainable_parameters": models.get("potential", {}).get(
                        "trainable"
                    ),
                }
            )
    return rows


def plot_comparisons(path: Path, rows: list[dict], evidence_level: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("collision_rate", "task_completion"),
        ("Collision-rate effect", "Task-completion effect"),
    ):
        selected = [row for row in rows if row["metric"] == metric]
        labels = [
            f"{row['factor']}\n{row['variant']} vs {row['reference']}"
            for row in selected
        ]
        positions = np.arange(len(selected))
        means = np.asarray([row["mean_difference"] for row in selected])
        lows = np.asarray([row["ci_low"] for row in selected])
        highs = np.asarray([row["ci_high"] for row in selected])
        colors = [
            "#1b9e77" if row["holm_reject_0_05"] else "#7570b3"
            for row in selected
        ]
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        for index, row in enumerate(selected):
            axis.errorbar(
                means[index],
                positions[index],
                xerr=[
                    [means[index] - lows[index]],
                    [highs[index] - means[index]],
                ],
                fmt="o",
                color=colors[index],
                capsize=3,
            )
            axis.text(
                0.98,
                positions[index],
                f"Holm p={row['holm_p']:.3g}",
                transform=axis.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7},
            )
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlabel("Variant minus registered reference (bootstrap 95% CI)")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle(
        f"Pre-registered chained UAV ablation ({evidence_level}; green = Holm rejection)"
    )
    figure.savefig(path, dpi=220)
    plt.close(figure)


def write_report(
    path: Path,
    audit: dict,
    config: dict,
    comparison_rows: list[dict],
    resource_rows: list[dict],
) -> None:
    inferential = config["level"] in {"pilot", "paper"} and len(config["seeds"]) > 1
    lines = [
        "# UAV 语义与控制链式消融报告",
        "",
        f"> 自动生成；证据等级：`{config['level']}`。"
        + ("" if inferential else " 本结果仅验证管线，不作效果推断。"),
        "",
        "## Artifact 与预注册状态",
        "",
        f"- 审计状态：`{audit['status']}`",
        f"- 变体/比较/种子：{audit['variant_count']} / "
        f"{len(config['ablation_contract']['comparisons'])} / {audit['seed_count']}",
        f"- checksum 条目：{audit['checksum_entry_count']}",
        f"- 多重检验族：{len(comparison_rows)} 个预注册主假设",
        f"- 警告：{'; '.join(audit['warnings']) if audit['warnings'] else '无'}",
        "",
        "## 链式主比较",
        "",
        "| Factor | Contrast | Metric | Δ | 95% CI | exact p | Holm p | Reject |",
        "|---|---|---|---:|---:|---:|---:|:---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['factor']} | {row['variant']} − {row['reference']} | "
            f"{row['metric']} | {row['mean_difference']:.4f} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
            f"{row['exact_p']:.6f} | {row['holm_p']:.6f} | "
            f"{'yes' if row['holm_reject_0_05'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## 运行资源",
            "",
            "| Variant | Seed | Wall s | Peak CUDA MiB | Text cache entries |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in resource_rows:
        wall = row["wall_time_seconds"]
        peak = row["cuda_peak_allocated_mb"]
        cache = row["text_model_cache_entries"]
        lines.append(
            f"| {row['variant']} | {row['seed']} | "
            f"{wall:.2f} | {peak:.1f} | {cache} |"
            if wall is not None and peak is not None
            else f"| {row['variant']} | {row['seed']} | n/a | n/a | n/a |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 每条效应均为 variant minus 其契约中注册的 reference；链式比较不可改写成全部相对 full。",
            "- `identity_oracle` 获得 canonical-label identity，不是自然语言理解基线。",
            "- `no_intent` 仍保留同一任务标签、奖励画像与 posture-derived mask，但 actor/critic 输入为全零；报告必须披露该 mask 侧信道。",
            "- CBF 是经验安全过滤器；碰撞改善不能表述为严格安全保证。",
            "- smoke 的单 seed 与极短训练只证明实现和统计流水线连通。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(output_dir: Path, study_dir: Path, config_path: Path) -> None:
    files = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "source_study": str(study_dir.resolve()),
        "source_config": str(config_path.resolve()),
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
    if "ablation_contract" not in config:
        raise ValueError("config has no ablation_contract")
    audit = validate_study_artifact(args.study_dir, config)
    if audit["status"] != "valid":
        raise RuntimeError(f"refusing to render invalid artifact: {audit['errors']}")
    summary = read_json(args.study_dir / "summary.json")
    mean_rows = build_mean_rows(summary, config)
    comparison_rows = build_comparison_rows(summary)
    resource_rows = build_resource_rows(args.study_dir, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "ablation_means.csv", mean_rows)
    write_csv(args.output_dir / "ablation_primary_comparisons.csv", comparison_rows)
    write_csv(args.output_dir / "resource_audit.csv", resource_rows)
    plot_comparisons(
        args.output_dir / "figure_ablation_primary.png",
        comparison_rows,
        str(config["level"]),
    )
    write_report(
        args.output_dir / "ABLATION_REPORT.md",
        audit,
        config,
        comparison_rows,
        resource_rows,
    )
    write_manifest(args.output_dir, args.study_dir, args.config)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(args.output_dir),
                "files": sorted(path.name for path in args.output_dir.iterdir()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
