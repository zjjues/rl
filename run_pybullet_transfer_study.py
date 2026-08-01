"""Run a checksummed cross-dynamics Crazyflie transfer study."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pybullet_transfer import TransferEpisodeConfig, evaluate_transfer_episode
from research_statistics import paired_difference_summary, summarize_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_checksums(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_versions() -> Dict[str, str]:
    names = ("numpy", "scipy", "pybullet", "gymnasium", "gym-pybullet-drones", "setuptools")
    return {name: importlib.metadata.version(name) for name in names}


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def result_card(spec: Dict[str, object], summary: Dict[str, object]) -> str:
    treatment = str(spec["treatment_key"])
    lines = [
        f"# PyBullet Transfer Result Card: {spec['study_id']}", "",
        f"- Evidence level: `{spec['level']}`", f"- Paired seeds: `{len(spec['seeds'])}`",
        "- Simulator: headless Crazyflie rigid-body/rotor dynamics through VelocityAviary",
        "- Scope: high-level controller and safety-layer transfer; this is not SITL, HIL, or real flight",
        "", "## Treatment summary", "",
        "| Metric | Mean | 95% bootstrap CI |", "|---|---:|---:|",
    ]
    for metric, values in summary["variants"][treatment]["aggregate"].items():
        ci = values["mean_ci"]
        lines.append(f"| {metric} | {values['mean']:.4f} | [{ci['low']:.4f}, {ci['high']:.4f}] |")
    lines.extend(["", "## Paired treatment differences", "",
                  "Differences are treatment minus baseline.", "",
                  "| Baseline | Metric | Mean difference | 95% bootstrap CI |",
                  "|---|---|---:|---:|"])
    for baseline, metrics in summary["comparisons"].items():
        for metric, values in metrics.items():
            ci = values["difference_ci"]
            lines.append(f"| {baseline} | {metric} | {values['mean_difference']:.4f} | "
                         f"[{ci['low']:.4f}, {ci['high']:.4f}] |")
    lines.extend(["", "## Interpretation guardrails", "",
                  "- Smoke and five-seed pilot intervals are diagnostic, not confirmatory.",
                  "- A rigid-body simulation improves dynamics validity but does not establish sim-to-real safety.",
                  "- The evaluated controller uses structured objective profiles; independent-language evidence is separate.",
                  "- Collision distance and linearized command-space constraints are reported separately.", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve() if args.config.is_absolute() else (ROOT / args.config).resolve()
    spec = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / "experiments" / str(spec["level"]) / str(spec["study_id"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing study: {output}")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "config.json")
    episode_base = dict(spec["episode"])
    all_results: Dict[str, List[Dict[str, object]]] = {}
    for variant in spec["variants"]:
        variant_key = str(variant["key"])
        all_results[variant_key] = []
        for seed in spec["seeds"]:
            conditions = []
            for scenario in spec["scenarios"]:
                for profile_key, profile in spec["profiles"].items():
                    episode_config = TransferEpisodeConfig(
                        **episode_base, filter_mode=str(variant["filter_mode"])
                    )
                    metrics = evaluate_transfer_episode(
                        int(seed), str(scenario), profile, episode_config
                    )
                    conditions.append({
                        "scenario": scenario, "profile": profile_key, "metrics": metrics
                    })
            per_seed = {
                "seed": int(seed), "variant": variant, "conditions": conditions,
                "aggregate": {
                    metric: float(sum(c["metrics"][metric] for c in conditions) / len(conditions))
                    for metric in conditions[0]["metrics"]
                },
            }
            all_results[variant_key].append(per_seed)
            write_json(output / variant_key / f"seed_{seed}" / "result.json", per_seed)

    bootstrap_seed = int(spec.get("bootstrap_seed", 20260801))
    summary: Dict[str, object] = {"variants": {}, "comparisons": {}}
    for variant_key, results in all_results.items():
        metrics = results[0]["aggregate"].keys()
        summary["variants"][variant_key] = {"aggregate": {
            metric: summarize_sample(
                [result["aggregate"][metric] for result in results], seed=bootstrap_seed
            ) for metric in metrics
        }}
    treatment = str(spec["treatment_key"])
    lower_is_better = {
        "minimum_pairwise_distance": False, "collision_step_fraction": True,
        "safety_violation_step_fraction": True,
        "goal_success_fraction": False, "final_goal_rmse": True,
        "normalized_command_energy": True, "mean_filter_correction": True,
        "solver_success_fraction": False, "constraint_max_violation": True,
        "mean_solver_time_ms": True, "safety_distance": False, "speed_limit_mps": False,
        "constraint_distance": False, "robust_margin": False,
    }
    for baseline, results in all_results.items():
        if baseline == treatment:
            continue
        summary["comparisons"][baseline] = {}
        for metric in results[0]["aggregate"]:
            summary["comparisons"][baseline][metric] = paired_difference_summary(
                [item["aggregate"][metric] for item in all_results[treatment]],
                [item["aggregate"][metric] for item in results],
                lower_is_better=lower_is_better[metric], seed=bootstrap_seed,
            )
    manifest = {
        "schema_version": 1, "study_id": spec["study_id"],
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_git_revision": git_revision(), "dirty_worktree_allowed": True,
        "upstream_simulator_revision": "e712698a05a80728b06572819dcf044596707754",
        "python": sys.version, "platform": platform.platform(), "packages": package_versions(),
        "claim_scope": "rigid-body simulator transfer; not SITL/HIL/real flight",
    }
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", manifest)
    (output / "RESULT_CARD.md").write_text(result_card(spec, summary), encoding="utf-8")
    write_checksums(output)
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
