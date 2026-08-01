"""Benchmark cold construction and cached semantic-profile inference."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from intent_generalization import load_generalization_suite  # noqa: E402
from imappo import IMAPPO  # noqa: E402
from run_research_study import build_config, read_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    return parser.parse_args()


def timed_samples(callback, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        callback()
        samples.append(time.perf_counter() - start)
    return samples


def summarize_seconds(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "mean_seconds": float(statistics.mean(samples)),
        "median_seconds": float(statistics.median(samples)),
        "p95_seconds": float(ordered[p95_index]),
        "min_seconds": float(min(samples)),
        "max_seconds": float(max(samples)),
    }


def parameter_count(module) -> int:
    if module is None:
        return 0
    return int(sum(parameter.numel() for parameter in module.parameters()))


def main() -> None:
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("repeats must be at least two")
    spec = read_json(args.config.resolve())
    variants = {str(item["key"]): item for item in spec["variants"]}
    if args.variant not in variants:
        raise ValueError(f"unknown variant: {args.variant}")
    cfg = build_config(spec, variants[args.variant], int(spec["seeds"][0]))
    construction_start = time.perf_counter()
    algo = IMAPPO(cfg)
    construction_seconds = time.perf_counter() - construction_start
    suite = load_generalization_suite(ROOT / str(spec["generalization"]["suite"]))
    entries = [
        (str(query["canonical_label"]), str(query["description"]))
        for query in suite["queries"]
    ]
    first_start = time.perf_counter()
    algo.encode_intent_queries(entries)
    first_query_batch_seconds = time.perf_counter() - first_start
    adapter = algo.objective_semantic_adapter
    if adapter is None:
        raise RuntimeError("selected variant does not have an objective semantic adapter")
    cached_profile_samples = timed_samples(
        lambda: adapter.predict_profiles(entries), args.repeats
    )
    repeated_full_encode_samples = timed_samples(
        lambda: algo.encode_intent_queries(entries), args.repeats
    )
    nli_module = getattr(getattr(adapter, "nli_model", None), "model", None)
    payload = {
        "schema_version": 1,
        "config": str(args.config.resolve()),
        "variant": args.variant,
        "suite_id": suite["suite_id"],
        "query_count": len(entries),
        "repeats": args.repeats,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": str(cfg.device),
        "encoder_model": cfg.intent_encoder_model,
        "encoder_revision": cfg.intent_encoder_revision,
        "nli_model": cfg.intent_nli_model if nli_module is not None else None,
        "nli_model_revision": (
            cfg.intent_nli_model_revision if nli_module is not None else None
        ),
        "construction_seconds": construction_seconds,
        "first_query_batch_seconds": first_query_batch_seconds,
        "cached_profile_batch": summarize_seconds(cached_profile_samples),
        "repeated_full_encode_batch": summarize_seconds(repeated_full_encode_samples),
        "cached_profile_per_query_mean_seconds": (
            statistics.mean(cached_profile_samples) / len(entries)
        ),
        "encoder_parameter_count": parameter_count(adapter.model),
        "nli_parameter_count": parameter_count(nli_module),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
