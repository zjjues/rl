"""Statistical summaries for paper-grade reinforcement-learning studies."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


def as_finite_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("at least one finite observation is required")
    return array


def interquartile_mean(values: Iterable[float]) -> float:
    """Return a deterministic 25%-trimmed mean with fractional boundary weights."""
    array = np.sort(as_finite_array(values))
    n = array.size
    lower = 0.25 * n
    upper = 0.75 * n
    total = 0.0
    weight = 0.0
    for idx, value in enumerate(array):
        overlap = max(0.0, min(idx + 1.0, upper) - max(float(idx), lower))
        total += overlap * float(value)
        weight += overlap
    return total / weight if weight > 0.0 else float(np.mean(array))


def bootstrap_interval(
    values: Iterable[float],
    statistic="mean",
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Dict[str, float]:
    array = as_finite_array(values)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    if statistic == "mean":
        fn = np.mean
    elif statistic == "median":
        fn = np.median
    elif statistic == "iqm":
        fn = interquartile_mean
    elif callable(statistic):
        fn = statistic
    else:
        raise ValueError(f"unsupported statistic: {statistic}")
    rng = np.random.default_rng(seed)
    if array.size == 1:
        value = float(array[0])
        return {
            "low": value,
            "high": value,
            "confidence": float(confidence),
            "n_resamples": int(n_resamples),
            "seed": int(seed),
        }
    estimates = np.empty(n_resamples, dtype=np.float64)
    chunk_size = 2048
    iqm_weights = None
    if statistic == "iqm":
        lower = 0.25 * array.size
        upper = 0.75 * array.size
        iqm_weights = np.asarray(
            [
                max(0.0, min(idx + 1.0, upper) - max(float(idx), lower))
                for idx in range(array.size)
            ],
            dtype=np.float64,
        )
    for start in range(0, n_resamples, chunk_size):
        end = min(start + chunk_size, n_resamples)
        indices = rng.integers(0, array.size, size=(end - start, array.size))
        samples = array[indices]
        if statistic == "mean":
            estimates[start:end] = samples.mean(axis=1)
        elif statistic == "median":
            estimates[start:end] = np.median(samples, axis=1)
        elif statistic == "iqm":
            sorted_samples = np.sort(samples, axis=1)
            estimates[start:end] = (
                sorted_samples @ iqm_weights / float(iqm_weights.sum())
            )
        else:
            estimates[start:end] = [float(fn(sample)) for sample in samples]
    alpha = (1.0 - confidence) / 2.0
    return {
        "low": float(np.quantile(estimates, alpha)),
        "high": float(np.quantile(estimates, 1.0 - alpha)),
        "confidence": float(confidence),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def summarize_sample(
    values: Iterable[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Dict[str, object]:
    array = as_finite_array(values)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "iqm": float(interquartile_mean(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean_ci": bootstrap_interval(array, "mean", confidence, n_resamples, seed),
        "iqm_ci": bootstrap_interval(array, "iqm", confidence, n_resamples, seed + 1),
        "raw": [float(value) for value in array],
    }


def paired_difference_summary(
    treatment: Sequence[float],
    baseline: Sequence[float],
    lower_is_better: bool,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Dict[str, object]:
    treatment_array = np.asarray(treatment, dtype=np.float64).reshape(-1)
    baseline_array = np.asarray(baseline, dtype=np.float64).reshape(-1)
    if treatment_array.shape != baseline_array.shape:
        raise ValueError("paired comparisons require arrays with identical shape")
    finite = np.isfinite(treatment_array) & np.isfinite(baseline_array)
    treatment_array = treatment_array[finite]
    baseline_array = baseline_array[finite]
    if treatment_array.size == 0:
        raise ValueError("paired comparisons require at least one finite pair")
    differences = treatment_array - baseline_array
    wins = differences < 0.0 if lower_is_better else differences > 0.0
    ties = differences == 0.0
    difference_std = (
        float(np.std(differences, ddof=1)) if differences.size > 1 else 0.0
    )
    return {
        "direction": "treatment_minus_baseline",
        "lower_is_better": bool(lower_is_better),
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "win_rate": float(np.mean(wins)),
        "tie_rate": float(np.mean(ties)),
        "standardized_effect_dz": (
            float(np.mean(differences) / difference_std)
            if difference_std > 1e-12 else 0.0
        ),
        "randomization_test": paired_randomization_test(
            treatment_array,
            baseline_array,
            seed=seed,
        ),
        "difference_ci": bootstrap_interval(
            differences,
            "mean",
            confidence,
            n_resamples,
            seed,
        ),
        "raw_differences": [float(value) for value in differences],
    }


def paired_randomization_test(
    treatment: Sequence[float],
    baseline: Sequence[float],
    *,
    n_resamples: int = 100_000,
    seed: int = 0,
    max_exact_pairs: int = 16,
) -> Dict[str, object]:
    """Two-sided paired sign-flip test for a zero mean difference."""

    treatment_array = np.asarray(treatment, dtype=np.float64).reshape(-1)
    baseline_array = np.asarray(baseline, dtype=np.float64).reshape(-1)
    if treatment_array.shape != baseline_array.shape:
        raise ValueError("paired randomization requires arrays with identical shape")
    finite = np.isfinite(treatment_array) & np.isfinite(baseline_array)
    differences = (treatment_array - baseline_array)[finite]
    if differences.size == 0:
        raise ValueError("paired randomization requires at least one finite pair")
    observed = abs(float(np.mean(differences)))
    tolerance = 1e-15
    if differences.size <= max_exact_pairs:
        count = 1 << differences.size
        indices = np.arange(count, dtype=np.uint64)[:, None]
        bits = (indices >> np.arange(differences.size, dtype=np.uint64)) & 1
        signs = bits.astype(np.float64) * 2.0 - 1.0
        estimates = np.abs(np.mean(signs * differences[None, :], axis=1))
        extreme = int(np.count_nonzero(estimates >= observed - tolerance))
        return {
            "alternative": "two-sided",
            "method": "exact_paired_sign_flip",
            "p_value": float(extreme / count),
            "permutations": int(count),
        }
    if n_resamples < 1_000:
        raise ValueError("Monte Carlo randomization requires at least 1000 resamples")
    rng = np.random.default_rng(seed)
    extreme = 0
    chunk_size = 4096
    for start in range(0, n_resamples, chunk_size):
        size = min(chunk_size, n_resamples - start)
        signs = rng.choice((-1.0, 1.0), size=(size, differences.size))
        estimates = np.abs(np.mean(signs * differences[None, :], axis=1))
        extreme += int(np.count_nonzero(estimates >= observed - tolerance))
    return {
        "alternative": "two-sided",
        "method": "monte_carlo_paired_sign_flip",
        "p_value": float((extreme + 1) / (n_resamples + 1)),
        "permutations": int(n_resamples),
        "seed": int(seed),
    }


def holm_adjust(p_values: Mapping[str, float], alpha: float = 0.05) -> Dict[str, object]:
    """Holm family-wise error correction keyed by stable hypothesis names."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    validated = {}
    for key, value in p_values.items():
        number = float(value)
        if not np.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"invalid p-value for {key!r}")
        validated[str(key)] = number
    ordered = sorted(validated, key=lambda key: (validated[key], key))
    adjusted: Dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, (count - rank) * validated[key])
        adjusted[key] = min(1.0, running)
    return {
        "method": "holm",
        "alpha": float(alpha),
        "family_size": count,
        "adjusted_p_values": {key: adjusted[key] for key in validated},
        "reject": {key: bool(adjusted[key] <= alpha) for key in validated},
    }


def performance_profile(
    scores_by_algorithm: Dict[str, Sequence[float]],
    thresholds: Optional[Sequence[float]] = None,
    lower_is_better: bool = False,
) -> Dict[str, object]:
    """Return empirical fractions meeting normalized performance thresholds."""
    arrays = {name: as_finite_array(values) for name, values in scores_by_algorithm.items()}
    lengths = {array.size for array in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("performance profiles require equally sized algorithm samples")
    stacked = np.vstack(list(arrays.values()))
    best = np.min(stacked, axis=0) if lower_is_better else np.max(stacked, axis=0)
    worst = np.max(stacked, axis=0) if lower_is_better else np.min(stacked, axis=0)
    span = np.abs(best - worst)
    normalized = {}
    for name, values in arrays.items():
        if lower_is_better:
            scores = np.divide(
                worst - values,
                span,
                out=np.ones_like(values),
                where=span > 1e-12,
            )
        else:
            scores = np.divide(
                values - worst,
                span,
                out=np.ones_like(values),
                where=span > 1e-12,
            )
        normalized[name] = np.clip(scores, 0.0, 1.0)
    resolved_thresholds = np.asarray(
        thresholds if thresholds is not None else np.linspace(0.0, 1.0, 101),
        dtype=np.float64,
    )
    return {
        "lower_is_better": bool(lower_is_better),
        "thresholds": resolved_thresholds.tolist(),
        "profiles": {
            name: [float(np.mean(values >= threshold)) for threshold in resolved_thresholds]
            for name, values in normalized.items()
        },
    }
