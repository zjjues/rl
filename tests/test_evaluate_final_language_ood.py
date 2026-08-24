from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_final_language_ood import (  # noqa: E402
    decide,
    summarize_relevance_scores,
    wilson_interval,
)


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(5, 100)
    assert low < 0.05 < high


def test_relevance_summary_reports_overall_and_split_far():
    result = summarize_relevance_scores(
        [0.1, 0.2, 0.9, 0.05],
        ["train", "train", "test", "test"],
        threshold=0.2,
    )
    assert result["overall"]["accepted_count"] == 2
    assert result["overall"]["false_accept_rate_at_frozen_threshold"] == 0.5
    assert result["by_split"]["train"]["false_accept_rate_at_frozen_threshold"] == 0.5
    assert result["by_split"]["test"]["false_accept_rate_at_frozen_threshold"] == 0.5


@pytest.mark.parametrize(
    ("far", "expected"), [(0.05, "pass"), (0.08, "caution"), (0.11, "fail")]
)
def test_registered_decision_boundaries(far, expected):
    assert decide(
        far, {"pass_if_far_at_most": 0.05, "caution_if_far_at_most": 0.1}
    ) == expected
