from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_external_language_ood import summarize_ood_profiles  # noqa: E402
from intent_objectives import OBJECTIVE_KEYS  # noqa: E402


def test_ood_summary_never_decodes_collision_and_reports_activation():
    profiles = np.ones((2, len(OBJECTIVE_KEYS)), dtype=np.float32)
    profiles[1, OBJECTIVE_KEYS.index("energy")] = 1.2
    summary = summarize_ood_profiles(profiles, thresholds=(0.1,))
    assert summary["collision_decoded"] is False
    assert summary["activation_rate_by_uncalibrated_threshold"]["0.1"] == 0.5
    assert summary["largest_deviation_class_counts"]["energy:high"] == 1


def test_ood_summary_rejects_wrong_profile_dimension():
    with pytest.raises(ValueError, match="one column"):
        summarize_ood_profiles(np.ones((2, len(OBJECTIVE_KEYS) + 1)))
