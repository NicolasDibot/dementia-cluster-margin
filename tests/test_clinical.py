import numpy as np
import pandas as pd
import pytest

from dementia_margin.clinical import estimands as clinical_estimands
from dementia_margin.rank_metrics import CachedRankMetric


@pytest.mark.parametrize("cohort", ["adams", "hcap"])
def test_clinical_decomposition_and_independent_metric_agree(cohort):
    rng = np.random.default_rng(19)
    phase = np.repeat([5, 6], 150)
    margin = rng.normal(size=300) + (phase - 5) * 1.2
    ordinal = rng.integers(0, 3, size=300)
    frame = pd.DataFrame({
        "margin_current": margin, "hard_current": (margin > 0).astype(int),
        "outcome": (ordinal == 2).astype(int), "outcome_ordinal": ordinal,
        "landmark_wave_tracker": phase, "cognition_only": rng.normal(size=300), "function_only": rng.normal(size=300),
    })
    weight = rng.uniform(.1, 2, 300)
    estimates = clinical_estimands(frame, weight, cohort)
    strata = phase if cohort == "adams" else np.zeros(len(frame))
    independent = CachedRankMetric(frame.outcome, frame[["hard_current", "margin_current"]],
                                   strata, binary=True).calculate(weight)
    np.testing.assert_allclose(independent, [estimates["auc::label"], estimates["auc::margin"]])
    assert abs(estimates["decomposition_error::two_cluster"]) < 1e-12
    assert abs(estimates["decomposition_error_ordinal::current"]) < 1e-12


def test_non_nested_label_is_rejected():
    frame = pd.DataFrame({"margin_current": [-1., 1.], "hard_current": [1, 0],
                          "outcome": [0, 1], "landmark_wave_tracker": [5, 5]})
    with pytest.raises(ValueError, match="nested hard state"):
        clinical_estimands(frame, np.ones(2), "adams")
