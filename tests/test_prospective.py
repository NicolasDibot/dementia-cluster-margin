import numpy as np
import pandas as pd

from dementia_margin.rank_metrics import CachedRankMetric, delayed_memory_impairment, informant_assistance_fraction


def test_rank_metric_matches_enumerated_pairs():
    y = np.array([0, 1, 1, 0, 0, 1])
    scores = np.array([[0, -.9], [1, .1], [0, -.4], [1, .3], [0, -.7], [1, .8]])
    weights = np.array([1., 2., 3., 2., 1., 3.])
    strata = np.array([0, 0, 0, 0, 1, 1])
    actual = CachedRankMetric(y, scores, strata, binary=True).calculate(weights)
    numerator, denominator = np.zeros(2), 0.
    for i in np.flatnonzero(y == 1):
        for j in np.flatnonzero(y == 0):
            if strata[i] == strata[j]:
                mass = weights[i] * weights[j]
                numerator += mass * ((scores[i] > scores[j]) + .5 * (scores[i] == scores[j]))
                denominator += mass
    np.testing.assert_allclose(actual, numerator / denominator)


def test_assistance_requires_ten_items():
    frame = pd.DataFrame({f"hibl005_{i}": [1., 5., np.nan] for i in range(1, 15)})
    frame.loc[2, [f"hibl005_{i}" for i in range(1, 10)]] = 1.
    np.testing.assert_allclose(informant_assistance_fraction(frame), [1., 0., np.nan], equal_nan=True)


def test_memory_requires_two_of_three_valid_tests():
    frame = pd.DataFrame({"WR_delayed_score": [1., 5., 8., -1.],
                          "LM_Story1_DelayedRecall_score": [1., 3., 6., np.nan],
                          "LM_Story2_DelayedRecall_score": [2., 12., 20., 3.]})
    result = delayed_memory_impairment(frame)
    assert result.iloc[:3].notna().all()
    assert pd.isna(result.iloc[3])
    assert result.iloc[0] > result.iloc[2]
