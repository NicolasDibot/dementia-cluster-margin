import numpy as np
import pandas as pd
import pytest

from dementia_margin.share_weighting import (
    BALANCE_FEATURES, COUNTRIES, entropy_balance, evaluate_weighting,
    fit_balance_specification,
)


def population():
    rng = np.random.default_rng(719)
    n = 96
    frame = pd.DataFrame({"mergeid": [f"synthetic{i}" for i in range(n)],
                          "hhid": [f"synthetic-household{i // 2}" for i in range(n)],
                          "country_name": np.repeat(COUNTRIES, n // 4),
                          "cciw_w9": np.ones(n), "dw_hcap": np.ones(n),
                          "cpsw_hcap": np.ones(n)})
    for name in BALANCE_FEATURES:
        frame[name] = rng.normal(size=n)
    frame["margin_two_cluster"] = rng.normal(size=n)
    frame["hard_two_cluster"] = frame.margin_two_cluster.gt(0).astype(int)
    frame["outcome"] = (frame.margin_two_cluster + rng.normal(size=n) > 0).astype(int)
    return frame


def test_entropy_recovers_target_moment_and_keeps_absent_rows_zero():
    design = np.array([[-1.], [0.], [1.]])
    weights, info = entropy_balance(design, [4., 0., 1.], ridge=0)
    assert weights[1] == 0
    assert abs(np.average(design[:, 0], weights=weights)) < 1e-7
    assert info["optimizer_acceptable"]


def test_target_standardization_and_outcome_independence():
    frame = population()
    frame.loc[0, "age_w7"] = np.nan
    spec = fit_balance_specification(frame)
    matrix = spec.transform(frame)
    assert np.allclose(np.average(matrix, axis=0, weights=frame.cciw_w9), 0, atol=1e-12)
    assert "age_w7__missing" in spec.terms
    changed = frame.assign(outcome=1 - frame.outcome)
    assert spec == fit_balance_specification(changed)


def test_paired_weighting_output_and_reproducibility():
    frame = population()
    first, info = evaluate_weighting(frame, frame, "outcome", binary=True, replicates=12, seed=9)
    second, _ = evaluate_weighting(frame, frame, "outcome", binary=True, replicates=12, seed=9)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 9
    assert info["entropy_refit_in_every_draw"]
    unweighted = first.loc[first.regime.eq("unweighted"), ["estimate", "ci95_low", "ci95_high"]].to_numpy()
    official = first.loc[first.regime.eq("official_cpsw"), ["estimate", "ci95_low", "ci95_high"]].to_numpy()
    assert np.allclose(unweighted, official)


def test_missing_target_member_is_not_silently_dropped():
    frame = population()
    with pytest.raises(ValueError, match="subset"):
        evaluate_weighting(frame, frame.iloc[1:], "outcome", binary=True, replicates=2)


def test_unknown_country_and_invalid_weights_fail():
    frame = population()
    with pytest.raises(ValueError, match="country"):
        fit_balance_specification(frame.assign(country_name="Other"))
    with pytest.raises(ValueError, match="positive"):
        fit_balance_specification(frame.assign(cciw_w9=-1))
