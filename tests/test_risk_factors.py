import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from dementia_margin.risk_factors import (
    Factor, compare_common_population, fit_weighted_linear,
    fit_weighted_logistic, prepare_panel,
)
from dementia_margin.risk_panel import share_alcohol, _inactive, _sensory
from dementia_margin.risk_panel import build_hrs_panel, read_scores
from pathlib import Path


def panel(n=900):
    rng = np.random.default_rng(17)
    baseline = -rng.uniform(0.05, 3, n)
    exposure = rng.binomial(1, 0.4, n)
    followup = baseline + 0.5 * exposure + rng.normal(0, 1.1, n)
    return pd.DataFrame({
        "cohort": "HRS", "participant_id": np.arange(n),
        "household_id": np.arange(n) // 2, "country": 1,
        "interval": rng.choice(["8-9", "9-10"], n),
        "age": rng.uniform(55, 90, n), "sex": rng.binomial(1, .55, n),
        "education": rng.choice([1, 2, 3], n), "weight": rng.uniform(.5, 2, n),
        "hard_baseline": 0, "hard_followup": (followup > 0).astype(int),
        "margin_baseline": baseline, "margin_followup": followup,
        "exposure": exposure,
    })


def test_primary_models_retain_entrants_and_have_identical_population():
    data = panel()
    data.loc[0, "exposure"] = np.nan
    data.loc[1, "margin_followup"] = np.nan
    estimates, flow = compare_common_population(data, [Factor("exposure", "Exposure")])
    common = estimates.query("population == 'common_population'")
    assert len(common) == 2
    assert set(common.status) == {"ok"}
    assert common.n.nunique() == common.weight_sum.nunique() == 1
    assert common.n.iloc[0] == len(data) - 2
    assert common.entries.min() > 0
    assert not common.comparison_is_power_test.any()
    assert not {"participant_id", "household_id"} & set(estimates)


def test_education_exposure_does_not_require_education_covariate():
    data = panel()
    data["education"] = np.nan
    result, _ = compare_common_population(data, [Factor("exposure", "Education", False)])
    assert set(result.status) == {"ok"}
    assert all("education" not in terms for terms in result.terms)


def test_invalid_geometry_and_duplicate_rows_are_rejected():
    data = panel()
    with pytest.raises(ValueError, match="two-cluster"):
        prepare_panel(data, geometry_n_clusters=3)
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_panel(pd.concat([data, data.iloc[[0]]]))
    data.loc[0, "hard_baseline"] = 1
    with pytest.raises(ValueError, match="disagrees"):
        prepare_panel(data)


def test_missing_unavailable_factor_not_relabelled_zero():
    data = panel()
    _, flow = compare_common_population(data, [Factor("not_measured", "Missing")])
    assert flow.status.iloc[0] == "factor_unavailable"


def test_common_scaling_does_not_depend_on_exposure_missingness():
    data = panel()
    data["second"] = data.exposure
    data.loc[:200, "second"] = np.nan
    _, flow = compare_common_population(data, [Factor("exposure", "A"), Factor("second", "B")])
    assert flow.delta_margin_sd.nunique() == 1
    assert flow.n_common.nunique() == 2


def test_linear_coefficients_and_sandwich_agree_with_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(33)
    x = np.column_stack([np.ones(400), rng.normal(size=(400, 2))])
    y = np.einsum("ni,i->n", x, np.array([1., .3, -.2])) + rng.normal(size=400)
    weight = rng.uniform(.5, 2., 400)
    groups = np.arange(400) // 2
    b, v = fit_weighted_linear(x, y, weight, groups)
    ref = sm.WLS(y, x, weights=weight).fit(cov_type="cluster", cov_kwds={"groups": groups})
    np.testing.assert_allclose(b, ref.params, atol=1e-10)
    np.testing.assert_allclose(v, ref.cov_params(), atol=1e-10)


def test_logistic_coefficients_and_sandwich_agree_with_manual_score():
    rng = np.random.default_rng(33)
    x = np.column_stack([np.ones(700), rng.normal(size=(700, 2))])
    y = rng.binomial(1, expit(np.einsum("ni,i->n", x, np.array([-.5, .4, -.3]))))
    w = rng.uniform(.5, 2., 700)
    groups = np.arange(700) // 2
    b, cov = fit_weighted_logistic(x, y, w, groups)
    p = expit(np.einsum("ni,i->n", x, b))
    assert np.max(np.abs(np.einsum("ni,n->i", x, w * (y - p)))) < 1e-6
    b2, cov2 = fit_weighted_logistic(x, y, w * 100, groups)
    np.testing.assert_allclose(b, b2, atol=1e-10)
    np.testing.assert_allclose(cov, cov2, atol=1e-10)


def test_logistic_no_events_fails_explicitly():
    with pytest.raises(ValueError, match="events and non-events"):
        fit_weighted_logistic(np.ones((30, 1)), np.zeros(30), np.ones(30), np.arange(30))


def test_factor_specific_weights_keep_both_models_on_same_records():
    data = panel()
    data["weight_exposure"] = data.weight * 2
    data.loc[:50, "weight_exposure"] = np.nan
    results, _ = compare_common_population(data, [Factor("exposure", "Exposure")])
    common = results.query("population == 'common_population'")
    assert common.n.tolist() == [849, 849]
    assert common.weight_sum.nunique() == 1


def test_alcohol_threshold_preserves_unknown_and_nondrinkers():
    wave6 = pd.DataFrame({"br039_": [1, 1, 5, -1], "br040_": [14, 15, np.nan, -1]})
    np.testing.assert_allclose(share_alcohol(wave6, 6), [0, 1, 0, np.nan], equal_nan=True)
    wave4 = pd.DataFrame({"br010_": [1, 2, 7, -1], "br019_": [2, 3, np.nan, 4], "br021_": [1, 1, 1, -1]})
    np.testing.assert_allclose(share_alcohol(wave4, 4), [0, 1, 0, np.nan], equal_nan=True)


def test_inactivity_and_vision_do_not_call_missing_values_healthy():
    inactive = _inactive(pd.Series([5, 4, 5]), pd.Series([5, 5, np.nan]), [1, 2, 3, 4, 5])
    np.testing.assert_allclose(inactive, [1, 0, np.nan], equal_nan=True)
    vision, hearing = _sensory(pd.Series([4, 2, np.nan]), pd.Series([np.nan, 3, 2]), pd.Series([4, 2, np.nan]))
    np.testing.assert_allclose(vision, [1, 0, np.nan], equal_nan=True)
    np.testing.assert_allclose(hearing, [1, 0, np.nan], equal_nan=True)


def test_multiplicity_family_preserved_when_one_factor_is_unavailable():
    data = panel()
    factors = [Factor("exposure", "Available")] + [Factor(f"missing{i}", "Missing") for i in range(11)]
    results, flow = compare_common_population(data, factors)
    np.testing.assert_allclose(results.p_bonferroni, np.minimum(1, results.p_value * 12))
    assert flow.status.eq("factor_unavailable").sum() == 11


def test_data_access_acknowledgement_precedes_input_reads(monkeypatch):
    monkeypatch.delenv("COHORT_DATA_ACCESS", raising=False)
    with pytest.raises(PermissionError, match="COHORT_DATA_ACCESS"):
        build_hrs_panel(Path("/does/not/exist"), Path("/also/missing"), Path("/still/missing"))


def test_source_scores_inside_repository_are_rejected():
    repo = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="outside Git"):
        read_scores(repo / "participant_scores.csv")
