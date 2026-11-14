import numpy as np
import pandas as pd
import pytest

from dementia_margin.share_hcap import construct_frames
from dementia_margin.rank_metrics import ASSISTANCE_COLUMNS


def inputs():
    ids = [f"synthetic{i}" for i in range(8)]
    margin = np.arange(8) - 5.5
    scores = pd.DataFrame({"mergeid": ids, "wave": 7, "margin_two_cluster": margin,
                           "hard_two_cluster": (margin > 0).astype(int),
                           "cognition_only": np.arange(8), "function_only": np.arange(8)})
    easy = pd.DataFrame({"mergeid": ids, "hhid": ids, "wave": 7, "country": 12,
                         "age": [50, 65, 68, 70, 72, 74, 78, 80], "female": 1,
                         "eduyears_mod": 12, "sphus": 3, "chronic_mod": 2,
                         "eurod": 1, "int_year": 2017, "int_month": 5})
    proxy = pd.DataFrame({"mergeid": ids, "r7proxy": 0})
    respondent = pd.DataFrame({"mergeid": ids, "country": 12,
                               "class_cognition_pooled": [1, 1, 2, 3, -1, 1, 2, 3],
                               "dw_hcap": 1., "cpsw_hcap": 1., "int_year_hcap_r": 2022,
                               "int_month_hcap_r": 7, "WR_delayed_score": np.arange(8),
                               "LM_Story1_DelayedRecall_score": [0, 1, 2, 3, 4, 5, 6, 6],
                               "LM_Story2_DelayedRecall_score": np.arange(8) * 2})
    informant = pd.DataFrame({"mergeid": ids, **{name: [1, 5, 1, 5, 1, 5, 1, 5] for name in ASSISTANCE_COLUMNS}})
    coverscreen = pd.DataFrame({"mergeid": ids, "deceased": 0, "interview": 1})
    weights = pd.DataFrame({"mergeid": ids, "cciw_w9": 1.})
    return scores, easy, proxy, respondent, informant, coverscreen, weights


def test_linkage_domains_and_outcomes_are_reconstructed():
    linked, target, metadata = construct_frames(*inputs())
    assert metadata["source_population_n"] == 7
    assert metadata["linked_hcap_n"] == 7
    assert metadata["target_n"] == 7
    assert linked.hcap_severe_cognitive_impairment.isna().sum() == 1
    assert linked.informant_assistance_fraction.isin([0, 1]).all()
    assert np.isclose(metadata["median_followup_years"], 5 + 2 / 12)
    assert set(linked.mergeid) == set(target.mergeid)


def test_outcome_missingness_does_not_change_landmark_or_target():
    pieces = list(inputs())
    pieces[3].loc[1, "class_cognition_pooled"] = -9
    pieces[4].loc[1, list(ASSISTANCE_COLUMNS[:5])] = -9
    linked, target, metadata = construct_frames(*pieces)
    assert len(target) == len(linked) == 7
    assert metadata["outcome_available"]["hcap_severe_cognitive_impairment"] == 5
    assert metadata["outcome_available"]["informant_assistance_fraction"] == 6


def test_duplicate_and_country_mismatch_fail():
    pieces = list(inputs())
    pieces[3] = pd.concat([pieces[3], pieces[3].iloc[[0]]])
    with pytest.raises(ValueError, match="unique"):
        construct_frames(*pieces)
    pieces = list(inputs())
    pieces[3].loc[2, "country"] = 16
    with pytest.raises(ValueError, match="Country mismatch"):
        construct_frames(*pieces)


def test_zero_target_weight_fails():
    pieces = list(inputs())
    pieces[6].loc[1, "cciw_w9"] = 0
    with pytest.raises(ValueError, match="positive"):
        construct_frames(*pieces)
