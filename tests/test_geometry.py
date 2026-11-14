import numpy as np
import pandas as pd
import pytest
from dementia_margin.geometry import signed_margin, source_benchmarks, ward_initial_partition


def test_two_separated_clouds_select_two():
    rng = np.random.default_rng(41)
    x = np.r_[rng.normal(-4, .2, (60, 5)), rng.normal(4, .2, (60, 5))]
    labels, info = ward_initial_partition(x)
    assert info["selected_clusters"] == 2
    assert np.unique(labels[:60]).size == np.unique(labels[60:]).size == 1
    assert labels[0] != labels[-1]


def test_margin_sign_bound_and_invariance():
    rng = np.random.default_rng(42)
    x, centers = rng.normal(size=(80, 5)), rng.normal(size=(2, 5))
    m, h = signed_margin(x, centers, 1)
    assert np.array_equal(h, (m > 0).astype(int))
    assert (np.abs(m) <= np.linalg.norm(centers[1]-centers[0])+1e-12).all()
    m2, h2 = signed_margin(x+7, centers+7, 1)
    assert np.allclose(m, m2)
    assert np.array_equal(h, h2)
    assert np.allclose(signed_margin(x, centers[::-1], 0)[0], m)


def test_benchmarks_use_source_not_target_normalization():
    s = pd.DataFrame({"tr20":[2,4,6],"adltot_s":[0,1,2],"iadltot1_s":[0,2,4]})
    t = s.iloc[[0]].copy()
    b = source_benchmarks(s,t,"adams")
    assert b.cognition_only.iloc[0] == 1
    assert b.function_only.iloc[0] == -1


def test_invalid_dimensions_rejected():
    with pytest.raises(ValueError):
        ward_initial_partition(np.zeros((20, 2)))
