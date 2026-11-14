import numpy as np
import pandas as pd
import pytest

from dementia_margin.stability import (
    METRICS, participant_subsample, profile_orientation,
    source_domain_burdens, stability_refits, summarize_stability,
)

from dementia_margin.geometry import FUNCTION_PROFILE_ITEMS as FUNCTION_ITEMS

def test_subsampling_keeps_every_wave_of_selected_person():
    ids = np.repeat(np.arange(50), 3)
    selected = participant_subsample(ids, np.random.default_rng(4))
    assert len(np.unique(ids[selected])) == 40
    assert np.all(selected.reshape(-1, 3).min(axis=1) == selected.reshape(-1, 3).max(axis=1))
    assert not set(ids[selected]) & set(ids[~selected])


def test_profiles_use_all_items_and_source_sample_standard_deviation():
    frame = pd.DataFrame({column: [0, 1, 2] for column in FUNCTION_ITEMS})
    for column in ["imrc", "dlrc", "tr20"]:
        frame[column] = [6, 4, 2]
    frame["hlthlma"] = ["hlthlma_0", "hlthlma_1", "hlthlma_2"]
    burden = source_domain_burdens(frame)
    assert np.allclose(burden.function, [-1, 0, 1])
    assert np.allclose(burden.cognition, [-1, 0, 1])


def test_orientation_requires_agreement_and_is_label_invariant():
    labels = np.repeat([1, 2], 3)
    burdens = np.array([[-1, -2]] * 3 + [[1, 2]] * 3)
    assert profile_orientation(labels, burdens) == 1
    assert profile_orientation(3 - labels, burdens) == 0
    with pytest.raises(ValueError, match="jointly"):
        profile_orientation(labels, burdens * [1, -1])


def test_stability_on_well_separated_clouds():
    rng = np.random.default_rng(31)
    source = np.r_[rng.normal(-5, 0.1, (60, 5)), rng.normal(5, 0.1, (60, 5))]
    labels = (source[:, 0] > 0).astype(int)
    target = np.r_[rng.normal(-5, .1, (20, 5)), rng.normal(5, .1, (20, 5))]
    from dementia_margin.geometry import signed_margin
    centers = np.vstack([source[labels == k].mean(axis=0) for k in (0, 1)])
    margin, target_labels = signed_margin(target, centers, 1)
    # The Ward cut is already nearest-centroid consistent for these clouds.
    result = stability_refits(source, np.repeat(np.arange(60), 2),
                             np.column_stack([source[:, 0]] * 2), labels,
                             target, margin, target_labels,
                             consolidate=lambda x, initial: initial, repeats=5)
    assert (result.heldout_adjusted_rand == 1).all()
    assert (result.target_label_agreement == 1).all()
    assert (result.target_margin_spearman > .99).all()
    summary = summarize_stability(result)
    assert set(summary) == set(METRICS)
    assert summary["heldout_adjusted_rand"]["percentile_10"] == 1


def test_stability_rejects_misaligned_primary_target_label():
    rng = np.random.default_rng(2)
    source, target = rng.normal(size=(30, 5)), rng.normal(size=(20, 5))
    with pytest.raises(ValueError, match="inconsistent"):
        stability_refits(source, np.arange(30), source[:, :2], np.zeros(30),
                         target, np.ones(20), np.zeros(20),
                         consolidate=lambda x, initial: initial, repeats=1)
