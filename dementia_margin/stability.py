"""Participant-level perturbations with frozen factor coordinates."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score
from .geometry import source_impairment_profiles, signed_margin, ward_initial_partition

METRICS = ("heldout_adjusted_rand", "target_label_agreement", "target_margin_spearman")

def source_domain_burdens(completed: pd.DataFrame, *, include_fluency=False):
    """Source-standardized full domain profiles, not the simpler benchmarks.

    Functional burden averages 24 binary items and the two functional totals;
    cognitive burden averages negative immediate, delayed and total recall.
    Encoded SHARE factor levels are decoded only with their exact column prefix.
    """
    return source_impairment_profiles(completed, "share" if include_fluency else "adams").rename(
        columns={"function_burden": "function", "cognition_burden": "cognition"}
    )[["function", "cognition"]]


def participant_subsample(person_ids, rng, fraction=0.8):
    people = np.asarray(person_ids).astype(str)
    unique = np.unique(people)
    if not 0 < fraction < 1 or len(unique) < 5:
        raise ValueError("Subsampling requires at least five people and fraction in (0,1)")
    n = int(np.floor(len(unique) * fraction))
    if not 1 <= n < len(unique):
        raise ValueError("Subsample must leave nonempty training and held-out sets")
    chosen = rng.choice(unique, size=n, replace=False)
    return np.isin(people, chosen)


def profile_orientation(labels, burdens):
    labels = np.asarray(labels, int)
    values = np.asarray(burdens, float)
    if set(np.unique(labels)) != {1, 2} or values.shape != (len(labels), 2):
        raise ValueError("Two nonempty clusters and two domain burdens required")
    profiles = np.vstack([values[labels == k].mean(axis=0) for k in (1, 2)])
    if not np.isfinite(profiles).all() or np.any(profiles[0] == profiles[1]):
        raise ValueError("Undefined or tied cluster profiles")
    worst = np.argmax(profiles, axis=0)
    if worst[0] != worst[1]:
        raise ValueError("No jointly most impaired cluster")
    return int(worst[0])


def stability_refits(source_coordinates, person_ids, source_burdens, primary_source_label,
                     target_coordinates, primary_target_margin, primary_target_label,
                     *, consolidate, repeats=100, fraction=0.8, seed=20261109,
                     progress=None):
    x, target = np.asarray(source_coordinates, float), np.asarray(target_coordinates, float)
    burden = np.asarray(source_burdens, float)
    source_label = np.asarray(primary_source_label, int)
    target_margin = np.asarray(primary_target_margin, float)
    target_label = np.asarray(primary_target_label, int)
    if (x.ndim != 2 or x.shape[1] != 5 or target.ndim != 2 or target.shape[1] != 5
            or burden.shape != (len(x), 2) or len(person_ids) != len(x)
            or source_label.shape != (len(x),) or target_margin.shape != (len(target),)
            or target_label.shape != (len(target),)):
        raise ValueError("Stability inputs have incompatible dimensions")
    if not all(np.isfinite(value).all() for value in [x, target, burden, target_margin]):
        raise ValueError("Stability inputs must be finite")
    if repeats < 1 or not np.isin(source_label, [0, 1]).all():
        raise ValueError("Invalid repetition count or primary labels")
    if not np.array_equal(target_margin > 0, target_label == 1):
        raise ValueError("Primary target label is inconsistent with margin")
    rng, records = np.random.default_rng(seed), []
    for replicate in range(repeats):
        selected = participant_subsample(person_ids, rng, fraction)
        initial, _ = ward_initial_partition(x[selected], min_clusters=2, max_clusters=2)
        labels = consolidate(x[selected], initial)
        severe = profile_orientation(labels, burden[selected])
        centers = np.vstack([x[selected][labels == k].mean(axis=0) for k in (1, 2)])
        _, training_label = signed_margin(x[selected], centers, severe)
        if not np.array_equal(training_label, (labels == severe + 1).astype(int)):
            raise RuntimeError("Refitted consolidation is not a nearest-centroid partition")
        _, heldout_label = signed_margin(x[~selected], centers, severe)
        refit_margin, refit_label = signed_margin(target, centers, severe)
        values = [adjusted_rand_score(source_label[~selected], heldout_label),
                  float(np.mean(refit_label == target_label)),
                  float(spearmanr(target_margin, refit_margin).statistic)]
        if not np.isfinite(values).all():
            raise RuntimeError("Stability metric is undefined")
        records.append({"replicate": replicate + 1, **dict(zip(METRICS, values))})
        if progress is not None:
            progress(replicate + 1, repeats)
    return pd.DataFrame(records)


def summarize_stability(replicates):
    return {metric: {"median": float(replicates[metric].median()),
                     "percentile_10": float(replicates[metric].quantile(0.1)),
                     "minimum": float(replicates[metric].min())}
            for metric in METRICS}
