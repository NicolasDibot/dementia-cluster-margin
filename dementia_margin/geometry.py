"""Ward clustering and signed centroid-distance margins in five dimensions."""
from __future__ import annotations
import numpy as np
import pandas as pd
import fastcluster
from scipy.cluster.hierarchy import cut_tree

FUNCTION_PROFILE_ITEMS = (
    "hlthlma", "walkra", "dressa", "batha", "eata", "beda", "toilta",
    "phonea", "medsa", "moneya", "shopa", "mealsa", "mapa", "housewka",
    "walk100a", "sita", "chaira", "climsa", "clim1a", "lifta", "stoopa",
    "armsa", "pusha", "dimea", "adltot_s", "iadltot1_s",
)

def source_impairment_profiles(source_completed, cohort):
    """Full source-standardized domains for orienting the impaired cluster.

    These are not the simpler cognition/function comparators. HRS uses all
    26 functional variables and the three recall variables.
    SHARE additionally includes verbal fluency in its cognitive domain; its
    functional domain uses all 26 variables.
    """
    cognition = ["imrc", "dlrc", "tr20"] + (["verbf"] if cohort == "share" else [])
    result = {}
    for domain, columns, sign in (
        ("cognition_burden", cognition, -1),
        ("function_burden", FUNCTION_PROFILE_ITEMS, 1),
    ):
        numeric = pd.DataFrame(index=source_completed.index)
        for column in columns:
            value = source_completed[column]
            if not pd.api.types.is_numeric_dtype(value):
                value = value.astype(str).str.removeprefix(column + "_")
            numeric[column] = pd.to_numeric(value, errors="raise")
        values = numeric.to_numpy(float)
        scale = values.std(axis=0, ddof=1)
        if not np.isfinite(values).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Source profile variables must be complete and nonconstant")
        result[domain] = sign * ((values - values.mean(axis=0)) / scale).mean(axis=1)
    return pd.DataFrame(result, index=source_completed.index)


def ward_initial_partition(coordinates, min_clusters=2, max_clusters=10):
    """HCPC inertia-ratio selection using Ward.D2 SSE increments.

    fastcluster's Ward heights are sqrt(2 * increase in within-cluster SSE).
    Consequently heights must be squared before constructing the inertia ratio.
    Sorting by the first coordinate reproduces HCPC's default ordering.
    """
    x = np.asarray(coordinates, dtype=float)
    if x.ndim != 2 or x.shape[1] != 5 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError("Expected a finite n-by-5 source coordinate matrix")
    maximum = min(int(max_clusters), len(x) - 1)
    if not 2 <= min_clusters <= maximum:
        raise ValueError("Invalid cluster-count bounds")
    order = np.argsort(x[:, 0], kind="stable")
    tree = fastcluster.linkage_vector(x[order], method="ward", metric="euclidean")
    cumulative_sse = np.r_[0.0, np.cumsum(tree[:, 2] ** 2 / 2.0)]
    candidates = np.arange(min_clusters, maximum + 1)
    before = cumulative_sse[len(x) - candidates + 1]
    after = cumulative_sse[len(x) - candidates]
    if np.any(before <= 0):
        raise ValueError("Degenerate geometry has zero within-cluster inertia")
    ratios = after / before
    selected = int(candidates[np.argmin(ratios)])
    ordered_labels = cut_tree(tree, n_clusters=[selected]).ravel().astype(int) + 1
    labels = np.empty(len(x), dtype=int)
    labels[order] = ordered_labels
    return labels, {
        "selected_clusters": selected,
        "minimum_clusters": int(min_clusters),
        "maximum_clusters": int(maximum),
        "inertia_ratios": dict(zip(map(str, candidates), map(float, ratios))),
        "ward_method": "Ward.D2; inertia increments = linkage_height_squared / 2",
    }


def signed_margin(coordinates, centers, severe_cluster):
    x, centers = np.asarray(coordinates, float), np.asarray(centers, float)
    if centers.shape != (2, 5) or x.ndim != 2 or x.shape[1] != 5:
        raise ValueError("Two five-dimensional centers are required")
    if severe_cluster not in (0, 1) or not np.isfinite(x).all() or not np.isfinite(centers).all():
        raise ValueError("Invalid geometry or orientation")
    distances = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
    margin = distances[:, 1 - severe_cluster] - distances[:, severe_cluster]
    if np.any(np.abs(margin) > np.linalg.norm(centers[1] - centers[0]) + 1e-10):
        raise RuntimeError("Triangle-inequality margin bound failed")
    return margin, (margin > 0).astype(int)


def source_benchmarks(source_completed, target_completed, cohort):
    """Unsupervised, equally weighted domain scores standardized in the source."""
    cognitive = ["tr20", "verbf"] if cohort == "share" else ["tr20"]
    functional = ["adltot_s", "iadltot1_s"]
    out = []
    for cols, sign in [(cognitive, -1), (functional, 1)]:
        source = source_completed[cols].astype(float)
        mu, sd = source.mean(), source.std(ddof=1)
        if (sd <= 0).any() or not np.isfinite(source).all().all():
            raise ValueError("Invalid completed source benchmark data")
        out.append(sign * ((target_completed[cols].astype(float) - mu) / sd).mean(axis=1))
    return pd.DataFrame({"cognition_only": out[0], "function_only": out[1]})
