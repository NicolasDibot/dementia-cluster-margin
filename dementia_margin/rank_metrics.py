"""Weighted pairwise discrimination and household bootstrap calculations."""
from __future__ import annotations
import numpy as np
import pandas as pd

def delayed_memory_impairment(frame: pd.DataFrame) -> pd.Series:
    """Delayed-memory outcome requiring two of three valid recall tests.

    Valid component values are standardized over the linked assessment frame
    (before dropping people with fewer than two tests), without using any
    predictor.
    """
    ranges = {"WR_delayed_score": 10, "LM_Story1_DelayedRecall_score": 6,
              "LM_Story2_DelayedRecall_score": 25}
    values = pd.DataFrame(index=frame.index)
    for column, maximum in ranges.items():
        value = pd.to_numeric(frame[column], errors="coerce")
        values[column] = value.where(value.between(0, maximum))
    scale = values.std(ddof=1)
    if scale.isna().any() or scale.le(0).any():
        raise ValueError("Delayed-memory test has no estimable scale")
    z = (values - values.mean()) / scale
    return (-z.mean(axis=1)).where(values.notna().sum(axis=1).ge(2))


class CachedRankMetric:
    """Weighted AUC, or continuous-outcome concordance, within given strata."""

    def __init__(self, outcome, scores, strata, *, binary: bool):
        self.y = np.asarray(outcome, float)
        self.scores = np.asarray(scores, float)
        self.strata = np.asarray(strata)
        self.binary = binary
        if self.scores.ndim != 2 or len(self.y) != len(self.scores):
            raise ValueError("Invalid metric input dimensions")
        if not np.isfinite(self.y).all() or not np.isfinite(self.scores).all():
            raise ValueError("Metrics require finite outcomes and scores")
        if binary and not np.isin(self.y, [0, 1]).all():
            raise ValueError("AUC requires a binary outcome")
        self.groups = []
        if binary:
            for value in pd.unique(self.strata):
                ids = np.flatnonzero(self.strata == value)
                group = []
                for j in range(self.scores.shape[1]):
                    order = ids[np.argsort(self.scores[ids, j], kind="stable")]
                    starts = np.r_[0, np.flatnonzero(np.diff(self.scores[order, j])) + 1]
                    group.append((order, starts))
                self.groups.append(group)
        else:
            first, second = np.triu_indices(len(self.y), 1)
            keep = (self.y[first] != self.y[second]) & (self.strata[first] == self.strata[second])
            self.first, self.second = first[keep], second[keep]
            dy = np.sign(self.y[self.first] - self.y[self.second])[:, None]
            ds = np.sign(self.scores[self.first] - self.scores[self.second])
            self.pair_score = np.where(ds * dy > 0, 1.0, np.where(ds == 0, 0.5, 0.0))

    def calculate(self, weights):
        w = np.asarray(weights, float)
        if not np.isfinite(w).all() or np.any(w < 0):
            raise ValueError("Metric weights must be finite and nonnegative")
        if self.binary:
            numerator = np.zeros(self.scores.shape[1])
            denominator = 0.0
            for group in self.groups:
                ids = group[0][0]
                denominator += w[ids][self.y[ids] == 1].sum() * w[ids][self.y[ids] == 0].sum()
                for j, (ids, starts) in enumerate(group):
                    pos = np.add.reduceat(w[ids] * self.y[ids], starts)
                    neg = np.add.reduceat(w[ids] * (1 - self.y[ids]), starts)
                    numerator[j] += np.dot(pos, np.cumsum(neg) - 0.5 * neg)
            return numerator / denominator if denominator > 0 else np.full(len(numerator), np.nan)
        mass = w[self.first] * w[self.second]
        return (np.einsum("ij,i->j", self.pair_score, mass) / mass.sum()
                if mass.sum() > 0 else np.full(self.scores.shape[1], np.nan))


def household_multipliers(frame: pd.DataFrame, rng) -> np.ndarray:
    """Paired resampling of households independently within each country."""
    result = np.zeros(len(frame), float)
    for _, rows in frame.reset_index(drop=True).groupby("country_name", sort=True):
        households = rows["hhid"].astype(str)
        unique = np.sort(households.unique())
        counts = np.bincount(rng.integers(0, len(unique), size=len(unique)), minlength=len(unique))
        mapping = dict(zip(unique, counts, strict=True))
        result[rows.index] = households.map(mapping).to_numpy(float)
    return result


ASSISTANCE_COLUMNS = tuple(f"hibl005_{index}" for index in range(1, 15))

def informant_assistance_fraction(frame: pd.DataFrame) -> pd.Series:
    """Fraction of 14 daily activities requiring help, with >=10 valid items."""
    components = pd.DataFrame(
        {
            column: pd.to_numeric(frame[column], errors="coerce").map(
                {1.0: 1.0, 5.0: 0.0}
            )
            for column in ASSISTANCE_COLUMNS
        },
        index=frame.index,
    )
    result = components.mean(axis=1, skipna=True)
    result.loc[components.notna().sum(axis=1) < 10] = np.nan
    return result
