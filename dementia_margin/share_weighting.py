"""SHARE-HCAP discrimination under released and entropy-balancing weights.

The finite target is fixed before assessment outcomes are used. Its weighted
Wave-7 moments, means for missing-value replacement and standard deviations
are held fixed during household resampling. Entropy weights are fitted anew
in every draw, starting from the released HCAP design weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from .rank_metrics import CachedRankMetric, household_multipliers

COUNTRIES = ("Denmark", "France", "Germany", "Italy")
BALANCE_FEATURES = (
    "age_w7", "age_squared_w7", "female_w7", "education_w7",
    "cognition_w7", "function_w7", "margin_w7", "label_w7", "proxy_w7",
    "self_rated_health_w7", "chronic_conditions_w7", "depression_w7",
)
REGIMES = ("unweighted", "official_cpsw", "entropy_transport")
ENTROPY_RIDGE = 1e-3


def positive_weights(weights):
    result = np.asarray(weights, float)
    if result.ndim != 1 or not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("Weights must be finite and positive")
    return result / result.mean()


@dataclass(frozen=True)
class BalanceSpecification:
    fills: dict[str, float]
    missing: tuple[str, ...]
    centers: dict[str, float]
    scales: dict[str, float]
    terms: tuple[str, ...]

    def transform(self, frame):
        raw = balance_terms(frame, self.fills, self.missing)
        return np.column_stack([
            (raw[name].to_numpy(float) - self.centers[name]) / self.scales[name]
            for name in self.terms
        ])


def balance_terms(frame, fills, missing):
    """Construct only specified baseline features, never HCAP outcomes."""
    out = pd.DataFrame(index=frame.index)
    for feature in BALANCE_FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[feature] = values.fillna(fills[feature])
        if feature in missing:
            out[f"{feature}__missing"] = values.isna().astype(float)
    if not frame.country_name.isin(COUNTRIES).all():
        raise ValueError("Unexpected country in the SHARE-HCAP balance population")
    # One reference country avoids the sum-to-one redundancy.
    for country in COUNTRIES[1:]:
        out[f"country__{country}"] = frame.country_name.eq(country).astype(float)
    return out


def fit_balance_specification(target):
    weights = positive_weights(target.cciw_w9)
    fills, missing = {}, []
    for feature in BALANCE_FEATURES:
        values = pd.to_numeric(target[feature], errors="coerce").to_numpy(float)
        valid = np.isfinite(values)
        if not valid.any():
            raise ValueError(f"Target feature is entirely missing: {feature}")
        fills[feature] = float(np.average(values[valid], weights=weights[valid]))
        if not valid.all():
            missing.append(feature)
    raw = balance_terms(target, fills, tuple(missing))
    centers, scales = {}, {}
    for name in raw:
        values = raw[name].to_numpy(float)
        mean = float(np.average(values, weights=weights))
        sd = float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))
        if sd > 1e-8:
            centers[name], scales[name] = mean, sd
    if not centers:
        raise ValueError("No varying target characteristics")
    return BalanceSpecification(fills, tuple(missing), centers, scales, tuple(centers))


def entropy_balance(design, base_weights, *, ridge=ENTROPY_RIDGE):
    """Exponential tilting towards zero-centered target moments.

    Zero base weights represent households absent from a bootstrap draw. The
    small, fixed ridge makes the dual finite when a rare baseline category is
    absent from a resample. The achieved imbalance is always returned.
    """
    matrix = np.asarray(design, float)
    base = np.asarray(base_weights, float)
    if (matrix.ndim != 2 or matrix.shape[0] != len(base)
            or not np.isfinite(matrix).all() or not np.isfinite(base).all()
            or (base < 0).any() or not (base > 0).any() or ridge < 0):
        raise ValueError("Invalid entropy-balancing inputs")
    active = base > 0
    x = matrix[active]
    log_base = np.log(base[active] / base[active].mean())

    def objective(beta):
        linear = log_base + np.einsum("ij,j->i", x, beta)
        norm = logsumexp(linear)
        probabilities = np.exp(linear - norm)
        value = norm + .5 * ridge * np.dot(beta, beta)
        gradient = np.einsum("ij,i->j", x, probabilities) + ridge * beta
        return float(value), gradient

    fit = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                   bounds=[(-30., 30.)] * x.shape[1],
                   options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-9, "maxls": 80})
    linear = log_base + np.einsum("ij,j->i", x, fit.x)
    probability = np.exp(linear - logsumexp(linear))
    result = np.zeros(len(base))
    result[active] = probability * len(base)
    moment_gap = float(np.max(np.abs(np.einsum("ij,i->j", x, probability))))
    if not np.isfinite(result).all() or not (result[active] > 0).all():
        raise RuntimeError("Entropy weights are not finite and positive")
    acceptable = bool(fit.success or moment_gap < 1e-5)
    return result, {
        "optimizer_acceptable": acceptable, "optimizer_success": bool(fit.success),
        "iterations": int(fit.nit), "max_standardized_moment_gap": moment_gap,
        "effective_sample_size": float(result.sum() ** 2 / np.square(result).sum()),
        "ridge": ridge,
    }


def evaluate_weighting(linked, target, outcome, *, binary, replicates=1000, seed=20261109):
    """Return paired label/margin discrimination for each weighting scheme."""
    if replicates < 2:
        raise ValueError("At least two bootstrap replicates are required")
    if linked.mergeid.duplicated().any() or target.mergeid.duplicated().any():
        raise ValueError("Assessment and target identifiers must be unique")
    complete = linked.loc[linked[outcome].notna()].copy().reset_index(drop=True)
    if not set(complete.mergeid).issubset(set(target.mergeid)):
        raise ValueError("Outcome participants are not a subset of the finite W9 target")
    required = [outcome, "hard_two_cluster", "margin_two_cluster", "dw_hcap", "cpsw_hcap"]
    if (complete[required].isna().any().any()
            or complete[["hhid", "country_name"]].isna().any().any()):
        raise ValueError("Incomplete predictors, design weights or household identifiers")
    if not np.array_equal(complete.margin_two_cluster > 0, complete.hard_two_cluster == 1):
        raise ValueError("Cluster label must equal the sign of the margin")
    spec = fit_balance_specification(target)
    design = spec.transform(complete)
    base = positive_weights(complete.dw_hcap)
    official = positive_weights(complete.cpsw_hcap)
    entropy, entropy_info = entropy_balance(design, base)
    if not entropy_info["optimizer_acceptable"]:
        raise RuntimeError("Point-sample entropy fit did not converge")
    weights = (np.ones(len(complete)), official, entropy)
    metric = CachedRankMetric(complete[outcome],
                              complete[["hard_two_cluster", "margin_two_cluster"]],
                              complete.country_name, binary=binary)
    point = np.vstack([metric.calculate(weight) for weight in weights])
    if not np.isfinite(point).all():
        raise ValueError("No comparable within-country outcome pairs")
    rng = np.random.default_rng(seed)
    draws, gaps, attempts, rejected = [], [], 0, 0
    while len(draws) < replicates and attempts < int(np.ceil(1.25 * replicates)):
        attempts += 1
        multiplier = household_multipliers(complete, rng)
        entropy_draw, info = entropy_balance(design, base * multiplier)
        if not info["optimizer_acceptable"]:
            rejected += 1
            continue
        current = np.vstack([metric.calculate(multiplier),
                             metric.calculate(official * multiplier),
                             metric.calculate(entropy_draw)])
        if np.isfinite(current).all():
            draws.append(current)
            gaps.append(info["max_standardized_moment_gap"])
    if len(draws) != replicates:
        raise RuntimeError("Too few valid paired household bootstrap draws")
    boot = np.asarray(draws)
    rows = []
    for k, regime in enumerate(REGIMES):
        estimates = [point[k, 0], point[k, 1], point[k, 1] - point[k, 0]]
        columns = [boot[:, k, 0], boot[:, k, 1], boot[:, k, 1] - boot[:, k, 0]]
        for name, estimate, values in zip(("label", "margin", "margin_minus_label"), estimates, columns):
            lo, hi = np.quantile(values, [.025, .975])
            rows.append({"outcome": outcome, "metric": "auc" if binary else "concordance",
                         "regime": regime, "estimand": name, "estimate": float(estimate),
                         "ci95_low": float(lo), "ci95_high": float(hi),
                         "n": len(complete), "events": int(complete[outcome].sum()) if binary else None})
    metadata = {"outcome": outcome, "sample_n": len(complete), "target_n": len(target),
                "replicates": replicates, "seed": seed, "attempted_draws": attempts,
                "entropy_rejected_draws": rejected, "entropy_point": entropy_info,
                "bootstrap_max_moment_gap": float(max(gaps)), "terms": list(spec.terms),
                "target_moments_fixed": True, "entropy_refit_in_every_draw": True,
                "bootstrap_unit": "household within country", "pair_strata": "country"}
    return pd.DataFrame(rows), metadata
