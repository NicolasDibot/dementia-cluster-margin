"""Compare margin change and cluster entry on identical participant-periods.

The two outcomes are different estimands, not interchangeable measures of
statistical power.  Neither a ratio of z statistics nor a count of significant
associations is converted into an equivalent sample size here.

Input is a prepared panel with a frozen two-cluster geometry. Participant keys
are used only in memory. The returned tables contain aggregate estimates only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm


@dataclass(frozen=True)
class Factor:
    name: str
    label: str
    adjust_education: bool = True


FACTORS = (
    Factor("less_education", "Less education", False),
    Factor("hearing_loss", "Hearing loss"),
    Factor("poor_vision", "Poor vision"),
    Factor("depression", "Depression"),
    Factor("physical_inactivity", "Physical inactivity"),
    Factor("diabetes", "Diabetes"),
    Factor("hypertension", "Hypertension"),
    Factor("high_cholesterol", "High cholesterol"),
    Factor("smoking", "Current smoking"),
    Factor("obesity", "Obesity"),
    Factor("excessive_alcohol", "Excessive alcohol"),
    Factor("no_social_activity", "No social activity"),
)
KEYS = ("cohort", "participant_id", "household_id", "country", "interval")
NUMERIC = (
    "age", "sex", "education", "weight", "hard_baseline", "hard_followup",
    "margin_baseline", "margin_followup",
)


def prepare_panel(panel: pd.DataFrame, *, geometry_n_clusters: int = 2) -> pd.DataFrame:
    """Validate semantics before selection; never select on follow-up state."""
    if geometry_n_clusters != 2:
        raise ValueError("This comparison requires the two-cluster geometry")
    required = {*KEYS, *NUMERIC}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Missing prepared-panel columns: {sorted(missing)}")
    out = panel.copy()
    if not set(out["cohort"].dropna()).issubset({"HRS", "SHARE"}):
        raise ValueError("Only HRS and SHARE are supported")
    if out.duplicated(["cohort", "participant_id", "interval"]).any():
        raise ValueError("Duplicate participant-period records")
    for name in NUMERIC:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    for suffix in ("baseline", "followup"):
        hard, margin = out[f"hard_{suffix}"], out[f"margin_{suffix}"]
        if not hard.dropna().isin([0, 1]).all():
            raise ValueError("Hard state must be binary")
        observed = hard.notna() & margin.notna()
        if not hard[observed].eq(margin[observed].gt(0).astype(int)).all():
            raise ValueError("Hard state disagrees with the signed margin")
    out["delta_margin"] = out["margin_followup"] - out["margin_baseline"]
    # This is a risk set, not the H=0 -> H=0 subgroup. Entrants remain included.
    eligible = out["hard_baseline"].eq(0) & out["age"].ge(55)
    eligible &= out[list(KEYS)].notna().all(axis=1)
    eligible &= np.isfinite(out[[
        "age", "sex", "weight", "hard_followup", "margin_baseline",
        "margin_followup", "delta_margin",
    ]]).all(axis=1)
    eligible &= out["weight"].gt(0)
    out = out.loc[eligible].copy()
    out["cluster_entry"] = out["hard_followup"].astype(int)
    # Prefix by country/cohort to prevent accidental pooling of household IDs.
    out["cluster_key"] = pd.MultiIndex.from_frame(
        out[["cohort", "country", "household_id"]].astype(str)
    ).factorize()[0]
    return out


def _design(frame: pd.DataFrame, factor: Factor, categorical_education: bool):
    x = pd.DataFrame(index=frame.index)
    x["intercept"] = 1.0
    x["exposure"] = frame[factor.name].astype(float)
    x["age"] = (frame["age"] - frame["age"].mean()) / 10
    x["age_squared"] = x["age"] ** 2
    x["sex"] = frame["sex"]
    margin = frame["margin_baseline"]
    sd = margin.std(ddof=0)
    x["margin_baseline"] = (margin - margin.mean()) / (sd if sd > 0 else 1)
    categories = ["interval"]
    if frame["cohort"].iloc[0] == "SHARE":
        categories.append("country")
    if factor.adjust_education:
        if categorical_education:
            categories.append("education")
        else:
            x["education"] = frame["education"]
    x = pd.concat([x, pd.get_dummies(
        frame[categories].astype(str), drop_first=True, dtype=float,
    )], axis=1)
    # A constant nuisance term is redundant, but a constant exposure is invalid.
    if frame[factor.name].nunique() != 2:
        raise ValueError("Exposure does not have both binary levels")
    keep = [c for c in x if c == "intercept" or x[c].nunique() > 1]
    x = x[keep]
    values = x.to_numpy(float)
    if not np.isfinite(values).all() or np.linalg.matrix_rank(values) != values.shape[1]:
        raise ValueError("Nonfinite or rank-deficient design")
    return values, list(x.columns)


def _cluster_covariance(x, residual_score, bread, clusters):
    _, inverse = np.unique(clusters, return_inverse=True)
    g = int(inverse.max()) + 1
    n, p = x.shape
    if g < 2 or n <= p:
        raise ValueError("Insufficient observations or household clusters")
    scores = np.zeros((g, p))
    np.add.at(scores, inverse, x * residual_score[:, None])
    meat = np.einsum("ni,nj->ij", scores, scores)
    correction = g / (g - 1) * (n - 1) / (n - p)
    covariance = correction * np.einsum("ij,jk,lk->il", bread, meat, bread)
    if not np.isfinite(covariance).all():
        raise FloatingPointError("Nonfinite cluster covariance")
    return (covariance + covariance.T) / 2


def fit_weighted_linear(x, y, weight, clusters):
    """WLS with household-cluster sandwich and CR1 finite-sample correction."""
    weight = np.asarray(weight, float) / np.mean(weight)
    information = np.einsum("ni,n,nj->ij", x, weight, x)
    bread = np.linalg.inv(information)
    beta = np.linalg.solve(information, np.einsum("ni,n,n->i", x, weight, y))
    residual = y - np.einsum("ni,i->n", x, beta)
    return beta, _cluster_covariance(x, weight * residual, bread, clusters)


def fit_weighted_logistic(x, y, weight, clusters, *, max_iter=100):
    """Survey-weighted logit; fail visibly on separation or nonconvergence."""
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Cluster entry must contain events and non-events")
    weight = np.asarray(weight, float) / np.mean(weight)
    beta = np.zeros(x.shape[1])

    def loglik(value):
        eta = np.einsum("ni,i->n", x, value)
        return np.sum(weight * (y * eta - np.logaddexp(0, eta)))

    for _ in range(max_iter):
        probability = expit(np.einsum("ni,i->n", x, beta))
        information = np.einsum("ni,n,nj->ij", x, weight * probability * (1 - probability), x)
        step = np.linalg.solve(information, np.einsum("ni,n->i", x, weight * (y - probability)))
        old = loglik(beta)
        for _ in range(30):
            candidate = beta + step
            if loglik(candidate) >= old - 1e-10:
                break
            step *= 0.5
        else:
            raise RuntimeError("Logistic line search failed")
        if not np.isfinite(candidate).all() or np.max(np.abs(candidate)) > 40:
            raise RuntimeError("Logistic separation or numerical divergence")
        beta = candidate
        if np.max(np.abs(step)) < 1e-9:
            break
    else:
        raise RuntimeError("Weighted logistic regression did not converge")
    probability = expit(np.einsum("ni,i->n", x, beta))
    information = np.einsum("ni,n,nj->ij", x, weight * probability * (1 - probability), x)
    bread = np.linalg.inv(information)
    return beta, _cluster_covariance(x, weight * (y - probability), bread, clusters)


def compare_common_population(
    panel: pd.DataFrame,
    factors: Iterable[Factor] = FACTORS,
    *,
    geometry_n_clusters: int = 2,
    delta_scales: Mapping[str, float] | None = None,
    education_categorical: Iterable[str] = ("HRS",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return effects and flow diagnostics without any participant-level output.

    Each factor has its own complete-case population, but both primary models
    use exactly the same rows, weights and covariates for that factor. Missing
    exposure is not recoded as absence. SDs are fixed within cohort before
    factor-specific missingness. Bonferroni uses the full factor family, including
    factors that cannot be estimated, separately within each outcome/cohort.
    """
    factors = tuple(factors)
    if not factors or len({f.name for f in factors}) != len(factors):
        raise ValueError("A nonempty set of distinct factors is required")
    core = prepare_panel(panel, geometry_n_clusters=geometry_n_clusters)
    estimates, flow = [], []
    for cohort, cohort_frame in core.groupby("cohort", sort=True):
        scale = (delta_scales or {}).get(cohort, cohort_frame["delta_margin"].std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Margin-change SD must be positive")
        for factor in factors:
            base = {"cohort": cohort, "factor": factor.name, "label": factor.label,
                    "n_base_risk_set": len(cohort_frame), "delta_margin_sd": float(scale)}
            if factor.name not in cohort_frame:
                flow.append({**base, "status": "factor_unavailable", "n_common": 0})
                continue
            values = pd.to_numeric(cohort_frame[factor.name], errors="coerce")
            if not values.notna().any():
                flow.append({**base, "status": "factor_unavailable", "n_common": 0})
                continue
            if not values.dropna().isin([0, 1]).all():
                raise ValueError(f"{factor.name} must be binary with missing preserved")
            eligible = values.notna()
            if factor.adjust_education:
                eligible &= np.isfinite(cohort_frame["education"])
            weight_column = f"weight_{factor.name}"
            if weight_column in cohort_frame:
                values_weight = pd.to_numeric(cohort_frame[weight_column], errors="coerce")
                eligible &= np.isfinite(values_weight) & values_weight.gt(0)
            frame = cohort_frame.loc[eligible].copy()
            frame[factor.name] = values[eligible]
            if weight_column in frame:
                frame["weight"] = values_weight[eligible]
            diagnostic = {**base, "n_common": len(frame),
                          "n_entries": int(frame["cluster_entry"].sum()),
                          "n_stayers": int(frame["cluster_entry"].eq(0).sum()),
                          "participants": frame["participant_id"].nunique(),
                          "households": frame["cluster_key"].nunique()}
            if len(frame) < 20 or frame[factor.name].nunique() != 2:
                flow.append({**diagnostic, "status": "insufficient_or_constant_exposure"})
                continue
            x, terms = _design(frame, factor, cohort in set(education_categorical))
            exposure_index = terms.index("exposure")
            group = frame["cluster_key"].to_numpy()
            weights = frame["weight"].to_numpy(float)
            specifications = [
                ("common_population", "margin_change", frame, x, weights, group, terms),
                ("common_population", "cluster_entry", frame, x, weights, group, terms),
            ]
            statuses = []
            for population, outcome, selected, design, w, g, model_terms in specifications:
                effect_base = {**diagnostic, "population": population, "outcome": outcome,
                               "n": len(selected), "entries": int(selected["cluster_entry"].sum()),
                               "weight_sum": float(np.sum(w)), "terms": ";".join(model_terms),
                               "comparison_is_power_test": False}
                try:
                    if outcome == "margin_change":
                        beta, covariance = fit_weighted_linear(design, selected["delta_margin"].to_numpy(float) / scale, w, g)
                    else:
                        beta, covariance = fit_weighted_logistic(design, selected["cluster_entry"].to_numpy(float), w, g)
                    value = float(beta[exposure_index])
                    se = float(np.sqrt(covariance[exposure_index, exposure_index]))
                    if not np.isfinite(se) or se <= 0:
                        raise FloatingPointError("Invalid standard error")
                    low, high = value - 1.95996398454 * se, value + 1.95996398454 * se
                    p = float(2 * norm.sf(abs(value / se)))
                    transform = np.exp if outcome == "cluster_entry" else float
                    estimates.append({**effect_base, "status": "ok", "coefficient": value,
                                      "standard_error": se, "estimate": transform(value),
                                      "ci95_low": transform(low), "ci95_high": transform(high),
                                      "p_value": p, "p_bonferroni": min(1.0, p * len(factors)),
                                      "effect_scale": "odds_ratio" if outcome == "cluster_entry" else "SD_change"})
                except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                    statuses.append(f"{population}/{outcome}: {error}")
                    estimates.append({**effect_base, "status": str(error)})
            flow.append({**diagnostic, "status": "; ".join(statuses) or "ok"})
    return pd.DataFrame(estimates), pd.DataFrame(flow)
