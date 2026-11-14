"""Evaluate fixed two-cluster scores against ADAMS or HCAP classifications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import digest, external, fresh_output, require_data_access
from .statistics import (
    _load_adams, _load_hcap, _binary_nested_estimands, _ordinal_nested_estimands,
    _stratified_auc, _survey_multiplicity, _prepare_association_predictors,
    _survey_logistic_association, _screen_association_table,
)


def estimands(frame, weights, cohort):
    strata = frame.landmark_wave_tracker.to_numpy() if cohort == "adams" else np.zeros(len(frame))
    result = _binary_nested_estimands(
        frame, weights, strata=strata, hard_column="hard_current",
        margin_column="margin_current", hard_name="label", margin_name="margin",
        contrast_name="margin_minus_label", family="two_cluster",
    )
    result.update(_ordinal_nested_estimands(frame, weights, strata=strata))
    for name in ("cognition_only", "function_only"):
        value = _stratified_auc(frame.outcome.to_numpy(), frame[name].to_numpy(), weights, strata)
        result[f"auc::{name}"] = value
        result[f"contrast::margin_minus_{name}"] = result["auc::margin"] - value
    return result


def load_evaluation(cohort, scores, respondent, tracker, summary=None):
    """Link fixed landmark scores to the eligible provider reference sample."""
    geometry = pd.read_csv(scores, dtype={"key": str}).rename(
        columns={"margin_two_cluster": "margin_current", "hard_two_cluster": "hard_current"})
    if not np.array_equal(geometry.margin_current > 0, geometry.hard_current == 1):
        raise ValueError("Labels must equal the sign of the margin")
    if cohort == "adams":
        return _load_adams(geometry, respondent, tracker)
    if cohort == "hcap" and summary is not None:
        return _load_hcap(geometry, respondent, summary, tracker)
    raise ValueError("A valid cohort and its reference files are required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=["adams", "hcap"], required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--respondent", type=Path, required=True)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    require_data_access()
    if args.bootstrap < 20:
        parser.error("At least 20 bootstrap replicates are required")
    if args.cohort == "hcap" and args.summary is None:
        parser.error("HCAP requires --summary")
    paths = {name: external(getattr(args, name)) for name in ("scores", "respondent", "tracker")}
    if args.summary is not None:
        paths["summary"] = external(args.summary)
    output = fresh_output(args.output)
    frame, design = load_evaluation(args.cohort, paths["scores"], paths["respondent"],
                                   paths["tracker"], paths.get("summary"))
    weight = frame.weight.to_numpy(float)
    point = estimands(frame, weight, args.cohort)
    rng = np.random.default_rng(args.seed)
    draws = []
    for _ in range(args.bootstrap):
        w = weight * _survey_multiplicity(design, rng).reindex(frame.key).to_numpy(float)
        try:
            value = estimands(frame, w, args.cohort)
            if np.isfinite(list(value.values())).all():
                draws.append(value)
        except ValueError:
            continue
    if len(draws) < .95 * args.bootstrap:
        raise RuntimeError("Fewer than 95% of bootstrap replicates succeeded")
    boot = pd.DataFrame(draws)
    pd.DataFrame([{"cohort": args.cohort, "estimand": name, "estimate": value,
                   "ci95_low": boot[name].quantile(.025), "ci95_high": boot[name].quantile(.975)}
                  for name, value in point.items()]).to_csv(output / "clinical_estimands.csv", index=False)
    distribution = []
    for level in sorted(frame.outcome_ordinal.unique()):
        selected = frame.outcome_ordinal.eq(level).to_numpy()
        inside = np.average(frame.hard_current.to_numpy()[selected], weights=weight[selected])
        distribution.append({"cohort": args.cohort, "diagnostic_level": int(level),
                             "percentage_outside": 100 * (1 - inside), "percentage_inside": 100 * inside})
    pd.DataFrame(distribution).to_csv(output / "diagnosis_distribution.csv", index=False)
    association, diagnostic = _survey_logistic_association(_prepare_association_predictors(frame), design, cohort=args.cohort)
    _screen_association_table(association).to_csv(output / "clinical_association.csv", index=False)
    (output / "analysis_metadata.json").write_text(json.dumps({
        "cohort": args.cohort, "seed": args.seed, "successful_replicates": len(draws),
        "participants": len(frame), "dementia": int(frame.outcome.sum()),
        "conditioning": "factor coordinates and two-cluster geometry fixed",
        "association": diagnostic, "input_sha256": {name: digest(path) for name, path in paths.items()},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
