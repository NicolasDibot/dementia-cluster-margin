"""Link SHARE Wave-7 scores to released HCAP outcomes and weighting inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import digest, external, fresh_output, require_data_access
from .rank_metrics import (ASSISTANCE_COLUMNS, delayed_memory_impairment,
                           informant_assistance_fraction)
from .share_weighting import evaluate_weighting

COUNTRY_NAMES = {12: "Germany", 16: "Italy", 17: "France", 18: "Denmark"}
MEMORY_COLUMNS = ("WR_delayed_score", "LM_Story1_DelayedRecall_score",
                  "LM_Story2_DelayedRecall_score")
OUTCOMES = {"hcap_severe_cognitive_impairment": True,
            "informant_assistance_fraction": False, "delayed_memory_impairment": False}
EASY_COLUMNS = ("mergeid", "hhid", "wave", "country", "age", "female", "eduyears_mod",
                "sphus", "chronic_mod", "eurod", "int_year", "int_month")
RESPONDENT_COLUMNS = ("mergeid", "country", "class_cognition_pooled", "dw_hcap", "cpsw_hcap",
                      "int_year_hcap_r", "int_month_hcap_r", *MEMORY_COLUMNS)


def valid_range(series, lower, upper):
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values.between(lower, upper))


def unique(frame, name):
    if frame.mergeid.isna().any() or frame.mergeid.astype(str).eq("").any() or frame.mergeid.duplicated().any():
        raise ValueError(f"{name} requires unique nonmissing participant identifiers")


def construct_frames(scores, easy, proxy, respondent, informant, coverscreen, weights):
    """Construct the landmark, assessment and finite-target frames in memory."""
    scores = scores.loc[pd.to_numeric(scores.wave, errors="coerce").eq(7)].copy()
    easy = easy.loc[pd.to_numeric(easy.wave, errors="coerce").eq(7)].copy()
    for frame, name in [(scores, "Wave-7 scores"), (easy, "EasySHARE Wave 7"),
                        (proxy, "Harmonized SHARE"), (respondent, "HCAP respondent"),
                        (informant, "HCAP informant"), (coverscreen, "Wave-9 coverscreen"),
                        (weights, "Wave-9 weights")]:
        unique(frame, name)
    keep_scores = ["mergeid", "margin_two_cluster", "hard_two_cluster", "cognition_only", "function_only"]
    source = scores[keep_scores].merge(easy[list(EASY_COLUMNS)], on="mergeid", how="left",
                                       validate="one_to_one", indicator=True)
    unmatched = int(source._merge.ne("both").sum())
    if unmatched > max(10, int(np.ceil(.01 * len(scores)))):
        raise ValueError("Wave-7 score linkage to EasySHARE is incomplete")
    source = source.loc[source._merge.eq("both")].drop(columns="_merge")
    source["age_w7"] = valid_range(source.age, 60, 110)
    source = source.loc[source.country.isin(COUNTRY_NAMES) & source.age_w7.notna()].copy()
    source["country_name"] = source.country.map(COUNTRY_NAMES)
    source = source.merge(proxy[["mergeid", "r7proxy"]], on="mergeid", how="left", validate="one_to_one")
    source["age_squared_w7"] = source.age_w7 ** 2
    source["female_w7"] = valid_range(source.female, 0, 1)
    source["education_w7"] = valid_range(source.eduyears_mod, 0, 30)
    source["cognition_w7"] = source.cognition_only
    source["function_w7"] = source.function_only
    source["margin_w7"] = source.margin_two_cluster
    source["label_w7"] = source.hard_two_cluster
    source["proxy_w7"] = valid_range(source.r7proxy, 0, 1)
    source["self_rated_health_w7"] = valid_range(source.sphus, 1, 5)
    source["chronic_conditions_w7"] = valid_range(source.chronic_mod, 0, 100)
    source["depression_w7"] = valid_range(source.eurod, 0, 12)
    if source[["hard_two_cluster", "margin_two_cluster"]].isna().any().any():
        raise ValueError("Landmark scores are incomplete")
    if not np.array_equal(source.margin_two_cluster > 0, source.hard_two_cluster == 1):
        raise ValueError("Cluster label must equal the sign of the margin")
    linked = source.merge(respondent[list(RESPONDENT_COLUMNS)], on="mergeid", how="inner",
                          validate="one_to_one", suffixes=("_w7", "_hcap"))
    linked = linked.merge(informant[["mergeid", *ASSISTANCE_COLUMNS]], on="mergeid", how="left",
                          validate="one_to_one")
    if not linked.country_w7.eq(linked.country_hcap).all():
        raise ValueError("Country mismatch between the landmark and HCAP assessment")
    classification = pd.to_numeric(linked.class_cognition_pooled, errors="coerce")
    linked["hcap_severe_cognitive_impairment"] = classification.eq(3).astype(float).where(classification.isin([1, 2, 3]))
    linked["informant_assistance_fraction"] = informant_assistance_fraction(linked)
    linked["delayed_memory_impairment"] = delayed_memory_impairment(linked)
    elapsed = 12 * (linked.int_year_hcap_r - linked.int_year) + linked.int_month_hcap_r - linked.int_month
    linked["wave7_to_hcap_years"] = elapsed / 12
    target = source.merge(coverscreen[["mergeid", "deceased", "interview"]], on="mergeid", how="left",
                          validate="one_to_one")
    target = target.loc[target.deceased.eq(0) & target.interview.eq(1)].copy()
    target = target.merge(weights[["mergeid", "cciw_w9"]], on="mergeid", how="left", validate="one_to_one")
    if target.cciw_w9.isna().any() or not np.isfinite(target.cciw_w9).all() or target.cciw_w9.le(0).any():
        raise ValueError("A finite-target participant lacks a valid positive Wave-9 weight")
    metadata = {"source_population_n": len(source), "wave7_scores_n": len(scores),
                "unmatched_easyshare_n": unmatched, "linked_hcap_n": len(linked), "target_n": len(target),
                "linked_outside_finite_target_n": len(set(linked.mergeid) - set(target.mergeid)),
                "median_followup_years": float(linked.wave7_to_hcap_years.median()),
                "outcome_available": {name: int(linked[name].notna().sum()) for name in OUTCOMES},
                "memory_standardization": "All linked HCAP participants, before requiring >=2 of 3 tests",
                "finite_target": "Wave-7 four-country age>=60 domain, alive and main-interviewed at Wave 9",
                "target_weight": "cciw_w9", "entropy_base_weight": "dw_hcap",
                "propensity_weight": "Released cpsw_hcap, not estimated using outcomes"}
    return linked.reset_index(drop=True), target.reset_index(drop=True), metadata


def assemble_frames(scores_path, easyshare_path, harmonized_share_path, respondent_path,
                    informant_path, coverscreen_path, weights_path):
    scores = pd.read_csv(scores_path, dtype={"mergeid": str})
    files = [(easyshare_path, EASY_COLUMNS), (harmonized_share_path, ["mergeid", "r7proxy"]),
             (respondent_path, RESPONDENT_COLUMNS), (informant_path, ["mergeid", *ASSISTANCE_COLUMNS]),
             (coverscreen_path, ["mergeid", "deceased", "interview"]), (weights_path, ["mergeid", "cciw_w9"])]
    frames = [pd.read_stata(path, columns=list(columns), convert_categoricals=False) for path, columns in files]
    return construct_frames(scores, *frames)


def run(config):
    require_data_access()
    names = ("scores", "easyshare", "harmonized_share", "hcap_respondent", "hcap_informant",
             "wave9_coverscreen", "wave9_weights")
    paths = {name: external(config[name]) for name in names}
    linked, target, metadata = assemble_frames(*(paths[name] for name in names))
    output = fresh_output(config["output"])
    estimates, diagnostics = [], {}
    for index, (outcome, binary) in enumerate(OUTCOMES.items()):
        rows, info = evaluate_weighting(linked, target, outcome, binary=binary,
                                        replicates=config.get("bootstrap", 1000),
                                        seed=config.get("seed", 20261109) + index)
        estimates.append(rows)
        diagnostics[outcome] = info
    result = pd.concat(estimates, ignore_index=True)
    result.to_csv(output / "share_hcap_estimands.csv", index=False)
    result.loc[result.regime.eq("unweighted")].to_csv(output / "prospective_estimands.csv", index=False)
    result.loc[result.estimand.eq("margin_minus_label")].to_csv(output / "supplementary_table_1.csv", index=False)
    # Licensed participant records remain outside Git with restrictive permissions.
    linked.to_csv(output / "evaluation_private.csv", index=False)
    target.to_csv(output / "target_private.csv", index=False)
    metadata.update({"evaluation": diagnostics, "input_sha256": {name: digest(path) for name, path in paths.items()}})
    (output / "share_hcap_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return result, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scores", "easyshare", "harmonized-share", "hcap-respondent", "hcap-informant",
                 "wave9-coverscreen", "wave9-weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    if args.bootstrap < 20:
        parser.error("At least 20 bootstrap replicates are required")
    run(vars(args))


if __name__ == "__main__":
    main()
