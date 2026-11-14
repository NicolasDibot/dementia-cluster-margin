"""Fit margin-change and cluster-entry models on identical populations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .io import digest, external, fresh_output, require_data_access
from .risk_factors import FACTORS, compare_common_population
from .risk_panel import build_hrs_panel, build_share_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("hrs-scores", "share-scores", "rand", "harmonized-hrs",
                 "harmonized-share", "share-raw", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    require_data_access()
    paths = {name: external(path) for name, path in vars(args).items()}
    if paths["output"].exists():
        raise FileExistsError("Use a new output directory")
    panels = {
        "HRS": build_hrs_panel(paths["hrs_scores"], paths["rand"], paths["harmonized_hrs"]),
        "SHARE": build_share_panel(paths["share_scores"], paths["harmonized_share"], paths["share_raw"]),
    }
    effects, flows = [], []
    for panel in panels.values():
        estimates, flow = compare_common_population(panel, FACTORS)
        effects.append(estimates)
        flows.append(flow)
    effects = pd.concat(effects, ignore_index=True)
    flow = pd.concat(flows, ignore_index=True)
    output = fresh_output(paths["output"])
    effects.to_csv(output / "common_population_effects.csv", index=False)
    flow.to_csv(output / "common_population_flow.csv", index=False)
    paired = effects.pivot(index=["cohort", "factor", "label"], columns="outcome",
                           values=["estimate", "ci95_low", "ci95_high", "p_value", "p_bonferroni", "n", "entries"])
    paired.columns = [f"{outcome}_{field}" for field, outcome in paired.columns]
    paired.reset_index().to_csv(output / "common_population_paired_table.csv", index=False)
    counts = []
    for (cohort, outcome), frame in effects.groupby(["cohort", "outcome"]):
        valid = frame.status.eq("ok")
        counts.append({"cohort": cohort, "outcome": outcome,
                       "factors_estimated": int(valid.sum()),
                       "bonferroni_significant": int((frame.loc[valid, "p_bonferroni"] < .05).sum())})
    pd.DataFrame(counts).to_csv(output / "association_counts.csv", index=False)
    metadata = {
        "population": "baseline H=0, age>=55; factor-specific complete cases, entrants retained",
        "uncertainty": "household-cluster CR1 sandwich; normal-Wald intervals and tests",
        "bonferroni_family": 12,
        "counts_interpretation": "descriptive, not a comparison of statistical power",
        "input_sha256": {name: digest(path) for name, path in paths.items()
                         if name not in ("output", "share_raw")},
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
