"""Refit clustering and evaluate stability in clinically assessed participants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .clinical import load_evaluation
from .fit import consolidate, read_indexed
from .io import digest, external, fresh_output, require_data_access
from .stability import source_domain_burdens, stability_refits, summarize_stability


def run(cohort, paths, output, *, repeats=100, seed=20261109, rscript="Rscript"):
    require_data_access()
    paths = {name: external(path) for name, path in paths.items()}
    source = read_indexed(paths["source_coordinates"])
    source_scores = read_indexed(paths["source_scores"]).loc[source.index]
    completed = read_indexed(paths["completed_predictors"]).loc[source.index]
    assessed, _ = load_evaluation(cohort, paths["target_scores"], paths["respondent"],
                                  paths["tracker"], paths.get("summary"))
    target_keys = assessed.key + "-" + assessed.landmark_wave.astype(int).astype(str)
    target = read_indexed(paths["target_coordinates"]).loc[target_keys]
    dimensions = list(source.filter(regex=r"^Dim[._]").columns)
    if len(dimensions) != 5 or any(name not in target for name in dimensions):
        raise ValueError("Five identically named source and target dimensions required")
    output = fresh_output(output)
    def progress(done, total):
        if done % 10 == 0 or done == total:
            print(f"{cohort}: {done}/{total} participant-level refits", flush=True)
    replicates = stability_refits(
        source[dimensions], source.index.str.rsplit("-", n=1).str[0],
        source_domain_burdens(completed), source_scores.hard_two_cluster,
        target[dimensions], assessed.margin_current, assessed.hard_current,
        consolidate=lambda x, labels: consolidate(x, labels, parent_dir=output, rscript=rscript),
        repeats=repeats, seed=seed, progress=progress,
    )
    replicates.to_csv(output / "stability_replicates.csv", index=False)
    report = {"cohort": cohort, "repeats": repeats, "seed": seed,
              "source_person_fraction": .8, "factor_coordinates_fixed": True,
              "target_population": "linked participants with an eligible reference diagnosis",
              "assessed_participants": len(assessed),
              "reference_diagnoses_used_in_clustering": False,
              "summary": summarize_stability(replicates),
              "input_sha256": {name: digest(path) for name, path in paths.items()}}
    (output / "stability_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("adams", "hcap"), required=True)
    for name in ("source-coordinates", "source-scores", "completed-predictors",
                 "target-coordinates", "target-scores", "respondent", "tracker", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    if args.repeats < 1 or (args.cohort == "hcap" and args.summary is None):
        parser.error("Positive repeats and all required reference files are needed")
    paths = {name: value for name, value in vars(args).items()
             if isinstance(value, Path) and name != "output"}
    run(args.cohort, paths, args.output, repeats=args.repeats, seed=args.seed, rscript=args.rscript)


if __name__ == "__main__":
    main()
