"""Run the manuscript analyses from provider files using one path configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .io import external, fresh_output, require_data_access

# HRS waves after the primary Wave 10 landmark at which HCAP participants are
# also scored (2012, 2014 and 2016), for the analysis by interval.
HCAP_LATER_WAVES = (11, 12, 13)


def figure_script(name):
    source = Path(__file__).resolve().parents[1] / "scripts" / name
    installed = Path(sys.prefix) / "share" / "dementia-margin" / "scripts" / name
    return source if source.exists() else installed


def build_plan(config):
    """Describe the ordered steps; no cohort file is opened by this function."""
    root = external(config["output_dir"])
    rscript = config.get("rscript", "Rscript")
    seed, bootstrap = int(config.get("seed", 20261109)), int(config.get("bootstrap", 1000))
    repeats = int(config.get("stability_repeats", 100))
    if bootstrap < 20 or repeats < 1:
        raise ValueError("At least 20 bootstrap draws and one stability refit are required")
    package = Path(__file__).resolve().parent
    hrs, share = config["hrs"], config["share"]
    steps, geometry_configs = [], {}
    def add(name, *command):
        steps.append({"name": name, "command": list(map(str, command))})
    def module(name, *args):
        return (sys.executable, "-m", f"dementia_margin.{name}", *args)
    for cohort in ("adams", "hcap"):
        folder, clinical = root / cohort, hrs[cohort]
        add(f"{cohort}_selection", *module("select_hrs", "--cohort", cohort,
            "--tracker", clinical["tracker"], "--output", folder / "selection"))
        add(f"{cohort}_preparation", rscript, package / "prepare_hrs.R", cohort,
            hrs["rand"], hrs["harmonized"], folder / "selection" / "selection.csv", folder / "prepared")
        add(f"{cohort}_mfa", rscript, package / "fit_hrs_mfa.R",
            folder / "prepared" / "prepared_predictors.rds", folder / "mfa")
        geometry_configs[cohort] = {
            "cohort": cohort, "rscript": rscript, "output_dir": str(folder / "geometry"),
            "source_coordinates": str(folder / "mfa" / "source_coordinates.csv"),
            "completed_predictors": str(folder / "mfa" / "completed_predictors.csv"),
            "target_coordinates": str(folder / "mfa" / "target_coordinates.csv"),
            "target_geometry": str(folder / "mfa" / "target_geometry.csv"),
        }
        add(f"{cohort}_geometry", *module("fit", "--config", root / "configuration" / f"{cohort}.json"))
        references = ["--respondent", clinical["respondent"], "--tracker", clinical["tracker"]]
        if cohort == "hcap":
            references += ["--summary", clinical["summary"]]
        add(f"{cohort}_clinical", *module("clinical", "--cohort", cohort,
            "--scores", folder / "geometry" / "target_scores.csv", *references,
            "--bootstrap", bootstrap, "--seed", seed, "--output", folder / "clinical"))
        add(f"{cohort}_stability", *module("stability_analysis", "--cohort", cohort,
            "--source-coordinates", folder / "mfa" / "source_coordinates.csv",
            "--source-scores", folder / "geometry" / "source_scores.csv",
            "--completed-predictors", folder / "mfa" / "completed_predictors.csv",
            "--target-coordinates", folder / "mfa" / "target_coordinates.csv",
            "--target-scores", folder / "geometry" / "target_scores.csv", *references,
            "--repeats", repeats, "--seed", seed, "--rscript", rscript, "--output", folder / "stability"))
        if cohort == "hcap":
            for wave in HCAP_LATER_WAVES:
                later = folder / "interval" / f"wave{wave}"
                geometry_configs[f"hcap_wave{wave}"] = {
                    **geometry_configs["hcap"], "output_dir": str(later / "geometry"),
                    "target_geometry": str(folder / "mfa" / f"target_geometry_wave{wave}.csv"),
                }
                add(f"hcap_wave{wave}_geometry", *module("fit", "--config", root / "configuration" / f"hcap_wave{wave}.json"))
                add(f"hcap_wave{wave}_clinical", *module("clinical", "--cohort", "hcap",
                    "--scores", later / "geometry" / "target_scores.csv", *references,
                    "--bootstrap", bootstrap, "--seed", seed, "--output", later / "clinical"))
    folder = root / "share"
    add("share_preparation", rscript, package / "prepare_share.R", share["harmonized"], folder / "prepared")
    add("share_mfa", rscript, package / "fit_share_mfa.R", folder / "prepared" / "prepared_predictors.rds", folder / "mfa")
    geometry_configs["share"] = {
        "cohort": "share", "rscript": rscript, "output_dir": str(folder / "geometry"),
        "source_coordinates": str(folder / "mfa" / "source_coordinates.csv"),
        "completed_predictors": str(folder / "mfa" / "completed_predictors.csv"),
    }
    add("share_geometry", *module("fit", "--config", root / "configuration" / "share.json"))
    add("share_hcap", *module("share_hcap", "--scores", folder / "geometry" / "target_scores.csv",
        "--easyshare", share["easyshare"], "--harmonized-share", share["harmonized"],
        "--hcap-respondent", share["hcap_respondent"], "--hcap-informant", share["hcap_informant"],
        "--wave9-coverscreen", share["wave9_coverscreen"], "--wave9-weights", share["wave9_weights"],
        "--bootstrap", bootstrap, "--seed", seed, "--output", folder / "hcap"))
    add("risk_factors", *module("risk_analysis", "--hrs-scores", root / "hcap" / "geometry" / "source_scores.csv",
        "--share-scores", folder / "geometry" / "source_scores.csv", "--rand", hrs["rand"],
        "--harmonized-hrs", hrs["harmonized"], "--harmonized-share", share["harmonized"],
        "--share-raw", share["raw_root"], "--output", root / "risk_factors"))
    add("tables", *module("manuscript_tables", "--analysis-root", root, "--output", root / "tables"))
    add("figures_1_3", sys.executable, figure_script("plot_method_and_validation.py"),
        "--clinical-estimates", root / "tables" / "clinical_estimands.csv", "--output-dir", root / "figures")
    add("figures_2_4", sys.executable, figure_script("plot_profiles_and_prospective.py"),
        "--profiles", root / "tables" / "cluster_profiles.csv",
        "--stability", root / "tables" / "stability_summary.csv",
        "--prospective", root / "tables" / "prospective_summary.csv", "--output-dir", root / "figures")
    add("figure_5", sys.executable, figure_script("plot_risk_factors.py"),
        "--input", root / "risk_factors" / "common_population_paired_table.csv", "--output-dir", root / "figures")
    return root, steps, geometry_configs


def input_paths(config):
    hrs, share = config["hrs"], config["share"]
    values = [hrs["rand"], hrs["harmonized"], *hrs["adams"].values(), *hrs["hcap"].values(), *share.values()]
    return [external(value) for value in values]


def run(config):
    require_data_access()
    root, steps, geometry_configs = build_plan(config)
    for path in input_paths(config):
        if not path.exists():
            raise FileNotFoundError(f"Required provider file or directory not found: {path.name}")
    fresh_output(root)
    for name in ("adams", "hcap", "share", "configuration", "logs"):
        (root / name).mkdir(mode=0o700)
    for cohort, values in geometry_configs.items():
        (root / "configuration" / f"{cohort}.json").write_text(json.dumps(values, indent=2) + "\n")
    (root / "configuration" / "pipeline.json").write_text(json.dumps(config, indent=2) + "\n")
    completed = []
    for step in steps:
        print(f"Running {step['name']}", flush=True)
        with (root / "logs" / f"{step['name']}.log").open("w") as stream:
            result = subprocess.run(step["command"], stdout=stream, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"Step {step['name']} failed; inspect its controlled log before rerunning")
        completed.append(step["name"])
        (root / "completed_steps.json").write_text(json.dumps(completed, indent=2) + "\n")
    print("Completed analyses, tables and figures. Review aggregate disclosure before sharing.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", action="store_true", help="Show the execution plan without opening cohort data")
    args = parser.parse_args()
    config = json.loads(external(args.config).read_text())
    if args.plan:
        _, steps, _ = build_plan(config)
        print("\n".join(step["name"] for step in steps))
    else:
        run(config)


if __name__ == "__main__":
    main()
