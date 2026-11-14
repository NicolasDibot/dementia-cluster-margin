"""Assemble manuscript tables and figure inputs from aggregate analysis outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .io import external, fresh_output
from .variables import variable_dictionary


CLINICAL_ROWS = (
    ("AUC, label", "auc::label"),
    ("AUC, margin", "auc::margin"),
    ("Difference", "contrast::margin_minus_label"),
    ("Contribution, outside the cluster", "contribution::two_cluster::H0"),
    ("Contribution, inside the cluster", "contribution::two_cluster::H1"),
    ("Within-cluster AUC of the margin, outside", "within::two_cluster::H0"),
    ("Within-cluster AUC of the margin, inside", "within::two_cluster::H1"),
)


def interval(row, decimals=3):
    return f"{row.estimate:.{decimals}f} ({row.ci95_low:.{decimals}f}–{row.ci95_high:.{decimals}f})"


def clinical_tables(directories):
    """Produce Tables 1 and 2 without altering estimates or intervals."""
    table1 = {"Cluster": ["Non-dementia-like", "Dementia-like"]}
    table2 = {"Measure": ["Participants (dementia)"] + [label for label, _ in CLINICAL_ROWS]
              + ["Concordance, three levels: label / margin"]}
    numeric, associations = [], []
    for cohort in ("adams", "hcap"):
        folder = Path(directories[cohort])
        distribution = pd.read_csv(folder / "diagnosis_distribution.csv").set_index("diagnostic_level")
        diagnoses = ("normal", "CIND" if cohort == "adams" else "MCI", "dementia")
        for level, diagnosis in enumerate(diagnoses):
            row = distribution.loc[level]
            table1[f"{cohort.upper()} {diagnosis}"] = [round(row.percentage_outside, 1), round(row.percentage_inside, 1)]
        estimates = pd.read_csv(folder / "clinical_estimands.csv").set_index("estimand")
        association = pd.read_csv(folder / "clinical_association.csv")
        metadata = json.loads((folder / "analysis_metadata.json").read_text())
        participants = (int(metadata["participants"]) // 10) * 10
        events = (int(metadata["dementia"]) // 10) * 10
        values = [f"≈{participants:,} (≈{events:,})"]
        for _, name in CLINICAL_ROWS:
            decimals = 4 if cohort == "hcap" and name == "contribution::two_cluster::H1" else 3
            values.append(interval(estimates.loc[name], decimals))
        hard = estimates.loc["ordinal_concordance::current_hard", "estimate"]
        margin = estimates.loc["ordinal_concordance::current_margin", "estimate"]
        difference = estimates.loc["ordinal_contrast::current_margin_minus_current_hard"]
        values.append(f"{hard:.3f} / {margin:.3f} (difference {interval(difference)})")
        table2["ADAMS consensus dementia" if cohort == "adams" else "HCAP algorithmic dementia"] = values
        numeric.append(estimates.reset_index())
        for name, term in (("label", "hard_current"), ("margin", "margin_within_state_z")):
            rows = association.loc[association.term.eq(term)]
            if len(rows) != 1:
                raise ValueError("Each clinical association term must be reported once")
            row = rows.iloc[0]
            numeric.append(pd.DataFrame([{"cohort": cohort, "estimand": f"or::{name}",
                                          "estimate": row.odds_ratio,
                                          "ci95_low": row.ci95_low, "ci95_high": row.ci95_high}]))
        associations.append(association)
    return pd.DataFrame(table1), pd.DataFrame(table2), pd.concat(numeric, ignore_index=True), pd.concat(associations, ignore_index=True)


def profile_table(geometry_directories):
    records = []
    for cohort, folder in geometry_directories.items():
        metadata = json.loads((Path(folder) / "geometry_metadata.json").read_text())
        severe = metadata["severe_cluster"] - 1
        for cluster, (cognition, function) in enumerate(metadata["source_profiles_cognition_function"]):
            label = int(cluster == severe)
            records.append({"cohort": cohort, "label": label,
                            "cognition_mean": cognition, "function_mean": function,
                            "share": metadata["severe_share"] if label else 1 - metadata["severe_share"],
                            "profile_definition": "mean of source-standardized full domain variables"})
    return pd.DataFrame(records)


def stability_table(stability_directories):
    rows = []
    for cohort, folder in stability_directories.items():
        summary = json.loads((Path(folder) / "stability_summary.json").read_text())["summary"]
        for metric, values in summary.items():
            rows.append({"cohort": cohort, "metric": metric,
                         "median": values["median"], "percentile_10": values["percentile_10"]})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = external(args.analysis_root), fresh_output(args.output)
    tables = clinical_tables({cohort: root / cohort / "clinical" for cohort in ("adams", "hcap")})
    names = ("table_1_diagnostic_distribution.csv", "table_2_clinical_validation.csv",
             "clinical_estimands.csv", "clinical_model_coefficients.csv")
    for name, table in zip(names, tables, strict=True):
        table.to_csv(output / name, index=False)
    profile_table({cohort: root / cohort / "geometry" for cohort in ("adams", "hcap", "share")}).to_csv(output / "cluster_profiles.csv", index=False)
    stability_table({cohort: root / cohort / "stability" for cohort in ("adams", "hcap")}).to_csv(output / "stability_summary.csv", index=False)
    prospective = pd.read_csv(root / "share" / "hcap" / "prospective_estimands.csv")
    prospective.to_csv(output / "prospective_summary.csv", index=False)
    pd.read_csv(root / "share" / "hcap" / "supplementary_table_1.csv").to_csv(output / "supplementary_table_1.csv", index=False)
    (output / "supplementary_table_2_variables.json").write_text(json.dumps(variable_dictionary(), indent=2) + "\n")


if __name__ == "__main__":
    main()
