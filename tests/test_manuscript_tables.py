import json

import pandas as pd

from dementia_margin.manuscript_tables import CLINICAL_ROWS, clinical_tables, profile_table, stability_table
from dementia_margin.variables import variable_dictionary


def test_tables_preserve_estimates_counts_and_diagnostic_denominators(tmp_path):
    directories = {}
    for cohort in ("adams", "hcap"):
        folder = tmp_path / cohort
        folder.mkdir()
        directories[cohort] = folder
        pd.DataFrame({"diagnostic_level": [0, 1, 2], "percentage_outside": [90., 70., 40.],
                      "percentage_inside": [10., 30., 60.]}).to_csv(folder / "diagnosis_distribution.csv", index=False)
        names = [name for _, name in CLINICAL_ROWS] + ["ordinal_concordance::current_hard",
                 "ordinal_concordance::current_margin", "ordinal_contrast::current_margin_minus_current_hard"]
        pd.DataFrame([{"cohort": cohort, "estimand": name, "estimate": .612345,
                       "ci95_low": .51, "ci95_high": .72} for name in names]).to_csv(folder / "clinical_estimands.csv", index=False)
        pd.DataFrame({"cohort": cohort, "term": ["hard_current", "margin_within_state_z"],
                      "odds_ratio": [2., 1.5], "ci95_low": [1.5, 1.1], "ci95_high": [3., 2.]}).to_csv(folder / "clinical_association.csv", index=False)
        (folder / "analysis_metadata.json").write_text(json.dumps({"participants": 109, "dementia": 27}))
    table1, table2, numeric, models = clinical_tables(directories)
    assert table1["ADAMS dementia"].tolist() == [40., 60.]
    assert table1["HCAP MCI"].sum() == 100
    assert table2["ADAMS consensus dementia"].iloc[0] == "≈100 (≈20)"
    assert table2["ADAMS consensus dementia"].iloc[1] == "0.612 (0.510–0.720)"
    assert numeric.loc[numeric.estimand.eq("auc::label"), "estimate"].eq(.612345).all()
    assert numeric.loc[numeric.estimand.eq("or::margin"), "estimate"].eq(1.5).all()
    assert len(models) == 4


def test_figure_tables_preserve_orientation_and_stability(tmp_path):
    folder = tmp_path / "example"
    folder.mkdir()
    (folder / "geometry_metadata.json").write_text(json.dumps({
        "severe_cluster": 1, "severe_share": .2,
        "source_profiles_cognition_function": [[1., 2.], [-.1, -.2]],
    }))
    profiles = profile_table({"share": folder})
    assert profiles.label.tolist() == [1, 0]
    assert profiles.function_mean.tolist() == [2., -.2]
    (folder / "stability_summary.json").write_text(json.dumps({"summary": {
        "target_label_agreement": {"median": .998, "percentile_10": .992}}}))
    stability = stability_table({"adams": folder})
    assert stability.iloc[0]["median"] == .998
    assert stability.iloc[0].percentile_10 == .992


def test_variable_dictionary_preserves_cohort_specific_blocks():
    dictionary = variable_dictionary()
    assert len(dictionary["HRS"]) == 34
    assert len(dictionary["SHARE"]) == 38
    for cohort, active_count in (("HRS", 29), ("SHARE", 32)):
        rows = dictionary[cohort]
        assert sum(row["role"] == "active" for row in rows) == active_count
        assert len({row["variable"] for row in rows}) == len(rows)
    hrs = {row["variable"]: row for row in dictionary["HRS"]}
    share = {row["variable"]: row for row in dictionary["SHARE"]}
    assert hrs["cogothp"]["role"] == "external_descriptive"
    assert share["cogothp"]["role"] == "active"
    assert share["cogimp"]["role"] == "active"
    assert "verbf" not in hrs and share["verbf"]["role"] == "active"
