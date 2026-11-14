"""Figures preserve supplied aggregate estimates and remain editable vectors."""

import importlib.util
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree as ET

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_profiles_and_prospective.py"
SPEC = importlib.util.spec_from_file_location("figure_inputs", SCRIPT)
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


def examples():
    profiles = [
        {"cohort": cohort, "label": label, "function_mean": 1.1 if label else -.2,
         "cognition_mean": .9 if label else -.1, "share": .1 if label else .9,
         "profile_definition": "Synthetic standardized functional and cognitive domains"}
        for cohort in PLOT.COHORTS for label in (0, 1)
    ]
    stability = [{"cohort": cohort, "metric": metric, "median": .998, "percentile_10": .992}
                 for cohort in PLOT.COHORTS[:2] for metric in PLOT.METRICS]
    prospective = [{"outcome": outcome, "estimand": estimand,
                    "estimate": .6 if estimand == "label" else .7,
                    "ci95_low": .55 if estimand == "label" else .63,
                    "ci95_high": .65 if estimand == "label" else .77}
                   for outcome in PLOT.OUTCOMES for estimand in ("label", "margin")]
    return profiles, stability, prospective


def test_figures_preserve_all_aggregate_values(tmp_path):
    datasets = dict(zip(("profiles", "stability", "prospective"), examples()))
    command = [sys.executable, str(SCRIPT), "--output-dir", str(tmp_path / "figures")]
    for name, rows in datasets.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"rows": rows}))
        command.extend(["--"+name, str(path)])
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    ns = {"s": "http://www.w3.org/2000/svg"}
    for figure, groups in [(2, 12), (4, 6)]:
        root = ET.parse(tmp_path / "figures" / f"Figure_{figure}.svg").getroot()
        assert len(root.findall(".//s:g", ns)) == groups
        assert not root.findall(".//s:image", ns)
        assert len(root.findall(".//s:text", ns)) > 15
        assert (tmp_path / "figures" / f"Figure_{figure}.png").stat().st_size > 1000
    actual = json.loads((tmp_path / "figures" / "Source_Data_Figure_4.json").read_text())["rows"]
    assert actual == datasets["prospective"]
    profiles_out = json.loads((tmp_path / "figures" / "Source_Data_Figure_2.json").read_text())
    assert profiles_out["profiles"] == datasets["profiles"]
    assert profiles_out["stability"] == datasets["stability"]
    tree = ET.parse(tmp_path / "figures" / "Figure_4.svg")
    for group, row in zip(tree.findall(".//s:g", ns), actual):
        for name in ("estimate", "ci95_low", "ci95_high"):
            assert float(group.attrib["data-"+name]) == row[name]


def test_incomplete_or_invalid_inputs_fail():
    profiles, stability, prospective = examples()
    with pytest.raises(ValueError, match="two clusters"):
        PLOT.profiles(profiles[:-1])
    profiles[0]["share"] = 1.0
    with pytest.raises(ValueError, match="fraction"):
        PLOT.profiles(profiles)
    stability[0]["median"] = .5
    with pytest.raises(ValueError, match="ordered"):
        PLOT.stability(stability)
    prospective[0]["ci95_low"] = .8
    with pytest.raises(ValueError, match="ordered"):
        PLOT.prospective(prospective)


def test_estimates_are_not_replaced_by_manuscript_values():
    _, _, rows = examples()
    rows[0]["estimate"] = .6123456789
    assert PLOT.prospective(rows)[0]["estimate"] == .6123456789


def test_clinical_figure_uses_optional_pipeline_values(tmp_path):
    script = SCRIPT.with_name("plot_method_and_validation.py")
    source = tmp_path / "clinical.csv"
    values = {
        "auc::label": (.7,.65,.75), "auc::margin": (.8,.75,.85),
        "contrast::margin_minus_label": (.1,.06,.14),
        "contribution::two_cluster::H0": (.08,.04,.12),
        "contribution::two_cluster::H1": (.02,.01,.03),
        "or::label": (4.,2.,8.), "or::margin": (1.8,1.3,2.5),
    }
    with source.open("w", newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["cohort","estimand","estimate","ci95_low","ci95_high"])
        for cohort in ("adams","hcap"):
            for estimand,triplet in values.items():
                writer.writerow([cohort,estimand,*triplet])
    output=tmp_path/"recomputed"
    result=subprocess.run([sys.executable,str(script),"--clinical-estimates",str(source),
                           "--output-dir",str(output)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    data=json.loads((output/"Source_Data_Figure_3.json").read_text())
    for cohort in ("ADAMS","HCAP"):
        assert data[cohort]["label"]==list(values["auc::label"])
        assert data[cohort]["margin_or"]==list(values["or::margin"])
        assert data[cohort]["share"]=="80.0%"
    default=tmp_path/"reported"
    result=subprocess.run([sys.executable,str(script),"--output-dir",str(default)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    expected={
        "Figure_1.svg":"43affa91e99804abb06ddb2dff8852a042d7c63fe6a1839c62905d821c51401f",
        "Figure_3.svg":"5f15bd7fded6ef2a0723533f0d07642969ede531f702e95734fc86a6f5894406",
    }
    for filename,digest in expected.items():
        assert hashlib.sha256((default/filename).read_bytes()).hexdigest()==digest
