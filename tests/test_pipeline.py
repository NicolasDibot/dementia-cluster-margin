import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from dementia_margin.pipeline import build_plan, run


def configuration(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    names = ("rand", "harmonized_hrs", "adams_tracker", "adams_respondent", "hcap_tracker",
             "hcap_respondent", "hcap_summary", "harmonized_share", "easyshare",
             "share_hcap_respondent", "share_hcap_informant", "wave9_coverscreen", "wave9_weights")
    paths = {}
    for name in names:
        path = inputs / name
        path.touch()
        paths[name] = str(path)
    (inputs / "raw").mkdir()
    return {"output_dir": str(tmp_path / "output"), "bootstrap": 20, "stability_repeats": 2,
            "hrs": {"rand": paths["rand"], "harmonized": paths["harmonized_hrs"],
                    "adams": {"tracker": paths["adams_tracker"], "respondent": paths["adams_respondent"]},
                    "hcap": {"tracker": paths["hcap_tracker"], "respondent": paths["hcap_respondent"], "summary": paths["hcap_summary"]}},
            "share": {"harmonized": paths["harmonized_share"], "easyshare": paths["easyshare"],
                      "hcap_respondent": paths["share_hcap_respondent"], "hcap_informant": paths["share_hcap_informant"],
                      "wave9_coverscreen": paths["wave9_coverscreen"], "wave9_weights": paths["wave9_weights"],
                      "raw_root": str(inputs / "raw")}}


def test_plan_has_all_manuscript_steps_and_matching_paths(tmp_path):
    config = configuration(tmp_path)
    root, steps, geometries = build_plan(config)
    names = [step["name"] for step in steps]
    assert len(names) == len(set(names))
    for cohort in ("adams", "hcap"):
        assert names.index(cohort + "_selection") < names.index(cohort + "_mfa") < names.index(cohort + "_clinical")
        assert geometries[cohort]["target_geometry"] == str(root / cohort / "mfa" / "target_geometry.csv")
        stability = next(s["command"] for s in steps if s["name"] == cohort + "_stability")
        assert "--respondent" in stability and "--tracker" in stability
        assert ("--summary" in stability) == (cohort == "hcap")
    assert names.index("share_hcap") < names.index("tables") < names.index("figures_2_4")
    assert names[-1] == "figure_5"
    assert not root.exists()


def test_driver_records_only_successfully_completed_steps(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    monkeypatch.setenv("COHORT_DATA_ACCESS", "authorised")
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr("dementia_margin.pipeline.subprocess.run", execute)
    run(config)
    root, steps, _ = build_plan(config)
    assert calls == [step["command"] for step in steps]
    assert json.loads((root / "completed_steps.json").read_text()) == [step["name"] for step in steps]
    for cohort in ("adams", "hcap", "share"):
        content = json.loads((root / "configuration" / f"{cohort}.json").read_text())
        assert content["cohort"] == cohort
        assert Path(content["output_dir"]).parent == root / cohort


def test_pipeline_python_cli_options_are_available(tmp_path):
    _, steps, _ = build_plan(configuration(tmp_path))
    checked = set()
    for step in steps:
        command = step["command"]
        if "-m" in command:
            entry = tuple(command[:3])
            flags = {value for value in command[3:] if value.startswith("--")}
        elif len(command) > 1 and command[1].endswith(".py"):
            entry = tuple(command[:2])
            flags = {value for value in command[2:] if value.startswith("--")}
        else:
            continue
        key = (entry, tuple(sorted(flags)))
        if key in checked:
            continue
        result = subprocess.run([*entry, "--help"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert all(flag in result.stdout for flag in flags)
        checked.add(key)
