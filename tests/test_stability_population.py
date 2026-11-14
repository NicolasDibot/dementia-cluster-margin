import json

import numpy as np
import pandas as pd

from dementia_margin.stability_analysis import run


def test_stability_uses_only_eligible_clinically_assessed_targets(tmp_path, monkeypatch):
    monkeypatch.setenv("COHORT_DATA_ACCESS", "authorised")
    source_keys = [f"{key}-5" for key in range(10)]
    target_keys = [f"{key}-5" for key in range(100, 104)]
    dimensions = [f"Dim.{i}" for i in range(1, 6)]
    source = pd.DataFrame(np.arange(50).reshape(10, 5), columns=dimensions)
    source.insert(0, "person_wave_key", source_keys)
    target = pd.DataFrame(np.arange(20).reshape(4, 5), columns=dimensions)
    target.insert(0, "person_wave_key", target_keys)
    scores = pd.DataFrame({"person_wave_key": source_keys, "hard_two_cluster": np.tile([0, 1], 5)})
    completed = pd.DataFrame({"person_wave_key": source_keys})
    paths = {}
    for name, frame in (("source_coordinates", source), ("target_coordinates", target),
                        ("source_scores", scores), ("completed_predictors", completed)):
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    for name in ("target_scores", "respondent", "tracker"):
        path = tmp_path / name
        path.touch()
        paths[name] = path
    assessed = pd.DataFrame({"key": ["102", "100"], "landmark_wave": 5,
                             "margin_current": [1., -1.], "hard_current": [1, 0]})
    monkeypatch.setattr("dementia_margin.stability_analysis.load_evaluation", lambda *a: (assessed, None))
    monkeypatch.setattr("dementia_margin.stability_analysis.source_domain_burdens", lambda x: np.ones((10, 2)))
    def refits(source_x, people, burdens, source_labels, target_x, margin, label, **kwargs):
        np.testing.assert_array_equal(target_x, target.loc[[2, 0], dimensions])
        np.testing.assert_array_equal(margin, [1., -1.])
        np.testing.assert_array_equal(label, [1, 0])
        return pd.DataFrame({"replicate": [1, 2], "heldout_adjusted_rand": [.99, 1.],
                             "target_label_agreement": [1., 1.], "target_margin_spearman": [.999, 1.]})
    monkeypatch.setattr("dementia_margin.stability_analysis.stability_refits", refits)
    output = tmp_path / "results"
    result = run("adams", paths, output, repeats=2)
    assert result["assessed_participants"] == 2
    assert not result["reference_diagnoses_used_in_clustering"]
    saved = json.loads((output / "stability_summary.json").read_text())
    assert saved["summary"]["target_label_agreement"]["median"] == 1.
