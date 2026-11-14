import json
import shutil

import numpy as np
import pandas as pd
import pytest

from dementia_margin.fit import fit
from dementia_margin.geometry import FUNCTION_PROFILE_ITEMS


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is not installed")
def test_geometry_from_prepared_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("COHORT_DATA_ACCESS", "authorised")
    rng = np.random.default_rng(41)
    x = np.r_[rng.normal(-4, .2, (60, 5)), rng.normal(4, .2, (60, 5))]
    target = np.r_[rng.normal(-4, .2, (10, 5)), rng.normal(4, .2, (10, 5))]
    source_keys = [f"{i}-5" for i in range(120)]
    target_keys = [f"{i}-5" for i in range(1000, 1020)]
    columns = [f"Dim.{i}" for i in range(1, 6)]
    for name, values, keys in [("source", x, source_keys), ("target", target, target_keys)]:
        frame = pd.DataFrame(values, columns=columns)
        frame.insert(0, "person_wave_key", keys)
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    burden = np.r_[x[:, 0], target[:, 0]]
    complete = pd.DataFrame({name: (burden > 0).astype(float) for name in FUNCTION_PROFILE_ITEMS})
    complete["imrc"] = 6 - burden / 2
    complete["dlrc"] = 5 - burden / 2
    complete["tr20"] = complete.imrc + complete.dlrc
    complete.insert(0, "person_wave_key", source_keys + target_keys)
    complete.to_csv(tmp_path / "complete.csv", index=False)
    pd.DataFrame({"key": np.arange(1000, 1020).astype(str), "landmark_wave": 5,
                  "age_years": 75, "female": np.tile([0, 1], 10), "education_level": 2}).to_csv(tmp_path / "target_metadata.csv", index=False)
    fit({"cohort": "adams", "source_coordinates": str(tmp_path / "source.csv"),
         "target_coordinates": str(tmp_path / "target.csv"), "completed_predictors": str(tmp_path / "complete.csv"),
         "target_geometry": str(tmp_path / "target_metadata.csv"), "output_dir": str(tmp_path / "output")})
    scores = pd.read_csv(tmp_path / "output" / "target_scores.csv")
    np.testing.assert_array_equal(scores.hard_two_cluster, [0] * 10 + [1] * 10)
    assert (scores.margin_two_cluster > 0).eq(scores.hard_two_cluster == 1).all()
    metadata = json.loads((tmp_path / "output" / "geometry_metadata.json").read_text())
    assert metadata["selected_clusters"] == 2
