"""Fit two clusters and score participants from prepared factor coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd

from .geometry import signed_margin, source_benchmarks, source_impairment_profiles, ward_initial_partition
from .io import digest, external, fresh_output, require_data_access


def consolidate(coordinates, initial, *, parent_dir, rscript="Rscript"):
    """Consolidate the Ward partition with R's Hartigan-Wong algorithm."""
    helper = Path(__file__).with_name("consolidate_clusters.R")
    with tempfile.TemporaryDirectory(prefix="consolidation_", dir=parent_dir) as tmp:
        folder = Path(tmp)
        frame = pd.DataFrame(coordinates, columns=[f"Dim.{i}" for i in range(1, 6)])
        frame.insert(0, "person_wave_key", np.arange(len(frame)).astype(str))
        frame["initial_cluster"] = initial
        frame.to_csv(folder / "input.csv", index=False)
        subprocess.run([rscript, str(helper), str(folder / "input.csv"),
                        str(folder / "labels.csv"), str(folder / "fit.json")],
                       check=True, capture_output=True, text=True)
        labels = pd.read_csv(folder / "labels.csv", dtype={"person_wave_key": str})
        if not np.array_equal(labels.person_wave_key, frame.person_wave_key):
            raise RuntimeError("Consolidation changed row order")
        return labels.cluster.to_numpy(int)


def read_indexed(path):
    frame = pd.read_csv(path, dtype={"person_wave_key": str}).set_index("person_wave_key")
    if not frame.index.is_unique:
        raise ValueError("Person-wave keys must be unique")
    return frame


def fit(config):
    require_data_access()
    cohort = config["cohort"]
    if cohort not in ("adams", "hcap", "share"):
        raise ValueError("Unknown cohort")
    names = ("source_coordinates", "completed_predictors")
    if cohort != "share":
        names += ("target_coordinates", "target_geometry")
    paths = {name: external(config[name]) for name in names}
    output = fresh_output(config["output_dir"])
    source = read_indexed(paths["source_coordinates"])
    completed = read_indexed(paths["completed_predictors"])
    dimensions = list(source.filter(regex=r"^Dim[._]").columns)
    if len(dimensions) != 5:
        raise ValueError("Exactly five factor dimensions are required")
    x = source[dimensions].to_numpy(float)
    initial, metadata = ward_initial_partition(x)
    if metadata["selected_clusters"] != 2:
        raise RuntimeError("The inertia criterion did not select two clusters")
    labels = consolidate(x, initial, parent_dir=output, rscript=config.get("rscript", "Rscript"))
    centres = np.vstack([x[labels == k].mean(axis=0) for k in (1, 2)])
    source_completed = completed.loc[source.index]
    burden = source_impairment_profiles(source_completed, cohort)
    profiles = np.vstack([burden.loc[labels == k].mean().to_numpy() for k in (1, 2)])
    if np.argmax(profiles[:, 0]) != np.argmax(profiles[:, 1]):
        raise RuntimeError("No cluster has both the highest cognitive and functional burden")
    severe = int(np.argmax(profiles[:, 0]))
    margin, hard = signed_margin(x, centres, severe)
    if not np.array_equal(hard, (labels == severe + 1).astype(int)):
        raise RuntimeError("Consolidated labels differ from nearest-centroid labels")
    scores = source_benchmarks(source_completed, source_completed, cohort)
    scores["margin_two_cluster"], scores["hard_two_cluster"] = margin, hard
    scores["wave"] = source.index.str.rsplit("-", n=1).str[-1].astype(int)
    scores.reset_index().to_csv(output / "source_scores.csv", index=False)
    source.reset_index().to_csv(output / "source_coordinates.csv", index=False)
    if cohort == "share":
        target = scores.loc[scores.wave.eq(7)].copy()
        target["mergeid"] = target.index.str.rsplit("-", n=1).str[0]
    else:
        metadata_frame = pd.read_csv(paths["target_geometry"], dtype={"key": str})
        if metadata_frame.key.duplicated().any():
            raise ValueError("Target geometry must contain one landmark per participant")
        target_keys = pd.Index(metadata_frame.key + "-" + metadata_frame.landmark_wave.astype(int).astype(str), name="person_wave_key")
        source_people = set(source.index.str.rsplit("-", n=1).str[0])
        if not source_people.isdisjoint(metadata_frame.key):
            raise RuntimeError("Clinical participants must not enter the source fit")
        coordinates = read_indexed(paths["target_coordinates"]).loc[target_keys, dimensions]
        tm, th = signed_margin(coordinates.to_numpy(float), centres, severe)
        target = source_benchmarks(source_completed, completed.loc[target_keys], cohort)
        target["margin_two_cluster"], target["hard_two_cluster"] = tm, th
        for column in ("key", "landmark_wave", "age_years", "female", "education_level"):
            target[column] = metadata_frame[column].to_numpy()
    target.reset_index().to_csv(output / "target_scores.csv", index=False)
    pd.DataFrame(centres, columns=dimensions).assign(cluster=[1, 2], severe=[severe == 0, severe == 1]).to_csv(output / "centroids.csv", index=False)
    metadata.update({"cohort": cohort, "source_person_waves": len(source),
                     "severe_share": float(hard.mean()),
                     "source_profiles_cognition_function": profiles.tolist(),
                     "severe_cluster": severe + 1,
                     "input_sha256": {name: digest(path) for name, path in paths.items()}})
    (output / "geometry_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    fit(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
