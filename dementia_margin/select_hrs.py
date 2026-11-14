"""Select HRS clinical target participants without reading outcome fields."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyreadstat

from .io import external, fresh_output, require_data_access

ADAMS_TRACKER_LAYOUT = {"HHID": (1, 6), "PN": (7, 9), "WAVESEL": (261, 261)}


def _key_from_hhid_pn(hhid: object, pn: object) -> str:
    left = str(hhid).strip().removesuffix(".0")
    right = str(pn).strip().removesuffix(".0")
    if not left.isdigit() or len(left) > 6 or not right.isdigit() or len(right) > 3:
        raise ValueError("Invalid fixed-width HHID or PN")
    return str(int(left.zfill(6) + right.zfill(3)))


def _read_adams_tracker_selection(path: Path) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    maximum = max(end for _, end in ADAMS_TRACKER_LAYOUT.values())
    with Path(path).open("r", encoding="cp1252", errors="strict", newline="") as stream:
        for line_number, raw in enumerate(stream, start=1):
            line = raw.rstrip("\r\n")
            if len(line) < maximum:
                raise ValueError(
                    f"ADAMS Tracker record {line_number} is shorter than the release layout"
                )
            record = {
                name: line[start - 1 : end].strip()
                for name, (start, end) in ADAMS_TRACKER_LAYOUT.items()
            }
            records.append(record)
    tracker = pd.DataFrame.from_records(records)
    if len(tracker) != 1770:
        raise ValueError("ADAMS Tracker must contain 1,770 records")
    wave = pd.to_numeric(tracker["WAVESEL"], errors="coerce")
    if wave.isna().any() or wave.value_counts().to_dict() != {2: 1105, 1: 665}:
        raise ValueError("ADAMS WAVESEL differs from the official counts")
    keys = [
        _key_from_hhid_pn(hhid, pn)
        for hhid, pn in zip(tracker["HHID"], tracker["PN"], strict=True)
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("ADAMS Tracker contains duplicate keys")
    return pd.DataFrame(
        {
            "key": keys,
            "landmark_wave": wave.astype(int).map({1: 5, 2: 6}),
            "target_source": "adams",
        }
    )


def _read_hcap_tracker_selection(path: Path) -> pd.DataFrame:
    try:
        tracker, _ = pyreadstat.read_dta(
            str(path), usecols=["HHID", "PN", "HCAP16RESP"]
        )
    except (ValueError, pyreadstat.ReadstatError) as error:
        raise ValueError("HCAP Tracker whitelist is unavailable") from error
    response = pd.to_numeric(tracker["HCAP16RESP"], errors="coerce")
    selected = tracker.loc[response.isin([1, 2, 3]), ["HHID", "PN"]]
    if len(tracker) != 46766 or len(selected) != 3496:
        raise ValueError("HCAP Tracker totals differ from the official release")
    keys = [
        _key_from_hhid_pn(hhid, pn)
        for hhid, pn in zip(selected["HHID"], selected["PN"], strict=True)
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("HCAP selection contains duplicate keys")
    return pd.DataFrame(
        {
            "key": keys,
            "landmark_wave": 10,
            "target_source": "hcap",
        }
    )



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("adams", "hcap"), required=True)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_data_access()
    tracker = external(args.tracker)
    reader = _read_adams_tracker_selection if args.cohort == "adams" else _read_hcap_tracker_selection
    selection = reader(tracker)
    output = fresh_output(args.output)
    selection.to_csv(output / "selection.csv", index=False)


if __name__ == "__main__":
    main()
