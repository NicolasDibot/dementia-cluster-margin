"""Build the paired risk-factor panel without modifying frozen geometries.

Readers use explicit variable whitelists. They return person-level data only
to the caller's controlled memory; this module never exports such records.
HRS cholesterol is deliberately unavailable: RAND's ``cholst`` is a
preventive cholesterol test, and medication use is not a diagnosis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

from .io import require_data_access


def _external(path: Path):
    repo = Path(__file__).resolve().parents[1]
    result = Path(path).expanduser().resolve()
    if result == repo or repo in result.parents:
        raise ValueError("Participant data and fitted objects must remain outside Git")
    return result


def clean(values: pd.Series, allowed=None, minimum=None, maximum=None):
    out = pd.to_numeric(values, errors="coerce")
    out = out.where(np.isfinite(out))
    if allowed is not None:
        out = out.where(out.isin(allowed))
    if minimum is not None:
        out = out.where(out >= minimum)
    if maximum is not None:
        out = out.where(out <= maximum)
    return out


def flag(values: pd.Series, threshold: float):
    return values.ge(threshold).astype(float).where(values.notna())


def _read(path: Path, columns):
    _, metadata = pyreadstat.read_dta(str(path), metadataonly=True)
    columns = list(dict.fromkeys(columns))
    present = [name for name in columns if name in metadata.column_names]
    frame, _ = pyreadstat.read_dta(str(path), usecols=present)
    for name in columns:
        if name not in frame:
            frame[name] = np.nan
    return frame


def _integer_key(values):
    value = pd.to_numeric(values, errors="raise")
    if value.isna().any() or not value.eq(np.floor(value)).all():
        raise ValueError("Invalid integer participant key")
    return value.astype("int64").astype(str)


def read_scores(path: Path):
    out = pd.read_csv(_external(path), dtype={"person_wave_key": str})
    required = {"person_wave_key", "wave", "margin_two_cluster", "hard_two_cluster"}
    if not required.issubset(out):
        raise ValueError("Two-cluster source score schema missing")
    parsed = out.person_wave_key.str.extract(r"^(?P<participant_id>.+)-(?P<parsed_wave>\d+)$")
    if parsed.isna().any().any():
        raise ValueError("Malformed participant-wave keys")
    if not pd.to_numeric(out.wave).eq(parsed.parsed_wave.astype(int)).all():
        raise ValueError("Wave mismatch")
    result = pd.DataFrame({
        "participant_id": parsed.participant_id, "wave": out.wave.astype(int),
        "margin": out.margin_two_cluster, "hard": out.hard_two_cluster,
    })
    if result.duplicated(["participant_id", "wave"]).any():
        raise ValueError("Duplicate frozen score")
    return result


def _join_pairs(scores, factors, pairs, cohort):
    output = []
    for first, second in pairs:
        start = scores.loc[scores.wave.eq(first)].drop(columns="wave").rename(
            columns={"margin": "margin_baseline", "hard": "hard_baseline"})
        end = scores.loc[scores.wave.eq(second)].drop(columns="wave").rename(
            columns={"margin": "margin_followup", "hard": "hard_followup"})
        paired = start.merge(end, on="participant_id", validate="one_to_one")
        paired = paired.merge(factors.loc[factors.wave.eq(first)], on="participant_id", validate="one_to_one")
        paired["interval"] = f"{first}-{second}"
        paired["cohort"] = cohort
        output.append(paired)
    return pd.concat(output, ignore_index=True)


def _sensory(near, distant, hearing):
    near, distant, hearing = [clean(x, allowed=[1, 2, 3, 4, 5]) for x in (near, distant, hearing)]
    bad = near.ge(4) | distant.ge(4)
    good = near.le(3) & distant.le(3)
    vision = pd.Series(np.select([bad, good], [1., 0.], np.nan), index=near.index)
    return vision, flag(hearing, 4)


def _inactive(vigorous, moderate, allowed):
    vigorous, moderate = clean(vigorous, allowed=allowed), clean(moderate, allowed=allowed)
    # Both components must be observed; missing is never inactivity.
    return (vigorous.eq(5) & moderate.eq(5)).astype(float).where(vigorous.notna() & moderate.notna())


def build_hrs_panel(scores_path: Path, rand_path: Path, harmonized_path: Path):
    """Build Waves 8->9 and 9->10; requires caller-authorised HRS execution."""
    require_data_access()
    scores_path, rand_path, harmonized_path = map(_external, (scores_path, rand_path, harmonized_path))
    static = ["hhidpn", "ragender", "raeduc"]
    stems = ["agey_m", "wtresp", "cesd", "vgactx", "mdactx", "diabe", "hibpe", "smoken", "bmi", "drink", "drinkd", "drinkn"]
    rand = _read(rand_path, static + [f"r{w}{s}" for w in (8, 9) for s in stems])
    rand["participant_id"] = _integer_key(rand.hhidpn)
    hcols = ["hhidpn", "hhid", "r9socwk", "r9scwtresp"] + [f"r{w}{s}" for w in (8, 9) for s in ("nsight", "dsight", "hearing")]
    harmonized = _read(harmonized_path, hcols)
    harmonized["participant_id"] = _integer_key(harmonized.hhidpn)
    data = rand.merge(harmonized.drop(columns="hhidpn"), on="participant_id", validate="one_to_one")
    edu = clean(data.raeduc, allowed=[1, 2, 3, 4, 5])
    rows = []
    for w in (8, 9):
        def col(stem):
            return data[f"r{w}{stem}"]
        frame = pd.DataFrame({
            "participant_id": data.participant_id, "household_id": data.hhid,
            "country": "USA", "wave": w,
            "age": clean(col("agey_m"), minimum=40, maximum=115),
            "sex": clean(data.ragender, allowed=[1, 2]),
            "education": np.select([edu.eq(1), edu.isin([2, 3]), edu.isin([4, 5])], [1., 2., 3.], np.nan),
            "weight": clean(col("wtresp"), minimum=0),
            "less_education": edu.eq(1).astype(float).where(edu.notna()),
            "depression": flag(clean(col("cesd"), minimum=0, maximum=8), 4),
            "physical_inactivity": _inactive(col("vgactx"), col("mdactx"), [1, 2, 3, 4, 5]),
            "diabetes": clean(col("diabe"), allowed=[0, 1]),
            "hypertension": clean(col("hibpe"), allowed=[0, 1]),
            "smoking": clean(col("smoken"), allowed=[0, 1]),
            "obesity": flag(clean(col("bmi"), minimum=10, maximum=100), 30),
        })
        frame["poor_vision"], frame["hearing_loss"] = _sensory(col("nsight"), col("dsight"), col("hearing"))
        days = clean(col("drinkd"), minimum=0, maximum=7)
        drinks = clean(col("drinkn"), minimum=0, maximum=98)
        weekly = days * drinks
        nondrinker = clean(col("drink"), allowed=[0, 1]).eq(0) | days.eq(0)
        weekly = weekly.mask(nondrinker, 0.)
        frame["excessive_alcohol"] = weekly.gt(14).astype(float).where(weekly.notna())
        # No self-reported high-cholesterol diagnosis is present in these two
        # harmonized products; do not substitute preventive screening or drugs.
        frame["high_cholesterol"] = np.nan
        frame["no_social_activity"] = np.nan
        frame["weight_no_social_activity"] = np.nan
        if w == 9:
            social = clean(data.r9socwk, allowed=[0, 1])
            frame["no_social_activity"] = 1 - social
            frame["weight_no_social_activity"] = clean(data.r9scwtresp, minimum=0)
        rows.append(frame)
    return _join_pairs(read_scores(scores_path), pd.concat(rows, ignore_index=True), [(8, 9), (9, 10)], "HRS")


def share_alcohol(raw: pd.DataFrame, wave: int):
    """Quantity threshold with explicit midpoint mapping for categorical days.

    Wave 4/5 frequency categories use weekly midpoints
    7, 5.5, 3.5, 1.5, 0.375, 0.125, 0.
    """
    if wave == 6:
        yesno = clean(raw.br039_, allowed=[1, 5])
        weekly = clean(raw.br040_, minimum=0)
        weekly = weekly.mask(yesno.eq(5), 0.)
    else:
        frequency = clean(raw.br010_, allowed=range(1, 8))
        days = frequency.map({1: 7., 2: 5.5, 3: 3.5, 4: 1.5, 5: .375, 6: .125, 7: 0.})
        weekly = days * clean(raw.br019_, minimum=0)
        never = clean(raw.br021_, allowed=[1, 5]).eq(5) | frequency.eq(7)
        weekly = weekly.mask(never, 0.)
    return weekly.gt(14).astype(float).where(weekly.notna())


def build_share_panel(scores_path: Path, harmonized_path: Path, raw_root: Path):
    scores_path, harmonized_path, raw_root = map(_external, (scores_path, harmonized_path, raw_root))
    pairs = [(1, 2), (4, 5), (5, 6), (6, 7)]
    firsts = [a for a, _ in pairs]
    static = ["mergeid", "hhid", "country", "ragender", "raedyrs", "raeducl"]
    stems = ["agey", "nsight", "dsight", "hearing", "eurod", "vgactx", "mdactx", "diabe", "hibpe", "hchole", "smoken", "bmi", "socyr"]
    weights = [f"ral{a}{b}wtrsp" for a, b in pairs]
    data = _read(harmonized_path, static + weights + [f"r{w}{s}" for w in firsts for s in stems])
    rows = []
    for w, followup in pairs:
        def col(stem):
            return data[f"r{w}{stem}"]
        education_level = clean(data.raeducl, allowed=[1, 2, 3])
        frame = pd.DataFrame({
            "participant_id": data.mergeid.astype(str), "household_id": data.hhid,
            "country": data.country, "wave": w,
            "age": clean(col("agey"), minimum=40, maximum=115),
            "sex": clean(data.ragender, allowed=[1, 2]),
            "education": clean(data.raedyrs, minimum=0, maximum=40),
            "weight": clean(data[f"ral{w}{followup}wtrsp"], minimum=0),
            "less_education": education_level.eq(1).astype(float).where(education_level.notna()),
            "depression": flag(clean(col("eurod"), minimum=0, maximum=12), 4),
            "physical_inactivity": _inactive(col("vgactx"), col("mdactx"), [2, 3, 4, 5]),
            "diabetes": clean(col("diabe"), allowed=[0, 1]),
            "hypertension": clean(col("hibpe"), allowed=[0, 1]),
            "high_cholesterol": clean(col("hchole"), allowed=[0, 1]),
            "smoking": clean(col("smoken"), allowed=[0, 1]),
            "obesity": flag(clean(col("bmi"), minimum=10, maximum=100), 30),
            "no_social_activity": 1 - clean(col("socyr"), allowed=[0, 1]) if w >= 4 else np.nan,
        })
        frame["poor_vision"], frame["hearing_loss"] = _sensory(col("nsight"), col("dsight"), col("hearing"))
        frame["excessive_alcohol"] = np.nan
        if w >= 4:
            path = raw_root / f"sharew{w}_rel9-0-0_ALL_datasets_stata" / f"sharew{w}_rel9-0-0_br.dta"
            fields = ["br039_", "br040_"] if w == 6 else ["br010_", "br019_", "br021_"]
            raw = _read(path, ["mergeid", *fields])
            alcohol = pd.DataFrame({"participant_id": raw.mergeid.astype(str), "alcohol": share_alcohol(raw, w)})
            frame = frame.merge(alcohol, on="participant_id", how="left", validate="one_to_one")
            frame["excessive_alcohol"] = frame.pop("alcohol")
        rows.append(frame)
    return _join_pairs(read_scores(scores_path), pd.concat(rows, ignore_index=True), pairs, "SHARE")
