"""Survey-weighted discrimination, decomposition and adjusted associations."""
from __future__ import annotations
import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping
import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import t as student_t
from sklearn.metrics import roc_auc_score

class InputError(ValueError):
    """Input data do not match the analysis specification."""


MINIMUM_CELL = 10


BOOTSTRAP_REPLICATES = 1000


BOOTSTRAP_MINIMUM_SUCCESS = 950


BOOTSTRAP_MAXIMUM_ATTEMPTS = 1200


ADAMS_WAVE_A_LAYOUT = {"HHID": (1, 6), "PN": (7, 9), "ADFDX1": (27, 28)}


ADAMS_TRACKER_LAYOUT = {
    "HHID": (1, 6),
    "PN": (7, 9),
    "AASSESS": (18, 18),
    "WAVESEL": (261, 261),
    "SECLUST": (284, 284),
    "SESTRAT": (285, 286),
    "AASAMPWT_F": (287, 292),
}


ADAMS_NORMAL_CODES = {31}


ADAMS_CIND_CODES = {*range(20, 31), 33}


ADAMS_DEMENTIA_CODES = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18, 19, 32}


ASSOCIATION_COMMON_TERMS = (
    "hard_current",
    "margin_within_state_z",
    "age_per_10_years",
    "female",
    "education_middle_vs_low",
    "education_high_vs_low",
)


class _AssociationReliabilityError(ValueError):
    """A model-estimation failure that suppresses only the secondary model."""


def _key_from_hhid_pn(hhid: object, pn: object) -> str:
    left = str(hhid).strip().removesuffix(".0")
    right = str(pn).strip().removesuffix(".0")
    if not left.isdigit() or len(left) > 6 or not right.isdigit() or len(right) > 3:
        raise InputError("Invalid HHID/PN")
    return str(int(left.zfill(6) + right.zfill(3)))


def _read_fixed_width(path: Path, layout: Mapping[str, tuple[int, int]]) -> pd.DataFrame:
    maximum = max(end for _, end in layout.values())
    records: list[dict[str, object]] = []
    with path.open("r", encoding="cp1252", errors="strict", newline="") as stream:
        for line_number, raw in enumerate(stream, start=1):
            line = raw.rstrip("\r\n")
            if len(line) < maximum:
                raise InputError(
                    f"Fixed-width record {line_number} is shorter than the specified layout"
                )
            records.append(
                {
                    name: (line[start - 1 : end].strip() or pd.NA)
                    for name, (start, end) in layout.items()
                }
            )
    return pd.DataFrame.from_records(records, columns=list(layout))


def _read_hcap_dta_whitelist(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read only the exact HCAP whitelist and restore canonical column names."""
    try:
        _, metadata = pyreadstat.read_dta(str(path), metadataonly=True)
        available_by_casefold: dict[str, str] = {}
        for available in metadata.column_names:
            folded = str(available).casefold()
            if folded in available_by_casefold:
                raise InputError(
                    "HCAP Stata schema contains case-ambiguous column names"
                )
            available_by_casefold[folded] = str(available)
        if any(column.casefold() not in available_by_casefold for column in columns):
            raise InputError(
                "HCAP required columns is unavailable"
            )
        provider_columns = [
            available_by_casefold[column.casefold()] for column in columns
        ]
        frame, _ = pyreadstat.read_dta(str(path), usecols=provider_columns)
    except (ValueError, pyreadstat.ReadstatError) as error:
        raise InputError(
            "HCAP required columns is unavailable"
        ) from error
    expected_by_casefold = {column.casefold(): column for column in columns}
    actual_by_casefold: dict[str, str] = {}
    for actual in frame.columns:
        folded = str(actual).casefold()
        if folded in actual_by_casefold:
            raise InputError(
                "HCAP Stata reader returned case-ambiguous column names"
            )
        actual_by_casefold[folded] = str(actual)
    if set(actual_by_casefold) != set(expected_by_casefold):
        raise InputError("HCAP column schema differs from specification")
    frame = frame.rename(
        columns={
            actual_by_casefold[folded]: canonical
            for folded, canonical in expected_by_casefold.items()
        }
    )
    return frame.loc[:, columns]


def _load_hcap(
    geometry: pd.DataFrame,
    respondent: Path,
    summary: Path,
    tracker: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    respondent_frame = _read_hcap_dta_whitelist(
        respondent,
        ["HHID", "PN", "HCAP16WGTR", "VS1HCAPDX_EAP", "R1IFLAG"],
    )
    summary_frame = _read_hcap_dta_whitelist(
        summary, ["HHID", "PN", "VS1HCAPDX_EAP"]
    )
    tracker_frame = _read_hcap_dta_whitelist(
        tracker,
        ["HHID", "PN", "HCAP16REL", "HCAP16RESP", "STRATUM", "SECU", "VERSION"],
    )
    if (
        len(respondent_frame) != 3496
        or len(summary_frame) != 3496
        or len(tracker_frame) != 46766
    ):
        raise InputError("HCAP provider row totals differ from specification")
    for frame in (respondent_frame, summary_frame, tracker_frame):
        frame["key"] = [
            _key_from_hhid_pn(hhid, pn)
            for hhid, pn in zip(frame["HHID"], frame["PN"], strict=True)
        ]
        if frame["key"].duplicated().any():
            raise InputError("HCAP provider key is duplicated")
    if set(respondent_frame["key"]) != set(summary_frame["key"]):
        raise InputError("HCAP respondent and summary keys differ")
    response = pd.to_numeric(tracker_frame["HCAP16RESP"], errors="coerce")
    response_counts = {code: int(response.eq(code).sum()) for code in (0, 1, 2, 3)}
    if response_counts != {0: 929, 1: 149, 2: 313, 3: 3034} or int(response.isna().sum()) != 42341:
        raise InputError("HCAP Tracker response counts differ from specification")
    release = pd.to_numeric(tracker_frame["HCAP16REL"], errors="coerce")
    released = release.between(1, 9, inclusive="both")
    version = pd.to_numeric(tracker_frame["VERSION"], errors="coerce")
    if (
        int(released.sum()) != 4425
        or int(release.eq(99).sum()) != 576
        or int(release.isna().sum()) != 41765
        or not released.eq(response.notna()).all()
        or version.isna().any()
        or not version.eq(1).all()
    ):
        raise InputError("HCAP Tracker release/version differs from specification")
    tracker_hcap = tracker_frame.loc[response.isin([1, 2, 3])].copy()
    if set(tracker_hcap["key"]) != set(respondent_frame["key"]):
        raise InputError("HCAP products do not match Tracker participants")
    hcap = respondent_frame.merge(
        summary_frame[["key", "VS1HCAPDX_EAP"]],
        on="key",
        how="inner",
        validate="one_to_one",
        suffixes=("__respondent", "__summary"),
    ).merge(
        tracker_hcap[["key", "HCAP16RESP", "STRATUM", "SECU"]],
        on="key",
        how="inner",
        validate="one_to_one",
    )
    respondent_dx = hcap["VS1HCAPDX_EAP__respondent"]
    summary_dx = hcap["VS1HCAPDX_EAP__summary"]
    if not (respondent_dx.eq(summary_dx) | (respondent_dx.isna() & summary_dx.isna())).all():
        raise InputError("HCAP respondent and summary diagnoses differ")
    hcap["VS1HCAPDX_EAP"] = summary_dx
    respondent_mode = pd.to_numeric(hcap["R1IFLAG"], errors="coerce")
    tracker_mode = pd.to_numeric(hcap["HCAP16RESP"], errors="coerce")
    if respondent_mode.isna().any() or not respondent_mode.eq(4 - tracker_mode).all():
        raise InputError("HCAP interview mode differs from Tracker")
    diagnosis = pd.to_numeric(hcap["VS1HCAPDX_EAP"], errors="coerce")
    if not set(diagnosis.dropna().astype(int)).issubset({1, 2, 3}):
        raise InputError("HCAP EAP criterion contains an invalid code")
    hcap["outcome"] = diagnosis.map({1: 0, 2: 0, 3: 1})
    hcap["outcome_ordinal"] = diagnosis.map({1: 0, 2: 1, 3: 2})
    hcap["weight"] = pd.to_numeric(hcap["HCAP16WGTR"], errors="coerce")
    hcap["stratum"] = pd.to_numeric(hcap["STRATUM"], errors="coerce").astype(int)
    hcap["psu"] = pd.to_numeric(hcap["SECU"], errors="coerce").astype(int)
    full_design = hcap[["key", "weight", "stratum", "psu"]].copy()
    domain = geometry.merge(
        hcap[
            ["key", "outcome", "outcome_ordinal", "weight", "stratum", "psu"]
        ].dropna(subset=["outcome", "outcome_ordinal"]),
        on="key",
        how="inner",
        validate="one_to_one",
    )
    domain[["outcome", "outcome_ordinal"]] = domain[
        ["outcome", "outcome_ordinal"]
    ].astype(int)
    return domain, full_design


def _load_adams(
    geometry: pd.DataFrame, wave_a_path: Path, tracker_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wave_a = _read_fixed_width(wave_a_path, ADAMS_WAVE_A_LAYOUT)
    tracker = _read_fixed_width(tracker_path, ADAMS_TRACKER_LAYOUT)
    if len(wave_a) != 856 or len(tracker) != 1770:
        raise InputError("ADAMS provider row totals differ from specification")
    for frame in (wave_a, tracker):
        frame["key"] = [
            _key_from_hhid_pn(hhid, pn)
            for hhid, pn in zip(frame["HHID"], frame["PN"], strict=True)
        ]
        if frame["key"].duplicated().any():
            raise InputError("ADAMS provider key is duplicated")
    assessed = pd.to_numeric(tracker["AASSESS"], errors="coerce")
    if assessed.value_counts().to_dict() != {5: 914, 1: 856}:
        raise InputError("ADAMS AASSESS counts differ from specification")
    design = tracker.loc[assessed.eq(1)].copy()
    if set(design["key"]) != set(wave_a["key"]):
        raise InputError("ADAMS assessed Tracker and Wave A keys differ")
    diagnosis_code = pd.to_numeric(wave_a["ADFDX1"], errors="coerce")
    allowed = ADAMS_NORMAL_CODES | ADAMS_CIND_CODES | ADAMS_DEMENTIA_CODES
    if diagnosis_code.isna().any() or not set(diagnosis_code.astype(int)).issubset(allowed):
        raise InputError("ADAMS ADFDX1 coding differs from specification")
    diagnosis_code = diagnosis_code.astype(int)
    wave_a["outcome"] = diagnosis_code.isin(ADAMS_DEMENTIA_CODES).astype(int)
    wave_a["outcome_ordinal"] = np.select(
        [
            diagnosis_code.isin(ADAMS_NORMAL_CODES),
            diagnosis_code.isin(ADAMS_CIND_CODES),
            diagnosis_code.isin(ADAMS_DEMENTIA_CODES),
        ],
        [0, 1, 2],
        default=-1,
    ).astype(int)
    wave_sel_all = pd.to_numeric(tracker["WAVESEL"], errors="coerce")
    if wave_sel_all.value_counts().to_dict() != {2: 1105, 1: 665}:
        raise InputError("ADAMS WAVESEL counts differ from specification")
    wave_sel = pd.to_numeric(design["WAVESEL"], errors="coerce")
    if wave_sel.isna().any() or not set(wave_sel.astype(int)).issubset({1, 2}):
        raise InputError("Assessed ADAMS WAVESEL is missing or invalid")
    design["landmark_wave_tracker"] = wave_sel.astype(int).map({1: 5, 2: 6})
    design["weight"] = pd.to_numeric(design["AASAMPWT_F"], errors="coerce")
    design["stratum"] = pd.to_numeric(design["SESTRAT"], errors="coerce").astype(int)
    design["psu"] = pd.to_numeric(design["SECLUST"], errors="coerce").astype(int)
    full_design = design[["key", "weight", "stratum", "psu", "landmark_wave_tracker"]]
    domain = geometry.merge(
        wave_a[["key", "outcome", "outcome_ordinal"]],
        on="key",
        how="inner",
        validate="one_to_one",
    ).merge(full_design, on="key", how="inner", validate="one_to_one")
    if not domain["landmark_wave"].astype(int).eq(domain["landmark_wave_tracker"]).all():
        raise InputError("ADAMS predictor row does not match WAVESEL")
    phase_one = domain["landmark_wave_tracker"].eq(5)
    if domain.loc[phase_one, "landmark_wave"].ne(5).any():
        raise InputError("WAVESEL phase 1 was scored at Wave 6")
    return domain, full_design[["key", "weight", "stratum", "psu"]]


def _binary_cell_is_disclosure_safe(frame: pd.DataFrame) -> bool:
    """An empty structural complement is safe; every nonempty cell needs 10/10."""
    if frame.empty:
        return True
    events = int(frame["outcome"].sum())
    controls = int((1 - frame["outcome"]).sum())
    return len(frame) >= MINIMUM_CELL and events >= MINIMUM_CELL and controls >= MINIMUM_CELL


def _prepare_association_predictors(domain: pd.DataFrame) -> pd.DataFrame:
    """Create the outcome-blind within-state margin scaling."""
    frame = domain.copy()
    required = (
        "hard_current", "margin_current", "age_years", "female",
        "education_level", "weight",
    )
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (
        frame[list(required)].isna().any(axis=None)
        or not np.isfinite(frame[list(required)].to_numpy(float)).all()
        or not set(frame["hard_current"].astype(int)).issubset({0, 1})
        or not set(frame["female"].astype(int)).issubset({0, 1})
        or not set(frame["education_level"].astype(int)).issubset({1, 2, 3})
        or (frame["age_years"] < 50).any()
        or (frame["age_years"] > 120).any()
        or (frame["weight"] <= 0).any()
    ):
        raise InputError("Association covariates differ from specification")
    hard = frame["hard_current"].astype(int).to_numpy()
    margin = frame["margin_current"].to_numpy(float)
    provider_weight = frame["weight"].to_numpy(float)
    residual = np.empty(len(frame), dtype=float)
    state_centers: dict[int, float] = {}
    for state in (0, 1):
        selected = hard == state
        if selected.sum() < MINIMUM_CELL:
            raise InputError(
                "Within-state margin scaling has an undersized hard state"
            )
        center = float(np.average(margin[selected], weights=provider_weight[selected]))
        state_centers[state] = center
        residual[selected] = margin[selected] - center
    pooled_sd = float(
        np.sqrt(np.average(np.square(residual), weights=provider_weight))
    )
    if not math.isfinite(pooled_sd) or pooled_sd <= 0:
        raise InputError("Within-state margin pooled SD is invalid")
    frame["margin_within_state_z"] = residual / pooled_sd
    age_center = float(np.average(frame["age_years"], weights=provider_weight))
    frame["age_per_10_years"] = (frame["age_years"] - age_center) / 10.0
    frame["education_middle_vs_low"] = frame["education_level"].eq(2).astype(int)
    frame["education_high_vs_low"] = frame["education_level"].eq(3).astype(int)
    frame.attrs["association_scaling"] = {
        "margin_h0_center": state_centers[0],
        "margin_h1_center": state_centers[1],
        "margin_pooled_within_state_sd": pooled_sd,
        "age_center_years": age_center,
    }
    return frame


def _association_terms(cohort: str) -> tuple[str, ...]:
    if cohort == "hcap":
        return ASSOCIATION_COMMON_TERMS
    if cohort == "adams":
        return ASSOCIATION_COMMON_TERMS + ("wavesel_w6_vs_w5",)
    raise InputError("Unknown association cohort")


def _association_design_matrix(
    frame: pd.DataFrame,
    cohort: str,
    *,
    reliability_context: bool = False,
) -> tuple[np.ndarray, tuple[str, ...]]:
    terms = _association_terms(cohort)
    columns: list[np.ndarray] = [np.ones(len(frame), dtype=float)]
    for term in terms:
        if term == "wavesel_w6_vs_w5":
            if "landmark_wave_tracker" not in frame:
                raise InputError("ADAMS WAVESEL covariate is absent")
            value = frame["landmark_wave_tracker"].astype(int).eq(6).to_numpy(float)
        else:
            value = pd.to_numeric(frame[term], errors="raise").to_numpy(float)
        columns.append(value)
    matrix = np.column_stack(columns)
    if (
        not np.isfinite(matrix).all()
        or np.linalg.matrix_rank(matrix) != matrix.shape[1]
    ):
        exception = (
            _AssociationReliabilityError
            if reliability_context
            else InputError
        )
        raise exception("Association design is nonfinite or rank deficient")
    return matrix, ("intercept",) + terms


def _fit_weighted_logistic(
    frame: pd.DataFrame,
    weight: np.ndarray,
    cohort: str,
    *,
    reliability_context: bool = False,
) -> tuple[dict[str, float], int]:
    """Fit the unpenalized survey-weighted logistic model by IRLS."""
    x, names = _association_design_matrix(
        frame, cohort, reliability_context=reliability_context
    )
    y = frame["outcome"].astype(int).to_numpy()
    sample_weight = np.asarray(weight, dtype=float)
    invalid_weight_support = (
        sample_weight.shape != (len(frame),)
        or not np.isfinite(sample_weight).all()
        or np.any(sample_weight < 0)
    )
    positive_weight = sample_weight > 0 if not invalid_weight_support else None
    invalid_replicate_support = (
        not invalid_weight_support
        and (
            sample_weight[y == 0].sum() <= 0
            or sample_weight[y == 1].sum() <= 0
            or np.linalg.matrix_rank(x[positive_weight]) != x.shape[1]
        )
    )
    if invalid_weight_support or invalid_replicate_support:
        exception = (
            _AssociationReliabilityError
            if reliability_context
            else InputError
        )
        raise exception("Association weights or outcome support are invalid")
    prevalence = float(np.average(y, weights=sample_weight))
    if not 0 < prevalence < 1:
        raise InputError("Association outcome class is empty")
    beta = np.zeros(x.shape[1], dtype=float)
    beta[0] = math.log(prevalence / (1.0 - prevalence))

    def log_likelihood(candidate: np.ndarray) -> float:
        if not np.isfinite(candidate).all() or np.max(np.abs(candidate)) > 25:
            return -math.inf
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            eta = x @ candidate
            value = np.sum(sample_weight * (y * eta - np.logaddexp(0.0, eta)))
        return float(value) if math.isfinite(float(value)) else -math.inf

    current_likelihood = log_likelihood(beta)
    converged = False
    iterations = 0
    for iterations in range(1, 101):
        if not np.isfinite(beta).all() or np.max(np.abs(beta)) > 25:
            raise _AssociationReliabilityError(
                "Association iterates indicate separation"
            )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            eta = np.clip(x @ beta, -35.0, 35.0)
            probability = 1.0 / (1.0 + np.exp(-eta))
            working_weight = sample_weight * probability * (1.0 - probability)
            information = x.T @ (working_weight[:, None] * x)
            gradient = x.T @ (sample_weight * (y - probability))
        if (
            not np.isfinite(information).all()
            or not np.isfinite(gradient).all()
            or np.linalg.cond(information) > 1e12
        ):
            raise _AssociationReliabilityError(
                "Association information matrix is singular"
            )
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError as exc:
            raise _AssociationReliabilityError(
                "Association information solve failed"
            ) from exc
        multiplier = 1.0
        accepted = False
        while multiplier >= 2.0 ** -20:
            candidate = beta + multiplier * step
            candidate_likelihood = log_likelihood(candidate)
            if math.isfinite(candidate_likelihood) and (
                candidate_likelihood >= current_likelihood - 1e-10
            ):
                accepted = True
                break
            multiplier /= 2.0
        if not accepted:
            raise _AssociationReliabilityError(
                "Association IRLS line search failed"
            )
        beta = candidate
        current_likelihood = candidate_likelihood
        if np.max(np.abs(multiplier * step)) <= 1e-9:
            converged = True
            break
    if not converged or not np.isfinite(beta).all() or np.max(np.abs(beta)) > 25:
        raise _AssociationReliabilityError(
            "Association model did not converge without separation"
        )
    return dict(zip(names, beta, strict=True)), iterations


def _association_disclosure_safe(frame: pd.DataFrame, cohort: str) -> bool:
    categorical: dict[str, tuple[int, ...]] = {
        "hard_current": (0, 1),
        "female": (0, 1),
        "education_level": (1, 2, 3),
    }
    if cohort == "adams":
        categorical["landmark_wave_tracker"] = (5, 6)
    for column, levels in categorical.items():
        for level in levels:
            atom = frame.loc[pd.to_numeric(frame[column], errors="raise").eq(level)]
            if atom.empty or not _binary_cell_is_disclosure_safe(atom):
                return False
    return True


def _survey_logistic_association(
    frame: pd.DataFrame,
    full_design: pd.DataFrame,
    *,
    cohort: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    design_strata = int(full_design["stratum"].nunique())
    design_psus = int(
        full_design[["stratum", "psu"]].drop_duplicates().shape[0]
    )
    design_degrees_of_freedom = design_psus - design_strata
    if design_degrees_of_freedom <= 0:
        raise InputError("Survey-logistic design df is nonpositive")
    terms = _association_terms(cohort)
    # Schema, covariates and rank are data requirements: they always abort.
    _association_design_matrix(frame, cohort)
    seed = 20260931 if cohort == "hcap" else 20260932

    def suppressed_result(
        status: str,
        *,
        successful: int,
        attempts: int,
        iterations: int | float,
        failures: Counter[str],
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        rows = [
            {
                "cohort": cohort,
                "model": "survey_logistic_dementia",
                "term": term,
                "scale": (
                    "per_within_hard_state_pooled_SD"
                    if term == "margin_within_state_z"
                    else "per_10_years" if term == "age_per_10_years"
                    else "odds_ratio"
                ),
                "odds_ratio": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "p_value": np.nan,
                "log_odds": np.nan,
                "rao_wu_bootstrap_se": np.nan,
                "design_degrees_of_freedom": design_degrees_of_freedom,
                "n": int(len(frame)),
                "events": int(frame["outcome"].sum()),
                "controls": int((1 - frame["outcome"]).sum()),
                "cell_gate_safe": False,
                "association_status": status,
            }
            for term in terms
        ]
        diagnostic = {
            "family": "survey_logistic_association",
            "weighting": "survey_weighted",
            "analysis_role": "secondary",
            "status": status,
            "seed": seed,
            "requested": BOOTSTRAP_REPLICATES,
            "successful": successful,
            "attempts": attempts,
            "point_irls_iterations": iterations,
            "design_degrees_of_freedom": design_degrees_of_freedom,
            "failures": json.dumps(
                dict(sorted(failures.items())), sort_keys=True
            ),
        }
        return pd.DataFrame(rows), diagnostic

    try:
        point, iterations = _fit_weighted_logistic(
            frame, frame["weight"].to_numpy(float), cohort
        )
    except _AssociationReliabilityError:
        point_failures: Counter[str] = Counter(
            {"point_nonconvergence_or_separation": 1}
        )
        return suppressed_result(
            "suppressed_point_nonconvergence",
            successful=0,
            attempts=0,
            iterations=np.nan,
            failures=point_failures,
        )
    observed: dict[str, list[float]] = {term: [] for term in terms}
    failures: Counter[str] = Counter()
    rng = np.random.default_rng(seed)
    attempts = 0
    while (
        attempts < BOOTSTRAP_MAXIMUM_ATTEMPTS
        and len(observed[terms[0]]) < BOOTSTRAP_REPLICATES
    ):
        attempts += 1
        multiplicity = _survey_multiplicity(full_design, rng)
        replicate_weight = (
            frame["weight"].to_numpy(float)
            * frame["key"].map(multiplicity).fillna(0).to_numpy(float)
        )
        try:
            estimate, _ = _fit_weighted_logistic(
                frame,
                replicate_weight,
                cohort,
                reliability_context=True,
            )
        except _AssociationReliabilityError:
            failures["nonconvergence_or_separation"] += 1
            continue
        if set(estimate) != {"intercept", *terms}:
            raise InputError(
                "Survey-logistic replicate coefficient schema differs from specification"
            )
        for term in terms:
            observed[term].append(float(estimate[term]))
    successful = len(observed[terms[0]])
    if successful < BOOTSTRAP_MINIMUM_SUCCESS:
        failures["insufficient_valid_replicates"] += 1
        return suppressed_result(
            "suppressed_insufficient_bootstrap",
            successful=successful,
            attempts=attempts,
            iterations=iterations,
            failures=failures,
        )
    safe = _association_disclosure_safe(frame, cohort)
    standard_errors = {
        term: float(np.asarray(observed[term], dtype=float).std(ddof=1))
        for term in terms
    }
    if any(
        not math.isfinite(standard_error) or standard_error <= 0
        for standard_error in standard_errors.values()
    ):
        failures["invalid_rao_wu_standard_error"] += 1
        return suppressed_result(
            "suppressed_invalid_standard_error",
            successful=successful,
            attempts=attempts,
            iterations=iterations,
            failures=failures,
        )
    rows = []
    for term in terms:
        standard_error = standard_errors[term]
        log_odds = float(point[term])
        critical = float(student_t.ppf(0.975, design_degrees_of_freedom))
        p_value = float(
            2.0
            * student_t.sf(
                abs(log_odds / standard_error), design_degrees_of_freedom
            )
        )
        low = log_odds - critical * standard_error
        high = log_odds + critical * standard_error
        rows.append(
            {
                "cohort": cohort,
                "model": "survey_logistic_dementia",
                "term": term,
                "scale": (
                    "per_within_hard_state_pooled_SD"
                    if term == "margin_within_state_z"
                    else "per_10_years" if term == "age_per_10_years"
                    else "odds_ratio"
                ),
                "odds_ratio": math.exp(log_odds),
                "ci95_low": math.exp(float(low)),
                "ci95_high": math.exp(float(high)),
                "p_value": p_value,
                "log_odds": log_odds,
                "rao_wu_bootstrap_se": standard_error,
                "design_degrees_of_freedom": design_degrees_of_freedom,
                "n": int(len(frame)),
                "events": int(frame["outcome"].sum()),
                "controls": int((1 - frame["outcome"]).sum()),
                "cell_gate_safe": safe,
                "association_status": "reliable",
            }
        )
    diagnostics = {
        "family": "survey_logistic_association",
        "weighting": "survey_weighted",
        "analysis_role": "secondary",
        "status": "reliable",
        "seed": seed,
        "requested": BOOTSTRAP_REPLICATES,
        "successful": successful,
        "attempts": attempts,
        "point_irls_iterations": iterations,
        "design_degrees_of_freedom": design_degrees_of_freedom,
        "failures": json.dumps(dict(sorted(failures.items())), sort_keys=True),
    }
    return pd.DataFrame(rows), diagnostics


def _screen_association_table(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    unsafe = ~output["cell_gate_safe"].astype(bool) | ~output[
        "association_status"
    ].eq("reliable")
    protected = [
        "odds_ratio", "ci95_low", "ci95_high", "p_value", "log_odds",
        "rao_wu_bootstrap_se", "n", "events", "controls",
    ]
    output.loc[unsafe, protected] = np.nan
    output["suppressed"] = unsafe
    return _round_participant_counts(output, ("n", "events", "controls"))


def _round_participant_counts(
    frame: pd.DataFrame, columns: tuple[str, ...]
) -> pd.DataFrame:
    """Round already disclosure-safe participant counts down to tens."""
    output = frame.copy()
    for column in columns:
        if column in output:
            numeric = pd.to_numeric(output[column], errors="coerce")
            output[column] = np.where(
                numeric.notna(), np.floor(numeric / 10.0) * 10.0, np.nan
            )
    return output


def _auc(y: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    if len(np.unique(y)) != 2 or weight[y == 1].sum() <= 0 or weight[y == 0].sum() <= 0:
        raise ValueError("empty outcome class")
    return float(roc_auc_score(y, score, sample_weight=weight))


def _binary_pair_mass(y: np.ndarray, weight: np.ndarray) -> float:
    return float(weight[y == 1].sum() * weight[y == 0].sum())


def _stratified_auc(
    y: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray,
    strata: np.ndarray,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for stratum in sorted(np.unique(strata)):
        selected = strata == stratum
        pair_mass = _binary_pair_mass(y[selected], weight[selected])
        if pair_mass <= 0:
            continue
        numerator += pair_mass * _auc(y[selected], score[selected], weight[selected])
        denominator += pair_mass
    if denominator <= 0:
        raise ValueError("zero within-stratum binary pair mass")
    return float(numerator / denominator)


def _stratified_binary_pair_mass(
    y: np.ndarray, weight: np.ndarray, strata: np.ndarray
) -> float:
    return float(
        sum(
            _binary_pair_mass(y[strata == stratum], weight[strata == stratum])
            for stratum in np.unique(strata)
        )
    )


def _ordinal_concordance(
    outcome: np.ndarray, score: np.ndarray, weight: np.ndarray
) -> float:
    """Weighted concordance over all pairs with unequal ordinal outcomes."""
    levels = sorted(
        level for level in np.unique(outcome) if weight[outcome == level].sum() > 0
    )
    if len(levels) < 2:
        raise ValueError("fewer than two positive-mass ordinal classes")
    numerator = 0.0
    denominator = 0.0
    for lower_index, lower in enumerate(levels[:-1]):
        for higher in levels[lower_index + 1 :]:
            selected = (outcome == lower) | (outcome == higher)
            pair_mass = (
                weight[outcome == lower].sum() * weight[outcome == higher].sum()
            )
            binary = (outcome[selected] == higher).astype(int)
            numerator += pair_mass * _auc(binary, score[selected], weight[selected])
            denominator += pair_mass
    if denominator <= 0:
        raise ValueError("zero ordinal pair mass")
    return float(numerator / denominator)


def _ordinal_pair_mass(outcome: np.ndarray, weight: np.ndarray) -> float:
    levels = sorted(np.unique(outcome))
    return float(
        sum(
            weight[outcome == lower].sum() * weight[outcome == higher].sum()
            for lower_index, lower in enumerate(levels[:-1])
            for higher in levels[lower_index + 1 :]
        )
    )


def _stratified_ordinal_concordance(
    outcome: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray,
    strata: np.ndarray,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for stratum in sorted(np.unique(strata)):
        selected = strata == stratum
        pair_mass = _ordinal_pair_mass(outcome[selected], weight[selected])
        if pair_mass <= 0:
            continue
        numerator += pair_mass * _ordinal_concordance(
            outcome[selected], score[selected], weight[selected]
        )
        denominator += pair_mass
    if denominator <= 0:
        raise ValueError("zero within-stratum ordinal pair mass")
    return float(numerator / denominator)


def _stratified_ordinal_pair_mass(
    outcome: np.ndarray, weight: np.ndarray, strata: np.ndarray
) -> float:
    return float(
        sum(
            _ordinal_pair_mass(
                outcome[strata == stratum], weight[strata == stratum]
            )
            for stratum in np.unique(strata)
        )
    )


def _binary_nested_estimands(
    frame: pd.DataFrame,
    weight: np.ndarray,
    *,
    strata: np.ndarray,
    hard_column: str,
    margin_column: str,
    hard_name: str,
    margin_name: str,
    contrast_name: str,
    family: str,
    key_prefix: str = "",
) -> dict[str, float]:
    y = frame["outcome"].astype(int).to_numpy()
    hard = frame[hard_column].astype(int).to_numpy()
    margin = frame[margin_column].to_numpy(float)
    if not np.array_equal(hard, (margin > 0).astype(int)):
        raise ValueError("nested hard state differs from margin > 0")
    auc_kind = f"{key_prefix}auc" if key_prefix else "auc"
    contrast_kind = f"{key_prefix}contrast" if key_prefix else "contrast"
    suffix = f"_{key_prefix.rstrip('_')}" if key_prefix else ""
    hard_auc = _stratified_auc(y, hard.astype(float), weight, strata)
    margin_auc = _stratified_auc(y, margin, weight, strata)
    estimates = {
        f"{auc_kind}::{hard_name}": hard_auc,
        f"{auc_kind}::{margin_name}": margin_auc,
        f"{contrast_kind}::{contrast_name}": margin_auc - hard_auc,
    }
    total_pair_mass = _stratified_binary_pair_mass(y, weight, strata)
    contribution_sum = 0.0
    for state in (0, 1):
        selected = hard == state
        within_auc = _stratified_auc(
            y[selected], margin[selected], weight[selected], strata[selected]
        )
        pair_weight = (
            _stratified_binary_pair_mass(
                y[selected], weight[selected], strata[selected]
            )
            / total_pair_mass
        )
        contribution = pair_weight * (within_auc - 0.5)
        estimates[f"within{suffix}::{family}::H{state}"] = within_auc
        estimates[f"pair_weight{suffix}::{family}::H{state}"] = pair_weight
        estimates[f"contribution{suffix}::{family}::H{state}"] = contribution
        contribution_sum += contribution
    estimates[f"decomposition{suffix}::{family}"] = contribution_sum
    estimates[f"decomposition_error{suffix}::{family}"] = (
        contribution_sum - (margin_auc - hard_auc)
    )
    return estimates


def _ordinal_nested_estimands(
    frame: pd.DataFrame, weight: np.ndarray, *, strata: np.ndarray
) -> dict[str, float]:
    ordinal = frame["outcome_ordinal"].astype(int).to_numpy()
    hard = frame["hard_current"].astype(int).to_numpy()
    margin = frame["margin_current"].to_numpy(float)
    if not np.array_equal(hard, (margin > 0).astype(int)):
        raise ValueError("nested ordinal hard state differs from margin > 0")
    hard_value = _stratified_ordinal_concordance(ordinal, hard, weight, strata)
    margin_value = _stratified_ordinal_concordance(
        ordinal, margin, weight, strata
    )
    estimates = {
        "ordinal_concordance::current_hard": hard_value,
        "ordinal_concordance::current_margin": margin_value,
        "ordinal_contrast::current_margin_minus_current_hard": (
            margin_value - hard_value
        ),
    }
    total_mass = _stratified_ordinal_pair_mass(ordinal, weight, strata)
    contribution_sum = 0.0
    for state in (0, 1):
        selected = hard == state
        within_mass = _stratified_ordinal_pair_mass(
            ordinal[selected], weight[selected], strata[selected]
        )
        within_value = _stratified_ordinal_concordance(
            ordinal[selected], margin[selected], weight[selected], strata[selected]
        )
        pair_weight = within_mass / total_mass
        contribution = pair_weight * (within_value - 0.5)
        estimates[f"within_ordinal::current::H{state}"] = within_value
        estimates[f"pair_weight_ordinal::current::H{state}"] = pair_weight
        estimates[f"contribution_ordinal::current::H{state}"] = contribution
        contribution_sum += contribution
    estimates["decomposition_ordinal::current"] = contribution_sum
    estimates["decomposition_error_ordinal::current"] = (
        contribution_sum - (margin_value - hard_value)
    )
    return estimates


def _survey_multiplicity(
    full_design: pd.DataFrame, rng: np.random.Generator
) -> pd.Series:
    multiplier = pd.Series(0.0, index=full_design["key"].astype(str))
    for _, rows in full_design.groupby("stratum", observed=True):
        psus = np.sort(rows["psu"].unique())
        n_h = len(psus)
        if n_h < 2:
            raise InputError(
                "Rao-Wu bootstrap requires at least two PSUs per stratum"
            )
        draw = rng.choice(psus, size=n_h - 1, replace=True)
        counts = Counter(draw.tolist())
        for psu, count in counts.items():
            keys = rows.loc[rows["psu"].eq(psu), "key"].astype(str)
            multiplier.loc[keys] = float(count) * n_h / (n_h - 1)
    return multiplier
