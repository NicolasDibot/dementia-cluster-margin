# Dementia cluster margin

Analysis code accompanying **A graded score recovers dementia signal hidden by binary labels in aging surveys**.

The analysis retains a two-cluster classification of cognitive and functional measures and adds a continuous score within each label. For factor coordinates `z`, the margin is

```text
M(z) = ||z - cN|| - ||z - cD||
H(z) = 1[M(z) > 0]
```

Here `cD` is the centroid of the cluster with greater cognitive and functional burden and `cN` is the other centroid. The margin is a difference of centroid distances, not a dementia probability. Since the binary label is its sign, discrimination gains arise exclusively from pairs with the same label.

## Scope

The package includes cohort predictor preparation, five-dimensional multiple factor analysis, two-cluster scoring and the analyses below:

| Analysis | Implementation |
| --- | --- |
| Outcome-free HRS clinical-target selection and RAND/Harmonized predictor preparation | `dementia_margin/select_hrs.py`, `prepare_hrs.R` |
| Harmonized SHARE predictor preparation and longitudinal sample selection | `dementia_margin/prepare_share.R` |
| HRS source-only imputation and supplementary MFA projection | `dementia_margin/fit_hrs_mfa.R`, `mfa_helpers.R` |
| SHARE imputation and MFA with the published variable grouping | `dementia_margin/fit_share_mfa.R` |
| Ward clustering, inertia-ratio selection and k-means consolidation | `dementia_margin/geometry.py`, `fit.py`, `consolidate_clusters.R` |
| Participant-level clustering stability with fixed coordinates and clinically assessed targets | `dementia_margin/stability.py`, `stability_analysis.py` |
| ADAMS and HCAP reference classifications, weighted AUC and exact decomposition | `dementia_margin/statistics.py`, `clinical.py` |
| Recall-only and function-only score comparisons | `dementia_margin/geometry.py`, `clinical.py` |
| HCAP discrimination by interval between the HRS interview and the 2016 assessment | `dementia_margin/prepare_hrs.R`, `fit_hrs_mfa.R`, `pipeline.py`, `manuscript_tables.py` |
| SHARE-HCAP linkage, outcomes, within-country discrimination and weighting analyses | `dementia_margin/share_hcap.py`, `share_weighting.py`, `rank_metrics.py` |
| Risk-factor comparisons on the same participant-periods | `dementia_margin/risk_panel.py`, `risk_factors.py`, `risk_analysis.py` |
| Figure 5 from paired aggregate model estimates | `scripts/plot_risk_factors.py` |
| Method diagram and clinical validation display (Figures 1 and 3) | `scripts/plot_method_and_validation.py` |
| Cluster profiles, stability and prospective validation (Figures 2 and 4) | `scripts/plot_profiles_and_prospective.py` |
| Tables 1–2, Extended Data Table 1, Supplementary Tables 1–2 and figure input tables | `dementia_margin/manuscript_tables.py`, `variables.py` |
| Provider-file-to-figure execution | `dementia_margin/pipeline.py` |

The pipeline links provider files, prepares predictors, fits the factor spaces and clusters, performs clinical and longitudinal analyses, and generates the manuscript tables and editable figures. Provider datasets must be obtained separately under the applicable access agreements.

Only analyses within this manuscript's scope are included. No participant data, fitted factor coordinates, fitted centroids or imputed records are distributed.

## Installation

Python 3.11 or later is required; the code was tested with Python 3.13.0. The pinned scientific packages are listed in `requirements.txt`. R is required for predictor preparation, factor analysis and clustering consolidation. The tested R version is 4.5.1 with FactoMineR 2.11, missMDA 1.19, jsonlite 2.0.0, haven 2.5.4 and tidyselect 1.2.1.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
Rscript -e 'install.packages(c("FactoMineR", "missMDA", "jsonlite", "haven", "tidyselect"), repos="https://cloud.r-project.org")'
python -m pytest -q
```

Tests use synthetic data only. R integration tests are skipped if Rscript or the required R packages are unavailable. An optional comparison with statsmodels is skipped when that package is absent; the remaining tests do not depend on it.

The relative-path examples below assume that the current directory is the cloned repository. After installing the package with `python -m pip install .`, the complete `python -m dementia_margin.pipeline` workflow can also run from another directory; its R helpers and figure scripts are included in the installation.

## Data access and storage

Researchers must obtain their own access to the HRS, ADAMS, HCAP, SHARE and SHARE-HCAP releases required by their analysis. Access is governed by the providers, not by this repository's software licence:

- HRS: <https://hrsdata.isr.umich.edu>
- SHARE Research Data Center: <https://share-eric.eu/data>

The code reads only the variables required for each analysis. Keep all cohort data and generated outputs **outside the Git checkout**, including when the checkout is private. Aggregate results also require disclosure review before redistribution. The commands require an explicit acknowledgement of the analyst's permissions:

```bash
export COHORT_DATA_ACCESS=authorised
```

This acknowledgement does not grant access, replace a data-use agreement, or extend permission to an unapproved person or computing environment.

## Inputs and execution

### Complete workflow

Save a JSON configuration outside the repository, replacing the example paths with the locations of your licensed provider files. The output directory must not already exist.

```json
{
  "output_dir": "/controlled/manuscript_results",
  "rscript": "Rscript",
  "seed": 20261109,
  "bootstrap": 1000,
  "stability_repeats": 100,
  "hrs": {
    "rand": "/controlled/randhrs1992_2022v1.dta",
    "harmonized": "/controlled/h_hrs_d.dta",
    "adams": {
      "respondent": "/controlled/ADAMS1AD_R.da",
      "tracker": "/controlled/ADAMS1TRK_R.da"
    },
    "hcap": {
      "respondent": "/controlled/hc16hp_r.dta",
      "summary": "/controlled/HC16HP_F.dta",
      "tracker": "/controlled/trk2022tr_r.dta"
    }
  },
  "share": {
    "harmonized": "/controlled/h_share_f.dta",
    "easyshare": "/controlled/easySHARE_rel9-0-0.dta",
    "hcap_respondent": "/controlled/sharew9_rel1-0-0_hcap_r.dta",
    "hcap_informant": "/controlled/sharew9_rel1-0-0_hcap_ff.dta",
    "wave9_coverscreen": "/controlled/sharew9_rel9-0-0_cv_r.dta",
    "wave9_weights": "/controlled/sharew9_rel9-0-0_gv_weights.dta",
    "raw_root": "/controlled/share_raw"
  }
}
```

```bash
python -m dementia_margin.pipeline --config /controlled/analysis.json
```

Add `--plan` to inspect the ordered steps without opening cohort files. The full workflow writes controlled cohort intermediates under `adams/`, `hcap/` (with the later HCAP interview waves under `hcap/interval/`) and `share/`, risk-model estimates under `risk_factors/`, manuscript tables under `tables/`, and editable SVG figures under `figures/`, with PNG renderings for Figures 2, 4 and 5. Execution logs and the resolved configuration remain in the same controlled output directory. Review disclosure requirements before releasing any generated aggregate files. Each stage is also available separately as described below.

The configured seed and bootstrap count govern discrimination analyses and SHARE-HCAP weighting; the seed also governs the stability refits. HRS factor analysis and k-means consolidation retain seed 20260901. The survey-adjusted clinical association models retain 1,000 replicates and fixed seeds 20260932 (ADAMS) and 20260931 (HCAP). These separate settings preserve the calculation used for each analysis.

### Predictor preparation

For HRS, select the entire ADAMS or HCAP target sample from the provider tracker before preparing predictors. The target selection does not use diagnostic outcomes. All selected participants are excluded from the source factor fit, including those without an eligible clinical assessment.

```bash
python -m dementia_margin.select_hrs --cohort adams \
  --tracker /controlled/ADAMS1TRK_R.da --output /controlled/adams_selection
python -m dementia_margin.select_hrs --cohort hcap \
  --tracker /controlled/trk2022tr_r.dta --output /controlled/hcap_selection

Rscript dementia_margin/prepare_hrs.R adams \
  /controlled/randhrs1992_2022v1.dta /controlled/h_hrs_d.dta \
  /controlled/adams_selection/selection.csv /controlled/adams_prepared
Rscript dementia_margin/prepare_hrs.R hcap \
  /controlled/randhrs1992_2022v1.dta /controlled/h_hrs_d.dta \
  /controlled/hcap_selection/selection.csv /controlled/hcap_prepared

Rscript dementia_margin/prepare_share.R \
  /controlled/h_share_f.dta /controlled/share_prepared
```

Each preparation command writes `prepared_predictors.rds` to a new output directory. HRS source participants must contribute the first two source waves, with the third retained when available: Waves 3–5 for the ADAMS period and Waves 8–10 for the HCAP period. The ADAMS target landmark is Wave 5 or 6 according to `WAVESEL`; the HCAP landmark is Wave 10. HCAP participants interviewed at Waves 11–13 (2012–2016) are also retained as targets, for the analysis by interval between the interview and the assessment. Target inclusion checks functional information, functional totals, cognitive information or a proxy interview, and at most 30% missing active predictors. Raw predictor values and the source/target roles are retained in the prepared frame. SHARE preparation applies the longitudinal inclusion and variable coding in its script to Harmonized SHARE release F. No diagnosis is used in either predictor preparation step.

### Factor analysis from prepared predictor frames

The R scripts accept the prepared RDS frames generated above, with unique person-wave row names and the exact column order shown in each script. Supplying an arbitrary selection of respondents does not reproduce the study.

```bash
Rscript dementia_margin/fit_hrs_mfa.R /controlled/adams_prepared/prepared_predictors.rds /controlled/adams_mfa
Rscript dementia_margin/fit_hrs_mfa.R /controlled/hcap_prepared/prepared_predictors.rds /controlled/hcap_mfa
Rscript dementia_margin/fit_share_mfa.R /controlled/share_prepared/prepared_predictors.rds /controlled/share_mfa
```

The output directory must be new and its parent directory must already exist. Both factor-analysis scripts export coordinates and completed predictors for the two-cluster step, together with the fitted objects and R session information, all under controlled storage. HRS frames produced by `prepare_hrs.R` also carry landmark metadata, enabling `fit_hrs_mfa.R` to export `target_geometry.csv` with completed landmark age, sex and education. For HCAP it also exports `target_geometry_wave11.csv`, `target_geometry_wave12.csv` and `target_geometry_wave13.csv`, which list the participants interviewed at each later wave. Older prepared frames without that metadata require a separately prepared `target_geometry` file.

For HRS, the `row_role` attribute identifies `source` and `target` rows. The `analysis_mode` attribute is `adams` or `hcap`. Source imputation uses regularized `imputeMFA` with five components. Missing supplementary values are completed with source-only medians or categorical modes, and supplementary individuals do not alter the active factor fit. The code checks source-fit invariance and preservation of complete target observations.

The SHARE grouping is retained exactly: functional items, functional totals, cognitive-context variables (`cogimp`, `cogothp`) and cognitive tests are active in MFA; wave, sex, age, education and weights are supplementary. HRS does not make proxy status active. These cohort-specific groupings must not be interchanged.

### Two-cluster scores

Run `python -m dementia_margin.fit --config /controlled/geometry.json`. The configuration is a JSON object with:

| Field | Content |
| --- | --- |
| `cohort` | `adams`, `hcap` or `share` |
| `source_coordinates` | CSV with unique `person_wave_key` and five `Dim.1`–`Dim.5` columns |
| `completed_predictors` | CSV with the same key and the completed active variables |
| `output_dir` | New directory outside the repository |
| `target_coordinates` | HRS only: five-dimensional coordinates of projected participants |
| `target_geometry` | HRS only: `key`, `landmark_wave`, `age_years`, `female`, `education_level` |
| `rscript` | Optional Rscript executable; default `Rscript` |

Person-wave keys take the form `participant-wave`. For HRS, source and target participants must be disjoint. Completed predictors contain the 24 functional items listed in `FUNCTION_PROFILE_ITEMS`, `adltot_s`, `iadltot1_s`, `imrc`, `dlrc` and `tr20`; SHARE additionally uses `verbf`. Values must use the same coding, imputation and source selection as the manuscript. HRS `female` is 0/1 and `education_level` is low/middle/high coded 1/2/3.

The cluster search considers K=2,…,10. This finite upper bound is an implementation choice; the manuscript specifies the lower bound. The code stops if the inertia criterion does not select two clusters. Ward centres initialize R's Hartigan–Wong k-means with at most ten iterations. No reference outcome is read during this step.

The output includes source and target scores and centroids. All are controlled participant-derived files and must remain outside the repository.

### ADAMS and HCAP

```bash
python -m dementia_margin.clinical --cohort adams \
  --scores /controlled/adams/target_scores.csv \
  --respondent /controlled/ADAMS1AD_R.da \
  --tracker /controlled/ADAMS1TRK_R.da \
  --output /controlled/results/adams

python -m dementia_margin.clinical --cohort hcap \
  --scores /controlled/hcap/target_scores.csv \
  --respondent /controlled/hc16hp_r.dta \
  --summary /controlled/HC16HP_F.dta \
  --tracker /controlled/trk2022tr_r.dta \
  --output /controlled/results/hcap
```

The ADAMS reader uses Wave A Early Version 6.0 and the Cross-Wave Tracker Early Version 7.0 fixed-width layouts. The HCAP reader uses the 2016 respondent and summary releases and the 2022 HRS tracker. Readers check released file dimensions, coding and linkage consistency; a later release can fail those checks and should not be substituted silently.

Discrimination is weighted by provider weights. ADAMS pairs are compared within sampling phase. AUC and ordinal-concordance intervals use 1,000 Rao–Wu resamples drawn from the full survey design before restricting to linked participants. The same replicate weights are used for each score and contrast. The adjusted model includes the label, within-label centred and scaled margin, age, sex, education and ADAMS phase. Its bootstrap standard errors use the survey design degrees of freedom. Bootstrap intervals may differ slightly across seeds and software versions.

To evaluate HCAP at a later interview wave, run `dementia_margin.fit` with `target_geometry` set to `target_geometry_wave11.csv`, `target_geometry_wave12.csv` or `target_geometry_wave13.csv`, then pass the resulting `target_scores.csv` to `dementia_margin.clinical`. The factor space, the clusters and the centroids are those of the Wave 8–10 source fit; only the wave at which HCAP participants are scored changes. The complete workflow runs the three later waves and assembles them with Wave 10 in `tables/extended_data_table_1.csv`.

### SHARE-HCAP comparison and weighting

```bash
python -m dementia_margin.share_hcap \
  --scores /controlled/share/geometry/target_scores.csv \
  --easyshare /controlled/easySHARE_rel9-0-0.dta \
  --harmonized-share /controlled/h_share_f.dta \
  --hcap-respondent /controlled/sharew9_rel1-0-0_hcap_r.dta \
  --hcap-informant /controlled/sharew9_rel1-0-0_hcap_ff.dta \
  --wave9-coverscreen /controlled/sharew9_rel9-0-0_cv_r.dta \
  --wave9-weights /controlled/sharew9_rel9-0-0_gv_weights.dta \
  --output /controlled/results/share_hcap
```

This command links Wave 7 scores for participants aged at least 60 years in Denmark, France, Germany and Italy to SHARE-HCAP. Outcomes are probable dementia, the 14-item informant assistance fraction (at least ten valid items), and the reversed mean of three standardized delayed-recall scores (at least two valid tests). The memory standardization reference is the linked assessment sample before the two-test restriction.

All comparisons use within-country pairs and paired household bootstrap draws within country. Supplementary Table 1 compares unweighted estimates, released calibrated propensity weights, and entropy-balancing weights. The finite target is the Wave 7 domain alive and main-interviewed at Wave 9, with Wave 9 calibration weights. Entropy balancing starts from HCAP design weights, uses only baseline covariates, and is refitted within every bootstrap draw. Target moments and their standardization are held fixed. The fixed ridge is 0.001; achieved balance and effective sample size are exported. Small changes in bootstrap interval bounds, particularly close to zero, should not be interpreted as a change in the underlying effect.

### Risk factors on a common population

```bash
python -m dementia_margin.risk_analysis \
  --hrs-scores /controlled/hrs/source_scores.csv \
  --share-scores /controlled/share/source_scores.csv \
  --rand /controlled/randhrs1992_2022v1.dta \
  --harmonized-hrs /controlled/h_hrs_d.dta \
  --harmonized-share /controlled/h_share_f.dta \
  --share-raw /controlled/share_raw \
  --output /controlled/results/risk_factors
```

The SHARE raw directory must contain the Release 9.0.0 Wave 4, 5 and 6 behaviour modules in their provider download directories. The panel uses HRS 8→9 and 9→10 and SHARE 1→2, 4→5, 5→6 and 6→7.

For each factor, both models retain the same eligible baseline-H0 observations, including people entering H1 at follow-up. Margin change is divided by its cohort-specific standard deviation before factor-specific exclusions. Models use identical weights and adjustment variables. Standard errors are clustered by household. Bonferroni correction uses the twelve-factor family separately for each cohort and outcome; HRS diagnosis-based high cholesterol is unavailable and is not replaced by screening or medication use. SHARE alcohol-frequency categories use weekly midpoint frequencies 7, 5.5, 3.5, 1.5, 0.375, 0.125 and 0.

The number of significant associations is descriptive. It is not converted into a comparison of power or an equivalent sample size.

### Figure 5

```bash
python scripts/plot_risk_factors.py \
  --input /controlled/results/risk_factors/common_population_paired_table.csv \
  --output-dir /controlled/figures
```

The script draws the existing coefficients and confidence intervals without fitting models. It produces editable SVG and high-resolution PNG versions. Font availability can change the PNG's typography without changing plotted values.

Figures 1 and 3 can be rendered with `python scripts/plot_method_and_validation.py --output-dir /controlled/figures`. By default, Figure 3 uses the reported aggregate estimates; supply `--clinical-estimates /controlled/manuscript_results/tables/clinical_estimands.csv` to draw newly computed estimates. Figures 2 and 4 are generated by `scripts/plot_profiles_and_prospective.py` from the exported cluster-profile, stability and prospective-estimate tables. All drawing scripts use supplied aggregate values and do not fit statistical models.

## Licence and citation

The software is distributed under the MIT licence. This licence does not apply to provider datasets. Citation metadata are provided in `CITATION.cff`.
