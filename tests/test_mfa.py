import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("cohort", ["hrs", "share"])
def test_prepared_mfa_workflow(tmp_path, monkeypatch, cohort):
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not installed")
    available = subprocess.run([rscript, "-e", 'quit(status=ifelse(requireNamespace("FactoMineR",quietly=TRUE)&&requireNamespace("missMDA",quietly=TRUE),0,1))'], capture_output=True)
    if available.returncode:
        pytest.skip("FactoMineR and missMDA are not installed")
    monkeypatch.setenv("COHORT_DATA_ACCESS", "authorised")
    input_path = tmp_path / "input.rds"
    output = tmp_path / "output"
    code = r'''
set.seed(67)
n <- 220L
functional <- c("hlthlma","walkra","dressa","batha","eata","beda","toilta",
 "phonea","medsa","moneya","shopa","mealsa","mapa","housewka","walk100a",
 "sita","chaira","climsa","clim1a","lifta","stoopa","armsa","pusha","dimea")
x <- data.frame(phase=factor(c(rep(c(3,4,5),length.out=200),rep(6,20))),
 ragender=factor(sample(1:2,n,TRUE)),newage=runif(n,50,90),raeducl=factor(sample(1:3,n,TRUE)))
for (name in functional) x[[name]] <- factor(sample(0:1,n,TRUE))
x$adltot_s <- runif(n,0,5)
x$iadltot1_s <- runif(n,0,5)
if (COHORT=="share") x$cogimp <- factor(sample(0:1,n,TRUE))
x$cogothp <- factor(sample(0:1,n,TRUE))
x$imrc <- runif(n,0,10)
x$dlrc <- runif(n,0,10)
x$tr20 <- x$imrc+x$dlrc
if (COHORT=="share") {
 x$verbf <- runif(n,0,30)
 x$wtresp <- runif(n,.1,2)
 x$ipw_weight <- runif(n,.1,2)
}
rownames(x) <- paste0(seq_len(n),"-",as.character(x$phase))
if (COHORT=="hrs") {
 attr(x,"row_role") <- c(rep("source",200),rep("target",20))
 attr(x,"analysis_mode") <- "adams"
 attr(x,"target_landmarks") <- data.frame(key=as.character(201:220),landmark_wave=6L)
}
x$imrc[c(2,8,15,205)] <- NA_real_
x$hlthlma[c(4,10,25,206)] <- NA
saveRDS(x,INPUT)
'''.replace("COHORT", json.dumps(cohort)).replace("INPUT", json.dumps(str(input_path)))
    subprocess.run([rscript, "-e", code], check=True, capture_output=True, text=True)
    script = Path(__file__).resolve().parents[1] / "dementia_margin" / f"fit_{cohort}_mfa.R"
    result = subprocess.run([rscript, str(script), str(input_path), str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    source = pd.read_csv(output / "source_coordinates.csv")
    assert len(source) == (200 if cohort == "hrs" else 220)
    assert source.filter(regex=r"^Dim[._]").shape[1] == 5
    assert np.isfinite(source.filter(regex=r"^Dim[._]").to_numpy()).all()
    completed = pd.read_csv(output / "completed_predictors.csv")
    assert not completed.isna().any().any()
    if cohort == "hrs":
        assert len(pd.read_csv(output / "target_coordinates.csv")) == 20
        metadata = pd.read_csv(output / "target_geometry.csv")
        assert len(metadata) == 20
        assert metadata.landmark_wave.eq(6).all()
        selected = completed.loc[completed.person_wave_key.isin([f"{key}-6" for key in range(201, 221)])]
        np.testing.assert_array_equal(metadata.age_years, selected.newage)
        np.testing.assert_array_equal(metadata.female, selected.ragender.eq(2).astype(int))
        np.testing.assert_array_equal(metadata.education_level, selected.raeducl)


@pytest.mark.parametrize("via_symlink", [False, True])
def test_r_rejects_repository_output(tmp_path, monkeypatch, via_symlink):
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not installed")
    monkeypatch.setenv("COHORT_DATA_ACCESS", "authorised")
    root = Path(__file__).resolve().parents[1]
    script = root / "dementia_margin" / "fit_hrs_mfa.R"
    helper = root / "dementia_margin" / "mfa_helpers.R"
    dummy = tmp_path / "not_cohort_data.txt"
    dummy.touch()
    if via_symlink:
        link = tmp_path / "linked_repository"
        link.symlink_to(root, target_is_directory=True)
        output = str(link / "unwritten_results")
    else:
        output = "unwritten_results"
    code = f"source({json.dumps(str(helper))}); analysis_paths(c({json.dumps(str(dummy))},{json.dumps(output)}),{json.dumps(str(script))})"
    result = subprocess.run([rscript, "-e", code], cwd=root, capture_output=True, text=True)
    assert result.returncode != 0
    assert "outside the repository" in result.stderr
    assert not (root / "unwritten_results").exists()
