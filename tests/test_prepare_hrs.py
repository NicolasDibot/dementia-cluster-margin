import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pyreadstat
import pytest

from dementia_margin.select_hrs import _read_adams_tracker_selection, _read_hcap_tracker_selection


def test_adams_tracker_selection(tmp_path):
    path = tmp_path / "tracker.da"
    lines = []
    for i in range(1770):
        line = list(" " * 261)
        line[:9] = f"{i + 1:06d}001"
        line[260] = "1" if i < 665 else "2"
        lines.append("".join(line))
    path.write_text("\n".join(lines) + "\n")
    selected = _read_adams_tracker_selection(path)
    assert list(selected.columns) == ["key", "landmark_wave", "target_source"]
    assert selected.landmark_wave.value_counts().to_dict() == {6: 1105, 5: 665}
    assert selected.key.iloc[0] == "1001"


def test_hcap_tracker_selection(tmp_path):
    frame = pd.DataFrame({
        "HHID": [f"{i + 1:06d}" for i in range(46766)],
        "PN": "001", "HCAP16RESP": np.r_[np.resize([1, 2, 3], 3496), np.zeros(43270)],
    })
    path = tmp_path / "tracker.dta"
    pyreadstat.write_dta(frame, str(path))
    selected = _read_hcap_tracker_selection(path)
    assert len(selected) == 3496
    assert selected.landmark_wave.eq(10).all()
    assert selected.key.iloc[0] == "1001"


@pytest.mark.parametrize("cohort", ["adams", "hcap"])
def test_raw_hrs_preparation(tmp_path, monkeypatch, cohort):
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not installed")
    available = subprocess.run([rscript, "-e", 'quit(status=ifelse(requireNamespace("haven",quietly=TRUE),0,1))'], capture_output=True)
    if available.returncode:
        pytest.skip("haven is not installed")
    monkeypatch.setenv("COHORT_DATA_ACCESS", "authorised")
    root = Path(__file__).resolve().parents[1]
    fixture = r'''
set.seed(62)
mode <- COHORT
target_n <- if(mode=="adams") 1770L else 3496L
keys <- c(1:400,10001:(10000+target_n))
n <- length(keys)
waves <- if(mode=="adams") 3:6 else 8:13
source_waves <- waves[1:3]
wide <- data.frame(hhidpn=keys,ragender=rep(1:2,length.out=n),raeduc=rep(1:5,length.out=n))
harm <- data.frame(hhidpn=keys)
stems <- c("hlthlm","walkra","dressa","batha","eata","beda","toilta","phonea","medsa",
 "moneya","shopa","mealsa","mapa","walk1a","sita","chaira","climsa","clim1a","lifta",
 "stoopa","armsa","pusha","dimea")
for(w in waves) {
 p <- paste0("r",w)
 wide[[paste0(p,"iwstat")]] <- 1
 wide[[paste0(p,"agey_m")]] <- runif(n,55,90)
 for(s in stems) wide[[paste0(p,s)]] <- rbinom(n,1,.3)
 wide[[paste0(p,"proxy")]] <- rbinom(n,1,.05)
 wide[[paste0(p,"imrc")]] <- sample(0:10,n,TRUE)
 wide[[paste0(p,"dlrc")]] <- sample(0:10,n,TRUE)
 wide[[paste0(p,"tr20")]] <- wide[[paste0(p,"imrc")]]+wide[[paste0(p,"dlrc")]]
 harm[[paste0(p,"limhwrk")]] <- rbinom(n,1,.3)
 harm[[paste0(p,"adltot6")]] <- sample(0:6,n,TRUE)
 harm[[paste0(p,"iadltot_h")]] <- sample(0:6,n,TRUE)
 # One proxy interview remains eligible without direct recall tests.
 for(s in c("imrc","dlrc","tr20")) wide[401:402,paste0(p,s)] <- NA_real_
 wide[401,paste0(p,"proxy")] <- 1
 wide[402,paste0(p,"proxy")] <- 0
 wide[403,paste0(p,stems)] <- NA_real_
 harm[403,paste0(p,"limhwrk")] <- NA_real_
 harm[404,paste0(p,c("adltot6","iadltot_h"))] <- NA_real_
 wide[405,paste0(p,stems[1:10])] <- NA_real_
 wide[406,paste0(p,"agey_m")] <- 49
}
# The source must contribute the first two waves; its third wave is optional.
wide[1,paste0("r",source_waves[1],"iwstat")] <- 0
wide[2,paste0("r",source_waves[3],"iwstat")] <- 0
landmark <- if(mode=="adams") c(rep(5,665),rep(6,1105)) else rep(10,target_n)
selection <- data.frame(key=as.character(10001:(10000+target_n)),landmark_wave=landmark,target_source=mode)
haven::write_dta(wide,file.path(DIR,"rand.dta"))
haven::write_dta(harm,file.path(DIR,"harm.dta"))
write.csv(selection,file.path(DIR,"selection.csv"),row.names=FALSE)
'''.replace("COHORT", json.dumps(cohort)).replace("DIR", json.dumps(str(tmp_path)))
    subprocess.run([rscript, "-e", fixture], capture_output=True, text=True, check=True)
    output = tmp_path / "prepared"
    result = subprocess.run([
        rscript, str(root / "dementia_margin" / "prepare_hrs.R"), cohort,
        str(tmp_path / "rand.dta"), str(tmp_path / "harm.dta"),
        str(tmp_path / "selection.csv"), str(output),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    check = r'''
x <- readRDS(INPUT)
role <- attr(x,"row_role")
key <- sub("-[^-]+$","",rownames(x))
stopifnot(ncol(x)==34L,sum(role=="source")==1196L,
 !any(key[role=="source"] %in% key[role=="target"]),
 !("1" %in% key),"10001" %in% key,
 !any(as.character(10002:10006) %in% key),
 all(role %in% c("source","target")))
selection <- attr(x,"target_landmarks")
later <- attr(x,"target_later_landmarks")
if(attr(x,"analysis_mode")=="adams") {
 stopifnot(!any(as.character(x$phase[key=="10001"])=="6"),length(later)==0L)
} else stopifnot(all(selection$landmark_wave==10L),identical(later,11:13),
 all(as.character(x$phase[role=="source"]) %in% c("8","9","10")),
 all(c("10","11","12","13") %in% as.character(x$phase[role=="target"])))
stopifnot(sum(key=="2")==2L)
'''.replace("INPUT", json.dumps(str(output / "prepared_predictors.rds")))
    result = subprocess.run([rscript, "-e", check], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
