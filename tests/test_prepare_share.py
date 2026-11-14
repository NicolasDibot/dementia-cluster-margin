"""Synthetic checks for the SHARE cohort recipe and released-file selection."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "dementia_margin" / "prepare_share.R"
RSCRIPT = shutil.which("Rscript")
pytestmark = pytest.mark.skipif(RSCRIPT is None, reason="Rscript is unavailable")

FIXTURE_R = r'''
make_fixture <- function() {
  set.seed(382L)
  n <- 600L
  visits <- sample(rep(1:6, each=100L))
  wide <- data.frame(mergeid=sprintf("synthetic-%04d",seq_len(n)),
    ragender=rep(c(1,2),length.out=n), raeducl=rep(c(1,2,3),length.out=n),
    rabyear=sample(1930:1950,n,replace=TRUE), rabmonth=sample(1:12,n,replace=TRUE),
    isocountry=sample(c(40,250,276),n,replace=TRUE))
  years <- c(2004L,2006L,2011L,2013L,2015L,2017L)
  for (j in seq_along(waves)) {
    prefix <- paste0("r",waves[j])
    wide[[paste0(prefix,"iwm")]] <- rep(6L,n)
    wide[[paste0(prefix,"iwy")]] <- ifelse(visits >= j,years[j],NA)
    wide[[paste0(prefix,"wtresp")]] <- runif(n,.5,2)
    for (stem in c(functional_binary,cognitive_flags)) {
      wide[[paste0(prefix,stem)]] <- sample(c(0L,1L,NA),n,replace=TRUE,prob=c(.6,.35,.05))
    }
    for (stem in functional_counts) wide[[paste0(prefix,stem)]] <- sample(0:6,n,replace=TRUE)
    wide[[paste0(prefix,"imrc")]] <- sample(c(0:10,NA),n,replace=TRUE)
    wide[[paste0(prefix,"dlrc")]] <- sample(c(0:10,NA),n,replace=TRUE)
    wide[[paste0(prefix,"tr20")]] <- wide[[paste0(prefix,"imrc")]] + wide[[paste0(prefix,"dlrc")]]
    wide[[paste0(prefix,"verbf")]] <- sample(c(0:30,NA),n,replace=TRUE)
  }
  wide
}
'''


def run_r(code):
    statement = f"source({json.dumps(str(SCRIPT))});\n{FIXTURE_R}\n{code}"
    result = subprocess.run([RSCRIPT, "-e", statement], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_prepared_schema_and_rows():
    run_r(r'''
    wide <- make_fixture()
    result <- prepare_share_predictors(wide,enforce_reference_counts=FALSE)
    stopifnot(identical(dim(result),c(2000L,38L)), identical(names(result),output_columns))
    stopifnot(identical(as.integer(table(result$phase)),c(500L,500L,400L,300L,200L,100L)))
    stopifnot(!anyDuplicated(rownames(result)), all(is.finite(result$ipw_weight)), all(result$ipw_weight>0))
    stopifnot(is.factor(result$phase), is.factor(result$cogimp), is.factor(result$cogothp))
    stopifnot(is.numeric(result$imrc), is.numeric(result$newage), anyNA(result$imrc))
    stopifnot(all(vapply(result[functional_binary],is.factor,logical(1))))
    original <- stack_interviews(wide)
    original <- original[original$mergeid %in% eligible_ids(original),]
    stopifnot(identical(result$newage,original$newage), identical(result$imrc,original$imrc))
    stopifnot(identical(is.na(result$hlthlma),is.na(original$hlthlma)))
    ''')


def test_selection_and_attrition_weights():
    run_r(r'''
    wide <- make_fixture()
    long <- stack_interviews(wide)
    selected <- eligible_ids(long)
    stopifnot(length(selected)==500L)
    weights <- attrition_weight_table(long,selected)
    w1 <- weights[weights$phase==1,]
    w2 <- weights[weights$phase==2,]
    stopifnot(identical(w1$mergeid,w2$mergeid),identical(w1$ipw_weight,w2$ipw_weight))
    # The age cutoff is assessed at Wave 1 and attendance must be consecutive.
    id <- selected[1]
    altered <- long
    altered$newage[altered$mergeid==id & altered$phase==1] <- 49.99
    stopifnot(!(id %in% eligible_ids(altered)))
    id <- long$mergeid[long$phase==7][1]
    altered <- long[!(long$mergeid==id & long$phase==4),]
    stopifnot(!(id %in% eligible_ids(altered)))
    ''')


def test_invalid_source_and_reference_counts_fail():
    run_r(r'''
    wide <- make_fixture()
    fail <- function(expr) inherits(tryCatch(expr,error=function(e)e),"error")
    incomplete <- wide
    incomplete$r1imrc <- NULL
    stopifnot(fail(prepare_share_predictors(incomplete,FALSE)))
    stopifnot(fail(prepare_share_predictors(wide,TRUE)))
    stopifnot(fail(prepare_share_predictors(rbind(wide,wide[1,]),FALSE)))
    ''')


def test_selective_dta_reader(tmp_path):
    dependencies = subprocess.run(
        [RSCRIPT, "-e", "quit(status=ifelse(requireNamespace('haven',quietly=TRUE) && requireNamespace('tidyselect',quietly=TRUE),0L,1L))"],
        capture_output=True, text=True,
    )
    if dependencies.returncode:
        pytest.skip("haven and tidyselect are unavailable")
    path = json.dumps(str(tmp_path / "synthetic.dta"))
    run_r(f'''
    wide <- make_fixture()
    wide$unrelated_field <- 42
    haven::write_dta(wide,{path})
    selected <- read_harmonized_share({path})
    stopifnot(setequal(names(selected),required_source_columns()))
    a <- prepare_share_predictors(wide,FALSE)
    b <- prepare_share_predictors(selected,FALSE)
    stopifnot(identical(a,b))
    ''')
