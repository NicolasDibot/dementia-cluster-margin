#!/usr/bin/env Rscript
# Published SHARE variable grouping, imputation and five-dimensional factor model.
suppressPackageStartupMessages({library(FactoMineR); library(missMDA)})
script_path <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)), mustWork=TRUE)
source(file.path(dirname(script_path), "mfa_helpers.R"))
paths <- analysis_paths(commandArgs(trailingOnly=TRUE), script_path)
input_path <- paths$input
output_dir <- paths$output
expected_columns <- c(
  "phase", "ragender", "newage", "raeducl",
  "hlthlma", "walkra", "dressa", "batha", "eata", "beda", "toilta",
  "phonea", "medsa", "moneya", "shopa", "mealsa", "mapa", "housewka",
  "walk100a", "sita", "chaira", "climsa", "clim1a", "lifta", "stoopa",
  "armsa", "pusha", "dimea", "adltot_s", "iadltot1_s", "cogimp",
  "cogothp", "imrc", "dlrc", "tr20", "verbf", "wtresp", "ipw_weight"
)
group_sizes <- c(1L, 1L, 1L, 1L, 24L, 2L, 2L, 4L, 2L)
group_types <- c("n", "n", "s", "n", "n", "s", "n", "s", "s")

input <- readRDS(input_path)
if (!is.data.frame(input)) stop("Prepared input must be a data frame")
if (!identical(names(input), expected_columns)) {
  missing <- setdiff(expected_columns, names(input))
  extra <- setdiff(names(input), expected_columns)
  stop(
    "Prepared input schema mismatch; missing: ",
    paste(missing, collapse = ", "), "; extra: ", paste(extra, collapse = ", ")
  )
}
if (nrow(input) < 3L || anyNA(rownames(input)) || anyDuplicated(rownames(input))) {
  stop("Input row names must provide unique person-wave keys")
}
if (sum(group_sizes) != ncol(input)) stop("Invalid specified MFA group structure")

imputed <- missMDA::imputeMFA(
  X = input,
  ncp = 5L,
  group = group_sizes,
  type = group_types,
  num.group.sup = 9L,
  method = "Regularized"
)
completed <- as.data.frame(imputed$completeObs)
if (!identical(rownames(completed), rownames(input))) {
  stop("Imputation changed person-wave row order")
}

mfa <- FactoMineR::MFA(
  base = completed,
  ncp = 5L,
  group = group_sizes,
  type = group_types,
  graph = FALSE,
  num.group.sup = c(1L, 2L, 3L, 4L, 9L)
)
coordinates <- as.data.frame(mfa$ind$coord)
if (ncol(coordinates) != 5L || !identical(rownames(coordinates), rownames(input))) {
  stop("MFA did not return the specified five-dimensional person-wave geometry")
}
write_keyed(coordinates,file.path(output_dir,"source_coordinates.csv"))
write_keyed(completed,file.path(output_dir,"completed_predictors.csv"))
saveRDS(mfa,file.path(output_dir,"mfa.rds"),compress="xz")
saveRDS(imputed,file.path(output_dir,"imputation.rds"),compress="xz")
writeLines(capture.output(sessionInfo()),file.path(output_dir,"sessionInfo.txt"))
