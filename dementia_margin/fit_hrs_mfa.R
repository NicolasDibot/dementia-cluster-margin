#!/usr/bin/env Rscript
# Source-only imputation and supplementary projection for the HRS factor model.
suppressPackageStartupMessages({library(FactoMineR); library(missMDA)})
script_path <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)), mustWork=TRUE)
source(file.path(dirname(script_path), "mfa_helpers.R"))
paths <- analysis_paths(commandArgs(trailingOnly=TRUE), script_path)
input_path <- paths$input
output_dir <- paths$output
functional_binary <- c(
  "hlthlma", "walkra", "dressa", "batha", "eata", "beda", "toilta",
  "phonea", "medsa", "moneya", "shopa", "mealsa", "mapa", "housewka", "walk100a",
  "sita", "chaira", "climsa", "clim1a", "lifta", "stoopa", "armsa",
  "pusha", "dimea"
)
expected_columns <- c(
  "phase", "ragender", "newage", "raeducl", functional_binary,
  "adltot_s", "iadltot1_s", "cogothp", "imrc", "dlrc", "tr20"
)
model_columns <- setdiff(expected_columns, "cogothp")
group_sizes <- c(1L, 1L, 1L, 1L, 24L, 2L, 3L)
group_types <- c("n", "n", "s", "n", "n", "s", "s")
active_columns <- c(functional_binary, "adltot_s", "iadltot1_s",
                    "imrc", "dlrc", "tr20")

input <- readRDS(input_path)
row_role <- attr(input,"row_role")
mode <- attr(input,"analysis_mode")
if (!is.data.frame(input) || !identical(names(input),expected_columns) ||
    !is.character(row_role) || length(row_role)!=nrow(input) ||
    !all(c("source","target") %in% row_role) ||
    !all(row_role %in% c("source","target","benchmark")) ||
    !(mode %in% c("adams","hcap")) || anyNA(rownames(input)) ||
    anyDuplicated(rownames(input))) stop("Invalid prepared HRS predictor frame")
source_index <- which(row_role=="source")
target_index <- which(row_role=="target")
supplementary_index <- which(row_role!="source")
# No active-at-imputation categorical modality may occur only among targets:
# otherwise droplevels could change the disjunctive table when targets are
# removed. Phase has an expected target-only follow-up level, so its entire
# group is explicitly supplementary in both imputeMFA and MFA.
phase_source_levels <- unique(as.character(input$phase[source_index]))
phase_supplementary_levels <- unique(as.character(input$phase[supplementary_index]))
phase_target_only <- setdiff(
  phase_supplementary_levels[!is.na(phase_supplementary_levels)],
  phase_source_levels[!is.na(phase_source_levels)]
)
expected_phase_target_only <- if (mode == "hcap") character(0) else "6"
if (!identical(phase_target_only, expected_phase_target_only)) {
  stop("Target-only phase modality differs from the specified supplementary level")
}
model_input <- input[, model_columns, drop = FALSE]
for (name in setdiff(
    names(model_input)[vapply(model_input, is.factor, logical(1))], "phase")) {
  source_levels <- unique(as.character(model_input[[name]][source_index]))
  target_levels <- unique(as.character(model_input[[name]][supplementary_index]))
  target_only <- setdiff(target_levels[!is.na(target_levels)],
                         source_levels[!is.na(source_levels)])
  if (length(target_only)) {
    stop(sprintf("Target-only categorical level in %s: %s", name,
                 paste(target_only, collapse = ", ")))
  }
}

fit_source_only <- function(frame) {
  set.seed(20260901L)
  imputation <- missMDA::imputeMFA(
    X = frame,
    ncp = 5L,
    group = group_sizes,
    type = group_types,
    method = "Regularized",
    ind.sup = NULL,
    num.group.sup = 1L
  )
  completed <- normalize_imputemfa_completion(
    imputation$completeObs, frame
  )
  mfa <- FactoMineR::MFA(
    base = completed,
    ncp = 5L,
    group = group_sizes,
    type = group_types,
    graph = FALSE,
    num.group.sup = c(1L, 2L, 3L, 4L),
    ind.sup = NULL
  )
  list(imputation = imputation, completed = completed, mfa = mfa)
}

source_input <- droplevels(model_input[source_index, , drop = FALSE])
source_primary_fit <- fit_source_only(source_input)
source_only_fit <- fit_source_only(source_input)
if (!identical(source_primary_fit$completed, source_only_fit$completed)) {
  stop("Independent source-only imputeMFA completions are not identical")
}

source_fill_values <- source_fill_values(
  source_input, seq_len(nrow(source_input))
)
combined_completed <- apply_source_fill_values(
  model_input, source_fill_values
)
for (name in names(combined_completed)) {
  if (is.factor(combined_completed[[name]])) {
    replacement <- as.character(source_primary_fit$completed[[name]])
    if (any(!(replacement %in% levels(combined_completed[[name]])))) {
      stop(sprintf("Source imputeMFA completion has an invalid level in %s", name))
    }
    combined_completed[[name]][source_index] <- replacement
  } else {
    combined_completed[[name]][source_index] <-
      source_primary_fit$completed[[name]]
  }
}
phase_projection <- harmonize_supplementary_phase(
  combined_completed, source_index, supplementary_index
)
combined_completed <- phase_projection$frame
if (anyNA(combined_completed) ||
    !identical(rownames(combined_completed), rownames(model_input))) {
  stop("Hybrid source-only/projection completion is incomplete or misaligned")
}
set.seed(20260901L)
combined_mfa <- FactoMineR::MFA(
  base = combined_completed,
  ncp = 5L,
  group = group_sizes,
  type = group_types,
  graph = FALSE,
  num.group.sup = c(1L, 2L, 3L, 4L),
  ind.sup = supplementary_index
)
combined_fit <- list(
  imputation = list(
    method = "source_imputeMFA_projection_source_median_mode",
    source_imputeMFA = source_primary_fit$imputation,
    source_fill_values = source_fill_values,
    supplementary_phase_projection = phase_projection[c(
      "source_levels", "source_reference",
      "observed_supplementary_only_levels", "method"
    )],
    target_rows_used_to_learn_fills = 0L,
    completeObs = combined_completed
  ),
  completed = combined_completed,
  mfa = combined_mfa
)
complete_target_index <- target_index[
  stats::complete.cases(input[target_index, active_columns, drop = FALSE])
]
if (!length(complete_target_index)) {
  stop("No target row has complete active predictor information")
}
complete_target_unchanged <- all(vapply(active_columns, function(name) {
  original <- input[[name]][complete_target_index]
  completed <- combined_completed[[name]][complete_target_index]
  if (is.factor(original)) {
    identical(as.character(original), as.character(completed))
  } else {
    identical(as.numeric(original), as.numeric(completed))
  }
}, logical(1)))
if (!complete_target_unchanged) {
  stop("Source-derived projection fills changed complete target information")
}

source_coordinates <- as.data.frame(combined_fit$mfa$ind$coord)
supplementary_coordinates <- as.data.frame(combined_fit$mfa$ind.sup$coord)
source_only_coordinates <- as.data.frame(source_only_fit$mfa$ind$coord)
if (!identical(rownames(source_coordinates), rownames(input)[source_index]) ||
    !identical(rownames(supplementary_coordinates), rownames(input)[supplementary_index]) ||
    !identical(rownames(source_only_coordinates), rownames(input)[source_index]) ||
    ncol(source_coordinates) != 5L || ncol(supplementary_coordinates) != 5L ||
    ncol(source_only_coordinates) != 5L ||
    any(!is.finite(as.matrix(source_coordinates))) ||
    any(!is.finite(as.matrix(supplementary_coordinates))) ||
    any(!is.finite(as.matrix(source_only_coordinates)))) {
  stop("MFA did not return aligned finite source and supplementary coordinates")
}

numeric_column <- function(value, name) {
  completed_numeric(value, name)
}
max_imputation_delta <- 0
for (name in active_columns) {
  combined_value <- numeric_column(combined_fit$completed[[name]][source_index], name)
  source_value <- numeric_column(source_only_fit$completed[[name]], name)
  spread <- stats::sd(source_value)
  if (!is.finite(spread) || spread <= 0) stop(sprintf("Source variable %s has zero variance", name))
  max_imputation_delta <- max(
    max_imputation_delta,
    max(abs((combined_value - source_value) / spread))
  )
}

procrustes_correlation <- function(reference, candidate) {
  x <- scale(as.matrix(reference), center = TRUE, scale = FALSE)
  y <- scale(as.matrix(candidate), center = TRUE, scale = FALSE)
  rotation <- svd(crossprod(y, x))
  aligned <- y %*% rotation$u %*% t(rotation$v)
  stats::cor(as.vector(x), as.vector(aligned))
}
coordinate_correlation <- procrustes_correlation(
  source_coordinates, source_only_coordinates
)
if (!is.finite(max_imputation_delta) || !is.finite(coordinate_correlation)) {
  stop("Supplementary-row influence diagnostics are not finite")
}
if (max_imputation_delta > 1e-7 || coordinate_correlation < 0.999999) {
  stop("Supplementary-row influence exceeds the imputation/MFA tolerance")
}

write_keyed(source_coordinates,file.path(output_dir,"source_coordinates.csv"))
write_keyed(supplementary_coordinates,file.path(output_dir,"target_coordinates.csv"))
write_keyed(combined_fit$completed,file.path(output_dir,"completed_predictors.csv"))
target_landmarks <- attr(input,"target_landmarks")
if (!is.null(target_landmarks)) {
  if (!is.data.frame(target_landmarks) ||
      !identical(names(target_landmarks),c("key","landmark_wave")) ||
      anyNA(target_landmarks) || anyDuplicated(target_landmarks$key)) {
    stop("Invalid target landmark metadata")
  }
  landmark_keys <- paste(target_landmarks$key,target_landmarks$landmark_wave,sep="-")
  target_landmarks <- target_landmarks[landmark_keys %in% rownames(input)[target_index],,drop=FALSE]
  landmark_keys <- paste(target_landmarks$key,target_landmarks$landmark_wave,sep="-")
  matched <- match(landmark_keys,rownames(combined_fit$completed))
  if (!nrow(target_landmarks) || anyNA(matched)) stop("No valid target landmarks")
  target_completed <- combined_fit$completed[matched,,drop=FALSE]
  target_geometry <- data.frame(
    key=target_landmarks$key,
    landmark_wave=target_landmarks$landmark_wave,
    age_years=completed_numeric(target_completed$newage,"newage"),
    female=as.integer(completed_numeric(target_completed$ragender,"ragender")==2),
    education_level=as.integer(completed_numeric(target_completed$raeducl,"raeducl"))
  )
  if (anyNA(target_geometry) || any(target_geometry$age_years<50 | target_geometry$age_years>120) ||
      any(!(target_geometry$education_level %in% 1:3))) stop("Invalid landmark covariates")
  write.csv(target_geometry,file.path(output_dir,"target_geometry.csv"),row.names=FALSE)
}
saveRDS(combined_fit$mfa,file.path(output_dir,"mfa.rds"),compress="xz")
saveRDS(combined_fit$imputation,file.path(output_dir,"imputation.rds"),compress="xz")
writeLines(capture.output(sessionInfo()),file.path(output_dir,"sessionInfo.txt"))
