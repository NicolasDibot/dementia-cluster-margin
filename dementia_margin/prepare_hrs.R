#!/usr/bin/env Rscript
# Prepare HRS source and supplementary person-waves without reading diagnoses.
suppressPackageStartupMessages(library(haven))
script_path <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)), mustWork=TRUE)
source(file.path(dirname(script_path), "mfa_helpers.R"))
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=5L) stop("Usage: Rscript prepare_hrs.R <adams|hcap> <rand.dta> <harmonized.dta> <selection.csv> <new-output-directory>")
mode <- args[1L]
if (!(mode %in% c("adams","hcap"))) stop("Mode must be adams or hcap")
paths <- analysis_paths(args[c(2L,5L)],script_path)
input_path <- paths$input
output_dir <- paths$output
repo <- normalizePath(file.path(dirname(script_path),".."),winslash="/",mustWork=TRUE)
external_input <- function(path) {
  path <- normalizePath(path,winslash="/",mustWork=TRUE)
  if (identical(path,repo) || startsWith(path,paste0(repo,"/"))) {
    stop("Cohort inputs must remain outside the repository")
  }
  path
}
harmonized_path <- external_input(args[3L])
selection_path <- external_input(args[4L])
source_waves <- if (mode=="hcap") 8:10 else 3:5
# HCAP participants are also kept at Waves 11-13 (2012-2016) for the analysis
# by interval between the HRS interview and the 2016 assessment.
later_landmarks <- if (mode=="hcap") 11:13 else integer(0)
target_waves <- if (mode=="hcap") c(8:10,later_landmarks) else 3:6
read_waves <- sort(unique(c(source_waves,target_waves)))

functional_binary <- c(
  "hlthlma", "walkra", "dressa", "batha", "eata", "beda", "toilta",
  "phonea", "medsa", "moneya", "shopa", "mealsa", "mapa", "housewka", "walk100a",
  "sita", "chaira", "climsa", "clim1a", "lifta", "stoopa", "armsa",
  "pusha", "dimea"
)
source_stems <- c(
  hlthlma = "hlthlm", walkra = "walkra", dressa = "dressa",
  batha = "batha", eata = "eata", beda = "beda", toilta = "toilta",
  phonea = "phonea", medsa = "medsa", moneya = "moneya",
  shopa = "shopa", mealsa = "mealsa", mapa = "mapa",
  walk100a = "walk1a", sita = "sita", chaira = "chaira",
  climsa = "climsa", clim1a = "clim1a", lifta = "lifta",
  stoopa = "stoopa", armsa = "armsa", pusha = "pusha", dimea = "dimea"
)
cognitive_scores <- c("imrc", "dlrc", "tr20")
output_columns <- c(
  "phase", "ragender", "newage", "raeducl", functional_binary,
  "adltot_s", "iadltot1_s", "cogothp", cognitive_scores
)

canonical_key <- function(value) {
  numeric <- suppressWarnings(as.numeric(value))
  if (anyNA(numeric) || any(!is.finite(numeric)) ||
      any(numeric <= 0) || any(numeric != floor(numeric))) {
    stop("HRS keys must be finite positive integers")
  }
  format(numeric, scientific = FALSE, trim = TRUE, digits = 15)
}

selection <- read.csv(selection_path,stringsAsFactors=FALSE,colClasses=c(key="character"))
expected_selection <- c("key","landmark_wave","target_source")
if (!identical(names(selection),expected_selection) || anyNA(selection) ||
    anyDuplicated(selection$key)) stop("Invalid unique-key selection schema")
selection$key <- canonical_key(selection$key)
selection$landmark_wave <- as.integer(selection$landmark_wave)
if (anyDuplicated(selection$key)) stop("Selection keys must be unique after normalization")
valid_selection <- if (mode=="hcap") {
  selection$landmark_wave==10L & selection$target_source=="hcap"
} else {
  selection$landmark_wave %in% c(5L,6L) & selection$target_source=="adams"
}
if (!all(valid_selection)) stop("Invalid selection timing or source")
if (nrow(selection)!=(if (mode=="hcap") 3496L else 1770L)) {
  stop("Selection row count differs from the official release")
}

static_columns <- c("hhidpn", "ragender", "raeduc")
wave_stems <- c(
  "iwstat", "agey_m", unname(source_stems), "proxy", cognitive_scores
)
rand_columns <- c(
  static_columns,
  unlist(lapply(read_waves, function(wave) paste0("r", wave, wave_stems)), use.names = FALSE)
)
forbidden <- grep(
  "adams|hcap|dement|diagnos|adfdx|langa|predicted",
  rand_columns, ignore.case = TRUE, value = TRUE
)
if (length(forbidden)) stop("Outcome-like fields are forbidden from the predictor stage")

wide <- haven::read_dta(
  input_path,
  col_select = tidyselect::all_of(rand_columns)
)
if (anyDuplicated(names(wide)) ||
    !setequal(names(wide), rand_columns) ||
    length(names(wide)) != length(rand_columns)) {
  stop("RAND HRS did not return the exact predictor whitelist")
}
wide <- wide[, rand_columns]
if (!identical(names(wide), rand_columns)) {
  stop("RAND HRS predictor whitelist could not be put in order")
}
wide$key <- canonical_key(wide$hhidpn)
if (anyDuplicated(wide$key)) stop("RAND HRS hhidpn must be unique")

harmonized_stems <- c("limhwrk", "adltot6", "iadltot_h")
harmonized_columns <- c(
  "hhidpn",
  unlist(lapply(read_waves, function(wave) paste0("r", wave, harmonized_stems)),
         use.names = FALSE)
)
harmonized <- haven::read_dta(
  harmonized_path,
  col_select = tidyselect::all_of(harmonized_columns)
)
if (anyDuplicated(names(harmonized)) ||
    !setequal(names(harmonized), harmonized_columns) ||
    length(names(harmonized)) != length(harmonized_columns)) {
  stop("Harmonized HRS did not return the exact derived-variable whitelist")
}
harmonized <- harmonized[, harmonized_columns]
if (!identical(names(harmonized), harmonized_columns)) {
  stop("Harmonized HRS derived-variable whitelist could not be put in order")
}
harmonized$key <- canonical_key(harmonized$hhidpn)
if (anyDuplicated(harmonized$key)) stop("Harmonized HRS keys must be unique")
harmonized_match <- match(wide$key, harmonized$key)
wide$harmonized_linked <- !is.na(harmonized_match)
for (column in setdiff(harmonized_columns, "hhidpn")) {
  wide[[column]] <- harmonized[[column]][harmonized_match]
}
target_keys <- selection$key
landmark_by_key <- setNames(selection$landmark_wave, selection$key)
plain_numeric <- function(value) suppressWarnings(as.numeric(value))
wave_frame <- function(wave) {
  prefix <- paste0("r", wave)
  get_wave <- function(stem) plain_numeric(wide[[paste0(prefix, stem)]])
  frame <- data.frame(
    key = wide$key,
    phase = wave,
    ragender = plain_numeric(wide$ragender),
    newage = get_wave("agey_m"),
    raeduc = plain_numeric(wide$raeduc),
    iwstat = get_wave("iwstat"),
    harmonized_linked = wide$harmonized_linked,
    cogothp = get_wave("proxy"),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  for (analysis_name in functional_binary) {
    frame[[analysis_name]] <- if (analysis_name == "housewka") {
      plain_numeric(wide[[paste0(prefix, "limhwrk")]])
    } else {
      get_wave(source_stems[[analysis_name]])
    }
  }
  for (score in cognitive_scores) frame[[score]] <- get_wave(score)
  frame$raeducl <- ifelse(
    frame$raeduc == 1, 1,
    ifelse(frame$raeduc %in% c(2, 3), 2,
           ifelse(frame$raeduc %in% c(4, 5), 3, NA))
  )
  frame$adltot_s <- plain_numeric(wide[[paste0(prefix, "adltot6")]])
  frame$iadltot1_s <- plain_numeric(wide[[paste0(prefix, "iadltot_h")]])
  if (anyDuplicated(names(frame))) {
    stop("Wave-frame predictor names must be unique")
  }
  frame
}

long <- do.call(rbind, lapply(read_waves, wave_frame))
rownames(long) <- NULL
validate_integer_support <- function(name, lower, upper) {
  observed <- long[[name]][!is.na(long[[name]])]
  if (any(!is.finite(observed)) || any(observed != floor(observed)) ||
      any(observed < lower | observed > upper)) {
    stop(sprintf("Observed %s values fall outside %d-%d support",
                 name, lower, upper))
  }
}
for (name in c("imrc", "dlrc")) validate_integer_support(name, 0L, 10L)
validate_integer_support("tr20", 0L, 20L)
for (name in c("adltot_s", "iadltot1_s")) {
  validate_integer_support(name, 0L, 6L)
}
complete_recall <- stats::complete.cases(long[, c("imrc", "dlrc", "tr20")])
recall_consistency_by_wave <- do.call(rbind, lapply(read_waves, function(wave) {
  selected <- complete_recall & long$phase == wave
  mismatch <- long$tr20[selected] !=
    long$imrc[selected] + long$dlrc[selected]
  data.frame(
    wave = wave,
    complete_recall_n = sum(selected),
    inconsistent_recall_n = sum(mismatch),
    inconsistent_fraction = if (sum(selected)) mean(mismatch) else NA_real_,
    stringsAsFactors = FALSE
  )
}))
if (any(recall_consistency_by_wave$inconsistent_fraction > 0.005,
        na.rm = TRUE)) {
  stop("Observed recall-component inconsistency exceeds the 0.5% gate")
}
interviewed <- !is.na(long$iwstat) & long$iwstat == 1 &
  !is.na(long$newage) & long$newage >= 50

active_columns <- c(
  functional_binary, "adltot_s", "iadltot1_s", cognitive_scores
)
row_feature_gate <- function(frame) {
  function_observed <- rowSums(!is.na(frame[, functional_binary, drop = FALSE])) >= 1
  total_observed <- rowSums(!is.na(frame[, c("adltot_s", "iadltot1_s"), drop = FALSE])) >= 1
  context_observed <- !is.na(frame$cogothp)
  cognition_observed <- rowSums(!is.na(frame[, cognitive_scores, drop = FALSE])) >= 1
  cognition_or_proxy <- cognition_observed |
    suppressWarnings(as.numeric(as.character(frame$cogothp))) == 1
  missing_fraction <- rowMeans(is.na(frame[, active_columns, drop = FALSE]))
  function_observed & total_observed & context_observed & cognition_or_proxy &
    missing_fraction <= 0.30
}
source_candidate <- long[
  !(long$key %in% target_keys) & long$phase %in% source_waves & interviewed,
  , drop = FALSE
]
first_keys <- unique(source_candidate$key[source_candidate$phase == source_waves[[1L]]])
second_keys <- unique(source_candidate$key[source_candidate$phase == source_waves[[2L]]])
source_keys <- intersect(first_keys, second_keys)
source <- source_candidate[source_candidate$key %in% source_keys, , drop = FALSE]
if (!all(source$harmonized_linked)) {
  stop("Every selected source participant must link one-to-one to Harmonized HRS")
}
source_counts <- table(source$key)
if (any(!(source_counts %in% c(2L, 3L)))) {
  stop("Source paths must contain the first two waves and optionally the third")
}
source_missingness <- colMeans(is.na(source[, active_columns, drop = FALSE]))
if (any(source_missingness >= 0.30)) {
  stop(paste(
    "A source item has at least 30% missingness:",
    paste(names(source_missingness)[source_missingness >= 0.30], collapse = ", ")
  ))
}

# ADAMS uses each participant's WAVESEL landmark (Wave 5 or 6).
# HCAP uses Waves 8-10 with Wave 10 as the primary landmark; participants
# observed at a later landmark (Waves 11-13) are retained as targets as well.
target_raw <- long[
  long$key %in% target_keys & long$phase %in% target_waves & interviewed,
  ,drop=FALSE
]
target_raw$landmark_wave <- unname(landmark_by_key[target_raw$key])
if (mode=="adams") {
  target_raw <- target_raw[target_raw$phase<=target_raw$landmark_wave,,drop=FALSE]
}
target_raw$feature_gate <- row_feature_gate(target_raw)
eligible_target_keys <- unique(target_raw$key[
  (target_raw$phase==target_raw$landmark_wave |
     target_raw$phase %in% later_landmarks) & target_raw$feature_gate
])
target <- target_raw[
  target_raw$key %in% eligible_target_keys & target_raw$feature_gate,,drop=FALSE
]
if (any(target$key %in% source$key)) stop("Source and target participants overlap")
analysis_row_columns <- names(long)
combined <- rbind(source[,analysis_row_columns,drop=FALSE],
                  target[,analysis_row_columns,drop=FALSE])
row_role <- c(rep("source",nrow(source)),rep("target",nrow(target)))
if (nrow(source)<1000L || nrow(target)<100L ||
    anyDuplicated(paste(combined$key,combined$phase))) {
  stop("Prepared source/target frame is too small or contains duplicate rows")
}

primary_phase_levels <- sort(unique(c(source_waves, target_waves)))
combined$phase <- factor(combined$phase, levels = primary_phase_levels)
combined$ragender <- factor(combined$ragender, levels = c(1, 2))
combined$raeducl <- factor(combined$raeducl, levels = c(1, 2, 3))
for (column in c(functional_binary, "cogothp")) {
  invalid <- !is.na(combined[[column]]) & !(combined[[column]] %in% c(0, 1))
  if (any(invalid)) stop(sprintf("%s contains values outside the 0/1 support", column))
  combined[[column]] <- factor(combined[[column]], levels = c(0, 1))
}

result <- combined[, output_columns, drop = FALSE]
rownames(result) <- paste(combined$key, combined$phase, sep = "-")
if (anyDuplicated(rownames(result)) || any(vapply(result, function(x) all(is.na(x)), logical(1)))) {
  stop("Prepared MFA input has duplicate keys or an all-missing column")
}
attr(result, "row_role") <- row_role
attr(result, "analysis_mode") <- mode
attr(result, "source_waves") <- source_waves
attr(result, "target_primary_wave") <- if (mode == "hcap") 10L else NA_integer_
attr(result,"target_landmarks") <- selection[,c("key","landmark_wave"),drop=FALSE]
attr(result,"target_later_landmarks") <- later_landmarks
attr(result,"source_item_missingness") <- source_missingness
attr(result,"recall_consistency_by_wave") <- recall_consistency_by_wave
saveRDS(result,file.path(output_dir,"prepared_predictors.rds"),compress="xz")
writeLines(capture.output(sessionInfo()),file.path(output_dir,"sessionInfo.txt"))
message(sprintf("Prepared %s: %s source rows and %s target rows",
                mode,nrow(source),nrow(target)))
