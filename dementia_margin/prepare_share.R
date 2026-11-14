#!/usr/bin/env Rscript
# Prepare the SHARE factor-analysis matrix from Harmonized SHARE Version F.

waves <- c(1L, 2L, 4L, 5L, 6L, 7L)

functional_binary <- c(
  "hlthlma", "walkra", "dressa", "batha", "eata", "beda", "toilta",
  "phonea", "medsa", "moneya", "shopa", "mealsa", "mapa", "housewka",
  "walk100a", "sita", "chaira", "climsa", "clim1a", "lifta", "stoopa",
  "armsa", "pusha", "dimea"
)

functional_counts <- c("adltot_s", "iadltot1_s")
cognitive_flags <- c("cogimp", "cogothp")
cognitive_scores <- c("imrc", "dlrc", "tr20", "verbf")

phenotype_stems <- c(
  functional_binary,
  functional_counts,
  cognitive_flags,
  cognitive_scores
)

output_columns <- c(
  "phase", "ragender", "newage", "raeducl",
  functional_binary,
  functional_counts,
  cognitive_flags,
  cognitive_scores,
  "wtresp", "ipw_weight"
)

country_names <- c(
  `40` = "Austria", `56` = "Belgium", `100` = "Bulgaria",
  `191` = "Croatia", `196` = "Cyprus", `203` = "Czech Republic",
  `208` = "Denmark", `233` = "Estonia", `246` = "Finland",
  `250` = "France", `276` = "Germany", `300` = "Greece",
  `348` = "Hungary", `372` = "Ireland", `376` = "Israel",
  `380` = "Italy", `428` = "Latvia", `440` = "Lithuania",
  `442` = "Luxembourg", `470` = "Malta", `528` = "Netherlands",
  `616` = "Poland", `620` = "Portugal", `642` = "Romania",
  `703` = "Slovakia", `705` = "Slovenia", `724` = "Spain",
  `752` = "Sweden", `756` = "Switzerland"
)

required_source_columns <- function() {
  static <- c(
    "mergeid", "ragender", "raeducl", "rabyear", "rabmonth",
    "isocountry"
  )
  time_varying <- c("iwm", "iwy", "wtresp", phenotype_stems)
  c(static, unlist(lapply(waves, function(wave) {
    paste0("r", wave, time_varying)
  }), use.names = FALSE))
}

plain_numeric <- function(value) {
  # haven tagged missing values remain NA after coercion.
  as.numeric(value)
}

month_date <- function(month, year) {
  month <- plain_numeric(month)
  year <- plain_numeric(year)
  text <- ifelse(
    is.na(month) | is.na(year),
    NA_character_,
    sprintf("%04d-%02d-01", as.integer(year), as.integer(month))
  )
  as.Date(text)
}

wave_frame <- function(wide, wave) {
  prefix <- paste0("r", wave)
  get_wave <- function(stem) plain_numeric(wide[[paste0(prefix, stem)]])

  frame <- data.frame(
    mergeid = as.character(wide$mergeid),
    phase = wave,
    iwm = get_wave("iwm"),
    iwy = get_wave("iwy"),
    wtresp = get_wave("wtresp"),
    ragender = plain_numeric(wide$ragender),
    raeducl = plain_numeric(wide$raeducl),
    rabyear = plain_numeric(wide$rabyear),
    rabmonth = plain_numeric(wide$rabmonth),
    isocountry = plain_numeric(wide$isocountry),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  for (stem in phenotype_stems) {
    frame[[stem]] <- get_wave(stem)
  }

  frame <- frame[!is.na(frame$iwy), , drop = FALSE]
  frame$country_name <- unname(country_names[as.character(frame$isocountry)])
  frame$newage <- round(
    as.numeric(
      month_date(frame$iwm, frame$iwy) -
        month_date(frame$rabmonth, frame$rabyear)
    ) / 365.25,
    2
  )
  frame
}

stack_interviews <- function(wide) {
  stacked <- do.call(rbind, lapply(waves, function(wave) wave_frame(wide, wave)))
  rownames(stacked) <- NULL
  stacked
}

eligible_ids <- function(long) {
  phase_path <- tapply(long$phase, long$mergeid, paste, collapse = "-")
  allowed <- c("1-2", "1-2-4", "1-2-4-5", "1-2-4-5-6", "1-2-4-5-6-7")
  consecutive <- names(phase_path)[phase_path %in% allowed]

  baseline <- long[long$phase == 1L, c("mergeid", "newage"), drop = FALSE]
  baseline_age <- setNames(baseline$newage, baseline$mergeid)
  consecutive[!is.na(baseline_age[consecutive]) & baseline_age[consecutive] >= 50]
}

stabilized_binary_weights <- function(frame, participation) {
  if (anyNA(frame$newage) || anyNA(frame$ragender) || anyNA(frame$country_name)) {
    stop("Missing age, sex or country in an attrition model", call. = FALSE)
  }
  if (length(unique(participation)) != 2L) {
    stop("Each attrition model requires participants and non-participants", call. = FALSE)
  }

  model_data <- data.frame(
    participation = as.integer(participation),
    newage = frame$newage,
    ragender = factor(frame$ragender),
    country_name = frame$country_name,
    stringsAsFactors = FALSE
  )
  numerator <- stats::glm(
    participation ~ 1,
    family = stats::binomial(link = "logit"),
    data = model_data,
    na.action = stats::na.fail
  )
  denominator <- stats::glm(
    participation ~ newage + ragender + country_name,
    family = stats::binomial(link = "logit"),
    data = model_data,
    na.action = stats::na.fail
  )
  numerator_probability <- stats::predict(numerator, type = "response")
  denominator_probability <- stats::predict(denominator, type = "response")
  observed_numerator <- ifelse(
    participation == 1L,
    numerator_probability,
    1 - numerator_probability
  )
  observed_denominator <- ifelse(
    participation == 1L,
    denominator_probability,
    1 - denominator_probability
  )
  observed_numerator / observed_denominator
}

attrition_weight_table <- function(long, selected_ids) {
  baseline <- long[
    long$phase == 1L & !is.na(long$newage) & long$newage >= 50,
    ,
    drop = FALSE
  ]
  targets <- c(2L, 4L, 5L, 6L, 7L)
  origins <- c(1L, 2L, 4L, 5L, 6L)
  tables <- vector("list", length(targets))

  for (index in seq_along(targets)) {
    origin <- origins[index]
    target <- targets[index]
    risk_set <- if (origin == 1L) {
      baseline
    } else {
      long[
        long$phase == origin & long$mergeid %in% selected_ids,
        ,
        drop = FALSE
      ]
    }
    target_ids <- long$mergeid[
      long$phase == target & long$mergeid %in% selected_ids
    ]
    participation <- risk_set$mergeid %in% target_ids
    interval_weight <- stabilized_binary_weights(risk_set, participation)
    retained <- participation
    tables[[index]] <- data.frame(
      mergeid = risk_set$mergeid[retained],
      phase = target,
      ipw_weight = interval_weight[retained],
      stringsAsFactors = FALSE
    )
  }

  # The baseline and Wave 2 rows share the first transition weight.
  wave_two <- tables[[1L]]
  wave_one <- wave_two
  wave_one$phase <- 1L
  all_weights <- do.call(rbind, c(list(wave_one), tables))
  rownames(all_weights) <- NULL
  all_weights
}

prepare_share_predictors <- function(wide, enforce_reference_counts = TRUE) {
  absent <- setdiff(required_source_columns(), names(wide))
  if (length(absent) > 0L) {
    stop(
      sprintf("Harmonized SHARE is missing: %s", paste(absent, collapse = ", ")),
      call. = FALSE
    )
  }

  if (anyNA(wide$mergeid) || anyDuplicated(wide$mergeid) ||
      any(!nzchar(as.character(wide$mergeid)))) {
    stop("Harmonized SHARE requires one nonempty unique mergeid per participant", call. = FALSE)
  }

  long <- stack_interviews(wide)
  selected_ids <- eligible_ids(long)
  selected <- long[long$mergeid %in% selected_ids, , drop = FALSE]
  weights <- attrition_weight_table(long, selected_ids)
  selected_key <- paste(selected$mergeid, selected$phase, sep = "\r")
  weight_key <- paste(weights$mergeid, weights$phase, sep = "\r")
  selected$ipw_weight <- weights$ipw_weight[match(selected_key, weight_key)]

  if (anyNA(selected$ipw_weight)) {
    stop("Attrition weights did not link to every selected row", call. = FALSE)
  }

  selected$phase <- factor(selected$phase, levels = waves)
  selected$ragender <- factor(selected$ragender, levels = c(1, 2))
  selected$raeducl <- factor(selected$raeducl, levels = c(1, 2, 3))
  for (column in c(functional_binary, cognitive_flags)) {
    selected[[column]] <- factor(selected[[column]], levels = c(0, 1))
  }

  result <- selected[, output_columns, drop = FALSE]
  rownames(result) <- paste(selected$mergeid, selected$phase, sep = "-")

  if (anyDuplicated(rownames(result))) {
    stop("Person-wave keys are not unique", call. = FALSE)
  }
  if (enforce_reference_counts) {
    expected_phase_counts <- c(`1` = 15278L, `2` = 15278L, `4` = 10008L,
                               `5` = 8418L, `6` = 6485L, `7` = 5533L)
    observed <- table(result$phase)
    if (!identical(as.integer(observed), as.integer(expected_phase_counts))) {
      stop(
        paste0(
          "Unexpected phase counts: ",
          paste(names(observed), as.integer(observed), sep = "=", collapse = ", ")
        ),
        call. = FALSE
      )
    }
    if (!identical(dim(result), c(61000L, 38L))) {
      stop("Reference reconstruction must have dimension 61,000 x 38", call. = FALSE)
    }
  }
  result
}

read_harmonized_share <- function(path) {
  if (!requireNamespace("haven", quietly = TRUE)) {
    stop("The R package 'haven' is required", call. = FALSE)
  }
  haven::read_dta(
    path,
    col_select = tidyselect::all_of(required_source_columns())
  )
}


main <- function(args = commandArgs(trailingOnly = TRUE)) {
  script_path <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)), mustWork=TRUE)
  source(file.path(dirname(script_path), "mfa_helpers.R"))
  paths <- analysis_paths(args, script_path)
  result <- prepare_share_predictors(read_harmonized_share(paths$input))
  saveRDS(result, file.path(paths$output, "prepared_predictors.rds"), compress="xz")
  writeLines(capture.output(sessionInfo()), file.path(paths$output, "sessionInfo.txt"))
  message(sprintf("Prepared %s person-wave rows and %s predictor columns", nrow(result), ncol(result)))
}

if (sys.nframe() == 0L) main()
