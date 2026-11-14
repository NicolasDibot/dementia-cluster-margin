# Completion and projection helpers for the HRS factor model.

completed_numeric <- function(value, name) {
  if (!is.character(name) || length(name) != 1L || !nzchar(name)) {
    stop("Completed-variable name must be one nonempty string")
  }
  if (is.factor(value) || is.character(value)) {
    labels <- as.character(value)
    output <- suppressWarnings(as.numeric(labels))
    unresolved <- is.na(output) & !is.na(labels)
    if (any(unresolved)) {
      prefix <- paste0(name, "_")
      if (any(!startsWith(labels[unresolved], prefix))) {
        stop(sprintf("Completed variable %s has an unexpected factor label", name))
      }
      output[unresolved] <- suppressWarnings(as.numeric(
        substring(labels[unresolved], nchar(prefix) + 1L)
      ))
    }
  } else {
    output <- suppressWarnings(as.numeric(value))
  }
  if (anyNA(output) || any(!is.finite(output))) {
    stop(sprintf("Completed variable %s cannot be represented as finite numeric", name))
  }
  output
}

source_fill_values <- function(frame, source_index) {
  if (!is.data.frame(frame) || !length(source_index) ||
      anyNA(source_index) || anyDuplicated(source_index) ||
      any(source_index < 1L | source_index > nrow(frame))) {
    stop("Source-only imputation requires valid unique source rows")
  }
  fills <- setNames(vector("list", ncol(frame)), names(frame))
  for (name in names(frame)) {
    value <- frame[[name]]
    observed <- value[source_index]
    observed <- observed[!is.na(observed)]
    if (!length(observed)) {
      stop(sprintf("Source-only imputation has no observed source value for %s", name))
    }
    if (is.factor(value)) {
      declared_levels <- levels(value)
      counts <- tabulate(
        match(as.character(observed), declared_levels),
        nbins = length(declared_levels)
      )
      maximum <- max(counts)
      tied <- which(counts == maximum)
      selected <- tied[[1L]]
      fills[[name]] <- list(
        kind = "factor_mode",
        value = declared_levels[[selected]],
        tied_mode_count = length(tied),
        tie_break = "first_declared_factor_level",
        source_observed_n = length(observed)
      )
    } else if (is.numeric(value) || is.integer(value)) {
      fills[[name]] <- list(
        kind = "numeric_median",
        value = as.numeric(stats::median(as.numeric(observed))),
        tied_mode_count = NA_integer_,
        tie_break = NA_character_,
        source_observed_n = length(observed)
      )
    } else {
      stop(sprintf("Source-only imputation does not support class of %s", name))
    }
  }
  fills
}

apply_source_fill_values <- function(frame, fill_values) {
  if (!is.data.frame(frame) || !is.list(fill_values) ||
      !identical(names(fill_values), names(frame))) {
    stop("Source-only imputation fill values do not match the frame schema")
  }
  completed <- frame
  for (name in names(completed)) {
    missing <- is.na(completed[[name]])
    fill <- fill_values[[name]]
    if (!is.list(fill) || !is.character(fill$kind) ||
        length(fill$kind) != 1L || length(fill$value) != 1L) {
      stop(sprintf("Source-only imputation fill metadata is invalid for %s", name))
    }
    if (is.factor(completed[[name]])) {
      if (!identical(fill$kind, "factor_mode") ||
          !(fill$value %in% levels(completed[[name]]))) {
        stop(sprintf("Source-only factor fill is invalid for %s", name))
      }
      completed[[name]][missing] <- fill$value
    } else if (is.numeric(completed[[name]]) || is.integer(completed[[name]])) {
      numeric_fill <- suppressWarnings(as.numeric(fill$value))
      if (!identical(fill$kind, "numeric_median") ||
          !is.finite(numeric_fill)) {
        stop(sprintf("Source-only numeric fill is invalid for %s", name))
      }
      completed[[name]][missing] <- numeric_fill
    } else {
      stop(sprintf("Source-only imputation does not support class of %s", name))
    }
  }
  if (anyNA(completed) || !identical(rownames(completed), rownames(frame))) {
    stop("Source-only imputation is incomplete or changed row order")
  }
  completed
}

harmonize_supplementary_phase <- function(
    frame, source_index, supplementary_index, phase_name = "phase") {
  if (!identical(phase_name, "phase")) {
    stop("Only the supplementary phase group may be harmonized")
  }
  if (!is.data.frame(frame) || !(phase_name %in% names(frame)) ||
      !is.factor(frame[[phase_name]]) || !length(source_index) ||
      anyNA(source_index) || anyDuplicated(source_index) ||
      any(source_index < 1L | source_index > nrow(frame)) ||
      anyNA(supplementary_index) || anyDuplicated(supplementary_index) ||
      any(supplementary_index < 1L | supplementary_index > nrow(frame)) ||
      length(intersect(source_index, supplementary_index)) ||
      !setequal(c(source_index, supplementary_index), seq_len(nrow(frame)))) {
    stop("Supplementary phase harmonization requires a valid partition")
  }
  phase <- frame[[phase_name]]
  source_value <- as.character(phase[source_index])
  supplementary_value <- as.character(phase[supplementary_index])
  if (anyNA(source_value) || anyNA(supplementary_value)) {
    stop("Supplementary phase harmonization requires completed phase values")
  }
  source_levels <- levels(phase)[levels(phase) %in% unique(source_value)]
  if (!length(source_levels) || any(!(source_value %in% source_levels))) {
    stop("Source phase modalities are absent or invalid")
  }
  counts <- tabulate(match(source_value, source_levels), nbins = length(source_levels))
  source_reference <- source_levels[[which.max(counts)]]
  unseen <- !(supplementary_value %in% source_levels)
  # Build a model-only factor with exactly the modalities estimable in active
  # source rows. Unsupported supplementary modalities first become NA, then
  # receive the source-only modal reference. This avoids FactoMineR creating a
  # zero-sum MCA indicator while leaving the original phase object untouched.
  model_phase <- factor(as.character(phase), levels = source_levels)
  model_phase[supplementary_index[unseen]] <- source_reference
  frame[[phase_name]] <- model_phase
  if (anyNA(frame[[phase_name]]) ||
      !identical(as.character(frame[[phase_name]][source_index]), source_value)) {
    stop("Supplementary phase harmonization changed source phase values")
  }
  list(
    frame = frame,
    source_levels = source_levels,
    source_reference = source_reference,
    observed_supplementary_only_levels = levels(phase)[
      levels(phase) %in% unique(supplementary_value[unseen])
    ],
    remapped_index = supplementary_index[unseen],
    method = paste(
      "source_modal_reference_for_unseen_supplementary_phase_only",
      "original_phase_retained_outside_MFA", sep = ";"
    )
  )
}

normalize_imputemfa_completion <- function(completed, template) {
  completed <- as.data.frame(completed)
  if (!is.data.frame(template) ||
      !identical(names(completed), names(template)) ||
      !identical(rownames(completed), rownames(template)) ||
      anyNA(completed)) {
    stop("Source imputeMFA completion is incomplete or misaligned")
  }
  normalized <- template
  for (name in names(template)) {
    if (is.factor(template[[name]])) {
      declared_levels <- levels(template[[name]])
      level_numeric <- suppressWarnings(as.numeric(declared_levels))
      values <- completed_numeric(completed[[name]], name)
      if (anyNA(level_numeric) || any(!is.finite(level_numeric)) ||
          anyDuplicated(level_numeric) || any(!(values %in% level_numeric))) {
        stop(sprintf("Source imputeMFA factor completion is outside levels for %s", name))
      }
      normalized[[name]] <- factor(
        declared_levels[match(values, level_numeric)],
        levels = declared_levels
      )
    } else if (is.numeric(template[[name]]) || is.integer(template[[name]])) {
      values <- suppressWarnings(as.numeric(completed[[name]]))
      if (anyNA(values) || any(!is.finite(values))) {
        stop(sprintf("Source imputeMFA numeric completion is invalid for %s", name))
      }
      normalized[[name]] <- values
    } else {
      stop(sprintf("Source imputeMFA completion does not support class of %s", name))
    }
  }
  if (anyNA(normalized)) stop("Normalized source imputeMFA completion contains NA")
  normalized
}


analysis_paths <- function(args, script_path) {
  if (length(args) != 2L) stop("Usage: Rscript <script.R> <prepared-input.rds> <new-output-directory>")
  if (!identical(Sys.getenv("COHORT_DATA_ACCESS"), "authorised")) {
    stop("Set COHORT_DATA_ACCESS=authorised only after obtaining cohort permissions")
  }
  repo <- normalizePath(file.path(dirname(script_path), ".."), winslash="/", mustWork=TRUE)
  input <- normalizePath(args[1L], winslash="/", mustWork=TRUE)
  output <- file.path(normalizePath(dirname(args[2L]),winslash="/",mustWork=TRUE),basename(args[2L]))
  for (path in c(input, output)) {
    if (identical(path,repo) || startsWith(path,paste0(repo,"/"))) {
      stop("Cohort inputs and fitted objects must remain outside the repository")
    }
  }
  if (file.exists(output)) stop("Output directory already exists")
  dir.create(output, recursive=FALSE, mode="0700")
  Sys.umask("0077")
  list(input=input, output=output)
}

write_keyed <- function(frame, path) {
  write.csv(data.frame(person_wave_key=rownames(frame), frame,
                       check.names=FALSE, row.names=NULL), path, row.names=FALSE)
}
