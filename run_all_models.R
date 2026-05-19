# ============================================================
# run_all_models.R
# Unified pipeline: shared preprocessing + 5 models (3-class)
# Saves each model's tuning process and final results as .rds
#
# Usage:
#   1. Place survey_results_public.csv in the working directory
#   2. Create a "results" folder: dir.create("results")
#   3. Run this script section by section, or source() the whole file
#   4. Output: results/gbm_3class.rds, results/dnn_3class.rds, etc.
#
# Note: DNN section requires {torch} and {luz} packages.
#       NB section requires {e1071}, {naivebayes}, {themis}, {recipes}.
#       RF/LR section requires {ranger}, {nnet}.
# ============================================================

dir.create("results", showWarnings = FALSE)

# ############################################################
#
#   PART 0: SHARED PREPROCESSING (DO NOT MODIFY)
#
# ############################################################

library(caret)
library(gbm)
library(tidyverse)
library(forcats)
library(ggplot2)

set.seed(123)

# ============================================================
# 0.1 Load data
# ============================================================

data_path <- "survey_results_public.csv"
df <- read_csv(data_path, show_col_types = FALSE)

# ============================================================
# 0.2 Helper functions
# ============================================================

convert_years <- function(x) {
  x <- as.character(x)
  x[x == "Less than 1 year"] <- "0.5"
  x[x == "More than 50 years"] <- "51"
  as.numeric(x)
}

safe_median <- function(x) {
  med <- median(x, na.rm = TRUE)
  ifelse(is.na(med), 0, med)
}

safe_quantile <- function(x, prob = 0.99) {
  if (all(is.na(x))) return(0)
  q <- quantile(x, probs = prob, na.rm = TRUE)
  ifelse(is.na(q), max(x, na.rm = TRUE), q)
}

safe_remove_vars <- function(data, vars) {
  vars <- vars[!is.na(vars) & vars != "" & vars %in% names(data)]
  if (length(vars) > 0) data <- data %>% select(-any_of(vars))
  data
}

is_near_zero_var <- function(x, freq_cut = 95 / 5, unique_cut = 10) {
  x <- as.character(x)
  x <- x[!is.na(x)]
  if (length(x) == 0) return(TRUE)
  tab <- sort(table(x), decreasing = TRUE)
  if (length(tab) == 1) return(TRUE)
  freq_ratio <- as.numeric(tab[1] / tab[2])
  percent_unique <- length(unique(x)) / length(x) * 100
  freq_ratio > freq_cut && percent_unique < unique_cut
}

# ============================================================
# 0.3 Create target variable and select raw predictors
# ============================================================

gbm_raw <- df %>%
  mutate(
    JobSat = as.numeric(JobSat),
    JobSat_Class = case_when(
      JobSat <= 1  ~ "Very_Dissatisfied",
      JobSat <= 3  ~ "Dissatisfied",
      JobSat <= 6  ~ "Neutral",
      JobSat <= 8  ~ "Satisfied",
      JobSat <= 10 ~ "Very_Satisfied",
      TRUE ~ NA_character_
    ),
    JobSat_Class = factor(JobSat_Class,
      levels = c("Very_Dissatisfied", "Dissatisfied", "Neutral",
                 "Satisfied", "Very_Satisfied")),
    YearsCode = convert_years(YearsCode),
    WorkExp = as.numeric(WorkExp),
    ConvertedCompYearly = as.numeric(ConvertedCompYearly)
  ) %>%
  select(any_of(c(
    "JobSat_Class",
    "Age", "EdLevel", "Employment", "RemoteWork",
    "WorkExp", "YearsCode", "OrgSize", "ICorPM",
    "Country", "DevType", "Industry",
    "AISelect", "AISent", "AIAcc", "AIThreat",
    "NewRole", "TechEndorseIntro", "PurchaseInfluence",
    "ToolCountWork", "ToolCountPersonal",
    "ConvertedCompYearly"
  ))) %>%
  filter(!is.na(JobSat_Class))

# ============================================================
# 0.4 Stratified train-test split BEFORE preprocessing
# ============================================================

train_index <- createDataPartition(gbm_raw$JobSat_Class, p = 0.8, list = FALSE)
train_raw <- gbm_raw[train_index, ]
test_raw  <- gbm_raw[-train_index, ]

# ============================================================
# 0.5 Fit preprocessing rules on training data only
# ============================================================

fit_gbm_preprocess <- function(train_data) {
  missing_rates <- sapply(train_data, function(x) mean(is.na(x)))
  high_missing_vars <- names(missing_rates)[
    missing_rates > 0.80 & names(missing_rates) != "JobSat_Class"]
  high_missing_vars <- high_missing_vars[!is.na(high_missing_vars) & high_missing_vars != ""]
  train_data_reduced <- safe_remove_vars(train_data, high_missing_vars)

  numeric_vars <- intersect(
    c("WorkExp", "YearsCode", "ToolCountWork", "ToolCountPersonal"),
    names(train_data_reduced))
  numeric_medians <- list()
  for (v in numeric_vars) numeric_medians[[v]] <- safe_median(train_data_reduced[[v]])

  workexp_cap <- if ("WorkExp" %in% names(train_data_reduced))
    safe_quantile(train_data_reduced$WorkExp, 0.99) else NA
  yearscode_cap <- if ("YearsCode" %in% names(train_data_reduced))
    safe_quantile(train_data_reduced$YearsCode, 0.99) else NA
  salary_cap <- if ("ConvertedCompYearly" %in% names(train_data_reduced))
    safe_quantile(train_data_reduced$ConvertedCompYearly, 0.99) else NA

  if ("ConvertedCompYearly" %in% names(train_data_reduced)) {
    salary_temp <- train_data_reduced$ConvertedCompYearly
    salary_temp[salary_temp < 0] <- NA
    salary_temp <- ifelse(salary_temp > salary_cap, salary_cap, salary_temp)
    log_salary_median <- safe_median(log1p(salary_temp))
  } else {
    log_salary_median <- 0
  }

  multi_response_candidates <- intersect(
    c("DevType", "Employment", "AISelect", "AISent", "AIAcc", "AIThreat",
      "NewRole", "TechEndorseIntro", "PurchaseInfluence"),
    names(train_data_reduced))
  multi_response_vars  <- c()
  multi_response_terms <- list()
  for (v in multi_response_candidates) {
    x <- as.character(train_data_reduced[[v]])
    if (any(str_detect(x[!is.na(x)], fixed(";")))) {
      multi_response_vars <- c(multi_response_vars, v)
      split_values <- trimws(unlist(strsplit(x[!is.na(x)], ";", fixed = TRUE)))
      split_values <- split_values[split_values != ""]
      value_counts <- sort(table(split_values), decreasing = TRUE)
      multi_response_terms[[v]] <- names(value_counts)[1:min(15, length(value_counts))]
    }
  }

  categorical_vars <- setdiff(
    intersect(
      c("Age", "EdLevel", "RemoteWork", "OrgSize", "ICorPM", "Country", "Industry",
        "AISelect", "AISent", "AIAcc", "AIThreat", "Employment", "DevType",
        "NewRole", "TechEndorseIntro", "PurchaseInfluence"),
      names(train_data_reduced)),
    multi_response_vars)
  top_levels <- list()
  for (v in categorical_vars) {
    x <- as.character(train_data_reduced[[v]])
    x[is.na(x)] <- "Missing"
    level_counts <- sort(table(x), decreasing = TRUE)
    top_levels[[v]] <- names(level_counts)[1:min(15, length(level_counts))]
  }

  list(
    high_missing_vars = high_missing_vars,
    numeric_vars = numeric_vars, numeric_medians = numeric_medians,
    workexp_cap = workexp_cap, yearscode_cap = yearscode_cap,
    salary_cap = salary_cap, log_salary_median = log_salary_median,
    categorical_vars = categorical_vars, top_levels = top_levels,
    multi_response_vars = multi_response_vars, multi_response_terms = multi_response_terms
  )
}

# ============================================================
# 0.6 Apply preprocessing rules
# ============================================================

apply_gbm_preprocess <- function(data, prep) {
  data <- safe_remove_vars(data, prep$high_missing_vars)

  if ("WorkExp"             %in% names(data)) data$Missing_WorkExp             <- ifelse(is.na(data$WorkExp), 1, 0)
  if ("YearsCode"           %in% names(data)) data$Missing_YearsCode           <- ifelse(is.na(data$YearsCode), 1, 0)
  if ("ConvertedCompYearly" %in% names(data)) data$Missing_ConvertedCompYearly <- ifelse(is.na(data$ConvertedCompYearly), 1, 0)
  if ("OrgSize"             %in% names(data)) data$Missing_OrgSize             <- ifelse(is.na(data$OrgSize), 1, 0)

  if ("WorkExp" %in% names(data)) {
    data$WorkExp[data$WorkExp < 0] <- NA
    data$WorkExp <- ifelse(!is.na(data$WorkExp) & data$WorkExp > prep$workexp_cap,
                           prep$workexp_cap, data$WorkExp)
    data$WorkExp[is.na(data$WorkExp)] <- prep$numeric_medians[["WorkExp"]]
  }
  if ("YearsCode" %in% names(data)) {
    data$YearsCode[data$YearsCode < 0] <- NA
    data$YearsCode <- ifelse(!is.na(data$YearsCode) & data$YearsCode > prep$yearscode_cap,
                             prep$yearscode_cap, data$YearsCode)
    data$YearsCode[is.na(data$YearsCode)] <- prep$numeric_medians[["YearsCode"]]
  }
  for (v in setdiff(prep$numeric_vars, c("WorkExp", "YearsCode"))) {
    if (v %in% names(data)) data[[v]][is.na(data[[v]])] <- prep$numeric_medians[[v]]
  }

  if ("ConvertedCompYearly" %in% names(data)) {
    data$ConvertedCompYearly[data$ConvertedCompYearly < 0] <- NA
    data$ConvertedCompYearly <- ifelse(
      !is.na(data$ConvertedCompYearly) & data$ConvertedCompYearly > prep$salary_cap,
      prep$salary_cap, data$ConvertedCompYearly)
    data$LogConvertedCompYearly <- log1p(data$ConvertedCompYearly)
    data$LogConvertedCompYearly[is.na(data$LogConvertedCompYearly)] <- prep$log_salary_median
    data$ConvertedCompYearly <- NULL
  }

  for (v in prep$categorical_vars) {
    if (v %in% names(data)) {
      x <- as.character(data[[v]])
      x[is.na(x)] <- "Missing"
      x[!(x %in% prep$top_levels[[v]])] <- "Other"
      data[[v]] <- factor(x, levels = unique(c(prep$top_levels[[v]], "Other", "Missing")))
    }
  }

  for (v in prep$multi_response_vars) {
    if (v %in% names(data)) {
      raw_x <- as.character(data[[v]])
      raw_x[is.na(raw_x)] <- ""
      padded_x <- paste0(";", raw_x, ";")
      for (term in prep$multi_response_terms[[v]]) {
        data[[paste0(v, "_", make.names(term))]] <-
          ifelse(str_detect(padded_x, fixed(paste0(";", term, ";"))), 1, 0)
      }
      data[[v]] <- NULL
    }
  }
  data
}

# ============================================================
# 0.7 Fit and apply preprocessing
# ============================================================

gbm_prep   <- fit_gbm_preprocess(train_raw)
train_data <- apply_gbm_preprocess(train_raw, gbm_prep)
test_data  <- apply_gbm_preprocess(test_raw,  gbm_prep)

levels_order <- c("Very_Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very_Satisfied")
train_data$JobSat_Class <- factor(train_data$JobSat_Class, levels = levels_order)
test_data$JobSat_Class  <- factor(test_data$JobSat_Class,  levels = levels_order)

# ============================================================
# 0.8 Derived feature: LeavingScore (ordinal encoding of NewRole)
# ============================================================

leaving_map <- c(
  "I have strongly considered changing my career and/or the industry I work in" = 4,
  "I have somewhat considered changing my career and/or the industry I work in" = 3,
  "I have transitioned into a new career and/or industry involuntarily"         = 2,
  "I have neither consider or transitioned into a new career or industry"       = 1,
  "I have transitioned into a new career and/or industry voluntarily"           = 0
)

add_leaving_score <- function(data, median_value) {
  if ("NewRole" %in% names(data)) {
    nr_char <- as.character(data$NewRole)
    data$LeavingScore <- leaving_map[nr_char]
    data$LeavingScore[is.na(data$LeavingScore)] <- median_value
  }
  data
}

train_leaving_raw <- leaving_map[as.character(train_data$NewRole)]
leaving_median <- median(train_leaving_raw, na.rm = TRUE)

train_data <- add_leaving_score(train_data, leaving_median)
test_data  <- add_leaving_score(test_data,  leaving_median)

cat("LeavingScore added. Training median:", leaving_median, "\n")

# ============================================================
# 0.9 Feature selection based on training data only
# ============================================================

predictor_names <- setdiff(names(train_data), "JobSat_Class")
nzv_vars <- predictor_names[sapply(train_data[predictor_names], is_near_zero_var)]
nzv_vars <- nzv_vars[!is.na(nzv_vars) & nzv_vars != "" & nzv_vars %in% names(train_data)]
train_data <- safe_remove_vars(train_data, nzv_vars)
test_data  <- safe_remove_vars(test_data,  nzv_vars)

numeric_predictors <- names(train_data)[
  sapply(train_data, is.numeric) & names(train_data) != "JobSat_Class"]
numeric_predictors <- numeric_predictors[
  !is.na(numeric_predictors) & numeric_predictors != "" &
    numeric_predictors %in% names(train_data)]
numeric_predictors <- numeric_predictors[
  sapply(train_data[numeric_predictors], function(x) sd(x, na.rm = TRUE) > 0)]

high_corr_vars <- character(0)
if (length(numeric_predictors) > 1) {
  corr_matrix <- cor(train_data[numeric_predictors], use = "pairwise.complete.obs")
  corr_matrix[is.na(corr_matrix)] <- 0
  high_corr_index <- findCorrelation(corr_matrix, cutoff = 0.90, names = FALSE)
  if (length(high_corr_index) > 0) high_corr_vars <- numeric_predictors[high_corr_index]
}
high_corr_vars <- high_corr_vars[
  !is.na(high_corr_vars) & high_corr_vars != "" & high_corr_vars %in% names(train_data)]
train_data <- safe_remove_vars(train_data, high_corr_vars)
test_data  <- safe_remove_vars(test_data,  high_corr_vars)

removed_variables <- list(
  high_missing_vars       = gbm_prep$high_missing_vars,
  near_zero_variance_vars = nzv_vars,
  high_correlation_vars   = high_corr_vars
)

cat("\n===== Preprocessing complete =====\n")
cat("Training set:", nrow(train_data), "rows x", ncol(train_data), "cols\n")
cat("Test set:    ", nrow(test_data),  "rows x", ncol(test_data),  "cols\n")
cat("Removed vars:\n"); print(removed_variables)

# ============================================================
# 0.10 Build 3-class datasets (all 5 models use this)
#
# Mapping:
#   Dissatisfied = Very_Dissatisfied + Dissatisfied
#   Neutral      = Neutral
#   Satisfied    = Satisfied + Very_Satisfied
# ============================================================

remap_to_3class <- function(x) {
  factor(
    case_when(
      x %in% c("Very_Dissatisfied", "Dissatisfied") ~ "Dissatisfied",
      x == "Neutral"                                ~ "Neutral",
      x %in% c("Satisfied", "Very_Satisfied")       ~ "Satisfied",
      TRUE ~ NA_character_
    ),
    levels = c("Dissatisfied", "Neutral", "Satisfied")
  )
}

train_3 <- train_data
test_3  <- test_data
train_3$JobSat_Class <- remap_to_3class(train_data$JobSat_Class)
test_3$JobSat_Class  <- remap_to_3class(test_data$JobSat_Class)

cat("\n===== 3-Class Distribution =====\n")
cat("Train:\n"); print(table(train_3$JobSat_Class))
cat("Test:\n");  print(table(test_3$JobSat_Class))

# ============================================================
# 0.11 Unified evaluation function (used by ALL models)
# ============================================================

unified_evaluate <- function(pred, obs, model_name) {
  lev <- levels(obs)
  pred <- factor(pred, levels = lev)
  obs  <- factor(obs,  levels = lev)

  cm <- confusionMatrix(data = pred, reference = obs)
  cm_tab <- table(Actual = obs, Predicted = pred)

  # Per-class metrics
  class_metrics <- data.frame(
    Class     = lev,
    Precision = NA_real_,
    Recall    = NA_real_,
    F1        = NA_real_,
    Support   = NA_integer_
  )

  for (i in seq_along(lev)) {
    cn <- lev[i]
    TP <- cm_tab[cn, cn]
    FP <- sum(cm_tab[, cn]) - TP
    FN <- sum(cm_tab[cn, ]) - TP

    prec <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
    rec  <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
    f1   <- ifelse((prec + rec) == 0, 0, 2 * prec * rec / (prec + rec))

    class_metrics$Precision[i] <- round(prec, 4)
    class_metrics$Recall[i]    <- round(rec, 4)
    class_metrics$F1[i]        <- round(f1, 4)
    class_metrics$Support[i]   <- as.integer(sum(cm_tab[cn, ]))
  }

  # Overall metrics
  overall_metrics <- data.frame(
    Model           = model_name,
    Accuracy        = round(as.numeric(cm$overall["Accuracy"]), 4),
    Kappa           = round(as.numeric(cm$overall["Kappa"]), 4),
    Macro_F1        = round(mean(class_metrics$F1), 4),
    Weighted_F1     = round(weighted.mean(class_metrics$F1, class_metrics$Support), 4),
    Macro_Precision = round(mean(class_metrics$Precision), 4),
    Macro_Recall    = round(mean(class_metrics$Recall), 4)
  )

  list(
    confusion_matrix = cm,
    class_metrics    = class_metrics,
    overall_metrics  = overall_metrics
  )
}

# ============================================================
# Shared macro F1 summary function for caret CV
# ============================================================

macro_f1_summary <- function(data, lev = NULL, model = NULL) {
  obs  <- factor(data$obs,  levels = lev)
  pred <- factor(data$pred, levels = lev)
  cm <- table(obs, pred)
  f1_scores <- sapply(lev, function(cn) {
    TP <- cm[cn, cn]
    FP <- sum(cm[, cn]) - TP
    FN <- sum(cm[cn, ]) - TP
    prec <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
    rec  <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
    ifelse((prec + rec) == 0, 0, 2 * prec * rec / (prec + rec))
  })
  c(Macro_F1 = mean(f1_scores), Accuracy = mean(obs == pred))
}

# ============================================================
# Shared class weight function (sqrt weighting)
# ============================================================

make_class_weights <- function(y) {
  y <- factor(y)
  cc <- table(y)
  raw_w <- sqrt(max(cc) / cc)
  w <- raw_w[as.character(y)]
  as.numeric(w)
}


# ############################################################
#
#   MODEL 1: GRADIENT BOOSTING (GBM)
#
# ############################################################

cat("\n\n========== MODEL 1: GBM (3-Class) ==========\n")

# --- Parallel backend ---
use_parallel <- FALSE
cl_par <- NULL
if (requireNamespace("doParallel", quietly = TRUE)) {
  avail <- parallel::detectCores()
  if (!is.na(avail) && avail > 1) {
    cl_par <- parallel::makeCluster(max(1, avail - 1))
    doParallel::registerDoParallel(cl_par)
    use_parallel <- TRUE
    cat("Parallel backend:", max(1, avail - 1), "cores\n")
  }
}

# --- CV control ---
ctrl_tune_gbm <- trainControl(
  method = "cv", number = 3,
  summaryFunction = macro_f1_summary,
  classProbs = FALSE, sampling = NULL,
  savePredictions = "final", allowParallel = use_parallel
)

ctrl_final_gbm <- trainControl(
  method = "none", classProbs = FALSE,
  sampling = NULL, allowParallel = use_parallel
)

# --- Tuning grid (same as original GBM 3-class code) ---
gbm_grid_3 <- expand.grid(
  n.trees           = c(500, 800, 1000),
  interaction.depth = c(3, 5),
  shrinkage         = c(0.03, 0.05),
  n.minobsinnode    = c(10, 20)
)

cat("GBM grid: ", nrow(gbm_grid_3), " combinations\n")

# --- Tuning on subset (as in original code) ---
set.seed(123)
tune_p_3     <- min(1, 12000 / nrow(train_3))
tune_index_3 <- createDataPartition(train_3$JobSat_Class, p = tune_p_3, list = FALSE)
tune_data_3  <- train_3[tune_index_3, ]
tune_weights_3 <- make_class_weights(tune_data_3$JobSat_Class)

set.seed(123)
gbm_tuned_3 <- train(
  JobSat_Class ~ ., data = tune_data_3,
  method = "gbm", trControl = ctrl_tune_gbm,
  tuneGrid = gbm_grid_3, metric = "Macro_F1",
  weights = tune_weights_3, verbose = FALSE
)

best_grid_gbm <- gbm_tuned_3$bestTune
cat("GBM best params:\n"); print(best_grid_gbm)

# --- Final model on full training set ---
final_weights_3 <- make_class_weights(train_3$JobSat_Class)

set.seed(123)
gbm_final_3 <- train(
  JobSat_Class ~ ., data = train_3,
  method = "gbm", trControl = ctrl_final_gbm,
  tuneGrid = best_grid_gbm,
  weights = final_weights_3, verbose = FALSE
)

# --- Evaluate ---
gbm_pred_3 <- predict(gbm_final_3, newdata = test_3)
gbm_eval_3 <- unified_evaluate(gbm_pred_3, test_3$JobSat_Class, "GBM")

cat("\n>>> GBM 3-Class Results\n")
print(gbm_eval_3$overall_metrics)
print(gbm_eval_3$class_metrics)

# --- Variable importance ---
gbm_varimp <- varImp(gbm_final_3)

# --- Save ---
saveRDS(list(
  model_name     = "Gradient Boosting (GBM)",
  tuning_grid    = gbm_grid_3,
  tuning_results = gbm_tuned_3$results,
  best_params    = best_grid_gbm,
  overall_metrics = gbm_eval_3$overall_metrics,
  class_metrics   = gbm_eval_3$class_metrics,
  confusion_matrix = gbm_eval_3$confusion_matrix,
  variable_importance = gbm_varimp
), file = "results/gbm_3class.rds")

cat("Saved: results/gbm_3class.rds\n")

# --- Stop parallel ---
if (!is.null(cl_par)) {
  parallel::stopCluster(cl_par)
  if (requireNamespace("foreach", quietly = TRUE)) foreach::registerDoSEQ()
  cl_par <- NULL
  cat("Parallel backend stopped.\n")
}


# ############################################################
#
#   MODEL 2: DNN (Deep Neural Network)
#
# ############################################################

cat("\n\n========== MODEL 2: DNN (3-Class) ==========\n")

library(torch)
library(luz)

# --- Device ---
if (cuda_is_available()) {
  device <- torch_device("cuda")
  cat("GPU detected.\n")
} else {
  device <- torch_device("cpu")
  cat("Using CPU.\n")
}

# --- 3-class labels (unified: Dissatisfied / Neutral / Satisfied) ---
dnn_label_levels <- c("Dissatisfied", "Neutral", "Satisfied")

# Map from 5-class to 3-class integer labels (1-based for torch)
remap_to_3class_int <- function(x) {
  x_char <- as.character(x)
  x_new <- case_when(
    x_char %in% c("Very_Dissatisfied", "Dissatisfied") ~ 1L,
    x_char == "Neutral"                                 ~ 2L,
    x_char %in% c("Satisfied", "Very_Satisfied")        ~ 3L
  )
  x_new
}

y_train_dnn <- remap_to_3class_int(train_data$JobSat_Class)
y_test_dnn  <- remap_to_3class_int(test_data$JobSat_Class)

# --- One-hot encoding + Z-score standardization ---
X_train_raw <- train_data %>% select(-JobSat_Class)
X_test_raw  <- test_data  %>% select(-JobSat_Class)

X_train_raw$.__split__ <- "train"
X_test_raw$.__split__  <- "test"
combined <- bind_rows(X_train_raw, X_test_raw) %>%
  mutate(across(where(is.character), as.factor))
split_col <- combined$.__split__
combined$.__split__ <- NULL
X_onehot   <- model.matrix(~ . - 1, data = combined)
X_train_oh <- X_onehot[split_col == "train", ]
X_test_oh  <- X_onehot[split_col == "test",  ]

train_means <- apply(X_train_oh, 2, mean)
train_sds   <- apply(X_train_oh, 2, sd)
train_sds[train_sds == 0] <- 1

X_train_scaled <- scale(X_train_oh, center = train_means, scale = train_sds)
X_test_scaled  <- scale(X_test_oh,  center = train_means, scale = train_sds)

# Clip to [-5, 5]
X_train_scaled <- pmax(pmin(X_train_scaled, 5), -5)
X_test_scaled  <- pmax(pmin(X_test_scaled,  5), -5)

X_train_mat <- as.matrix(X_train_scaled)
X_test_mat  <- as.matrix(X_test_scaled)

n_features <- ncol(X_train_mat)
n_classes_dnn <- 3

cat("DNN features after one-hot:", n_features, "\n")

# --- Model architecture ---
dnn_model <- nn_module(
  classname = "JobSatDNN",
  initialize = function(n_features, n_classes,
                        hidden1 = 128, hidden2 = 64, hidden3 = 32,
                        dropout_rate = 0.4) {
    self$fc1     <- nn_linear(n_features, hidden1)
    self$fc2     <- nn_linear(hidden1,    hidden2)
    self$fc3     <- nn_linear(hidden2,    hidden3)
    self$fc4     <- nn_linear(hidden3,    n_classes)
    self$dropout <- nn_dropout(p = dropout_rate)
  },
  forward = function(x) {
    x <- self$dropout(nnf_relu(self$fc1(x)))
    x <- self$dropout(nnf_relu(self$fc2(x)))
    x <- self$dropout(nnf_relu(self$fc3(x)))
    self$fc4(x)
  }
)

# --- Focal Loss ---
focal_loss_module <- nn_module(
  classname = "FocalLoss",
  initialize = function(weight = NULL, gamma = 2.0) {
    self$gamma  <- gamma
    self$weight <- weight
  },
  forward = function(input, target) {
    log_probs <- nnf_log_softmax(input, dim = 2)
    log_pt    <- log_probs$gather(2, target$unsqueeze(2))$squeeze(2)
    pt        <- log_pt$exp()
    focal_w   <- (1 - pt)^self$gamma
    loss      <- -log_pt * focal_w
    if (!is.null(self$weight)) {
      alpha_t <- self$weight[target]
      loss    <- alpha_t * loss
    }
    loss$mean()
  }
)

# --- Class weights (sqrt + Neutral x1.5 boost) ---
dnn_counts   <- table(factor(y_train_dnn, levels = 1:3))
dnn_weights  <- as.numeric(sqrt(length(y_train_dnn) / (n_classes_dnn * dnn_counts)))
dnn_weights  <- dnn_weights / sum(dnn_weights) * n_classes_dnn
dnn_weights[2] <- dnn_weights[2] * 1.5  # Neutral boost
dnn_weights  <- dnn_weights / sum(dnn_weights) * n_classes_dnn
names(dnn_weights) <- dnn_label_levels
class_weights_tensor <- torch_tensor(dnn_weights, dtype = torch_float())$to(device = device)

cat("DNN class weights:", round(dnn_weights, 3), "\n")

# --- Helper functions ---
get_hidden_dims <- function(hidden_size) {
  if (hidden_size == "large") list(h1 = 128, h2 = 64, h3 = 32)
  else                        list(h1 = 64,  h2 = 32, h3 = 16)
}

predict_classes_dnn <- function(model, X_mat, device) {
  model$eval()
  with_no_grad({
    X_tensor <- torch_tensor(X_mat, dtype = torch_float())$to(device = device)
    logits   <- model(X_tensor)
    as.integer(torch_argmax(logits, dim = 2)$cpu())
  })
}

# --- Grid search: 2x2x2 = 8 combos, 3-fold CV ---
set.seed(123)
torch_manual_seed(123)

dnn_grid <- expand.grid(
  lr           = c(0.001, 0.0005),
  dropout_rate = c(0.3, 0.5),
  hidden_size  = c("large", "small"),
  stringsAsFactors = FALSE
)

n_folds_dnn <- 3
fold_index_dnn <- createFolds(factor(y_train_dnn, levels = 1:3),
                              k = n_folds_dnn, list = TRUE)

cat("DNN grid search:", nrow(dnn_grid), "combos x", n_folds_dnn, "folds\n\n")

dnn_grid_results <- vector("list", nrow(dnn_grid))

for (i in seq_len(nrow(dnn_grid))) {
  lr           <- dnn_grid$lr[i]
  dropout_rate <- dnn_grid$dropout_rate[i]
  hidden_size  <- dnn_grid$hidden_size[i]
  dims         <- get_hidden_dims(hidden_size)

  cat(sprintf(">>> DNN combo %d/%d | lr=%.4f | dropout=%.1f | hidden=%s\n",
              i, nrow(dnn_grid), lr, dropout_rate, hidden_size))

  fold_mf1 <- numeric(n_folds_dnn)

  for (k in seq_len(n_folds_dnn)) {
    val_idx   <- fold_index_dnn[[k]]
    train_idx <- setdiff(seq_len(nrow(X_train_mat)), val_idx)

    X_cv_train <- X_train_mat[train_idx, ]
    y_cv_train <- y_train_dnn[train_idx]
    X_cv_val   <- X_train_mat[val_idx, ]
    y_cv_val   <- y_train_dnn[val_idx]

    # Per-fold class weights
    cv_counts  <- table(factor(y_cv_train, levels = 1:3))
    cv_w <- as.numeric(sqrt(length(y_cv_train) / (n_classes_dnn * cv_counts)))
    cv_w <- cv_w / sum(cv_w) * n_classes_dnn
    cv_w[2] <- cv_w[2] * 1.5
    cv_w <- cv_w / sum(cv_w) * n_classes_dnn
    cv_wt <- torch_tensor(cv_w, dtype = torch_float())$to(device = device)

    make_ds <- dataset(
      name = "CVDataset",
      initialize = function(X, y) {
        self$X <- torch_tensor(X, dtype = torch_float())
        self$y <- torch_tensor(y, dtype = torch_long())
      },
      .getitem = function(i) list(x = self$X[i, ], y = self$y[i]),
      .length  = function()  nrow(self$X)
    )

    cv_train_dl <- dataloader(make_ds(X_cv_train, y_cv_train),
                              batch_size = 256, shuffle = TRUE)
    cv_val_dl   <- dataloader(make_ds(X_cv_val, y_cv_val),
                              batch_size = 256, shuffle = FALSE)

    cv_focal <- focal_loss_module(weight = cv_wt, gamma = 2.0)

    fitted_cv <- tryCatch({
      dnn_model %>%
        setup(loss = cv_focal, optimizer = optim_adam,
              metrics = list(luz_metric_accuracy())) %>%
        set_hparams(n_features = n_features, n_classes = n_classes_dnn,
                    hidden1 = dims$h1, hidden2 = dims$h2, hidden3 = dims$h3,
                    dropout_rate = dropout_rate) %>%
        set_opt_hparams(lr = lr, weight_decay = 1e-4) %>%
        fit(data = cv_train_dl, epochs = 80, valid_data = cv_val_dl,
            accelerator = accelerator(device_placement = TRUE,
                                      cpu = !cuda_is_available()),
            callbacks = list(luz_callback_early_stopping(patience = 12)))
    }, error = function(e) {
      cat(sprintf("    Fold %d failed: %s\n", k, conditionMessage(e)))
      NULL
    })

    if (is.null(fitted_cv)) { fold_mf1[k] <- NA; next }

    y_cv_pred <- predict_classes_dnn(fitted_cv$model, X_cv_val, device)
    # Compute macro F1 manually
    pred_f <- factor(dnn_label_levels[y_cv_pred], levels = dnn_label_levels)
    true_f <- factor(dnn_label_levels[y_cv_val],  levels = dnn_label_levels)
    cm_cv  <- table(true_f, pred_f)
    f1s <- sapply(dnn_label_levels, function(cn) {
      TP <- cm_cv[cn, cn]; FP <- sum(cm_cv[, cn]) - TP; FN <- sum(cm_cv[cn, ]) - TP
      p <- ifelse((TP+FP)==0, 0, TP/(TP+FP)); r <- ifelse((TP+FN)==0, 0, TP/(TP+FN))
      ifelse((p+r)==0, 0, 2*p*r/(p+r))
    })
    fold_mf1[k] <- mean(f1s)

    cat(sprintf("    Fold %d | Macro F1 = %.4f\n", k, fold_mf1[k]))
  }

  dnn_grid_results[[i]] <- list(
    lr = lr, dropout_rate = dropout_rate, hidden_size = hidden_size,
    cv_mf1_mean = mean(fold_mf1, na.rm = TRUE),
    cv_mf1_sd   = sd(fold_mf1, na.rm = TRUE),
    fold_mf1    = fold_mf1
  )
  cat(sprintf("    CV Macro F1: %.4f +/- %.4f\n\n",
              dnn_grid_results[[i]]$cv_mf1_mean,
              dnn_grid_results[[i]]$cv_mf1_sd))
}

# --- DNN tuning results table ---
dnn_tuning_df <- do.call(rbind, lapply(dnn_grid_results, function(r) {
  data.frame(lr = r$lr, dropout = r$dropout_rate, hidden = r$hidden_size,
             CV_Macro_F1 = round(r$cv_mf1_mean, 4),
             CV_Macro_F1_SD = round(r$cv_mf1_sd, 4))
}))
dnn_tuning_df <- dnn_tuning_df[order(-dnn_tuning_df$CV_Macro_F1), ]
cat("DNN grid results:\n"); print(dnn_tuning_df, row.names = FALSE)

best_dnn_idx <- which.max(sapply(dnn_grid_results, function(r) r$cv_mf1_mean))
best_dnn     <- dnn_grid_results[[best_dnn_idx]]

cat(sprintf("\nDNN best: lr=%.4f | dropout=%.1f | hidden=%s\n",
            best_dnn$lr, best_dnn$dropout_rate, best_dnn$hidden_size))

# --- Train final DNN model ---
best_dnn_dims <- get_hidden_dims(best_dnn$hidden_size)

jobsat_ds <- dataset(
  name = "JobSatFull",
  initialize = function(X, y) {
    self$X <- torch_tensor(X, dtype = torch_float())
    self$y <- torch_tensor(y, dtype = torch_long())
  },
  .getitem = function(i) list(x = self$X[i, ], y = self$y[i]),
  .length  = function()  nrow(self$X)
)

final_train_dl <- dataloader(jobsat_ds(X_train_mat, y_train_dnn),
                             batch_size = 256, shuffle = TRUE)
final_test_dl  <- dataloader(jobsat_ds(X_test_mat,  y_test_dnn),
                             batch_size = 256, shuffle = FALSE)

set.seed(123); torch_manual_seed(123)
final_focal <- focal_loss_module(weight = class_weights_tensor, gamma = 2.0)

dnn_final_model <- dnn_model %>%
  setup(loss = final_focal, optimizer = optim_adam,
        metrics = list(luz_metric_accuracy())) %>%
  set_hparams(n_features = n_features, n_classes = n_classes_dnn,
              hidden1 = best_dnn_dims$h1, hidden2 = best_dnn_dims$h2,
              hidden3 = best_dnn_dims$h3, dropout_rate = best_dnn$dropout_rate) %>%
  set_opt_hparams(lr = best_dnn$lr, weight_decay = 1e-4) %>%
  fit(data = final_train_dl, epochs = 80, valid_data = final_test_dl,
      accelerator = accelerator(device_placement = TRUE,
                                cpu = !cuda_is_available()),
      callbacks = list(
        luz_callback_lr_scheduler(torch::lr_reduce_on_plateau, patience = 5, factor = 0.5),
        luz_callback_early_stopping(patience = 15)),
      verbose = TRUE)

cat("\nDNN final model trained.\n")

# --- Evaluate ---
y_pred_dnn <- predict_classes_dnn(dnn_final_model$model, X_test_mat, device)
# Convert integer predictions to factor with unified labels
pred_factor_dnn <- factor(dnn_label_levels[y_pred_dnn], levels = dnn_label_levels)
obs_factor_dnn  <- factor(dnn_label_levels[y_test_dnn], levels = dnn_label_levels)

dnn_eval_3 <- unified_evaluate(pred_factor_dnn, obs_factor_dnn, "DNN")

cat("\n>>> DNN 3-Class Results\n")
print(dnn_eval_3$overall_metrics)
print(dnn_eval_3$class_metrics)

# --- Training history ---
dnn_history <- dnn_final_model$records$metrics
dnn_epochs  <- length(dnn_history$train)
dnn_history_df <- data.frame(
  epoch      = 1:dnn_epochs,
  train_loss = sapply(dnn_history$train, function(x) x$loss),
  valid_loss = sapply(dnn_history$valid, function(x) x$loss),
  train_acc  = sapply(dnn_history$train, function(x) x$acc),
  valid_acc  = sapply(dnn_history$valid, function(x) x$acc)
)

# --- Save (note: torch model objects may not serialize well with saveRDS,
#     so we save metrics/results only, not the model object itself) ---
saveRDS(list(
  model_name       = "Deep Neural Network (DNN)",
  tuning_grid      = dnn_grid,
  tuning_results   = dnn_tuning_df,
  best_params      = data.frame(lr = best_dnn$lr,
                                dropout = best_dnn$dropout_rate,
                                hidden = best_dnn$hidden_size),
  overall_metrics  = dnn_eval_3$overall_metrics,
  class_metrics    = dnn_eval_3$class_metrics,
  confusion_matrix = dnn_eval_3$confusion_matrix,
  training_history = dnn_history_df,
  grid_results_raw = dnn_grid_results
), file = "results/dnn_3class.rds")

cat("Saved: results/dnn_3class.rds\n")


# ############################################################
#
#   MODEL 3: NAIVE BAYES
#
#   IMPORTANT FIX: Uses shared train_3/test_3 (not re-split).
#   Uses SMOTE + 3-fold CV tuning of laplace/usekernel.
#
# ############################################################

cat("\n\n========== MODEL 3: Naive Bayes (3-Class) ==========\n")

library(e1071)
library(recipes)
library(themis)

# --- Prepare NB data from shared train_3/test_3 ---
# Convert all factors to numeric for SMOTE, keep JobSat_Class as factor
train_nb <- train_3
test_nb  <- test_3

# Identify factor columns (excluding target)
factor_cols_nb <- names(train_nb)[sapply(train_nb, is.factor)]
factor_cols_nb <- setdiff(factor_cols_nb, "JobSat_Class")

# Convert factors to numeric for SMOTE
train_nb[factor_cols_nb] <- lapply(train_nb[factor_cols_nb], as.numeric)
test_nb[factor_cols_nb]  <- lapply(test_nb[factor_cols_nb],  as.numeric)

cat("NB pre-SMOTE class distribution:\n")
print(table(train_nb$JobSat_Class))

# --- SMOTE oversampling (over_ratio=0.8 as in original NB 3-class code) ---
rec_nb <- recipe(JobSat_Class ~ ., data = train_nb) %>%
  step_smote(JobSat_Class, over_ratio = 0.8, seed = 123)
rec_prep_nb   <- prep(rec_nb, training = train_nb)
train_bal_nb  <- bake(rec_prep_nb, new_data = NULL)

cat("NB post-SMOTE class distribution:\n")
print(table(train_bal_nb$JobSat_Class))

# --- CV tuning ---
if (requireNamespace("doParallel", quietly = TRUE)) {
  cl_nb <- parallel::makeCluster(max(1, parallel::detectCores() - 1))
  doParallel::registerDoParallel(cl_nb)
} else {
  cl_nb <- NULL
}

ctrl_nb <- trainControl(
  method = "cv", number = 3,
  classProbs = TRUE,
  summaryFunction = multiClassSummary
)

tune_grid_nb <- expand.grid(
  laplace   = c(0, 1, 5),
  usekernel = c(FALSE, TRUE),
  adjust    = 1
)

set.seed(123)
nb_model_3 <- train(
  JobSat_Class ~ .,
  data      = train_bal_nb,
  method    = "naive_bayes",
  trControl = ctrl_nb,
  tuneGrid  = tune_grid_nb,
  metric    = "Mean_F1"
)

if (!is.null(cl_nb)) {
  parallel::stopCluster(cl_nb)
  if (requireNamespace("foreach", quietly = TRUE)) foreach::registerDoSEQ()
}

cat("NB best params:\n"); print(nb_model_3$bestTune)

# --- Evaluate on shared test set ---
nb_pred_3 <- predict(nb_model_3, newdata = test_nb)
nb_eval_3 <- unified_evaluate(nb_pred_3, test_nb$JobSat_Class, "Naive Bayes")

cat("\n>>> NB 3-Class Results\n")
print(nb_eval_3$overall_metrics)
print(nb_eval_3$class_metrics)

# --- Save ---
saveRDS(list(
  model_name       = "Naive Bayes",
  tuning_grid      = tune_grid_nb,
  tuning_results   = nb_model_3$results,
  best_params      = nb_model_3$bestTune,
  overall_metrics  = nb_eval_3$overall_metrics,
  class_metrics    = nb_eval_3$class_metrics,
  confusion_matrix = nb_eval_3$confusion_matrix
), file = "results/nb_3class.rds")

cat("Saved: results/nb_3class.rds\n")


# ############################################################
#
#   MODEL 4: RANDOM FOREST
#
# ############################################################

cat("\n\n========== MODEL 4: Random Forest (3-Class) ==========\n")

library(ranger)

# --- Class weight function (power weighting, as in original RF code) ---
make_class_weights_rf <- function(y, weight_power = 0.75) {
  cc <- table(y)
  cw <- (max(cc) / cc) ^ weight_power
  as.numeric(cw)
}

# --- CV controls ---
ctrl_rf <- trainControl(
  method = "cv", number = 3,
  summaryFunction = macro_f1_summary,
  savePredictions = "final", classProbs = FALSE
)

ctrl_rf_down <- trainControl(
  method = "cv", number = 3,
  summaryFunction = macro_f1_summary,
  savePredictions = "final", classProbs = FALSE,
  sampling = "down"
)

p_rf <- ncol(train_3) - 1

# --- Grid for class-weighted RF ---
rf_grid_w <- expand.grid(
  mtry = unique(pmin(p_rf, c(6, 8, 10, 12, 16))),
  splitrule = "extratrees",
  min.node.size = c(10, 15, 20)
)

# --- Grid for downsampled RF ---
rf_grid_d <- expand.grid(
  mtry = unique(pmin(p_rf, c(6, 8, 10, 12))),
  splitrule = "extratrees",
  min.node.size = c(10, 15, 20)
)

all_rf_models  <- list()
all_rf_results <- list()
rf_counter     <- 1

# --- Class-weighted RF (3 weight powers) ---
for (wp in c(0.75, 0.85, 0.95)) {
  cw <- make_class_weights_rf(train_3$JobSat_Class, wp)
  set.seed(123)
  rf_fit <- train(
    JobSat_Class ~ ., data = train_3,
    method = "ranger", trControl = ctrl_rf,
    tuneGrid = rf_grid_w, metric = "Macro_F1",
    num.trees = 100, importance = "impurity",
    respect.unordered.factors = "order",
    class.weights = cw
  )
  res_tmp <- rf_fit$results %>%
    mutate(sample_method = "weights", weight_power = wp,
           model_id = paste0("weights_", wp))
  all_rf_models[[paste0("weights_", wp)]] <- rf_fit
  all_rf_results[[rf_counter]] <- res_tmp
  rf_counter <- rf_counter + 1
}

# --- Downsampled RF ---
set.seed(123)
rf_down <- train(
  JobSat_Class ~ ., data = train_3,
  method = "ranger", trControl = ctrl_rf_down,
  tuneGrid = rf_grid_d, metric = "Macro_F1",
  num.trees = 100, importance = "impurity",
  respect.unordered.factors = "order"
)
res_down <- rf_down$results %>%
  mutate(sample_method = "downsample", weight_power = 0, model_id = "downsample")
all_rf_models[["downsample"]] <- rf_down
all_rf_results[[rf_counter]] <- res_down

# --- Select best RF ---
rf_tuning_table <- bind_rows(all_rf_results) %>% arrange(desc(Macro_F1))
best_rf_row     <- rf_tuning_table %>% slice(1)
best_rf_id      <- as.character(best_rf_row$model_id[[1]])
best_rf_model   <- all_rf_models[[best_rf_id]]

cat("RF best params:\n"); print(best_rf_row)

# --- Evaluate ---
rf_pred_3 <- predict(best_rf_model, newdata = test_3)
rf_eval_3 <- unified_evaluate(rf_pred_3, test_3$JobSat_Class, "Random Forest")

cat("\n>>> RF 3-Class Results\n")
print(rf_eval_3$overall_metrics)
print(rf_eval_3$class_metrics)

# --- Save ---
saveRDS(list(
  model_name       = "Random Forest",
  tuning_table     = rf_tuning_table,
  best_params      = best_rf_row,
  overall_metrics  = rf_eval_3$overall_metrics,
  class_metrics    = rf_eval_3$class_metrics,
  confusion_matrix = rf_eval_3$confusion_matrix
), file = "results/rf_3class.rds")

cat("Saved: results/rf_3class.rds\n")


# ############################################################
#
#   MODEL 5: WEIGHTED MULTINOMIAL LOGISTIC REGRESSION
#
# ############################################################

cat("\n\n========== MODEL 5: Multinomial Logistic Regression (3-Class) ==========\n")

library(nnet)

# --- Case weights (power=0.75, same as original code) ---
make_case_weights_lr <- function(y, weight_power = 0.75) {
  cc <- table(y)
  cw <- (max(cc) / cc) ^ weight_power
  as.numeric(cw[as.character(y)])
}

# --- Dummy encoding ---
x_train_lr <- train_3 %>% select(-JobSat_Class)
x_test_lr  <- test_3  %>% select(-JobSat_Class)
y_train_lr <- train_3$JobSat_Class
y_test_lr  <- test_3$JobSat_Class

dummy_lr <- dummyVars(~ ., data = x_train_lr, fullRank = TRUE)
x_train_dummy <- predict(dummy_lr, newdata = x_train_lr) %>% as.data.frame()
x_test_dummy  <- predict(dummy_lr, newdata = x_test_lr)  %>% as.data.frame()

# Remove near-zero variance from dummy columns
nzv_idx_lr <- nearZeroVar(x_train_dummy, saveMetrics = FALSE)
if (length(nzv_idx_lr) > 0) {
  x_train_dummy <- x_train_dummy[, -nzv_idx_lr]
  x_test_dummy  <- x_test_dummy[,  -nzv_idx_lr]
}

case_weights_lr <- make_case_weights_lr(y_train_lr, 0.75)

cat("LR input dims:", dim(x_train_dummy), "\n")

# --- CV tuning ---
ctrl_lr <- trainControl(
  method = "cv", number = 3,
  summaryFunction = macro_f1_summary,
  savePredictions = "final", classProbs = FALSE
)

multinom_grid <- expand.grid(decay = c(0, 0.001, 0.01, 0.1))

set.seed(123)
lr_model_3 <- train(
  x = x_train_dummy, y = y_train_lr,
  method = "multinom", trControl = ctrl_lr,
  tuneGrid = multinom_grid, metric = "Macro_F1",
  preProcess = c("center", "scale"),
  weights = case_weights_lr,
  trace = FALSE, maxit = 300, MaxNWts = 100000
)

cat("LR best params:\n"); print(lr_model_3$bestTune)

# --- Evaluate ---
lr_pred_3 <- predict(lr_model_3, newdata = x_test_dummy)
lr_eval_3 <- unified_evaluate(lr_pred_3, y_test_lr, "Multinomial Logistic Regression")

cat("\n>>> LR 3-Class Results\n")
print(lr_eval_3$overall_metrics)
print(lr_eval_3$class_metrics)

# --- Save ---
saveRDS(list(
  model_name       = "Multinomial Logistic Regression",
  tuning_grid      = multinom_grid,
  tuning_results   = lr_model_3$results,
  best_params      = lr_model_3$bestTune,
  overall_metrics  = lr_eval_3$overall_metrics,
  class_metrics    = lr_eval_3$class_metrics,
  confusion_matrix = lr_eval_3$confusion_matrix
), file = "results/lr_3class.rds")

cat("Saved: results/lr_3class.rds\n")


# ############################################################
#
#   FINAL COMPARISON TABLE
#
# ############################################################

cat("\n\n")
cat("============================================================\n")
cat("       FINAL 3-CLASS MODEL COMPARISON\n")
cat("============================================================\n\n")

final_comparison <- bind_rows(
  gbm_eval_3$overall_metrics,
  dnn_eval_3$overall_metrics,
  nb_eval_3$overall_metrics,
  rf_eval_3$overall_metrics,
  lr_eval_3$overall_metrics
) %>%
  arrange(desc(Macro_F1))

print(final_comparison)

# Save the comparison table
saveRDS(final_comparison, file = "results/model_comparison.rds")

# Save preprocessing info for the report
saveRDS(list(
  removed_variables = removed_variables,
  leaving_median    = leaving_median,
  train_3_distribution = table(train_3$JobSat_Class),
  test_3_distribution  = table(test_3$JobSat_Class),
  n_train = nrow(train_3),
  n_test  = nrow(test_3),
  n_features_final = ncol(train_3) - 1,
  feature_names = setdiff(names(train_3), "JobSat_Class")
), file = "results/preprocessing_info.rds")

cat("\nSaved: results/model_comparison.rds\n")
cat("Saved: results/preprocessing_info.rds\n")
cat("\n===== ALL DONE =====\n")
