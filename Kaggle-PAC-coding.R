#!/usr/bin/env Rscript
#
# run_pac_pipeline.R
# Full pipeline for PAC competition (data load -> preprocess -> vtreat -> sparse -> xgb.cv -> train -> predict)
#
# Usage:
#   Edit 'data_path' below to point to folder containing analysisData.csv and scoringData.csv
#   Run in R: source("run_pac_pipeline.R")
#   or CLI: Rscript run_pac_pipeline.R
#

## ---------- Configuration ----------
data_path <- "./data/raw"   # <-- 修改为你的数据目录（包含 analysisData.csv & scoringData.csv）
train_file <- file.path(data_path, "analysisData.csv")
scoring_file <- file.path(data_path, "scoringData.csv")

out_dir      <- "."         # 根输出目录（脚本会在此创建 models/, data/processed/, submissions/）
processed_dir <- file.path(out_dir, "data", "processed")
models_dir    <- file.path(out_dir, "models")
subs_dir      <- file.path(out_dir, "submissions")

use_vtreat_design_from_file <- FALSE
vtreat_plan_path <- file.path(models_dir, "vtreat_plan.rds")  # 如果你已有 plan，可设置 TRUE 并指定路径

# XGBoost settings
max_rounds <- 5000
cv_folds   <- 5
early_stop_rounds <- 50
xgb_eta    <- 0.03
xgb_max_depth <- 6
xgb_subsample <- 0.8
xgb_colsample <- 0.8

# seed
seed_val <- 10946

# Create directories
dir.create(processed_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(models_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(subs_dir, recursive = TRUE, showWarnings = FALSE)

## ---------- Dependency checks ----------
pkgs <- c("data.table","vtreat","Matrix","xgboost","parallel","methods","utils")
missing_pkgs <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop("Missing required packages: ", paste(missing_pkgs, collapse = ", "),
       "\nPlease install them, e.g.: install.packages(c(", paste(missing_pkgs, collapse = ","), "))")
}
# load libs
library(data.table); library(vtreat); library(Matrix); library(xgboost); library(parallel)

cat("Starting pipeline (seed =", seed_val, ")\n")
set.seed(seed_val)

## ---------- Load data ----------
if (!file.exists(train_file)) stop("Train file not found: ", train_file)
if (!file.exists(scoring_file)) stop("Scoring file not found: ", scoring_file)

cat("Reading data...\n")
dt <- fread(train_file, na.strings = c("", "NA"))
scoring <- fread(scoring_file, na.strings = c("", "NA"))
cat("Train dims:", dim(dt), "Scoring dims:", dim(scoring), "\n")

## ---------- Preprocessing: numeric imputation w/ training means & factor alignment ----------
cat("Preprocessing: numeric imputation and factor alignment...\n")
num_cols  <- names(dt)[sapply(dt, is.numeric)]
char_cols <- names(dt)[sapply(dt, is.character)]

# compute training means for numeric columns
train_means <- dt[, lapply(.SD, mean, na.rm = TRUE), .SDcols = num_cols]

# fill NA in training (vectorized)
dt[, (num_cols) := Map(function(col, m) { col[is.na(col)] <- m; col }, .SD = .SD, m = as.numeric(train_means)), .SDcols = num_cols]

# fill NA in scoring using training means (avoid test leakage)
for (j in seq_along(num_cols)) {
  colname <- num_cols[j]
  if (colname %in% names(scoring)) {
    set(scoring, which(is.na(scoring[[colname]])), colname, train_means[[colname]])
  } else {
    # create column in scoring if missing (rare)
    scoring[[colname]] <- train_means[[colname]]
  }
}

# Align factor levels from training set
for (cname in char_cols) {
  if (!cname %in% names(scoring)) scoring[[cname]] <- NA_character_
  levs <- unique(dt[[cname]])
  dt[[cname]] <- factor(dt[[cname]], levels = levs)
  scoring[[cname]] <- factor(scoring[[cname]], levels = levs)
}

# Save processed copies
fwrite(dt, file = file.path(processed_dir, "train_filled.csv"))
fwrite(scoring, file = file.path(processed_dir, "scoring_filled.csv"))
cat("Saved processed data to", processed_dir, "\n")

## ---------- Feature engineering: squared terms (optional, example subset) ----------
cat("Feature engineering: adding squared terms for selected numeric candidates...\n")
# choose numeric features to square - user can modify this section
num_candidates <- setdiff(num_cols, c("id","price"))
num_to_square <- head(num_candidates, 14)  # example: first 14 numeric columns (customize as needed)

for (nm in num_to_square) {
  new_nm <- paste0(nm, "_sq")
  dt[[new_nm]] <- dt[[nm]]^2
  scoring[[new_nm]] <- scoring[[nm]]^2
}
cat("Added squared terms for:", paste(num_to_square, collapse = ", "), "\n")

## ---------- vtreat: design on training only or load existing plan ----------
if (use_vtreat_design_from_file && file.exists(vtreat_plan_path)) {
  cat("Loading vtreat plan from file:", vtreat_plan_path, "\n")
  trt_plan <- readRDS(vtreat_plan_path)
} else {
  cat("Designing vtreat plan on training data (this may take some time)...\n")
  varlist <- setdiff(names(dt), c("id", "price"))
  trt_plan <- designTreatmentsZ(dframe = dt, varlist = varlist)
  saveRDS(trt_plan, file = vtreat_plan_path)
  cat("Saved vtreat plan to", vtreat_plan_path, "\n")
}

# choose vtreat vars recommended ('clean' & 'lev')
score_frame <- trt_plan$scoreFrame
use_vars <- as.character(score_frame[score_frame$code %in% c("clean", "lev"), "varName"])
cat("Number of vtreat features to use:", length(use_vars), "\n")

# prepare
train_prep <- prepare(trt_plan, dt, varRestriction = use_vars)
test_prep  <- prepare(trt_plan, scoring, varRestriction = use_vars)
cat("Prepared vtreat datasets: train_prep dims:", dim(train_prep), "test_prep dims:", dim(test_prep), "\n")

y_train <- dt$price

## ---------- Sparse matrices & xgb DMatrix ----------
cat("Converting to sparse model matrix...\n")
X_train <- sparse.model.matrix(~ . - 1, data = as.data.frame(train_prep))
X_test  <- sparse.model.matrix(~ . - 1, data = as.data.frame(test_prep))
cat("Sparse dims: train:", dim(X_train), "test:", dim(X_test), "\n")

dtrain <- xgb.DMatrix(data = X_train, label = y_train, missing = NA)

## ---------- xgboost params & GPU detection ----------
cat("Preparing xgboost parameters and detecting GPU availability...\n")
params <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eta = xgb_eta,
  max_depth = xgb_max_depth,
  subsample = xgb_subsample,
  colsample_bytree = xgb_colsample,
  min_child_weight = 1,
  eval_metric = "rmse"
)

# Simple GPU detection: if xgboost supports 'gpu_hist' try it (works on Linux with GPU-enabled xgboost)
gpu_ok <- FALSE
try({
  # attempt small gpu call; if it errors then gpu not available
  if ("gpu_hist" %in% names(xgb.parameters())) {
    # NOTE: many xgboost builds won't show gpu_hist here; we test environment var fallback
    NULL
  }
}, silent = TRUE)

# fallback: check env var CUDA_VISIBLE_DEVICES or OS
if (nzchar(Sys.getenv("CUDA_VISIBLE_DEVICES", unset = "")) && Sys.info()[['sysname']] != "Windows") {
  params$tree_method <- "gpu_hist"
  gpu_ok <- TRUE
  cat("GPU environment variable detected; using 'gpu_hist'.\n")
} else {
  params$tree_method <- "hist"
  cat("Using CPU tree_method = 'hist'. If you have GPU-enabled xgboost, set CUDA_VISIBLE_DEVICES and re-run.\n")
}

nthread_val <- max(1, detectCores() - 1)
cat("nthread set to", nthread_val, "\n")

## ---------- xgb.cv to find best nrounds ----------
cat("Starting xgb.cv (this may take a while)...\n")
cv <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = max_rounds,
  nfold = cv_folds,
  early_stopping_rounds = early_stop_rounds,
  verbose = 1,
  showsd = TRUE,
  nthread = nthread_val,
  prediction = FALSE
)
best_nrounds <- cv$best_iteration
cat("xgb.cv finished. Best nrounds =", best_nrounds, "\n")

## ---------- Final training on full training set ----------
cat("Training final xgboost model with nrounds =", best_nrounds, "...\n")
bst <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,
  watchlist = list(train = dtrain),
  print_every_n = 50,
  nthread = nthread_val
)

# Save model artifacts
xgb_model_path <- file.path(models_dir, "xgb_final.model")
xgb_rds_path   <- file.path(models_dir, "xgb_final.rds")
cat("Saving model to", xgb_model_path, "and", xgb_rds_path, "...\n")
xgb.save(bst, xgb_model_path)
saveRDS(bst, xgb_rds_path)

## ---------- Predict on test & export ----------
cat("Predicting on scoring/test dataset...\n")
pred_test <- predict(bst, newdata = X_test)
submission <- data.table(id = if ("id" %in% names(scoring)) scoring$id else seq_len(nrow(scoring)),
                         price = pred_test)
out_sub_path <- file.path(subs_dir, paste0("submission_xgb_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv"))
fwrite(submission, file = out_sub_path)
cat("Saved submission to", out_sub_path, "\n")

## ---------- Feature importance (top 30) ----------
cat("Computing feature importance (top 30)...\n")
imp <- xgb.importance(feature_names = colnames(X_train), model = bst)
top_imp <- head(imp, 30)
print(top_imp)
# save importance
fwrite(top_imp, file = file.path(models_dir, "feature_importance_top30.csv"))

cat("Pipeline finished successfully.\n")
