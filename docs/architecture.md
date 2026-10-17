# Architecture

## Data flow

```
data/raw/{train,test,RUL}_FD001.txt
        │  DataIngestion
        ▼
data/processed/{train,test}_set_001.csv        (named columns, no labels yet)
        │  DataTransformation
        ▼
artifacts/X_train_processed.csv, y_train.csv   (PC1..PC10, RUL capped at 125)
artifacts/X_test_processed.csv,  y_test.csv    (last cycle per test engine)
artifacts/preprocessor.pkl                     (StandardScaler + PCA)
        │  ExperimentSplitter  (by engine_id)
        ▼
artifacts/X_train_proper.csv, y_train_proper.csv   (80% of engines)
artifacts/X_cal.csv, y_cal.csv                     (20% of engines)
        │  XGBoostModelTrainer
        ▼
models/xgboost_model.pkl
        │  ConformalCalibrator
        ▼
artifacts/residuals.pkl, conformal_results.csv
```

Each stage is a DVC stage in `dvc.yaml`, so `dvc repro` re-runs only what changed.

## Modules

| Module | Responsibility | Does not do |
|---|---|---|
| `components/data_ingestion.py` | Read whitespace-separated C-MAPSS files, assign column names (`engine_id`, `cycle`, 3 operational settings, `s1`–`s21`), write CSVs | Labelling, scaling |
| `components/data_transformation.py` | Add capped RUL labels, drop 7 constant/uninformative sensors, fit `StandardScaler` + `PCA` on train only, transform train and test, save the fitted preprocessor | Calibration split, model training |
| `components/experiment_splitter.py` | Split training data into *train proper* and *calibration* by `engine_id` (seeded) | Anything else. It exists only because of conformal prediction |
| `components/base_model_training.py` | Grid search (3-fold CV, RMSE) over Linear Regression, Random Forest, SVR and XGBoost; write per-model grid results and a top-5 table | Final model selection (done in notebooks) |
| `pipeline/xgboost_training.py` | Train the chosen XGBoost model with the hyperparameters from `configs/` | |
| `pipeline/gp_training.py` | Train a Gaussian Process (Matérn kernel) on downsampled data. Not part of the DVC pipeline | |
| `components/conformal_calibration.py` | Compute absolute residuals on the calibration set, the finite-sample-corrected quantile `q̂`, and evaluate coverage and width on the test set | |
| `components/conformal_plots.py` | Plot a single prediction interval, returned as base64 PNG for the API | |
| `utils.py`, `logger.py`, `exception.py` | `save_object` (dill), `evaluate_model` / `get_best_model`, timestamped log files in `logs/`, `CustomException` with file and line info | |

Configuration values (RUL cap, PCA components, α, hyperparameters, GP settings) live in
`configs/data_constants_config.py`. Component paths are `@dataclass` configs at the top of each module.

## Design decisions

- **Engine-level calibration split.** Each engine is an independent degradation trajectory. Splitting by row would put
  early cycles of an engine in training and later cycles in calibration, so the calibration residuals would not be
  truly held out and conformal's exchangeability assumption would break. The split is by `engine_id` with a fixed seed.
- **Preprocessing fitted on train only.** The scaler and PCA are fitted on the training sensors, then applied with
  `.transform()` to the calibration and test data, and saved for inference.
- **RUL capped at 125.** Early in an engine's life degradation is not observable, so labels use the common
  piece-wise linear model `RUL = min(max_cycle − cycle, 125)`. The test ground truth is capped the same way and
  predictions are clipped to `[0, 125]` in the demo.
- **Test set = last cycle per engine.** This is the C-MAPSS convention: `RUL_FD001.txt` holds the true RUL at each test
  engine's final observed cycle.
- **XGBoost over Random Forest.** Random Forest scored marginally better in the grid search, but both reach the same
  ~92% coverage and XGBoost gives slightly narrower intervals and a far smaller train/test gap, so it was chosen. See
  [modeling.md](modeling.md).
- **GP kept out of DVC.** Training it is slow and memory-hungry, so it stays a notebook/script experiment.
