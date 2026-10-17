# Modeling

## Dataset

NASA C-MAPSS FD001: one operating condition and one fault mode. Training data runs 100 engines to failure
(16,778 rows after dropping the calibration engines, 3,852 calibration rows). The test set has 100 engines that stop
before failure, with the true remaining life given in `RUL_FD001.txt`.

## Preprocessing

1. **Labels.** `RUL = min(max_cycle − cycle, 125)` for each training row.
2. **Sensor selection.** Drop `s1, s5, s6, s10, s16, s18, s19`, leaving 14 informative sensors. The operational
   settings are not used.
3. **Scaling + PCA.** `StandardScaler` then `PCA(n_components=10)`, fitted on training rows only. The EDA notebook
   examines how many components explain 95% of the variance.
4. **Test features.** Last observed cycle of each test engine.

## Model comparison

3-fold grid search on the proper training set, evaluated on the 100 test engines (`artifacts/top5_models.csv`):

| Model | Best params | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|
| Random Forest | `n_estimators=256` | 16.39 | 11.50 | 0.833 |
| **XGBoost** | `learning_rate=0.1, n_estimators=32` | **16.42** | **11.56** | **0.832** |
| SVR | `C=1, kernel=rbf` | 18.88 | 12.90 | 0.778 |
| Linear Regression | – | 20.83 | 16.64 | 0.730 |

Random Forest is marginally ahead on point accuracy, but it overfits more (train RMSE 6.7 vs test 16.4, while XGBoost
is 16.9 vs 16.4), and under conformal prediction it gave slightly wider intervals. XGBoost was chosen.

## Conformal prediction

Split conformal prediction on the calibration set (n = 3,852 rows, 20 engines):

1. Residuals `R_i = |y_i − ŷ_i|` on the calibration set (`artifacts/residuals.pkl`).
2. `q̂` is the `⌈(1−α)(n+1)⌉ / n` quantile of the residuals.
3. Interval `[ŷ − q̂, ŷ + q̂]`.

Results on the 100 test engines, α = 0.1 (target coverage 90%), from `notebooks/pca/03_conformal_results.ipynb`:

| Model | q̂ | Coverage | Avg width | Meets guarantee |
|---|---|---|---|---|
| **XGBoost** | 31.21 | 0.92 | 62.4 | yes |
| Random Forest | 31.74 | 0.92 | 63.5 | yes |
| Linear Regression | 36.37 | 0.94 | 72.7 | yes |
| SVR | 34.80 | 0.89 | 69.6 | no |

The interval has constant width because it uses a single global `q̂`. It does not adapt to how uncertain a particular
engine is. With only 100 test engines, coverage estimates are noisy (±3 points is one standard error).

## Gaussian Process experiment

`pipeline/gp_training.py` and `notebooks/pca/04_gp_results.ipynb` fit a GP with kernel
`10² · Matérn(length_scale=0.5 per PC, ν=2.5) + WhiteKernel`, `alpha=0.01`, on a downsampled training set (every
5th row once there are more than 2,500). Notebook results: RMSE 17.56, MAE 12.22, R² 0.808, 92% coverage, average
width 68.4. It also gives a predictive standard deviation per engine, which the conformal models do not. It is slower
and uses more memory, so it is not part of the DVC pipeline.

## Reproducibility notes

- Calibration split uses seed 42 and 20% of engines; the GP uses `random_state=42`.
- The two tables above come from different runs (grid search vs. notebook refit), so Random Forest and XGBoost RMSE
  differ in the third decimal.
