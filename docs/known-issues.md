# Known issues and remaining limitations

## Fixed
- **Inference path:** `src/pipeline/predict_pipeline.py` now runs: correct imports, loads the preprocessor/model/residuals
  from `ARTIFACTS_DIR` / `MODELS_DIR`, returns the prediction with its interval, and treats the GP as optional.
  `ConformalCalibrator.predict_interval()` was added.
- **`app.py`:** completed as a FastAPI server (`/health`, `/predict`) and covered by `tests/test_inference.py`.
  `Dockerfile.api` now copies `configs/` and uses the git-tracked `demo_assets/`.
- `base_model_training.py`: removed the stray `#INCOMPLETE HERE` marker.
- `xgboost_training.py` now logs held-out test RMSE / R² (the test set is still never used for fitting).
- `gp_training.py`: both downsampling steps come from `DOWNSAMPLING_STEP`; `LENGTH_SCALES` is no longer double-wrapped.
- `data_ingestion.py` no longer runs data transformation when executed as a script (DVC has its own stage for it).
- `requirements.txt` is now a short list of runtime dependencies (ranges, no self-referencing install);
  training extras are in `requirements-dev.txt`.

## Still open
- `dvc.yaml` covers XGBoost and conformal stages only; the baseline grid search and GP training are manual. The grid
  search saves its winner to `artifacts/model.pkl`, but the pipeline's final model is the separately trained
  `models/xgboost_model.pkl`.
- MLflow is only used in `notebooks/pca/04_gp_results.ipynb` (tracking data in `mlflow.db`); `.dvc/config` defines no
  remote, so DVC-tracked data is local only.
- `src/components/conformal_plots.py` is not used by the serving path.
- `demo_assets/` is a copy of the trained artifacts (`artifacts/`, `models/` are git-ignored), so it must be refreshed
  after retraining.
- `.venv/`, `logs/` and `mlflow.db` live in the project directory (all git-ignored).

## Statistical caveats (inherent to the method)
- Coverage is measured on 100 test engines, so it is a noisy estimate.
- Intervals have a single constant width, regardless of the engine's state.
- The conformal calibration set has correlated rows (many cycles per engine), so the finite-sample guarantee is
  approximate rather than exact.
