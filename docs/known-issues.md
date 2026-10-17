# Known issues and unfinished work

Found while documenting the code. Nothing below has been changed.

## Inference path
- `src/pipeline/predict_pipeline.py` does not run:
  - it imports `PCA` from `sklearn.preprocessing` (it lives in `sklearn.decomposition`);
  - it calls `.transform` on `self.config.preprocessor_path`, which is a string, not the loaded preprocessor;
  - it uses `self.model` and `calibrator.predict_interval(...)`, neither of which exists (the model is
    `self.xgboost_model` and `ConformalCalibrator` has no such method);
  - `predict()` ends before returning a result;
  - it loads `models/gp_model.pkl`, but `gp_training.py` saves to `artifacts/gp_model.pkl`.
- `app.py` stops after loading the preprocessor and model: no `q_hat` calculation, request schemas or routes.
- `src/components/conformal_plots.py` is used by the unfinished API path only.

## Pipeline
- `base_model_training.py` has an `#INCOMPLETE HERE` marker left in the code. The best model is saved to
  `artifacts/model.pkl`, but the pipeline's final model is the separately trained `models/xgboost_model.pkl`.
- `xgboost_training.py` loads the test data but never uses it.
- `gp_training.py` hard-codes the training downsampling step (5) while the test step comes from config. Also
  `LENGTH_SCALES` is already a list and is wrapped in another list.
- `data_ingestion.py` runs data transformation when executed as a script, in addition to the separate DVC
  `data_transformation` stage.
- `dvc.yaml` covers XGBoost and conformal stages only; the baseline grid search and GP training are manual.
- MLflow is only used in `notebooks/pca/04_gp_results.ipynb` (tracking data in `mlflow.db`); the `src/` pipeline does not
  log to it. `.dvc/config` defines no remote, so DVC-tracked data is local only.

## Statistics
- Coverage is measured on 100 test engines, so it is a noisy estimate.
- Intervals have a single constant width, regardless of the engine's state.
- The conformal calibration set has correlated rows (many cycles per engine), so the finite-sample guarantee is
  approximate rather than exact.

## Housekeeping
- `requirements.txt` is an unfiltered `pip freeze` with a self-referencing editable install on line 1.
- The environment folder `.venv/`, `logs/` and `mlflow.db` live in the project directory.
