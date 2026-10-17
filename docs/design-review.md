# Design review

What the original code did against common software-design principles, and what changed. "Open" means not fixed.

| # | Problem found | Principle | Fix |
|---|---|---|---|
| 1 | `predict_pipeline.py` called methods and attributes that did not exist (`self.model`, `calibrator.predict_interval`), imported `PCA` from the wrong module and never returned a result | Broken **contract between components** (the caller and callee disagreed); unfinished abstraction | Rewritten against the real classes; `ConformalCalibrator.predict_interval` added; tested |
| 2 | `data_ingestion.py` also ran data transformation when executed | **Single Responsibility** | Script only ingests; transformation stays its own DVC stage |
| 3 | `app.py` held a class with mutable class-level attributes as a global "model store" and stopped before any route | Hidden global state; incomplete module | Completed as a FastAPI app whose pipeline is created in the lifespan hook. **Open:** the pipeline is still a module-level variable |
| 4 | Artifact paths hard-coded in dataclass fields | **Dependency Inversion** (high-level code depends on concrete paths) | `PredictPipelineConfig` reads `ARTIFACTS_DIR` / `MODELS_DIR` or accepts an injected config |
| 5 | Magic number `step = 5` repeated next to a config constant; `LENGTH_SCALES` wrapped twice | **DRY**, constants out of config | Both steps use `DOWNSAMPLING_STEP`; length scales validated |
| 6 | `xgboost_training.py` loaded test data it never used | **YAGNI** / **Interface Segregation** (dependency on unused inputs) | Test set now used to report held-out RMSE/R² |
| 7 | My own Streamlit app re-implemented scaler → PCA → XGBoost → interval and duplicated `FEATURE_COLS`/`RUL_CAP` | **DRY**, single source of truth (introduced by me, then fixed) | App and API both call `PredictPipeline.predict_frame`; same numbers before and after |
| 8 | `requirements.txt` was an unfiltered `pip freeze` with a self-referencing editable install; `setup.py` read it | Environment coupling; mixes runtime and tooling dependencies | Runtime `requirements.txt` + `requirements-dev.txt`, as ranges |
| 9 | `logger.py` crashed on a read-only file system | Robustness at the boundary | Falls back to stderr |
| 10 | Copies of the trained artifacts live in `demo_assets/` because `artifacts/` and `models/` are git-ignored | **DRY** (two copies to keep in sync) | **Open:** documented; refresh `demo_assets/` after retraining |
