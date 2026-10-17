## 03. Remaining Useful Life Prediction

Predict how many operating cycles a turbofan engine has left (RUL) from its sensor readings, and report an
uncertainty interval with a statistical coverage guarantee using **split conformal prediction**.

- **Data:** NASA C-MAPSS turbofan degradation simulation, subset **FD001** (100 train engines, 100 test engines).
- **Model:** 14 informative sensors → `StandardScaler` → `PCA(10)` → `XGBoost` → RUL, with a conformal interval of
  `prediction ± q̂`.
- **Result (FD001 test set, α = 0.1):** RMSE ≈ 16.4 cycles, R² ≈ 0.83, empirical coverage 92% (target ≥ 90%),
  interval width ≈ 62 cycles.
- **Tooling:** DVC pipeline, MLflow (GP experiment), Streamlit demo (Docker / Streamlit Community Cloud), FastAPI server.

## Documentation

| Doc | What it covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Pipeline design, module responsibilities, data flow, key design decisions |
| [docs/setup.md](docs/setup.md) | Installation, getting the data, running the DVC pipeline and the demo |
| [docs/modeling.md](docs/modeling.md) | Preprocessing, model comparison, conformal prediction, Gaussian Process experiment, results |
| [docs/deployment.md](docs/deployment.md) | Streamlit demo, Docker images, Streamlit Community Cloud, FastAPI server |
| [docs/known-issues.md](docs/known-issues.md) | Unfinished parts and known bugs, so you know what not to rely on |

## Project structure
```
├── app.py                      FastAPI inference server (work in progress)
├── streamlit_app.py            Streamlit demo
├── artifacts                   Processed data, preprocessor, residuals, grid-search results (DVC outputs)
├── configs
│   └── data_constants_config.py   RUL cap, PCA size, alpha, model hyperparameters
├── data
│   ├── processed               CSVs written by ingestion
│   └── raw                     Original C-MAPSS files
├── demo_assets                 Model + data bundled for the Streamlit demo
├── docs                        Project documentation
├── logs
├── models                      Trained model pickles
├── notebooks
│   ├── pca                     EDA, model training, conformal results, GP results
│   └── without_pca             XGBoost and Lasso experiments on the raw sensors
├── dvc.yaml                    Reproducible pipeline definition
├── Dockerfile                  Streamlit demo image
├── Dockerfile.api              FastAPI image (work in progress)
├── setup.py
└── src
    ├── components
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   ├── experiment_splitter.py
    │   ├── base_model_training.py      Grid search over 4 baseline models
    │   ├── conformal_calibration.py
    │   └── conformal_plots.py
    ├── pipeline
    │   ├── xgboost_training.py
    │   ├── gp_training.py
    │   └── predict_pipeline.py         (work in progress)
    ├── exception.py
    ├── logger.py
    └── utils.py
```

## Quick start

```bash
pip install -r requirements-dev.txt   # runtime deps + dvc, mlflow, pytest
dvc repro                              # raw data -> processed -> split -> XGBoost -> conformal results
streamlit run streamlit_app.py         # demo uses the bundled demo_assets/, no training needed
```

See [docs/setup.md](docs/setup.md) for details.

## Live demo (Streamlit, Docker)

An interactive demo lives in `streamlit_app.py` (model + sample data bundled in `demo_assets/`).

```bash
docker build -t rul-demo .
docker run -p 7860:7860 rul-demo      # http://localhost:7860
```

**Streamlit Community Cloud:** point a new app at this repo, branch `main`, file `streamlit_app.py`
(Python 3.11). `requirements.txt` holds only the runtime dependencies.

**REST API:** `Dockerfile.api` serves `POST /predict` and `GET /health` (docs at `/docs`):

```bash
docker build -f Dockerfile.api -t rul-api .
docker run -p 8000:8000 rul-api
```

Run the tests with `pip install -r requirements-dev.txt && pytest`.

## References
- Saxena & Goebel, *Turbofan Engine Degradation Simulation Data Set*, NASA Ames Prognostics Data Repository.
- Shafer & Vovk (2008), *A Tutorial on Conformal Prediction*, JMLR 9.
- Tibshirani (2023), *Conformal Prediction*, Advanced Topics in Statistical Learning.
