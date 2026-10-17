## 03. Remaining Useful Life Prediction

Predict how many operating cycles a turbofan engine has left (RUL) from its sensor readings, and report an
uncertainty interval with a statistical coverage guarantee using **split conformal prediction**.

- **Data:** NASA C-MAPSS turbofan degradation simulation, subset **FD001** (100 train engines, 100 test engines).
- **Model:** 14 informative sensors → `StandardScaler` → `PCA(10)` → `XGBoost` → RUL, with a conformal interval of
  `prediction ± q̂`.
- **Result (FD001 test set, α = 0.1):** RMSE ≈ 16.4 cycles, R² ≈ 0.83, empirical coverage 92% (target ≥ 90%),
  interval width ≈ 62 cycles.
- **Tooling:** DVC pipeline, MLflow (GP experiment), Streamlit demo (Docker / Hugging Face Space), FastAPI server (work in progress).

## Documentation

| Doc | What it covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Pipeline design, module responsibilities, data flow, key design decisions |
| [docs/setup.md](docs/setup.md) | Installation, getting the data, running the DVC pipeline and the demo |
| [docs/modeling.md](docs/modeling.md) | Preprocessing, model comparison, conformal prediction, Gaussian Process experiment, results |
| [docs/deployment.md](docs/deployment.md) | Streamlit demo, Docker images, Hugging Face Space, FastAPI server |
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
├── deploy                      Hugging Face Space README
├── docs                        Project documentation
├── logs
├── models                      Trained model pickles
├── notebooks
│   ├── pca                     EDA, model training, conformal results, GP results
│   └── without_pca             XGBoost and Lasso experiments on the raw sensors
├── scripts/deploy_space.py     Upload the demo to a Hugging Face Space
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
pip install -r requirements.txt
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

Deploy to a Hugging Face Docker Space:

```bash
export HF_TOKEN=hf_xxx
python scripts/deploy_space.py <username>/rul-prediction
```

`Dockerfile.api` is the (work-in-progress) FastAPI inference-server image.

## References
- Saxena & Goebel, *Turbofan Engine Degradation Simulation Data Set*, NASA Ames Prognostics Data Repository.
- Shafer & Vovk (2008), *A Tutorial on Conformal Prediction*, JMLR 9.
- Tibshirani (2023), *Conformal Prediction*, Advanced Topics in Statistical Learning.
