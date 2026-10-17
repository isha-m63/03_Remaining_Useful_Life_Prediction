# Setup

## Requirements
- Python 3.10 (development) or 3.11 (Docker images)
- `pip`; Docker if you want to run the containers
- Optionally DVC (already in `requirements.txt`)

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is a full `pip freeze` and its first line is an editable install of this repo from GitHub. If that
line causes trouble, delete it and run `pip install -e .` instead. For the demo alone,
`pip install -r requirements-demo.txt` is enough.

## Data

The raw C-MAPSS files (`train_FD00x.txt`, `test_FD00x.txt`, `RUL_FD00x.txt`) belong in `data/raw/`. Only **FD001** is
used by the pipeline. The folder also contains the dataset's `readme.txt` and the *Damage Propagation Modeling* paper.

## Run the pipeline

```bash
dvc repro            # all stages in dvc.yaml
```

Or run stages by hand from the repo root (modules import `src.*` and `configs.*`):

```bash
python src/components/data_ingestion.py        # also triggers data transformation
python src/components/data_transformation.py
python src/components/experiment_splitter.py
python src/components/base_model_training.py   # optional: grid search of 4 baseline models
python src/pipeline/xgboost_training.py
python src/components/conformal_calibration.py
python src/pipeline/gp_training.py             # optional, slow, not in DVC
```

Outputs: `artifacts/` (data, preprocessor, residuals, `conformal_results.csv`) and `models/xgboost_model.pkl`.
Logs go to `logs/<timestamp>.log`. DVC stages are listed in [architecture.md](architecture.md).

## Run the demo

```bash
streamlit run streamlit_app.py
```

The demo reads everything from `demo_assets/` (preprocessor, XGBoost model, residuals, test data) and does not need the
pipeline to have run. If you retrain, copy the new `preprocessor.pkl`, `xgboost_model.pkl` and `residuals.pkl` there.

## Notebooks

| Notebook | Purpose |
|---|---|
| `pca/01_eda.ipynb` | Data checks, distributions, PCA components needed for 95% variance |
| `pca/02_initial_model_training.ipynb` | PCs vs RUL, baseline model metrics |
| `pca/03_conformal_results.ipynb` | Conformal intervals for the top models, coverage vs width |
| `pca/04_gp_results.ipynb` | Gaussian Process point predictions and intervals |
| `without_pca/01_xgboost_training.ipynb`, `02_lasso_reg_training.ipynb` | Experiments on scaled raw sensors |
