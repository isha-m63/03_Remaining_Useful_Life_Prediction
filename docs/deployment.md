# Deployment

## Streamlit demo

`streamlit_app.py` loads `demo_assets/{preprocessor,xgboost_model,residuals}.pkl` plus the FD001 test data and ground
truth. The sidebar has a miscoverage slider (α from 0.02 to 0.30); `q̂` is recomputed live from the saved residuals.

Tabs:
- **Engine explorer.** Pick a test engine and a cycle, and see the predicted RUL, the interval, the true RUL, and the
  prediction trajectory over time.
- **Manual sensor input.** Sliders for the 14 sensors, starting from a real engine's last cycle.
- **Fleet overview.** RMSE, empirical coverage and mean width across all 100 test engines, a predicted vs. true plot, and
  a per-engine table.

## Docker (demo)

```bash
docker build -t rul-demo .
docker run -p 7860:7860 rul-demo       # http://localhost:7860
```

The `Dockerfile` uses `python:3.11-slim`, installs `libgomp1` (needed by XGBoost) and `requirements.txt`, copies
only `streamlit_app.py` and `demo_assets/`, and runs as a non-root user (uid 1000).

## Streamlit Community Cloud

Create an app from this repository: branch `main`, main file `streamlit_app.py`, Python 3.11. It installs
`requirements.txt` and reads the bundled `demo_assets/`; nothing needs to be trained first.

## FastAPI server

`app.py` loads the preprocessor, XGBoost model and calibration residuals once at startup (through
`src/pipeline/predict_pipeline.py`) and exposes:

- `GET /health`
- `POST /predict` with `{"sensor_readings": [14 floats], "alpha": 0.1}` (optional `alpha`), returning
  `rul_prediction`, `lower_bound`, `upper_bound`, `interval_width` and `confidence_level`. The sensor order is listed by
  `/health` and in `FEATURE_COLS`. Interactive docs are at `/docs`.

```bash
docker build -f Dockerfile.api -t rul-api .
docker run -p 8000:8000 rul-api
```

`Dockerfile.api` runs `uvicorn app:app` on port 8000 with `ARTIFACTS_DIR=MODELS_DIR=demo_assets`. Outside Docker the
defaults are `artifacts/` and `models/`. Tests live in `tests/` (`pytest`).
