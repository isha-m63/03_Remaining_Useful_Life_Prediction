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

The `Dockerfile` uses `python:3.11-slim`, installs `libgomp1` (needed by XGBoost) and `requirements-demo.txt`, copies
only `streamlit_app.py` and `demo_assets/`, and runs as a non-root user (uid 1000, as Hugging Face requires).

## Hugging Face Space

```bash
export HF_TOKEN=hf_xxx                 # token with write access
python scripts/deploy_space.py <username>/<space-name>
```

The script creates a public Docker Space and uploads only the `Dockerfile`, `requirements-demo.txt`,
`streamlit_app.py`, `demo_assets/` and `deploy/space_README.md` (as the Space's `README.md`).

## FastAPI server (work in progress)

`app.py` and `Dockerfile.api` are intended to serve predictions over HTTP. Currently `app.py` only defines the model
store and an artifact loader; it has no routes yet. `Dockerfile.api` is a two-stage build on `python:3.11-slim` that
copies `app.py`, `src/`, `artifacts/` and `models/` and runs `uvicorn app:app` on port 8000. Artifact directories can
be overridden with `ARTIFACTS_DIR` and `MODELS_DIR`. See [known-issues.md](known-issues.md).
