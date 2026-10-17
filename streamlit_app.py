"""
Streamlit demo: Remaining Useful Life (RUL) prediction for turbofan engines
(NASA C-MAPSS FD001) with split-conformal prediction intervals.

Pipeline: 14 sensors -> StandardScaler -> PCA(10) -> XGBoost -> RUL
Interval: point prediction +/- conformal quantile of calibration residuals.
"""
import os
from pathlib import Path

# The bundled, git-tracked copy of the trained artifacts (set before the pipeline reads its config)
ASSETS = Path(__file__).parent / "demo_assets"
os.environ.setdefault("ARTIFACTS_DIR", str(ASSETS))
os.environ.setdefault("MODELS_DIR", str(ASSETS))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from configs.data_constants_config import RUL_CAP
from src.pipeline.predict_pipeline import FEATURE_COLS, PredictPipeline

st.set_page_config(page_title="Turbofan RUL Prediction", page_icon="✈️", layout="wide")


@st.cache_resource
def load_pipeline() -> PredictPipeline:
    return PredictPipeline()


@st.cache_data
def load_data():
    test = pd.read_csv(ASSETS / "test_set_001.csv")
    rul = pd.read_csv(ASSETS / "RUL_FD001.txt", header=None, names=["RUL"])["RUL"]
    return test, rul


def predict(X_raw: pd.DataFrame) -> np.ndarray:
    return load_pipeline().predict_frame(X_raw)["rul_prediction"].to_numpy()


pipeline = load_pipeline()
test_df, rul_final = load_data()

st.title("✈️ Turbofan Engine — Remaining Useful Life")
st.caption(
    "NASA C-MAPSS FD001 · StandardScaler → PCA(10) → XGBoost · "
    "split-conformal prediction intervals"
)

alpha = st.sidebar.slider(
    "Mis-coverage α", 0.02, 0.30, 0.10, 0.01,
    help="The interval is built to contain the true RUL at least (1 − α) of the time.",
)
q_hat = pipeline.conformal_quantile(alpha)
st.sidebar.metric("Target coverage", f"{(1 - alpha):.0%}")
st.sidebar.metric("Interval half-width (cycles)", f"±{q_hat:.1f}")
st.sidebar.markdown(
    "RUL labels are capped at 125 cycles (piece-wise linear degradation model), "
    "so predictions are clipped to [0, 125]."
)

tab_engine, tab_manual, tab_fleet = st.tabs(["Engine explorer", "Manual sensor input", "Fleet overview"])

# ---------------------------------------------------------------- Engine explorer
with tab_engine:
    engine_ids = sorted(test_df["engine_id"].unique())
    c1, c2 = st.columns([1, 3])
    engine = c1.selectbox("Test engine", engine_ids, index=0)
    eng_df = test_df[test_df["engine_id"] == engine].sort_values("cycle").reset_index(drop=True)
    max_cycle = int(eng_df["cycle"].max())
    obs_cycle = c1.slider("Observed up to cycle", int(eng_df["cycle"].min()), max_cycle, max_cycle)

    preds = predict(eng_df)
    true_final = float(min(rul_final.iloc[engine - 1], RUL_CAP))
    true_curve = np.minimum(rul_final.iloc[engine - 1] + (max_cycle - eng_df["cycle"].to_numpy()), RUL_CAP)

    idx = int((eng_df["cycle"] <= obs_cycle).sum() - 1)
    p, lo, hi = preds[idx], max(preds[idx] - q_hat, 0), min(preds[idx] + q_hat, RUL_CAP)
    true_now = true_curve[idx]

    c1.metric("Predicted RUL (cycles)", f"{p:.0f}")
    c1.metric(f"{1 - alpha:.0%} interval", f"[{lo:.0f}, {hi:.0f}]")
    c1.metric("True RUL (ground truth)", f"{true_now:.0f}", delta=f"{p - true_now:+.0f} error", delta_color="off")
    if lo <= true_now <= hi:
        c1.success("True RUL inside interval")
    else:
        c1.warning("True RUL outside interval")

    fig, ax = plt.subplots(figsize=(8, 4))
    cyc = eng_df["cycle"].to_numpy()
    ax.fill_between(cyc, np.clip(preds - q_hat, 0, RUL_CAP), np.clip(preds + q_hat, 0, RUL_CAP),
                    alpha=0.2, label=f"{1 - alpha:.0%} conformal interval")
    ax.plot(cyc, preds, label="Predicted RUL")
    ax.plot(cyc, true_curve, "k--", label="True RUL")
    ax.axvline(obs_cycle, color="red", lw=1, ls=":")
    ax.set_xlabel("Cycle"); ax.set_ylabel("RUL (cycles)"); ax.set_title(f"Engine {engine}")
    ax.legend(); ax.grid(alpha=0.3)
    c2.pyplot(fig)
    c2.caption("The red line marks the point in time at which the prediction on the left is made.")

# ---------------------------------------------------------------- Manual input
with tab_manual:
    st.write("Adjust the 14 informative sensor readings (starting from a real engine's last cycle) and see the prediction update.")
    seed_engine = st.selectbox("Start from engine", engine_ids, index=0, key="seed")
    seed = test_df[test_df["engine_id"] == seed_engine].sort_values("cycle").iloc[-1]
    cols = st.columns(4)
    vals = {}
    for i, s in enumerate(FEATURE_COLS):
        lo_s, hi_s = float(test_df[s].min()), float(test_df[s].max())
        spread = max(hi_s - lo_s, 1e-6)
        vals[s] = cols[i % 4].slider(
            s, lo_s - 0.1 * spread, hi_s + 0.1 * spread, float(seed[s]),
            step=spread / 200, key=f"{seed_engine}_{s}", format="%.4f",
        )
    p = predict(pd.DataFrame([vals]))[0]
    m1, m2 = st.columns(2)
    m1.metric("Predicted RUL (cycles)", f"{p:.0f}")
    m2.metric(f"{1 - alpha:.0%} interval", f"[{max(p - q_hat, 0):.0f}, {min(p + q_hat, RUL_CAP):.0f}]")

# ---------------------------------------------------------------- Fleet overview
with tab_fleet:
    last = test_df.sort_values("cycle").groupby("engine_id").tail(1).sort_values("engine_id")
    yhat = predict(last)
    y = np.minimum(rul_final.to_numpy(), RUL_CAP)
    inside = (y >= yhat - q_hat) & (y <= yhat + q_hat)
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    k1, k2, k3 = st.columns(3)
    k1.metric("RMSE on 100 test engines", f"{rmse:.1f} cycles")
    k2.metric("Empirical coverage", f"{inside.mean():.0%}", help=f"Target ≥ {1 - alpha:.0%}")
    k3.metric("Mean interval width", f"{2 * q_hat:.1f} cycles")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(y, yhat, yerr=q_hat, fmt="o", ms=3, alpha=0.6, elinewidth=0.8)
    ax.plot([0, RUL_CAP], [0, RUL_CAP], "k--", lw=1)
    ax.set_xlabel("True RUL"); ax.set_ylabel("Predicted RUL"); ax.grid(alpha=0.3)
    st.pyplot(fig)
    table = pd.DataFrame({"engine": last["engine_id"].to_numpy(), "cycles observed": last["cycle"].to_numpy(),
                          "true RUL": y, "predicted": yhat.round(1), "in interval": inside})
    st.dataframe(table, use_container_width=True, hide_index=True)
