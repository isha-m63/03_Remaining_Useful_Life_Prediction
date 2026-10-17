"""Inference-path tests against the bundled, git-tracked artifacts in demo_assets/."""
import os

os.environ["ARTIFACTS_DIR"] = "demo_assets"
os.environ["MODELS_DIR"] = "demo_assets"

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from src.pipeline.predict_pipeline import FEATURE_COLS, PredictPipeline


@pytest.fixture(scope="module")
def last_cycle_row():
    df = pd.read_csv("demo_assets/test_set_001.csv")
    return df[df.engine_id == 1].sort_values("cycle").iloc[-1][FEATURE_COLS].tolist()


def test_predict_returns_valid_interval(last_cycle_row):
    out = PredictPipeline().predict(last_cycle_row)
    assert 0 <= out["lower_bound"] <= out["rul_prediction"] <= out["upper_bound"] <= 125
    assert out["confidence_level"] == pytest.approx(0.9)


def test_smaller_alpha_gives_wider_interval(last_cycle_row):
    p = PredictPipeline()
    q = {a: p.calibrator.compute_quantile(a) for a in (0.05, 0.2)}
    assert q[0.05] > q[0.2]


def test_wrong_number_of_sensors_is_rejected():
    with pytest.raises(Exception):
        PredictPipeline().predict([0.0] * 3)


def test_api_roundtrip(last_cycle_row):
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/predict", json={"sensor_readings": last_cycle_row})
        assert r.status_code == 200 and "rul_prediction" in r.json()
        assert client.post("/predict", json={"sensor_readings": [1.0]}).status_code == 422
