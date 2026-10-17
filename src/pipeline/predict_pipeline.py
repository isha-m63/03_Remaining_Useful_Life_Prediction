import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import dill
import numpy as np
import pandas as pd

from configs.data_constants_config import ALPHA, RUL_CAP
from src.components.conformal_calibration import ConformalCalibrator
from src.exception import CustomException
from src.logger import logging

#Non-dropped sensor columns, must match FEATURE_COLS in data_transformation.py
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
SENSOR_COLS  = [f"s{i}" for i in range(1, 22)]
FEATURE_COLS = [c for c in SENSOR_COLS if c not in DROP_SENSORS]


@dataclass
class PredictPipelineConfig:
    """Where the trained artifacts live. Override with ARTIFACTS_DIR / MODELS_DIR
    (e.g. both = "demo_assets" for the bundled, git-tracked copy)."""
    artifacts_dir: Path = field(default_factory=lambda: Path(os.getenv("ARTIFACTS_DIR", "artifacts")))
    models_dir: Path = field(default_factory=lambda: Path(os.getenv("MODELS_DIR", "models")))

    @property
    def preprocessor_path(self) -> Path:
        return self.artifacts_dir / "preprocessor.pkl"

    @property
    def residuals_path(self) -> Path:
        return self.artifacts_dir / "residuals.pkl"

    @property
    def xgboost_model_path(self) -> Path:
        return self.models_dir / "xgboost_model.pkl"

    @property
    def gp_model_path(self) -> Path:
        return self.artifacts_dir / "gp_model.pkl"      #written by gp_training.py


class PredictPipeline:
    def __init__(self, config: PredictPipelineConfig | None = None, load_gp: bool = False):
        self.config = config or PredictPipelineConfig()
        self.gp_model = None

        try:
            #dill can load both dill- and pickle-serialised objects
            with open(self.config.preprocessor_path, "rb") as f:
                self.preprocessor = dill.load(f)
            logging.info("Preprocessor loaded: %s", self.config.preprocessor_path)

            with open(self.config.xgboost_model_path, "rb") as f:
                self.model = dill.load(f)
            logging.info("Model loaded: %s (%s)", self.config.xgboost_model_path, type(self.model).__name__)

            self.calibrator = ConformalCalibrator()
            self.calibrator.config.residuals_file_path = str(self.config.residuals_path)
            self.calibrator.load_residuals()

            #The GP is an optional experiment, not required for serving
            if load_gp and self.config.gp_model_path.exists():
                with open(self.config.gp_model_path, "rb") as f:
                    self.gp_model = dill.load(f)
                logging.info("GP model loaded: %s", self.config.gp_model_path)

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, sensor_readings: list[float], alpha: float = ALPHA) -> dict:
        """sensor_readings: the 14 informative sensor values, in FEATURE_COLS order."""
        try:
            if len(sensor_readings) != len(FEATURE_COLS):
                raise ValueError(f"Expected {len(FEATURE_COLS)} sensor values, got {len(sensor_readings)}.")
            if not 0.0 < alpha < 1.0:
                raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

            #Step 1: raw sensors -> scaled -> PCA
            X_raw = pd.DataFrame([sensor_readings], columns=FEATURE_COLS)
            X_transformed = self.preprocessor.transform(X_raw)
            X_df = pd.DataFrame(X_transformed, columns=[f"PC{i+1}" for i in range(X_transformed.shape[1])])

            #Step 2: point prediction + conformal interval (clipped to the valid RUL range)
            lower_arr, upper_arr = self.calibrator.predict_interval(self.model, X_df, alpha)
            point_pred = float(np.clip(self.model.predict(X_df)[0], 0, RUL_CAP))
            lower = float(np.clip(lower_arr[0], 0, RUL_CAP))
            upper = float(np.clip(upper_arr[0], 0, RUL_CAP))

            logging.info("Prediction = %.1f  interval = [%.1f, %.1f]  width = %.1f",
                         point_pred, lower, upper, upper - lower)

            return {
                "rul_prediction": point_pred,
                "lower_bound": lower,
                "upper_bound": upper,
                "interval_width": upper - lower,
                "confidence_level": 1 - alpha,
            }

        except Exception as e:
            raise CustomException(e, sys)
