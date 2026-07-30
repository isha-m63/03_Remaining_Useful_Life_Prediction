import os
import pickle
import sys
from dataclasses import dataclass 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PCA
from configs.data_constants_config import ALPHA
from src.components.conformal_calibration import ConformalCalibrator
from src.components.conformal_plots import ConformalPlots
from src.exception import CustomException
from src.logger import logging
 
#Non-dropped sensor columns, must match FEATURE_COLS in data_transformation.py
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
SENSOR_COLS  = [f"s{i}" for i in range(1, 22)]
FEATURE_COLS = [c for c in SENSOR_COLS if c not in DROP_SENSORS]

@dataclass
class PredictPipelineConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
    xgboost_model_path: str = os.path.join("models", "xgboost_model.pkl")
    gp_model_path: str = os.path.join("models", "gp_model.pkl")


class PredictPipeline:
    def __init__(self):
        self.config = PredictPipelineConfig()

        try: 
            with open(self.config.preprocessor_path, "rb") as f:
                self.preprocessor = pickle.load(f)
            logging.info("Preprocessor loaded: %s", self.config.preprocessor_path)

            with open(self.config.xgboost_model_path, "rb") as f:
                self.xgboost_model = pickle.load(f)
            self.model_name = type(self.xgboost_model).__name__
            logging.info("XGBoost Model loaded:  %s (%s)", self.config.xgboost_model_path, self.model_name)
 
            self.calibrator = ConformalCalibrator()
            self.calibrator.load_residuals()
            logging.info("Residuals loaded.")

            with open(self.config.gp_model_path, "rb") as f:
                    self.gp_model = pickle.load(f)
            self.model_name = type(self.gp_model).__name__
            logging.info("GP Model loaded:  %s (%s)", self.config.gp_model_path, self.model_name)
 
        except Exception as e:
            raise CustomException(e, sys)


    def predict(self, sensor_readings: list[float], alpha: float = ALPHA) -> dict:
        try:
            if len(sensor_readings) != len(FEATURE_COLS):
                raise ValueError(f"Expected {len(FEATURE_COLS)} sensor values, got {len(sensor_readings)}.")
 
            # Step 1: raw sensors - scaled - PCA
            X_raw = pd.DataFrame([sensor_readings], columns=FEATURE_COLS)
            X_transformed = self.config.preprocessor_path.transform(X_raw)
            X_df = pd.DataFrame(X_transformed, columns=[f"PC{i+1}" for i in range(X_transformed.shape[1])])
 
            # Step 2: point prediction + conformal interval
            lower_arr, upper_arr = self.calibrator.predict_interval(self.model, X_df, alpha)
            point_pred = float(self.model.predict(X_df)[0])
            lower = float(lower_arr[0])
            upper = float(upper_arr[0])
 
            logging.info(
                "Prediction = %.1f  interval = [%.1f, %.1f]  width = %.1f",
                point_pred, lower, upper, upper - lower,
            )
 