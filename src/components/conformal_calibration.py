
"""
Conformal Calibrator

Transforms a fitted regression model's point predictions into prediction intervals using split conformal prediction.

Workflow

Offline:
    1. Predict on calibration set.
    2. Compute absolute residuals.
    3. Save residuals.

Online:
    1. Load saved residuals.
    2. Compute conformal quantile.
    3. Predict on test/inference data.
    4. Form interval:
            [prediction - q_hat,
             prediction + q_hat]
"""

import os
import sys
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from configs.data_constants_config import ALPHA
from src.exception import CustomException
from src.logger import logging


@dataclass
class ConformalCalibratorConfig:
    model_path: str = os.path.join("models", "xgboost_model.pkl")
    X_cal_path: str = os.path.join("artifacts", "X_cal.csv")
    y_cal_path: str = os.path.join("artifacts", "y_cal.csv")
    X_test_path: str = os.path.join("artifacts", "X_test_processed.csv")
    y_test_path: str = os.path.join("artifacts", "y_test.csv")

    residuals_file_path: str = os.path.join("artifacts", "residuals.pkl")



class ConformalCalibrator:

    def __init__(self):
        self.config = ConformalCalibratorConfig()
        self.residuals: np.ndarray | None = None

    
#Offline calibration
    

    def fit(self,model, X_cal: pd.DataFrame, y_cal: pd.Series,) -> "ConformalCalibrator":
        try:
            logging.info("Calibrating conformal predictor")
            y_pred = model.predict(X_cal)
            self.residuals = np.abs(y_cal - y_pred)
            with open(self.config.residuals_file_path, "wb") as f:
                pickle.dump(self.residuals, f)

            logging.info( "Saved %d residuals to %s", len(self.residuals), self.config.residuals_file_path,)
            return self

        except Exception as e:
            raise CustomException(e, sys)

#Online
    def load_residuals(self) -> "ConformalCalibrator":
        try:
            with open(self.config.residuals_file_path, "rb") as f:
                self.residuals = pickle.load(f)

            logging.info("Loaded %d residuals", len(self.residuals))
            return self

        except Exception as e:
            raise CustomException(e, sys)

    def compute_quantile(self, alpha: float = ALPHA) -> float:
        try:
            if self.residuals is None:
                raise ValueError("Residuals are not loaded. Call fit() or load_residuals() first." )
            
            n = len(self.residuals)
            level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0,)
            q_hat = np.quantile(self.residuals, level)

            logging.info( "Computed q_hat = %.6f",  q_hat)
            return q_hat

        except Exception as e:
            raise CustomException(e, sys)

     #Evaluation
    def evaluate_conformal_intervals( self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                                     alpha: float = ALPHA, save_path: str | None = None) -> pd.DataFrame:

        """
        Evaluate conformal prediction intervals using previously calibrated residuals.
        """
        try:
            logging.info("Evaluating conformal intervals")

            #Load offline residuals
            self.load_residuals()
            q_hat = self.compute_quantile(alpha)
            y_pred = model.predict(X_test)
            lower = y_pred - q_hat
            upper = y_pred + q_hat

            nominal = 1 - alpha

            coverage = np.mean( (y_test >= lower) & (y_test <= upper))
            avg_width = np.mean(upper - lower)

            results = pd.DataFrame(
                [{
                    "q_hat": round(q_hat, 6),
                    "coverage": round(coverage, 4),
                    "avg_width": round(avg_width, 6),
                    "meets_guarantee": coverage >= nominal,
                    "coverage_gap": round(
                        coverage - nominal,
                        4,
                    ),

                    # Hidden plotting columns
                    "_lower": lower,
                    "_upper": upper,
                    "_point": y_pred,
                }]
            )

            logging.info("Coverage = %.4f | Width = %.4f | Valid = %s",coverage, avg_width, coverage >= nominal)

            if save_path is not None:

                public_cols = [
                    "q_hat",
                    "coverage",
                    "avg_width",
                    "meets_guarantee",
                    "coverage_gap",
                ]

                results[public_cols].to_csv(
                    save_path,
                    index=False,
                )

                logging.info("Saved evaluation to %s", save_path)

            return results

        except Exception as e:
            raise CustomException(e, sys)




if __name__ == "__main__":
    config = ConformalCalibratorConfig()

    #Load trained model
    with open(config.model_path, "rb") as f:
        model = pickle.load(f)

    #Load calibration data
    X_cal = pd.read_csv(config.X_cal_path)
    y_cal = pd.read_csv(config.y_cal_path).squeeze("columns")

    #Load test data
    X_test = pd.read_csv(config.X_test_path)
    y_test = pd.read_csv(config.y_test_path).squeeze("columns")

    calibrator = ConformalCalibrator()

    
    #Offline calibration
    calibrator.fit( model=model, X_cal=X_cal, y_cal=y_cal)

   
    #Online evaluation
    results = calibrator.evaluate_conformal_intervals(
        model=model,
        X_test=X_test,
        y_test=y_test,
        alpha=ALPHA,
        save_path="artifacts/conformal_results.csv")

