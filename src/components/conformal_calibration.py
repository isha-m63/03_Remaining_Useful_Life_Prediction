"""
conformal_calibrator step turns each fitted baseline's point predictions into intervals, using the calibration set.

After the baseline models are fitted using X_train_proper and y_train_proper, the conformal_calibrator step is to take an already fitted 
point predictor and compute the nonconformity quantile on calibration data.

Steps: 
 1. For each baseline, predict on X_cal → compute residuals |y_cal - y_hat_cal|
 2. Store residuals (the quantile is computed per ALPHA)
 3. At evaluation time: predict_interval(X_test, ALPHA) wraps the baseline's  point prediction with ± q_hat(ALPHA)

"""

import os
import sys
import pickle
import numpy as np
from dataclasses import dataclass, field

import pandas as pd

from configs.data_constants_config import ALPHA
from src.exception import CustomException
from src.logger import logging

@dataclass
class ConformalCalibratorConfig:
    residuals_file_path: str = os.path.join('artifacts', 'residuals.pkl')
    #ALPHA: float = 0.1      #Gives 90% prediction intervals by default, can be changed at evaluation time


class ConformalCalibrator:
    def __init__(self):
        self.config = ConformalCalibratorConfig()

    def fit(self, model, X_cal: pd.DataFrame, y_cal: pd.Series):
        try:
            logging.info("Fitting conformal calibrator...")
            y_hat_cal = model.predict(X_cal)
            residuals = abs(y_cal - y_hat_cal)
            with open(self.config.residuals_file_path, 'wb') as f:
                pickle.dump(residuals, f)
            logging.info("Conformal calibrator fitted and residuals saved")
        except Exception as e:
            raise CustomException(e, sys)

    def predict_interval(self, model, X_test: pd.DataFrame, ALPHA: float):
        try:
            logging.info("Predicting intervals using conformal calibrator")
            with open(self.config.residuals_file_path, 'rb') as f:
                residuals = pickle.load(f)
            n = len(residuals)
            q_hat = np.quantile(residuals, np.ceil((1 - ALPHA) * (n + 1)) / n)
            y_hat_test = model.predict(X_test)
            lower_bound = y_hat_test - q_hat
            upper_bound = y_hat_test + q_hat
            return lower_bound, upper_bound
        except Exception as e:
            raise CustomException(e, sys)
    