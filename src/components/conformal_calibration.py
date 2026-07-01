"""
conformal_calibrator step turns each fitted baseline's point predictions into intervals, using the calibration set.

After the baseline models are fitted using X_train_proper and y_train_proper, the conformal_calibrator step is to take an already fitted 
point predictor and compute the nonconformity quantile on calibration data.

Steps: 
 1. For each baseline, predict on X_cal → compute residuals |y_cal - y_hat_cal|
 2. Store residuals (the quantile is computed per alpha)
 3. At evaluation time: predict_interval(X_test, alpha) wraps the baseline's  point prediction with ± q_hat(alpha)

"""

import os
import sys
import pickle
from dataclasses import dataclass, field

import pandas as pd
 
from src.exception import CustomException
from src.logger import logging

@dataclass
class ConformalCalibratorConfig:
    residuals_file_path: str = os.path.join('artifacts', 'residuals.pkl')
    