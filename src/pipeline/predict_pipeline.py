from __future__ import annotations
 
import os
import pickle
import sys
 
import numpy as np
import pandas as pd
 
from configs.data_constants_config import ALPHA
from src.components.conformal_calibration import ConformalCalibrator
from src.components.conformal_plots import generate_prediction_plot
from src.exception import CustomException
from src.logger import logging
 
#Non-dropped sensor columns, must match FEATURE_COLS in data_transformation.py
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
SENSOR_COLS  = [f"s{i}" for i in range(1, 22)]
FEATURE_COLS = [c for c in SENSOR_COLS if c not in DROP_SENSORS]

