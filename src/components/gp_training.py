
import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


from src.utils import save_object, evaluate_model
from src.logger import logging
from src.exception import CustomException
