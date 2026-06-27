import sys
import os
import pickle
from dataclasses import dataclass
 
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from configs.data_constants_config import RUL_CAP, PCA_COMPONENETS
 
from src.exception import CustomException
from src.logger import logging

SENSOR_COLS  = [f"s{i}" for i in range(1, 22)]
FEATURE_COLS = [c for c in SENSOR_COLS]  # 14 sensors

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str  = os.path.join('artifacts', 'preprocessor.pkl')
    X_train_path: str = os.path.join("artifacts", "X_train.csv")
    y_train_path: str = os.path.join("artifacts", "y_train.csv")
    X_test_path:  str = os.path.join("artifacts", "X_test.csv")
    y_test_path:  str = os.path.join("artifacts", "y_test.csv")

class DataTransformation:
    """
    Steps:
      1. Add piecewise-linear RUL labels to the training set.
      2. Fit StandardScaler + PCA on training sensor rows only.
      3. Transform train and test sets using those fitted objects.
      4. Save the fitted preprocessor to artifacts/ for inference reuse.
 
    What this class does NOT do:
      - Split train into train_proper / calibration. That split only exists
        because of conformal prediction — it lives in ExperimentSplitter.
      - Choose PCA n_components. That belongs in config.
      - Train any model.
    """

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try: 
            transform_pipeline = Pipeline(
                steps = [
                    ('scaler', StandardScaler()), 
                    ('pca', PCA())
                ]
            )
            logging.info('Preprocessing complete - sensor columns are scaled and PCA transformed')
            return transform_pipeline

        except Exception as e: 
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try: 
            train_df = pd.read_csv(train_path)
            test_path = pd.read_csv(test_path)
            logging.info('Reading training and testing data is complete')

            logging.info('Obtaining preprocessing object')
            preprocessing_object = self.get_data_transformer_object()
            target_column_name = 'RUL'

        except Exception as e: 
            raise CustomException(e, sys)
        