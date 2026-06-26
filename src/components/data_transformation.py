import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
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

        except: 
            pass