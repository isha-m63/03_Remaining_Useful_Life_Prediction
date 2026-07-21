import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from configs.data_constants_config import LENGTH_SCALES, NU, DOWNSAMPLING_STEP
from src.utils import save_object, evaluate_model
from src.logger import logging
from src.exception import CustomException

@dataclass
class GPModelTrainerConfig: 
    trained_model_file_path = os.path.join('artifacts', 'gp_model.pkl')


class GPModelTrainer: 
    def __init__(self):
        self.model_trainer_config = GPModelTrainerConfig()

    def initiate_model_trainer(self, X_train_proper_path: str, y_train_proper_path: str, 
                               X_test_path: str, y_test_path: str):
        try: 
            logging.info('Load train and test data for Gaussian Process tuning')

            X_train = pd.read_csv(X_train_proper_path)
            y_train = pd.read_csv(y_train_proper_path).values.ravel()
            X_test = pd.read_csv(X_test_path)
            y_test = pd.read_csv(y_test_path).values.ravel()
 
            logging.info("Original data size: %d rows", len(X_train))

            X_train_gp = X_train.copy()
            y_train_gp = y_train.copy()

            X_test_gp = X_test.copy()
            y_test_gp = y_test.copy()

            #Downsampling to prevent OOM (Killed) crash for large datasets
            if len(X_train_gp) > 2500:
                step = 5
                X_train_gp = X_train_gp.iloc[::step].reset_index(drop=True)
                y_train_gp = y_train_gp[::step]

            if len(X_test_gp) > 1000:
                step = DOWNSAMPLING_STEP
                X_test_gp = X_test_gp.iloc[::step].reset_index(drop=True)
                y_test_gp = y_test_gp[::step]

            logging.info("GPRegressorTrainer: training on %d rows, %d features",
                len(X_train_gp), X_train_gp.shape[1])

            # Tuned kernel (replace values below with your notebook's best values if different)
            length_scales = [LENGTH_SCALES] * X_train_gp.shape[1]

            kernel = (
                ConstantKernel(constant_value=100.0)
                * Matern(length_scale = length_scales, nu = NU)
                + WhiteKernel(
                    noise_level=1.0,
                    noise_level_bounds=(1e-3, 1e3),
                )
            )

            gp_model = GaussianProcessRegressor(
                kernel = kernel,
                alpha = 0.01,
                normalize_y = True,
                n_restarts_optimizer = 5,
                random_state = 42,
            )

            logging.info("Training Gaussian Process model")
            gp =  gp_model.fit(X_train_gp, y_train_gp)

            logging.info("Training completed")
            
            # Save the trained object as 'gp_model.pkl'
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=gp
            )

            logging.info( "Gaussian Process model saved to %s",self.model_trainer_config.trained_model_file_path)

            return gp

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = GPModelTrainer()
    trainer.initiate_model_trainer(
        X_train_proper_path = "artifacts/X_train_proper.csv",
        y_train_proper_path = "artifacts/y_train_proper.csv",
        X_test_path = "artifacts/X_test_processed.csv",
        y_test_path = "artifacts/y_test.csv"
    )