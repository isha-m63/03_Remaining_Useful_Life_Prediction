import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.utils import save_object, evaluate_model
from src.logger import logging
from src.exception import CustomException
from configs.data_constants_config import LEARNING_RATE, N_ESTIMATORS

@dataclass
class XGBoostModelTrainerConfig: 
    trained_model_file_path = os.path.join('artifacts', 'xgboost_model.pkl')


class XGBoostModelTrainer: 
    def __init__(self):
        self.model_trainer_config = XGBoostModelTrainerConfig()

    def initiate_model_trainer(self, X_train_proper_path: str, y_train_proper_path: str, 
                               X_test_path: str, y_test_path: str):
        try: 
            logging.info('Load train and test data for XGBoost model training')

            X_train = pd.read_csv(X_train_proper_path)
            y_train = pd.read_csv(y_train_proper_path).values.ravel()
            X_test = pd.read_csv(X_test_path)
            y_test = pd.read_csv(y_test_path).values.ravel()
 
            
            logging.info("XGBoostTrainer: training on %d rows, %d features",
                len(X_train), X_train.shape[1])

            #Define the best fit model
            xgboost_model = XGBRegressor(learning_rate = LEARNING_RATE, n_estimators = N_ESTIMATORS)

            logging.info("Training XGBoost model")
            xgboost =  xgboost_model.fit(X_train, y_train)

            logging.info("Training completed")
            
            # Save the trained object as 'gp_model.pkl'
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=xgboost
            )

            logging.info( "XGBoost model saved to %s",self.model_trainer_config.trained_model_file_path)

            return xgboost

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = XGBoostModelTrainer()
    trainer.initiate_model_trainer(
        X_train_proper_path = "artifacts/X_train_proper.csv",
        y_train_proper_path = "artifacts/y_train_proper.csv",
        X_test_path = "artifacts/X_test_processed.csv",
        y_test_path = "artifacts/y_test.csv"
    )