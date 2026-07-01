import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.utils import save_object, evaluate_model
from src.logger import logging
from src.exception import CustomException

'''
Models used: Linear Regression, Random Forest Regressor, SVR, XGBoost Regressor
Note: GP training should not be included here.
'''
@dataclass
class ModelTrainerConfig: 
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer: 
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train_proper_path: str, y_train_proper_path: str, 
                               X_test_path: str, y_test_path: str):
        try: 
            logging.info('Load train and test data')

            X_train = pd.read_csv(X_train_proper_path)
            y_train = pd.read_csv(y_train_proper_path).values.ravel()
            X_test = pd.read_csv(X_test_path)
            y_test = pd.read_csv(y_test_path).values.ravel()
 
            logging.info(
                "BaselineTrainer: training on %d rows, %d features",
                len(X_train), X_train.shape[1],
            )

            #X_train, y_train, X_test, y_test = #INCOMPLETE HERE

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "SVR": SVR(),
                "XGBRegressor": XGBRegressor(),
            }

            params = {
                "Linear Regression":{},
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "SVR":{
                    'kernel':['linear','poly','rbf','sigmoid'],
                    'C':[0.1,0.5,1, 2, 3],
                    'degree':[2,3,4],
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
               
            }


            model_report: dict = evaluate_model (
                X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, param = params
            ) 

            print("Model Report:")
            print(model_report)

            #Get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            #Get best model name from dictionary key
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(f"Best Model: {best_model_name}")
            print(f"Best Test R sqaure: {best_model_score:.4f}")

            if best_model_score < 0.6: 
                raise CustomException ("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset")

            '''
            #Load preprocessing file if new data is coming
            preprocessor_path = #INCOMPLETE HERE
            
            '''
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path, 
                obj = best_model
            )

            #Predicted output for test data
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square


        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_model_trainer(
        X_train_proper_path = "artifacts/X_train_proper.csv",
        y_train_proper_path = "artifacts/y_train_proper.csv",
        X_test_path = "artifacts/X_test_processed.csv",
        y_test_path = "artifacts/y_test.csv",
        #preprocessor_path = "artifacts/preprocessor.pkl"
    )
 