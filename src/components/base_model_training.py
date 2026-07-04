import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.utils import save_object, evaluate_model, get_best_model
from src.logger import logging
from src.exception import CustomException

'''
Models used: Linear Regression, Random Forest Regressor, SVR, XGBoost Regressor
Note: GP training should not be included here.
'''
@dataclass
class ModelTrainerConfig: 
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    top5_results_path: str = os.path.join('artifacts', 'top5_models.csv')



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

            best_model_name, ranking_df = get_best_model(model_report)

            # save top-5 to csv
            top5 = ranking_df.head(5)
            top5.to_csv(self.model_trainer_config.top5_results_path, index=False)
            logging.info("Top-5 models saved to %s", self.model_trainer_config.top5_results_path)

            print("\nTop-5 models (ranked by RMSE → MAE → R²):")
            print(top5.to_string(index=False))

            # --- threshold check on RMSE instead of R² ---
            best_metrics = model_report[best_model_name]
            print(f"\nBest model : {best_model_name}")
            print(f"  test RMSE: {best_metrics['test_rmse']:.4f}")
            print(f"  test MAE : {best_metrics['test_mae']:.4f}")
            print(f"  test R²  : {best_metrics['test_r2']:.4f}")

            if best_metrics["test_r2"] < 0.6:
                raise CustomException("No best model found — R² below 0.6")

            best_model = models[best_model_name]
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            logging.info("Best model saved: %s", best_model_name)

            return best_model_name, best_model, best_metrics


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
 