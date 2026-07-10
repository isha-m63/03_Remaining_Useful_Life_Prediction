import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

from src.utils import save_object, evaluate_model, get_best_model
from src.logger import logging
from src.exception import CustomException

@dataclass
class GPModelTrainerConfig: 
    trained_model_file_path = os.path.join('artifacts', 'gp_model.pkl')
    top5_results_path: str = os.path.join('artifacts', 'top2_gp_models.csv')


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
 
            logging.info(
                "GPRegressorTrainer: training on %d rows, %d features",
                len(X_train), X_train.shape[1],
            )

            # We use GaussianProcessRegressor as the base model
            models = {
                "Gaussian Process": GaussianProcessRegressor(normalize_y=True, random_state=42)
            }

            # Hyperparameter space mapping kernel choices and regularization (alpha)
            params = {
                "Gaussian Process": {
                    'kernel': [
                        RBF(length_scale=1.0, length_scale_bounds=(0.1, 10)),
                        RBF(length_scale=2, length_scale_bounds=(0.1, 10)),
                        Matern(length_scale=1.0, length_scale_bounds=(0.1, 10), nu=2.5), 
                        Matern(length_scale=2, length_scale_bounds=(0.1, 10), nu=2.5)
                    ],
                    'alpha': [0.01, 0.05, 0.1, 0.5, 1.0] # Value added to the diagonal of the kernel matrix during fitting
                }
            }

            # Utilizing your existing utils structure for evaluation
            model_report: dict = evaluate_model(
                X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, param = params
            ) 

            best_model_name, ranking_df = get_best_model(model_report)

            # Save top-2 iterations to csv (or as many as evaluated if total combos < 5)
            top2 = ranking_df.head(2)
            top2.to_csv(self.model_trainer_config.top2_results_path, index=False)
            logging.info("Top 2 GP configs saved to %s", self.model_trainer_config.top2_results_path)

            print("\nTop configurations (ranked by RMSE - MAE - R2):")
            print(top2.to_string(index=False))

            #Threshold check
            best_metrics = model_report[best_model_name]
            print(f"\nBest model configuration: {best_model_name}")
            print(f"  test RMSE: {best_metrics['test_rmse']:.4f}")
            print(f"  test MAE : {best_metrics['test_mae']:.4f}")
            print(f"  test R2  : {best_metrics['test_r2']:.4f}")

            if best_metrics["test_r2"] < 0.6:
                raise CustomException("No suitable GP model configuration found - R2 below 0.6")

            best_model = models[best_model_name]
            
            # Save the trained object as 'gp_model.pkl'
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            logging.info("Best GP model saved successfully.")

            return best_model_name, best_model, best_metrics

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