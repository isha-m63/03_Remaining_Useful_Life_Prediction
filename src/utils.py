import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e: 
        raise CustomException(e, sys)
    


def evaluate_model (X_train, y_train, X_test, y_test, models, param):
    try: 
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            model_name = list(models.keys())[i]

            gs = GridSearchCV(model, para, cv=3, scoring='neg_root_mean_squared_error')
            gs.fit(X_train, y_train)

            # Save every hyperparameter combination tested
            results_df = pd.DataFrame(gs.cv_results_)
            results_df.to_csv(
                os.path.join("artifacts", f"{model_name}_gridsearch_results.csv"),
                index=False
            )

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            report[model_name] = {
                "train_r2":   r2_score(y_train, y_train_pred),
                "test_r2":    r2_score(y_test, y_test_pred),
                "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
                "test_rmse":  np.sqrt(mean_squared_error(y_test, y_test_pred)),
                "train_mae":  mean_absolute_error(y_train, y_train_pred),
                "test_mae":   mean_absolute_error(y_test, y_test_pred),
                "best_params": gs.best_params_,
            }

        return report

    except Exception as e: 
        raise CustomException(e, sys)
    


def get_best_model(report: dict):
    try:
        """
        Rank by test_rmse (asc) → test_mae (asc) → test_r2 (desc).
        Returns (best_model_name, sorted_df_of_all_models).
        """
        rows = []
        for name, metrics in report.items():
            rows.append({
                "model":       name,
                "test_rmse":   metrics["test_rmse"],
                "test_mae":    metrics["test_mae"],
                "test_r2":     metrics["test_r2"],
                "train_rmse":  metrics["train_rmse"],
                "train_mae":   metrics["train_mae"],
                "train_r2":    metrics["train_r2"],
                "best_params": str(metrics["best_params"]),
            })

        df = pd.DataFrame(rows)
        df = df.sort_values(
            by=["test_rmse", "test_mae", "test_r2"],
            ascending=[True, True, False]
        ).reset_index(drop=True)

        return df.iloc[0]["model"], df
    
    except Exception as e: 
        raise CustomException(e, sys)
            
