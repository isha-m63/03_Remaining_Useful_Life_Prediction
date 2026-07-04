import logging
import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from configs.data_constants_config import ALPHA
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
        Rank by test_rmse (asc) - test_mae (asc) - test_r2 (desc).
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
            

def evaluate_conformal(
    models: dict,           # {model_name: fitted_sklearn_model}
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    alpha: float = ALPHA,
    save_path: str = "artifacts/conformal_results.csv",
) -> pd.DataFrame:
    """
    For each model in `models`:
      1. Compute calibration residuals  Ri = |y_cal_i - y_hat_cal_i|
      2. Compute q_hat = ⌈(1-alpha)(n+1)⌉ / n  quantile of residuals
         (finite-sample corrected quantile, eq. 6/7, Tibshirani notes)
      3. Form test intervals  [y_hat_test - q_hat,  y_hat_test + q_hat]
         (eq. 11, Tibshirani notes)
      4. Record empirical coverage and average width
         (eq. 12, Tibshirani notes)
 
    Returns a DataFrame with one row per model, sorted by avg_width
    among models that meet the coverage guarantee.
    """
    try:
        n = len(y_cal)
        # finite-sample quantile level  (Tibshirani eq. 6 / 7)
        level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
 
        rows = []
        for name, model in models.items():
            logging.info("Conformal evaluation: %s", name)
 
            # --- calibration ---
            y_hat_cal = model.predict(X_cal)
            residuals = np.abs(y_cal - y_hat_cal)
            q_hat = np.quantile(residuals, level)
 
            # --- test intervals (eq. 11) ---
            y_hat_test = model.predict(X_test)
            lower = y_hat_test - q_hat
            upper = y_hat_test + q_hat
 
            # --- metrics (eq. 12) ---
            coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
            avg_width = float(np.mean(upper - lower))
 
            rows.append({
                "model":     name,
                "q_hat":     q_hat,
                "coverage":  coverage,
                "avg_width": avg_width,
                # store arrays for plotting
                "_lower":    lower,
                "_upper":    upper,
                "_point":    y_hat_test,
            })
 
        results_df = pd.DataFrame(rows)
 
        # rank: valid models first (coverage >= 1-alpha), then by width
        nominal = 1 - alpha
        results_df["_valid"] = results_df["coverage"] >= nominal
        results_df = results_df.sort_values(
            ["_valid", "avg_width"], ascending=[False, True]
        ).reset_index(drop=True)
 
        # save the public columns
        public_cols = ["model", "q_hat", "coverage", "avg_width"]
        results_df[public_cols].to_csv(save_path, index=False)
        logging.info("Conformal results saved to %s", save_path)
 
        return results_df
 
    except Exception as e:
        raise CustomException(e, sys)
 