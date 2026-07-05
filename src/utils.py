import logging
import os
import sys
import numpy as np
import pandas as pd
import dill
import matplotlib.pyplot as plt
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
            #residuals = Ri = max(f_lower(xi) - yi,  yi - f_upper(xi))
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
 
#Check the function below
def plot_conformal_results(
    results_df: pd.DataFrame,
    y_test: np.ndarray,
    alpha: float = ALPHA,
    out_dir: str = "artifacts",
    max_points: int = 80,
):
    """
    Three plots grounded in the two metrics from the papers:
 
    Plot 1 – Error-bar panels (one per model)
             Shows point predictions ± interval, coloured by coverage hit/miss.
             Directly illustrates validity (eq. 12, Tibshirani).
 
    Plot 2 – Coverage vs nominal level bar chart
             Each bar shows empirical coverage; red dashed line = 1-alpha target.
             A bar below the line means the guarantee is violated.
 
    Plot 3 – Width vs Coverage scatter
             The core efficiency/validity tradeoff discussed in Section 2.1
             (Tibshirani): valid models sit to the right of the nominal line;
             among those, lower width = better.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        nominal   = 1 - alpha
        model_names = results_df["model"].tolist()
        n_models  = len(model_names)
        colors    = plt.cm.tab10(np.linspace(0, 0.6, n_models))
 
        # subsample for readable error bars
        sort_idx = np.argsort(y_test)
        step     = max(1, len(y_test) // max_points)
        idx      = sort_idx[::step]
        y_sub    = y_test[idx]
 
        #Error-bars
        fig, axes = plt.subplots(
            n_models, 1, figsize=(13, 3.5 * n_models), sharex=False
        )
        if n_models == 1:
            axes = [axes]
 
        for ax, (_, row), color in zip(axes, results_df.iterrows(), colors):
            name = row["model"]
            lo   = row["_lower"][idx]
            hi   = row["_upper"][idx]
            pt   = row["_point"][idx]
            covered = (y_sub >= lo) & (y_sub <= hi)
 
            for xi, (l, h, cov) in enumerate(zip(lo, hi, covered)):
                ax.plot(
                    [xi, xi], [l, h],
                    color="green" if cov else "red",
                    alpha=0.5, linewidth=1.2,
                )
 
            ax.scatter(range(len(idx)), y_sub, s=18, color="black",
                       zorder=3, label="y_true")
            ax.scatter(range(len(idx)), pt, s=10, color=color,
                       zorder=4, alpha=0.8, label="ŷ")
            ax.set_title(
                f"{name}  |  coverage={row['coverage']:.3f}  "
                f"(nominal={nominal:.2f})  avg_width={row['avg_width']:.3f}",
                fontsize=10,
            )
            ax.set_ylabel("y")
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True, alpha=0.3)
 
        axes[-1].set_xlabel("Test samples (sorted by y_true)")
        fig.suptitle(
            f"Conformal prediction intervals  (alpha={alpha}, nominal={nominal:.0%})\n"
            "Green = covered, Red = missed",
            fontsize=12, y=1.01,
        )
        plt.tight_layout()
        p1 = os.path.join(out_dir, "conformal_plot1_errorbars.png")
        plt.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close()
        logging.info("Saved %s", p1)
 
        #Coverage bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        coverages = results_df["coverage"].values
        bars = ax.bar(model_names, coverages, color=colors,
                      alpha=0.8, edgecolor="black")
        ax.axhline(nominal, color="red", linestyle="--", linewidth=1.5,
                   label=f"Nominal {nominal:.0%}  (1 - alpha)")
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Empirical coverage")
        ax.set_title(
            "Empirical coverage per model\n"
            "(bars below red line violate the coverage guarantee)"
        )
        ax.legend()
        for bar, val in zip(bars, coverages):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        p2 = os.path.join(out_dir, "conformal_plot2_coverage.png")
        plt.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close()
        logging.info("Saved %s", p2)
 
        # Width vs coverage scatter
        fig, ax = plt.subplots(figsize=(7, 5))
        widths = results_df["avg_width"].values
 
        for i, (name, cov, w) in enumerate(
            zip(model_names, coverages, widths)
        ):
            ax.scatter(cov, w, s=120, color=colors[i],
                       edgecolors="black", zorder=3)
            ax.annotate(name, (cov, w),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
 
        ax.axvline(nominal, color="red", linestyle="--", linewidth=1.5,
                   label=f"Nominal {nominal:.0%}")
        ax.set_xlabel("Empirical coverage")
        ax.set_ylabel("Average interval width")
        ax.set_title(
            "Efficiency vs Validity\n"
            "Valid models: right of red line. "
            "Best = valid + lowest width."
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p3 = os.path.join(out_dir, "conformal_plot3_width_vs_coverage.png")
        plt.savefig(p3, dpi=150, bbox_inches="tight")
        plt.close()
        logging.info("Saved %s", p3)
 
    except Exception as e:
        raise CustomException(e, sys)
 