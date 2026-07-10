"""
prediction_plot.py

Simple plotting utility for FastAPI.

Generates a prediction interval plot for a single model and returns it
as a PNG image that FastAPI can send directly to the client.
"""


import io
import os
import sys
import base64
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import gaussian_kde
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
 
from configs.data_constants_config import ALPHA
from src.exception import CustomException
from src.logger import logging
 
 
@dataclass
class ConformalPlotsConfig:
    out_dir: str  =  os.path.join("artifacts")
 
 
class ConformalPlots:
    def __init__(self):
        self.config  =  ConformalPlotsConfig()
        residual_path = os.path.join("artifacts", "residuals.pkl")

        with open(residual_path, "rb") as f:
            self.residuals = np.asarray(pickle.load(f))


    def plot_single_prediction( self, point_pred: float, lower: float, upper: float, model_name: str, coverage: float, avg_width: float, 
                               alpha: float  =  ALPHA) -> str:
        """Generate a single-engine prediction interval chart for the API response.
 
        Returns the chart as a base64-encoded PNG string.
        Embed in HTML as: <img src = "data:image/png;base64,{returned_string}">
 
        Args:
            point_pred: Point prediction y_hat in RUL cycles.
            lower: Lower interval bound.
            upper: Upper interval bound.
            model_name: Display label for the chart title.
            alpha: Miscoverage level.
 
        Returns:
            Base64-encoded PNG string.
        """
        try:
            nominal  =  1 - alpha
            width  =  upper - lower
        
            fig =  plt.figure(figsize = (12,7), constrained_layout=True,)
            gs  =  GridSpec(2,2, figure = fig, width_ratios = [3,1], height_ratios = [1,1], hspace = 0.1, wspace = 0.3)

            ax_pdf  =  fig.add_subplot(gs[0,0])
            ax_interval  =  fig.add_subplot(gs[1,0])
            ax_cov  =  fig.add_subplot(gs[0,1])
            ax_tradeoff  =  fig.add_subplot(gs[1,1])

            #PDF plot
            scores = self.residuals  # absolute residuals / conformal scores

            # Mirror the scores so the density is centered on the prediction
            sym_scores = np.concatenate([-scores, scores])
            kde = gaussian_kde(sym_scores)
            margin = scores.max()
            x = np.linspace(point_pred - margin, point_pred + margin, 500)
            pdf = kde(x - point_pred)

            #ax_pdf.hist(res, bins=30, density=True, alpha=0.25, color="steelblue", label="Calibration residuals")
            ax_pdf.plot(x, pdf, color="navy", lw=2, label="KDE")
            ax_pdf.fill_between(x, pdf, where=(x >= lower) & (x <= upper), alpha=0.35,color="steelblue")

            ax_pdf.axvline(point_pred, color="red", lw=2, ls="--", label="Prediction",)
            ax_pdf.axvline(lower, color="black", ls=":")
            ax_pdf.axvline(upper, color="black", ls=":")
            ax_pdf.axvline(lower, color="black", ls=":")

            ax_pdf.set_xlim(0,125)
            ax_pdf.set_ylabel("Relative density")
            ax_pdf.set_title("Prediction uncertainty")
            ax_pdf.grid(alpha=.25)  

            #Errorbar
            ax_interval.errorbar(
                [point_pred], [0],
                xerr = [[point_pred - lower], [upper - point_pred]],
                fmt = 'o', color = 'navy', zorder = 5, capsize  =  5,
                #label = rf"$\hat{{y}}$  =  ${point_pred:.2f}^{+(upper - point_pred):.2f}_{-(point_pred - lower):.2f}$ cycles",
            )
            ax_interval.set_yticks([])
            ax_interval.set_xlabel("Remaining Useful Life (cycles)", fontsize = 11)
            ax_interval.set_title(
                f"{model_name}  -  RUL Prediction\n"
                f"Point estimate: {point_pred:.1f} cycles  |  "
                f"{nominal:.0%} interval: [{lower:.1f}, {upper:.1f}]  |  "
                f"Width: {width:.1f} cycles"
            )
            ax_interval.grid(True, axis = "x", alpha = 0.3)
            ax_interval.set_xlim(0, 125)
            #ax_interval.set_title(f"Prediction = {point_pred:.1f} cycles\n" f"{nominal:.0%} PI = [{lower:.1f}, {upper:.1f}]")

            #Coverage bar
            ax_cov.barh( ["Coverage"], [coverage], height=0.45, color="steelblue", alpha=0.7, edgecolor="black")
            ax_cov.axvline(nominal, color="red", ls="--", lw=2)
            ax_cov.text(coverage, 0, f"{coverage:.1%}", ha="left", va="center", fontsize=10)
            ax_cov.set_xlim(0,1)
            ax_cov.set_title("Empirical Coverage")
            ax_cov.grid(axis="x", alpha=.3)

            #Width vs coverage tradeoff
            ax_tradeoff.scatter(coverage, avg_width, s=220, color="dodgerblue", edgecolors="black")
            ax_tradeoff.axvline(nominal, color="red",ls="--")

            ax_tradeoff.annotate(model_name, (coverage, avg_width), xytext=(6,6), textcoords="offset points")
            ax_tradeoff.set_xlim(max(0.7, nominal-0.15), 1.0)
            ax_tradeoff.set_xlabel("Coverage")
            ax_tradeoff.set_ylabel("Avg Width")
            ax_tradeoff.set_title("Efficiency vs Validity")
            ax_tradeoff.grid(alpha=.3)

            fig.suptitle( f"{model_name} Conformal Prediction Dashboard", fontsize=15, fontweight="bold")

 
            buf  =  io.BytesIO()
            plt.savefig(buf, format = "png", dpi = 130, bbox_inches = "tight")
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")
 
        except Exception as e:
            raise CustomException(e, sys)
 
 
if __name__  ==  "__main__":
    # Test generate a sample single-prediction chart and save to disk
    plotter  =  ConformalPlots()
 
    b64  =  plotter.plot_single_prediction(point_pred = 47.2, lower = 16.0, upper = 78.4, coverage = 0.92, avg_width = 62.4,
        model_name = "XGBRegressor",)
 
    out_path  =  os.path.join("artifacts", "test_single_prediction.png")
    os.makedirs("artifacts", exist_ok = True)
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64))
    logging.info("Test chart saved - %s", out_path)
 