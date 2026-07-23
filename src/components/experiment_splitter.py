"""
ExperimentSplitter generates a calibration set out of the processed training data.

Why this is a seaparate module (not part of DataTransformation):

  DataTransformation's job is to load raw data, apply PCA/scaling, save artifacts.
  It knows nothing about how many models you're training or whether you need
  conformal prediction.

  ExperimentSplitter's job is given the already-transformed training data,
  split it by engine_id into train_proper and calibration.

  The split only exists because of conformal prediction. If you drop
  conformal from the project, ExperimentSplitter disappears entirely
  DataTransformation stays unchanged. That's the test for whether a module
  belongs here, can you remove it without touching anything else?

Why split by engine ID (not by row index)?

  Each engine is one independent degradation trajectory. The training data
  has ~20,000 rows from 100 engines. If you split by row index (e.g. first
  80% of rows = train, last 20% = cal), you might put cycles 1-150 of engine
  42 in train and cycles 151-200 of the same engine in calibration.

  That leaks: the calibration residuals would be computed on data the model
  has partially seen (the early life of that engine). Conformal's exchangeability
  assumption requires calibration points to be truly held-out.

  Engine-level split is the only correct approach for time series with
  multiple independent entities.

Where this step is taken in the pipeline:

  DataIngestion - DataTransformation - [ExperimentSplitter] - ModelTrainer
                                             
                                   runs here, after PCA is applied,
                                   before any model sees the data
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class ExperimentSplitterConfig: 
    X_train_proper_path: str = os.path.join("artifacts", "X_train_proper.csv")     #Use this for model training
    y_train_proper_path: str = os.path.join("artifacts", "y_train_proper.csv")     #Use this
    X_cal_path: str = os.path.join("artifacts", "X_cal.csv")
    y_cal_path: str = os.path.join("artifacts", "y_cal.csv")

    train_data_raw_csv_path: str = os.path.join('data/processed', 'train_set_001.csv')
    X_train_processed_path: str = os.path.join("artifacts", "X_train_processed.csv")
    y_train_path: str = os.path.join("artifacts", "y_train.csv")
    calibration_fraction: float = 0.2   #Fraction of engines held out for calibration
    random_seed: int = 42

class ExperimentSplitter:
    """
    Splits the processed training data into train_proper and calibration,
    by engine_id.

    Inputs:  X_train_processed (IC1..ICn), y_train (RUL), original train_df (for engine_IDs saved in train_data_raw_csv_path)
    Outputs: X_train_proper, y_train_proper, X_cal, y_cal
    """
    def __init__(self):
        self.experiment_splitter_config = ExperimentSplitterConfig()

    def initiate_data_split(self, X_train_processed_path: str, y_train_path: str, train_data_raw_csv_path: str):
      """
        Split processed training data by engine_id.

        Args:
            X_train_processed: Processed feature matrix (IC1..ICn), shape (n_rows, n_ICs). Must have the same row order as train_df_raw.
            y_train: RUL targets, shape (n_rows,). Same row order.
            train_df_raw: The original (pre-PCA) train dataframe, needed only for the engine_id column. Row order must match X_train.

        Returns:
            X_train_proper, y_train_proper, X_cal, y_cal
      """
      try: 
          train_df_raw = pd.read_csv(train_data_raw_csv_path)
          X_train = pd.read_csv(X_train_processed_path)
          y_train = pd.read_csv(y_train_path)

          engine_ids = train_df_raw["engine_id"].values
          unique_engines = np.unique(engine_ids)
          n_engines = len(unique_engines)

          rng = np.random.default_rng(self.experiment_splitter_config.random_seed)
          shuffled = rng.permutation(unique_engines)

          n_cal = max(1, int(n_engines * self.experiment_splitter_config.calibration_fraction))
          cal_engines   = set(shuffled[:n_cal])
          train_engines = set(shuffled[n_cal:])

          logging.info("Engine split: %d train_proper and %d calibration (out of %d total, seed=%d)",
                len(train_engines), len(cal_engines), n_engines, self.experiment_splitter_config.random_seed)
          
          # Boolean masks over rows (not engines)
          cal_mask   = np.isin(engine_ids, list(cal_engines))
          train_mask = ~cal_mask

          X_train_proper = X_train[train_mask].reset_index(drop=True)
          y_train_proper = y_train[train_mask].reset_index(drop=True)
          X_cal = X_train[cal_mask].reset_index(drop=True)
          y_cal = y_train[cal_mask].reset_index(drop=True)

          logging.info("Split sizes - train_proper: %d rows, calibration: %d rows", len(X_train_proper), len(X_cal))
          
          os.makedirs("artifacts", exist_ok=True)
          X_train_proper.to_csv(self.experiment_splitter_config.X_train_proper_path, index=False)
          y_train_proper.to_csv(self.experiment_splitter_config.y_train_proper_path, index=False, header=True)
          X_cal.to_csv(self.experiment_splitter_config.X_cal_path, index=False)
          y_cal.to_csv(self.experiment_splitter_config.y_cal_path, index=False, header=True)
          logging.info("Calibration split saved to artifacts/")

          return X_train_proper, y_train_proper, X_cal, y_cal
      
      except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = ExperimentSplitter()
    obj.initiate_data_split(
        X_train_processed_path = obj.experiment_splitter_config.X_train_processed_path, 
        y_train_path = obj.experiment_splitter_config.y_train_path, 
        train_data_raw_csv_path = obj.experiment_splitter_config.train_data_raw_csv_path
    )



 
    