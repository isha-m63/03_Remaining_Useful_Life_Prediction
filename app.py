
"""
app.py for FastAPI inference server for RUL prediction.
 
This file's ONLY job:
    1. Load trained model artifacts (once, at startup)
    2. Accept HTTP POST requests with sensor readings
    3. Run the same preprocessing used during training
    4. Return RUL point prediction + conformal prediction interval
 
What this file does NOT do:
    - Train models (DVC + training pipeline)
    - Store data (that's a database, not an API)
    - Call DVC or MLflow (those are training-time tools)
"""

import pickle
import logging
import dill
import os
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pathlib import Path
from pydantic import BaseModel
from configs.data_constants_config import ALPHA

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s",)
logger = logging.getLogger(__name__)


#Model store - loaded once at startup, reused for every request
 
class ModelStore:
    """Holds all artifacts needed for inference."""
    preprocessor = None    # sklearn Pipeline (scaler + PCA)
    model = None           # XGBoost 
    q_hat: float = None    # conformal quantile for interval construction
    feature_names: list[str] = None   # column order expected by preprocessor
 
model_store = ModelStore()

def load_artifacts() -> None:
    """
    Load all pkl artifacts into memory.
    Called once at server startup via the lifespan hook.
 
    Why dill instead of pickle?
        dill can serialize lambda functions, closures, and some sklearn
        objects that standard pickle can't. If your preprocessor or model
        was saved with dill, you must load with dill. Harmless if not.
    """
    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
    models_dir    = Path(os.getenv("MODELS_DIR", "models"))
 
    logger.info(f"Loading artifacts from {artifacts_dir} and {models_dir}")
 
    with open(artifacts_dir / "preprocessor.pkl", "rb") as f:
        model_store.preprocessor = dill.load(f)
    logger.info("Preprocessor loaded")
 
    with open(models_dir / "xgboost_model.pkl", "rb") as f:
        model_store.model = dill.load(f)
    logger.info("Model loaded")
