
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
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s",)
logger = logging.getLogger(__name__)

artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading artifacts...")
    with open("artifacts/preprocessor.pkl", "rb") as f:
        artifacts["preprocessor"] = pickle.load(f)
    with open("models/xgboost_model.pkl", "rb") as f:
        artifacts["model"] = pickle.load(f)
    with open("artifacts/residuals.pkl", "rb") as f:
        residuals = pickle.load(f)