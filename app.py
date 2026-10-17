"""
FastAPI inference server for RUL prediction.

    uvicorn app:app --host 0.0.0.0 --port 8000

Loads the trained artifacts once at startup, then answers POST /predict with a
point RUL prediction and a split-conformal prediction interval.

Artifacts are read from ARTIFACTS_DIR / MODELS_DIR (default: artifacts/ and models/;
use "demo_assets" for the bundled copy).

Training is not done here (DVC pipeline), and no data is stored.
"""
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from configs.data_constants_config import ALPHA
from src.pipeline.predict_pipeline import FEATURE_COLS, PredictPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

pipeline: PredictPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = PredictPipeline()
    logger.info("Artifacts loaded")
    yield


app = FastAPI(title="Turbofan RUL prediction", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    sensor_readings: list[float] = Field(
        ..., min_length=len(FEATURE_COLS), max_length=len(FEATURE_COLS),
        description=f"The {len(FEATURE_COLS)} informative sensors in this order: {', '.join(FEATURE_COLS)}",
    )
    alpha: float = Field(ALPHA, gt=0.0, lt=1.0, description="Mis-coverage level; the interval covers 1 - alpha.")


class PredictResponse(BaseModel):
    rul_prediction: float
    lower_bound: float
    upper_bound: float
    interval_width: float
    confidence_level: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if pipeline is not None else "loading", "features": FEATURE_COLS}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        return PredictResponse(**pipeline.predict(request.sensor_readings, request.alpha))
    except Exception as e:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=422, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
