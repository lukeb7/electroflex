from __future__ import annotations

import os
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from electroflex_contracts import Forecaster, ModelSpec

class ContextPoint(BaseModel):
    timestamp: str
    demand_kw: float

class PredictRequest(BaseModel):
    context: list[ContextPoint] = Field(min_length=2)
    horizon: Optional[int] = Field(default=None, ge=1)

class ForecastPoint(BaseModel):
    timestamp: str
    demand_kw_pred: float

class PredictResponse(BaseModel):
    forecast: list[ForecastPoint]

def _load_model(model_path: Path) -> Forecaster:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    with model_path.open("rb") as f:
        model = pickle.load(f)
    return model

def _load_spec(model_spec_path: Path) -> ModelSpec:
    if not model_spec_path.exists():
        raise FileNotFoundError(f"Model spec not found: {model_spec_path}")
    return ModelSpec.from_json(model_spec_path)

def create_app(
    *,
    model_path: Path,
    model_spec_path: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="ElectroFlex API", version="0.1.0")

    model: Forecaster = _load_model(model_path)
    spec: ModelSpec | None = _load_spec(model_spec_path) if model_spec_path else None

    @app.get("/health")
    def health() -> dict:
        payload: dict = {"status": "ok"}
        payload["model_spec"] = asdict(spec) if spec is not None else None
        return payload

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        horizon = req.horizon or (spec.horizon_default if spec is not None else 24)

        context_df = pd.DataFrame([p.model_dump() for p in req.context])
        context_df["timestamp"] = pd.to_datetime(context_df["timestamp"], utc=True)

        try:
            forecast_df = model.predict(context=context_df, horizon=horizon)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        if list(forecast_df.columns) != ["timestamp", "demand_kw_pred"]:
            raise HTTPException(
                status_code=500,
                detail="Model returned invalid forecast schema",
            )

        forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"], utc=True)

        forecast = [
            ForecastPoint(
                timestamp=ts.isoformat().replace("+00:00", "Z"),
                demand_kw_pred=float(val),
            )
            for ts, val in zip(
                forecast_df["timestamp"],
                forecast_df["demand_kw_pred"],
            )
        ]

        return PredictResponse(forecast=forecast)

    return app

def _default_app() -> FastAPI:
    model_path = Path(
        os.getenv("ELECTROFLEX_MODEL_PATH", "artifacts/latest/model/model.pkl")
    )

    spec_path_env = os.getenv(
        "ELECTROFLEX_MODEL_SPEC_PATH",
        "artifacts/latest/model/model_spec.json",
    )
    spec_path = Path(spec_path_env) if spec_path_env else None

    return create_app(model_path=model_path, model_spec_path=spec_path)

# Import-safe default app:
# - Works with uvicorn when artifacts exist
# - Does NOT break pytest collection when they do not
try:
    app = _default_app()
except FileNotFoundError:
    app = FastAPI(title="ElectroFlex API", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model_spec": None,
            "note": (
                "Default model not found. "
                "Set ELECTROFLEX_MODEL_PATH / ELECTROFLEX_MODEL_SPEC_PATH "
                "or create the app explicitly via create_app(...)."
            ),
        }