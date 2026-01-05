from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from electroflex_contracts import Forecaster, ModelSpec

@dataclass(frozen=True)
class EvalConfig:
    frequency: str = "h"
    horizon: int = 24
    context_points: int = 48


def _mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    err = (y_true - y_pred).abs()
    return float(err.mean())


def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    err2 = (y_true - y_pred) ** 2
    return float(math.sqrt(err2.mean()))


def evaluate(
    *,
    series_path: Path,
    model_path: Path,
    out_path: Path,
    config: EvalConfig | None = None,
    model_spec_path: Path | None = None,
) -> Path:
    config = config or EvalConfig()

    series_path = Path(series_path)
    model_path = Path(model_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not series_path.exists():
        raise FileNotFoundError(f"Processed series not found: {series_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    series = pd.read_parquet(series_path)

    required = {"timestamp", "demand_kw"}
    if not required.issubset(series.columns):
        raise ValueError("Processed series must contain columns: 'timestamp', 'demand_kw'")

    series = series.sort_values("timestamp").reset_index(drop=True)

    min_len = config.context_points + config.horizon
    if len(series) < min_len:
        raise ValueError(f"Not enough rows for evaluation: need >= {min_len}, got {len(series)}")

    context = series.iloc[-(config.context_points + config.horizon) : -config.horizon].copy()
    actual = series.iloc[-config.horizon :].copy()

    context["timestamp"] = pd.to_datetime(context["timestamp"], utc=True)
    actual["timestamp"] = pd.to_datetime(actual["timestamp"], utc=True)

    with model_path.open("rb") as f:
        model = pickle.load(f)

    forecaster: Forecaster = model

    forecast = forecaster.predict(context=context, horizon=config.horizon)

    if list(forecast.columns) != ["timestamp", "demand_kw_pred"]:
        raise ValueError("Forecast must have columns: ['timestamp', 'demand_kw_pred']")

    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"], utc=True)

    joined = actual.merge(forecast, on="timestamp", how="inner")
    if len(joined) == 0:
        raise ValueError("No overlapping timestamps between actual and forecast")

    mae = _mae(joined["demand_kw"], joined["demand_kw_pred"])
    rmse = _rmse(joined["demand_kw"], joined["demand_kw_pred"])

    payload: dict[str, Any] = {
        "mae": mae,
        "rmse": rmse,
        "horizon": config.horizon,
        "context_points": config.context_points,
        "n_eval": int(len(joined)),
    }

    if model_spec_path is not None and Path(model_spec_path).exists():
        spec = ModelSpec.from_json(model_spec_path)
        payload["model_spec"] = asdict(spec)

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path