from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from electroflex_api import create_app
from electroflex_data.preprocess import PreprocessConfig, preprocess
from electroflex_eval import EvalConfig, evaluate
from electroflex_train import train


def test_end_to_end_smoke_pipeline(tmp_path: Path) -> None:
    # Arrange: raw CSV in temp dir
    raw_path = tmp_path / "raw.csv"
    raw_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=500, freq="15min", tz="UTC").astype(str),
            "demand_kw": [1.0 + (i % 20) * 0.05 for i in range(500)],
        }
    )
    raw_df.to_csv(raw_path, index=False)

    # 1) preprocess -> series.parquet
    series_path = tmp_path / "series.parquet"
    preprocess(
        raw_path=raw_path,
        out_path=series_path,
        config=PreprocessConfig(timestamp_col="timestamp", demand_col="demand_kw", resample_rule="h"),
    )
    assert series_path.exists()

    # 2) train -> model artifacts
    model_dir = tmp_path / "model"
    train(series_path=series_path, out_dir=model_dir, frequency="h", horizon_default=24)
    model_path = model_dir / "model.pkl"
    spec_path = model_dir / "model_spec.json"
    assert model_path.exists()
    assert spec_path.exists()

    # 3) eval -> metrics.json
    metrics_path = tmp_path / "metrics.json"
    evaluate(
        series_path=series_path,
        model_path=model_path,
        out_path=metrics_path,
        config=EvalConfig(frequency="h", horizon=24, context_points=48),
        model_spec_path=spec_path,
    )
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["n_eval"] > 0
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0

    # 4) api -> /predict
    app = create_app(model_path=model_path, model_spec_path=spec_path)
    client = TestClient(app)

    processed = pd.read_parquet(series_path).sort_values("timestamp")
    context = processed.tail(48)

    payload = {
        "context": [
            {"timestamp": pd.to_datetime(ts, utc=True).isoformat(), "demand_kw": float(val)}
            for ts, val in zip(context["timestamp"], context["demand_kw"])
        ],
        "horizon": 12,
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "forecast" in body
    assert len(body["forecast"]) == 12