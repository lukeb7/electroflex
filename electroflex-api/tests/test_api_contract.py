from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from electroflex_api import create_app
from electroflex_train import train


def test_api_health_and_predict_contract(tmp_path: Path) -> None:
    # Arrange: create synthetic processed series
    series_path = tmp_path / "series.parquet"
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC"),
            "demand_kw": [1.0 + (i % 10) * 0.1 for i in range(200)],
        }
    )
    df.to_parquet(series_path, index=False)

    # Train model artifact into tmp_path
    model_dir = tmp_path / "model"
    train(series_path=series_path, out_dir=model_dir, frequency="h", horizon_default=24)

    model_path = model_dir / "model.pkl"
    spec_path = model_dir / "model_spec.json"

    app = create_app(model_path=model_path, model_spec_path=spec_path)
    client = TestClient(app)

    # Health
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_spec"]["model_name"] == "mean_baseline"

    # Predict
    context = df.tail(48).copy()
    payload = {
        "context": [
            {"timestamp": ts.isoformat(), "demand_kw": float(val)}
            for ts, val in zip(context["timestamp"], context["demand_kw"])
        ],
        "horizon": 12,
    }

    r2 = client.post("/predict", json=payload)
    assert r2.status_code == 200
    out = r2.json()

    assert "forecast" in out
    assert len(out["forecast"]) == 12
    assert set(out["forecast"][0].keys()) == {"timestamp", "demand_kw_pred"}