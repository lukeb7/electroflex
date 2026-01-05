from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from electroflex_contracts import ModelSpec, Forecaster
from electroflex_train.train import train


def test_train_writes_model_and_spec_and_predicts(tmp_path: Path) -> None:
    # Arrange: create a tiny processed series parquet
    series_path = tmp_path / "series.parquet"
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC"),
            "demand_kw": [1.0, 1.2, 1.1, 1.3, 1.25, 1.15, 1.4, 1.35, 1.3, 1.2],
        }
    )
    df.to_parquet(series_path, index=False)

    out_dir = tmp_path / "model"

    # Act
    train_out = train(
        series_path=series_path,
        out_dir=out_dir,
        frequency="h",
        horizon_default=24,
        model_name="mean_baseline",
        model_version="0.1.0",
    )

    # Assert: training returns the output directory
    assert train_out == out_dir
    assert out_dir.exists()

    model_path = out_dir / "model.pkl"
    spec_path = out_dir / "model_spec.json"

    assert model_path.exists()
    assert spec_path.exists()

    # Assert: spec round-trip
    spec = ModelSpec.from_json(spec_path)
    assert spec.model_name == "mean_baseline"
    assert spec.model_version == "0.1.0"
    assert spec.frequency == "h"
    assert spec.horizon_default == 24
    assert spec.trained_at_utc.endswith("Z")

    # Assert: model loads and conforms to Forecaster contract
    with model_path.open("rb") as f:
        model = pickle.load(f)

    forecaster: Forecaster = model  # type-level contract check

    # Use last few rows as context
    context = df.tail(3).copy()
    forecast = forecaster.predict(context=context, horizon=5)

    assert list(forecast.columns) == ["timestamp", "demand_kw_pred"]
    assert len(forecast) == 5
    assert pd.api.types.is_datetime64_any_dtype(forecast["timestamp"])
    assert pd.api.types.is_numeric_dtype(forecast["demand_kw_pred"])

    # Forecast timestamps should be strictly increasing and after the last context timestamp
    assert forecast["timestamp"].is_monotonic_increasing
    assert forecast["timestamp"].iloc[0] > pd.to_datetime(context["timestamp"].iloc[-1], utc=True)
