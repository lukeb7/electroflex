from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from electroflex_eval import EvalConfig, evaluate
from electroflex_train import train


def test_evaluate_writes_metrics_with_required_keys(tmp_path: Path) -> None:
    # Arrange: synthetic processed series
    series_path = tmp_path / "series.parquet"
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC"),
            "demand_kw": [1.0 + (i % 10) * 0.1 for i in range(200)],
        }
    )
    df.to_parquet(series_path, index=False)

    # Train baseline model artifact
    model_dir = tmp_path / "model"
    train(series_path=series_path, out_dir=model_dir, frequency="h", horizon_default=24)

    model_path = model_dir / "model.pkl"
    spec_path = model_dir / "model_spec.json"

    out_path = tmp_path / "metrics.json"

    # Act
    evaluate(
        series_path=series_path,
        model_path=model_path,
        out_path=out_path,
        config=EvalConfig(frequency="h", horizon=24, context_points=48),
        model_spec_path=spec_path,
    )

    # Assert: artifact exists and contains required metrics
    assert out_path.exists()

    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert "mae" in payload
    assert "rmse" in payload
    assert "horizon" in payload
    assert "n_eval" in payload

    assert isinstance(payload["mae"], (int, float))
    assert isinstance(payload["rmse"], (int, float))
    assert payload["mae"] >= 0
    assert payload["rmse"] >= 0
    assert payload["horizon"] == 24
    assert payload["n_eval"] > 0

    assert "model_spec" in payload
    assert payload["model_spec"]["model_name"] == "mean_baseline"
