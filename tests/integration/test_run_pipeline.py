import json
from pathlib import Path

import pandas as pd

from scripts.run_pipeline import run_pipeline


def test_run_pipeline_writes_immutable_run_artifacts(tmp_path: Path) -> None:
    # Arrange: synthetic processed series
    series_path = tmp_path / "series.parquet"
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC"),
            "demand_kw": [1.0 + (i % 10) * 0.1 for i in range(200)],
        }
    )
    df.to_parquet(series_path, index=False)

    artifacts_root = tmp_path / "artifacts"

    # Act
    run_dir = run_pipeline(
        series_path=series_path,
        artifacts_root=artifacts_root,
        data_fingerprint="test_fingerprint",
        frequency="h",
        horizon=24,
        context_points=48,
        primary_metric="rmse",
    )

    # Assert
    assert run_dir.exists()
    assert (run_dir / "model" / "model.pkl").exists()
    assert (run_dir / "model" / "model_spec.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "run_spec.json").exists()

    rs = json.loads((run_dir / "run_spec.json").read_text(encoding="utf-8"))
    assert rs["data_fingerprint"] == "test_fingerprint"
    assert rs["primary_metric"] == "rmse"