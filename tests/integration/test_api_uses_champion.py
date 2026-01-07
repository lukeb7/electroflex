import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from electroflex_api.app import _default_app
from electroflex_train import train


def test_api_loads_champion_by_default(tmp_path: Path):
    artifacts_root = tmp_path / "artifacts"
    (artifacts_root / "registry").mkdir(parents=True)

    # Create synthetic data
    series_path = tmp_path / "series.parquet"
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC"),
            "demand_kw": [1.0] * 100,
        }
    )
    df.to_parquet(series_path, index=False)

    # Train model into run directory
    run_dir = artifacts_root / "runs" / "run_x01" / "model"
    train(series_path=series_path, out_dir=run_dir, frequency="h", horizon_default=24)

    # Write champion.json
    champion = {
        "run_id": "run_x01",
        "primary_metric": "rmse",
        "metric_value": 0.5,
        "trained_at": "2026-01-01T00:00:00Z",
        "git_sha": "abc1234",
        "model_path": str(run_dir).replace("\\", "/"),
    }
    (artifacts_root / "registry" / "champion.json").write_text(
        json.dumps(champion), encoding="utf-8"
    )

    # Start API with cwd set to tmp_path
    client = TestClient(_default_app(artifacts_root=artifacts_root))

    r = client.get("/health")
    assert r.status_code == 200