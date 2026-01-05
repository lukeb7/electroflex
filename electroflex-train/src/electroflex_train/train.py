from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from electroflex_contracts import ModelSpec
from electroflex_train.baseline import MeanForecaster

def train(
    *,
    series_path: Path,
    out_dir: Path,
    frequency: str = "h",
    horizon_default: int = 24,
    model_name: str = "mean_baseline",
    model_version: str = "0.1.0",
) -> Path:
    series_path = Path(series_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not series_path.exists():
        raise FileNotFoundError(f"Processed series not found: {series_path}")

    series = pd.read_parquet(series_path)

    if "timestamp" not in series.columns or "demand_kw" not in series.columns:
        raise ValueError("Processed series must contain columns: 'timestamp', 'demand_kw'")

    model = MeanForecaster(frequency=frequency).fit(series)

    model_path = out_dir / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    trained_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    spec = ModelSpec(
        model_name=model_name,
        model_version=model_version,
        frequency=frequency,
        horizon_default=horizon_default,
        trained_at_utc=trained_at_utc,
    )

    spec_path = out_dir / "model_spec.json"
    spec.to_json(spec_path)

    return out_dir