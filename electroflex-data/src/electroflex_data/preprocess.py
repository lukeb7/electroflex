from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

REQUIRED_OUTPUT_COLUMNS = ["timestamp", "demand_kw"]

@dataclass(frozen=True)
class PreprocessConfig:
    timestamp_col: str = "timestamp"
    demand_col: str = "demand_kw"
    resample_rule: Optional[str] = "H"  # e.g. "H", "15T", or None

def preprocess(raw_path: Path, out_path: Path, config: PreprocessConfig | None = None,) -> Path:

    config = config or PreprocessConfig()

    raw_path = Path(raw_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    df = pd.read_csv(raw_path)

    if config.timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column '{config.timestamp_col}'")
    if config.demand_col not in df.columns:
        raise ValueError(f"Missing demand column '{config.demand_col}'")

    df = df[[config.timestamp_col, config.demand_col]].copy()
    df.rename(
        columns={
            config.timestamp_col: "timestamp",
            config.demand_col: "demand_kw",
        },
        inplace=True,
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["demand_kw"] = pd.to_numeric(df["demand_kw"], errors="coerce")

    df = df.dropna(subset=["timestamp", "demand_kw"])
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    if config.resample_rule:
        df = (
            df.set_index("timestamp")
            .resample(config.resample_rule)
            .mean(numeric_only=True)
            .dropna()
            .reset_index()
        )

    df["demand_kw"] = df["demand_kw"].astype(float)
    df = df[REQUIRED_OUTPUT_COLUMNS]

    df.to_parquet(out_path, index=False)

    return out_path