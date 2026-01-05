from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

@dataclass
class MeanForecaster:
    frequency: str = "h"
    mean_kw: Optional[float] = None

    def fit(self, series: pd.DataFrame) -> "MeanForecaster":
        if "demand_kw" not in series.columns:
            raise ValueError("series must contain 'demand_kw'")
        mean = pd.to_numeric(series["demand_kw"], errors="coerce").dropna().mean()
        if pd.isna(mean):
            raise ValueError("cannot fit MeanForecaster: demand_kw contains no numeric values")
        self.mean_kw = float(mean)
        return self

    def predict(self, context: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if horizon <= 0:
            raise ValueError("horizon must be > 0")
        if "timestamp" not in context.columns:
            raise ValueError("context must contain 'timestamp'")
        if self.mean_kw is None:
            raise ValueError("model is not fitted (mean_kw is None)")

        last_ts = pd.to_datetime(context["timestamp"].iloc[-1], utc=True)
        future_ts = pd.date_range(last_ts, periods=horizon + 1, freq=self.frequency, tz="UTC")[1:]

        return pd.DataFrame(
            {
                "timestamp": future_ts,
                "demand_kw_pred": [self.mean_kw] * horizon,
            }
        )