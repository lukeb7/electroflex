from __future__ import annotations

import pandas as pd

from electroflex_contracts import Forecaster

class DummyForecaster:
    def predict(self, context: pd.DataFrame, horizon: int) -> pd.DataFrame:
        last_ts = pd.to_datetime(context["timestamp"].iloc[-1], utc=True)
        future_ts = pd.date_range(last_ts, periods=horizon + 1, freq="h", tz="UTC")[1:]
        return pd.DataFrame(
            {
                "timestamp": future_ts,
                "demand_kw_pred": [1.0] * horizon,
            }
        )

def test_forecaster_contract_schema() -> None:
    model: Forecaster = DummyForecaster()

    context = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
            "demand_kw": [1.0, 1.2],
        }
    )

    forecast = model.predict(context=context, horizon=3)

    assert list(forecast.columns) == ["timestamp", "demand_kw_pred"]
    assert len(forecast) == 3
    assert pd.api.types.is_datetime64_any_dtype(forecast["timestamp"])
    assert pd.api.types.is_numeric_dtype(forecast["demand_kw_pred"])