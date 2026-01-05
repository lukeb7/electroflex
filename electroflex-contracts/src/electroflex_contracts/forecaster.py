from __future__ import annotations

from typing import Protocol

import pandas as pd

class Forecaster(Protocol):
    def predict(self, context: pd.DataFrame, horizon: int) -> pd.DataFrame:
        ...