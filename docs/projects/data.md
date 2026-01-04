# electroflex-data

## Purpose
Transform raw household electricity consumption data into a canonical, model-ready
time series artifact with a stable schema.

## Inputs
- Raw dataset (local file; excluded from git)

Expected columns (configurable):
- timestamp column (e.g. `timestamp` or dataset-specific name)
- demand column (e.g. `demand_kw` or dataset-specific name)

## Outputs
- `data/processed/series.parquet`

Schema:
- `timestamp` (datetime, UTC)
- `demand_kw` (float)

## Contract invariants
- `timestamp` is sorted ascending and unique
- rows with invalid timestamps or non-numeric demand are dropped
- optional resampling to a fixed frequency (default hourly)

## How to run (Python)
```python
from pathlib import Path
from electroflex_data.preprocess import preprocess, PreprocessConfig

preprocess(
    raw_path=Path("data/raw/<yourfile>.csv"),
    out_path=Path("data/processed/series.parquet"),
    config=PreprocessConfig(
        timestamp_col="timestamp",
        demand_col="demand_kw",
        resample_rule="H",
    ),
)
```

## Tests
test_preprocess_contract.py enforces:
- output schema
- timestamp ordering/uniqueness
- numeric demand
- deterministic handling of duplicates and invalid rows