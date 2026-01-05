# electroflex-train

## Purpose
Train forecasting models from processed electricity demand time series and
produce versioned model artifacts.

## Inputs
- `series.parquet`
  - columns: `timestamp`, `demand_kw`

## Outputs
- `model.pkl` — serialized forecaster
- `model_spec.json` — metadata describing the model

## Model contract
- Models must implement the `Forecaster` interface defined in `electroflex-contracts`
- `predict(context, horizon)` returns:
  - `timestamp`
  - `demand_kw_pred`

## Current implementation
- `MeanForecaster`: constant mean baseline used for system integration

## Tests
- `test_train_contract.py` enforces artifact creation and contract compliance
