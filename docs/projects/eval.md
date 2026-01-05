# electroflex-eval

## Purpose
Evaluate a trained forecaster offline and produce stable metrics artifacts that can
be used for regression testing and (later) CI quality gates.

## Inputs
- `series.parquet`
  - columns: `timestamp`, `demand_kw`
- `model.pkl`
  - a serialized object implementing the `Forecaster` contract
- (optional) `model_spec.json`
  - metadata stored alongside the model artifact

## Outputs
- `metrics.json`

Minimum keys:
- `mae`
- `rmse`
- `horizon`
- `context_points`
- `n_eval`

## Evaluation approach (MVP)
- Uses the most recent `context_points` rows as context
- Evaluates on the following `horizon` points
- Aligns forecast and actual by timestamp

## Tests
- `test_eval_contract.py` enforces:
  - metrics artifact exists
  - required keys are present
  - metrics are numeric and non-negative
  - evaluation successfully loads and uses the trained model