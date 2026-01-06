# electroflex-contracts

## Purpose
Define shared interfaces and schemas used across ElectroFlex components.

This package exists to keep component boundaries clean:
- `electroflex-train` owns implementations
- `electroflex-eval` and `electroflex-api` depend only on contracts + artifacts

## Exports

### Interfaces
- `Forecaster`
  - Common interface for models consumed by eval and API.
  - Requires `predict(context, horizon)` to return forecast points with:
    - `timestamp`
    - `demand_kw_pred`

### Schemas
- `ModelSpec`
  - Stable metadata stored alongside a trained model artifact (e.g. default horizon, model name, etc.).

- `RunSpec`
  - Metadata for an immutable training/evaluation run.
  - Intended to support traceability and future promotion/registry workflows without requiring a full ML platform.

  Recommended fields:
  - `run_id`
  - `trained_at`
  - `git_sha` (optional locally; populated in CI)
  - `data_fingerprint`
  - `model_path`
  - `primary_metric`
  - `primary_metric_value`

## Why this exists
This package prevents `electroflex-eval` and `electroflex-api` from depending on training internals.
Only contracts are shared; implementations remain isolated.