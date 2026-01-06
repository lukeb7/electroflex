# ElectroFlex pipeline

## Purpose
Describe the end-to-end ElectroFlex workflow and the artifact flow between components.

This is intentionally “production-shaped” but lightweight: artifacts are the integration surface,
and contracts are the stability layer.

## Components
- `electroflex-data`: raw → canonical time series (`series.parquet`)
- `electroflex-train`: time series → model artifacts (`model.pkl`, `model_spec.json`)
- `electroflex-eval`: model + data → metrics (`metrics.json`)
- `electroflex-api`: serves a trained forecaster via HTTP using the `Forecaster` contract
- `electroflex-contracts`: shared interfaces + schemas

## Canonical artifacts

### Processed data
- `data/processed/series.parquet`
  - schema: `timestamp` (UTC datetime), `demand_kw` (float)

### Model artifacts (single model)
- `model.pkl`
- `model_spec.json`

### Evaluation artifact
- `metrics.json`

## Run artifacts (standard layout)
ElectroFlex uses an immutable “run folder” convention to support reproducibility and automation.

- `artifacts/`
  - `runs/<run_id>/`
    - `model/` (or `model.pkl`)
    - `metrics.json`
    - `run_spec.json`
  - `registry/`
    - `champion.json` (planned: pointer to the current production model)

At this commit, `artifacts/runs/` and `artifacts/registry/` exist and `RunSpec` is defined in contracts.
The pipeline runner and registry updates are introduced in subsequent commits.

## Current workflow (manual)
Today, each component can be run independently, producing the artifacts listed above:

1) Preprocess
- Input: raw dataset (excluded from git)
- Output: `data/processed/series.parquet`

2) Train
- Input: `series.parquet`
- Output: `model.pkl`, `model_spec.json`

3) Evaluate
- Input: `series.parquet`, `model.pkl`, optional `model_spec.json`
- Output: `metrics.json`

4) Serve
- API loads a model artifact (via environment variables) and exposes `/health` and `/predict`.

## Next steps (closing the loop)
Planned incremental changes to make the workflow “one command is truth” and close the loop:

- Add `scripts/run_pipeline.py` to orchestrate: train → eval → write `artifacts/runs/<run_id>/...`
- Add `scripts/promote_if_better.py` to implement deterministic promotion and update `artifacts/registry/champion.json`
- Update the API to serve the champion by default and expose `/version`
- Add a scheduled GitHub Actions workflow to retrain/evaluate/promote automatically