# electroflex-api

## Purpose
Expose a trained forecaster via a stable HTTP interface.

The API depends only on:
- the `Forecaster` contract (electroflex-contracts)
- a serialized model artifact (`model.pkl`)
- optional model metadata (`model_spec.json`)

## Endpoints

### `GET /health`
Returns service status and model metadata when available.

Response:
- `status`: `"ok"`
- `model_spec`: object or `null`

### `POST /predict`
Request:
- `context`: list of points with `timestamp` and `demand_kw`
- `horizon`: optional integer; defaults to `model_spec.horizon_default` if available

Response:
- `forecast`: list of points with `timestamp` and `demand_kw_pred`

## Running locally
Set environment variables to point to an existing trained model:

- `ELECTROFLEX_MODEL_PATH`
- `ELECTROFLEX_MODEL_SPEC_PATH`

Example:

```bash
uvicorn electroflex_api.app:app --reload --port 8000
```
## Tests
test_api_contract.py enforces:
- /health returns status and model metadata
- /predict returns correct response schema and forecast length
- API successfully loads and calls the trained model