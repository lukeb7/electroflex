from __future__ import annotations

import json
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from electroflex_contracts.run_spec import RunSpec
from electroflex_eval import EvalConfig, evaluate
from electroflex_train import train
from scripts.promote_if_better import promote_if_better


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _make_run_id(now: datetime) -> str:
    ts = now.strftime("%Y-%m-%dT%H%M%SZ")
    suffix = secrets.token_hex(3)  # 6 hex chars
    return f"{ts}_{suffix}"


def _try_get_git_sha() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return r.stdout.strip()
    except Exception:
        return None


def run_pipeline(
    *,
    series_path: Path,
    artifacts_root: Path,
    data_fingerprint: str,
    frequency: str = "h",
    horizon: int = 24,
    context_points: int = 48,
    primary_metric: str = "rmse",
) -> Path:
    """
    Orchestrate train -> eval and write immutable run artifacts.

    Returns the run directory: artifacts_root/runs/<run_id>
    """
    now = _utc_now()
    run_id = _make_run_id(now)

    run_dir = artifacts_root / "runs" / run_id
    model_dir = run_dir / "model"
    metrics_path = run_dir / "metrics.json"
    run_spec_path = run_dir / "run_spec.json"

    model_dir.mkdir(parents=True, exist_ok=True)

    # 1) TRAIN -> artifacts/runs/<run_id>/model/{model.pkl, model_spec.json}
    train(series_path=series_path, out_dir=model_dir, frequency=frequency, horizon_default=horizon)

    model_path = model_dir / "model.pkl"
    spec_path = model_dir / "model_spec.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Expected model at {model_path}")

    # 2) EVAL -> artifacts/runs/<run_id>/metrics.json
    evaluate(
        series_path=series_path,
        model_path=model_path,
        out_path=metrics_path,
        config=EvalConfig(frequency=frequency, horizon=horizon, context_points=context_points),
        model_spec_path=spec_path if spec_path.exists() else None,
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if primary_metric not in metrics:
        raise KeyError(f"Primary metric '{primary_metric}' not found in metrics.json keys={list(metrics.keys())}")

    rs = RunSpec(
        run_id=run_id,
        trained_at=now,
        git_sha=_try_get_git_sha(),
        data_fingerprint=data_fingerprint,
        model_path=str(model_dir).replace("\\", "/"),
        primary_metric=primary_metric,
        primary_metric_value=float(metrics[primary_metric]),
    )

    run_spec_path.write_text(
        json.dumps(rs.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    decision = promote_if_better(
        candidate_run_dir=run_dir,
        artifacts_root=artifacts_root,
        improvement_threshold=0.01,
    )

    print(f"[run_pipeline] promotion: promote={decision.promote} reason={decision.reason}")

    return run_dir


if __name__ == "__main__":
    # Minimal default runner (optional). Keeps this usable manually without committing to a full CLI.
    default_series = Path("data/processed/series.parquet")
    default_artifacts = Path("artifacts")

    run_dir = run_pipeline(
        series_path=default_series,
        artifacts_root=default_artifacts,
        data_fingerprint="dev_fingerprint",
    )
    print(f"[run_pipeline] wrote {run_dir}")