import json
from datetime import datetime, timezone
from pathlib import Path

from electroflex_contracts.run_spec import RunSpec
from scripts.promote_if_better import promote_if_better


def write_run(tmp: Path, run_id: str, rmse: float) -> Path:
    run_dir = tmp / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "model").mkdir(parents=True, exist_ok=True)

    (run_dir / "metrics.json").write_text(json.dumps({"rmse": rmse}, indent=2), encoding="utf-8")

    rs = RunSpec(
        run_id=run_id,
        trained_at=datetime.now(timezone.utc),
        git_sha="abc1234",
        data_fingerprint="test_fp",
        model_path=str(run_dir / "model").replace("\\", "/"),
        primary_metric="rmse",
        primary_metric_value=rmse,
    )
    (run_dir / "run_spec.json").write_text(
        json.dumps(rs.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return run_dir


def test_promotes_first_candidate_and_updates_champion(tmp_path: Path):
    artifacts_root = tmp_path / "artifacts"
    candidate = write_run(artifacts_root, "run_a01", rmse=1.0)

    d = promote_if_better(candidate_run_dir=candidate, artifacts_root=artifacts_root, improvement_threshold=0.01)
    assert d.promote is True

    champion = json.loads((artifacts_root / "registry" / "champion.json").read_text(encoding="utf-8"))
    assert champion["run_id"] == "run_a01"
    assert champion["metric_value"] == 1.0


def test_does_not_promote_worse_candidate(tmp_path: Path):
    artifacts_root = tmp_path / "artifacts"
    a = write_run(artifacts_root, "run_a01", rmse=1.0)
    promote_if_better(candidate_run_dir=a, artifacts_root=artifacts_root, improvement_threshold=0.01)

    b = write_run(artifacts_root, "run_b01", rmse=1.001) # worse than 1.0
    d = promote_if_better(candidate_run_dir=b, artifacts_root=artifacts_root, improvement_threshold=0.01)
    assert d.promote is False

    champion = json.loads((artifacts_root / "registry" / "champion.json").read_text(encoding="utf-8"))
    assert champion["run_id"] == "run_a01"


def test_promotes_better_candidate(tmp_path: Path):
    artifacts_root = tmp_path / "artifacts"
    a = write_run(artifacts_root, "run_a01", rmse=1.0)
    promote_if_better(candidate_run_dir=a, artifacts_root=artifacts_root, improvement_threshold=0.01)

    b = write_run(artifacts_root, "run_b01", rmse=0.98)  # better than 0.99 required
    d = promote_if_better(candidate_run_dir=b, artifacts_root=artifacts_root, improvement_threshold=0.01)
    assert d.promote is True

    champion = json.loads((artifacts_root / "registry" / "champion.json").read_text(encoding="utf-8"))
    assert champion["run_id"] == "run_b01"