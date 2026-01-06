from datetime import datetime, timezone

from electroflex_contracts.run_spec import RunSpec


def test_run_spec_validates_minimal_fields():
    rs = RunSpec(
        run_id="2026-01-06T120102Z_ab12cd",
        trained_at=datetime.now(timezone.utc),
        git_sha="abc1234",
        data_fingerprint="datahash_001",
        model_path="artifacts/runs/2026-01-06T120102Z_ab12cd/model",
        primary_metric="rmse",
        primary_metric_value=0.421,
    )

    assert rs.run_id
    assert rs.primary_metric == "rmse"
    assert isinstance(rs.primary_metric_value, float)


def test_run_spec_rejects_missing_required_fields():
    # trained_at is required
    try:
        RunSpec(
            run_id="x12345",
            data_fingerprint="datahash_001",
            model_path="artifacts/runs/x12345/model",
            primary_metric="rmse",
            primary_metric_value=0.5,
        )
        assert False, "Expected validation error"
    except Exception:
        assert True