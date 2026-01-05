from __future__ import annotations

from pathlib import Path

from electroflex_contracts import ModelSpec

def test_model_spec_json_roundtrip(tmp_path: Path) -> None:
    spec = ModelSpec(
        model_name="mean_baseline",
        model_version="0.1.0",
        frequency="H",
        horizon_default=24,
        trained_at_utc="2026-01-01T00:00:00Z",
    )

    out = tmp_path / "model_spec.json"
    spec.to_json(out)

    loaded = ModelSpec.from_json(out)
    assert loaded == spec