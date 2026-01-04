from pathlib import Path
import pandas as pd

from electroflex_data.preprocess import preprocess, PreprocessConfig

def test_preprocess_writes_parquet_with_contract(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T01:00:00Z",
                "2026-01-01T01:00:00Z",
                "bad_timestamp",
            ],
            "demand_kw": [1.0, 2.0, 3.0, 999.0],
        }
    ).to_csv(raw, index=False)

    out = tmp_path / "series.parquet"

    config = PreprocessConfig(resample_rule=None)

    preprocess(raw_path=raw, out_path=out, config=config)

    result = pd.read_parquet(out)

    assert list(result.columns) == ["timestamp", "demand_kw"]
    assert result["timestamp"].is_monotonic_increasing
    assert result["timestamp"].is_unique
    assert len(result) == 2
    assert float(result["demand_kw"].iloc[1]) == 3.0