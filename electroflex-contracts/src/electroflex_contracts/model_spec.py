from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    model_version: str
    frequency: str
    horizon_default: int
    trained_at_utc: str

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @staticmethod
    def from_json(path: str | Path) -> "ModelSpec":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return ModelSpec(**data)