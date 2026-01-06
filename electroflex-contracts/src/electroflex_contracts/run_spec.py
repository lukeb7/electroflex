from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class RunSpec(BaseModel):
    run_id: str = Field(..., min_length=6)
    trained_at: datetime

    # Optional locally (handy), but should be populated in CI.
    git_sha: Optional[str] = Field(default=None, min_length=7)

    # Fingerprint of the dataset used for this run (hash/etag/version string).
    data_fingerprint: str = Field(..., min_length=6)

    # Path to the model artifact for this run (relative to repo root is fine).
    model_path: str = Field(..., min_length=1)

    # The "decision metric" used for promotion.
    primary_metric: str = Field(..., min_length=1)
    primary_metric_value: float