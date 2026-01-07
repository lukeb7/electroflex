from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from electroflex_contracts.run_spec import RunSpec


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str
    candidate_metric: float
    champion_metric: Optional[float]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_run_spec(run_dir: Path) -> RunSpec:
    return RunSpec.model_validate(_read_json(run_dir / "run_spec.json"))


def load_metrics(run_dir: Path) -> dict[str, Any]:
    return _read_json(run_dir / "metrics.json")


def decide_promotion(
    *,
    candidate_metric: float,
    champion_metric: Optional[float],
    improvement_threshold: float,
) -> PromotionDecision:
    if champion_metric is None:
        return PromotionDecision(
            promote=True,
            reason="no_champion",
            candidate_metric=candidate_metric,
            champion_metric=None,
        )

    required = champion_metric * (1.0 - improvement_threshold)
    if candidate_metric <= required:
        return PromotionDecision(
            promote=True,
            reason=f"better_than_champion_by_{improvement_threshold:.3f}",
            candidate_metric=candidate_metric,
            champion_metric=champion_metric,
        )

    return PromotionDecision(
        promote=False,
        reason=f"not_better_than_champion_by_{improvement_threshold:.3f}",
        candidate_metric=candidate_metric,
        champion_metric=champion_metric,
    )


def promote_if_better(
    *,
    candidate_run_dir: Path,
    artifacts_root: Path = Path("artifacts"),
    improvement_threshold: float = 0.01,
) -> PromotionDecision:
    registry_dir = artifacts_root / "registry"
    champion_path = registry_dir / "champion.json"

    run_spec = load_run_spec(candidate_run_dir)
    metrics = load_metrics(candidate_run_dir)

    primary = run_spec.primary_metric
    if primary not in metrics:
        raise KeyError(f"Primary metric '{primary}' not found in metrics.json keys={list(metrics.keys())}")

    candidate_metric = float(metrics[primary])

    champion_metric: Optional[float] = None
    if champion_path.exists():
        champion = _read_json(champion_path)
        champion_metric = float(champion["metric_value"])

    decision = decide_promotion(
        candidate_metric=candidate_metric,
        champion_metric=champion_metric,
        improvement_threshold=improvement_threshold,
    )

    if decision.promote:
        champion_payload = {
            "run_id": run_spec.run_id,
            "primary_metric": run_spec.primary_metric,
            "metric_value": candidate_metric,
            "trained_at": run_spec.trained_at.isoformat(),
            "git_sha": run_spec.git_sha,
            "model_path": run_spec.model_path,
        }
        _write_json(champion_path, champion_payload)

    return decision


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Promote a candidate run if it beats the current champion.")
    parser.add_argument("--candidate-run-dir", required=True)
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--improvement-threshold", type=float, default=0.01)
    args = parser.parse_args()

    decision = promote_if_better(
        candidate_run_dir=Path(args.candidate_run_dir),
        artifacts_root=Path(args.artifacts_root),
        improvement_threshold=args.improvement_threshold,
    )

    print(
        json.dumps(
            {
                "promote": decision.promote,
                "reason": decision.reason,
                "candidate_metric": decision.candidate_metric,
                "champion_metric": decision.champion_metric,
            },
            indent=2,
            sort_keys=True,
        )
    )