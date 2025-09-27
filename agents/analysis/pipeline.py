"""Analysis stage that derives insights from normalized documents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from agents.normalize.schema import DocumentData


@dataclass
class AnalysisResult:
    """Container for derived metrics."""

    metrics: Dict[str, float] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)


@dataclass
class AnalysisConfig:
    """Configuration for analysis calculations."""

    enable_portfolio_checks: bool = True


class AnalysisStage:
    """Perform lightweight analytics on normalized documents."""

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        self._config = config or AnalysisConfig()

    def run(self, batch: list[DocumentData]) -> list[DocumentData]:
        """Attach analysis metadata to each document."""

        for document in batch:
            document.metadata.setdefault("analysis", AnalysisResult(metrics={}, notes={}).__dict__)
        return batch
