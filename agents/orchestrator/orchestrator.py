"""Pipeline orchestrator for the multi-agent financial analysis system."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol

from agents.normalize.schema import DocumentData


class PipelineStage(Protocol):
    """Protocol for individual pipeline stages."""

    def run(self, batch: list[DocumentData]) -> list[DocumentData]:
        """Process a batch of documents and return the updated batch."""


@dataclass
class OrchestratorConfig:
    """Configuration for the pipeline orchestrator."""

    batch_size: int = 4


class PipelineOrchestrator:
    """Coordinates the ingest → parse → analyze → verify → synthesize pipeline."""

    def __init__(self, stages: Iterable[PipelineStage], config: OrchestratorConfig | None = None) -> None:
        self._stages: List[PipelineStage] = list(stages)
        self._config = config or OrchestratorConfig()

    def run(self, initial_batch: list[DocumentData]) -> list[DocumentData]:
        """Run the configured pipeline over the provided batch."""

        batch = initial_batch
        for stage in self._stages:
            batch = stage.run(batch)
        return batch

    def add_stage(self, stage: PipelineStage) -> None:
        """Append a stage to the pipeline at runtime."""

        self._stages.append(stage)

    @property
    def config(self) -> OrchestratorConfig:
        return self._config
