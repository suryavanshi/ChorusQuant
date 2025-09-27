"""Synthesis stage responsible for final report generation."""
from __future__ import annotations

from dataclasses import dataclass

from agents.normalize.schema import DocumentData


@dataclass
class SynthesizeConfig:
    """Configuration for synthesis outputs."""

    output_format: str = "markdown"


class SynthesizeStage:
    """Produce final outputs informed by upstream stages."""

    def __init__(self, config: SynthesizeConfig | None = None) -> None:
        self._config = config or SynthesizeConfig()

    def run(self, batch: list[DocumentData]) -> list[DocumentData]:
        """Attach synthesized output placeholders to each document."""

        for document in batch:
            document.metadata.setdefault(
                "report",
                {
                    "format": self._config.output_format,
                    "content": f"Report for {document.document_id} pending implementation.",
                },
            )
        return batch
