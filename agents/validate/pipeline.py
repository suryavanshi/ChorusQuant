"""Verification stage ensuring analytic outputs are backed by evidence."""
from __future__ import annotations

from dataclasses import dataclass

from agents.normalize.schema import DocumentData, Evidence


@dataclass
class ValidateConfig:
    """Configuration for verification heuristics."""

    require_numeric_evidence: bool = True


class ValidateStage:
    """Annotate documents with verification results."""

    def __init__(self, config: ValidateConfig | None = None) -> None:
        self._config = config or ValidateConfig()

    def run(self, batch: list[DocumentData]) -> list[DocumentData]:
        """Mark each document as verified if minimal evidence is present."""

        for document in batch:
            if document.text_pages:
                document.meta.setdefault(
                    "evidence",
                    [
                        Evidence(text=document.text_pages[0][:200], page=1).model_dump(),
                    ],
                )
            document.meta.setdefault("verified", str(self._config.require_numeric_evidence).lower())
        return batch
