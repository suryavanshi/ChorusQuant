"""Normalization stage that maps raw extracts to structured schemas."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from agents.normalize.schema import DocumentData


@dataclass
class NormalizeConfig:
    """Configuration for the normalization stage."""

    enable_llm_backfill: bool = True


class NormalizeStage:
    """Normalize ingested documents into canonical ``DocumentData`` structures."""

    def __init__(self, config: NormalizeConfig | None = None) -> None:
        self._config = config or NormalizeConfig()

    def run(self, batch: list[DocumentData]) -> list[DocumentData]:
        """Perform basic normalization and passthrough for now."""

        for document in batch:
            document.metadata.setdefault("normalized", "true")
        return batch
