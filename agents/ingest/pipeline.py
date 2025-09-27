"""Ingest stage responsible for loading raw documents."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from agents.normalize.schema import DocumentData


@dataclass
class IngestConfig:
    """Configuration for the ingest stage."""

    input_paths: List[Path]


class IngestStage:
    """Load documents from disk or remote storage into ``DocumentData`` objects."""

    def __init__(self, config: IngestConfig) -> None:
        self._config = config

    def run(self, batch: list[DocumentData]) -> list[DocumentData]:
        """Populate the batch with newly ingested documents."""

        documents = []
        for path in self._config.input_paths:
            pages: list[str] = []
            if path.exists():
                if path.suffix.lower() in {".txt", ".md"}:
                    pages = [path.read_text(encoding="utf-8")]
                else:
                    # Binary formats (e.g. PDF) are currently left empty as placeholders.
                    pages = [""]

            documents.append(
                DocumentData(
                    pages=pages,
                    metadata={
                        "document_id": path.stem,
                        "source_path": str(path),
                        "mime_type": path.suffix.lower().lstrip("."),
                    },
                )
            )
        return documents
