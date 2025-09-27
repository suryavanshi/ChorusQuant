"""Planning utilities for orchestrating pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from agents.normalize.schema import DocumentData


@dataclass
class PlanStep:
    """Represents a single planned action for a document."""

    name: str
    description: str


@dataclass
class DocumentPlan:
    """A set of planned steps to execute for a document."""

    document_id: str
    steps: List[PlanStep]


def make_default_plan(document: DocumentData) -> DocumentPlan:
    """Create a deterministic plan covering all pipeline stages for a document."""

    steps = [
        PlanStep(name="ingest", description="Load raw document bytes and metadata."),
        PlanStep(name="parse", description="Extract text, tables, and visual artifacts."),
        PlanStep(name="analyze", description="Compute financial insights from normalized data."),
        PlanStep(name="verify", description="Cross-check assertions and collect evidence."),
        PlanStep(name="synthesize", description="Generate human-readable and machine-readable outputs."),
    ]
    return DocumentPlan(document_id=document.metadata.get("document_id", "unknown"), steps=steps)
