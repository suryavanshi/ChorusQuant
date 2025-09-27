"""Utilities for running the lightweight demo pipeline used in CLI shims."""
from __future__ import annotations

from pathlib import Path
from typing import List

from agents.analysis.pipeline import AnalysisStage
from agents.ingest.pipeline import IngestConfig, IngestStage
from agents.normalize.map_to_schema import NormalizeStage
from agents.normalize.schema import DocumentData
from agents.orchestrator.orchestrator import PipelineOrchestrator
from agents.synthesize.pipeline import SynthesizeStage
from agents.validate.pipeline import ValidateStage


def run_demo_pipeline(input_path: Path) -> DocumentData:
    """Execute the minimal ingest→normalize→analyze→validate→synthesize pipeline."""

    if input_path.is_dir():
        raise ValueError("Input path must reference a single file, not a directory.")

    stages = [
        IngestStage(IngestConfig(input_paths=[input_path])),
        NormalizeStage(),
        AnalysisStage(),
        ValidateStage(),
        SynthesizeStage(),
    ]
    orchestrator = PipelineOrchestrator(stages=stages)
    documents: List[DocumentData] = orchestrator.run([])
    if not documents:
        raise RuntimeError("Ingest stage did not return any documents.")

    document = documents[0]
    if not document.pages:
        document.pages = [""]
    document.metadata.setdefault("pipeline", "demo")
    return document
