"""CLI stub for running an end-to-end demo of the pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from agents.analysis.pipeline import AnalysisStage
from agents.ingest.pipeline import IngestConfig, IngestStage
from agents.normalize.map_to_schema import NormalizeStage
from agents.orchestrator.orchestrator import PipelineOrchestrator
from agents.synthesize.pipeline import SynthesizeStage
from agents.validate.pipeline import ValidateStage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a demo pipeline over sample documents.")
    parser.add_argument("--in", dest="input_dir", type=Path, required=True, help="Input directory containing documents.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = sorted(p for p in args.input_dir.glob("*") if p.is_file())

    ingest_stage = IngestStage(IngestConfig(input_paths=input_paths))
    pipeline = PipelineOrchestrator(
        stages=[
            ingest_stage,
            NormalizeStage(),
            AnalysisStage(),
            ValidateStage(),
            SynthesizeStage(),
        ]
    )

    documents = pipeline.run([])
    for document in documents:
        print(document.model_dump_json(indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
