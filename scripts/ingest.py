"""CLI entry-point for running the ingest stage."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.ingest.pipeline import IngestConfig, IngestStage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest financial documents into the pipeline store.")
    parser.add_argument("--in", dest="input_dir", type=Path, required=True, help="Directory containing source documents.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    input_paths: List[Path] = sorted(p for p in input_dir.glob("*") if p.is_file())

    ingest_stage = IngestStage(IngestConfig(input_paths=input_paths))
    documents = ingest_stage.run([])
    for document in documents:
        print(document.model_dump_json(indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
