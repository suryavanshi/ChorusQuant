"""CLI stub for running evaluation harnesses."""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pipeline outputs against golden references.")
    parser.add_argument("--pred", dest="pred_path", type=Path, required=True, help="Predictions JSONL file.")
    parser.add_argument("--gold", dest="gold_path", type=Path, required=True, help="Golden reference directory or file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Evaluation stub - predictions: {args.pred_path}, gold: {args.gold_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
