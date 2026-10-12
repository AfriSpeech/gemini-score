"""Command-line interface for GeminiScore."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __version__
from .core import GeminiScore
from .similarity import DEFAULT_MODEL, GeminiAuthError


def _load_lines_from_file(path: str) -> List[str]:
    """Reads non-empty stripped lines from a text file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\r\n") for line in f]


def _load_from_tabular(
    path: str,
    pred_col: str = "prediction",
    ref_col: str = "reference",
) -> tuple[List[str], List[str]]:
    """Reads predictions and references from a CSV or JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    predictions: List[str] = []
    references: List[str] = []
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        import csv

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                predictions.append(row.get(pred_col, ""))
                references.append(row.get(ref_col, ""))
    elif ext in (".json", ".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            if ext == ".jsonl":
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    predictions.append(item.get(pred_col, ""))
                    references.append(item.get(ref_col, ""))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        predictions.append(item.get(pred_col, ""))
                        references.append(item.get(ref_col, ""))
                elif isinstance(data, dict):
                    preds = data.get(pred_col, [])
                    refs = data.get(ref_col, [])
                    predictions.extend([str(p) for p in preds])
                    references.extend([str(r) for r in refs])
    else:
        raise ValueError(f"Unsupported tabular file extension: {ext}. Use .csv, .json, or .jsonl")

    return predictions, references


def build_parser() -> argparse.ArgumentParser:
    """Constructs the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="gemini-score",
        description=(
            "GeminiScore: Semantic MT evaluation metric using Google Gemini "
            "multilingual text embeddings."
        ),
    )

    # Input modes
    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument(
        "--pred",
        "-p",
        dest="predictions_file",
        help="Path to text file containing target predictions (one per line).",
    )
    input_group.add_argument(
        "--ref",
        "-r",
        dest="references_file",
        help="Path to text file containing references (one per line).",
    )
    input_group.add_argument(
        "--file",
        "-f",
        dest="input_file",
        help="Path to CSV, JSON, or JSONL file containing predictions and references.",
    )
    input_group.add_argument(
        "--pred-col",
        default="prediction",
        help="Column/key name for predictions in tabular file (default: 'prediction').",
    )
    input_group.add_argument(
        "--ref-col",
        default="reference",
        help="Column/key name for references in tabular file (default: 'reference').",
    )

    # Configuration options
    config_group = parser.add_argument_group("Model and Evaluation Configuration")
    config_group.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"Gemini embedding model identifier (default: {DEFAULT_MODEL}).",
    )
    config_group.add_argument(
        "--scale",
        choices=["percent", "unit", "raw"],
        default="percent",
        help="Score scaling: 'percent' (0-100, default), 'unit' (0-1), or 'raw' (-1 to 1).",
    )
    config_group.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding requests (default: 64).",
    )
    config_group.add_argument(
        "--api-key",
        help="Gemini API key (defaults to GEMINI_API_KEY / GOOGLE_API_KEY environment variable).",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output",
        "-o",
        help="Save evaluation results to output file (.json or .csv).",
    )
    output_group.add_argument(
        "--format",
        choices=["summary", "json", "score-only"],
        default="summary",
        help="Terminal output format: 'summary' (default), 'json', or 'score-only'.",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """CLI entry point for GeminiScore."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    predictions: List[str] = []
    references: List[str] = []

    if parsed_args.input_file:
        try:
            predictions, references = _load_from_tabular(
                parsed_args.input_file,
                pred_col=parsed_args.pred_col,
                ref_col=parsed_args.ref_col,
            )
        except Exception as exc:
            print(f"Error loading tabular file: {exc}", file=sys.stderr)
            return 1
    elif parsed_args.predictions_file and parsed_args.references_file:
        try:
            predictions = _load_lines_from_file(parsed_args.predictions_file)
            references = _load_lines_from_file(parsed_args.references_file)
        except Exception as exc:
            print(f"Error loading input files: {exc}", file=sys.stderr)
            return 1
    else:
        parser.print_help(file=sys.stderr)
        print(
            "\nError: Please provide inputs via --file, or via --pred and --ref.",
            file=sys.stderr,
        )
        return 1

    if len(predictions) != len(references):
        print(
            f"Error: Number of predictions ({len(predictions)}) does not match "
            f"number of references ({len(references)}).",
            file=sys.stderr,
        )
        return 1

    try:
        scorer = GeminiScore(
            api_key=parsed_args.api_key,
            model=parsed_args.model,
            scale=parsed_args.scale,
            batch_size=parsed_args.batch_size,
        )
    except GeminiAuthError as auth_err:
        print(f"Authentication Error: {auth_err}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Initialization Error: {exc}", file=sys.stderr)
        return 1

    try:
        result = scorer.score(
            predictions=predictions,
            references=references,
        )
    except Exception as exc:
        print(f"Evaluation Error: {exc}", file=sys.stderr)
        return 1

    if parsed_args.format == "score-only":
        print(f"{result.score:.4f}")
    elif parsed_args.format == "json":
        print(result.to_json())
    else:
        print(result.summary())

    if parsed_args.output:
        out_ext = os.path.splitext(parsed_args.output)[1].lower()
        try:
            if out_ext == ".csv":
                df = result.to_dataframe()
                df.to_csv(parsed_args.output, index=False)
            else:
                with open(parsed_args.output, "w", encoding="utf-8") as f:
                    f.write(result.to_json())
            if parsed_args.format in ("json", "score-only"):
                print(f"Saved detailed results to {parsed_args.output}", file=sys.stderr)
            else:
                print(f"\nSaved detailed results to {parsed_args.output}")
        except Exception as exc:
            print(f"Failed to save output to {parsed_args.output}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
