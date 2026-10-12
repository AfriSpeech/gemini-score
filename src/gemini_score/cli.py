"""Command-line interface for GeminiScore."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __version__
from .core import GeminiScore
from .languages import is_supported_language, list_supported_languages, resolve_language
from .similarity import DEFAULT_SIMILARITY_MODEL
from .translator import DEFAULT_GEMINI_MODEL, GeminiAuthError


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
    lang_col: Optional[str] = None,
) -> tuple[List[str], List[str], Optional[List[str]]]:
    """Reads predictions, references, and optional language from CSV or JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    predictions: List[str] = []
    references: List[str] = []
    languages: Optional[List[str]] = [] if lang_col else None

    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        import csv

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                predictions.append(row.get(pred_col, ""))
                references.append(row.get(ref_col, ""))
                if languages is not None and lang_col:
                    languages.append(row.get(lang_col, ""))
    elif ext in (".json", ".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            if ext == ".jsonl":
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    predictions.append(item.get(pred_col, ""))
                    references.append(item.get(ref_col, ""))
                    if languages is not None and lang_col:
                        languages.append(item.get(lang_col, ""))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        predictions.append(item.get(pred_col, ""))
                        references.append(item.get(ref_col, ""))
                        if languages is not None and lang_col:
                            languages.append(item.get(lang_col, ""))
                elif isinstance(data, dict):
                    preds = data.get(pred_col, [])
                    refs = data.get(ref_col, [])
                    predictions.extend([str(p) for p in preds])
                    references.extend([str(r) for r in refs])
                    if languages is not None and lang_col:
                        languages.extend([str(l) for l in data.get(lang_col, [])])
    else:
        raise ValueError(f"Unsupported tabular file extension: {ext}. Use .csv, .json, or .jsonl")

    return predictions, references, languages


def build_parser() -> argparse.ArgumentParser:
    """Constructs the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="gemini-score",
        description=(
            "GeminiScore: Semantic MT evaluation metric for African languages "
            "using Gemini back-translation and static Model2Vec sentence similarity."
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
        help="Path to text file containing English references (one per line).",
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
    input_group.add_argument(
        "--lang-col",
        default=None,
        help="Optional column/key name for language in tabular file.",
    )
    input_group.add_argument(
        "--language",
        "-l",
        dest="language",
        help="African language name or ISO code (e.g. 'yor', 'swh', 'twi', 'Yoruba').",
    )

    # Scorer options
    config_group = parser.add_argument_group("Model and Evaluation Configuration")
    config_group.add_argument(
        "--model",
        "-m",
        default=DEFAULT_GEMINI_MODEL,
        help=f"Gemini model identifier (default: {DEFAULT_GEMINI_MODEL}, requires Flash 3.0+).",
    )
    config_group.add_argument(
        "--similarity-model",
        default=DEFAULT_SIMILARITY_MODEL,
        help=f"Model2Vec static embedding model (default: {DEFAULT_SIMILARITY_MODEL}).",
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
        default=16,
        help="Batch size for structured Gemini prompt translations (default: 16).",
    )
    config_group.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent translation worker threads (default: 4).",
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

    # Language utility options
    info_group = parser.add_argument_group("Language Utilities (afriso)")
    info_group.add_argument(
        "--list-languages",
        action="store_true",
        help="List all supported African languages in the afriso database.",
    )
    info_group.add_argument(
        "--info-language",
        metavar="LANG",
        help="Look up and display metadata for a specific African language.",
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

    # 1. Utility: List languages
    if parsed_args.list_languages:
        langs = list_supported_languages()
        print(f"Total African languages supported via afriso: {len(langs)}")
        for l in sorted(langs):
            print(f"  - {l}")
        return 0

    # 2. Utility: Look up specific language
    if parsed_args.info_language:
        info = resolve_language(parsed_args.info_language)
        print(f"Language Name : {info.name}")
        print(f"Prompt Name   : {info.prompt_name}")
        print(f"ISO 639-3     : {info.iso639_3}")
        print(f"ISO 639-1     : {info.iso639_1}")
        print(f"Family        : {info.family}")
        print(f"Regions       : {', '.join(info.regions) if info.regions else 'N/A'}")
        print(f"Countries     : {', '.join(info.countries) if info.countries else 'N/A'}")
        print(f"Is African    : {info.is_african}")
        return 0

    # 3. Load inputs
    predictions: List[str] = []
    references: List[str] = []
    languages: Optional[List[str]] = None

    if parsed_args.input_file:
        try:
            predictions, references, languages = _load_from_tabular(
                parsed_args.input_file,
                pred_col=parsed_args.pred_col,
                ref_col=parsed_args.ref_col,
                lang_col=parsed_args.lang_col,
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

    # Validate language
    active_language = languages if (languages and len(languages) > 0) else parsed_args.language
    if not active_language:
        print(
            "Error: Language must be specified using --language/-l or --lang-col in the dataset.",
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

    # Initialize evaluator
    try:
        scorer = GeminiScore(
            api_key=parsed_args.api_key,
            model=parsed_args.model,
            similarity_model=parsed_args.similarity_model,
            scale=parsed_args.scale,
            batch_size=parsed_args.batch_size,
            max_workers=parsed_args.workers,
        )
    except GeminiAuthError as auth_err:
        print(f"Authentication Error: {auth_err}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Initialization Error: {exc}", file=sys.stderr)
        return 1

    # Execute scoring
    try:
        result = scorer.score(
            predictions=predictions,
            references=references,
            language=active_language,
        )
    except Exception as exc:
        print(f"Evaluation Error: {exc}", file=sys.stderr)
        return 1

    # Print output
    if parsed_args.format == "score-only":
        print(f"{result.score:.4f}")
    elif parsed_args.format == "json":
        print(result.to_json())
    else:
        print(result.summary())

    # Save to file if requested
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
