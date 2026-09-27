"""Prepare the exact 16-animal trial dataset used for model fitting."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd

from violations_glm.paths import processed_data_dir
from violations_glm.preprocessing import (
    dataframe_digest,
    dataset_counts,
    file_sha256,
    prepare_modeling_dataset,
)

SUBMITTED_COUNTS = {"rows": 1_866_003, "sessions": 5_210, "animals": 16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to all_animals_trained_threshold.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=processed_data_dir() / "all_animals_trained_cleaned.parquet",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Optional submitted cleaned parquet for exact logical comparison.",
    )
    parser.add_argument(
        "--allow-count-difference",
        action="store_true",
        help="Do not require the submitted row/session/animal counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    frame = pd.read_parquet(source)
    output, stages = prepare_modeling_dataset(frame)
    counts = dataset_counts(output).__dict__
    if not args.allow_count_difference and counts != SUBMITTED_COUNTS:
        raise ValueError(f"Expected {SUBMITTED_COUNTS}, observed {counts}")

    logical_digest = dataframe_digest(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)

    manifest = {
        "producer": "scripts/prepare_data.py",
        "input": {
            "path": source.name,
            "sha256": file_sha256(source),
        },
        "stages": stages,
        "output": {
            "path": args.output.name,
            "sha256": file_sha256(args.output),
            "logical_sha256": logical_digest,
            **counts,
        },
        "filters": {
            "animals": "16-animal submitted cohort",
            "minimum_trials_per_session": 50,
            "maximum_violation_rate": 0.99,
            "require_nonmissing_hit_and_violation_rates": True,
        },
    }

    if args.reference:
        del frame, output
        gc.collect()
        reference = pd.read_parquet(args.reference)
        reference_digest = dataframe_digest(reference)
        manifest["reference"] = {
            "path": args.reference.name,
            "sha256": file_sha256(args.reference),
            "logical_sha256": reference_digest,
            **dataset_counts(reference).__dict__,
        }
        if logical_digest != reference_digest:
            raise ValueError("Generated data do not match the submitted reference")

    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["output"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
