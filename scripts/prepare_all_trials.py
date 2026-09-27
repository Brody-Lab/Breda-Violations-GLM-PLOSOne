"""Convert the reconstructed all-stage CSV to a publication-only Parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from violations_glm.config import ALL_ANIMAL_IDS
from violations_glm.paths import processed_data_dir
from violations_glm.preprocessing import dataset_counts, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=processed_data_dir() / "all_animals_cleaned.parquet",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    frame = frame.loc[frame["animal_id"].isin(ALL_ANIMAL_IDS)].copy()
    missing = sorted(set(ALL_ANIMAL_IDS) - set(frame["animal_id"].unique()))
    if missing:
        raise ValueError(f"Missing publication animals: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)

    manifest = {
        "producer": "scripts/prepare_all_trials.py",
        "input": {
            "filename": args.input.name,
            "sha256": file_sha256(args.input),
        },
        "output": {
            "filename": args.output.name,
            "sha256": file_sha256(args.output),
            **dataset_counts(frame).__dict__,
        },
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest["output"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
