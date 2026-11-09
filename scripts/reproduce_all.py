"""Orchestrate cached figure reproduction or a full publication refit."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from violations_glm.paths import processed_data_dir


def run(*arguments: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        [sys.executable, *arguments],
        check=True,
        env={**os.environ, **(env or {})},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cached", "full"), default="cached")
    parser.add_argument(
        "--all-training-data",
        type=Path,
        help="Required in full mode: all_animals_cleaned.csv.",
    )
    args = parser.parse_args()

    if args.mode == "full":
        if args.all_training_data is None:
            parser.error("--all-training-data is required in full mode")
        run(
            "scripts/prepare_all_trials.py",
            "--input",
            str(args.all_training_data),
        )
        run("scripts/prepare_full_pipeline.py")
        run(
            "scripts/prepare_revision_behavior.py",
            "--all-training-data",
            str(args.all_training_data),
        )
        for script in ("fit_fig03.py", "fit_fig04.py", "fit_fig05.py"):
            run(f"scripts/{script}", "--validate-legacy")
        run("scripts/compute_fig03_autocorrelation.py")

    required = processed_data_dir() / "all_animals_trained_cleaned.parquet"
    if not required.is_file():
        raise FileNotFoundError(
            f"{required} is required for cached notebook reproduction"
        )
    run("scripts/capture_environment.py")
    collection = "current" if args.mode == "full" else "legacy"
    run(
        "scripts/export_fig05_animal_deltas.py",
        env={"VIOLATIONS_GLM_MODEL_COLLECTION": collection},
    )
    run("scripts/prepare_fig06_across_animals.py")
    run("scripts/prepare_fig06_within_animal.py")
    run(
        "scripts/verify_cached.py",
        env={"VIOLATIONS_GLM_MODEL_COLLECTION": collection},
    )


if __name__ == "__main__":
    main()
