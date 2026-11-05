"""Prepare Supporting Figure S2 stimulus-pair summaries from all-stage trials."""

from __future__ import annotations

import argparse
from pathlib import Path

from violations_glm.config import MODEL_ANIMAL_IDS
from violations_glm.io import load_all_trials_df
from violations_glm.paths import results_dir
from violations_glm.transforms import stimulus_pair_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=results_dir() / "stimulus_pairs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trials = load_all_trials_df(args.data)
    model_trials = trials.loc[trials["animal_id"].isin(MODEL_ANIMAL_IDS)].copy()
    summary = stimulus_pair_summary(model_trials)

    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(args.output / "stimulus_pair_summary.parquet", index=False)
    print(
        f"Wrote stimulus-pair summary for {summary['animal_id'].nunique()} animals "
        f"({len(summary)} pairs) to {args.output}"
    )


if __name__ == "__main__":
    main()
