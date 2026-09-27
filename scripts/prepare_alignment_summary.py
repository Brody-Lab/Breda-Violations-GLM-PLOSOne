"""Collect per-animal old/new reconstruction alignment summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from violations_glm.config import ALL_ANIMAL_IDS
from violations_glm.paths import results_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-repo", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=results_dir() / "alignment",
    )
    args = parser.parse_args()

    source = args.legacy_repo / "data/processed/dataset_alignment"
    summaries = []
    for animal_id in ALL_ANIMAL_IDS:
        path = source / f"{animal_id}_alignment_df.csv"
        frame = pd.read_csv(path)
        summaries.append(frame)
    summary = pd.concat(summaries, ignore_index=True)

    wide = summary.pivot(index="animal_id", columns="source")
    deltas = pd.DataFrame({"animal_id": ALL_ANIMAL_IDS})
    deltas["delta_n_sessions"] = (
        wide["n_sessions"]["new"].reindex(ALL_ANIMAL_IDS).to_numpy()
        - wide["n_sessions"]["old"].reindex(ALL_ANIMAL_IDS).to_numpy()
    )
    for metric in (
        "n_trials_per_session",
        "session_avg_violation",
        "session_avg_violation_stage_4",
    ):
        deltas[f"delta_{metric}"] = (
            wide[metric]["new_no_touts"].reindex(ALL_ANIMAL_IDS).to_numpy()
            - wide[metric]["old"].reindex(ALL_ANIMAL_IDS).to_numpy()
        )

    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "alignment_summary.csv", index=False)
    deltas.to_csv(args.output / "alignment_deltas.csv", index=False)


if __name__ == "__main__":
    main()
