"""Collect per-animal old/new reconstruction alignment summaries for S1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from violations_glm.config import ALL_ANIMAL_IDS, MODEL_ANIMAL_IDS
from violations_glm.paths import processed_data_dir, results_dir

EXAMPLE_ANIMALS = ("W065", "W068", "W089")
EXAMPLE_ALIGNMENT_ANIMAL = "W089"


def _alignment_tables(legacy_repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = legacy_repo / "data/processed/dataset_alignment"
    summaries = []
    for animal_id in ALL_ANIMAL_IDS:
        path = source / f"{animal_id}_alignment_df.csv"
        summaries.append(pd.read_csv(path))
    summary = pd.concat(summaries, ignore_index=True)

    wide = summary.pivot(index="animal_id", columns="source")
    deltas = pd.DataFrame({"animal_id": list(ALL_ANIMAL_IDS)})
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
    return summary, deltas


def _example_animal_sessions(legacy_repo: Path) -> pd.DataFrame:
    """Session rates from the public Akrami rat_behavior table (violations end)."""
    path = legacy_repo / "data/raw/rat_behavior.csv"
    frame = pd.read_csv(path)
    frame["violation"] = frame["hit"].isnull()
    example = (
        frame.loc[frame["subject_id"].isin(EXAMPLE_ANIMALS)]
        .groupby(["subject_id", "session"], as_index=False)
        .agg(
            hit_rate=("hit", "mean"),
            viol_rate=("violation", "mean"),
            stage=("training_stage", "max"),
        )
    )
    return example.rename(columns={"subject_id": "animal_id"})


def _session_violation_stats(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby(["animal_id", "session"])["violation"].mean().agg(
        mean_violation="mean",
        std_violation="std",
    )


def _dataset_comparison(legacy_repo: Path) -> tuple[pd.DataFrame, dict]:
    """Reproduce the source notebook old / new_all / new_match_old_viols prints."""
    old = pd.read_csv(
        legacy_repo / "data/cleaned/old_dataset/old_violation_data_copy.csv"
    )
    old = old.query("animal_id in @MODEL_ANIMAL_IDS").copy()

    new_all = pd.read_csv(legacy_repo / "data/cleaned/all_animals_cleaned.csv")
    new_all = new_all.query("animal_id in @MODEL_ANIMAL_IDS").copy()
    new_match = new_all.query("session_relative_to_old < 200").copy()

    rows = []
    for label, subset in (
        ("old", old),
        ("new_all", new_all),
        ("new_match_old_viols", new_match),
    ):
        stats = _session_violation_stats(subset)
        rows.append(
            {
                "index": "mean_violation",
                "violation": float(stats["mean_violation"]),
                "data_type": label,
            }
        )
        rows.append(
            {
                "index": "std_violation",
                "violation": float(stats["std_violation"]),
                "data_type": label,
            }
        )

    counts = {
        "n_trials_new_all": int(len(new_all)),
        "n_sessions_new_all": int(new_all.groupby("animal_id")["session"].max().sum()),
    }
    return pd.DataFrame(rows), counts


def _best_session_shift(old_hits: pd.Series, new_hits: pd.Series) -> int:
    """Choose new→old session offset that maximizes early-session hit correlation."""
    best_shift = 0
    best_corr = -1.0
    for shift in range(-5, 6):
        paired = []
        for session, value in new_hits.items():
            old_session = session + shift
            if old_session in old_hits.index:
                paired.append((value, old_hits.loc[old_session]))
        if len(paired) < 50:
            continue
        new_vals, old_vals = zip(*paired[:200], strict=True)
        corr = float(pd.Series(new_vals).corr(pd.Series(old_vals)))
        if corr > best_corr:
            best_corr = corr
            best_shift = shift
    return best_shift


def _example_alignment_curves(
    legacy_repo: Path,
    *,
    animal_id: str = EXAMPLE_ALIGNMENT_ANIMAL,
) -> tuple[pd.DataFrame, dict]:
    """Session hit rate and trial counts for S1 panels C–D (W089 mega-fig subplots).

    Reproduces ``DatasetVisualizer.plot_hit_rates`` / ``plot_trial_counts`` using the
    public Akrami table (old) and cleaned reconstructed trials (new). Cleaned new
    data already excludes timeout trials, matching the hit-rate mega-fig panel
    (without timeouts); trial counts therefore approximate the with-timeouts panel.
    """
    old = pd.read_csv(legacy_repo / "data/raw/rat_behavior.csv")
    old = old.loc[old["subject_id"] == animal_id].copy()
    old_sessions = old.groupby("session", as_index=True).agg(
        hit_rate=("hit", "mean"),
        n_trials=("trial", "max"),
    )

    new_path = processed_data_dir() / "all_animals_cleaned.parquet"
    new = pd.read_parquet(new_path)
    new = new.loc[new["animal_id"] == animal_id].copy()
    new_sessions = new.groupby("session", as_index=True).agg(
        hit_rate=("hit", "mean"),
        n_trials=("trial", "max"),
    )

    shift = _best_session_shift(old_sessions["hit_rate"], new_sessions["hit_rate"])
    new_aligned = new_sessions.reset_index()
    new_aligned["session"] = new_aligned["session"] + shift

    old_long = old_sessions.reset_index().assign(source="old", animal_id=animal_id)
    new_long = new_aligned.assign(source="new", animal_id=animal_id)
    curves = pd.concat([old_long, new_long], ignore_index=True)
    curves = curves.sort_values(["source", "session"]).reset_index(drop=True)
    meta = {
        "animal_id": animal_id,
        "new_session_shift": int(shift),
        "n_sessions_old": int(old_sessions.shape[0]),
        "n_sessions_new": int(new_sessions.shape[0]),
    }
    return curves, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-repo", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=results_dir() / "alignment",
    )
    args = parser.parse_args()

    summary, deltas = _alignment_tables(args.legacy_repo)
    example_sessions = _example_animal_sessions(args.legacy_repo)
    comparison, counts = _dataset_comparison(args.legacy_repo)
    alignment_curves, alignment_meta = _example_alignment_curves(args.legacy_repo)

    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "alignment_summary.csv", index=False)
    deltas.to_csv(args.output / "alignment_deltas.csv", index=False)
    example_sessions.to_csv(args.output / "example_animal_sessions.csv", index=False)
    comparison.to_csv(args.output / "dataset_comparison_stats.csv", index=False)
    (args.output / "dataset_counts.json").write_text(json.dumps(counts, indent=2))
    alignment_curves.to_csv(args.output / "example_alignment_curves.csv", index=False)
    (args.output / "example_alignment_meta.json").write_text(
        json.dumps(alignment_meta, indent=2)
    )


if __name__ == "__main__":
    main()
