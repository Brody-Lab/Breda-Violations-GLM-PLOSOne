"""Prepare revised Figure 1 and Supporting Figure S5 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from violations_glm.config import ANIMAL_IDS, TRAINED_DELAYS
from violations_glm.io import (
    load_all_trials_df,
    load_threshold_sessions,
    load_trials_df,
)
from violations_glm.paths import results_dir
from violations_glm.stats import (
    animal_level_association,
    delay_violation_statistics,
    hit_violation_regressions,
    post_violation_effect,
    relative_delay_statistics,
    session_progress_statistics,
)
from violations_glm.transforms import (
    animal_delay_violation_rates,
    animal_session_edge_rates,
    animal_session_performance_means,
    animal_session_progress_violation_rates,
    animal_training_predictors,
    animal_violation_rates,
    relative_delay_trials,
    select_example_sessions,
    session_performance_percentages,
    session_window_trials,
    summarize_session_windows,
    trim_session_edges,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument(
        "--all-training-data",
        type=Path,
        help="Optional all_animals_cleaned.csv for the earlier-stage panel.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=results_dir() / "revision_behavior",
    )
    return parser.parse_args()


def training_summary(path: Path, trained: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    columns = ["animal_id", "session", "training_stage", "violation"]
    all_training = pd.read_csv(path, usecols=columns)
    all_training = all_training.loc[all_training["animal_id"].isin(ANIMAL_IDS)].copy()
    early = (
        all_training.loc[all_training["training_stage"] <= 3]
        .groupby("animal_id", as_index=False)
        .agg(
            early_violation_rate=("violation", "mean"),
            early_trials=("violation", "size"),
            early_sessions=("session", "nunique"),
        )
    )
    first_stage_four = (
        all_training.loc[all_training["training_stage"] == 4]
        .groupby("animal_id", as_index=False)
        .agg(first_stage_four_session=("session", "min"))
    )
    current = trained.groupby("animal_id", as_index=False).agg(
        stage_four_violation_rate=("violation", "mean")
    )
    summary = early.merge(first_stage_four, on="animal_id").merge(
        current, on="animal_id"
    )

    statistics = {}
    for predictor in ("early_violation_rate", "first_stage_four_session"):
        pearson = pearsonr(summary[predictor], summary["stage_four_violation_rate"])
        spearman = spearmanr(summary[predictor], summary["stage_four_violation_rate"])
        statistics[predictor] = {
            "pearson_r": pearson.statistic,
            "pearson_p": pearson.pvalue,
            "spearman_r": spearman.statistic,
            "spearman_p": spearman.pvalue,
            "n_animals": len(summary),
        }
    return summary, statistics


TRAINING_PREDICTORS = (
    "trained_hit_rate",
    "fixation_stage_viol_rate",
    "fixation_stage_sessions",
    "sessions_to_trained",
)


def write_training_predictor_artifacts(animals: pd.DataFrame, output: Path) -> None:
    all_trials = load_all_trials_df()
    model_trials = all_trials["animal_id"].isin(animals["animal_id"])
    print(
        f"training predictors: excluded {int((~model_trials).sum())} trials from "
        f"non-model animals {sorted(set(all_trials.loc[~model_trials, 'animal_id']))}"
    )
    predictors = animal_training_predictors(
        all_trials.loc[model_trials], load_threshold_sessions(), animals
    )
    statistics = {
        predictor: animal_level_association(predictors, predictor, "trained_viol_rate")
        for predictor in TRAINING_PREDICTORS
    }
    predictors.to_csv(output / "suppS5_training_predictors.csv", index=False)
    (output / "suppS5_training_predictor_statistics.json").write_text(
        json.dumps(statistics, indent=2, sort_keys=True) + "\n"
    )


def write_delay_artifacts(trials: pd.DataFrame, output: Path) -> None:
    common = trials["delay_time"].isin(TRAINED_DELAYS)
    print(
        f"delay length: excluded {int((~common).sum())} trials with delays "
        f"outside {TRAINED_DELAYS}"
    )
    delay_rates = animal_delay_violation_rates(trials)

    labeled = relative_delay_trials(trials)
    print(
        f"relative delay: excluded {len(trials) - len(labeled)} first-of-session trials"
    )
    relative_rates = animal_violation_rates(labeled, ["relative_delay"])
    matched_rates = animal_violation_rates(
        labeled.loc[labeled["delay_time"].isin(TRAINED_DELAYS)],
        ["delay_time", "relative_delay"],
    )

    statistics = {
        "delay_length": delay_violation_statistics(delay_rates),
        "relative_delay": relative_delay_statistics(relative_rates, matched_rates),
    }
    delay_rates.to_csv(output / "suppS5_delay_violation_rates.csv", index=False)
    relative_rates.to_csv(
        output / "suppS5_relative_delay_violation_rates.csv", index=False
    )
    matched_rates.to_csv(
        output / "suppS5_relative_delay_matched_rates.csv", index=False
    )
    (output / "suppS5_delay_statistics.json").write_text(
        json.dumps(statistics, indent=2, sort_keys=True) + "\n"
    )


def write_session_position_artifacts(trials: pd.DataFrame, output: Path) -> None:
    edge_trials = 10
    progress_trimmed = animal_session_progress_violation_rates(
        trials, edge_trials=edge_trials
    )
    progress_all = animal_session_progress_violation_rates(trials, edge_trials=0)
    n_sessions = trials.groupby(["animal_id", "session"]).ngroups
    print(
        f"session progress: excluded {2 * edge_trials * n_sessions} trials "
        f"(first and last {edge_trials} of {n_sessions} sessions) for trimmed bins"
    )
    statistics = {
        "edge_trials_excluded": edge_trials,
        "trimmed": session_progress_statistics(progress_trimmed),
        "all_trials": session_progress_statistics(progress_all),
    }
    progress_trimmed.to_csv(output / "suppS5_session_progress_rates.csv", index=False)
    progress_all.to_csv(
        output / "suppS5_session_progress_rates_all_trials.csv", index=False
    )
    animal_session_edge_rates(trials).to_csv(
        output / "suppS5_session_edge_rates.csv", index=False
    )
    (output / "suppS5_session_position_statistics.json").write_text(
        json.dumps(statistics, indent=2, sort_keys=True) + "\n"
    )


def write_history_robustness_artifacts(trials: pd.DataFrame, output: Path) -> None:
    edge_trials = 10
    trimmed = trim_session_edges(trials, edge_trials)
    payload = {
        "edge_trials_excluded": edge_trials,
        "full": post_violation_effect(trials),
        "trimmed": post_violation_effect(trimmed),
    }
    (output / "suppS5_history_robustness.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def write_hit_violation_artifacts(trials: pd.DataFrame, output: Path) -> pd.DataFrame:
    sessions = session_performance_percentages(trials)
    animals = animal_session_performance_means(sessions)
    statistics, residuals = hit_violation_regressions(sessions, animals)

    sessions.to_parquet(output / "suppS5_hit_violation_sessions.parquet", index=False)
    animals.to_parquet(output / "suppS5_hit_violation_animals.parquet", index=False)
    residuals.to_parquet(
        output / "suppS5_hit_violation_mixedlm_residuals.parquet", index=False
    )
    (output / "suppS5_hit_violation_statistics.json").write_text(
        json.dumps(statistics, indent=2, sort_keys=True) + "\n"
    )
    return animals


def main() -> None:
    args = parse_args()
    trials = load_trials_df(args.data)
    args.output.mkdir(parents=True, exist_ok=True)

    examples = select_example_sessions(trials)
    examples.to_parquet(args.output / "fig01_example_sessions.parquet", index=False)

    windows = session_window_trials(trials)
    windows.to_parquet(args.output / "suppS5_session_windows.parquet", index=False)
    summarize_session_windows(windows).to_parquet(
        args.output / "suppS5_session_window_summary.parquet", index=False
    )

    animals = write_hit_violation_artifacts(trials, args.output)
    write_training_predictor_artifacts(animals, args.output)
    write_delay_artifacts(trials, args.output)
    write_session_position_artifacts(trials, args.output)
    write_history_robustness_artifacts(trials, args.output)

    if args.all_training_data:
        summary, statistics = training_summary(args.all_training_data, trials)
        summary.to_csv(args.output / "suppS5_training_summary.csv", index=False)
        (args.output / "suppS5_training_statistics.json").write_text(
            json.dumps(statistics, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
