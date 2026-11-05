"""Reusable behavioral transforms for publication figures."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from violations_glm.config import TRAINED_DELAYS

CHOICE_LABELS = {0: "Left", 1: "Right", 2: "Violation"}
RELATIVE_DELAY_ORDER = ("Same or shorter", "Longer")


def session_performance_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per session with hit rate, trial count, and violation rate."""
    return (
        frame.groupby(["animal_id", "session"], as_index=False)
        .agg(
            hit_rate=("hit", "mean"),
            n_trials=("trial", "max"),
            viol_rate=("violation", "mean"),
        )
        .sort_values(["animal_id", "session"])
        .reset_index(drop=True)
    )


def session_performance_percentages(frame: pd.DataFrame) -> pd.DataFrame:
    """Session performance with hit and violation rates on a percent scale."""
    summary = session_performance_summary(frame)
    summary = summary.copy()
    summary["hit_rate"] = summary["hit_rate"] * 100
    summary["viol_rate"] = summary["viol_rate"] * 100
    return summary


def animal_session_performance_means(session_summary: pd.DataFrame) -> pd.DataFrame:
    """Mean session hit rate, violation rate, and trial count per animal."""
    return (
        session_summary.groupby("animal_id", as_index=False)
        .agg(
            hit_rate=("hit_rate", "mean"),
            viol_rate=("viol_rate", "mean"),
            n_trials=("n_trials", "mean"),
        )
        .sort_values("animal_id")
        .reset_index(drop=True)
    )


def animal_training_predictors(
    all_trials: pd.DataFrame,
    thresholds: pd.DataFrame,
    trained_animals: pd.DataFrame,
    *,
    fixation_stage: int = 1,
) -> pd.DataFrame:
    """One row per trained animal with training-history predictors.

    Fixation-stage and trained rates are animal means of session percentages.
    ``sessions_to_trained`` is the session of the last 72.5% rolling hit-rate
    crossing (``hit_threshold``); trained sessions start after it.
    """
    stage = all_trials.loc[
        (all_trials["training_stage"] == fixation_stage)
        & all_trials["animal_id"].isin(trained_animals["animal_id"])
    ]
    stage_sessions = session_performance_percentages(stage)
    fixation = stage_sessions.groupby("animal_id", as_index=False).agg(
        fixation_stage_sessions=("session", "nunique"),
        fixation_stage_viol_rate=("viol_rate", "mean"),
    )
    trained = trained_animals.loc[:, ["animal_id", "hit_rate", "viol_rate"]].rename(
        columns={"hit_rate": "trained_hit_rate", "viol_rate": "trained_viol_rate"}
    )
    training_time = thresholds.loc[:, ["animal_id", "hit_threshold"]].rename(
        columns={"hit_threshold": "sessions_to_trained"}
    )
    predictors = trained.merge(fixation, on="animal_id", validate="one_to_one").merge(
        training_time, on="animal_id", validate="one_to_one"
    )
    if len(predictors) != len(trained_animals):
        missing = sorted(
            set(trained_animals["animal_id"]) - set(predictors["animal_id"])
        )
        raise ValueError(f"Missing training predictors for animals: {missing}")
    return predictors.sort_values("animal_id").reset_index(drop=True)


def animal_violation_rates(frame: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    """Trial-pooled violation percentage per animal and condition."""
    rates = frame.groupby(["animal_id", *by], as_index=False).agg(
        viol_rate=("violation", "mean"),
        mean_delay_time=("delay_time", "mean"),
        n_trials=("violation", "size"),
    )
    rates["viol_rate"] = rates["viol_rate"] * 100
    return rates


def animal_delay_violation_rates(
    trials: pd.DataFrame, *, delays: Sequence[float] = TRAINED_DELAYS
) -> pd.DataFrame:
    """Violation percentage per animal at each common delay length."""
    return animal_violation_rates(
        trials.loc[trials["delay_time"].isin(delays)], ["delay_time"]
    )


def relative_delay_trials(trials: pd.DataFrame) -> pd.DataFrame:
    """Label each trial by its delay relative to the previous trial.

    The first trial of each session has no previous trial and is dropped.
    """
    ordered = trials.sort_values(["animal_id", "session", "trial"])
    previous = ordered.groupby(["animal_id", "session"], sort=False)[
        "delay_time"
    ].shift()
    labeled = ordered.assign(previous_delay_time=previous).dropna(
        subset=["previous_delay_time"]
    )
    labeled["relative_delay"] = np.where(
        labeled["delay_time"] > labeled["previous_delay_time"],
        RELATIVE_DELAY_ORDER[1],
        RELATIVE_DELAY_ORDER[0],
    )
    return labeled


def session_position_trials(trials: pd.DataFrame) -> pd.DataFrame:
    """Add each trial's position from the session start and end.

    ``position`` counts from 0 at the first trial, ``trials_from_end`` from 0
    at the last trial, and ``session_progress`` runs from 0 to 1.
    """
    ordered = trials.sort_values(["animal_id", "session", "trial"])
    grouped = ordered.groupby(["animal_id", "session"], sort=False)
    position = grouped.cumcount()
    n_trials = grouped["trial"].transform("size")
    return ordered.assign(
        position=position,
        trials_from_end=n_trials - 1 - position,
        session_progress=position / (n_trials - 1),
    )


def trim_session_edges(trials: pd.DataFrame, n_edge: int) -> pd.DataFrame:
    """Drop the first and last ``n_edge`` trials of each session."""
    if n_edge <= 0:
        return trials.copy()
    positioned = session_position_trials(trials)
    keep = (positioned["position"] >= n_edge) & (
        positioned["trials_from_end"] >= n_edge
    )
    return trials.loc[keep.index[keep]].copy()


def animal_session_progress_violation_rates(
    trials: pd.DataFrame, *, n_bins: int = 10, edge_trials: int = 10
) -> pd.DataFrame:
    """Violation percentage per animal in equal fractions of each session.

    Normalizing by session length lets sessions of any length contribute to
    every bin. When ``edge_trials`` is positive, the first and last trials of
    each session are dropped so edge spikes do not mask gradual drift.
    """
    positioned = session_position_trials(trials)
    if edge_trials > 0:
        core = positioned.loc[
            (positioned["position"] >= edge_trials)
            & (positioned["trials_from_end"] >= edge_trials)
        ]
    else:
        core = positioned
    core = core.assign(
        progress_bin=np.minimum(
            (core["session_progress"] * n_bins).astype(int), n_bins - 1
        )
    )
    rates = core.groupby(["animal_id", "progress_bin"], as_index=False).agg(
        viol_rate=("violation", "mean"), n_trials=("violation", "size")
    )
    rates["viol_rate"] = rates["viol_rate"] * 100
    rates["session_progress"] = (rates["progress_bin"] + 0.5) * 100 / n_bins
    return rates


def animal_session_edge_rates(
    trials: pd.DataFrame, *, n_edge: int = 20
) -> pd.DataFrame:
    """Violation and hit percentages per animal at trials aligned to session edges.

    ``trial_offset`` is 1..n_edge from the session start and -n_edge..-1 from
    the session end (-1 is the final trial). Hit rate is over non-violation
    trials.
    """
    positioned = session_position_trials(trials)
    start = positioned.loc[positioned["position"] < n_edge].assign(
        aligned_to="Start", trial_offset=lambda frame: frame["position"] + 1
    )
    end = positioned.loc[positioned["trials_from_end"] < n_edge].assign(
        aligned_to="End", trial_offset=lambda frame: -(frame["trials_from_end"] + 1)
    )
    rates = (
        pd.concat([start, end], ignore_index=True)
        .groupby(["animal_id", "aligned_to", "trial_offset"], as_index=False)
        .agg(
            viol_rate=("violation", "mean"),
            hit_rate=("hit", "mean"),
            n_sessions=("session", "nunique"),
        )
    )
    rates[["viol_rate", "hit_rate"]] = rates[["viol_rate", "hit_rate"]] * 100
    return rates


def session_choice_percentages(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-session choice percentages for Left, Right, and Violation."""
    choice_dist = (
        frame.groupby(["animal_id", "session"])["choice"]
        .value_counts(normalize=True)
        .reset_index(name="proportion")
    )
    choice_dist["Choice"] = choice_dist["choice"].map(CHOICE_LABELS)
    choice_dist["percentage"] = (choice_dist["proportion"] * 100).round(2)
    return choice_dist


def session_window_trials(
    frame: pd.DataFrame,
    *,
    window_size: int = 50,
    minimum_session_trials: int = 100,
    maximum_session_violation_rate: float = 0.5,
) -> pd.DataFrame:
    """Extract matched start, middle, and end windows from valid sessions."""
    outputs: list[pd.DataFrame] = []
    for _, group in frame.groupby(["animal_id", "session"], sort=False):
        ordered = group.sort_values("trial")
        if (
            len(ordered) <= minimum_session_trials
            or ordered["violation"].mean() >= maximum_session_violation_rate
        ):
            continue
        middle_start = len(ordered) // 2 - window_size // 2
        windows = {
            "start": ordered.iloc[:window_size],
            "middle": ordered.iloc[middle_start : middle_start + window_size],
            "end": ordered.iloc[-window_size:],
        }
        for window, selected in windows.items():
            output = selected[["animal_id", "session", "trial", "violation"]].copy()
            output["window"] = window
            output["relative_trial"] = range(1, len(output) + 1)
            outputs.append(output)
    return pd.concat(outputs, ignore_index=True)


def summarize_session_windows(window_trials: pd.DataFrame) -> pd.DataFrame:
    """Average within animal before population-level plotting."""
    return window_trials.groupby(
        ["animal_id", "window", "relative_trial"], as_index=False
    ).agg(violation_rate=("violation", "mean"), sessions=("session", "nunique"))


def select_example_sessions(
    frame: pd.DataFrame,
    *,
    animal_id: str = "W065",
    sessions: tuple[int, ...] = (249, 250, 251),
) -> pd.DataFrame:
    selected = frame.loc[
        (frame["animal_id"] == animal_id) & frame["session"].isin(sessions)
    ].copy()
    missing = sorted(set(sessions) - set(selected["session"].unique()))
    if missing:
        raise ValueError(f"Missing example sessions for {animal_id}: {missing}")
    return selected.sort_values(["session", "trial"])


def stimulus_pair_summary(
    frame: pd.DataFrame,
    *,
    min_training_stage: int = 4,
) -> pd.DataFrame:
    """Per-animal stimulus-pair trial counts for stage ≥ 4.

    One row per ``(animal_id, s_a, s_b)`` with trial count and ``n_stim_pairs``
    for Supporting Figure S2 titles. Animals are ordered alphabetically.
    """
    required = {"animal_id", "training_stage", "s_a", "s_b"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"stimulus_pair_summary missing columns: {missing}")

    plot_df = frame.loc[frame["training_stage"] >= min_training_stage]
    summary = (
        plot_df.groupby(["animal_id", "s_a", "s_b"], as_index=False)
        .size()
        .rename(columns={"size": "trial_count"})
        .sort_values(["animal_id", "s_a", "s_b"])
        .reset_index(drop=True)
    )
    n_pairs = (
        summary.groupby("animal_id", sort=True)
        .size()
        .rename("n_stim_pairs")
        .reset_index()
    )
    return summary.merge(n_pairs, on="animal_id", validate="many_to_one")
