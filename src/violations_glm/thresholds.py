"""Training-window thresholding used for Supporting Figure S3."""

from __future__ import annotations

import numpy as np
import pandas as pd

VIOLATION_TAIL_ANIMALS = ("W065", "W080", "W083", "W088")


def session_performance(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["animal_id", "session", "training_stage"], as_index=False)
        .agg(hit=("hit", "mean"), violation=("violation", "mean"))
        .sort_values(["animal_id", "session"])
        .reset_index(drop=True)
    )


def rolling_performance(frame: pd.DataFrame) -> pd.DataFrame:
    performance = session_performance(frame)
    performance["rolling_mean_hit"] = performance.groupby("animal_id", sort=False)[
        "hit"
    ].transform(lambda values: values.rolling(21, min_periods=1).mean())
    performance["rolling_mean_violation"] = performance.groupby(
        "animal_id", sort=False
    )["violation"].transform(lambda values: values.rolling(7, min_periods=1).mean())
    return performance


def _last_low_to_high_crossing(
    group: pd.DataFrame, column: str, threshold: float
) -> float:
    values = group[column]
    crossed = (values.shift() < threshold) & (values >= threshold)
    sessions = group.loc[crossed, "session"]
    return float(sessions.max()) if not sessions.empty else np.nan


def threshold_sessions(rolling: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for animal_id, group in rolling.groupby("animal_id", sort=False):
        stage_four = group.loc[group["training_stage"] > 3]
        hit_threshold = _last_low_to_high_crossing(
            stage_four, "rolling_mean_hit", 0.725
        )
        violation_threshold = _last_low_to_high_crossing(
            stage_four, "rolling_mean_violation", 0.5
        )
        if animal_id not in VIOLATION_TAIL_ANIMALS:
            violation_threshold = np.nan
        rows.append(
            {
                "animal_id": animal_id,
                "hit_threshold": hit_threshold,
                "violation_threshold": violation_threshold,
            }
        )
    thresholds = pd.DataFrame(rows)
    if thresholds["hit_threshold"].isna().any():
        missing = thresholds.loc[
            thresholds["hit_threshold"].isna(), "animal_id"
        ].tolist()
        raise ValueError(f"No 72.5% hit-rate crossing for animals: {missing}")
    return thresholds


def apply_training_thresholds(
    frame: pd.DataFrame, thresholds: pd.DataFrame
) -> pd.DataFrame:
    outputs = []
    indexed = thresholds.set_index("animal_id")
    for animal_id, group in frame.groupby("animal_id", sort=False):
        hit = indexed.loc[animal_id, "hit_threshold"]
        violation = indexed.loc[animal_id, "violation_threshold"]
        selected = group.loc[group["session"] > hit]
        if not np.isnan(violation):
            selected = selected.loc[selected["session"] < violation]
        outputs.append(selected)
    output = pd.concat(outputs, ignore_index=True)
    output["animal_id_session"] = (
        output["animal_id"] + "_s" + output["session"].astype(int).astype(str)
    )
    return output
