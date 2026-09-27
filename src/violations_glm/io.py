"""Named loaders with publication-data validation."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from violations_glm.config import ALL_ANIMAL_IDS, MODEL_ANIMAL_IDS
from violations_glm.paths import model_summaries_dir, processed_data_dir, results_dir

MODEL_COLUMNS = {
    "animal_id",
    "session",
    "trial",
    "s_a",
    "s_b",
    "choice",
    "violation",
    "correct_side",
    "n_prev_trial_not_started",
}


def _validate_columns(
    frame: pd.DataFrame, required: Iterable[str], *, source: Path
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _validate_animals(
    frame: pd.DataFrame,
    *,
    expected: tuple[str, ...],
    source: Path,
) -> None:
    observed = set(frame["animal_id"].unique())
    unexpected = sorted(observed - set(expected))
    if unexpected:
        raise ValueError(f"{source} contains non-publication animals: {unexpected}")


def load_trials_df(path: Path | None = None) -> pd.DataFrame:
    """Load the final 16-animal trial dataset used by the manuscript."""
    source = path or processed_data_dir() / "all_animals_trained_cleaned.parquet"
    if not source.is_file():
        raise FileNotFoundError(
            f"Publication trial data not found at {source}. "
            "Run scripts/prepare_data.py or set VIOLATIONS_GLM_DATA_DIR."
        )
    frame = pd.read_parquet(source)
    _validate_columns(frame, MODEL_COLUMNS, source=source)
    _validate_animals(frame, expected=MODEL_ANIMAL_IDS, source=source)
    return frame


def load_all_trials_df(path: Path | None = None) -> pd.DataFrame:
    """Load all reconstructed training stages for publication supplements."""
    source = path or processed_data_dir() / "all_animals_cleaned.parquet"
    if not source.is_file():
        raise FileNotFoundError(
            f"All-stage trial data not found at {source}. "
            "Prepare it from the Zenodo all_animals_cleaned.csv artifact."
        )
    frame = pd.read_parquet(source)
    _validate_columns(
        frame,
        {"animal_id", "session", "trial", "training_stage", "hit", "violation"},
        source=source,
    )
    _validate_animals(frame, expected=ALL_ANIMAL_IDS, source=source)
    return frame


def load_model_summary(
    artifact: str,
    *,
    kind: str = "fit_models",
    collection: str | None = None,
) -> pd.DataFrame:
    """Load a stable model-fit summary, never a serialized experiment object."""
    collection = collection or os.environ.get(
        "VIOLATIONS_GLM_MODEL_COLLECTION", "legacy"
    )
    source = model_summaries_dir() / collection / artifact / f"{kind}.parquet"
    if not source.is_file():
        raise FileNotFoundError(source)
    return pd.read_parquet(source)


def _read_table(source: Path) -> pd.DataFrame:
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix == ".csv":
        return pd.read_csv(source)
    return pd.read_parquet(source)


def load_autocorrelation_summary(*, edges_trimmed: bool = False) -> pd.DataFrame:
    """Load the corrected Figure 3 session-level autocorrelation table."""
    directory = (
        "fig03_autocorrelation_edges_trimmed"
        if edges_trimmed
        else "fig03_autocorrelation"
    )
    return _read_table(results_dir() / directory / "summary.parquet")


def load_alignment_deltas() -> pd.DataFrame:
    """Load per-animal reconstruction alignment deltas for Supporting Figure S1."""
    return _read_table(results_dir() / "alignment" / "alignment_deltas.csv")


def load_session_window_summary() -> pd.DataFrame:
    """Load Supporting Figure S5 session-position violation rates."""
    return _read_table(
        results_dir() / "revision_behavior" / "suppS5_session_window_summary.parquet"
    )


def load_training_summary() -> pd.DataFrame:
    """Load Supporting Figure S5 earlier-stage versus stage-4 summaries."""
    return _read_table(
        results_dir() / "revision_behavior" / "suppS5_training_summary.csv"
    )


def load_training_statistics() -> dict:
    """Load Supporting Figure S5 correlation statistics."""
    source = results_dir() / "revision_behavior" / "suppS5_training_statistics.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text())


def load_training_predictors() -> pd.DataFrame:
    """Load per-animal training-history predictors of trained violation rate."""
    return _read_table(
        results_dir() / "revision_behavior" / "suppS5_training_predictors.csv"
    )


def load_training_predictor_statistics() -> dict:
    """Load animal-level associations with trained violation rate."""
    source = (
        results_dir()
        / "revision_behavior"
        / "suppS5_training_predictor_statistics.json"
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text())


def load_delay_violation_rates() -> pd.DataFrame:
    """Load per-animal violation rates at each common delay length."""
    return _read_table(
        results_dir() / "revision_behavior" / "suppS5_delay_violation_rates.csv"
    )


def load_relative_delay_violation_rates(*, matched: bool = False) -> pd.DataFrame:
    """Load per-animal violation rates by delay relative to the previous trial.

    ``matched=True`` returns rates split further by the current delay.
    """
    stem = (
        "suppS5_relative_delay_matched_rates"
        if matched
        else ("suppS5_relative_delay_violation_rates")
    )
    return _read_table(results_dir() / "revision_behavior" / f"{stem}.csv")


def load_delay_statistics() -> dict:
    """Load delay-length and relative-delay test statistics."""
    source = results_dir() / "revision_behavior" / "suppS5_delay_statistics.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text())


def load_session_progress_violation_rates(
    *, include_edges: bool = False
) -> pd.DataFrame:
    """Load per-animal violation rates by fraction of session completed."""
    stem = (
        "suppS5_session_progress_rates_all_trials"
        if include_edges
        else "suppS5_session_progress_rates"
    )
    return _read_table(results_dir() / "revision_behavior" / f"{stem}.csv")


def load_history_robustness() -> dict:
    """Load full vs edge-trimmed post-violation effect summaries."""
    source = results_dir() / "revision_behavior" / "suppS5_history_robustness.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text())


def load_session_edge_rates() -> pd.DataFrame:
    """Load per-animal violation and hit rates aligned to session start and end."""
    return _read_table(
        results_dir() / "revision_behavior" / "suppS5_session_edge_rates.csv"
    )


def load_session_position_statistics() -> dict:
    """Load the per-animal session-progress slope test."""
    source = (
        results_dir() / "revision_behavior" / "suppS5_session_position_statistics.json"
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text())


def load_hit_violation_sessions() -> pd.DataFrame:
    """Load session-level hit and violation rates on a percent scale."""
    return _read_table(
        results_dir() / "revision_behavior" / "suppS5_hit_violation_sessions.parquet"
    )


def load_hit_violation_animals() -> pd.DataFrame:
    """Load animal-mean session hit and violation rates on a percent scale."""
    return _read_table(
        results_dir() / "revision_behavior" / "suppS5_hit_violation_animals.parquet"
    )


def load_hit_violation_residuals() -> pd.DataFrame:
    """Load MixedLM fitted values and residuals for the S5 diagnostic plot."""
    return _read_table(
        results_dir()
        / "revision_behavior"
        / "suppS5_hit_violation_mixedlm_residuals.parquet"
    )


def load_hit_violation_statistics() -> dict:
    """Load MixedLM, OLS, and animal-mean regression summaries for S5 Panel A."""
    source = (
        results_dir() / "revision_behavior" / "suppS5_hit_violation_statistics.json"
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text())


def load_rolling_performance() -> pd.DataFrame:
    """Load rolling hit and violation rates for Supporting Figure S3."""
    return _read_table(
        results_dir() / "data_preparation" / "rolling_performance.parquet"
    )


def load_threshold_sessions() -> pd.DataFrame:
    """Load trained-window crossings for Supporting Figure S3."""
    return _read_table(results_dir() / "data_preparation" / "threshold_sessions.csv")
