"""Deterministic preparation of the publication modeling dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from violations_glm.config import ANIMAL_IDS


@dataclass(frozen=True)
class DatasetCounts:
    rows: int
    sessions: int
    animals: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_digest(frame: pd.DataFrame) -> str:
    """Hash ordered values, columns, and dtypes independent of file encoding."""
    digest = hashlib.sha256()
    digest.update(json.dumps(frame.columns.tolist()).encode())
    digest.update(json.dumps(frame.dtypes.astype(str).tolist()).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def dataset_counts(frame: pd.DataFrame) -> DatasetCounts:
    sessions = frame[["animal_id", "session"]].drop_duplicates().shape[0]
    return DatasetCounts(
        rows=len(frame),
        sessions=sessions,
        animals=frame["animal_id"].nunique(),
    )


def select_publication_animals(frame: pd.DataFrame) -> pd.DataFrame:
    """Restrict an upstream artifact to the submitted 16-animal GLM cohort."""
    selected = frame.loc[frame["animal_id"].isin(ANIMAL_IDS)].copy()
    observed = set(selected["animal_id"].unique())
    missing = sorted(set(ANIMAL_IDS) - observed)
    if missing:
        raise ValueError(f"Input data are missing publication animals: {missing}")
    return selected


def add_animal_session_id(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["animal_id_session"] = (
        output["animal_id"].astype(str)
        + "_s"
        + output["session"].astype(int).astype(str)
    )
    return output


def filter_modeling_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the session-quality filter used for the submitted manuscript."""
    session_stats = (
        frame.groupby(["animal_id", "session"], sort=False)
        .agg(
            hit_rate=("hit", "mean"),
            n_trials=("trial", "max"),
            violation_rate=("violation", "mean"),
        )
        .reset_index()
    )
    valid = session_stats.loc[
        (session_stats["n_trials"] >= 50)
        & (session_stats["violation_rate"] <= 0.99)
        & session_stats["hit_rate"].notna()
        & session_stats["violation_rate"].notna(),
        ["animal_id", "session"],
    ]
    return frame.merge(valid, on=["animal_id", "session"], how="inner")


def prepare_modeling_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build the exact trial table consumed by publication model fits."""
    required = {"animal_id", "session", "trial", "hit", "violation"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input data are missing required columns: {missing}")

    stages: dict[str, dict] = {"input": asdict(dataset_counts(frame))}
    publication = select_publication_animals(frame)
    stages["publication_animals"] = asdict(dataset_counts(publication))
    with_ids = add_animal_session_id(publication)
    output = filter_modeling_sessions(with_ids)
    stages["session_quality_filter"] = asdict(dataset_counts(output))
    return output, stages
