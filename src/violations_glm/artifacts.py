"""Stable model-summary serialization and legacy regression checks."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from violations_glm.fitting import ExperimentSpec
from violations_glm.preprocessing import file_sha256


def write_fit_artifact(
    destination: Path,
    *,
    fits: pd.DataFrame,
    nulls: pd.DataFrame | None,
    spec: ExperimentSpec,
    data_path: Path,
    random_state: int,
    session_splits: dict[str, dict[str, list[int]]],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    fits.to_parquet(destination / "fit_models.parquet", index=False)
    if nulls is not None:
        nulls.to_parquet(destination / "null_models.parquet", index=False)
    metadata = {
        "spec": asdict(spec),
        "data": {
            "filename": data_path.name,
            "sha256": file_sha256(data_path),
        },
        "random_state": random_state,
        "fit_rows": len(fits),
        "animal_ids": sorted(fits["animal_id"].unique()),
        "session_splits": session_splits,
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def compare_fit_tables(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    rtol: float = 1e-7,
    atol: float = 1e-5,
    ignore_weight_models: frozenset[str] = frozenset(),
    ignore_numeric_models: frozenset[str] = frozenset(),
    numeric_columns: tuple[str, ...] = ("nll", "train_nll"),
) -> dict:
    keys = ["animal_id", "model_name", "sigma"]
    if "tau" in observed.columns or "tau" in expected.columns:
        keys.append("tau")

    expected_subset = expected.merge(
        observed[keys].drop_duplicates(), on=keys, how="inner"
    )
    observed_sorted = observed.sort_values(keys).reset_index(drop=True)
    expected_sorted = expected_subset.sort_values(keys).reset_index(drop=True)
    if len(observed_sorted) != len(expected_sorted):
        raise AssertionError(
            f"Fit row mismatch: {len(observed_sorted)} != {len(expected_sorted)}"
        )

    exact_columns = [
        "animal_id",
        "model_name",
        "model_type",
        "sigma",
        "features",
        "n_train_trials",
        "n_test_trials",
    ]
    if "tau" in keys:
        exact_columns.append("tau")
    for column in exact_columns:
        if not observed_sorted[column].equals(expected_sorted[column]):
            raise AssertionError(f"Mismatch in exact column {column}")

    numeric_differences: dict[str, float] = {}
    numeric_rows = observed_sorted.loc[
        ~observed_sorted["model_name"].isin(ignore_numeric_models)
    ]
    expected_numeric_rows = expected_sorted.loc[
        ~expected_sorted["model_name"].isin(ignore_numeric_models)
    ]
    for column in numeric_columns:
        observed_values = numeric_rows[column].astype(float).to_numpy()
        expected_values = expected_numeric_rows[column].astype(float).to_numpy()
        if observed_values.size == 0:
            numeric_differences[column] = 0.0
            continue
        numeric_differences[column] = float(
            np.max(np.abs(observed_values - expected_values))
        )
        np.testing.assert_allclose(
            observed_values,
            expected_values,
            rtol=rtol,
            atol=atol,
        )

    maximum_weight_difference = 0.0
    weight_rows = observed_sorted.loc[
        ~observed_sorted["model_name"].isin(ignore_weight_models)
    ]
    expected_weight_rows = expected_sorted.loc[
        ~expected_sorted["model_name"].isin(ignore_weight_models)
    ]
    if weight_rows.empty:
        numeric_differences["weights"] = 0.0
        numeric_differences["ignored_weight_models"] = sorted(ignore_weight_models)
        numeric_differences["ignored_numeric_models"] = sorted(ignore_numeric_models)
        return numeric_differences
    for observed_json, expected_json in zip(
        weight_rows["weights"], expected_weight_rows["weights"], strict=True
    ):
        observed_weights = np.asarray(json.loads(observed_json))
        expected_weights = np.asarray(json.loads(expected_json))
        maximum_weight_difference = max(
            maximum_weight_difference,
            float(np.max(np.abs(observed_weights - expected_weights))),
        )
        np.testing.assert_allclose(
            observed_weights,
            expected_weights,
            rtol=rtol,
            atol=atol,
        )
    numeric_differences["weights"] = maximum_weight_difference
    numeric_differences["ignored_weight_models"] = sorted(ignore_weight_models)
    numeric_differences["ignored_numeric_models"] = sorted(ignore_numeric_models)
    return numeric_differences
