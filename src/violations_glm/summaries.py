"""Transform stable model artifacts into figure-ready tables."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


def best_fits(
    fits: pd.DataFrame, *, groups: tuple[str, ...] = ("animal_id",)
) -> pd.DataFrame:
    indices = fits.groupby(list(groups), observed=False)["nll"].idxmin()
    return fits.loc[indices].reset_index(drop=True)


def unpack_weights(fits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, fit in fits.iterrows():
        features = json.loads(fit["features"])
        weights = np.asarray(json.loads(fit["weights"]))
        if weights.ndim == 1:
            weights = weights[:, None]
            classes = ("R-L",)
        else:
            classes = ("L", "R", "V")
        for feature_index, feature in enumerate(features):
            for class_index, weight_class in enumerate(classes):
                row = {
                    "animal_id": fit["animal_id"],
                    "model_name": fit["model_name"],
                    "sigma": fit["sigma"],
                    "nll": fit["nll"],
                    "feature": feature,
                    "weight_class": weight_class,
                    "weight": weights[feature_index, class_index],
                }
                if "tau" in fit:
                    row["tau"] = fit["tau"]
                rows.append(row)
    return pd.DataFrame(rows)


def bits_per_trial(fits: pd.DataFrame, nulls: pd.DataFrame) -> pd.DataFrame:
    selected = best_fits(fits, groups=("animal_id", "model_name"))
    columns = [
        "animal_id",
        "model_name",
        "model_type",
        "nll",
        "n_test_trials",
    ]
    combined = pd.concat([selected[columns], nulls[columns]], ignore_index=True)
    outputs = []
    for _, group in combined.groupby("animal_id", sort=False):
        group = group.copy()
        null_nll = group.loc[group["model_name"] == "null", "nll"].iloc[0]
        group["bits_per_trial"] = (null_nll - group["nll"]) / (
            group["n_test_trials"] * np.log(2)
        )
        base = group.loc[group["model_name"] == "base", "bits_per_trial"]
        if not base.empty:
            group["bits_per_trial_relative_to_base"] = (
                group["bits_per_trial"] - base.iloc[0]
            )
        outputs.append(group)
    return pd.concat(outputs, ignore_index=True)


def animal_model_deltas(
    fits: pd.DataFrame,
    nulls: pd.DataFrame,
    *,
    model_name: str,
) -> pd.DataFrame:
    """Return per-animal bits/trial for one model relative to the base model."""
    comparison = bits_per_trial(fits, nulls)
    return (
        comparison.loc[comparison["model_name"] == model_name]
        .loc[:, ["animal_id", "bits_per_trial", "bits_per_trial_relative_to_base"]]
        .sort_values("animal_id")
        .reset_index(drop=True)
    )
