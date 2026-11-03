"""Transform stable model artifacts into figure-ready tables."""

from __future__ import annotations

import json
import math

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


def _best_fit_rows(
    fits: pd.DataFrame,
    base_model_name: str,
    new_model_name: str,
) -> list[tuple[float, float]]:
    """Return per-animal (base_nll, new_nll) for the best test fit of each model."""
    selected = best_fits(fits, groups=("animal_id", "model_name"))
    pairs: list[tuple[float, float]] = []
    for animal_id, group in selected.groupby("animal_id", sort=False):
        base_rows = group.loc[group["model_name"] == base_model_name]
        new_rows = group.loc[group["model_name"] == new_model_name]
        if base_rows.empty or new_rows.empty:
            continue
        pairs.append((float(base_rows["nll"].iloc[0]), float(new_rows["nll"].iloc[0])))
    return pairs


def overall_likelihood_ratio(
    fits: pd.DataFrame,
    *,
    base_model_name: str,
    new_model_name: str,
) -> dict[str, float]:
    """Pool log-likelihood improvements across animals (submitted Fig 3 comparison)."""
    pairs = _best_fit_rows(fits, base_model_name, new_model_name)
    if not pairs:
        raise ValueError("No animals have both the base and new model entries.")
    overall_diff_bits = 0.0
    for base_nll, new_nll in pairs:
        diff_bits = (base_nll - new_nll) / np.log(2)
        overall_diff_bits += diff_bits
    log10_overall_ratio = overall_diff_bits * math.log10(2)
    try:
        overall_ratio = 2**overall_diff_bits
    except OverflowError:
        overall_ratio = float("inf")
    if math.isinf(2**overall_diff_bits):
        overall_ratio = float("inf")
    return {
        "overall_ratio": overall_ratio,
        "log10_overall_ratio": log10_overall_ratio,
    }


def geometric_mean_likelihood_ratio(
    fits: pd.DataFrame,
    *,
    base_model_name: str,
    new_model_name: str,
    threshold_bits: float = 700,
) -> dict[str, float]:
    """Geometric mean of per-animal likelihood ratios (submitted Fig 3 comparison)."""
    pairs = _best_fit_rows(fits, base_model_name, new_model_name)
    diffs = [base_nll - new_nll for base_nll, new_nll in pairs]
    if not diffs:
        raise ValueError("No animals have both the base and new model entries.")
    mean_diff = float(np.mean(diffs))
    log10_geometric_mean_ratio = mean_diff / np.log(10)
    if mean_diff > threshold_bits:
        geometric_mean_ratio = float("inf")
    else:
        try:
            geometric_mean_ratio = math.exp(mean_diff)
        except OverflowError:
            geometric_mean_ratio = float("inf")
    return {
        "geometric_mean_ratio": geometric_mean_ratio,
        "log10_geometric_mean_ratio": log10_geometric_mean_ratio,
    }


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
