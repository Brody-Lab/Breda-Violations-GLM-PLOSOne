"""Deterministic, configuration-driven publication model fitting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from violations_glm.config import RANDOM_STATE, TEST_SIZE
from violations_glm.features import build_design_matrix, build_labels
from violations_glm.models import (
    BinaryLogisticRegression,
    MultinomialLogisticRegression,
)

ModelType = Literal["binary", "multi"]

STANDARD_SIGMAS = (0.07, 0.13, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
STANDARD_TAUS = (
    0.51,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    12.0,
    15.0,
    20.0,
    25.0,
    30.0,
    40.0,
    50.0,
    100.0,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_type: ModelType
    operations: tuple[tuple[str, str], ...]
    lr_only_eval: bool = False


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    models: tuple[ModelSpec, ...]
    sigmas: tuple[float, ...] = STANDARD_SIGMAS
    taus: tuple[float, ...] = ()
    null_mode: ModelType | None = None


def split_sessions(
    frame: pd.DataFrame,
    *,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    return train_test_split(
        frame["session"].unique(),
        test_size=test_size,
        random_state=random_state,
    )


def apply_split(
    design: pd.DataFrame,
    labels: np.ndarray,
    train_sessions: np.ndarray,
    test_sessions: np.ndarray,
    *,
    lr_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    train_mask = design["session"].isin(train_sessions).to_numpy()
    test_mask = design["session"].isin(test_sessions).to_numpy()
    train_design = design.loc[train_mask].drop(columns="session")
    test_design = design.loc[test_mask].drop(columns="session")
    train_labels = labels[train_mask]
    test_labels = labels[test_mask]
    if lr_only:
        violations = np.all(test_labels == np.array([0, 0, 1]), axis=1)
        test_design = test_design.iloc[np.flatnonzero(~violations)]
        test_labels = test_labels[~violations]
    return train_design, test_design, train_labels, test_labels


def null_model(
    frame: pd.DataFrame,
    test_sessions: np.ndarray,
    *,
    mode: ModelType,
) -> dict:
    labels = frame["choice"]
    if mode == "binary":
        labels = labels.loc[labels != 2]
    test_labels = labels.loc[frame.loc[labels.index, "session"].isin(test_sessions)]
    counts = test_labels.value_counts().to_dict()
    total = len(test_labels)
    probabilities = {key: count / total for key, count in counts.items()}
    nll = -sum(counts[key] * np.log(probabilities[key]) for key in counts)
    return {
        "model_name": "null",
        "model_type": mode,
        "nll": nll,
        "n_test_trials": total,
        "n_train_trials": len(frame) - total,
        **{
            f"p_{label}": probabilities.get(index, 0)
            for index, label in enumerate(("L", "R", "V"))
        },
        **{
            f"n_{label}": counts.get(index, 0)
            for index, label in enumerate(("L", "R", "V"))
        },
    }


def run_experiment(
    frame: pd.DataFrame,
    spec: ExperimentSpec,
    *,
    taus_by_animal: dict[str, float] | None = None,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    fit_rows: list[dict] = []
    null_rows: list[dict] = []
    binary_rng = np.random.default_rng(random_state)

    for animal_id, animal_frame in frame.groupby("animal_id", sort=False):
        train_sessions, test_sessions = split_sessions(
            animal_frame, random_state=random_state
        )
        if spec.null_mode:
            null_rows.append(
                {
                    "animal_id": animal_id,
                    **null_model(
                        animal_frame,
                        test_sessions,
                        mode=spec.null_mode,
                    ),
                }
            )

        tau_values: tuple[float | None, ...]
        if spec.taus:
            tau_values = spec.taus
        elif any(
            "filtered" in operation or operation == "filt_prev_violation"
            for model in spec.models
            for _, operation in model.operations
        ):
            if taus_by_animal is None or animal_id not in taus_by_animal:
                raise ValueError(f"No submitted tau for {animal_id}")
            tau_values = (taus_by_animal[animal_id],)
        else:
            tau_values = (None,)

        for tau in tau_values:
            for model_spec in spec.models:
                design = build_design_matrix(
                    animal_frame,
                    list(model_spec.operations),
                    tau=tau,
                )
                design, labels = build_labels(
                    animal_frame,
                    design,
                    model_type=model_spec.model_type,
                )
                split = apply_split(
                    design,
                    labels,
                    train_sessions,
                    test_sessions,
                    lr_only=model_spec.lr_only_eval,
                )
                train_x, test_x, train_y, test_y = split
                for sigma in spec.sigmas:
                    if model_spec.model_type == "multi":
                        model = MultinomialLogisticRegression(sigma)
                    else:
                        model = BinaryLogisticRegression(sigma, rng=binary_rng)
                    weights = model.fit(train_x, train_y)
                    if model_spec.model_type == "multi":
                        test_nll = model.evaluate(
                            test_x,
                            test_y,
                            lr_only=model_spec.lr_only_eval,
                        )
                        train_nll = model.evaluate(train_x, train_y)
                    else:
                        test_nll = model.evaluate(test_x, test_y)
                        train_nll = model.evaluate(train_x, train_y)
                    row = {
                        "animal_id": animal_id,
                        "model_name": model_spec.name,
                        "model_type": model_spec.model_type,
                        "nll": test_nll,
                        "train_nll": train_nll,
                        "sigma": sigma,
                        "features": json.dumps(
                            test_x.columns.tolist(), separators=(",", ":")
                        ),
                        "weights": json.dumps(weights.tolist(), separators=(",", ":")),
                        "n_train_trials": len(train_x),
                        "n_test_trials": len(test_x),
                    }
                    if spec.taus:
                        row["tau"] = tau
                    fit_rows.append(row)

    fits = pd.DataFrame(fit_rows)
    nulls = pd.DataFrame(null_rows) if null_rows else None
    return fits, nulls


def best_fits(
    fits: pd.DataFrame, *, groups: tuple[str, ...] = ("animal_id",)
) -> pd.DataFrame:
    indices = fits.groupby(list(groups), observed=False)["nll"].idxmin()
    return fits.loc[indices].reset_index(drop=True)
