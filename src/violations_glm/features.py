"""Publication GLM feature construction, preserving submitted behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd


def standardize(values: pd.Series) -> pd.Series:
    return (values - values.mean()) / values.std()


def shift_within_sessions(
    values: pd.Series, sessions: pd.Series, shift_size: int = 1
) -> pd.Series:
    shifted = values.shift(shift_size).fillna(0)
    first = sessions.diff().ne(0)
    first_n = first.copy()
    for offset in range(1, shift_size):
        first_n |= first.shift(offset).fillna(False)
    return shifted * ~first_n


def previous_violation(frame: pd.DataFrame) -> pd.Series:
    encoded = frame["choice"].replace({1: 0, 2: 1})
    return shift_within_sessions(encoded, frame["session"]).astype(int)


def previous_choice(frame: pd.DataFrame) -> pd.Series:
    encoded = frame["choice"].replace({0: -1, 2: 0})
    return shift_within_sessions(encoded, frame["session"]).astype(int)


def previous_correct_side(frame: pd.DataFrame) -> pd.Series:
    encoded = frame["correct_side"].replace({0: -1})
    previous = shift_within_sessions(encoded, frame["session"])
    previous_event = shift_within_sessions(frame["violation"], frame["session"]).astype(
        bool
    )
    return (previous * ~previous_event).astype(int)


def previous_average_stimulus(
    frame: pd.DataFrame, *, mask_previous_violation: bool
) -> pd.Series:
    average = (frame["s_a"] + frame["s_b"]) / 2
    previous = shift_within_sessions(standardize(average), frame["session"])
    if mask_previous_violation:
        previous_event = shift_within_sessions(
            frame["violation"], frame["session"]
        ).astype(bool)
        previous = previous * ~previous_event
    return previous


def exponential_filter(
    values: pd.Series,
    sessions: pd.Series,
    tau: float,
    *,
    length_factor: int = 5,
) -> pd.Series:
    """Apply the submitted session-wise convolution and global max scaling."""
    kernel = exponential_kernel(tau, length_factor=length_factor)
    output = pd.Series(index=values.index, dtype=float)
    for session_id, session_values in values.groupby(sessions, sort=True):
        filtered = np.convolve(session_values, kernel, mode="full")[
            : len(session_values)
        ]
        output.loc[sessions == session_id] = filtered
    return output / output.max()


def exponential_kernel(tau: float, *, length_factor: int = 5) -> np.ndarray:
    """Create the finite exponential kernel used by submitted model features."""
    return np.array(
        [np.exp(-index / tau) for index in range(length_factor * round(tau))]
    )


def filtered_previous_violation(frame: pd.DataFrame, tau: float) -> pd.Series:
    return exponential_filter(previous_violation(frame), frame["session"], tau)


def build_design_matrix(
    frame: pd.DataFrame,
    operations: list[tuple[str, str]],
    *,
    tau: float | None = None,
) -> pd.DataFrame:
    """Build columns in declared order using publication feature operations."""
    output = pd.DataFrame(index=frame.index)
    filtered: pd.Series | None = None
    for name, operation in operations:
        if operation == "session":
            values = frame["session"].copy()
        elif operation == "bias":
            values = pd.Series(1, index=frame.index, dtype=int)
        elif operation in {"s_a", "s_b"}:
            values = standardize(frame[operation])
        elif operation == "prev_violation":
            values = previous_violation(frame)
        elif operation == "filt_prev_violation":
            if tau is None:
                raise ValueError("tau is required for filtered features")
            filtered = filtered_previous_violation(frame, tau)
            values = filtered
        elif operation == "prev_stim_unmasked":
            values = previous_average_stimulus(frame, mask_previous_violation=False)
        elif operation == "prev_stim_masked":
            values = previous_average_stimulus(frame, mask_previous_violation=True)
        elif operation == "prev_correct":
            values = previous_correct_side(frame)
        elif operation == "prev_choice":
            values = previous_choice(frame)
        elif operation == "sa_binary_interaction":
            values = standardize(frame["s_a"]) * previous_violation(frame)
        elif operation == "sb_binary_interaction":
            values = standardize(frame["s_b"]) * previous_violation(frame)
        elif operation in {"sa_filtered_interaction", "sb_filtered_interaction"}:
            if tau is None:
                raise ValueError("tau is required for filtered interactions")
            if filtered is None:
                filtered = filtered_previous_violation(frame, tau)
            stimulus = "s_a" if operation.startswith("sa_") else "s_b"
            values = standardize(frame[stimulus]) * filtered
        else:
            raise ValueError(f"Unknown feature operation: {operation}")
        output[name] = values
    return output


def build_labels(
    frame: pd.DataFrame, design: pd.DataFrame, *, model_type: str
) -> tuple[pd.DataFrame, np.ndarray]:
    if model_type == "multi":
        labels = pd.get_dummies(frame["choice"].replace({0: 0, 1: 1, 2: 2})).to_numpy(
            dtype=int, copy=True
        )
        return design, labels
    if model_type == "binary":
        labels = frame["choice"].replace({0: 0, 1: 1, 2: np.nan})
        valid = labels.dropna().index
        return design.loc[valid], labels.loc[valid].to_numpy(dtype=int, copy=True)
    raise ValueError(f"Unknown model type: {model_type}")
