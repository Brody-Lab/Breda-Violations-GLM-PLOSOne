import numpy as np
import pandas as pd

from violations_glm.features import (
    exponential_filter,
    previous_violation,
    shift_within_sessions,
)


def test_shift_masks_session_boundaries() -> None:
    values = pd.Series([1, 2, 3, 4])
    sessions = pd.Series([10, 10, 11, 11])
    assert shift_within_sessions(values, sessions).tolist() == [0, 1, 0, 3]


def test_previous_violation_uses_prior_trial_only() -> None:
    frame = pd.DataFrame({"choice": [2, 0, 2, 1], "session": [1, 1, 2, 2]})
    assert previous_violation(frame).tolist() == [0, 1, 0, 1]


def test_exponential_filter_resets_each_session_and_scales_globally() -> None:
    values = pd.Series([1, 0, 1, 0])
    sessions = pd.Series([1, 1, 2, 2])
    observed = exponential_filter(values, sessions, tau=1)
    np.testing.assert_allclose(observed, [1, np.exp(-1), 1, np.exp(-1)])
