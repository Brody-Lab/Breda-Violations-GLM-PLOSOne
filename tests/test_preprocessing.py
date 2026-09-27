import pandas as pd

from violations_glm.config import ANIMAL_IDS
from violations_glm.preprocessing import (
    add_animal_session_id,
    filter_modeling_sessions,
)


def _session(
    animal_id: str,
    session: int,
    *,
    trials: int,
    violations: int = 0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "animal_id": animal_id,
            "session": session,
            "trial": range(1, trials + 1),
            "hit": [1.0] * trials,
            "violation": [1] * violations + [0] * (trials - violations),
        }
    )


def test_animal_session_id_matches_submitted_format() -> None:
    frame = _session(ANIMAL_IDS[0], 12, trials=2)
    observed = add_animal_session_id(frame)
    assert observed["animal_id_session"].unique().tolist() == ["W051_s12"]


def test_session_quality_filter_preserves_boundary_values() -> None:
    frame = pd.concat(
        [
            _session(ANIMAL_IDS[0], 1, trials=50),
            _session(ANIMAL_IDS[0], 2, trials=49),
            _session(ANIMAL_IDS[0], 3, trials=100, violations=99),
            _session(ANIMAL_IDS[0], 4, trials=100, violations=100),
        ],
        ignore_index=True,
    )
    observed = filter_modeling_sessions(frame)
    assert sorted(observed["session"].unique()) == [1, 3]
