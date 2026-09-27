from violations_glm.config import (
    ALL_ANIMAL_IDS,
    MODEL_ANIMAL_IDS,
    PSYCHOMETRIC_ANIMALS,
)
from violations_glm.publication_specs import PUBLICATION_SPECS


def test_eight_submitted_fit_specs_are_defined() -> None:
    assert len(PUBLICATION_SPECS) == 8
    assert {name.split("_fig", 1)[1][:1] for name in PUBLICATION_SPECS} == {
        "3",
        "4",
        "5",
    }


def test_all_specs_preserve_publication_sigma_grid() -> None:
    for spec in PUBLICATION_SPECS.values():
        assert spec.sigmas == (0.07, 0.13, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def test_psychometric_subset_contains_only_publication_animals() -> None:
    assert set(PSYCHOMETRIC_ANIMALS) < set(MODEL_ANIMAL_IDS)


def test_reconstruction_and_modeling_cohorts_are_explicit() -> None:
    assert set(ALL_ANIMAL_IDS) - set(MODEL_ANIMAL_IDS) == {"W082"}
