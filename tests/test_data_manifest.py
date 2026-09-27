import json

from violations_glm.paths import processed_data_dir


def test_submitted_modeling_dataset_manifest() -> None:
    manifest = json.loads(
        (processed_data_dir() / "all_animals_trained_cleaned.manifest.json").read_text()
    )
    assert manifest["output"] == {
        "animals": 16,
        "logical_sha256": (
            "151f0dc0b958dd104343d1b56ac54ebc3b083bebe9acd710e4ce124d6e60cf9a"
        ),
        "path": "all_animals_trained_cleaned.parquet",
        "rows": 1_866_003,
        "sessions": 5_210,
        "sha256": ("947750d40daca77b77f84371d415a1391e3f9ba4910e118abf12b0bfca136c54"),
    }
    assert (
        manifest["output"]["logical_sha256"] == manifest["reference"]["logical_sha256"]
    )


def test_all_training_dataset_manifest_includes_reconstruction_cohort() -> None:
    manifest = json.loads(
        (processed_data_dir() / "all_animals_cleaned.manifest.json").read_text()
    )
    assert manifest["output"]["animals"] == 17
    assert manifest["output"]["rows"] == 3_034_200
    assert manifest["output"]["sessions"] == 9_952
