from pathlib import Path

from violations_glm import paths


def test_repository_paths_are_absolute() -> None:
    assert paths.REPO_ROOT.is_absolute()
    assert paths.figures_dir() == paths.REPO_ROOT / "figures"
    assert paths.model_summaries_dir() == paths.REPO_ROOT / "results/model_summaries"


def test_data_directory_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VIOLATIONS_GLM_DATA_DIR", str(tmp_path))
    assert paths.data_dir() == tmp_path.resolve()
