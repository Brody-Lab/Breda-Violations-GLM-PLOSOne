import nbformat

from violations_glm.paths import REPO_ROOT


def test_publication_notebooks_are_committed_without_outputs() -> None:
    notebooks = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
    assert notebooks
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is None, path
                assert not cell.outputs, path


def test_figure_notebook_names_are_complete() -> None:
    observed = {path.name for path in (REPO_ROOT / "notebooks").glob("*.ipynb")}
    expected = {
        *(f"fig{number:02d}.ipynb" for number in range(1, 7)),
        *(f"suppS{number}.ipynb" for number in range(1, 7)),
    }
    assert observed == expected
