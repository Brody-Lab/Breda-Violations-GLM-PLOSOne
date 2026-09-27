"""Resolve repository data, result, and figure paths."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Return the data directory, allowing an external Zenodo data checkout."""
    override = os.environ.get("VIOLATIONS_GLM_DATA_DIR")
    return Path(override).expanduser().resolve() if override else REPO_ROOT / "data"


def raw_data_dir() -> Path:
    return data_dir() / "raw"


def processed_data_dir() -> Path:
    return data_dir() / "processed"


def results_dir() -> Path:
    return REPO_ROOT / "results"


def model_summaries_dir() -> Path:
    return results_dir() / "model_summaries"


def figures_dir() -> Path:
    return REPO_ROOT / "figures"
