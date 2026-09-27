"""Export trusted legacy fit objects to stable regression-test artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import dill
import numpy as np
import pandas as pd
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.ndarray, pd.Index, pd.Series)):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value):
        module = getattr(value, "__module__", "unknown")
        qualname = getattr(value, "__qualname__", type(value).__qualname__)
        return f"{module}.{qualname}"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def stable_fit_table(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in ("features", "weights"):
        if column in output:
            output[column] = output[column].map(
                lambda value: json.dumps(jsonable(value), separators=(",", ":"))
            )
    return output


def export_experiment(source: Path, destination: Path, provenance: dict) -> None:
    with gzip.open(source, "rb") as stream:
        experiment = dill.load(stream)

    stem = source.name.removesuffix(".pkl.gz")
    artifact_dir = destination / stem
    artifact_dir.mkdir(parents=True, exist_ok=True)

    fit_models = getattr(experiment, "fit_models", None)
    if not isinstance(fit_models, pd.DataFrame):
        raise TypeError(f"{source.name} has no fit_models DataFrame")
    stable_fit_table(fit_models).to_parquet(
        artifact_dir / "fit_models.parquet", index=False
    )

    null_models = getattr(experiment, "null_models", None)
    if isinstance(null_models, pd.DataFrame):
        null_models.to_parquet(artifact_dir / "null_models.parquet", index=False)
    elif isinstance(null_models, list) and null_models:
        pd.concat(null_models, ignore_index=True).to_parquet(
            artifact_dir / "null_models.parquet", index=False
        )

    metadata = {
        **provenance,
        "experiment_class": (
            f"{type(experiment).__module__}.{type(experiment).__qualname__}"
        ),
        "params": jsonable(getattr(experiment, "params", {})),
        "random_state": jsonable(getattr(experiment, "random_state", None)),
        "test_size": jsonable(getattr(experiment, "test_size", None)),
        "fit_rows": len(fit_models),
        "fit_columns": fit_models.columns.tolist(),
    }
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-repo",
        type=Path,
        required=True,
        help="Path to the trusted violations-multinomial checkout.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/legacy_fit_manifest.yml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/model_summaries/legacy"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = yaml.safe_load(args.manifest.read_text())
    source_root = args.legacy_repo / manifest["source"]["directory"]

    legacy_src = args.legacy_repo / "src"
    sys.path.insert(0, str(legacy_src))
    __import__("violmulti")

    args.output.mkdir(parents=True, exist_ok=True)
    for artifact in manifest["artifacts"]:
        source = source_root / artifact["filename"]
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != artifact["bytes"]:
            raise ValueError(f"Unexpected size for {source}")
        observed_hash = sha256(source)
        if observed_hash != artifact["sha256"]:
            raise ValueError(f"Checksum mismatch for {source}")
        export_experiment(
            source,
            args.output,
            {
                **artifact,
                "source_commit": manifest["source"]["commit"],
            },
        )
        print(f"exported {artifact['filename']}")


if __name__ == "__main__":
    main()
