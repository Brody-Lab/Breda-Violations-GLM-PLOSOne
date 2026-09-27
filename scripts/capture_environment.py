"""Capture numerical-library and platform provenance for fit validation."""

from __future__ import annotations

import contextlib
import io
import json
import platform
import sys

import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn

from violations_glm.paths import results_dir


def main() -> None:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        np.show_config()
    provenance = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
        "numpy_configuration": buffer.getvalue(),
    }
    destination = results_dir() / "provenance"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "environment.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
