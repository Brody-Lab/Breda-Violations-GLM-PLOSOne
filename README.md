# Breda et al. — Violations GLM analyses

Reproducible data preparation, model fitting, statistics, and figure code for
the PLOS ONE manuscript by Breda et al.

This repository is the publication source of truth. The historical
`violations-multinomial` repository is retained only as migration provenance;
Dynamax, GLM-HMM, and unrelated exploratory analyses are intentionally excluded.

## Setup

The project uses Python 3.10 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev --extra legacy
uv run python -m ipykernel install --user \
  --name violations-glm --display-name "Python (violations-glm)"
```

Use `--extra legacy` only when exporting the trusted historical `dill` model
objects. Normal figure reproduction does not deserialize pickle files.

## Reproduction paths

Two workflows are supported:

1. **Cached reproduction** loads curated processed data and stable model-summary
   files, then executes the figure notebooks.
2. **Full reproduction** prepares the analysis dataset, refits every model used
   in Figures 3–5, validates the results against the submitted fits, and then
   executes the notebooks.

Cached mode selects the checksum-verified `legacy` summary collection. Full
mode writes a separate `current` collection, validates it against the legacy
baselines, and makes the notebooks plot those new fits.

Cached reproduction:

```bash
uv run python scripts/reproduce_all.py --mode cached
```

Full data preparation, Figure 3–5 refitting, legacy regression checks,
10,000-permutation autocorrelation analysis, and notebook execution:

```bash
uv run python scripts/reproduce_all.py --mode full \
  --all-training-data /path/to/all_animals_cleaned.csv
```

The full path performs every submitted sigma/tau fit and is expected to take
several hours on a laptop. It is intentionally separate from the fast cached
path.

Run a one-animal parity check before a full fit:

```bash
uv run violations-glm fit \
  2025_07_19_fig3_fit_prev_violation_rs47 \
  --animal W051 --validate-legacy --collection smoke
```

Run the fast verification suite independently:

```bash
uv run pytest -q
uv run ruff check src scripts tests notebooks
uv run python scripts/verify_cached.py
```

## Repository layout

```text
configs/                 Committed model-fitting parameters
data/
  raw/                   Zenodo inputs; not tracked by Git
  processed/             Publication-only processed datasets
figures/                 Generated panels and manuscript composites
notebooks/               One plot-focused notebook per figure/supplement
results/model_summaries/ Stable tabular outputs consumed by notebooks
scripts/                 Data preparation, fitting, and orchestration
src/violations_glm/      Reusable analysis package
tests/                   Fast unit and regression tests
```

The data directory can be stored elsewhere by setting
`VIOLATIONS_GLM_DATA_DIR`.

The reconstructed all-training dataset contains 17 animals, including W082.
The submitted GLM cohort contains 16 animals and excludes W082 because it did
not enter the trained-session modeling dataset. Cohort-specific constants and
loader validation prevent these two scopes from being mixed.

## Figures

The manuscript contains Figures 1–5 and Supporting Figures S1–S4. Revision 1
adds Supporting Figure S5, cited with Figure 1 in the Results. S1–S4 retain
their submitted numbering because they are already cited earlier in the
manuscript. The code uses S5 as the working number; confirm it against the
final manuscript ordering before submission.

Migration and revision status are tracked in `MIGRATION.md` and `REVISION.md`.

## Data and model artifacts

Compact processed datasets and stable model summaries are suitable for Git.
Raw/large data and trusted historical fit objects will be deposited on Zenodo
with checksums. The final DOI will be added before release.

## License

Code is available under the MIT License. Data reuse terms will follow the
associated Zenodo record.