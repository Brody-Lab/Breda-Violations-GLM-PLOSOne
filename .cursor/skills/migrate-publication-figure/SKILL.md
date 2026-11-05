---
name: migrate-publication-figure
description: >-
  Migrates one figure or supplement from the historical violations-multinomial
  repository into Breda-Violations-GLM-PLOSOne. Use when porting, revising,
  validating, or documenting a manuscript figure, or when panels look wrong
  despite having the right letter structure.
disable-model-invocation: true
---

# Migrate a publication figure

## Priorities

1. **Fidelity** — Recreate the exact publication analysis and plots unless the
   user directs a change. Assume the original was correct; do not redesign
   aggregations, plot types, colors, axis limits, or stats “for clarity.”
2. **Readable reproduction** — A newcomer should clone the repo, run the
   notebook, and get the published panels. Prefer straightforward panel code
   over wrappers and clever helpers used once.
3. **Package boundary** — Aggregation/transforms and model fits live in
   `src/violations_glm/` (+ thin prepare scripts). Notebooks load, plot, and
   print.

## Workflow

1. **Map the source** — Read the submitted figure/caption, revision notes, and
   the source plot notebook. List every markdown section and code cell: what
   it plots, which stats it prints, axis limits, colors, and output filenames.
2. **Map the destination** — Read the current `figNN.ipynb` / `suppSN.ipynb`,
   overview outputs, and `scripts/verify_cached.py` stems. Note missing panels,
   wrong plot types, commented saves, and scratch cells.
3. **Decide stay / move / split** — Keep panels that belong on the main figure;
   move blocks the user relocates (e.g. regressions to a supplement) and
   renumber subsequent panels + stems together.
4. **Extract reusable logic**
   - Transforms → `transforms.py` (session aggregates, filters).
   - Fits / summaries → `stats.py` + `scripts/prepare_*.py` writing parquet/CSV/JSON.
   - Loaders → named functions in `io.py` (no absolute paths, no bare
     `pd.read_*` in notebooks).
5. **Build the notebook** following `.cursor/rules/notebooks.mdc`:
   - One markdown + one code cell per panel.
   - Shared `sns.set_context` / RC once in imports or settings.
   - Call `configure_publication_dpi()` in imports so display and saves are
     300 DPI (`save_figure` defaults to `FIGURE_DPI = 300`).
   - Active `save_figure` stems matching the overview and `verify_cached.py`.
   - No model fitting in the notebook; print loaded summaries.
6. **Verify** — Execute top to bottom (fresh kernel after package edits).
   Confirm panel PNGs match the original visuals and printed stats match the
   source. Update `verify_cached.py`, `REVISION.md`, and `MIGRATION.md`.
7. **Clear outputs** before finishing.

## Common failure modes

- Destination has correct panel letters but **different plot geometry**
  (animal bars vs session histograms, trial-weighted vs session-weighted
  choice %). Treat as wrong until proven intentional.
- Fitting MixedLM/OLS/sklearn inside the notebook instead of loading artifacts.
- Repeating `with sns.plotting_context(...)` in every panel cell.
- Leaving scratch cells (`obj?`) or commented `save_figure` calls.
- Updating stems in the notebook but not in `verify_cached.py`.
- Stale kernel after adding `config` / loader exports → mysterious ImportError.

## Source of truth

- Historical plots: `violations-multinomial/notebooks/publication/`
- Publication repo: this project; notebooks under `notebooks/`
- Do not invent new analyses for the manuscript figure unless asked.
- Mapping table, acceptance criteria, and Agent prompt template:
  [MIGRATION_FIDELITY.md](../../../MIGRATION_FIDELITY.md)
- Optional source inventory: `uv run python scripts/migration_brief.py <source.ipynb>`
