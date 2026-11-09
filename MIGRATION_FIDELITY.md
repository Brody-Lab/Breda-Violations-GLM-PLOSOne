# Figure migration fidelity contract

Use this document for every port from `violations-multinomial/notebooks/publication/`
into this repository. It complements [MIGRATION.md](MIGRATION.md) (checklist status) and
the Agent skill [`.cursor/skills/migrate-publication-figure/SKILL.md`](.cursor/skills/migrate-publication-figure/SKILL.md).

## When to use

- Destination notebook exists but plots or printed stats do not match the submitted
  publication notebook (skeleton migration).
- A new panel or supplement is added and must match historical styling.

**Default mode: fidelity-only.** Apply [REVISION.md](REVISION.md) changes only when that
file (or a linked `PLOSPublication/REVISION*` note) explicitly names the figure.

## Acceptance criteria

| Area | Requirement |
|------|-------------|
| Plots | Same geometry, aggregation, colors, axis limits, labels, legend placement, and `figsize` as the source notebook unless a revision doc directs a change. |
| Statistics | Every `print`, displayed return value, and narrative blockquote from the source is preserved (section order may follow the source). |
| Analysis | No model fitting or heavy aggregation in notebooks. Load via `violations_glm.io` and committed `scripts/*.py` outputs. |
| Notebook structure | Overview header, imports (+ optional settings), then panel cells. Extra cells are fine during **Phase A**; trim in **Phase B**. |
| Outputs | Each panel PNG uses `save_figure` at 300 DPI; stems match the notebook overview and [scripts/verify_cached.py](scripts/verify_cached.py). |
| Commit | Code cells have no saved outputs (`tests/test_notebooks.py`). |

## Two phases

### Phase A — Parity (do first)

1. Generate or read a migration brief: `uv run python scripts/migration_brief.py <source.ipynb>`.
2. Map each source plot and stat block to destination cells.
3. Wire loaders/scripts; port plot code and stat cells from the source.
4. Run `uv run python scripts/verify_cached.py --notebook <dest.ipynb>`.
5. Compare PNGs to `violations-multinomial/notebooks/publication/figures/` when available.

### Phase B — Trim (optional)

1. Collapse to one markdown + one code cell per panel ([`.cursor/rules/notebooks.mdc`](.cursor/rules/notebooks.mdc)).
2. Move reused logic into `src/violations_glm/`; delete scratch cells.
3. Re-run verify and clear outputs.

## Source → destination map

Historical repo root: `violations-multinomial/notebooks/publication/`.

| Deliverable | Destination | Primary source notebook(s) | Scripts / loaders |
|-------------|-------------|----------------------------|-------------------|
| Figure 1 | [notebooks/fig01.ipynb](notebooks/fig01.ipynb) | `@fig1_trained_data_stats.ipynb` | `prepare_data.py`, `io.load_trials_df`, revision behavior loaders |
| Figure 2 | [notebooks/fig02.ipynb](notebooks/fig02.ipynb) | (schematic; no `@` composite notebook) | — |
| Figure 3 | [notebooks/fig03.ipynb](notebooks/fig03.ipynb) | `@fig3_prev_viol.ipynb` | `compute_fig03_autocorrelation.py`, `fit_fig03.py`, `load_autocorrelation_summary`, `load_model_summary` |
| Figure 4 | [notebooks/fig04.ipynb](notebooks/fig04.ipynb) | `@fig4_viol_stim_intrxs_multi.ipynb` | `fit_fig04.py`, `load_model_summary`, `violations_glm.psychometrics` |
| Figure 5 | [notebooks/fig05.ipynb](notebooks/fig05.ipynb) | `@fig5_model_comparisons.ipynb` | `fit_fig05.py`, `summaries.bits_per_trial`, likelihood-ratio helpers |
| Figure 6 (Revision 1) | [notebooks/fig06.ipynb](notebooks/fig06.ipynb) | — (new) | `export_fig05_animal_deltas.py`, `prepare_fig06_across_animals.py`, `prepare_fig06_within_animal.py`, `load_fig06_*` |
| S1 | [notebooks/suppS1.ipynb](notebooks/suppS1.ipynb) | `figS1_new_dataset_alignment.ipynb` | `prepare_alignment_summary.py`, `load_alignment_deltas`, `load_alignment_example_sessions`, `load_alignment_dataset_*` |
| S2 | [notebooks/suppS2.ipynb](notebooks/suppS2.ipynb) | `figS2_stimulus_pairs_by_animal.ipynb` | `prepare_stimulus_pair_summary.py`, `load_stimulus_pair_summary` |
| S3 | [notebooks/suppS3.ipynb](notebooks/suppS3.ipynb) | `figS3_create_trained_threshold_df.ipynb` | `prepare_full_pipeline.py`, `prepare_data.py` (notebook mostly visualizes pipeline output) |
| S4 | [notebooks/suppS4.ipynb](notebooks/suppS4.ipynb) | Schematic (no dedicated `@figS4`; compare Fig 3 intuition / filter demos if needed) | `violations_glm.features.exponential_kernel` |
| S5 | [notebooks/suppS5.ipynb](notebooks/suppS5.ipynb) | Split from original S5 draft; training-history panels | `prepare_revision_behavior.py`, revision loaders |
| S6 | [notebooks/suppS6.ipynb](notebooks/suppS6.ipynb) | Within-session structure (autocorr robustness → Fig 3) | `compute_fig03_autocorrelation.py --edge-trials 10`, revision loaders |

Fit-only source notebooks (not the main plot composite) remain inputs to `scripts/fit_fig*.py`
and golden summaries under `results/model_summaries/legacy/`.

## Agent prompt template (one figure)

In **Agent mode**, attach the source and destination notebooks, then paste (replace placeholders):

```text
Use the migrate-publication-figure skill and MIGRATION_FIDELITY.md.
Fidelity-only migration for FIG_ID: match submitted plots and all printed stats
from SOURCE_NOTEBOOK to DEST_NOTEBOOK. Load data and fits via scripts and
violations_glm loaders only—no fitting in the notebook. Phase A: port everything
needed for parity; update verify_cached.py stems to match save_figure calls.
Do not change REVISION.md unless this figure has an open revision item.
```

Example for Figure 4:

```text
Use the migrate-publication-figure skill and MIGRATION_FIDELITY.md.
Fidelity-only migration for fig04: match submitted plots and all printed stats
from violations-multinomial/notebooks/publication/@fig4_viol_stim_intrxs_multi.ipynb
to notebooks/fig04.ipynb. Load data and fits via scripts/fit_fig04.py and
violations_glm loaders only. Phase A parity; update verify_cached.py.
```

Run **one figure per session** when possible.

## Verification

```bash
# Inventory source notebook (plots, prints, saves)
uv run python scripts/migration_brief.py ../violations-multinomial/notebooks/publication/@fig4_viol_stim_intrxs_multi.ipynb

# Execute destination notebook and check PNGs
uv run python scripts/verify_cached.py --notebook fig04.ipynb

# Fast notebook hygiene
uv run pytest tests/test_notebooks.py -q
```

Compare printed statistics to source cell outputs or legacy summaries in
`results/model_summaries/legacy/` for GLM figures.

## Fidelity status (manual)

Update [MIGRATION.md](MIGRATION.md) when a figure completes Phase A review:

- **Fidelity complete** — plots and stats match source (or documented revision).
- **Skeleton** — notebook runs and saves panels but has not been checked against source.
