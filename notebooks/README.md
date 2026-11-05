# Figure notebooks

Each notebook reproduces one manuscript figure or supporting figure using the
package loaders and stable result artifacts. Notebooks contain plotting and
panel-specific narrative only; data preparation and model fitting live in
`src/` and `scripts/`.

## Migrating a figure from the historical repo

Full process: [MIGRATION_FIDELITY.md](../MIGRATION_FIDELITY.md).

1. **Brief the source notebook** (plots, saves, prints):

   ```bash
   uv run python scripts/migration_brief.py \
     ../violations-multinomial/notebooks/publication/@fig4_viol_stim_intrxs_multi.ipynb \
     -o docs/migration_briefs/fig04.md
   ```

2. **Agent mode** — attach source + destination notebooks, then paste the prompt
   template from MIGRATION_FIDELITY.md (one figure per session).

3. **Verify**:

   ```bash
   uv run python scripts/verify_cached.py --notebook fig04.ipynb
   uv run pytest tests/test_notebooks.py -q
   ```

4. **Update** [MIGRATION.md](../MIGRATION.md) fidelity row when Phase A review is done.

Cursor applies [`.cursor/rules/migrate-figure-fidelity.mdc`](../.cursor/rules/migrate-figure-fidelity.mdc)
when you edit notebooks here; invoke the **migrate-publication-figure** skill for full workflow steps.
