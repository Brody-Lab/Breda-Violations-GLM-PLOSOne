# Revision 1 code checklist

Source requests and author notes remain in
`PLOSPublication/REVISION1_Breda2026PLOSOne/`. This file links code outputs to
the manuscript changes they support.

## Figure 1 and Supporting Figure S5

### Code / notebook (done)

- [x] Finalize revised Figure 1 panel placement: example sequences are Panel B;
      session histograms and choice distribution are Panels C–F (shifted one
      letter from the submitted B–E layout; choice is E, violation rate is F).
- [x] Restore submitted session histograms and choice distribution as
      Figure 1 panels C–F (identical visuals to the original publication
      notebook panels B–E).
- [x] Move hit-versus-violation scatter and MixedLM into Figure 1 Panel G
      (session points + animal means; across-animal OLS not reported in text).
- [x] Plot representative raw behavioral sequences as Figure 1 Panel B
      (`fig01b_example_sequences`; DPI 300 via `configure_publication_dpi`).
- [x] Plot violation rate by trial position within sessions (S5 panels d–f).
- [x] Add S5 to the Results with Figure 1; retain S1–S4 numbering.

### Manuscript text (done)

- [x] Update the Figure 1 panel legend / in-text figure description for the
      revised B–F layout.
- [x] Update the Results section for the revised Figure 1 (and S5 linkage).

### Design / Overleaf (remaining)

- [ ] Add task-event timing to Figure 1a in Affinity Designer (not coded;
      composite `figures/fig01_FULL.png`). Panel A is the only outstanding
      visual before the figure is final.
- [ ] Address the Revision 1 comments that requested the Figure 1 changes
      (task timing, raw sequences, panel reordering / S5 move) in the
      response-to-reviewers text.
- [ ] Once Panel A is complete, export the full Figure 1 composite and upload
      it to the Overleaf manuscript project.

### Supplement scope (resolved)

- [x] Split the original seven-panel S5 into two supplements so each one is
      cited from a single main figure: **S5** (training history, cited with
      Figure 1) and **S6** (within-session structure, cited with Figure 3).
- [x] S5 panels a–c: training-history univariate scatters (fixation-stage
      violation rate r = 0.48, p = 0.06; sessions in stage 1 and sessions to
      trained n.s.). Addresses R1 #20.
- [x] S6 panels a–d: session progress (10 bins, all trials), start/end edge
      alignment, and the full vs. edge-trimmed autocorrelation overlay.
      Addresses R3 #25.
- [x] Delay-length and relative-delay analyses dropped from the paper (letter
      only for R3 #24). Hit-vs-violation moved into Figure 1 (now panel f
      after the hit/violation histograms merged into panel d).
- [x] Finalize `suppS5.ipynb` and `suppS6.ipynb` with `save_figure` stems and
      `verify_cached.py` expectations.
- [ ] Cached artifacts in `results/revision_behavior/` keep their original
      `suppS5_*` prefixes even where they now feed S6 panels; the loaders in
      `io.py` are named by content, so this is cosmetic only.

## Figure 3

- [x] Recompute the autocorrelation null distribution with 10,000 permutations
      (submitted notebook used 100 shuffles).
- [x] Use random seed 47, one reusable null array, and empirical p-values
      `(count + 1) / (n + 1)`.
- [x] Apply Bonferroni correction over lags 1–20 only (41-lag correction removed).
- [x] Restore Figure 3 notebook panels to submitted plot styling; Panel A loads
      `scripts/compute_fig03_autocorrelation.py` output (circle/X significance
      markers at ±20 lags). GLM panels load `scripts/fit_fig03.py` summaries.
- [ ] Update panel 3a caption, Methods, and Results from the same output.
      After Bonferroni correction over lags 1–20, 18 positive lags remain
      significant (lags 1–17 and 19; lags 18 and 20 do not).
- [x] Session-edge robustness for Figure 3 autocorrelation: after dropping the
      first and last 10 trials of each session, Bonferroni-significant positive
      autocorrelation remains at lags 1–7 (lag-1 observed 0.117 → 0.078).
      Model-free post-violation effect remains positive in all 16 animals
      (12.1 → 9.9 pp; both p < 1e-7). Submitted Fig 3–5 GLM fits unchanged.
      Artifacts: `fig03_autocorrelation_edges_trimmed/`, `suppS5_history_robustness.json`.
- [x] Refit and regression-test all Figure 3 GLMs against submitted artifacts.
      All four current summaries match (held-out NLL difference < 2.3e-4;
      weights < 2.4e-6). The tau sweep is the largest residual.

## Figure 4

- [x] Set the psychometric x-range to −2 through 2.
- [x] Apply the requested panel titles and axis labels.
- [x] Correct the “Psychometric” spelling.
- [x] Refit and regression-test both Figure 4 interaction models. Both
      current summaries match the submitted fits (held-out NLL difference
      < 5e-5; weights < 3e-6).

## Figure 5

- [x] Refit and regression-test both submitted comparison experiments.
      Multinomial comparison matches submitted NLLs/weights. The binary
      history models match as well; only the unused binary base optimizer
      remains unstable and is excluded from that check.
- [x] Supply model-summary evidence for the cross-animal variability response.
      The filtered-interaction model improves binary bits/trial for 12 of 16
      animals (mean +0.00050; range −0.0053 in W065 to +0.0049 in W074).

## Figure 6 (new)

- [x] Explain cross-animal variability in the Figure 5B improvement
      (`prepare_fig06_across_animals.py`). Improvement correlates with the
      session-averaged post-violation drop in % correct (r = 0.86,
      p = 2.4e-5) and negatively with trained violation rate (r = −0.76,
      p = 7.2e-4; Spearman −0.54, p = 0.029); trained hit rate does not
      (r = 0.35, p = 0.18).
- [x] Within-animal session split at the median violation rate
      (`prepare_fig06_within_animal.py`). The post-violation drop is larger in
      low-violation sessions (4.08 vs 2.95 pp; paired t-test p = 2.2e-4; 15 of
      16 animals). Post hoc; violation timing within the trial is not recorded.
- [x] Repeat-violation control (text only; printed in `fig06.ipynb` Panel C).
      The drop is not significantly related to repeat-violation probability
      (r = −0.36, p = 0.16); violation rate still predicts the drop controlling
      for it (partial r = −0.84, p = 4.8e-5), although the two correlate
      (r = 0.83). The draft Supporting Figure S7 recovery curve was dropped as
      an indirect test.
- [ ] Synchronize Results, Methods, and Discussion text with
      `REVISION1_Breda2026PLOSOne/revision_fig6_variability.md`.

## Release

- [ ] Replace provisional GitHub and Zenodo references with final URLs/DOI.
- [ ] Synchronize generated statistics with the manuscript and response letter.
- [ ] Reconcile the manuscript's “sixteen rats” description with the
      3,034,200-trial reconstruction count, which includes W082 (17 animals);
      the trained GLM cohort correctly excludes W082 and contains 16 animals.
- [ ] Reconcile the manuscript's 9,968-session reconstruction count with the
      9,952 distinct animal-session pairs in `all_animals_cleaned.csv`.
