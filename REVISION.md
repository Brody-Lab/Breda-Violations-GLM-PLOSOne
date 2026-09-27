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
- [x] Move hit-versus-violation scatters and MixedLM/OLS/sklearn diagnostics
      from the original Figure 1F block into Supporting Figure S5 Panel A
      (both session+animal and animal+regression figures retained).
- [x] Plot representative raw behavioral sequences as Figure 1 Panel B
      (`fig01b_example_sequences`; DPI 300 via `configure_publication_dpi`).
- [x] Plot violation rate by trial position within sessions (now S5 Panel B).
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

### Open analysis decision

- [ ] Decide whether earlier training-stage and delay-length analyses belong in
      S5 or remain discussion-only. The current S5 includes early-training
      and time-to-stage-4 panels (C–D); delay-length is not implemented.
- [ ] Finalize the draft `suppS5.ipynb` (univariate predictors of trained
      violation rate; no panels saved yet). It keeps the submitted
      hit/violation MixedLM (β = −0.342) and animal-mean OLS (R² = 0.190) and
      adds animal-level (N = 16) associations with stage 1 violation rate
      (r = 0.48, p = 0.060), sessions in stage 1 (r = −0.13, p = 0.63), and
      sessions to trained (r = 0.20, p = 0.46). Session position replaces the
      scratchpad's first/middle/last-50 windows: excluding the first and last
      10 trials, violations rise 0.35 pp per 10% of the session (13 of 16
      rats, p = 0.0003; 16.3% to 18.5%), and edge-aligned trials show the
      start spike (95% on trial 1) and end ramp (54% on the final trial, with
      hit rate falling to 56%). The stages 1–3 / first-stage-4 panels remain
      only in the scratchpad.

## Figure 3

- [x] Recompute the autocorrelation null distribution with 10,000 permutations.
- [x] Use random seed 47 and empirical p-values `(count + 1) / (n + 1)`.
- [x] Apply Bonferroni correction over lags 1–20 only.
- [ ] Update panel 3a, its caption, Methods, and Results from the same output.
      After Bonferroni correction over lags 1–20, 18 lags remain significant;
      lags 18 and 20 do not.
- [x] Session-edge robustness for Figure 3 autocorrelation: after dropping the
      first and last 10 trials of each session, Bonferroni-significant positive
      autocorrelation remains at lags 1–5 (lag-1 observed 0.117 → 0.078).
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

## Release

- [ ] Replace provisional GitHub and Zenodo references with final URLs/DOI.
- [ ] Synchronize generated statistics with the manuscript and response letter.
- [ ] Reconcile the manuscript's “sixteen rats” description with the
      3,034,200-trial reconstruction count, which includes W082 (17 animals);
      the trained GLM cohort correctly excludes W082 and contains 16 animals.
- [ ] Reconcile the manuscript's 9,968-session reconstruction count with the
      9,952 distinct animal-session pairs in `all_animals_cleaned.csv`.
