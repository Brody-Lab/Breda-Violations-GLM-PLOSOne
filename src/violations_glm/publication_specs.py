"""Exact model specifications for the eight submitted fit artifacts."""

from __future__ import annotations

from violations_glm.fitting import (
    STANDARD_TAUS,
    ExperimentSpec,
    ModelSpec,
)

SINGLE_SHARED = (
    ("session", "session"),
    ("bias", "bias"),
    ("s_a", "s_a"),
    ("s_b", "s_b"),
)
SINGLE_TRAILING = (
    ("prev_stim_avg", "prev_stim_unmasked"),
    ("prev_correct", "prev_correct"),
    ("prev_choice", "prev_choice"),
)
COMPARE_SHARED = (
    ("session", "session"),
    ("bias", "bias"),
    ("s_a", "s_a"),
    ("s_b", "s_b"),
    ("prev_avg_stim", "prev_stim_masked"),
    ("prev_correct", "prev_correct"),
    ("prev_choice", "prev_choice"),
)


def fig03_raw() -> ExperimentSpec:
    return ExperimentSpec(
        name="2025_07_19_fig3_fit_prev_violation_rs47",
        models=(
            ModelSpec(
                "prev_violation",
                "multi",
                SINGLE_SHARED
                + (("prev_violation", "prev_violation"),)
                + SINGLE_TRAILING,
            ),
        ),
    )


def fig03_filtered() -> ExperimentSpec:
    return ExperimentSpec(
        name="2025_07_19_fig3_fit_filt_prev_violation_rs47",
        models=(
            ModelSpec(
                "prev_violation",
                "multi",
                SINGLE_SHARED
                + (("filt_prev_violation", "filt_prev_violation"),)
                + SINGLE_TRAILING,
            ),
        ),
    )


def fig03_tau_sweep() -> ExperimentSpec:
    return ExperimentSpec(
        name="2025_07_19_fig3_sweep_prev_viol_taus_rs47",
        models=(
            ModelSpec(
                "tau_sweep_prev_filt_violation",
                "multi",
                SINGLE_SHARED
                + SINGLE_TRAILING
                + (("filt_prev_violation", "filt_prev_violation"),),
            ),
        ),
        taus=STANDARD_TAUS,
    )


def fig03_compare() -> ExperimentSpec:
    return ExperimentSpec(
        name="2025_07_19_fig3_compare_prev_viol_rs47",
        models=(
            ModelSpec("base", "multi", COMPARE_SHARED),
            ModelSpec(
                "prev_viol_binary",
                "multi",
                COMPARE_SHARED + (("prev_violation", "prev_violation"),),
            ),
            ModelSpec(
                "prev_viol_filtered",
                "multi",
                COMPARE_SHARED + (("filt_prev_violation", "filt_prev_violation"),),
            ),
        ),
        null_mode="multi",
    )


def fig04_binary_interactions() -> ExperimentSpec:
    operations = (
        ("session", "session"),
        ("bias", "bias"),
        ("s_a", "s_a"),
        ("sa_x_prev_viol", "sa_binary_interaction"),
        ("s_b", "s_b"),
        ("sb_x_prev_viol", "sb_binary_interaction"),
        ("prev_violation", "prev_violation"),
    ) + SINGLE_TRAILING
    return ExperimentSpec(
        name="2025_07_22_fig4_fit_binary_prev_viol_stim_intrx_n_rs47",
        models=(ModelSpec("prev_viol_binary_itrx", "multi", operations),),
    )


def fig04_filtered_interactions() -> ExperimentSpec:
    operations = (
        ("session", "session"),
        ("bias", "bias"),
        ("s_a", "s_a"),
        ("sa_x_prev_viol", "sa_filtered_interaction"),
        ("s_b", "s_b"),
        ("sb_x_prev_viol", "sb_filtered_interaction"),
        ("filt_prev_violation", "filt_prev_violation"),
    ) + SINGLE_TRAILING
    return ExperimentSpec(
        name="2025_07_22_fig4_fit_filtered_prev_viol_stim_intrx_n_rs47",
        models=(ModelSpec("prev_viol_filtered_itrx", "multi", operations),),
    )


def _fig05_models(*, binary_base: bool) -> tuple[ModelSpec, ...]:
    lr_only = binary_base
    return (
        ModelSpec("base", "binary" if binary_base else "multi", COMPARE_SHARED),
        ModelSpec(
            "prev_viol_binary",
            "multi",
            COMPARE_SHARED + (("prev_violation", "prev_violation"),),
            lr_only,
        ),
        ModelSpec(
            "prev_viol_binary_itrx",
            "multi",
            COMPARE_SHARED
            + (
                ("prev_violation", "prev_violation"),
                ("sa_x_prev_viol", "sa_binary_interaction"),
                ("sb_x_prev_viol", "sb_binary_interaction"),
            ),
            lr_only,
        ),
        ModelSpec(
            "prev_viol_filtered",
            "multi",
            COMPARE_SHARED + (("filt_prev_violation", "filt_prev_violation"),),
            lr_only,
        ),
        ModelSpec(
            "prev_viol_filtered_itrx",
            "multi",
            COMPARE_SHARED
            + (
                ("filt_prev_violation", "filt_prev_violation"),
                ("sa_x_prev_viol", "sa_filtered_interaction"),
                ("sb_x_prev_viol", "sb_filtered_interaction"),
            ),
            lr_only,
        ),
    )


def fig05_multinomial() -> ExperimentSpec:
    return ExperimentSpec(
        name="2025_07_22_fig5_compare_multi_intrxs_rs47",
        models=_fig05_models(binary_base=False),
        null_mode="multi",
    )


def fig05_binary_comparison() -> ExperimentSpec:
    return ExperimentSpec(
        name="2025_07_22_fig5_compare_to_binary_rs47",
        models=_fig05_models(binary_base=True),
        null_mode="binary",
    )


PUBLICATION_SPECS = {
    spec.name: spec
    for spec in (
        fig03_raw(),
        fig03_filtered(),
        fig03_tau_sweep(),
        fig03_compare(),
        fig04_binary_interactions(),
        fig04_filtered_interactions(),
        fig05_multinomial(),
        fig05_binary_comparison(),
    )
}
