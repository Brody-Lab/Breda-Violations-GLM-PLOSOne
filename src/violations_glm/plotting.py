"""Shared plotting helpers for publication panels."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from violations_glm.config import MODEL_ANIMAL_IDS
from violations_glm.paths import figures_dir

FIGURE_DPI = 300


def configure_publication_dpi(dpi: int = FIGURE_DPI) -> None:
    """Set display and save DPI for all figures in this process."""
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi


def animal_palette() -> dict[str, tuple[float, float, float]]:
    """Fixed husl color per publication animal, shared across panels."""
    colors = sns.color_palette("husl", len(MODEL_ANIMAL_IDS))
    return dict(zip(MODEL_ANIMAL_IDS, colors, strict=True))


def animal_association_scatter(
    ax: Axes,
    animals: pd.DataFrame,
    x: str,
    y: str,
    statistics: Mapping[str, Any],
    *,
    palette: Mapping[str, Any],
    legend: bool = False,
    show_n: bool = True,
) -> None:
    """Animal-level scatter with the OLS fit and a Pearson r/p annotation."""
    sns.scatterplot(
        data=animals,
        x=x,
        y=y,
        hue="animal_id",
        palette=palette,
        s=75,
        legend=legend,
        ax=ax,
    )
    if legend:
        ax.legend(title="Animal ID", frameon=False, loc="upper right")
    x_values = np.linspace(animals[x].min(), animals[x].max(), 100)
    ax.plot(
        x_values,
        statistics["intercept"] + statistics["slope"] * x_values,
        color="black",
        linewidth=2,
    )
    stats_text = (
        f"r = {statistics['pearson_r']:.2f}\n"
        f"p = {statistics['pearson_p']:.3g}"
    )
    if show_n:
        stats_text += f"\nN = {statistics['n_animals']}"
    ax.text(
        0.04,
        0.96,
        stats_text,
        transform=ax.transAxes,
        va="top",
    )
    sns.despine(ax=ax)


_STATS_CORNERS = {
    "upper left": (0.04, 0.96, "left", "top"),
    "upper right": (0.96, 0.96, "right", "top"),
    "lower left": (0.04, 0.04, "left", "bottom"),
    "lower right": (0.96, 0.04, "right", "bottom"),
}


def colored_animal_scatter(
    ax: Axes,
    animals: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    statistics: Mapping[str, Any],
    *,
    cmap: str,
    norm: Normalize,
    fit_line: bool = True,
    annotate_animals: bool = True,
    stats_corner: Literal[
        "upper left", "upper right", "lower left", "lower right"
    ] = "upper left",
) -> PathCollection:
    """Labeled animal scatter with points colored by a third variable.

    Returns the scatter artist so the caller can attach a colorbar if wanted.
    """
    points = ax.scatter(
        animals[x],
        animals[y],
        c=animals[color],
        cmap=cmap,
        norm=norm,
        s=90,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )
    if annotate_animals:
        for _, row in animals.iterrows():
            ax.annotate(
                row["animal_id"],
                (row[x], row[y]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=10,
            )
    if fit_line:
        x_values = np.linspace(animals[x].min(), animals[x].max(), 100)
        ax.plot(
            x_values,
            statistics["intercept"] + statistics["slope"] * x_values,
            color="black",
            linewidth=2,
        )
    x_text, y_text, ha, va = _STATS_CORNERS[stats_corner]
    ax.text(
        x_text,
        y_text,
        f"r = {statistics['pearson_r']:.2f}\np = {statistics['pearson_p']:.3g}",
        transform=ax.transAxes,
        ha=ha,
        va=va,
    )
    sns.despine(ax=ax)
    return points


def save_figure(
    figure: Figure,
    stem: str,
    *,
    output_dir: Path | None = None,
    dpi: int = FIGURE_DPI,
) -> Path:
    """Save a publication panel as a tightly bounded PNG at FIGURE_DPI."""
    destination = output_dir or figures_dir()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{stem}.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def plot_stim_pairs(
    animal_summary: pd.DataFrame,
    ax: Axes | None = None,
    *,
    alpha: float = 0.5,
) -> None:
    """Scatter one animal's stimulus pairs; marker size encodes trial count."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(
        animal_summary["s_a"],
        animal_summary["s_b"],
        marker=",",
        s=animal_summary["trial_count"] / 100,
        alpha=alpha,
    )
    animal_id = animal_summary["animal_id"].iloc[0]
    n_stim_pairs = int(animal_summary["n_stim_pairs"].iloc[0])
    ax.axline(xy1=(60, 60), slope=1, color="lightgray", linestyle="--")
    ax.set(
        xlabel="Stimulus A dB",
        ylabel="Stimulus B dB",
        title=f"{animal_id} Stimulus Pairs: N = {n_stim_pairs}",
    )


def plot_stimulus_pairs_by_animal(
    summary: pd.DataFrame,
    *,
    alpha: float = 0.5,
    n_rows: int = 4,
    n_cols: int = 4,
) -> tuple[Figure, Any]:
    """4×4 grid of stimulus-pair scatters for Supporting Figure S2."""
    n_animals = summary["animal_id"].nunique()
    print(f"Number of animals: {n_animals}")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    for index, (_animal_id, animal) in enumerate(summary.groupby("animal_id", sort=True)):
        row_idx, col_idx = np.unravel_index(index, (n_rows, n_cols))
        current_ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]
        plot_stim_pairs(animal, current_ax, alpha=alpha)

    for ax_idx in range(n_animals, n_rows * n_cols):
        fig.delaxes(axes.flatten()[ax_idx] if n_rows > 1 else axes[ax_idx])

    plt.tight_layout()
    return fig, axes
