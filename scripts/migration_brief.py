"""Summarize a publication source notebook for fidelity migration."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _cell_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _extract_saves(source: str) -> list[str]:
    patterns = (
        r'save_figure\s*\(\s*[^,]+,\s*["\']([^"\']+)["\']',
        r'savefig\s*\(\s*["\']([^"\']+)["\']',
        r'plt\.savefig\s*\(\s*["\']([^"\']+)["\']',
    )
    names: list[str] = []
    for pattern in patterns:
        names.extend(re.findall(pattern, source))
    return names


def _is_stat_cell(source: str) -> bool:
    markers = (
        "print(",
        "display(",
        "compute_overall_ll_ratio",
        "compute_geometric_mean_ll_ratio",
        "overall_likelihood_ratio",
        "geometric_mean_likelihood_ratio",
        ".agg(",
        "mixedlm",
        "linregress",
        "ttest",
        "pearsonr",
        "spearmanr",
    )
    return any(marker in source for marker in markers)


def _is_plot_cell(source: str) -> bool:
    if _extract_saves(source):
        return True
    plot_markers = (
        "plt.subplots",
        "sns.",
        "ax.plot",
        "ax.scatter",
        "heatmap",
        "histplot",
        "pointplot",
        "lineplot",
        "barplot",
    )
    return any(marker in source for marker in plot_markers)


def brief_notebook(path: Path) -> str:
    notebook = json.loads(path.read_text())
    lines = [
        f"# Migration brief: {path.name}",
        "",
        f"Path: `{path}`",
        "",
    ]

    plot_sections: list[str] = []
    stat_sections: list[str] = []
    other_code = 0

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = _cell_source(cell)
        stripped = source.strip()
        if not stripped:
            continue
        saves = _extract_saves(source)
        figsize = re.findall(r"figsize\s*=\s*\([^)]+\)", source)
        legend = re.findall(r"legend\s*\([^)]*\)", source, re.DOTALL)
        header = f"## Code cell {index}"
        if saves:
            block = [header, "", "**Saves:**"]
            block.extend(f"- `{name}`" for name in saves)
            if figsize:
                block.append("")
                block.append(f"**figsize:** `{figsize[0]}`")
            if legend:
                block.append("")
                block.append(f"**legend:** `{legend[0][:120]}...`" if len(legend[0]) > 120 else f"**legend:** `{legend[0]}`")
            block.append("")
            block.append("```python")
            block.append(stripped[:2000] + ("..." if len(stripped) > 2000 else ""))
            block.append("```")
            plot_sections.append("\n".join(block))
        elif _is_stat_cell(source):
            block = [header, "", "**Statistics / prints**", "", "```python", stripped[:2000], "```"]
            stat_sections.append("\n".join(block))
        elif _is_plot_cell(source):
            block = [header, "", "**Plot (no save detected)**", "", "```python", stripped[:1500], "```"]
            plot_sections.append("\n".join(block))
        else:
            other_code += 1

    lines.append(f"- Code cells scanned: {sum(1 for c in notebook['cells'] if c['cell_type']=='code')}")
    lines.append(f"- Plot/save sections: {len(plot_sections)}")
    lines.append(f"- Stat sections: {len(stat_sections)}")
    lines.append(f"- Other code cells: {other_code}")
    lines.append("")

    if plot_sections:
        lines.append("# Plots and saves")
        lines.append("")
        lines.extend(plot_sections)
        lines.append("")

    if stat_sections:
        lines.append("# Statistics and prints")
        lines.append("")
        lines.extend(stat_sections)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path, help="Source .ipynb path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write brief to this file (default: stdout)",
    )
    args = parser.parse_args()
    if not args.notebook.is_file():
        raise SystemExit(f"Not found: {args.notebook}")
    text = brief_notebook(args.notebook.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
