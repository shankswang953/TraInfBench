#!/usr/bin/env python3
"""Stack the three human-cerebral W2 comparisons in one A4-safe column."""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from trainfbench_plot_style import apply_nature_rc, method_style


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/human_cerebral_publication_redraw_10pt"
RNA_INPUT = ROOT / (
    "results/human_cerebral_full_biological_interpretability/"
    "10_raw_cell_fate_recovery/numeric_tables_by_time/"
    "appendix_all_methods_rna_w2_by_time.csv"
)
ORIGINAL_ATAC_INPUT = ROOT / (
    "results/human_cerebral_full_biological_interpretability/"
    "13_original_atac_w2_current/"
    "common_T_all_methods_original_cell_uniform_by_time.csv"
)
ORIGINAL_ATAC_REFERENCE = ROOT / (
    "results/human_cerebral_full_biological_interpretability/"
    "13_original_atac_w2_current/"
    "observed_metacell_to_original_atac_by_time.csv"
)
METACELL_ATAC_INPUT = ROOT / (
    "results/human_cerebral_full_biological_interpretability/"
    "14_metacell_atac_w2_all7time/method_summary_by_time.csv"
)

METHODS = (
    ("COATI bal. (Sync)", "COATI balanced"),
    ("COATI unbal. (Sync)", "COATI unbalanced"),
    ("CytoBridge bal.", "CytoBridge balanced"),
    ("CytoBridge unbal.", "CytoBridge unbalanced"),
    ("MIOFlow", "MIOFlow"),
    ("TrajectoryNet", "TrajectoryNet"),
    ("RNA-only bal.", "Balanced RNA-only"),
    ("RNA-only unbal.", "Unbalanced RNA-only"),
)
DISPLAY_ORDER = tuple(display for display, _ in METHODS)


def setup_style() -> None:
    apply_nature_rc(font_size=10.0)
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 10.0,
        "font.weight": "normal",
        "axes.titlesize": 10.0,
        "axes.titleweight": "normal",
        "axes.labelsize": 10.0,
        "axes.labelweight": "normal",
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def visual(style_name: str) -> tuple[str, str, str]:
    style = method_style(style_name)
    face = style.color if style.markerfacecolor is None else style.markerfacecolor
    edge = style.color if style.markeredgecolor is None else style.markeredgecolor
    if style_name == "TrajectoryNet":
        edge = face
    return style.marker, face, edge


def method_handle(display: str, style_name: str) -> Line2D:
    marker, face, edge = visual(style_name)
    return Line2D(
        [], [], linestyle="none", marker=marker, markersize=5.5,
        markerfacecolor=face, markeredgecolor=edge, markeredgewidth=0.65,
        label=display.replace(" (Sync)", ""),
    )


def finish_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#E3E3E3", linewidth=0.55, zorder=0)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#666666")
        spine.set_linewidth(0.75)


def draw_panel(
    ax: plt.Axes,
    table_path: Path,
    times: tuple[str, ...],
    title: str,
    ylabel: str,
    reference: pd.Series | None = None,
) -> None:
    table = pd.read_csv(table_path).set_index("Method").reindex(DISPLAY_ORDER)
    values = table.loc[:, list(times)]
    if values.isna().any().any():
        raise ValueError(f"Missing values in {table_path}")

    centers = np.arange(len(times), dtype=float)
    right_offset = 0.18 if reference is not None else 0.23
    offsets = np.linspace(-0.23, right_offset, len(METHODS))
    plotted = []
    for offset, (display, style_name) in zip(offsets, METHODS):
        marker, face, edge = visual(style_name)
        y = values.loc[display].to_numpy(float)
        plotted.extend(y.tolist())
        ax.scatter(
            centers + offset, y, s=34, marker=marker,
            facecolors=face, edgecolors=edge, linewidths=0.65, zorder=3,
        )

    if reference is not None:
        ref = reference.reindex(times).to_numpy(float)
        plotted.extend(ref.tolist())
        ax.scatter(
            centers + 0.245, ref, s=30, marker="o", facecolors="none",
            edgecolors="#777777", linewidths=0.85, zorder=4,
        )

    lo, hi = min(plotted), max(plotted)
    pad = max(1e-4, 0.08 * (hi - lo))
    ax.set_xlim(-0.48, len(times) - 0.52)
    ax.set_ylim(max(0, lo - pad), hi + pad)
    ax.set_xticks(centers, times)
    ax.set_xlabel("Time", labelpad=1.5)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_title(title, loc="left", pad=3, fontweight="normal")
    finish_axis(ax)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f"three_w2_comparisons_vertical_a4.{suffix}"
             for suffix in ("pdf", "png", "svg")]
    existing = [path for path in paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to overwrite; pass --overwrite: " + str(existing))

    setup_style()
    # Match the 3.50 in RNA figure width and use the full A4 page height.
    fig, axes = plt.subplots(3, 1, figsize=(3.50, 11.69))
    draw_panel(
        axes[0], RNA_INPUT, ("D7", "D9", "D11", "D12", "D18", "D21"),
        "RNA vs original cells",
        r"Sliced $W_2$ to original cells $\downarrow$",
    )
    original_atac_reference = pd.read_csv(
        ORIGINAL_ATAC_REFERENCE, index_col=0
    ).iloc[:, 0]
    draw_panel(
        axes[1], ORIGINAL_ATAC_INPUT, ("D7", "D9", "D11", "D21"),
        "ATAC vs original cells",
        r"Sliced $W_2$ to original cells $\downarrow$",
        reference=original_atac_reference,
    )
    draw_panel(
        axes[2], METACELL_ATAC_INPUT,
        ("D4", "D7", "D9", "D11", "D12", "D18", "D21"),
        "ATAC vs metacells",
        r"Sliced $W_2$ to metacells $\downarrow$",
    )

    handles = [method_handle(display, style_name)
               for display, style_name in METHODS]
    handles.append(Line2D(
        [], [], linestyle="none", marker="o", markersize=5.2,
        markerfacecolor="none", markeredgecolor="#777777",
        markeredgewidth=0.85, label="Observed metacells",
    ))
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.006),
        ncol=2, frameon=False, handletextpad=0.25, columnspacing=0.72,
        labelspacing=0.24,
    )
    fig.subplots_adjust(
        left=0.225, right=0.985, top=0.972, bottom=0.12, hspace=0.30,
    )

    for path in paths:
        fig.savefig(
            path, dpi=600 if path.suffix == ".png" else None,
            facecolor="white",
        )
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
