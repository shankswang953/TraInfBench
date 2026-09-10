#!/usr/bin/env python3
"""Compact portrait 4x2 original-cell versus metacell RNA/ATAC UMAP figure."""

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

from trainfbench_plot_style import NATURE_CUD


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/human_cerebral_publication_redraw_10pt"
RNA_COORDS = ROOT / "results/human_cerebral_original_metacell_umaps/umap_coordinates.csv.gz"
ATAC_COORDS = ROOT / "results/human_cerebral_original_atac_frozen_lsi12_umap/umap_coordinates.csv.gz"

DAYS = (4, 7, 9, 11, 12, 16, 18, 21)
TIME_COLORS = {
    4: "#440154",
    7: "#414487",
    9: "#2A788E",
    11: "#22A884",
    12: "#7AD151",
    16: "#B8DE29",
    18: "#F6C445",
    21: "#E76F51",
}
CELL_TYPES = ("RG", "IPC", "EN", "IN", "Other")
CELLTYPE_COLORS = {
    "RG": NATURE_CUD["blue"],
    "IPC": NATURE_CUD["orange"],
    "EN": NATURE_CUD["bluish_green"],
    "IN": NATURE_CUD["vermillion"],
    "Other": "#999999",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "font.weight": "normal",
            "axes.titlesize": 10.0,
            "axes.titleweight": "normal",
            "legend.fontsize": 10.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def limits(arrays: tuple[np.ndarray, ...]) -> tuple[tuple[float, float], tuple[float, float]]:
    all_xy = np.vstack(arrays)
    low = all_xy.min(axis=0)
    high = all_xy.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    return (
        (float(low[0] - 0.025 * span[0]), float(high[0] + 0.025 * span[0])),
        (float(low[1] - 0.025 * span[1]), float(high[1] + 0.025 * span[1])),
    )


def style_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, pad=2, linespacing=0.92, fontweight="normal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#666666")
        spine.set_linewidth(0.65)


def scatter_day(axis: plt.Axes, frame: pd.DataFrame, size: float) -> None:
    for day in DAYS:
        local = frame.loc[frame.day.eq(day)]
        if local.empty:
            continue
        axis.scatter(
            local.UMAP1,
            local.UMAP2,
            s=size,
            color=TIME_COLORS[day],
            alpha=0.58,
            linewidths=0,
            rasterized=True,
        )


def scatter_celltype(axis: plt.Axes, frame: pd.DataFrame, size: float) -> None:
    for label in ("Other", "IPC", "RG", "EN", "IN"):
        local = frame.loc[frame.cell_type.eq(label)]
        if local.empty:
            continue
        axis.scatter(
            local.UMAP1,
            local.UMAP2,
            s=size,
            color=CELLTYPE_COLORS[label],
            alpha=0.52 if label == "Other" else 0.62,
            linewidths=0,
            rasterized=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    target = output / "original_vs_metacell_rna_atac_umap.pdf"
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {target}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    rna = pd.read_csv(RNA_COORDS, low_memory=False)
    atac = pd.read_csv(ATAC_COORDS, low_memory=False)
    frames = (
        rna.loc[rna.representation.eq("Original RNA")].copy(),
        rna.loc[rna.representation.eq("Metacell RNA")].copy(),
        atac.loc[atac.representation.eq("Original ATAC raw")].copy(),
        atac.loc[atac.representation.eq("Metacell ATAC raw")].copy(),
    )
    if any(frame.empty for frame in frames):
        raise ValueError("A required original-cell or metacell UMAP representation is missing")

    setup_style()
    figure, axes = plt.subplots(4, 2, figsize=(3.85, 7.05))
    figure.subplots_adjust(
        left=0.13,
        right=0.995,
        bottom=0.145,
        top=0.965,
        wspace=0.08,
        hspace=0.09,
    )
    titles = ("Original RNA", "RNA metacells", "Original ATAC", "ATAC metacells")
    pair_limits = (
        limits((frames[0][["UMAP1", "UMAP2"]].to_numpy(), frames[1][["UMAP1", "UMAP2"]].to_numpy())),
        limits((frames[2][["UMAP1", "UMAP2"]].to_numpy(), frames[3][["UMAP1", "UMAP2"]].to_numpy())),
    )
    for row, (frame, title) in enumerate(zip(frames, titles)):
        size = 0.35 if row in (0, 2) else 0.60
        scatter_day(axes[row, 0], frame, size)
        scatter_celltype(axes[row, 1], frame, size)
        style_axis(axes[row, 0], "Time point" if row == 0 else "")
        style_axis(axes[row, 1], "Cell type" if row == 0 else "")
        axes[row, 0].set_ylabel(title, rotation=90, labelpad=3)
        current_limits = pair_limits[0 if row < 2 else 1]
        for axis in axes[row, :]:
            axis.set_xlim(*current_limits[0])
            axis.set_ylim(*current_limits[1])

    time_handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=5.0,
               markerfacecolor=TIME_COLORS[day], markeredgewidth=0, label=f"D{day}")
        for day in DAYS
    ]
    celltype_handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=5.0,
               markerfacecolor=CELLTYPE_COLORS[label], markeredgewidth=0, label=label)
        for label in CELL_TYPES
    ]
    figure.legend(
        handles=time_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.050),
        ncol=4,
        frameon=False,
        handletextpad=0.15,
        columnspacing=0.75,
    )
    figure.legend(
        handles=celltype_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        ncol=5,
        frameon=False,
        handletextpad=0.2,
        columnspacing=0.75,
    )

    for suffix in ("pdf", "png", "svg"):
        figure.savefig(
            output / f"original_vs_metacell_rna_atac_umap.{suffix}",
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.01,
            facecolor="white",
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
