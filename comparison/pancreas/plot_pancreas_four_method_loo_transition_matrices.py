#!/usr/bin/env python
"""Plot moscot-style pancreas E14.5-to-E15.5 LOO transition matrices."""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from plot_pancreas_coati_celltype_transition_matrix import (
    CELL_TYPE_COLORS,
    CELL_TYPE_ORDER,
    DISPLAY_NAMES,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARISON = (
    ROOT / "results/pancreas_film20k_cytobridge20k_loo_time1_comparison"
)
DEFAULT_OUTPUT = (
    ROOT / "results/pancreas_four_method_loo_transition_matrices_20k_balanced_refresh"
)

METHODS = (
    (
        "COATI balanced",
        "coati_balanced_strict_loo20k_c_y0p5_transition_matrix.csv",
        "uniform",
        "coati_balanced",
    ),
    (
        "COATI unbalanced",
        "coati_filmunbalanced_all1_strict_loo20k_c_y0p5_native_mass_transition_matrix.csv",
        "native particle mass",
        "coati_unbalanced",
    ),
    (
        "CytoBridge balanced",
        "cytobridge_balanced_20k_transition_matrix.csv",
        "uniform",
        "cytobridge_balanced",
    ),
    (
        "CytoBridge unbalanced",
        "cytobridge_unbalanced_20k_native_mass_transition_matrix.csv",
        "native particle mass",
        "cytobridge_unbalanced",
    ),
)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def load_matrix(path: Path) -> tuple[np.ndarray, float]:
    frame = pd.read_csv(path, index_col=0)
    missing_rows = set(CELL_TYPE_ORDER) - set(frame.index.astype(str))
    missing_cols = set(CELL_TYPE_ORDER) - set(frame.columns.astype(str))
    if missing_rows or missing_cols:
        raise ValueError(
            f"{path}: missing rows={sorted(missing_rows)}, missing cols={sorted(missing_cols)}"
        )
    values = frame.loc[list(CELL_TYPE_ORDER), list(CELL_TYPE_ORDER)].to_numpy(
        dtype=np.float64
    )
    if np.any(values < -1e-12):
        raise ValueError(f"{path}: transition matrix contains negative entries")
    row_error = float(np.max(np.abs(values.sum(axis=1) - 1.0)))
    if row_error > 1e-8:
        raise ValueError(f"{path}: row-sum error={row_error}")
    return values, row_error


def add_celltype_strips(
    top_ax: plt.Axes, left_ax: plt.Axes
) -> None:
    strip_colors = [CELL_TYPE_COLORS[cell_type] for cell_type in CELL_TYPE_ORDER]
    strip_cmap = ListedColormap(strip_colors)
    top_ax.imshow(
        np.arange(len(CELL_TYPE_ORDER))[None, :],
        cmap=strip_cmap,
        aspect="auto",
        interpolation="nearest",
    )
    left_ax.imshow(
        np.arange(len(CELL_TYPE_ORDER))[:, None],
        cmap=strip_cmap,
        aspect="auto",
        interpolation="nearest",
    )
    top_ax.set_axis_off()
    left_ax.set_axis_off()


def plot_single(
    matrix: np.ndarray,
    method: str,
    weighting: str,
    stem: str,
    output_dir: Path,
) -> None:
    fig = plt.figure(figsize=(8.25, 8.8), facecolor="white")
    grid = fig.add_gridspec(
        4,
        3,
        width_ratios=(0.18, 1.0, 0.05),
        height_ratios=(0.055, 1.0, 0.34, 0.075),
        left=0.27,
        right=0.94,
        bottom=0.10,
        top=0.90,
        wspace=0.015,
        hspace=0.015,
    )
    top_ax = fig.add_subplot(grid[0, 1])
    left_ax = fig.add_subplot(grid[1, 0])
    ax = fig.add_subplot(grid[1, 1])
    color_ax = fig.add_subplot(grid[3, 1])
    add_celltype_strips(top_ax, left_ax)

    image = ax.imshow(
        matrix,
        cmap="viridis",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )
    ticks = np.arange(len(CELL_TYPE_ORDER))
    display = [DISPLAY_NAMES[cell_type] for cell_type in CELL_TYPE_ORDER]
    ax.set_xticks(ticks, display, rotation=58, ha="right", rotation_mode="anchor")
    ax.set_yticks(ticks, display)
    ax.set_xlabel("Target cell type (E15.5)", labelpad=8)
    ax.set_ylabel("Source cell type (E14.5)", labelpad=8)
    ax.tick_params(length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_color("#303030")
        spine.set_linewidth(0.8)
    colorbar = fig.colorbar(
        image, cax=color_ax, orientation="horizontal", ticks=(0.0, 0.5, 1.0)
    )
    colorbar.set_label("Cell-type transition probability", labelpad=2)
    colorbar.ax.tick_params(length=2, pad=2)
    fig.suptitle(
        f"{method}: E14.5 → E15.5",
        x=0.61,
        y=0.955,
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.61,
        0.925,
        f"strict LOO, 20k; {weighting} weighting within source cell type",
        ha="center",
        fontsize=8.5,
        color="#4A4A4A",
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output_dir / f"{stem}_e145_to_e155_transition_matrix.{suffix}",
            dpi=500 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def plot_four_panel(
    matrices: dict[str, np.ndarray], output_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 11.0), facecolor="white")
    labels = [DISPLAY_NAMES[cell_type] for cell_type in CELL_TYPE_ORDER]
    ticks = np.arange(len(CELL_TYPE_ORDER))
    image = None
    for index, ((method, _, weighting, _), ax) in enumerate(
        zip(METHODS, axes.ravel())
    ):
        image = ax.imshow(
            matrices[method],
            cmap="viridis",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(f"{method}\n({weighting})", fontsize=10, fontweight="bold")
        row, col = divmod(index, 2)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels if row == 1 else [], rotation=58, ha="right")
        ax.set_yticklabels(labels if col == 0 else [])
        ax.tick_params(length=0)
        if col == 0:
            ax.set_ylabel("Source cell type (E14.5)")
    assert image is not None
    fig.subplots_adjust(
        left=0.16,
        right=0.98,
        bottom=0.25,
        top=0.93,
        hspace=0.18,
        wspace=0.08,
    )
    color_ax = fig.add_axes((0.30, 0.060, 0.40, 0.018))
    colorbar = fig.colorbar(
        image,
        cax=color_ax,
        orientation="horizontal",
        ticks=(0.0, 0.5, 1.0),
    )
    colorbar.set_label("Cell-type transition probability")
    fig.text(0.5, 0.135, "Target cell type (E15.5)", ha="center", fontsize=10)
    fig.suptitle(
        "Pancreas E14.5 → E15.5 strict LOO transition matrices",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output_dir / f"all_four_e145_to_e155_transition_matrices.{suffix}",
            dpi=500 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    expected = tuple(
        args.output_dir / f"{stem}_e145_to_e155_transition_matrix.png"
        for _, _, _, stem in METHODS
    ) + (
        args.output_dir / "all_four_e145_to_e155_transition_matrices.png",
        args.output_dir / "manifest.json",
    )
    if any(path.exists() for path in expected) and not args.overwrite:
        raise FileExistsError("Outputs exist; pass --overwrite to replace them")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    matrices: dict[str, np.ndarray] = {}
    manifest_rows: list[dict[str, object]] = []
    for method, filename, weighting, stem in METHODS:
        path = args.comparison_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        matrix, row_error = load_matrix(path)
        matrices[method] = matrix
        pd.DataFrame(
            matrix, index=CELL_TYPE_ORDER, columns=CELL_TYPE_ORDER
        ).rename_axis(index="source_cell_type", columns="target_cell_type").to_csv(
            args.output_dir / f"{stem}_e145_to_e155_transition_matrix.csv"
        )
        plot_single(matrix, method, weighting, stem, args.output_dir)
        manifest_rows.append(
            {
                "method": method,
                "input_matrix": str(path),
                "weighting": weighting,
                "max_row_sum_error": row_error,
            }
        )
    plot_four_panel(matrices, args.output_dir)
    pd.DataFrame(manifest_rows).to_csv(
        args.output_dir / "matrix_inputs_and_validation.csv", index=False
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "task": "moscot-style four-method pancreas LOO transition matrices",
                "source_stage": "E14.5",
                "target_stage": "E15.5 (held out during training)",
                "training_iterations": 20000,
                "soft_celltype_assignment": "50-NN voting against observed E15.5 RNA PCA",
                "matrix_normalization": "Each source-cell-type row sums to one",
                "balanced_weighting": "uniform particles within source type",
                "unbalanced_weighting": "native learned particle mass within source type",
                "color_scale": [0.0, 1.0],
                "cell_type_order": list(CELL_TYPE_ORDER),
                "methods": manifest_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(pd.DataFrame(manifest_rows).to_string(index=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
