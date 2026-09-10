#!/usr/bin/env python
"""Plot observed composition changes and six-method signed residuals."""

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
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EARLY = (
    ROOT
    / "results/pancreas_six_method_full_e145_e155_lineage_composition_15nn_20k_cy05"
    / "six_method_composition_change_long.csv"
)
DEFAULT_LATE = (
    ROOT
    / "results/pancreas_six_method_full_e155_e165_lineage_composition_15nn_20k_cy05"
    / "six_method_composition_change_long.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/pancreas_six_method_full_composition_residual_heatmap_15nn_20k_cy05"
)

CELL_TYPES = (
    "Ngn3 low",
    "Ngn3 high cycling",
    "Ngn3 high",
    "Eps. progenitors",
    "Fev+",
    "Fev+ Alpha",
    "Fev+ Beta",
    "Fev+ Delta",
    "Alpha",
    "Beta",
    "Delta",
    "Epsilon",
)
CELL_TYPE_LABELS = tuple(
    "Ngn3 high cyc."
    if celltype == "Ngn3 high cycling"
    else "Eps. prog."
    if celltype == "Eps. progenitors"
    else celltype
    for celltype in CELL_TYPES
)
METHODS = (
    "COATI balanced C_y=0.5",
    "COATI unbalanced C_y=0.5",
    "CytoBridge balanced",
    "CytoBridge unbalanced",
    "TrajectoryNet",
    "MIOFlow GAGA10",
)
METHOD_LABELS = (
    "COATI bal.",
    "COATI unbal.",
    "CytoBridge bal.",
    "CytoBridge unbal.",
    "TrajectoryNet",
    "MIOFlow",
)
INTERVALS = (
    ("E14.5→E15.5", DEFAULT_EARLY),
    ("E15.5→E16.5", DEFAULT_LATE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early", type=Path, default=DEFAULT_EARLY)
    parser.add_argument("--late", type=Path, default=DEFAULT_LATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_interval(path: Path, interval: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    data = pd.read_csv(path)
    required = {
        "method",
        "celltype",
        "observed_delta_pp",
        "predicted_delta_pp",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    observed = (
        data.drop_duplicates("celltype")
        .set_index("celltype")
        .loc[list(CELL_TYPES), "observed_delta_pp"]
        .to_numpy(dtype=float)
    )
    predicted = (
        data.pivot(index="celltype", columns="method", values="predicted_delta_pp")
        .loc[list(CELL_TYPES), list(METHODS)]
        .to_numpy(dtype=float)
    )
    residual = predicted - observed[:, None]
    rows = []
    for celltype_index, celltype in enumerate(CELL_TYPES):
        for method_index, method in enumerate(METHODS):
            rows.append(
                {
                    "interval": interval,
                    "celltype": celltype,
                    "method": method,
                    "observed_delta_pp": observed[celltype_index],
                    "predicted_delta_pp": predicted[celltype_index, method_index],
                    "residual_pp": residual[celltype_index, method_index],
                    "absolute_residual_pp": abs(
                        residual[celltype_index, method_index]
                    ),
                }
            )
    return observed, residual, pd.DataFrame(rows)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "font.weight": "normal",
            "axes.titlesize": 10.0,
            "axes.titleweight": "normal",
            "axes.labelsize": 10.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot(
    interval_data: list[tuple[str, np.ndarray, np.ndarray]],
    output_dir: Path,
) -> float:
    configure_style()
    limit = max(
        float(np.max(np.abs(residual))) for _, _, residual in interval_data
    )
    limit = float(np.ceil(limit * 2.0) / 2.0)
    norm = Normalize(vmin=0.0, vmax=limit)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "absolute_residual",
        ("#FFFFFF", "#FEE5D9", "#FC9272", "#CB181D", "#67000D"),
    )

    fig = plt.figure(figsize=(3.64, 4.35), facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 1.0),
        left=0.30,
        right=0.985,
        top=0.72,
        bottom=0.065,
        wspace=0.14,
    )
    heatmaps = []
    column_labels = METHOD_LABELS
    for column, (interval, observed, residual) in enumerate(interval_data):
        panel_ax = fig.add_subplot(grid[0, column])
        display = np.abs(residual)
        heatmaps.append(panel_ax.imshow(display, cmap=cmap, norm=norm, aspect="auto"))

        panel_ax.set_xticks(np.arange(len(column_labels)), column_labels)
        panel_ax.xaxis.tick_top()
        panel_ax.tick_params(axis="x", length=0, pad=3)
        for label in panel_ax.get_xticklabels():
            label.set_rotation(90)
            label.set_ha("left")
            label.set_va("center")
            label.set_rotation_mode("anchor")
        panel_ax.set_yticks(
            np.arange(len(CELL_TYPES)),
            CELL_TYPE_LABELS if column == 0 else [""] * len(CELL_TYPES),
        )
        panel_ax.tick_params(axis="y", length=0, pad=4)
        panel_ax.set_xticks(
            np.arange(-0.5, len(column_labels), 1.0), minor=True
        )
        panel_ax.set_yticks(
            np.arange(-0.5, len(CELL_TYPES), 1.0), minor=True
        )
        panel_ax.grid(which="minor", color="white", linewidth=0.55)
        panel_ax.tick_params(which="minor", bottom=False, left=False)
        for spine in panel_ax.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.55)

        for boundary in (3.5, 7.5):
            panel_ax.axhline(boundary, color="#333333", linewidth=0.8)
        panel_ax.set_xlabel(interval, labelpad=4)

    colorbar_ax = fig.add_axes((0.035, 0.785, 0.205, 0.022))
    colorbar = fig.colorbar(
        heatmaps[0], cax=colorbar_ax, orientation="horizontal"
    )
    midpoint = 0.5 * limit
    colorbar.set_ticks([0.0, midpoint, limit])
    colorbar.set_ticklabels(["0", f"{midpoint:g}", f"{limit:g}"])
    colorbar.ax.tick_params(length=2.5, width=0.55, pad=2)
    colorbar.outline.set_linewidth(0.55)
    colorbar.ax.set_title("Residual\n(percentage)", loc="left", pad=3)

    stem = output_dir / "pancreas_six_method_full_composition_residual_heatmap"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            stem.with_suffix(f".{suffix}"),
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.035,
            facecolor="white",
        )
    plt.close(fig)
    return limit


def main() -> None:
    args = parse_args()
    sentinel = args.output_dir / "pancreas_six_method_full_composition_residual_heatmap.png"
    if sentinel.exists() and not args.overwrite:
        raise FileExistsError(f"{sentinel} exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = {"E14.5→E15.5": args.early, "E15.5→E16.5": args.late}
    interval_data = []
    long_frames = []
    for interval, _ in INTERVALS:
        observed, residual, long = load_interval(source_paths[interval], interval)
        interval_data.append((interval, observed, residual))
        long_frames.append(long)
    combined = pd.concat(long_frames, ignore_index=True)
    combined.to_csv(args.output_dir / "composition_residuals_long.csv", index=False)
    color_limit = plot(interval_data, args.output_dir)

    manifest = {
        "task": "Six-method full-trained pancreas composition residual heatmap",
        "intervals": list(source_paths),
        "inputs": {key: str(value) for key, value in source_paths.items()},
        "residual": (
            "absolute value of predicted composition change minus observed "
            "composition change, percentage points"
        ),
        "celltypes": list(CELL_TYPES),
        "methods": list(METHODS),
        "color_limit_pp": color_limit,
        "display": (
            "10 pt Arial; intervals side by side; observed column omitted; "
            "all cells are absolute residuals; no MAE; shared scale; "
            "horizontal colorbar in the upper-left margin"
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
