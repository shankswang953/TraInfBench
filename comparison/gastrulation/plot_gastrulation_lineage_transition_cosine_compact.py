#!/usr/bin/env python
"""Plot compact RNA/ATAC transition-cosine comparison tables."""

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
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "common"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from trainfbench_plot_style import apply_nature_rc, method_style  # noqa: E402


DEFAULT_INPUT = (
    ROOT
    / "results/gastrulation_lineage_gene_peak_module_comparison"
    / "magnitude_comparison"
    / "method_lineage_transition_cosine.csv"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.parent
METHODS = (
    ("COATI balanced", "COATI bal."),
    ("COATI unbalanced", "COATI unbal."),
    ("CytoBridge balanced", "CytoBridge bal."),
    ("CytoBridge unbalanced", "CytoBridge unbal."),
    ("MIOFlow", "MIOFlow"),
    ("TrajectoryNet", "TrajectoryNet"),
)
GROUPS = (
    ("Spinal cord", "RNA", 0.0),
    ("Spinal cord", "ATAC", 1.0),
    ("Somitic mesoderm", "RNA", 2.25),
    ("Somitic mesoderm", "ATAC", 3.25),
)


def _draw_stage(axis: plt.Axes, data: pd.DataFrame, stage: str) -> None:
    subset = data[data["stage"].eq(stage)].set_index(
        ["lineage", "modality", "method"]
    )
    y_positions = np.arange(len(METHODS))[::-1].astype(float)
    axis.set_xlim(-1.75, 4.25)
    axis.set_ylim(-0.55, 7.35)
    axis.axis("off")

    axis.text(
        1.55,
        7.18,
        f"{stage} transition cosine",
        ha="center",
        va="top",
        fontsize=10.5,
    )
    axis.text(0.5, 6.42, "Spinal cord", ha="center", va="center", fontsize=9.2)
    axis.text(
        2.75,
        6.42,
        "Somitic mesoderm",
        ha="center",
        va="center",
        fontsize=9.2,
    )
    for _, modality, x in GROUPS:
        axis.text(x + 0.27, 5.87, modality, ha="center", va="center", fontsize=8.4)
        axis.plot(
            [x - 0.08, x - 0.08],
            [-0.25, 5.42],
            color="#D9D9D9",
            linewidth=0.65,
            zorder=0,
        )
    axis.plot(
        [1.93, 1.93],
        [-0.35, 6.7],
        color="#C8C8C8",
        linewidth=0.75,
        zorder=0,
    )

    for y, (method, display) in zip(y_positions, METHODS):
        axis.text(
            -0.16,
            y,
            display,
            ha="right",
            va="center",
            fontsize=8.1,
        )
        style = method_style(method)
        for lineage, modality, x in GROUPS:
            value = float(subset.loc[(lineage, modality, method), "transition_cosine"])
            axis.plot(
                x + 0.06,
                y,
                marker=style.marker,
                markersize=4.2,
                color=style.color,
                markerfacecolor=style.color,
                markeredgecolor="white",
                markeredgewidth=0.4,
                zorder=2,
            )
            axis.text(
                x + 0.19,
                y,
                f"{value:.3f}",
                ha="left",
                va="center",
                fontsize=8.1,
            )


def _save(fig: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output / f"{stem}.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.04,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input.resolve())
    apply_nature_rc(font_size=8.0)

    stages = [stage for stage in ("E8.0", "E8.5") if stage in set(data["stage"])]
    fig, axes = plt.subplots(len(stages), 1, figsize=(6.55, 5.35))
    axes = np.atleast_1d(axes)
    for axis, stage in zip(axes, stages):
        _draw_stage(axis, data, stage)
    fig.suptitle(
        "NMP branch-transition cosine",
        fontsize=11.5,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.958,
        "Fixed lineage genes and training-selected cis peaks · higher is better",
        ha="center",
        va="top",
        fontsize=7.6,
    )
    fig.subplots_adjust(left=0.03, right=0.995, bottom=0.02, top=0.925, hspace=0.18)
    _save(fig, output, "lineage_transition_cosine_compact")
    plt.close(fig)

    for stage in stages:
        single, axis = plt.subplots(1, 1, figsize=(6.55, 2.8))
        _draw_stage(axis, data, stage)
        single.subplots_adjust(left=0.03, right=0.995, bottom=0.02, top=0.98)
        stem = f"lineage_transition_cosine_{stage.lower().replace('.', '')}"
        _save(single, output, stem)
        plt.close(single)

    print(f"Wrote compact cosine plots to {output}")


if __name__ == "__main__":
    main()
