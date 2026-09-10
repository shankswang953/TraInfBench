#!/usr/bin/env python
"""Plot the strict-LOO external ATAC peak-contrast Spearman table."""

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
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from trainfbench_plot_style import apply_nature_rc, method_style


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "results/palate_strict_loo_external_epigenetic_evidence_with_rna_only"
    / "loo_external_epigenetic_metrics.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "results/palate_strict_loo_external_epigenetic_evidence_with_rna_only"
)

METHODS = (
    ("COATI-B", "COATI bal.", "COATI balanced"),
    ("COATI-U", "COATI unbal.", "COATI unbalanced"),
    ("CytoBridge-B", "CytoBridge bal.", "CytoBridge balanced"),
    ("CytoBridge-U", "CytoBridge unbal.", "CytoBridge unbalanced"),
    ("MIOFlow", "MIOFlow", "MIOFlow"),
    ("TrajectoryNet", "TrajectoryNet", "TrajectoryNet"),
    ("RNAonly-B", "OT(RNA)", "Balanced RNA-only"),
    ("RNAonly-U", "UOT(RNA)", "Unbalanced RNA-only"),
)

COLUMNS = (
    ("loo_time1", "H3K27ac", "E13.5\nH3K27ac"),
    ("loo_time2", "H3K27ac", "E14.0\nH3K27ac"),
    ("loo_time1", "TF-bound active CRE", "E13.5\nTF CRE"),
    ("loo_time2", "TF-bound active CRE", "E14.0\nTF CRE"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_values(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    values = np.empty((len(METHODS), len(COLUMNS)), dtype=float)
    for row_index, (method, _, _) in enumerate(METHODS):
        for column_index, (scenario, family, _) in enumerate(COLUMNS):
            selected = frame[
                frame["scenario"].eq(scenario)
                & frame["family"].eq(family)
                & frame["method"].eq(method)
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"Expected one row for {scenario}, {family}, {method}; "
                    f"found {len(selected)}"
                )
            values[row_index, column_index] = float(
                selected.iloc[0]["peak_contrast_spearman"]
            )
    return values


def save_figure(figure: plt.Figure, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "palate_external_peak_contrast_spearman_table"
    outputs: list[Path] = []
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        kwargs: dict[str, object] = {"bbox_inches": "tight", "pad_inches": 0.015}
        if suffix == "png":
            kwargs["dpi"] = 450
        figure.savefig(path, **kwargs)
        outputs.append(path)
    return outputs


def main() -> None:
    args = parse_args()
    values = load_values(args.input)
    apply_nature_rc(font_size=10)

    figure = plt.figure(figsize=(4.13, 2.75), facecolor="white")
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.5,
        0.955,
        "External regulatory peak contrast recovery",
        ha="center",
        va="top",
        color="#222222",
        fontweight="normal",
    )
    axis.text(
        0.5,
        0.865,
        r"Spearman $\rho$(predicted, observed) $\uparrow$",
        ha="center",
        va="top",
        color="#222222",
    )

    method_left = 0.035
    metric_centers = np.asarray([0.405, 0.585, 0.765, 0.935])
    header_y = 0.715
    for center, (_, _, label) in zip(metric_centers, COLUMNS):
        axis.text(
            center,
            header_y,
            label,
            ha="center",
            va="center",
            color="#222222",
            fontweight="normal",
            linespacing=0.95,
        )
    axis.plot([0.025, 0.995], [0.615, 0.615], color="#777777", lw=0.65)

    row_centers = np.linspace(0.545, 0.085, len(METHODS))
    row_height = 0.060
    for row_index, (center_y, (_, display, style_name)) in enumerate(
        zip(row_centers, METHODS)
    ):
        if row_index % 2 == 0:
            axis.add_patch(
                Rectangle(
                    (0.025, center_y - row_height / 2),
                    0.97,
                    row_height,
                    facecolor="#F5F5F5",
                    edgecolor="none",
                    zorder=0,
                )
            )
        style = method_style(style_name)
        axis.add_patch(
            Rectangle(
                (method_left, center_y - 0.017),
                0.028,
                0.034,
                facecolor=style.color,
                edgecolor=style.color,
                linewidth=0.6,
            )
        )
        axis.text(
            method_left + 0.04,
            center_y,
            display,
            ha="left",
            va="center",
            color="#222222",
        )
        for column_index, center_x in enumerate(metric_centers):
            value = values[row_index, column_index]
            axis.text(
                center_x,
                center_y,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="#222222",
                fontweight="normal",
            )

    outputs = save_figure(figure, args.output_dir)
    plt.close(figure)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
