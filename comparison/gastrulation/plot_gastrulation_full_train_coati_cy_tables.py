#!/usr/bin/env python
"""Render two 6 x 9 raw-value tables for the COATI C_y sweep."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from trainfbench_plot_style import METHOD_STYLES


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "results/gastrulation_full_train_coati_cy_sweep_revised"
    / "full_train_coati_cy_sweep_revised_metrics.csv"
)
DEFAULT_OUTPUT = ROOT / "results/gastrulation_full_train_coati_cy_sweep_revised"
CY_VALUES = tuple(round(i / 10, 1) for i in range(1, 10))
ROWS = (
    ("RNA support ↑", "rna_target_support_recall"),
    ("ATAC support ↑", "atac_target_support_recall"),
    (r"RNA $W_2$ ↓", "rna_w2"),
    (r"ATAC $W_2$ ↓", "atac_w2"),
    ("RNA comp. similarity ↑", "rna_composition_similarity"),
    ("ATAC comp. similarity ↑", "atac_composition_similarity"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def format_pair(stage_frame: pd.DataFrame, metric: str, cy: float) -> str:
    values = {}
    for stage in ("E8.0", "E8.5"):
        match = stage_frame.loc[
            stage_frame["stage"].eq(stage) & np.isclose(stage_frame["C_y"], cy),
            metric,
        ]
        if len(match) != 1:
            return ""
        values[stage] = float(match.iloc[0])
    return f"{values['E8.0']:.3f}\n{values['E8.5']:.3f}"


def draw_table(frame: pd.DataFrame, variant: str, output_dir: Path) -> list[Path]:
    method_name = f"COATI {variant}"
    method_key = "COATI balanced" if variant == "balanced" else "COATI unbalanced"
    method_color = METHOD_STYLES[method_key].color
    outputs = [
        output_dir / f"gastrulation_full_train_coati_{variant}_cy_raw_table.{suffix}"
        for suffix in ("png", "svg", "pdf")
    ]

    fig, ax = plt.subplots(figsize=(8.0, 2.65))
    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.02, top=0.99)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    left = 0.018
    label_right = 0.225
    right = 0.995
    header_top = 0.740
    header_bottom = 0.645
    body_bottom = 0.035
    row_height = (header_bottom - body_bottom) / len(ROWS)
    col_width = (right - label_right) / len(CY_VALUES)

    ax.text(
        0.5,
        0.955,
        method_name,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="normal",
    )
    ax.text(
        0.5,
        0.865,
        r"Full-train metrics; columns: $C_y$; cells: E8.0 (top), E8.5 (bottom)",
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )
    ax.add_patch(
        Rectangle(
            (left, header_bottom),
            right - left,
            header_top - header_bottom,
            facecolor=method_color,
            alpha=0.11,
            edgecolor="none",
        )
    )
    ax.text(
        left + 0.006,
        0.5 * (header_top + header_bottom),
        "Metric",
        ha="left",
        va="center",
        fontsize=10,
    )
    for j, cy in enumerate(CY_VALUES):
        x = label_right + (j + 0.5) * col_width
        ax.text(
            x,
            0.5 * (header_top + header_bottom),
            f"{cy:.1f}",
            ha="center",
            va="center",
            fontsize=10,
        )

    variant_frame = frame.loc[frame["variant"].eq(variant)].copy()
    for i, (label, metric) in enumerate(ROWS):
        y_top = header_bottom - i * row_height
        y_bottom = y_top - row_height
        y = 0.5 * (y_top + y_bottom)
        if i % 2 == 1:
            ax.add_patch(
                Rectangle(
                    (left, y_bottom),
                    right - left,
                    row_height,
                    facecolor="#F5F5F5",
                    edgecolor="none",
                )
            )
        ax.text(
            left + 0.006,
            y,
            label,
            ha="left",
            va="center",
            fontsize=10,
        )
        for j, cy in enumerate(CY_VALUES):
            value = format_pair(variant_frame, metric, cy)
            x = label_right + (j + 0.5) * col_width
            ax.text(
                x,
                y,
                value,
                ha="center",
                va="center",
                fontsize=10,
                linespacing=1.15,
                color="#777777" if value == "" else "#000000",
            )

    for j in range(len(CY_VALUES) + 1):
        x = label_right + j * col_width
        ax.plot([x, x], [body_bottom, header_top], color="#D8D8D8", linewidth=0.55)
    for i in range(len(ROWS) + 1):
        y = header_bottom - i * row_height
        linewidth = 1.0 if i in (0, 2, 4, 6) else 0.55
        ax.plot([left, right], [y, y], color="#B8B8B8", linewidth=linewidth)
    ax.plot([left, right], [header_top, header_top], color="#777777", linewidth=0.8)
    ax.plot([label_right, label_right], [body_bottom, header_top], color="#A0A0A0", linewidth=0.8)

    for path in outputs:
        kwargs = {"facecolor": "white"}
        if path.suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected = [
        args.output_dir / f"gastrulation_full_train_coati_{variant}_cy_raw_table.{suffix}"
        for variant in ("balanced", "unbalanced")
        for suffix in ("png", "svg", "pdf")
    ]
    existing = [path for path in expected if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite outputs; pass --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    frame = pd.read_csv(args.input)
    for variant in ("balanced", "unbalanced"):
        for path in draw_table(frame, variant, args.output_dir):
            print(path)


if __name__ == "__main__":
    main()
