#!/usr/bin/env python
"""Plot a compact publication table for paired E8.5 NMP T validation."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PREFIX = (
    ROOT
    / "results/gastrulation_crispr_t_e85_to_e875"
    / "e85_nmp_t_mapping_validation_table"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    rows = [
        ("Ordinary E8.5", "86.9%", "0.312", "0.0294"),
        ("CRISPR WT", "83.9%", "0.364", "0.0519"),
        ("T-KO", "78.7%", "0.431", "0.0719"),
    ]
    headers = (
        "Dataset",
        "T–ATAC\nagreement",
        "Paired\ndistance",
        "Sinkhorn\ndiv.",
    )

    fig = plt.figure(figsize=(3.85, 1.35))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(
        0.5,
        0.91,
        "Paired E8.5 NMP validation of T",
        ha="center",
        va="center",
        fontsize=12,
        fontfamily="Arial",
        fontweight="normal",
    )

    column_x = (0.025, 0.385, 0.675, 0.885)
    alignments = ("left", "center", "center", "center")
    header_y = 0.68
    for x, label, alignment in zip(column_x, headers, alignments):
        ax.text(
            x,
            header_y,
            label,
            ha=alignment,
            va="center",
            fontsize=10,
            fontfamily="Arial",
            fontweight="normal",
            linespacing=0.95,
            multialignment="center",
        )

    line_color = "#9A9A9A"
    ax.plot((0.02, 0.98), (0.55, 0.55), color=line_color, linewidth=0.65)

    row_y = (0.42, 0.25, 0.08)
    for row_idx, (y, values) in enumerate(zip(row_y, rows)):
        for x, value, alignment in zip(column_x, values, alignments):
            ax.text(
                x,
                y,
                value,
                ha=alignment,
                va="center",
                fontsize=10,
                fontfamily="Arial",
                fontweight="normal",
            )
        if row_idx < len(rows) - 1:
            ax.plot(
                (0.02, 0.98),
                (y - 0.09, y - 0.09),
                color="#D7D7D7",
                linewidth=0.45,
            )

    fig.savefig(
        args.output_prefix.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.015,
    )
    fig.savefig(
        args.output_prefix.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.015,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
