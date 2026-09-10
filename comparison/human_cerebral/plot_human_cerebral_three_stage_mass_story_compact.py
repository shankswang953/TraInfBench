#!/usr/bin/env python3
"""Render the three-stage mass-allocation table at half-A4 width."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from plot_human_cerebral_three_stage_mass_story import COLORS, METHODS, ROWS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/human_cerebral_three_stage_mass_story_v1"

EVIDENCE_LABELS = (
    "S-phase partial rho\nD7 to D9",
    "S-phase partial rho\nD9 to D11",
    "S-phase partial rho",
    "Non-S regionalization-\ngene AUROC",
    "Late-lineage gene\nAUROC (pooled)",
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "font.weight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def text(
    axis: plt.Axes,
    x: float,
    y: float,
    value: str,
    *,
    ha: str = "left",
    color: str = "#222222",
    linespacing: float = 1.0,
) -> None:
    axis.text(
        x,
        y,
        value,
        transform=axis.transAxes,
        ha=ha,
        va="center",
        fontsize=10.0,
        fontweight="normal",
        color=color,
        linespacing=linespacing,
    )


def draw(output: Path) -> None:
    configure_style()
    fig = plt.figure(figsize=(4.08, 3.40), facecolor="white")
    axis = fig.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    left, right, bottom, top = 0.006, 0.994, 0.008, 0.992
    header_bottom = 0.82
    widths = (0.10, 0.36, 0.18, 0.18, 0.18)
    xs = [left]
    for width in widths:
        xs.append(xs[-1] + (right - left) * width)
    body_rows = ROWS[:-1]
    row_height = (header_bottom - bottom) / len(body_rows)

    # Stage-level bands.
    group_sizes = (2, 2, 1)
    consumed = 0
    for group, group_size in enumerate(group_sizes):
        y_top = header_bottom - consumed * row_height
        y_bottom = y_top - group_size * row_height
        consumed += group_size
        axis.add_patch(
            Rectangle(
                (left, y_bottom),
                right - left,
                y_top - y_bottom,
                transform=axis.transAxes,
                facecolor="#F5F5F5" if group % 2 else "#FBFBFB",
                edgecolor="none",
            )
        )

    # Outer frame and column rules.
    axis.add_patch(
        Rectangle(
            (left, bottom),
            right - left,
            top - bottom,
            transform=axis.transAxes,
            facecolor="none",
            edgecolor="#555555",
            linewidth=0.8,
        )
    )
    for x in xs[1:-1]:
        axis.plot([x, x], [bottom, top], color="#C8C8C8", lw=0.55, transform=axis.transAxes)
    axis.plot([left, right], [header_bottom, header_bottom], color="#666666", lw=0.75, transform=axis.transAxes)

    # Group boundaries and within-group evidence-row rules.
    group_boundaries = {2, 4}
    for row in range(1, len(body_rows)):
        y = header_bottom - row * row_height
        if row in group_boundaries:
            axis.plot([left, right], [y, y], color="#777777", lw=0.7, transform=axis.transAxes)
        else:
            axis.plot([xs[1], right], [y, y], color="#D9D9D9", lw=0.45, transform=axis.transAxes)

    # Header.
    text(axis, xs[0] + 0.008, (header_bottom + top) / 2, "Stage")
    text(axis, xs[1] + 0.012, (header_bottom + top) / 2, "Evidence")
    header_labels = ("COATI\nunbal.", "UOT\n(RNA)", "CytoBridge\nunbal.")
    for col, (method, label) in enumerate(zip(METHODS, header_labels), start=2):
        center = (xs[col] + xs[col + 1]) / 2
        text(axis, center, (header_bottom + top) / 2, label, ha="center", linespacing=1.08)

    # Body.
    stage_labels = ("D7\nto\nD11", "D12\nto\nD18", "D18\nto\nD21")
    consumed = 0
    for stage, group_size in zip(stage_labels, group_sizes):
        center_y = header_bottom - (consumed + group_size / 2) * row_height
        consumed += group_size
        text(axis, xs[0] + 0.008, center_y, stage, linespacing=1.08)

    for index, (row, evidence) in enumerate(zip(body_rows, EVIDENCE_LABELS)):
        center_y = header_bottom - (index + 0.5) * row_height
        text(axis, xs[1] + 0.012, center_y, evidence, linespacing=1.18)
        for col, method in enumerate(METHODS, start=2):
            center = (xs[col] + xs[col + 1]) / 2
            value = str(row[method]).replace("−", "-")
            text(axis, center, center_y, value, ha="center")

    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(
            output / f"three_stage_mass_allocation_story_compact.{suffix}",
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.01,
            facecolor="white",
        )
    plt.close(fig)

    caption = (
        "Stage-dependent biological alignment of modeled mass allocation. "
        "D7-D11 represents neural induction with cycling-coupled allocation; "
        "D12-D18 represents the shift toward telencephalic regionalization; "
        "D18-D21 represents regional fate consolidation. Values summarize "
        "stage-appropriate gene-program associations for COATI unbalanced, "
        "UOT(RNA), and CytoBridge unbalanced."
    )
    (output / "three_stage_mass_allocation_story_compact_caption.txt").write_text(caption + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    target = args.output_dir / "three_stage_mass_allocation_story_compact.pdf"
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {target}; pass --overwrite")
    draw(args.output_dir.resolve())


if __name__ == "__main__":
    main()
