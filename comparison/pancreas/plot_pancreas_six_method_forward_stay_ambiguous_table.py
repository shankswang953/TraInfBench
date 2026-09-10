#!/usr/bin/env python
"""Render six-method transient-core forward/stay/reverse/unknown tables."""

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
import pandas as pd

from trainfbench_plot_style import method_style


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "results/pancreas_six_method_forward_stay_forbidden_20k"
    / "transient_core_transition_summary.csv"
)
DEFAULT_OUTPUT = ROOT / "results/pancreas_six_method_forward_stay_reverse_other_20k"

METHOD_ORDER = (
    "COATI balanced 20k",
    "COATI unbalanced 20k C_y=0.5",
    "CytoBridge balanced 20k",
    "CytoBridge unbalanced 20k",
    "TrajectoryNet 20k",
    "MIOFlow GAGA10 20k",
)
METHOD_LABELS = {
    "COATI balanced 20k": "COATI bal.",
    "COATI unbalanced 20k C_y=0.5": "COATI unbal.",
    "CytoBridge balanced 20k": "CytoBridge bal.",
    "CytoBridge unbalanced 20k": "CytoBridge unbal.",
    "TrajectoryNet 20k": "TrajectoryNet",
    "MIOFlow GAGA10 20k": "MIOFlow",
}

FULL_METHOD_ORDER = (
    "COATI balanced C_y=0.5",
    "COATI unbalanced C_y=0.5",
    "CytoBridge balanced",
    "CytoBridge unbalanced",
    "TrajectoryNet",
    "MIOFlow GAGA10",
)
FULL_METHOD_LABELS = {
    "COATI balanced C_y=0.5": "COATI bal.",
    "COATI unbalanced C_y=0.5": "COATI unbal.",
    "CytoBridge balanced": "CytoBridge bal.",
    "CytoBridge unbalanced": "CytoBridge unbal.",
    "TrajectoryNet": "TrajectoryNet",
    "MIOFlow GAGA10": "MIOFlow",
}


def prepare_table(frame: pd.DataFrame) -> pd.DataFrame:
    observed = set(frame["method"].astype(str))
    if observed == set(METHOD_ORDER):
        method_order = METHOD_ORDER
        method_labels = METHOD_LABELS
    elif observed == set(FULL_METHOD_ORDER):
        method_order = FULL_METHOD_ORDER
        method_labels = FULL_METHOD_LABELS
    else:
        raise ValueError(f"Unexpected methods: {frame['method'].tolist()}")
    indexed = frame.set_index("method").loc[list(method_order)]
    result = pd.DataFrame(
        {
            "Method": [method_labels[method] for method in method_order],
            "Forward": indexed["forward_F"].to_numpy(dtype=float),
            "Stay": indexed["stay"].to_numpy(dtype=float),
            "Reverse": indexed["reverse"].to_numpy(dtype=float),
            "Unknown": indexed["other_forbidden"].to_numpy(dtype=float),
        }
    )
    error = (
        result[["Forward", "Stay", "Reverse", "Unknown"]].sum(axis=1) - 1
    ).abs().max()
    if error > 1e-8:
        raise ValueError(f"Component row-sum error={error}")
    return result


def draw_table(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    title: str,
    output_stem: str,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "font.weight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # 4.1 inches is just under half the width of an A4 page.
    fig, ax = plt.subplots(figsize=(4.09, 2.65), facecolor="white")
    ax.axis("off")
    ax.set_title(
        title,
        fontsize=10.0,
        fontweight="normal",
        pad=-2,
    )
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)

    display = frame.copy()
    display["Method"] = display["Method"].str.replace(
        "CytoBridge ", "CytoBridge\n", regex=False
    )
    for column in display.columns[1:]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    table = ax.table(
        cellText=display.to_numpy(),
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        colWidths=(0.30, 0.175, 0.175, 0.175, 0.175),
        bbox=(0.0, 0.01, 1.0, 0.97),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.0)

    header_colors = (
        "#E7E7E7",
        "#DCEFD9",
        "#E7E7E7",
        "#E8E1F2",
        "#F7EBD9",
    )
    n_rows, n_cols = display.shape
    for col in range(n_cols):
        cell = table[(0, col)]
        cell.set_facecolor(header_colors[col])
        cell.set_edgecolor("#555555")
        cell.set_linewidth(0.65)
        cell.get_text().set_fontweight("normal")
        cell.get_text().set_fontsize(10.0)
    for row in range(1, n_rows + 1):
        for col in range(n_cols):
            cell = table[(row, col)]
            cell.set_facecolor("#FFFFFF" if row % 2 else "#F7F7F7")
            cell.set_edgecolor("#B8B8B8")
            cell.set_linewidth(0.45)
            if col == 0:
                cell.get_text().set_ha("left")
                cell.get_text().set_linespacing(0.85)
                cell.PAD = 0.085
    # Emphasize values, not a selected method.
    for row in range(1, n_rows + 1):
        table[(row, 1)].get_text().set_color("#237A30")
        table[(row, 3)].get_text().set_color("#5E3C99")
        table[(row, 4)].get_text().set_color("#8C510A")

    # Match the method-label accents used by the Edge/Node summary table.
    fig.canvas.draw()
    for row, method in enumerate(frame["Method"], start=1):
        cell = table[(row, 0)]
        x0, y0 = cell.get_xy()
        width = cell.get_width()
        height = cell.get_height()
        ax.add_patch(
            Rectangle(
                (x0 + 0.025 * width, y0 + 0.20 * height),
                0.025 * width,
                0.60 * height,
                transform=ax.transAxes,
                facecolor=method_style(str(method)).color,
                edgecolor="none",
                zorder=5,
            )
        )

    stem = output_dir / output_stem
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            stem.with_suffix(f".{suffix}"),
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.025,
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--title",
        default="Transient endocrine core: E14.5 → held-out E15.5",
    )
    parser.add_argument(
        "--output-stem",
        default="pancreas_six_method_forward_stay_reverse_other_table",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_png = args.output_dir / f"{args.output_stem}.png"
    if output_png.exists() and not args.overwrite:
        raise FileExistsError(f"{output_png} exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = prepare_table(pd.read_csv(args.input))
    table.to_csv(
        args.output_dir / f"{args.output_stem}_values.csv",
        index=False,
    )
    draw_table(
        table,
        args.output_dir,
        title=args.title,
        output_stem=args.output_stem,
    )
    print(table.to_string(index=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
