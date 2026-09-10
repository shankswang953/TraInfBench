#!/usr/bin/env python
"""Plot the two full-trained pancreas top-two target tables for an appendix."""

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
from matplotlib.patches import Rectangle
import pandas as pd

from trainfbench_plot_style import method_style


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_E155 = (
    ROOT
    / "results/pancreas_full_moscot_transition_trends_15nn_20k_cy05"
    / "full_all_source_top_transitions.csv"
)
DEFAULT_E165 = (
    ROOT
    / "results/pancreas_full_e155_to_e165_transition_trends_15nn_20k_cy05"
    / "full_e155_to_e165_all_source_top_transitions.csv"
)
DEFAULT_OUTPUT = ROOT / "results/pancreas_full_top_two_targets_appendix_15nn_20k_cy05"

METHODS = (
    ("COATI balanced C_y=0.5", "COATI bal."),
    ("COATI unbalanced C_y=0.5", "COATI unbal."),
    ("CytoBridge balanced", "CytoBridge bal."),
    ("CytoBridge unbalanced", "CytoBridge unbal."),
    ("TrajectoryNet", "TrajectoryNet"),
    ("MIOFlow GAGA10", "MIOFlow"),
)

SOURCES = (
    ("Alpha", "Alpha", "Alpha"),
    ("Beta", "Beta", "Beta"),
    ("Delta", "Delta", "Delta"),
    ("Fev+ Alpha", "Fev+ α", "Alpha"),
    ("Fev+ Beta", "Fev+ β", "Beta"),
    ("Fev+ Delta", "Fev+ δ", "Delta"),
    ("Eps. progenitors", "Eps. p.", "Fev+ Delta"),
    ("Epsilon", "Epsilon", "Alpha"),
)

GROUPS = (
    ("Mature identity", 0, 3),
    ("Committed Fev+", 3, 3),
    ("δ/ε", 6, 1),
    ("ε route", 7, 1),
)

FONT_SIZE = 10.0
METHOD_WIDTH_INCHES = 1.38
DATA_COLUMN_WIDTH_INCHES = (5.80 - METHOD_WIDTH_INCHES) / len(SOURCES)
LEFT_PANEL_WIDTH_INCHES = METHOD_WIDTH_INCHES + len(SOURCES) * DATA_COLUMN_WIDTH_INCHES
RIGHT_PANEL_WIDTH_INCHES = len(SOURCES) * DATA_COLUMN_WIDTH_INCHES
FIGURE_WIDTH_INCHES = LEFT_PANEL_WIDTH_INCHES + RIGHT_PANEL_WIDTH_INCHES + 0.12
TITLE_HEIGHT_INCHES = 0.40
GROUP_HEIGHT_INCHES = 0.30
HEADER_HEIGHT_INCHES = 0.72
ROW_HEIGHT_INCHES = 0.68


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e155", type=Path, default=DEFAULT_E155)
    parser.add_argument("--e165", type=Path, default=DEFAULT_E165)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def short_target(value: str) -> str:
    return {
        "Alpha": "α",
        "Beta": "β",
        "Delta": "δ",
        "Epsilon": "ε",
        "Fev+": "F+",
        "Fev+ Alpha": "Fα",
        "Fev+ Beta": "Fβ",
        "Fev+ Delta": "Fδ",
        "Ngn3 low": "N3l",
        "Ngn3 high": "N3h",
        "Ngn3 high cycling": "N3c",
        "Eps. progenitors": "Ep",
        "Mat. Acinar": "Mat-ac",
        "Imm. Acinar": "Imm-ac",
        "Prlf. Ductal": "Prlf-d",
        "Ductal": "Ductal",
    }.get(value, value)


def compact_probability(value: float) -> str:
    return f"{value:.3f}".removeprefix("0")


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": FONT_SIZE,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_cell(
    axis: plt.Axes,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    fill: str,
    *,
    border: str,
    linewidth: float = 0.55,
) -> None:
    axis.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor=fill,
            edgecolor=border,
            linewidth=linewidth,
        )
    )


def validate_table(table: pd.DataFrame, path: Path) -> pd.DataFrame:
    indexed = table.set_index(["method", "source"])
    for method, _ in METHODS:
        for source, _, _ in SOURCES:
            if (method, source) not in indexed.index:
                raise KeyError(f"Missing {(method, source)} in {path}")
    return indexed


def draw_panel(
    axis: plt.Axes,
    indexed: pd.DataFrame,
    *,
    title: str,
    show_method: bool,
    panel_width: float,
) -> None:
    border = "#B8B8B8"
    title_fill = "#FFFFFF"
    group_fill = "#E5E5E5"
    header_fill = "#F1F1F1"
    method_fill = "#F7F7F7"
    first_fill = "#E5F3E2"
    second_fill = "#FFF2C7"
    absent_fill = "#F7E8E8"
    text_color = "#202020"
    muted = "#666666"

    method_width = METHOD_WIDTH_INCHES if show_method else 0.0
    data_width = DATA_COLUMN_WIDTH_INCHES
    title_height = TITLE_HEIGHT_INCHES
    group_height = GROUP_HEIGHT_INCHES
    header_height = HEADER_HEIGHT_INCHES
    row_height = ROW_HEIGHT_INCHES
    panel_height = title_height + group_height + header_height + row_height * len(METHODS)

    axis.set_xlim(0.0, panel_width)
    axis.set_ylim(panel_height, 0.0)
    axis.axis("off")

    draw_cell(
        axis,
        0.0,
        panel_width,
        0.0,
        title_height,
        title_fill,
        border=title_fill,
        linewidth=0.0,
    )
    axis.text(
        0.0,
        0.5 * title_height,
        title,
        ha="left",
        va="center",
        color=text_color,
    )

    group_y0 = title_height
    if show_method:
        draw_cell(
            axis,
            0.0,
            method_width,
            group_y0,
            group_y0 + group_height,
            group_fill,
            border=border,
        )
    for label, start, span in GROUPS:
        x0 = method_width + start * data_width
        x1 = x0 + span * data_width
        draw_cell(
            axis,
            x0,
            x1,
            group_y0,
            group_y0 + group_height,
            group_fill,
            border=border,
        )
        axis.text(
            0.5 * (x0 + x1),
            group_y0 + 0.5 * group_height,
            label,
            ha="center",
            va="center",
            color=text_color,
        )

    header_y0 = group_y0 + group_height
    if show_method:
        draw_cell(
            axis,
            0.0,
            method_width,
            header_y0,
            header_y0 + header_height,
            header_fill,
            border=border,
        )
        axis.text(
            0.08,
            header_y0 + 0.5 * header_height,
            "Method",
            ha="left",
            va="center",
            color=text_color,
        )
    for column, (_, display, expected) in enumerate(SOURCES):
        x0 = method_width + column * data_width
        x1 = x0 + data_width
        draw_cell(
            axis,
            x0,
            x1,
            header_y0,
            header_y0 + header_height,
            header_fill,
            border=border,
        )
        axis.text(
            0.5 * (x0 + x1),
            header_y0 + 0.25,
            display,
            ha="center",
            va="center",
            color=text_color,
        )
        axis.text(
            0.5 * (x0 + x1),
            header_y0 + 0.50,
            f"→ {short_target(expected)}",
            ha="center",
            va="center",
            color=muted,
        )

    data_y0 = header_y0 + header_height
    for row_index, (method, display_method) in enumerate(METHODS):
        y0 = data_y0 + row_index * row_height
        y1 = y0 + row_height
        if show_method:
            draw_cell(
                axis,
                0.0,
                method_width,
                y0,
                y1,
                method_fill,
                border=border,
            )
            swatch = method_style(display_method).color
            axis.add_patch(
                Rectangle(
                    (0.08, y0 + 0.11),
                    0.07,
                    row_height - 0.22,
                    facecolor=swatch,
                    edgecolor="none",
                )
            )
            axis.text(
                0.21,
                0.5 * (y0 + y1),
                display_method,
                ha="left",
                va="center",
                color=text_color,
            )

        for column, (source, _, expected) in enumerate(SOURCES):
            record = indexed.loc[(method, source)]
            top = str(record["top_target"])
            second = str(record["second_target"])
            fill = (
                first_fill
                if top == expected
                else second_fill
                if second == expected
                else absent_fill
            )
            x0 = method_width + column * data_width
            x1 = x0 + data_width
            draw_cell(axis, x0, x1, y0, y1, fill, border=border)
            axis.text(
                0.5 * (x0 + x1),
                0.5 * (y0 + y1),
                f"{short_target(top)} {compact_probability(float(record['top_probability']))}\n"
                f"{short_target(second)} {compact_probability(float(record['second_probability']))}",
                ha="center",
                va="center",
                color=text_color,
                linespacing=1.20,
            )


def main() -> None:
    args = parse_args()
    output_stem = args.output_dir / "pancreas_full_top_two_targets_appendix"
    outputs = tuple(output_stem.with_suffix(f".{extension}") for extension in ("png", "pdf", "svg"))
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise FileExistsError("Outputs exist; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configure_plotting()
    e155 = validate_table(pd.read_csv(args.e155), args.e155)
    e165 = validate_table(pd.read_csv(args.e165), args.e165)

    panel_height = (
        TITLE_HEIGHT_INCHES
        + GROUP_HEIGHT_INCHES
        + HEADER_HEIGHT_INCHES
        + ROW_HEIGHT_INCHES * len(METHODS)
    )
    figure_height = panel_height
    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES, figure_height), dpi=300)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(LEFT_PANEL_WIDTH_INCHES, RIGHT_PANEL_WIDTH_INCHES),
        wspace=0.025,
    )
    draw_panel(
        fig.add_subplot(grid[0, 0]),
        e155,
        title="Predicted targets at E15.5",
        show_method=True,
        panel_width=LEFT_PANEL_WIDTH_INCHES,
    )
    draw_panel(
        fig.add_subplot(grid[0, 1]),
        e165,
        title="Predicted targets at E16.5",
        show_method=False,
        panel_width=RIGHT_PANEL_WIDTH_INCHES,
    )

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    for output in outputs:
        fig.savefig(
            output,
            dpi=600 if output.suffix == ".png" else None,
            bbox_inches="tight",
            pad_inches=0.025,
            facecolor="white",
        )
    plt.close(fig)

    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "e155_input": str(args.e155.resolve()),
                "e165_input": str(args.e165.resolve()),
                "display": "full-width appendix table; 10 pt Arial; normal font weight",
                "cell_content": "top two target cell types and row-normalized transition probabilities",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
