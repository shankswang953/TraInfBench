#!/usr/bin/env python
"""Render a compact Edge-versus-Node ATAC fidelity results table."""

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

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from trainfbench_plot_style import method_style


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EDGE = (
    ROOT
    / "results/pancreas_transition_target_atac_module_support_15nn_20k_cy05"
    / "observed_target_atac_profile_weighted_summary.csv"
)
DEFAULT_NODE = (
    ROOT
    / "results/pancreas_distribution_atac_module_fidelity_15nn_20k_cy05"
    / "celltype_atac_module_distribution_biological_group_summary.csv"
)
DEFAULT_OUTPUT = ROOT / "results/pancreas_edge_node_atac_summary_table_15nn_20k_cy05"
METHODS = (
    "COATI bal.",
    "COATI unbal.",
    "CytoBridge bal.",
    "CytoBridge unbal.",
    "TrajectoryNet",
    "MIOFlow",
)
NODE_GROUPS = ("Overall", "Progenitor", "Committed", "Mature")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge", type=Path, default=DEFAULT_EDGE)
    parser.add_argument("--node", type=Path, default=DEFAULT_NODE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_values(edge_path: Path, node_path: Path) -> pd.DataFrame:
    edge = pd.read_csv(edge_path).set_index("method")
    node = pd.read_csv(node_path)
    node = node[node["weighting"].eq("matched weighting")]
    node_pivot = node.pivot(
        index="method_label",
        columns="biological_group",
        values="observed_cell_number_weighted_centroid_srmse",
    )

    rows: list[dict[str, object]] = []
    for method in METHODS:
        rows.append(
            {
                "method": method,
                "Edge": float(edge.loc[method, "cell_number_weighted_srmse"]),
                **{
                    group: float(node_pivot.loc[method, group])
                    for group in NODE_GROUPS
                },
            }
        )
    return pd.DataFrame(rows)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "font.weight": "normal",
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
    *,
    facecolor: str,
    edgecolor: str = "#B8B8B8",
    linewidth: float = 0.55,
) -> None:
    axis.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
    )


def plot_table(values: pd.DataFrame, output_dir: Path) -> None:
    configure_plotting()
    x_edges = np.asarray((0.0, 1.65, 2.67, 3.6625, 4.655, 5.6475, 6.64))
    top_height = 0.64
    sub_height = 0.78
    row_height = 0.72
    data_top = top_height + sub_height
    data_bottom = data_top + row_height * len(values)
    total_height = data_bottom

    fig, axis = plt.subplots(figsize=(4.10, 3.00))
    axis.set_xlim(x_edges[0], x_edges[-1])
    axis.set_ylim(total_height, 0.0)
    axis.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)

    neutral_header = "#E7E7E7"
    edge_header = "#D9EAF7"
    edge_subheader = "#EAF3F9"
    node_header = "#DDEDD7"
    node_subheader = "#EDF5EA"
    best_edge = "#C5DEEF"
    best_node = "#D5E9CF"

    draw_cell(
        axis, x_edges[0], x_edges[1], 0.0, data_top,
        facecolor=neutral_header,
    )
    axis.text(
        0.5 * (x_edges[0] + x_edges[1]),
        0.5 * data_top,
        "Method",
        ha="center",
        va="center",
    )

    draw_cell(
        axis, x_edges[1], x_edges[2], 0.0, top_height,
        facecolor=edge_header,
    )
    axis.text(
        0.5 * (x_edges[1] + x_edges[2]),
        0.5 * top_height,
        "Edge",
        ha="center",
        va="center",
    )
    draw_cell(
        axis, x_edges[2], x_edges[-1], 0.0, top_height,
        facecolor=node_header,
    )
    axis.text(
        0.5 * (x_edges[2] + x_edges[-1]),
        0.5 * top_height,
        "Node",
        ha="center",
        va="center",
    )

    headers = ("Weighted\nsRMSE ↓", "Overall ↓", "Prog. ↓", "Commit. ↓", "Mature ↓")
    for column, header in enumerate(headers, start=1):
        fill = edge_subheader if column == 1 else node_subheader
        draw_cell(
            axis,
            x_edges[column],
            x_edges[column + 1],
            top_height,
            data_top,
            facecolor=fill,
        )
        axis.text(
            0.5 * (x_edges[column] + x_edges[column + 1]),
            top_height + 0.5 * sub_height,
            header,
            ha="center",
            va="center",
            linespacing=0.95,
        )

    numeric_columns = ("Edge", *NODE_GROUPS)
    minima = {column: float(values[column].min()) for column in numeric_columns}
    for row_index, row in values.iterrows():
        y0 = data_top + row_height * row_index
        y1 = y0 + row_height
        base_fill = "#FFFFFF" if row_index % 2 == 0 else "#FAFAFA"
        draw_cell(
            axis, x_edges[0], x_edges[1], y0, y1, facecolor=base_fill
        )
        swatch = method_style(str(row["method"])).color
        axis.add_patch(
            Rectangle(
                (x_edges[0] + 0.10, y0 + 0.18),
                0.10,
                row_height - 0.36,
                facecolor=swatch,
                edgecolor="none",
            )
        )
        display_method = str(row["method"]).replace("CytoBridge ", "CytoBridge\n")
        axis.text(
            x_edges[0] + 0.27,
            0.5 * (y0 + y1),
            display_method,
            ha="left",
            va="center",
            linespacing=0.85,
        )

        for column_index, column in enumerate(numeric_columns, start=1):
            is_best = np.isclose(float(row[column]), minima[column])
            if is_best:
                fill = best_edge if column == "Edge" else best_node
            else:
                fill = base_fill
            draw_cell(
                axis,
                x_edges[column_index],
                x_edges[column_index + 1],
                y0,
                y1,
                facecolor=fill,
            )
            axis.text(
                0.5 * (x_edges[column_index] + x_edges[column_index + 1]),
                0.5 * (y0 + y1),
                f"{float(row[column]):.3f}",
                ha="center",
                va="center",
                fontweight="bold" if is_best else "normal",
            )

    axis.plot(
        [x_edges[1], x_edges[1]], [0.0, data_bottom],
        color="#707070", linewidth=1.0,
    )
    axis.plot(
        [x_edges[2], x_edges[2]], [0.0, data_bottom],
        color="#707070", linewidth=1.15,
    )
    stem = output_dir / "pancreas_edge_node_atac_summary_table"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            stem.with_suffix(f".{extension}"),
            dpi=600 if extension == "png" else None,
            bbox_inches="tight",
            pad_inches=0.025,
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_png = args.output_dir / "pancreas_edge_node_atac_summary_table.png"
    if output_png.exists() and not args.overwrite:
        raise FileExistsError(f"{output_png} exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    values = load_values(args.edge, args.node)
    values.to_csv(args.output_dir / "pancreas_edge_node_atac_summary_table_values.csv", index=False)
    plot_table(values, args.output_dir)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "edge_input": str(args.edge.resolve()),
                "node_input": str(args.node.resolve()),
                "edge_metric": "cell-number-weighted edge-conditioned standardized RMSE for RNA transition probability >= 1%",
                "node_metric": "observed-E15.5-cell-number-weighted target-celltype centroid standardized RMSE",
                "node_groups": list(NODE_GROUPS),
                "display": "10 pt Arial; 4.10 inch width; Edge and Node color blocks; sliced W2 omitted",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(values.to_string(index=False))


if __name__ == "__main__":
    main()
