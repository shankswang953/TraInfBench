#!/usr/bin/env python3
"""Redraw the selected human-cerebral panels in a compact 10-pt style."""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from plot_human_cerebral_source_line_w2_grouped_dots import prepare as prepare_source
from trainfbench_plot_style import apply_nature_rc, method_style


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/human_cerebral_publication_redraw_10pt"
SOURCE_INPUT = ROOT / "results/human_cerebral_full_biological_interpretability/09_source_line_group_w2/source_line_time_scores.csv"
RNA_INPUT = ROOT / "results/human_cerebral_full_biological_interpretability/10_raw_cell_fate_recovery/numeric_tables_by_time/appendix_all_methods_rna_w2_by_time.csv"
ORIGINAL_ATAC_INPUT = ROOT / "results/human_cerebral_full_biological_interpretability/13_original_atac_w2_current/common_T_all_methods_original_cell_uniform_by_time.csv"
ORIGINAL_ATAC_REFERENCE = ROOT / "results/human_cerebral_full_biological_interpretability/13_original_atac_w2_current/observed_metacell_to_original_atac_by_time.csv"
METACELL_ATAC_INPUT = ROOT / "results/human_cerebral_full_biological_interpretability/14_metacell_atac_w2_all7time/method_summary_by_time.csv"
GLI3_INPUT = ROOT / "results/human_cerebral_gli3_8peak_source_line_main/gli3_8peak_source_line_numeric_values_d9_d21.csv"

METHODS = (
    ("COATI bal.", "COATI bal. (Sync)", "COATI balanced"),
    ("COATI unbal.", "COATI unbal. (Sync)", "COATI unbalanced"),
    ("CytoBridge bal.", "CytoBridge bal.", "CytoBridge balanced"),
    ("CytoBridge unbal.", "CytoBridge unbal.", "CytoBridge unbalanced"),
    ("MIOFlow", "MIOFlow", "MIOFlow"),
    ("TrajectoryNet", "TrajectoryNet", "TrajectoryNet"),
    ("OT(RNA)", "RNA-only bal.", "Balanced RNA-only"),
    ("UOT(RNA)", "RNA-only unbal.", "Unbalanced RNA-only"),
)
DISPLAY_ORDER = tuple(row[1] for row in METHODS)
SOURCE_ORDER = ("409b2", "h9", "hoik1", "wibj2")
SOURCE_LABELS = ("409B2\niPSC", "H9\nESC", "HOIK1\niPSC", "WIBJ2\niPSC")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_style() -> None:
    apply_nature_rc(font_size=10.0)
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 10.0,
        "font.weight": "normal",
        "axes.titlesize": 10.0,
        "axes.titleweight": "normal",
        "axes.labelsize": 10.0,
        "axes.labelweight": "normal",
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.0,
        "legend.title_fontsize": 10.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def visual(style_name: str) -> tuple[str, str, str]:
    style = method_style(style_name)
    face = style.color if style.markerfacecolor is None else style.markerfacecolor
    edge = style.color if style.markeredgecolor is None else style.markeredgecolor
    if style_name == "TrajectoryNet":
        edge = face  # explicitly remove the legacy black outline
    return style.marker, face, edge


def legend_handle(label: str, style_name: str) -> Line2D:
    marker, face, edge = visual(style_name)
    return Line2D([], [], linestyle="none", marker=marker, markersize=5.8,
                  markerfacecolor=face, markeredgecolor=edge,
                  markeredgewidth=0.75, label=label.replace(" (Sync)", ""))


def finish_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#E3E3E3", linewidth=0.55, zorder=0)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#666666")
        spine.set_linewidth(0.75)


def save(fig: plt.Figure, output: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("pdf", "png", "svg"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(path, dpi=600 if suffix == "png" else None,
                    bbox_inches="tight", pad_inches=0.015, facecolor="white")
        paths.append(str(path.resolve()))
    plt.close(fig)
    return paths


def plot_source(data: pd.DataFrame, modality: str, output: Path) -> list[str]:
    setup_style()
    local = data.loc[data.modality.eq(modality)]
    fig, ax = plt.subplots(figsize=(4.05, 3.45))
    centers = np.arange(4, dtype=float)
    offsets = np.linspace(-0.245, 0.245, 8)
    handles = []
    for index, (old, display, style_name) in enumerate(METHODS):
        values = (local.loc[local.method.eq(old)].set_index("source_line")
                  .reindex(SOURCE_ORDER).sum_w2.to_numpy(float))
        marker, face, edge = visual(style_name)
        ax.scatter(centers + offsets[index], values, s=38,
                   marker=marker, facecolors=face, edgecolors=edge,
                   linewidths=0.7, zorder=3)
        handles.append(legend_handle(display, style_name))
    lo, hi = float(local.sum_w2.min()), float(local.sum_w2.max())
    pad = max(0.045, 0.07 * (hi - lo))
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(max(0, lo - pad), hi + pad)
    ax.set_xticks(centers, SOURCE_LABELS)
    ax.set_xlabel("Source cell line", labelpad=2)
    ax.set_ylabel(r"Summed $W_2$ (six times) $\downarrow$")
    ax.set_title(modality, pad=3, fontweight="normal")
    finish_axis(ax)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=2, frameon=False, handletextpad=0.35, columnspacing=0.9,
               labelspacing=0.25)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.94, bottom=0.36)
    return save(fig, output, f"source_line_{modality.lower()}_w2")


def plot_time(table_path: Path, times: tuple[str, ...], ylabel: str,
              output: Path, stem: str, reference: pd.Series | None = None) -> list[str]:
    setup_style()
    table = pd.read_csv(table_path).set_index("Method").reindex(DISPLAY_ORDER)
    if table.loc[:, list(times)].isna().any().any():
        raise ValueError(f"Missing values in {table_path}")
    fig, ax = plt.subplots(figsize=(4.05, 3.75 if reference is not None else 3.55))
    centers = np.arange(len(times), dtype=float)
    offsets = np.linspace(-0.23, 0.18 if reference is not None else 0.23, 8)
    handles = []
    for index, (_, display, style_name) in enumerate(METHODS):
        marker, face, edge = visual(style_name)
        ax.scatter(centers + offsets[index], table.loc[display, list(times)],
                   s=39, marker=marker, facecolors=face, edgecolors=edge,
                   linewidths=0.7, zorder=3)
        handles.append(legend_handle(display, style_name))
    values = table.loc[:, list(times)].to_numpy(float).ravel()
    if reference is not None:
        ref = reference.reindex(times).to_numpy(float)
        ax.scatter(centers + 0.245, ref, s=34, marker="o", facecolors="none",
                   edgecolors="#777777", linewidths=0.9, zorder=4)
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=5.5,
                              markerfacecolor="none", markeredgecolor="#777777",
                              markeredgewidth=0.9, label="Observed metacells"))
        values = np.concatenate([values, ref])
    lo, hi = float(values.min()), float(values.max())
    pad = max(1e-4, 0.07 * (hi - lo))
    ax.set_ylim(max(0, lo - pad), hi + pad)
    ax.set_xlim(-0.48, len(times) - 0.52)
    ax.set_xticks(centers, times)
    ax.set_xlabel("Time", labelpad=2)
    ax.set_ylabel(ylabel)
    finish_axis(ax)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=2, frameon=False, handletextpad=0.3,
               columnspacing=0.7, labelspacing=0.25)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.98,
                        bottom=0.36 if reference is None else 0.41)
    return save(fig, output, stem)


def plot_gli3(output: Path) -> list[str]:
    setup_style()
    data = pd.read_csv(GLI3_INPUT)
    source_names = ("409b2", "H9", "Hoik1", "Wibj2", "Mean")
    labels = ("409B2\niPSC", "H9\nESC", "HOIK1\niPSC", "WIBJ2\niPSC", "Mean")
    model_order = (data[["model_id", "display_method", "style_name", "method_order"]]
                   .drop_duplicates().sort_values("method_order"))
    matrix = data.pivot(index="model_id", columns="source_line",
                        values="three_gene_equal_coupling").reindex(
                            index=model_order.model_id, columns=source_names)
    if matrix.isna().any().any():
        raise ValueError("Incomplete GLI3 table")

    fig = plt.figure(figsize=(3.80, 2.65), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    left, right, bottom, top = 0.006, 0.994, 0.015, 0.985
    label_right, header_bottom = 0.326, 0.75
    col_width = (right - label_right) / 5
    centers = [label_right + (i + 0.5) * col_width for i in range(5)]
    ax.text(left + 0.012, 0.855, "Method", ha="left", va="center",
            fontsize=10, fontweight="normal")
    ax.text(label_right + 2 * col_width, 0.945, "Source cell line",
            ha="center", va="center", fontsize=10, fontweight="normal")
    for x, label in zip(centers, labels):
        ax.text(x, 0.845, label, ha="center", va="center", fontsize=10,
                fontweight="normal", linespacing=0.95)
    ax.plot([left, right], [header_bottom, header_bottom], color="#777777", lw=0.7)
    row_height = (header_bottom - bottom) / 8
    lookup = data.set_index(["model_id", "source_line"])
    for i, row in enumerate(model_order.itertuples(index=False)):
        yc = header_bottom - (i + 0.5) * row_height
        if i % 2 == 0:
            ax.add_patch(Rectangle((left, yc - row_height / 2), right - left,
                                   row_height, facecolor="#F5F5F5", edgecolor="none"))
        _, face, edge = visual(row.style_name)
        ax.add_patch(Rectangle((left + 0.007, yc - 0.012), 0.019, 0.024,
                               facecolor=face, edgecolor=edge, linewidth=0.55))
        ax.text(left + 0.028, yc, row.display_method, ha="left", va="center",
                fontsize=10, fontweight="normal")
        for x, source in zip(centers, source_names):
            value = lookup.loc[(row.model_id, source), "three_gene_equal_coupling"]
            ax.text(x, yc, f"{value:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="normal")
    ax.plot([label_right, label_right], [bottom, top], color="#B5B5B5", lw=0.55)
    for i in range(1, 5):
        x = label_right + i * col_width
        ax.plot([x, x], [bottom, header_bottom], color="#DEDEDE", lw=0.45)
    ax.add_patch(Rectangle((left, bottom), right - left, top - bottom,
                           facecolor="none", edgecolor="#666666", linewidth=0.8,
                           clip_on=False))
    return save(fig, output, "gli3_8peak_source_line_numeric_table")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    expected = [
        "source_line_rna_w2.pdf", "source_line_atac_w2.pdf",
        "rna_original_cells_w2_by_time.pdf", "original_atac_w2_by_time.pdf",
        "metacell_atac_w2_all7time.pdf", "gli3_8peak_source_line_numeric_table.pdf",
    ]
    existing = [output / name for name in expected if (output / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to overwrite; pass --overwrite: " + str(existing))
    output.mkdir(parents=True, exist_ok=True)

    outputs = []
    source = prepare_source(SOURCE_INPUT)
    outputs += plot_source(source, "RNA", output)
    outputs += plot_source(source, "ATAC", output)
    outputs += plot_time(RNA_INPUT, ("D7", "D9", "D11", "D12", "D18", "D21"),
                         r"RNA sliced $W_2$ to original cells $\downarrow$", output,
                         "rna_original_cells_w2_by_time")
    ref = pd.read_csv(ORIGINAL_ATAC_REFERENCE, index_col=0).iloc[:, 0]
    outputs += plot_time(ORIGINAL_ATAC_INPUT, ("D7", "D9", "D11", "D21"),
                         r"ATAC sliced $W_2$ to original cells $\downarrow$", output,
                         "original_atac_w2_by_time", reference=ref)
    outputs += plot_time(METACELL_ATAC_INPUT,
                         ("D4", "D7", "D9", "D11", "D12", "D18", "D21"),
                         r"ATAC sliced $W_2$ to metacells $\downarrow$", output,
                         "metacell_atac_w2_all7time")
    outputs += plot_gli3(output)

    inputs = [SOURCE_INPUT, RNA_INPUT, ORIGINAL_ATAC_INPUT, ORIGINAL_ATAC_REFERENCE,
              METACELL_ATAC_INPUT, GLI3_INPUT]
    manifest = {
        "purpose": "Compact publication redraw of the six user-selected human-cerebral panels",
        "font": "Arial 10 pt throughout; normal weight only",
        "maximum_width_inches": 4.05,
        "trajectorynet_outline": "No black outline; marker edge equals yellow face",
        "spacing": "tight bounding box, 0.015 inch pad",
        "inputs": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in inputs],
        "outputs": outputs,
        "source_scripts": [
            str((ROOT / "comparison/human_cerebral/plot_human_cerebral_source_line_w2_grouped_dots.py").resolve()),
            str((ROOT / "comparison/human_cerebral/plot_human_cerebral_raw_cell_rna_w2_all_methods_by_time.py").resolve()),
            str((ROOT / "comparison/human_cerebral/analyze_human_cerebral_original_atac_w2_current.py").resolve()),
            str((ROOT / "comparison/human_cerebral/analyze_human_cerebral_common_t_atac_recovery_by_time.py").resolve()),
            str((ROOT / "comparison/human_cerebral/plot_human_cerebral_gli3_8peak_source_line_numeric_table.py").resolve()),
        ],
        "note": "Values were not recomputed; only the presentation was redrawn.",
    }
    (output / "redraw_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
