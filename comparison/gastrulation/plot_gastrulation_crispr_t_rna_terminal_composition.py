#!/usr/bin/env python
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

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from trainfbench_plot_style import gastrulation_celltype_color


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = ROOT / "results/gastrulation_crispr_t_e85_to_e875"
DEFAULT_INPUT = DEFAULT_RESULT_DIR / "e875_NMP_knn_fate_summary.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_DIR / "rna_terminal_composition_k20"

SYNC_FAMILIES = ("BSOT", "USOT")
BASELINES = (
    "MIOFlow",
    "CytoBridge balanced",
    "CytoBridge unbalanced",
    "TIGON",
    "TrajectoryNet",
)
METHODS = list(SYNC_FAMILIES) + list(BASELINES)
DISPLAY = {
    "BSOT": "COATI bal.",
    "USOT": "COATI unbal.",
    "MIOFlow": "MIOFlow",
    "CytoBridge balanced": "CytoBridge bal.",
    "CytoBridge unbalanced": "CytoBridge unbal.",
    "TIGON": "TIGON",
    "TrajectoryNet": "TrajectoryNet",
}

KO_COLOR = "#2C7FB8"
WT_COLOR = "#E67E22"
FATE_COLORS = {
    "spinal": gastrulation_celltype_color("Spinal cord"),
    "retained_NMP": gastrulation_celltype_color("NMP"),
    "paraxial": gastrulation_celltype_color("Paraxial mesoderm"),
    "somitic": gastrulation_celltype_color("Somitic mesoderm"),
    "mesoderm": "#D55E00",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12,
            "font.weight": "normal",
            "axes.titlesize": 12,
            "axes.titleweight": "normal",
            "axes.labelsize": 12,
            "axes.labelweight": "normal",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 12,
            "figure.titleweight": "normal",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.default": "regular",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.9,
        }
    )


def _draw_metric(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    prefix: str,
    title: str,
    xlim: tuple[float, float],
    show_ylabels: bool,
    show_xlabel: bool,
    show_legend: bool,
) -> None:
    y = np.arange(len(METHODS), dtype=float)
    offsets = {"T_KO": -0.16, "WT": 0.16}
    colors = {"T_KO": KO_COLOR, "WT": WT_COLOR}
    labels = {"T_KO": "T-KO", "WT": "WT"}
    bar_height = 0.22
    label_offset = 0.012 * (xlim[1] - xlim[0])

    for method_index, method in enumerate(METHODS):
        for genotype in ("T_KO", "WT"):
            y_center = y[method_index] + offsets[genotype]
            column = f"{prefix}_{genotype}_pct"
            if method in SYNC_FAMILIES:
                selected = frame[
                    frame["method"].str.startswith(method + " ")
                ].sort_values("method")
                values = selected[column].to_numpy(float)
                mean = float(values.mean())
                sd = float(values.std(ddof=1))
                ax.barh(
                    y_center,
                    mean,
                    height=bar_height,
                    color=colors[genotype],
                    edgecolor="none",
                    alpha=1.0,
                    zorder=2,
                )
                jitter = np.linspace(-0.052, 0.052, len(values))
                ax.scatter(
                    values,
                    np.full_like(values, y_center) + jitter,
                    s=11,
                    facecolor="white",
                    edgecolor=colors[genotype],
                    linewidth=0.7,
                    alpha=0.48,
                    zorder=4,
                )
                ax.errorbar(
                    mean,
                    y_center,
                    xerr=sd,
                    fmt="none",
                    ecolor="#222222",
                    elinewidth=1.2,
                    capsize=2.5,
                    capthick=1.1,
                    zorder=5,
                )
                value = mean
                label_anchor = max(mean + sd, float(values.max()))
                weight = "normal"
            else:
                selected = frame[frame["method"].eq(method)]
                value = float(selected.iloc[0][column])
                ax.barh(
                    y_center,
                    value,
                    height=bar_height,
                    color=colors[genotype],
                    edgecolor="none",
                    alpha=1.0,
                    zorder=2,
                )
                label_anchor = value
                weight = "normal"

            ax.text(
                label_anchor + label_offset,
                y_center,
                f"{value:.1f}",
                ha="left",
                va="center",
                fontweight=weight,
                zorder=6,
            )

    ax.set_xlim(*xlim)
    ax.set_ylim(len(METHODS) - 0.55, -0.55)
    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels(
            [DISPLAY[method] for method in METHODS],
            fontsize=10,
            fontweight="normal",
        )
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.tick_params(axis="x", labelsize=10)
    if show_xlabel:
        ax.set_xlabel("NMP descendants (%)")
    ax.set_title(title, loc="left", fontweight="normal", pad=8)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Patch(facecolor=KO_COLOR, edgecolor="none", label=labels["T_KO"]),
        Patch(facecolor=WT_COLOR, edgecolor="none", label=labels["WT"]),
    ]
    if show_legend:
        ax.legend(
            handles=handles,
            frameon=False,
            ncol=2,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.015),
            borderaxespad=0,
            handlelength=1.3,
            columnspacing=1.0,
        )


def _plot_metric(
    frame: pd.DataFrame,
    *,
    prefix: str,
    title: str,
    xlim: tuple[float, float],
    output_prefix: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(3.8, 3.75))
    _draw_metric(
        ax,
        frame,
        prefix=prefix,
        title=title,
        xlim=xlim,
        show_ylabels=True,
        show_xlabel=True,
        show_legend=False,
    )

    fig.subplots_adjust(left=0.40, right=0.98, bottom=0.15, top=0.85)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_prefix.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )
    fig.savefig(
        output_prefix.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )
    plt.close(fig)


def _plot_combined(
    frame: pd.DataFrame,
    *,
    specs: tuple[tuple[str, str, tuple[float, float], str], ...],
    readout_label: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.65), sharey=True)
    for index, (ax, (prefix, title, xlim, _)) in enumerate(zip(axes, specs)):
        _draw_metric(
            ax,
            frame,
            prefix=prefix,
            title=title,
            xlim=xlim,
            show_ylabels=index == 0,
            show_xlabel=False,
            show_legend=False,
        )
    handles = [
        Patch(facecolor=KO_COLOR, edgecolor="none", label="T-KO"),
        Patch(facecolor=WT_COLOR, edgecolor="none", label="WT"),
    ]
    fig.text(
        0.22,
        0.965,
        f"{readout_label} readout",
        ha="left",
        va="top",
        fontweight="normal",
    )
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        handlelength=1.3,
        columnspacing=1.0,
    )
    fig.supxlabel("Predicted NMP descendants (%)", x=0.60, y=0.02)
    fig.subplots_adjust(
        left=0.22,
        right=0.995,
        bottom=0.16,
        top=0.84,
        wspace=0.34,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )
    fig.savefig(
        output_path.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )
    plt.close(fig)


def _draw_genotype_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    genotype: str,
    show_ylabels: bool,
) -> None:
    y = np.arange(len(METHODS), dtype=float)
    fates = (
        ("spinal", "Spinal cord"),
        ("retained_NMP", "Retained NMP"),
        ("mesoderm", "Mesoderm"),
    )
    # Keep the three fate readouts distinct in the vertically compressed panel.
    offsets = (-0.27, 0.0, 0.27)
    bar_height = 0.16

    for fate_index, ((prefix, _), offset) in enumerate(zip(fates, offsets)):
        column = f"{prefix}_{genotype}_pct"
        color = FATE_COLORS[prefix]
        for method_index, method in enumerate(METHODS):
            y_center = y[method_index] + offset
            if method in SYNC_FAMILIES:
                selected = frame[
                    frame["method"].str.startswith(method + " ")
                ].sort_values("method")
                values = selected[column].to_numpy(float)
                value = float(values.mean())
                sd = float(values.std(ddof=1))
                ax.barh(
                    y_center,
                    value,
                    height=bar_height,
                    color=color,
                    edgecolor="none",
                    zorder=2,
                )
                jitter = np.linspace(-0.022, 0.022, len(values))
                ax.scatter(
                    values,
                    np.full_like(values, y_center) + jitter,
                    s=8,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.55,
                    alpha=0.55,
                    zorder=4,
                )
                ax.errorbar(
                    value,
                    y_center,
                    xerr=sd,
                    fmt="none",
                    ecolor="#333333",
                    elinewidth=0.8,
                    capsize=1.8,
                    capthick=0.8,
                    zorder=5,
                )
                label_anchor = max(value + sd, float(values.max()))
            else:
                selected = frame[frame["method"].eq(method)]
                value = float(selected.iloc[0][column])
                ax.barh(
                    y_center,
                    value,
                    height=bar_height,
                    color=color,
                    edgecolor="none",
                    zorder=2,
                )
                label_anchor = value

            # Place every value consistently just beyond the right-hand end of
            # its bar (and beyond the error bar for COATI parameter sweeps).
            text_x = max(label_anchor + 1.0, 1.0)
            ax.text(
                text_x,
                y_center,
                f"{value:.1f}",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="normal",
                clip_on=False,
                zorder=6,
            )

    ax.set_xlim(0.0, 100.0)
    ax.set_xticks((0.0, 50.0, 100.0))
    ax.set_ylim(len(METHODS) - 0.55, -0.55)
    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels(
            [DISPLAY[method] for method in METHODS],
            fontsize=10,
            fontweight="normal",
        )
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_title(
        "T-KO" if genotype == "T_KO" else "WT",
        loc="left",
        fontweight="normal",
        pad=3,
    )
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_by_genotype(
    frame: pd.DataFrame,
    *,
    output_prefix: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(4.05, 3.65), sharex=True, sharey=True)
    _draw_genotype_panel(
        axes[0],
        frame,
        genotype="T_KO",
        show_ylabels=True,
    )
    _draw_genotype_panel(
        axes[1],
        frame,
        genotype="WT",
        show_ylabels=False,
    )
    # The right panel shares the same 0–100% scale; omit its zero tick label
    # so the two compact panels do not create a visually merged "1000".
    axes[1].set_xticks((50.0, 100.0))
    handles = [
        Patch(
            facecolor=FATE_COLORS["spinal"],
            edgecolor="none",
            label="Spinal cord",
        ),
        Patch(
            facecolor=FATE_COLORS["retained_NMP"],
            edgecolor="none",
            label="Retained NMP",
        ),
        Patch(
            facecolor=FATE_COLORS["mesoderm"],
            edgecolor="none",
            label="Mesoderm",
        ),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.625, 0.995),
        handlelength=1.0,
        columnspacing=0.75,
        handletextpad=0.45,
        prop={"family": "Arial", "size": 10, "weight": "normal"},
    )
    fig.supxlabel(
        "Predicted NMP descendants (%)",
        x=0.61,
        y=0.005,
        fontsize=12,
        fontweight="normal",
    )
    fig.subplots_adjust(
        left=0.34,
        right=0.995,
        bottom=0.14,
        top=0.78,
        wspace=0.12,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_prefix.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.01,
        facecolor="white",
    )
    fig.savefig(
        output_prefix.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.01,
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create terminal-composition plots for one readout comparing "
            "CRISPR T-KO with WT for each trajectory method."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--readout", choices=("RNA", "ATAC"), default="RNA")
    args = parser.parse_args()

    _style()
    frame = pd.read_csv(args.input)
    frame = frame[
        frame["k"].eq(args.k) & frame["readout"].eq(args.readout)
    ].copy()
    if frame.empty:
        raise ValueError(
            f"No {args.readout} rows found for k={args.k} in {args.input}"
        )
    readout_stem = args.readout.lower()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    values_path = (
        args.output_dir / f"{readout_stem}_wt_vs_tko_values_k{args.k}.csv"
    )
    frame.to_csv(values_path, index=False)
    print(f"wrote: {values_path}")

    specs = (
        ("spinal", "Spinal cord", (0.0, 40.0), "spinal_cord"),
        ("retained_NMP", "Retained NMP", (0.0, 100.0), "retained_nmp"),
        ("mesoderm", "Mesoderm", (0.0, 3.7), "mesoderm"),
    )
    for prefix, title, xlim, stem in specs:
        output_prefix = (
            args.output_dir / f"{readout_stem}_terminal_{stem}_k{args.k}"
        )
        _plot_metric(
            frame,
            prefix=prefix,
            title=f"{args.readout} · {title}",
            xlim=xlim,
            output_prefix=output_prefix,
        )
        print(f"wrote: {output_prefix.with_suffix('.png')}")
        print(f"wrote: {output_prefix.with_suffix('.pdf')}")
    combined_path = (
        args.output_dir / f"{readout_stem}_terminal_composition_k{args.k}.png"
    )
    _plot_combined(
        frame,
        specs=specs,
        readout_label=args.readout,
        output_path=combined_path,
    )
    print(f"wrote: {combined_path}")
    print(f"wrote: {combined_path.with_suffix('.pdf')}")

    genotype_prefix = (
        args.output_dir
        / f"{readout_stem}_terminal_composition_by_genotype_k{args.k}"
    )
    _plot_by_genotype(frame, output_prefix=genotype_prefix)
    print(f"wrote: {genotype_prefix.with_suffix('.png')}")
    print(f"wrote: {genotype_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
