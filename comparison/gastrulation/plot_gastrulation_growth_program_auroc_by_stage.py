#!/usr/bin/env python
"""Plot stage-resolved growth-program rank AUROC for all unbalanced methods."""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from trainfbench_plot_style import (
    METHOD_STYLES,
    apply_nature_rc,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "results/gastrulation_genomewide_growth_programs_by_setting/"
    "rank_enrichment_by_setting_stage.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/gastrulation_genomewide_growth_programs_by_setting/figures"
)
DEFAULT_PAPER_OUTPUT = ROOT / "output/pdf/gastrulation_growth_programs"

STAGES = ["E7.5", "E8.0", "E8.5", "E8.75"]
STAGE_POSITIONS = np.array([0.0, 1.0, 2.0, 2.5], dtype=float)
PANELS = [
    ("S phase", "S phase"),
    (
        "Lineage maturation (non-cycle)",
        "Forward lineage\nprogression",
    ),
]
BASELINES = [
    "CytoBridge unbalanced",
    "TIGON",
]
METHOD_PLOT_STYLES = {
    "USOT Sync": METHOD_STYLES["COATI unbalanced"],
    "CytoBridge unbalanced": METHOD_STYLES["CytoBridge unbalanced"],
    "TIGON": METHOD_STYLES["TIGON"],
}
DISPLAY_NAMES = {
    "USOT Sync": "COATI unbal.",
    "CytoBridge unbalanced": "CytoBridge unbal.",
    "TIGON": "TIGON",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--paper-output-dir", type=Path, default=DEFAULT_PAPER_OUTPUT
    )
    parser.add_argument("--c-y", type=float, default=0.3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ordered_values(
    frame: pd.DataFrame,
    method: str,
    gene_set: str,
) -> np.ndarray:
    local = frame[
        frame["method"].eq(method) & frame["gene_set"].eq(gene_set)
    ].copy()
    local["stage"] = pd.Categorical(
        local["stage"], categories=STAGES, ordered=True
    )
    local = local.sort_values("stage")
    if local["stage"].astype(str).tolist() != STAGES:
        raise ValueError(f"Missing or duplicated stages for {method}, {gene_set}")
    return local["rank_auroc_equal_celltype_mean"].to_numpy(float)


def sync_values(
    frame: pd.DataFrame,
    gene_set: str,
    c_y: float,
) -> np.ndarray:
    local = frame[
        frame["method"].eq("USOT Sync")
        & frame["gene_set"].eq(gene_set)
        & np.isclose(frame["C_y"].to_numpy(float), c_y)
    ].copy()
    local["stage"] = pd.Categorical(
        local["stage"], categories=STAGES, ordered=True
    )
    local = local.sort_values("stage")
    if local["stage"].astype(str).tolist() != STAGES:
        raise ValueError(
            f"Missing or duplicated stages for USOT Sync C_y={c_y:g}, "
            f"{gene_set}"
        )
    return local["rank_auroc_equal_celltype_mean"].to_numpy(float)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.paper_output_dir.mkdir(parents=True, exist_ok=True)
    c_y_tag = f"{args.c_y:.1f}".replace(".", "")
    suffix = f"publication_cy{c_y_tag}"
    png = args.output_dir / f"growth_program_rank_auroc_by_stage_{suffix}.png"
    pdf = args.output_dir / f"growth_program_rank_auroc_by_stage_{suffix}.pdf"
    s_phase_pdf = (
        args.output_dir / f"s_phase_rank_auroc_by_stage_{suffix}.pdf"
    )
    progression_pdf = (
        args.output_dir
        / f"forward_lineage_progression_rank_auroc_by_stage_{suffix}.pdf"
    )
    values_csv = (
        args.output_dir
        / f"growth_program_rank_auroc_plot_values_{suffix}.csv"
    )
    paper_pdf = args.paper_output_dir / pdf.name
    paper_s_phase_pdf = args.paper_output_dir / s_phase_pdf.name
    paper_progression_pdf = args.paper_output_dir / progression_pdf.name
    targets = (
        png,
        pdf,
        s_phase_pdf,
        progression_pdf,
        values_csv,
        paper_pdf,
        paper_s_phase_pdf,
        paper_progression_pdf,
    )
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")

    frame = pd.read_csv(args.input)
    frame = frame[
        frame["gene_set"].isin([gene_set for gene_set, _ in PANELS])
    ].copy()

    apply_nature_rc(font_size=10)
    mpl.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(4.1, 2.8),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    x = STAGE_POSITIONS
    exported: list[dict[str, object]] = []

    for ax, (gene_set, title) in zip(axes, PANELS):
        sync = sync_values(frame, gene_set, args.c_y)
        sync_style = METHOD_PLOT_STYLES["USOT Sync"]
        ax.plot(
            x,
            sync,
            color=sync_style.color,
            marker=sync_style.marker,
            linestyle=sync_style.linestyle,
            linewidth=sync_style.linewidth,
            markersize=sync_style.markersize,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=4,
        )
        for stage, value in zip(STAGES, sync):
            exported.append(
                {
                    "gene_set": gene_set,
                    "stage": stage,
                    "method": DISPLAY_NAMES["USOT Sync"],
                    "C_y": args.c_y,
                    "summary": "selected C_y setting",
                    "value": value,
                }
            )

        for method in BASELINES:
            values = ordered_values(frame, method, gene_set)
            style = METHOD_PLOT_STYLES[method]
            ax.plot(
                x,
                values,
                color=style.color,
                marker=style.marker,
                linestyle=style.linestyle,
                linewidth=style.linewidth,
                markersize=style.markersize,
                markerfacecolor=style.color,
                markeredgecolor="white",
                markeredgewidth=0.7,
                zorder=3,
            )
            for stage, value in zip(STAGES, values):
                exported.append(
                    {
                        "gene_set": gene_set,
                        "stage": stage,
                        "method": DISPLAY_NAMES[method],
                        "C_y": np.nan,
                        "summary": "native setting",
                        "value": value,
                    }
                )

        ax.axhline(
            0.5,
            color="#3F3F3F",
            linestyle=(0, (3, 2)),
            linewidth=1.0,
            zorder=0,
        )
        ax.set_title(title, fontsize=12, fontweight="normal", pad=5)
        ax.set_xticks(x, STAGES)
        ax.set_xlim(x[0] - 0.08, x[-1] + 0.08)
        tick_labels = ax.get_xticklabels()
        for tick_label in tick_labels:
            tick_label.set_rotation(0)
            tick_label.set_horizontalalignment("center")
        tick_labels[-2].set_horizontalalignment("center")
        tick_labels[-1].set_horizontalalignment("left")
        ax.set_ylim(0.18, 0.86)
        ax.set_yticks([0.2, 0.4, 0.5, 0.6, 0.8])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", width=0.8, length=3)

    axes[0].set_ylabel("Gene-set rank AUROC")

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_PLOT_STYLES["USOT Sync"].color,
            marker=METHOD_PLOT_STYLES["USOT Sync"].marker,
            linewidth=METHOD_PLOT_STYLES["USOT Sync"].linewidth,
            markersize=METHOD_PLOT_STYLES["USOT Sync"].markersize,
            label=rf"{DISPLAY_NAMES['USOT Sync']} ($C_y={args.c_y:g}$)",
        ),
    ]
    for method in BASELINES:
        style = METHOD_PLOT_STYLES[method]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style.color,
                marker=style.marker,
                linestyle=style.linestyle,
                markerfacecolor=style.color,
                markeredgecolor="white",
                linewidth=style.linewidth,
                markersize=style.markersize,
                label=DISPLAY_NAMES[method],
            )
        )
    handles.append(
        Line2D(
            [0],
            [0],
            color="#3F3F3F",
            linestyle=(0, (3, 2)),
            linewidth=1,
            label="Random (0.5)",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.030),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.5,
        prop={"family": "Arial", "size": 10},
    )
    fig.subplots_adjust(
        left=0.17,
        right=0.94,
        top=0.91,
        bottom=0.35,
        wspace=0.48,
    )

    pd.DataFrame(exported).to_csv(values_csv, index=False)
    fig.savefig(
        png,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.01,
        facecolor="white",
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.01,
        facecolor="white",
    )

    for axis, path in zip(axes, (s_phase_pdf, progression_pdf)):
        extent = axis.get_tightbbox(fig.canvas.get_renderer()).transformed(
            fig.dpi_scale_trans.inverted()
        )
        extent = extent.expanded(1.04, 1.10)
        fig.savefig(path, bbox_inches=extent, pad_inches=0.02)

    plt.close(fig)
    shutil.copyfile(pdf, paper_pdf)
    shutil.copyfile(s_phase_pdf, paper_s_phase_pdf)
    shutil.copyfile(progression_pdf, paper_progression_pdf)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Wrote {s_phase_pdf}")
    print(f"Wrote {progression_pdf}")
    print(f"Wrote {paper_pdf}")
    print(f"Wrote {paper_s_phase_pdf}")
    print(f"Wrote {paper_progression_pdf}")
    print(f"Wrote {values_csv}")


if __name__ == "__main__":
    main()
