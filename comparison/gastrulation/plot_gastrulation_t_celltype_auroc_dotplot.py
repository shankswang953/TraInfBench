#!/usr/bin/env python
"""Plot 5-NN cell-type AUROCs for full and strict-LOTO T maps."""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

from trainfbench_plot_style import NATURE_CUD, apply_nature_rc  # noqa: E402


INPUT = ROOT / "results/gastrulation_t_celltype_auroc/t_celltype_auroc.csv"
OUTPUT_DIR = ROOT / "results/gastrulation_t_celltype_auroc"
STEM = OUTPUT_DIR / "t_celltype_auroc_5nn_dotplot"

MODEL_ORDER = ("ATAC atlas reference", "Full T", "Strict LOTO T")
DISPLAY = {
    "Forebrain/Midbrain/Hindbrain": "Fore-/mid-/hindbrain",
    "Haematoendothelial progenitors": "Haemato-endothelial prog.",
    "Caudal Mesoderm": "Caudal mesoderm",
}


def main() -> None:
    data = pd.read_csv(INPUT)
    strict_order = (
        data[data["model"].eq("Strict LOTO T")]
        .groupby("celltype", observed=True)["one_vs_rest_auroc"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    y_lookup = {celltype: index for index, celltype in enumerate(strict_order)}

    apply_nature_rc()
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
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.20, 7.10),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.265,
        right=0.985,
        top=0.825,
        bottom=0.115,
        wspace=0.08,
    )

    for ax, stage in zip(axes, ("E8.0", "E8.5")):
        local = data[data["stage"].eq(stage)]
        pivot = local.pivot(
            index="celltype",
            columns="model",
            values="one_vs_rest_auroc",
        )
        for celltype, row in pivot.iterrows():
            values = row.reindex(MODEL_ORDER).dropna().to_numpy(float)
            if len(values) >= 2:
                y_value = y_lookup[celltype]
                ax.plot(
                    [values.min(), values.max()],
                    [y_value, y_value],
                    color="#BDBDBD",
                    linewidth=0.7,
                    zorder=1,
                )

        for model, style in (
            (
                "ATAC atlas reference",
                {
                    "marker": "x",
                    "s": 26,
                    "color": NATURE_CUD["black"],
                    "linewidths": 1.0,
                },
            ),
            (
                "Full T",
                {
                    "marker": "o",
                    "s": 34,
                    "facecolors": "white",
                    "edgecolors": NATURE_CUD["sky_blue"],
                    "linewidths": 1.1,
                },
            ),
            (
                "Strict LOTO T",
                {
                    "marker": "D",
                    "s": 29,
                    "facecolors": NATURE_CUD["vermillion"],
                    "edgecolors": NATURE_CUD["vermillion"],
                    "linewidths": 0.8,
                },
            ),
        ):
            model_data = local[local["model"].eq(model)]
            y = [y_lookup[celltype] for celltype in model_data["celltype"]]
            ax.scatter(
                model_data["one_vs_rest_auroc"],
                y,
                zorder=3,
                label=model,
                **style,
            )

        macro = (
            local.groupby("model", observed=True)["one_vs_rest_auroc"]
            .mean()
            .reindex(MODEL_ORDER)
        )
        ax.set_title(
            f"{stage}\nLOTO T macro AUROC = {macro['Strict LOTO T']:.3f}",
            pad=5,
            fontweight="normal",
        )
        ax.set_xlim(0.845, 1.002)
        ax.set_xticks([0.85, 0.90, 0.95, 1.00])
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.6)
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(
        np.arange(len(strict_order)),
        [DISPLAY.get(celltype, celltype) for celltype in strict_order],
    )
    axes[0].set_ylim(len(strict_order) - 0.5, -0.5)
    axes[1].tick_params(labelleft=False)

    fig.suptitle(
        "Cross-fitted RNA-to-ATAC mapping by cell type",
        x=0.625,
        y=0.985,
        fontsize=12,
        fontweight="normal",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    display_labels = [
        "ATAC atlas reference",
        "Full T",
        "Stage-specific LOTO T",
    ]
    fig.legend(
        handles,
        display_labels,
        loc="upper center",
        bbox_to_anchor=(0.625, 0.940),
        ncol=3,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.supxlabel("One-vs-rest AUROC", x=0.625, y=0.045)
    fig.text(
        0.625,
        0.014,
        "LOTO T excludes E8.0 in the left panel and E8.5 in the right panel; distance-weighted 5-NN",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#333333",
    )
    fig.savefig(STEM.with_suffix(".pdf"), facecolor="white")
    fig.savefig(STEM.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)
    print(STEM.with_suffix(".pdf"))
    print(STEM.with_suffix(".png"))


if __name__ == "__main__":
    main()
