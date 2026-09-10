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

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trainfbench_plot_style import gastrulation_celltype_color


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path(
    "external/COATI/Gastrulation/data/"
    "gastrulation_rna_processed.h5ad"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/gastrulation_crispr_t_e85_to_e875_gaga10_tigon_ae10"
    / "observed_wt_e85_e875_nmp_lineage"
)

STAGES = ("E8.5", "E8.75")
LINEAGE_CELLTYPES = (
    "Caudal epiblast",
    "Caudal neurectoderm",
    "NMP",
    "Spinal cord",
    "Caudal Mesoderm",
    "Paraxial mesoderm",
    "Somitic mesoderm",
)
DISPLAY_GROUPS = (
    "NMP",
    "Spinal cord",
    "Paraxial mesoderm",
    "Somitic mesoderm",
    "Other posterior",
)
COLORS = {
    "NMP": gastrulation_celltype_color("NMP"),
    "Spinal cord": gastrulation_celltype_color("Spinal cord"),
    "Paraxial mesoderm": gastrulation_celltype_color("Paraxial mesoderm"),
    "Somitic mesoderm": gastrulation_celltype_color("Somitic mesoderm"),
    "Other posterior": "#B3B3B3",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "font.weight": "normal",
            "axes.titlesize": 12,
            "axes.titleweight": "normal",
            "axes.labelsize": 12,
            "axes.labelweight": "normal",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def _summaries(adata: ad.AnnData) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = adata.obs[["stage", "sample_name", "celltype"]].copy()
    obs["stage"] = obs["stage"].astype(str)
    obs["sample_name"] = obs["sample_name"].astype(str)
    obs["celltype"] = obs["celltype"].astype(str)
    obs = obs[
        obs["stage"].isin(STAGES) & obs["celltype"].isin(LINEAGE_CELLTYPES)
    ].copy()
    obs["display_group"] = obs["celltype"].where(
        obs["celltype"].isin(DISPLAY_GROUPS[:-1]), "Other posterior"
    )

    pooled = (
        obs.groupby(["stage", "display_group"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    pooled["lineage_total"] = pooled.groupby("stage", observed=True)[
        "n_cells"
    ].transform("sum")
    pooled["within_lineage_percent"] = (
        100.0 * pooled["n_cells"] / pooled["lineage_total"]
    )

    replicate = (
        obs.groupby(["stage", "sample_name", "display_group"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    replicate["lineage_total"] = replicate.groupby(
        ["stage", "sample_name"], observed=True
    )["n_cells"].transform("sum")
    replicate["within_lineage_percent"] = (
        100.0 * replicate["n_cells"] / replicate["lineage_total"]
    )
    return pooled, replicate


def _plot(pooled: pd.DataFrame, output_prefix: Path) -> None:
    matrix = (
        pooled.pivot(
            index="stage",
            columns="display_group",
            values="within_lineage_percent",
        )
        .reindex(index=STAGES, columns=DISPLAY_GROUPS)
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(4.02, 1.82))
    y = np.array([1.0, 0.0])
    left = np.zeros(len(STAGES), dtype=float)
    for group in DISPLAY_GROUPS:
        values = matrix[group].to_numpy(float)
        ax.barh(
            y,
            values,
            left=left,
            height=0.48,
            color=COLORS[group],
            edgecolor="white",
            linewidth=0.6,
        )
        for row, (start, value) in enumerate(zip(left, values)):
            if value < 4.0:
                continue
            ax.text(
                start + value / 2.0,
                y[row],
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="normal",
                color="#111111",
            )
        left += values

    ax.set_xlim(0.0, 100.0)
    ax.set_xticks((0.0, 50.0, 100.0))
    ax.set_yticks(y, STAGES)
    ax.set_xlabel("Observed NMP-lineage composition (%)", labelpad=5)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.42, 1.42)

    fig.subplots_adjust(left=0.13, right=0.995, top=0.98, bottom=0.28)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_prefix.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    fig.savefig(
        output_prefix.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot ordinary-WT observed NMP-lineage composition at E8.5 and E8.75."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    _style()
    adata = ad.read_h5ad(args.input, backed="r")
    pooled, replicate = _summaries(adata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(args.output_dir / "observed_composition_pooled.csv", index=False)
    replicate.to_csv(
        args.output_dir / "observed_composition_by_replicate.csv", index=False
    )
    _plot(pooled, args.output_dir / "observed_nmp_lineage_e85_e875")


if __name__ == "__main__":
    main()
