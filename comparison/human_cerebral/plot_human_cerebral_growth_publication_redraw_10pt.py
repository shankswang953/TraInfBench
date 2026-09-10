#!/usr/bin/env python3
"""Redraw the two human-cerebral growth panels in the compact paper style."""

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
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/human_cerebral_publication_redraw_10pt"
BREADTH_INPUT = (
    ROOT
    / "results/human_cerebral_growth_breadth_cy0p5_v1/02_cross_time"
    / "eligible_panel_rankings.csv"
)
HEATMAP_INPUT = (
    ROOT
    / "results/human_cerebral_full_biological_interpretability/03_mass_growth"
    / "growth_gene_summary_across_cy.csv.gz"
)

MODEL_ORDER = (
    "coati_sync_unbalanced_cy0.5_s0_i40000",
    "coati_rna_only_unbalanced_s0_i30000",
    "cytobridge_unbalanced_s42_i30000",
)
MODEL_LABEL = {
    MODEL_ORDER[0]: "COATI unbal.",
    MODEL_ORDER[1]: "UOT(RNA)",
    MODEL_ORDER[2]: "CytoBridge unbal.",
}
MODEL_COLOR = {
    MODEL_ORDER[0]: "#0072B2",
    MODEL_ORDER[1]: "#000000",
    MODEL_ORDER[2]: "#CC79A7",
}
MODEL_MARKER = {
    MODEL_ORDER[0]: "o",
    MODEL_ORDER[1]: "s",
    MODEL_ORDER[2]: "v",
}
INTERVALS = ("D7→D9", "D9→D11", "D11→D12", "D12→D18", "D18→D21")
HEAT_INTERVALS = ("D4→D7",) + INTERVALS
MODULES = (
    "Apoptosis/stress",
    "Dorsal",
    "Early neural ectoderm",
    "Excitatory maturation",
    "G2/M",
    "IPC/neurogenesis",
    "Inhibitory maturation",
    "Non-telencephalic",
    "Proliferation",
    "Radial glia/progenitor",
    "S phase",
    "Telencephalic",
    "Transient pre-DV",
    "Ventral",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_style() -> None:
    mpl.rcParams.update(
        {
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
        }
    )


def box_axis(axis: plt.Axes) -> None:
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#666666")
        spine.set_linewidth(0.75)


def save(fig: plt.Figure, output: Path, stem: str, *, tight: bool = False) -> list[str]:
    paths: list[str] = []
    for suffix in ("pdf", "png", "svg"):
        path = output / f"{stem}.{suffix}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.01} if tight else {}
        fig.savefig(
            path,
            dpi=600 if suffix == "png" else None,
            facecolor="white",
            **kwargs,
        )
        paths.append(str(path.resolve()))
    plt.close(fig)
    return paths


def plot_cross_time(output: Path) -> list[str]:
    setup_style()
    data = pd.read_csv(BREADTH_INPUT)
    data = data[
        data.panel.eq("eligible_lineage")
        & data.line.ne("ALL")
        & data.interval.isin(INTERVALS)
        & data.model_id.isin(MODEL_ORDER)
    ].copy()
    if data.groupby(["model_id", "interval"]).size().min() != 4:
        raise ValueError("Expected four source-cell-line values per method and interval")

    fig, axis = plt.subplots(figsize=(4.05, 3.15))
    x = np.arange(len(INTERVALS), dtype=float)
    offsets = (-0.19, 0.0, 0.19)
    for offset, model_id in zip(offsets, MODEL_ORDER):
        for xpos, interval in enumerate(INTERVALS):
            values = data.loc[
                data.model_id.eq(model_id) & data.interval.eq(interval), "auroc_abs"
            ].to_numpy(float)
            mean = float(values.mean())
            axis.errorbar(
                xpos + offset,
                mean,
                yerr=np.asarray([[mean - values.min()], [values.max() - mean]]),
                fmt=MODEL_MARKER[model_id],
                ms=5.3,
                mfc=MODEL_COLOR[model_id],
                mec=MODEL_COLOR[model_id],
                mew=0.6,
                ecolor=MODEL_COLOR[model_id],
                elinewidth=0.9,
                capsize=2.2,
                capthick=0.9,
                zorder=3,
            )
    axis.axhline(0.5, color="#999999", lw=0.7, ls=(0, (3, 3)), zorder=1)
    axis.set_xlim(-0.48, len(INTERVALS) - 0.52)
    axis.set_ylim(0.48, 0.92)
    axis.set_xticks(x, INTERVALS, rotation=32, ha="right")
    axis.set_xlabel("Source-stage interval", labelpad=1)
    axis.set_ylabel("Stage-eligible gene AUROC\n(|partial Spearman|)")
    axis.grid(axis="y", color="#E3E3E3", linewidth=0.55, zorder=0)
    axis.grid(axis="x", visible=False)
    box_axis(axis)
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=MODEL_MARKER[mid],
            markersize=5.3,
            markerfacecolor=MODEL_COLOR[mid],
            markeredgecolor=MODEL_COLOR[mid],
            label=MODEL_LABEL[mid],
        )
        for mid in MODEL_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.7,
    )
    fig.subplots_adjust(left=0.205, right=0.985, bottom=0.265, top=0.835)
    return save(fig, output, "cross_time_gene_allocation_auroc")


def plot_heatmap(output: Path) -> list[str]:
    setup_style()
    data = pd.read_csv(HEATMAP_INPUT)
    data = data[data.outcome.eq("realized")]
    heat = (
        data.pivot(index="module", columns="interval", values="median_partial_spearman")
        .reindex(index=MODULES, columns=HEAT_INTERVALS)
    )
    if heat.isna().any().any():
        raise ValueError("Incomplete growth-programme heatmap matrix")

    fig = plt.figure(figsize=(4.05, 4.35), facecolor="white")
    axis = fig.add_axes((0.43, 0.15, 0.54, 0.70))
    image = axis.imshow(
        heat.to_numpy(float),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-0.4,
        vmax=0.4,
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(len(HEAT_INTERVALS)), HEAT_INTERVALS, rotation=42, ha="right")
    axis.set_yticks(np.arange(len(MODULES)), MODULES)
    axis.tick_params(length=2.5, width=0.65, pad=2)
    box_axis(axis)
    caxis = fig.add_axes((0.55, 0.90, 0.40, 0.022))
    colorbar = fig.colorbar(image, cax=caxis, orientation="horizontal")
    colorbar.set_ticks((-0.4, -0.2, 0.0, 0.2, 0.4))
    colorbar.outline.set_linewidth(0.75)
    colorbar.outline.set_edgecolor("#666666")
    fig.text(0.75, 0.985, "Partial Spearman", ha="center", va="top")
    return save(fig, output, "growth_programme_partial_spearman_heatmap", tight=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    expected = (
        output / "cross_time_gene_allocation_auroc.pdf",
        output / "growth_programme_partial_spearman_heatmap.pdf",
    )
    existing = [path for path in expected if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to overwrite; pass --overwrite: " + str(existing))
    output.mkdir(parents=True, exist_ok=True)

    outputs = plot_cross_time(output) + plot_heatmap(output)
    manifest = {
        "description": "Compact 10-pt redraw of two human-cerebral growth panels",
        "rules": {
            "font": "Arial 10 pt",
            "bold": False,
            "maximum_width_inches": 4.05,
            "full_axis_box": True,
            "iteration_counts_in_labels": False,
        },
        "inputs": {
            str(BREADTH_INPUT.resolve()): sha256(BREADTH_INPUT),
            str(HEATMAP_INPUT.resolve()): sha256(HEATMAP_INPUT),
        },
        "outputs": outputs,
    }
    (output / "growth_redraw_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
