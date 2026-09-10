#!/usr/bin/env python
"""Plot E15.5 strict-LOO RNA and mapped-ATAC cell-type compositions."""

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
import numpy as np
import pandas as pd

from trainfbench_plot_style import apply_nature_rc


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "results/pancreas_strict_loo_time1_six_metrics_20k_with_trajectorynet_mioflow"
    / "pancreas_strict_loo_time1_composition.csv"
)
DEFAULT_OUTPUT = ROOT / "results/pancreas_strict_loo_e155_celltype_composition"

METHODS = (
    ("Observed", None),
    ("COATI bal.", "COATI balanced 20k"),
    ("COATI unbal.", "COATI unbalanced 20k C_y=0.5"),
    ("CytoBridge bal.", "CytoBridge balanced 20k"),
    ("CytoBridge unbal.", "CytoBridge unbalanced 20k"),
    ("MIOFlow", "MIOFlow GAGA10 20k"),
    ("TrajectoryNet", "TrajectoryNet 20k"),
)

# Biological progression order used in the moscot pancreas transition figures.
CELL_TYPES = (
    "Mat. Acinar",
    "Imm. Acinar",
    "Prlf. Ductal",
    "Ductal",
    "Ngn3 low",
    "Ngn3 high cycling",
    "Ngn3 high",
    "Eps. progenitors",
    "Fev+",
    "Fev+ Alpha",
    "Fev+ Beta",
    "Fev+ Delta",
    "Alpha",
    "Beta",
    "Delta",
    "Epsilon",
)

# Exact cell-type palette stored in pancreas rna_adata.h5ad, remapped to the
# biological order above.
CELL_TYPE_COLORS = {
    "Alpha": "#1f77b4",
    "Beta": "#ff7f0e",
    "Delta": "#279e68",
    "Ductal": "#d62728",
    "Eps. progenitors": "#aa40fc",
    "Epsilon": "#8c564b",
    "Fev+": "#e377c2",
    "Fev+ Alpha": "#b5bd61",
    "Fev+ Beta": "#42f5ec",
    "Fev+ Delta": "#aec7e8",
    "Imm. Acinar": "#ffeb3b",
    "Mat. Acinar": "#98df8a",
    "Ngn3 high": "#ff9896",
    "Ngn3 high cycling": "#adf542",
    "Ngn3 low": "#c5b0d5",
    "Prlf. Ductal": "#f7b6d2",
}

LEGEND_LABELS = {
    "Mat. Acinar": "Mat. acinar",
    "Imm. Acinar": "Imm. acinar",
    "Prlf. Ductal": "Prlf. ductal",
    "Ductal": "Ductal",
    "Ngn3 low": r"Ngn3$^{low}$",
    "Ngn3 high cycling": r"Ngn3$^{high}$ cycling",
    "Ngn3 high": r"Ngn3$^{high}$",
    "Eps. progenitors": "Eps. prog.",
    "Fev+": r"Fev$^{+}$",
    "Fev+ Alpha": r"Fev$^{+}$ alpha",
    "Fev+ Beta": r"Fev$^{+}$ beta",
    "Fev+ Delta": r"Fev$^{+}$ delta",
    "Alpha": "Alpha",
    "Beta": "Beta",
    "Delta": "Delta",
    "Epsilon": "Epsilon",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_table(composition: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for modality in ("RNA", "ATAC"):
        modality_rows = composition.loc[composition["modality"].eq(modality)]
        observed = (
            modality_rows.groupby("celltype", observed=True)["observed_fraction"]
            .first()
            .reindex(CELL_TYPES, fill_value=0.0)
        )
        for display, method in METHODS:
            if method is None:
                values = observed
            else:
                values = (
                    modality_rows.loc[modality_rows["method"].eq(method)]
                    .set_index("celltype")["predicted_fraction"]
                    .reindex(CELL_TYPES, fill_value=0.0)
                )
            total = float(values.sum())
            if not np.isclose(total, 1.0, rtol=0.0, atol=1e-6):
                raise ValueError(f"{modality} {display} sums to {total}")
            for celltype, fraction in values.items():
                rows.append(
                    {
                        "modality": modality,
                        "display": display,
                        "method": "Observed" if method is None else method,
                        "celltype": celltype,
                        "fraction": float(fraction),
                    }
                )
    return pd.DataFrame(rows)


def plot_composition(table: pd.DataFrame, output_dir: Path) -> None:
    apply_nature_rc(font_size=10.0)
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
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # 4.05 in = 10.29 cm, safely below half an A4 page width (10.5 cm).
    fig, axes = plt.subplots(2, 1, figsize=(4.05, 5.2), sharex=True, facecolor="white")
    y = np.arange(len(METHODS), dtype=np.float64)
    displays = [item[0] for item in METHODS]
    for ax, modality in zip(axes, ("RNA", "ATAC")):
        left = np.zeros(len(METHODS), dtype=np.float64)
        for celltype in CELL_TYPES:
            values = np.asarray(
                [
                    table.loc[
                        table["modality"].eq(modality)
                        & table["display"].eq(display)
                        & table["celltype"].eq(celltype),
                        "fraction",
                    ].iloc[0]
                    for display in displays
                ],
                dtype=np.float64,
            )
            ax.barh(
                y,
                values,
                left=left,
                height=0.64,
                color=CELL_TYPE_COLORS[celltype],
                edgecolor="none",
                linewidth=0,
                zorder=2,
            )
            left += values
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(len(METHODS) - 0.35, -0.65)
        ax.set_xticks([0.0, 0.5, 1.0], ["0", "50%", "100%"])
        ax.set_yticks(y, displays)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.tick_params(axis="x", length=2.5, width=0.7, pad=2)
        ax.set_title(modality, fontweight="normal", pad=3)
        ax.axvline(0.5, color="#D0D0D0", lw=0.6, zorder=0)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.7)

    handles = [
        mpl.patches.Patch(
            facecolor=CELL_TYPE_COLORS[celltype],
            edgecolor="none",
            label=LEGEND_LABELS[celltype],
        )
        for celltype in CELL_TYPES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.040),
        ncol=2,
        frameon=False,
        fontsize=10.0,
        prop={"family": "Arial", "size": 10.0, "weight": "normal"},
        handlelength=0.8,
        handleheight=0.8,
        handletextpad=0.3,
        columnspacing=1.0,
        labelspacing=0.35,
    )
    fig.suptitle(
        "Held-out E15.5 cell-type composition",
        fontsize=10.0,
        fontweight="normal",
        y=0.982,
    )
    fig.subplots_adjust(
        left=0.39,
        right=0.985,
        top=0.89,
        bottom=0.43,
        hspace=0.15,
    )
    stem = output_dir / "pancreas_strict_loo_e155_rna_atac_celltype_composition"
    options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.02}
    fig.savefig(stem.with_suffix(".png"), dpi=600, **options)
    fig.savefig(stem.with_suffix(".pdf"), **options)
    fig.savefig(stem.with_suffix(".svg"), **options)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} is not empty; pass --overwrite to replace its files"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    composition = pd.read_csv(args.input)
    table = build_table(composition)
    table.to_csv(args.output_dir / "celltype_composition_long.csv", index=False)
    plot_composition(table, args.output_dir)
    manifest = {
        "task": "Pancreas strict-LOO E15.5 RNA and mapped-ATAC cell-type composition",
        "observed": "Observed E15.5 cell-number fractions",
        "predicted": (
            "Predicted E15.5 fractions from 1NN cell-type assignments; native "
            "particle-mass weighting for unbalanced models and uniform weighting otherwise"
        ),
        "celltype_order": list(CELL_TYPES),
        "input": str(args.input.resolve()),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
