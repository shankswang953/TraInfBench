#!/usr/bin/env python
"""Plot the prespecified supplementary palate peak--gene trajectory examples.

The figure uses the same trajectory decoding and z-score convention as the
four-pair main-text figure.  It compares COATI-B and COATI-U side by side and
is sized to the width of an A4 page.  No observed pseudobulk values or
correlation scores are displayed.
"""

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
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results/palate_coati_peak_gene_trajectory_time_classes_cy05"
DEFAULT_RANKING = (
    ROOT
    / "results/palate_peak_gene_max_correlation_supported_pairs"
    / "all_peak_gene_max_correlation_rankings.csv"
)
DEFAULT_OUTPUT = ROOT / "results/palate_peak_gene_max_correlation_supported_pairs"

METHODS = ("COATI-B", "COATI-U")
RNA_COLOR = "#0072B2"
ATAC_COLOR = "#D55E00"

# Prespecified from the complete strict-distal, branch-matched H3K27ac table.
# The final row is retained as an evidence-first TF-bound active CRE rather
# than because it has one of the largest trajectory correlations.
PAIR_SPECS = (
    {
        "gene": "Inhba",
        "peak": "chr13-16202489-16203399",
        "cohort": "same_lineage",
        "branch": "anterior",
        "evidence_label": "",
    },
    {
        "gene": "Shox2",
        "peak": "chr3-66812541-66813491",
        "cohort": "same_lineage",
        "branch": "anterior",
        "evidence_label": "Shox2 cis-linked",
    },
    {
        "gene": "Prickle1",
        "peak": "chr15-93706299-93706832",
        "cohort": "CNC_to_branch",
        "branch": "posterior",
        "evidence_label": "",
    },
    {
        "gene": "Cyp26b1",
        "peak": "chr6-84656712-84657529",
        "cohort": "same_lineage",
        "branch": "anterior",
        "evidence_label": "",
    },
    {
        "gene": "Sim2",
        "peak": "chr16-94249883-94250668",
        "cohort": "same_lineage",
        "branch": "posterior",
        "evidence_label": "",
    },
    {
        "gene": "Shox2",
        "peak": "chr3-66584742-66586482",
        "cohort": "CNC_to_branch",
        "branch": "anterior",
        "evidence_label": "",
    },
    {
        "gene": "Shox2",
        "peak": "chr3-66967527-66967810",
        "cohort": "CNC_to_branch",
        "branch": "anterior",
        "evidence_label": "TF-bound active CRE",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_plot() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "font.weight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def trajectory_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = float(np.std(values, ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / scale


def select_one(frame: pd.DataFrame, spec: dict[str, str], method: str) -> pd.Series:
    selected = frame[
        frame["model"].eq(method)
        & frame["cohort"].eq(spec["cohort"])
        & frame["branch"].eq(spec["branch"])
        & frame["gene"].eq(spec["gene"])
        & frame["peak"].eq(spec["peak"])
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one row for {method}, {spec['gene']}, {spec['peak']}; "
            f"found {len(selected)}"
        )
    return selected.iloc[0]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "palate_coati_supplementary_peak_gene_pairs_a4"
    targets = [
        stem.with_suffix(".png"),
        stem.with_suffix(".pdf"),
        stem.with_suffix(".svg"),
        args.output_dir / "supplementary_peak_gene_pairs_a4_values.csv",
    ]
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError("Outputs exist; pass --overwrite")

    ranking = pd.read_csv(args.ranking)
    curves = pd.read_csv(args.input_dir / "trajectory_time_curves.csv.gz")
    selected_rows: list[dict[str, object]] = []

    configure_plot()
    figure, axes = plt.subplots(
        len(PAIR_SPECS),
        len(METHODS),
        figsize=(8.27, 11.15),
        sharex=True,
        sharey=True,
    )
    for column, method in enumerate(METHODS):
        axes[0, column].text(
            0.5,
            1.34,
            method,
            transform=axes[0, column].transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="normal",
        )
    for row, spec in enumerate(PAIR_SPECS):
        for column, method in enumerate(METHODS):
            axis = axes[row, column]
            score = select_one(ranking, spec, method)
            local = curves[
                curves["model"].eq(method)
                & curves["cohort"].eq(spec["cohort"])
                & curves["branch"].eq(spec["branch"])
                & curves["gene"].eq(spec["gene"])
                & curves["peak"].eq(spec["peak"])
            ].sort_values("time_index")
            if len(local) < 5:
                raise RuntimeError(f"Missing trajectory curve for {method} {spec}")
            x = local["embryonic_day"].to_numpy(float)
            rna = savgol_filter(np.log1p(local["rna_cpm"]), 5, 2, mode="interp")
            atac = savgol_filter(np.log1p(local["atac_cp10k"]), 5, 2, mode="interp")
            axis.plot(x, trajectory_zscore(rna), color=RNA_COLOR, lw=1.35, label="RNA")
            axis.plot(
                x,
                trajectory_zscore(atac),
                color=ATAC_COLOR,
                lw=1.35,
                ls="--",
                label="ATAC",
            )
            target = "Anterior" if spec["branch"] == "anterior" else "Posterior"
            source = "CNC progenitor" if spec["cohort"] == "CNC_to_branch" else target
            offset_kb = float(score["signed_offset_bp"]) / 1000.0
            if column == 0:
                title = f"{spec['gene']}  {offset_kb:+.1f} kb"
                if spec["evidence_label"]:
                    title += f"  ({spec['evidence_label']})"
                axis.set_title(title, loc="left", pad=2, fontsize=9.5)
            axis.text(
                0.02,
                0.92,
                f"{source} → {target}; lag {float(score['lag_at_max_days']):+.2f} d",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8.0,
            )
            axis.axhline(0, color="0.72", lw=0.5)
            axis.grid(axis="y", color="0.92", lw=0.45)
            axis.set_xlim(12.45, 14.60)
            axis.set_xticks([12.5, 13.5, 14.5], ["E12.5", "E13.5", "E14.5"])
            axis.spines[["top", "right"]].set_visible(False)
            selected_rows.append(
                {
                    "display_order": row + 1,
                    "model": method,
                    **spec,
                    "signed_offset_bp": int(score["signed_offset_bp"]),
                    "max_derivative_cosine": float(score["max_derivative_cosine"]),
                    "lag_at_max_days": float(score["lag_at_max_days"]),
                    "branch_matched_h3k27ac": bool(score["branch_matched_h3k27ac"]),
                    "branch_matched_tf_cre": bool(score["branch_matched_tf_cre"]),
                    "shox2_cis_linked": bool(score["shox2_cis_linked"]),
                }
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.52, 0.997),
    )
    figure.supylabel("Trajectory z-score", x=0.018, fontsize=10)
    figure.supxlabel("Embryonic stage", y=0.018, fontsize=10)
    figure.subplots_adjust(
        left=0.095,
        right=0.985,
        top=0.945,
        bottom=0.065,
        wspace=0.16,
        hspace=0.53,
    )
    figure.savefig(targets[0], dpi=400, facecolor="white")
    figure.savefig(targets[1], facecolor="white")
    figure.savefig(targets[2], facecolor="white")
    plt.close(figure)
    pd.DataFrame(selected_rows).to_csv(targets[3], index=False)
    for path in targets:
        print(path)


if __name__ == "__main__":
    main()
