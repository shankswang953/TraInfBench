#!/usr/bin/env python
"""Plot six supplementary COATI-B palate peak--gene pairs in a 2x3 grid."""

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

from plot_palate_peak_gene_supplementary_pairs_a4 import (
    ATAC_COLOR,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    DEFAULT_RANKING,
    PAIR_SPECS,
    RNA_COLOR,
    configure_plot,
    select_one,
    trajectory_zscore,
)


METHOD = "COATI-B"
DISPLAY_PAIRS = PAIR_SPECS[:6]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "palate_coati_b_supplementary_peak_gene_pairs_2x3"
    targets = [
        stem.with_suffix(".png"),
        stem.with_suffix(".pdf"),
        stem.with_suffix(".svg"),
        args.output_dir / "coati_b_supplementary_peak_gene_pairs_2x3_values.csv",
    ]
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError("Outputs exist; pass --overwrite")

    ranking = pd.read_csv(args.ranking)
    curves = pd.read_csv(args.input_dir / "trajectory_time_curves.csv.gz")
    rows: list[dict[str, object]] = []
    configure_plot()
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(8.27, 4.85),
        sharex=True,
        sharey=True,
    )

    for axis, spec in zip(axes.ravel(), DISPLAY_PAIRS):
        score = select_one(ranking, spec, METHOD)
        local = curves[
            curves["model"].eq(METHOD)
            & curves["cohort"].eq(spec["cohort"])
            & curves["branch"].eq(spec["branch"])
            & curves["gene"].eq(spec["gene"])
            & curves["peak"].eq(spec["peak"])
        ].sort_values("time_index")
        if len(local) < 5:
            raise RuntimeError(f"Missing trajectory curve for {spec}")
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
        axis.set_title(f"{spec['gene']}  {offset_kb:+.1f} kb", loc="left", pad=2)
        axis.text(
            0.025,
            0.93,
            f"{source} → {target}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            linespacing=1.0,
        )
        axis.axhline(0, color="0.72", lw=0.5)
        axis.grid(axis="y", color="0.92", lw=0.45)
        axis.set_xlim(12.45, 14.60)
        axis.set_xticks([12.5, 13.5, 14.5], ["E12.5", "E13.5", "E14.5"])
        axis.spines[["top", "right"]].set_visible(False)
        rows.append(
            {
                "model": METHOD,
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
        bbox_to_anchor=(0.52, 0.995),
    )
    figure.supylabel("Trajectory z-score", x=0.018, fontsize=10)
    figure.supxlabel("Embryonic stage", y=0.018, fontsize=10)
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.89,
        bottom=0.14,
        wspace=0.18,
        hspace=0.42,
    )
    figure.savefig(targets[0], dpi=400, facecolor="white")
    figure.savefig(targets[1], facecolor="white")
    figure.savefig(targets[2], facecolor="white")
    plt.close(figure)
    pd.DataFrame(rows).to_csv(targets[3], index=False)
    for path in targets:
        print(path)


if __name__ == "__main__":
    main()
