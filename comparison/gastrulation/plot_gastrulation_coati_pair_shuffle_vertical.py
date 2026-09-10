#!/usr/bin/env python3
"""Plot the COATI observed-pair versus mismatched-pair null vertically."""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT / "results" / "gastrulation_coati_b_temporal_pair_shuffle_validation"
)
LINEAGES = ("NMP-retaining", "Neural", "Mesoderm")
COATI_BLUE = "#0072B2"
NULL_GREY = "#B8B8B8"
GRID_GREY = "#D9D9D9"
TEXT_GREY = "#4D4D4D"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args()


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "font.weight": "normal",
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "axes.labelsize": 10,
            "axes.labelweight": "normal",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def p_label(value: float, permutations: int) -> str:
    if np.isclose(value, 1.0 / (permutations + 1)):
        return rf"$P_{{perm}}<10^{{-{int(np.log10(permutations))}}}$"
    return rf"$P_{{perm}}={value:.3g}$"


def plot_vertical(
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    configure_plotting()
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(3.95, 5.95),
        sharex=True,
        sharey=True,
    )

    for row_index, (axis, lineage) in enumerate(zip(axes, LINEAGES)):
        local = scores.loc[scores["lineage"].eq(lineage)]
        mismatched = local.loc[
            local["pairing"].eq("Mismatched pair"), "S_max"
        ].dropna().to_numpy()
        observed = local.loc[
            local["pairing"].eq("Observed pair"), "S_max"
        ].dropna().to_numpy()

        violin = axis.violinplot(
            [mismatched, observed],
            positions=[0, 1],
            widths=0.62,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.18,
        )
        for body, color in zip(violin["bodies"], [NULL_GREY, COATI_BLUE]):
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.82)

        medians = [float(np.median(mismatched)), float(np.median(observed))]
        axis.scatter(
            [0, 1],
            medians,
            marker="D",
            s=22,
            c=[TEXT_GREY, "black"],
            zorder=4,
        )
        for x_value, median in enumerate(medians):
            axis.text(
                x_value,
                median + 0.10,
                f"{median:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        info = summary.loc[summary["lineage"].eq(lineage)].iloc[0]
        axis.set_title(lineage, loc="left", pad=2)
        axis.text(
            0.99,
            0.04,
            f"AUC={info.common_language_AUC:.3f}; "
            f"{p_label(float(info.median_pairing_permutation_p), int(info.pairing_permutations))}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
        )
        axis.set_xlim(-0.55, 1.55)
        axis.set_ylim(-1.05, 1.18)
        axis.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        axis.axhline(0, color=GRID_GREY, linewidth=0.6, zorder=0)
        axis.grid(axis="y", color=GRID_GREY, linewidth=0.45, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", length=0)
        if row_index < len(axes) - 1:
            axis.tick_params(labelbottom=False)

    axes[-1].set_xticks(
        [0, 1],
        [
            f"Mismatched\n(n={int(summary['n_mismatched_pairs'].iloc[0]):,})",
            f"Observed\n(n={int(summary['n_observed_pairs'].iloc[0]):,})",
        ],
    )
    figure.supylabel(
        r"Temporal concordance, $S_{\max}$",
        x=0.02,
        fontsize=10,
    )
    figure.suptitle(
        "Observed pair identities versus mismatched-pair null",
        x=0.55,
        y=0.985,
        fontsize=10,
        fontweight="normal",
    )
    figure.legend(
        handles=[
            Patch(facecolor=NULL_GREY, edgecolor="none", label="Mismatched pair"),
            Patch(facecolor=COATI_BLUE, edgecolor="none", label="Observed pair"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.55, 0.955),
        frameon=False,
        ncol=2,
        handlelength=1.1,
        columnspacing=1.2,
    )
    figure.subplots_adjust(
        left=0.22,
        right=0.98,
        top=0.88,
        bottom=0.11,
        hspace=0.26,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "coati_temporal_pair_shuffle_validation_vertical"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(
        stem.with_suffix(".png"),
        dpi=450,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    scores = pd.read_csv(
        args.input_dir / "observed_and_mismatched_pair_scores.csv.gz"
    )
    summary = pd.read_csv(args.input_dir / "pair_shuffle_lineage_summary.csv")
    plot_vertical(scores, summary, args.output_dir)
    print(
        f"wrote: {args.output_dir / 'coati_temporal_pair_shuffle_validation_vertical.pdf'}"
    )


if __name__ == "__main__":
    main()
