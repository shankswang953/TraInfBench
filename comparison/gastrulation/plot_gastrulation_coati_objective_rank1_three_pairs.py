#!/usr/bin/env python
"""Plot the objective rank-1 peak--gene pair from each COATI lineage."""

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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from trainfbench_plot_style import NATURE_CUD, apply_nature_rc


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT / "results" / "gastrulation_coati_b_objective_lineage_temporal"
)
DEFAULT_OUTPUT = DEFAULT_INPUT

# Panel order is fixed before plotting; each row is the rank-1 candidate in the
# corresponding lineage, not a manually selected example.
PANELS = (
    ("t1", "NMP-retaining"),
    ("t2", "Neural"),
    ("t3", "Mesoderm"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = float(np.nanstd(values, ddof=0))
    if not np.isfinite(scale) or scale == 0:
        return np.zeros_like(values)
    return (values - float(np.nanmean(values))) / scale


def plot_figure(
    curves: pd.DataFrame,
    audit: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    rank1 = audit.loc[audit["rank_within_lineage"].eq(1)].copy()
    rank1 = rank1.set_index("lineage", verify_integrity=True)

    missing = [lineage for _, lineage in PANELS if lineage not in rank1.index]
    if missing:
        raise ValueError(f"Missing rank-1 candidate(s) for: {missing}")

    apply_nature_rc(font_size=10)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(3.95, 2.25),
        sharex=True,
        sharey=True,
    )

    selected_rows: list[dict[str, object]] = []
    for panel_index, (axis, (tag, lineage)) in enumerate(zip(axes, PANELS)):
        row = rank1.loc[lineage]
        local = curves.loc[
            curves["candidate_id"].eq(row["candidate_id"])
            & curves["lineage"].eq(lineage)
        ].sort_values("normalized_time")
        if local.empty:
            raise ValueError(
                f"No curve for {row['candidate_id']} in {lineage}"
            )

        day = local["embryonic_day_display"].to_numpy(float)
        rna = zscore(local["rna_log1p_cpm_smoothed"].to_numpy(float))
        atac = zscore(local["atac_log1p_cp10k_smoothed"].to_numpy(float))
        axis.plot(
            day,
            rna,
            color=NATURE_CUD["blue"],
            linewidth=1.5,
            solid_capstyle="round",
        )
        axis.plot(
            day,
            atac,
            color=NATURE_CUD["vermillion"],
            linestyle="--",
            dashes=(3.0, 1.7),
            linewidth=1.5,
            dash_capstyle="round",
        )

        offset_kb = float(row["signed_transcriptional_offset_bp"]) / 1000.0
        axis.set_title(lineage, pad=5, fontsize=10)
        axis.text(
            0.02,
            0.98,
            f"{row['gene']} · {offset_kb:+.1f} kb",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            linespacing=1.02,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 0.15,
            },
            zorder=5,
        )

        axis.axhline(0, color="0.68", linewidth=0.5, zorder=0)
        axis.grid(axis="y", color="0.90", linewidth=0.45, zorder=0)
        axis.set_xlim(7.5, 8.75)
        axis.set_ylim(-2.35, 2.35)
        axis.set_yticks([-2, 0, 2])
        axis.spines[["top", "right"]].set_visible(False)
        tick_labels = ["E7.5", "t1", "t2", "t3"]
        if panel_index > 0:
            tick_labels[0] = ""
        if panel_index < len(axes) - 1:
            tick_labels[-1] = ""
        axis.set_xticks([7.5, 8.0, 8.5, 8.75], tick_labels)
        axis.tick_params(axis="x", labelrotation=0, pad=2.0)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("center")
        if panel_index > 0:
            axis.tick_params(labelleft=False)

        selected_rows.append(
            {
                "panel": tag,
                "lineage": lineage,
                "candidate_id": row["candidate_id"],
                "gene": row["gene"],
                "peak": row["peak"],
                "signed_transcriptional_offset_bp": int(
                    row["signed_transcriptional_offset_bp"]
                ),
                "rank_within_lineage": int(row["rank_within_lineage"]),
                "S_max": float(row["S_max"]),
            }
        )

    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=NATURE_CUD["blue"],
                linewidth=1.5,
                label="RNA",
            ),
            Line2D(
                [0],
                [0],
                color=NATURE_CUD["vermillion"],
                linestyle="--",
                dashes=(3.0, 1.7),
                linewidth=1.5,
                label="ATAC",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.57, 0.995),
        frameon=False,
        ncol=2,
        handlelength=2.3,
        columnspacing=1.4,
        borderaxespad=0,
    )
    figure.supylabel("z-score", x=0.015, fontsize=10)
    figure.subplots_adjust(
        left=0.14,
        right=0.995,
        top=0.80,
        bottom=0.24,
        wspace=0.12,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "objective_rank1_three_pairs_compact"
    figure.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)

    selected = pd.DataFrame(selected_rows)
    selected.to_csv(
        output_dir / "objective_rank1_three_pairs_compact_selection.csv",
        index=False,
    )
    return selected


def main() -> None:
    args = parse_args()
    curves = pd.read_csv(args.input_dir / "trajectory_time_curves.csv.gz")
    audit = pd.read_csv(
        args.input_dir / "external_support_postranking_audit.csv"
    )
    selected = plot_figure(curves, audit, args.output_dir)
    print(selected.to_string(index=False))
    print(f"wrote: {args.output_dir / 'objective_rank1_three_pairs_compact.png'}")


if __name__ == "__main__":
    main()
