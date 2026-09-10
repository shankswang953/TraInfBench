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

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from trainfbench_plot_style import METHOD_STYLES


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = ROOT / "results/gastrulation_crispr_t_e85_to_e875"
DEFAULT_INPUT = DEFAULT_RESULT_DIR / "e875_NMP_knn_fate_summary.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_DIR / "cross_modal_response_1nn"

SYNC = {
    "BSOT": {"color": "#0072B2", "marker": "o"},
    "USOT": {"color": "#D55E00", "marker": "D"},
}
SYNC_DISPLAY = {
    "BSOT": "COATI bal.",
    "USOT": "COATI unbal.",
}
BASELINES = (
    "MIOFlow",
    "CytoBridge balanced",
    "CytoBridge unbalanced",
    "TIGON",
    "TrajectoryNet",
)
DISPLAY = {
    "MIOFlow": "MIOFlow",
    "CytoBridge balanced": "CytoBridge bal.",
    "CytoBridge unbalanced": "CytoBridge unbal.",
    "TIGON": "TIGON",
    "TrajectoryNet": "TrajectoryNet",
}

EXPECTED_FILL = "#E4F1E6"
EXPECTED_TEXT = "#356A42"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 10,
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


def _paired(frame: pd.DataFrame, method_filter: pd.Series, column: str) -> pd.DataFrame:
    selected = frame.loc[method_filter, ["method", "readout", column]].copy()
    paired = selected.pivot(index="method", columns="readout", values=column)
    missing = {"RNA", "ATAC"}.difference(paired.columns)
    if missing:
        raise ValueError(f"Missing readouts {sorted(missing)} for {column}")
    return paired.sort_index()


def _is_expected_direction(fate: str, delta_rna: float, delta_atac: float) -> bool:
    """Define the shaded response-direction criterion used in the figure."""
    if fate == "Spinal cord":
        return bool(delta_rna > 0 and delta_atac > 0)
    return bool(delta_rna * delta_atac > 0)


def _build_points(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for fate, column in (
        ("Spinal cord", "spinal_delta_pp"),
        ("Retained NMP", "retained_NMP_delta_pp"),
    ):
        for family in SYNC:
            paired = _paired(
                frame,
                frame["method"].str.startswith(family + " "),
                column,
            )
            for method, row in paired.iterrows():
                records.append(
                    {
                        "fate": fate,
                        "method": method,
                        "family": family,
                        "kind": "C_y setting",
                        "delta_RNA_pp": float(row["RNA"]),
                        "delta_ATAC_pp": float(row["ATAC"]),
                        "expected_both": _is_expected_direction(
                            fate, float(row["RNA"]), float(row["ATAC"])
                        ),
                    }
                )
            records.append(
                {
                    "fate": fate,
                    "method": family,
                    "family": family,
                    "kind": "mean",
                    "delta_RNA_pp": float(paired["RNA"].mean()),
                    "delta_ATAC_pp": float(paired["ATAC"].mean()),
                    "expected_both": _is_expected_direction(
                        fate,
                        float(paired["RNA"].mean()),
                        float(paired["ATAC"].mean()),
                    ),
                }
            )
        for method in BASELINES:
            paired = _paired(frame, frame["method"].eq(method), column)
            row = paired.iloc[0]
            records.append(
                {
                    "fate": fate,
                    "method": method,
                    "family": method,
                    "kind": "baseline",
                    "delta_RNA_pp": float(row["RNA"]),
                    "delta_ATAC_pp": float(row["ATAC"]),
                    "expected_both": _is_expected_direction(
                        fate, float(row["RNA"]), float(row["ATAC"])
                    ),
                }
            )
    points = pd.DataFrame.from_records(records)
    points["cross_modal_gap_pp"] = (
        points["delta_ATAC_pp"] - points["delta_RNA_pp"]
    ).abs()
    return points


LABEL_OFFSETS = {
    "Spinal cord": {
        "BSOT": (7, 8),
        "USOT": (7, -12),
        "MIOFlow": (8, -12),
        "CytoBridge balanced": (8, -12),
        "CytoBridge unbalanced": (8, -10),
        "TIGON": (8, 10),
        "TrajectoryNet": (-90, 13),
    },
    "Retained NMP": {
        "BSOT": (-75, -14),
        "USOT": (7, 8),
        "MIOFlow": (10, -18),
        "CytoBridge balanced": (-12, 20),
        "CytoBridge unbalanced": (-118, -10),
        "TIGON": (10, 10),
        "TrajectoryNet": (7, -11),
    },
}


def _draw_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    points: pd.DataFrame,
    *,
    fate: str,
    column: str,
    lim: tuple[float, float],
) -> None:
    lower, upper = lim
    if fate == "Spinal cord":
        ax.add_patch(
            Rectangle(
                (0, 0),
                upper,
                upper,
                facecolor=EXPECTED_FILL,
                edgecolor="none",
                zorder=0,
            )
        )
        expected_labels = [((upper - 0.45, upper - 0.7), "right", "top")]
    else:
        ax.add_patch(
            Rectangle(
                (lower, lower),
                -lower,
                -lower,
                facecolor=EXPECTED_FILL,
                edgecolor="none",
                zorder=0,
            )
        )
        ax.add_patch(
            Rectangle(
                (0, 0),
                upper,
                upper,
                facecolor=EXPECTED_FILL,
                edgecolor="none",
                zorder=0,
            )
        )
        expected_labels = [
            ((lower + 0.65, lower + 0.65), "left", "bottom"),
        ]

    ax.plot(lim, lim, color="#9E9E9E", linewidth=0.9, linestyle="--", zorder=1)
    ax.axhline(0, color="#222222", linewidth=1.0, zorder=1)
    ax.axvline(0, color="#222222", linewidth=1.0, zorder=1)
    for expected_xy, expected_ha, expected_va in expected_labels:
        ax.text(
            *expected_xy,
            "Expected\ndirection",
            color=EXPECTED_TEXT,
            ha=expected_ha,
            va=expected_va,
            fontsize=10,
            fontfamily="Arial",
            linespacing=0.92,
            fontweight="normal",
            zorder=2,
        )

    panel_points = points[points["fate"].eq(fate)]
    for family, style in SYNC.items():
        paired = _paired(
            frame,
            frame["method"].str.startswith(family + " "),
            column,
        )
        x = paired["RNA"].to_numpy(float)
        y = paired["ATAC"].to_numpy(float)
        ax.scatter(
            x,
            y,
            s=28,
            marker=style["marker"],
            facecolor="white",
            edgecolor=style["color"],
            linewidth=0.9,
            alpha=0.55,
            zorder=3,
        )
        mean_x = float(x.mean())
        mean_y = float(y.mean())
        ax.errorbar(
            mean_x,
            mean_y,
            xerr=float(x.std(ddof=1)),
            yerr=float(y.std(ddof=1)),
            fmt="none",
            ecolor=style["color"],
            elinewidth=1.3,
            capsize=3,
            capthick=1.1,
            alpha=0.9,
            zorder=4,
        )
        ax.scatter(
            [mean_x],
            [mean_y],
            s=105,
            marker=style["marker"],
            facecolor=style["color"],
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )
    for method in BASELINES:
        row = panel_points[
            panel_points["method"].eq(method)
            & panel_points["kind"].eq("baseline")
        ].iloc[0]
        method_style = METHOD_STYLES[method]
        edgecolor = (
            "none"
            if method == "TrajectoryNet"
            else (
                method_style.markeredgecolor
                if method_style.markeredgecolor is not None
                else method_style.color
            )
        )
        x = float(row["delta_RNA_pp"])
        y = float(row["delta_ATAC_pp"])
        ax.scatter(
            [x],
            [y],
            s=64,
            marker=method_style.marker,
            facecolor=method_style.color,
            edgecolor=edgecolor,
            linewidth=0.0 if method == "TrajectoryNet" else 1.0,
            zorder=4,
        )
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(fate, loc="left", fontweight="normal", pad=8)
    ax.set_xlabel(r"$\Delta$RNA (T-KO $-$ WT, pp)")
    ax.set_ylabel(r"$\Delta$ATAC (T-KO $-$ WT, pp)")
    ax.grid(color="#E0E0E0", linewidth=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _legend_handles() -> list[mlines.Line2D]:
    handles = [
        mlines.Line2D(
            [],
            [],
            marker=SYNC["BSOT"]["marker"],
            linestyle="none",
            markerfacecolor=SYNC["BSOT"]["color"],
            markeredgecolor="white",
            markersize=8,
            label="COATI bal. mean",
        ),
        mlines.Line2D(
            [],
            [],
            marker=SYNC["USOT"]["marker"],
            linestyle="none",
            markerfacecolor=SYNC["USOT"]["color"],
            markeredgecolor="white",
            markersize=8,
            label="COATI unbal. mean",
        ),
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#777777",
            alpha=0.6,
            markersize=5,
            label=r"individual $C_y$",
        ),
    ]
    for method in BASELINES:
        style = METHOD_STYLES[method]
        edgecolor = (
            "none"
            if method == "TrajectoryNet"
            else (
                style.markeredgecolor
                if style.markeredgecolor is not None
                else style.color
            )
        )
        handles.append(
            mlines.Line2D(
                [],
                [],
                marker=style.marker,
                linestyle="none",
                markerfacecolor=style.color,
                markeredgecolor=edgecolor,
                markeredgewidth=0.0 if method == "TrajectoryNet" else 1.0,
                markersize=7,
                label=DISPLAY[method],
            )
        )
    return handles


def _save_individual(
    frame: pd.DataFrame,
    points: pd.DataFrame,
    *,
    fate: str,
    column: str,
    lim: tuple[float, float],
    output_prefix: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(4.45, 4.15))
    _draw_panel(ax, frame, points, fate=fate, column=column, lim=lim)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.16, top=0.88)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_prefix.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    fig.savefig(
        output_prefix.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    plt.close(fig)


def _save_combined(
    frame: pd.DataFrame,
    points: pd.DataFrame,
    *,
    specs: tuple[tuple[str, str, tuple[float, float], str], ...],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.20, 4.55))
    for ax, (fate, column, lim, _) in zip(axes, specs):
        _draw_panel(ax, frame, points, fate=fate, column=column, lim=lim)
        ax.set_ylabel("")
    axes[0].set_xlabel("")
    fig.supylabel(
        r"$\Delta$ATAC (T-KO $-$ WT, pp)",
        x=0.13,
        fontsize=10,
        fontfamily="Arial",
    )
    fig.legend(
        handles=_legend_handles(),
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.51, 0.997),
        columnspacing=0.55,
        handletextpad=0.3,
        labelspacing=0.22,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(
        left=0.20,
        right=0.985,
        bottom=0.09,
        top=0.78,
        hspace=0.36,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    fig.savefig(
        output_path.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot RNA-versus-ATAC CRISPR T-KO response changes for terminal "
            "NMP fate proportions."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args()

    _style()
    frame = pd.read_csv(args.input)
    frame = frame[frame["k"].eq(args.k)].copy()
    if frame.empty:
        raise ValueError(f"No rows found for k={args.k} in {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    points = _build_points(frame)
    points_path = args.output_dir / f"cross_modal_response_points_k{args.k}.csv"
    points.to_csv(points_path, index=False)
    print(f"wrote: {points_path}")

    specs = (
        ("Spinal cord", "spinal_delta_pp", (-9.0, 9.0), "spinal_cord"),
        ("Retained NMP", "retained_NMP_delta_pp", (-15.0, 8.0), "retained_nmp"),
    )
    for fate, column, lim, stem in specs:
        output_prefix = args.output_dir / f"{stem}_response_scatter_k{args.k}"
        _save_individual(
            frame,
            points,
            fate=fate,
            column=column,
            lim=lim,
            output_prefix=output_prefix,
        )
        print(f"wrote: {output_prefix.with_suffix('.png')}")
        print(f"wrote: {output_prefix.with_suffix('.pdf')}")

    combined_path = args.output_dir / f"cross_modal_response_scatter_k{args.k}.png"
    _save_combined(frame, points, specs=specs, output_path=combined_path)
    print(f"wrote: {combined_path}")
    print(f"wrote: {combined_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
