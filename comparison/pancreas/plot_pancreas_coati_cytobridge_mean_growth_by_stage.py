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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "results"
    / "pancreas_unbalanced_growth_gene_programs_20k"
    / "per_cell_instantaneous_growth.csv.gz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "pancreas_coati_cytobridge_mean_growth_by_stage"
)

STAGES = ["E14.5", "E15.5", "E16.5"]
CELL_TYPES = [
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
]
DISPLAY_LABELS = {
    "Ngn3 high cycling": "Ngn3 high cyc.",
    "Eps. progenitors": "Eps. prog.",
    "Fev+ Alpha": "Fev+ α",
    "Fev+ Beta": "Fev+ β",
    "Fev+ Delta": "Fev+ δ",
}
METHODS = {
    "COATI": "coati_c_y_0_5__growth_raw",
    "CytoBridge": "cytobridge__growth_raw",
}
COLORS = {"COATI": "#D55E00", "CytoBridge": "#CC79A7"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare COATI and CytoBridge mean growth by pancreas stage and cell type."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage in STAGES:
        stage_frame = frame.loc[frame["stage"].eq(stage)].copy()
        for method, column in METHODS.items():
            stage_mean = float(stage_frame[column].mean())
            stage_sd = float(stage_frame[column].std(ddof=0))
            if not np.isfinite(stage_sd) or stage_sd <= 0:
                raise ValueError(f"Non-positive growth SD for {method} at {stage}.")
            stage_frame["growth_z"] = (stage_frame[column] - stage_mean) / stage_sd
            for celltype in CELL_TYPES:
                values = stage_frame.loc[stage_frame["celltype"].eq(celltype)]
                if values.empty:
                    continue
                rows.append(
                    {
                        "stage": stage,
                        "method": method,
                        "celltype": celltype,
                        "n_cells": int(len(values)),
                        "stage_growth_mean": stage_mean,
                        "stage_growth_sd": stage_sd,
                        "mean_growth_raw": float(values[column].mean()),
                        "mean_growth_centered": float(values[column].mean() - stage_mean),
                        "mean_growth_z": float(values["growth_z"].mean()),
                        "negative_fraction_raw": float((values[column] < 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def draw(summary: pd.DataFrame, output_dir: Path) -> None:
    configure_plotting()
    y = np.arange(len(CELL_TYPES))[::-1]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.0, 4.55),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.10},
    )

    max_abs = float(np.nanmax(np.abs(summary["mean_growth_z"])))
    limit = np.ceil((max_abs + 0.08) * 5) / 5

    for ax, stage in zip(axes, STAGES):
        stage_summary = summary.loc[summary["stage"].eq(stage)]
        values: dict[str, np.ndarray] = {}
        for method in METHODS:
            indexed = stage_summary.loc[stage_summary["method"].eq(method)].set_index(
                "celltype"
            )
            values[method] = indexed.reindex(CELL_TYPES)["mean_growth_z"].to_numpy()

        for row_y, coati, cyto in zip(y, values["COATI"], values["CytoBridge"]):
            if np.isfinite(coati) and np.isfinite(cyto):
                ax.plot(
                    [coati, cyto],
                    [row_y, row_y],
                    color="#B7B7B7",
                    linewidth=1.0,
                    zorder=1,
                )

        for method, marker in [("COATI", "o"), ("CytoBridge", "D")]:
            ax.scatter(
                values[method],
                y,
                s=27,
                marker=marker,
                color=COLORS[method],
                edgecolor="white",
                linewidth=0.45,
                label=method,
                zorder=3,
            )

        ax.axvline(0, color="#666666", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_title(stage, pad=3, fontweight="normal")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-0.6, len(CELL_TYPES) - 0.4)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        [DISPLAY_LABELS.get(celltype, celltype) for celltype in CELL_TYPES]
    )
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    fig.supxlabel("Mean growth (within-stage z-score)", y=0.025, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.57, 1.005),
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.205, right=0.995, bottom=0.13, top=0.91)

    stem = output_dir / "coati_cytobridge_mean_growth_by_stage"
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        args.output_dir / "coati_cytobridge_mean_growth_by_stage.png",
        args.output_dir / "coati_cytobridge_mean_growth_by_stage.pdf",
        args.output_dir / "coati_cytobridge_mean_growth_by_stage.svg",
        args.output_dir / "coati_cytobridge_mean_growth_by_stage.csv",
    ]
    if not args.overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "Output exists; rerun with --overwrite: "
                + ", ".join(str(path) for path in existing)
            )

    frame = pd.read_csv(args.input)
    required = {"stage", "celltype", *METHODS.values()}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    summary = summarize(frame)
    summary.to_csv(outputs[-1], index=False)
    draw(summary, args.output_dir)


if __name__ == "__main__":
    main()
