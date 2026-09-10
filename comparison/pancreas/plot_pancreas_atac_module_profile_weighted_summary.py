#!/usr/bin/env python
"""Summarize edge-specific pancreas ATAC-module deviations for the main text."""

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trainfbench_plot_style import method_style


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = (
    ROOT / "results/pancreas_transition_target_atac_module_support_15nn_20k_cy05"
)
METHOD_ORDER = (
    "COATI bal.",
    "COATI unbal.",
    "CytoBridge bal.",
    "CytoBridge unbal.",
    "TrajectoryNet",
    "MIOFlow",
)
PRIMARY_WEIGHTING = "matched transition-matrix weighting"
MIN_TRANSITION = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def summarize(profile: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    selected = profile[profile["weighting"].eq(PRIMARY_WEIGHTING)].copy()
    selected = selected.merge(
        counts[["target_cell_type", "n_cells"]],
        on="target_cell_type",
        how="left",
        validate="many_to_one",
    )
    if selected["n_cells"].isna().any():
        missing = selected.loc[selected["n_cells"].isna(), "target_cell_type"].unique()
        raise ValueError(f"Missing observed E15.5 cell counts for {missing.tolist()}")
    selected["supported"] = (
        selected["rna_transition_probability"].ge(MIN_TRANSITION)
        & selected["observed_target_atac_profile_srmse"].notna()
    )

    records: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        group = selected[selected["method_label"].eq(method)].copy()
        if group.empty:
            raise ValueError(f"No rows found for {method}")
        kept = group[group["supported"]]
        if kept.empty:
            weighted_srmse = np.nan
        else:
            weighted_srmse = float(
                np.average(
                    kept["observed_target_atac_profile_srmse"],
                    weights=kept["n_cells"],
                )
            )
        total_weight = float(group["n_cells"].sum())
        supported_weight = float(kept["n_cells"].sum())
        records.append(
            {
                "method": method,
                "cell_number_weighted_srmse": weighted_srmse,
                "cell_number_weighted_supported_edge_coverage": (
                    supported_weight / total_weight
                ),
                "supported_edges": int(group["supported"].sum()),
                "total_edges": int(len(group)),
                "supported_edge_cell_weight": supported_weight,
                "total_edge_cell_weight": total_weight,
                "all_edge_cell_number_weighted_srmse": float(
                    np.average(
                        group["observed_target_atac_profile_srmse"],
                        weights=group["n_cells"],
                    )
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "font.weight": "normal",
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def plot(summary: pd.DataFrame, output_dir: Path) -> None:
    configure_plotting()
    values = summary.set_index("method").loc[list(METHOD_ORDER)]
    positions = np.arange(len(METHOD_ORDER))
    colors = [method_style(method).color for method in METHOD_ORDER]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(3.85, 2.55),
        sharey=True,
        gridspec_kw={"width_ratios": (1.18, 0.82)},
    )
    error = values["cell_number_weighted_srmse"].to_numpy(dtype=float)
    coverage = 100.0 * values[
        "cell_number_weighted_supported_edge_coverage"
    ].to_numpy(dtype=float)

    axes[0].barh(positions, error, color=colors, height=0.58)
    axes[0].set_yticks(positions, METHOD_ORDER)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.0, max(0.40, float(np.nanmax(error)) + 0.045))
    axes[0].set_xticks((0.0, 0.2, 0.4))
    axes[0].set_title("Weighted sRMSE ↓", pad=4)
    axes[0].set_xlabel("Standardized RMSE")
    for y, value in zip(positions, error, strict=True):
        axes[0].text(value + 0.008, y, f"{value:.3f}", va="center", ha="left")

    axes[1].barh(positions, coverage, color=colors, height=0.58)
    axes[1].set_xlim(0.0, 108.0)
    axes[1].set_xticks((0, 50, 100))
    axes[1].set_title("Edge coverage ↑", pad=4)
    axes[1].set_xlabel("Cell-weighted (%)")
    for y, value in enumerate(coverage):
        axes[1].text(
            value - 2.0,
            y,
            f"{value:.1f}",
            va="center",
            ha="right",
        )

    for axis in axes:
        axis.grid(axis="x", color="#E1E1E1", linewidth=0.6, zorder=0)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0, pad=3)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    fig.subplots_adjust(left=0.35, right=0.985, top=0.88, bottom=0.23, wspace=0.22)

    stem = output_dir / "observed_target_atac_profile_weighted_summary"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            stem.with_suffix(f".{extension}"),
            dpi=600 if extension == "png" else None,
            bbox_inches="tight",
            pad_inches=0.035,
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    profile_path = args.input_dir / "observed_target_atac_profile_similarity_by_edge.csv"
    counts_path = args.input_dir / "observed_e155_target_atac_module_specificity.csv"
    output_csv = args.output_dir / "observed_target_atac_profile_weighted_summary.csv"
    output_png = args.output_dir / "observed_target_atac_profile_weighted_summary.png"
    if (output_csv.exists() or output_png.exists()) and not args.overwrite:
        raise FileExistsError("Summary outputs exist; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize(pd.read_csv(profile_path), pd.read_csv(counts_path))
    summary.to_csv(output_csv, index=False)
    plot(summary, args.output_dir)
    (args.output_dir / "observed_target_atac_profile_weighted_summary_manifest.json").write_text(
        json.dumps(
            {
                "profile_input": str(profile_path.resolve()),
                "cell_count_input": str(counts_path.resolve()),
                "weighting": PRIMARY_WEIGHTING,
                "minimum_RNA_transition_probability": MIN_TRANSITION,
                "mean_definition": (
                    "Mean standardized RMSE over supported prespecified edges, "
                    "weighted by the observed E15.5 count of each edge's target "
                    "cell type. Because every target has two prespecified edges, "
                    "this equals averaging the two edge errors within each target "
                    "and then weighting targets by observed cell number."
                ),
                "coverage_definition": (
                    "Fraction of total target-cell edge weight contributed by edges "
                    "with RNA transition probability >= 1%."
                ),
                "appendix_figure": "observed_target_atac_profile_distance_matched_weighting",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
