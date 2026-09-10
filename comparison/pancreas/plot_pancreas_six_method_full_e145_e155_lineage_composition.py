#!/usr/bin/env python
"""Compare six full-trained methods on E14.5-to-E15.5 composition change."""

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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import plot_pancreas_six_method_full_e155_e165_lineage_composition as late
from plot_pancreas_coati_celltype_transition_matrix import (
    CELL_TYPE_ORDER,
    soft_label_probabilities,
)
from trainfbench_plot_style import METHOD_STYLES


ROOT = Path(__file__).resolve().parents[2]
MOSCOT_DATA = Path("external/COATI/moscot/data")
DEFAULT_INPUT = (
    ROOT
    / "results/pancreas_full_moscot_transition_trends_15nn_20k_cy05"
    / "full_e145_to_e155_endpoints.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/pancreas_six_method_full_e145_e155_lineage_composition_15nn_20k_cy05"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def calculate(input_path: Path, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = np.load(MOSCOT_DATA / "rna_time_data.npz", allow_pickle=False)
    labels = np.load(MOSCOT_DATA / "celltype_sub_by_stage.npz", allow_pickle=True)
    scale = late.load_scale(MOSCOT_DATA / "primal_norm_params.pt")
    target_reference = (
        np.asarray(data["time_1"], dtype=np.float32) / scale
    ).astype(np.float32, copy=False)
    source_labels = np.asarray(labels["time_0"]).astype(str)
    target_labels = np.asarray(labels["time_1"]).astype(str)
    source_fraction = late.observed_fraction(source_labels)
    target_fraction = late.observed_fraction(target_labels)

    long_rows: list[dict[str, object]] = []
    with np.load(input_path, allow_pickle=True) as cache:
        methods = tuple(np.asarray(cache["method_order"]).astype(str))
        if methods != late.METHOD_KEYS:
            raise ValueError(f"Unexpected method order: {methods}")
        for index, method in enumerate(methods):
            endpoint = np.asarray(cache[f"endpoint_{index}"], dtype=np.float32)
            if endpoint.shape[0] != len(source_labels):
                raise ValueError(
                    f"{method}: endpoint/source mismatch "
                    f"{endpoint.shape[0]} != {len(source_labels)}"
                )
            probabilities = soft_label_probabilities(
                endpoint,
                target_reference,
                target_labels,
                k,
            )
            weights = late.endpoint_weights(cache, index, len(source_labels))
            predicted_fraction = weights @ probabilities
            for celltype_index, celltype in enumerate(CELL_TYPE_ORDER):
                long_rows.append(
                    {
                        "method": method,
                        "method_label": late.METHOD_LABELS[method],
                        "weighting": (
                            "native endpoint mass"
                            if f"log_mass_{index}" in cache.files
                            else "uniform particles"
                        ),
                        "celltype": celltype,
                        "observed_source_fraction": source_fraction[celltype_index],
                        "observed_target_fraction": target_fraction[celltype_index],
                        "predicted_target_fraction": predicted_fraction[celltype_index],
                        "observed_delta_pp": 100.0
                        * (target_fraction[celltype_index] - source_fraction[celltype_index]),
                        "predicted_delta_pp": 100.0
                        * (predicted_fraction[celltype_index] - source_fraction[celltype_index]),
                    }
                )

    long = pd.DataFrame(long_rows)
    pooled_counts = pd.Series(
        {
            celltype: int(np.sum(source_labels == celltype))
            + int(np.sum(target_labels == celltype))
            for celltype in late.LINEAGE_CELL_TYPES
        },
        dtype=np.float64,
    )
    cell_number_weights = pooled_counts / pooled_counts.sum()
    error_rows: list[dict[str, object]] = []
    for method in late.METHOD_KEYS:
        selected = (
            long[
                long["method"].eq(method)
                & long["celltype"].isin(late.LINEAGE_CELL_TYPES)
            ]
            .set_index("celltype")
            .loc[list(late.LINEAGE_CELL_TYPES)]
        )
        absolute_error = (
            selected["predicted_delta_pp"] - selected["observed_delta_pp"]
        ).abs()
        error_rows.append(
            {
                "method": method,
                "method_label": late.METHOD_LABELS[method],
                "cell_number_weighted_mae_pp": float(
                    absolute_error.dot(cell_number_weights)
                ),
                "weighting": selected["weighting"].iloc[0],
            }
        )
    return long, pd.DataFrame(error_rows)


def plot_table(long: pd.DataFrame, errors: pd.DataFrame, output_dir: Path) -> None:
    late.style()
    observed = (
        long.drop_duplicates("celltype")
        .set_index("celltype")
        .loc[list(late.LINEAGE_CELL_TYPES), "observed_delta_pp"]
    )
    method_values = {
        method: long[long["method"].eq(method)]
        .set_index("celltype")
        .loc[list(late.LINEAGE_CELL_TYPES), "predicted_delta_pp"]
        for method in late.METHOD_KEYS
    }
    error_values = errors.set_index("method")["cell_number_weighted_mae_pp"]

    rows: list[list[str]] = []
    for celltype in late.LINEAGE_CELL_TYPES:
        rows.append(
            [celltype, f"{observed.loc[celltype]:+.2f}"]
            + [
                f"{method_values[method].loc[celltype]:+.2f}"
                for method in late.METHOD_KEYS
            ]
        )
    rows.append(
        ["Cell-number-weighted\nMAE", "—"]
        + [f"{error_values.loc[method]:.2f}" for method in late.METHOD_KEYS]
    )

    columns = (
        "Cell type",
        "Observed\nΔ",
        "COATI\nbal.",
        "COATI\nunbal.",
        "CytoBridge\nbal.",
        "CytoBridge\nunbal.",
        "TrajectoryNet",
        "MIOFlow",
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.15), facecolor="white")
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=(0.0, 0.015, 1.0, 0.94),
        colWidths=(0.225, 0.10, 0.105, 0.105, 0.125, 0.125, 0.12, 0.095),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    method_colors = [
        METHOD_STYLES[late.STYLE_KEYS[method]].color for method in late.METHOD_KEYS
    ]
    method_colors[late.METHOD_KEYS.index("TrajectoryNet")] = "#7A7200"
    summary_row = len(rows)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#C3C3C3")
        cell.set_linewidth(0.55)
        cell.get_text().set_fontfamily("Arial")
        cell.get_text().set_fontsize(10)
        cell.get_text().set_weight("normal")
        cell.set_facecolor("#EEEEEE" if row in (0, summary_row) else "white")
        if column == 0:
            cell.get_text().set_ha("left")
        if row == 0 and column >= 2:
            cell.get_text().set_color(method_colors[column - 2])
    for column in range(len(columns)):
        table[(summary_row, column)].set_height(
            table[(summary_row, column)].get_height() * 1.08
        )

    ax.set_title(
        "E14.5→E15.5 full-trained composition change (percentage)",
        fontsize=10,
        fontweight="normal",
        pad=1,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01)
    stem = output_dir / "pancreas_six_method_e145_e155_lineage_composition_table"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            stem.with_suffix(f".{suffix}"),
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.035,
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.k <= 0:
        raise ValueError("--k must be positive")
    sentinel = (
        args.output_dir
        / "pancreas_six_method_e145_e155_lineage_composition_table.png"
    )
    if sentinel.exists() and not args.overwrite:
        raise FileExistsError(f"{sentinel} exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    long, errors = calculate(args.input, args.k)
    long.to_csv(args.output_dir / "six_method_composition_change_long.csv", index=False)
    errors.to_csv(
        args.output_dir / "six_method_cell_number_weighted_mae.csv", index=False
    )
    plot_table(long, errors, args.output_dir)
    manifest = {
        "task": "Six-method full E14.5-to-E15.5 lineage composition change",
        "input": str(args.input),
        "start": "all observed E14.5 RNA cells",
        "classifier": f"{args.k}-NN soft voting against observed E15.5 RNA PCA50",
        "methods": list(late.METHOD_KEYS),
        "weighting": {
            "balanced_and_neural_baselines": "uniform particles",
            "unbalanced": "native endpoint mass normalized across the E14.5 cohort",
        },
        "delta": "predicted E15.5 fraction minus observed E14.5 fraction (percentage points)",
        "observed_delta": "observed E15.5 fraction minus observed E14.5 fraction (percentage points)",
        "weighted_mae": (
            "absolute delta error weighted by pooled observed E14.5+E15.5 cell "
            "numbers within the 12 displayed endocrine-lineage states"
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(errors.to_string(index=False))


if __name__ == "__main__":
    main()
