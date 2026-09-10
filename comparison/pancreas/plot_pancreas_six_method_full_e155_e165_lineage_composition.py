#!/usr/bin/env python
"""Compare six full-trained methods on E15.5-to-E16.5 composition change."""

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
import sys


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

from plot_pancreas_coati_celltype_transition_matrix import (  # noqa: E402
    CELL_TYPE_ORDER,
    soft_label_probabilities,
)
from trainfbench_plot_style import METHOD_STYLES  # noqa: E402


MOSCOT_DATA = Path("external/COATI/moscot/data")
DEFAULT_INPUT = (
    ROOT
    / "results/pancreas_full_e155_to_e165_transition_trends_15nn_20k_cy05"
    / "full_e155_to_e165_endpoints.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/pancreas_six_method_full_e155_e165_lineage_composition_15nn_20k_cy05"
)

LINEAGE_CELL_TYPES = (
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

METHOD_KEYS = (
    "COATI balanced C_y=0.5",
    "COATI unbalanced C_y=0.5",
    "CytoBridge balanced",
    "CytoBridge unbalanced",
    "TrajectoryNet",
    "MIOFlow GAGA10",
)
METHOD_LABELS = {
    "COATI balanced C_y=0.5": "COATI bal.",
    "COATI unbalanced C_y=0.5": "COATI unbal.",
    "CytoBridge balanced": "CytoBridge bal.",
    "CytoBridge unbalanced": "CytoBridge unbal.",
    "TrajectoryNet": "TrajectoryNet",
    "MIOFlow GAGA10": "MIOFlow",
}
STYLE_KEYS = {
    "COATI balanced C_y=0.5": "COATI balanced",
    "COATI unbalanced C_y=0.5": "COATI unbalanced",
    "CytoBridge balanced": "CytoBridge balanced",
    "CytoBridge unbalanced": "CytoBridge unbalanced",
    "TrajectoryNet": "TrajectoryNet",
    "MIOFlow GAGA10": "MIOFlow",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_scale(path: Path) -> float:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return float(np.asarray(payload["scale"]).reshape(-1)[0])


def endpoint_weights(cache: np.lib.npyio.NpzFile, index: int, n: int) -> np.ndarray:
    key = f"log_mass_{index}"
    if key not in cache.files:
        return np.full(n, 1.0 / n, dtype=np.float64)
    log_mass = np.asarray(cache[key], dtype=np.float64).reshape(-1)
    if log_mass.shape != (n,) or not np.isfinite(log_mass).all():
        raise ValueError(f"Invalid {key}: shape={log_mass.shape}")
    mass = np.exp(log_mass - np.max(log_mass))
    return mass / mass.sum()


def observed_fraction(labels: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.mean(labels == cell_type) for cell_type in CELL_TYPE_ORDER],
        dtype=np.float64,
    )


def calculate(input_path: Path, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = np.load(MOSCOT_DATA / "rna_time_data.npz", allow_pickle=False)
    scale = load_scale(MOSCOT_DATA / "primal_norm_params.pt")
    target_reference = (
        np.asarray(data["time_2"], dtype=np.float32) / scale
    ).astype(np.float32, copy=False)

    long_rows: list[dict[str, object]] = []
    with np.load(input_path, allow_pickle=True) as cache:
        methods = tuple(np.asarray(cache["method_order"]).astype(str))
        if methods != METHOD_KEYS:
            raise ValueError(f"Unexpected method order: {methods}")
        source_labels = np.asarray(cache["source_labels"]).astype(str)
        target_labels = np.asarray(cache["target_labels"]).astype(str)
        source_fraction = observed_fraction(source_labels)
        target_fraction = observed_fraction(target_labels)

        for index, method in enumerate(methods):
            endpoint = np.asarray(cache[f"endpoint_{index}"], dtype=np.float32)
            probabilities = soft_label_probabilities(
                endpoint,
                target_reference,
                target_labels,
                k,
            )
            weights = endpoint_weights(cache, index, len(source_labels))
            predicted_fraction = weights @ probabilities
            for celltype_index, celltype in enumerate(CELL_TYPE_ORDER):
                long_rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
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
            for celltype in LINEAGE_CELL_TYPES
        },
        dtype=np.float64,
    )
    cell_number_weights = pooled_counts / pooled_counts.sum()
    error_rows: list[dict[str, object]] = []
    for method in METHOD_KEYS:
        selected = (
            long[
                long["method"].eq(method)
                & long["celltype"].isin(LINEAGE_CELL_TYPES)
            ]
            .set_index("celltype")
            .loc[list(LINEAGE_CELL_TYPES)]
        )
        absolute_error = (
            selected["predicted_delta_pp"] - selected["observed_delta_pp"]
        ).abs()
        error_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "cell_number_weighted_mae_pp": float(
                    absolute_error.dot(cell_number_weights)
                ),
                "weighting": selected["weighting"].iloc[0],
            }
        )
    return long, pd.DataFrame(error_rows)


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "font.weight": "normal",
            "axes.titlesize": 10.0,
            "axes.titleweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_table(long: pd.DataFrame, errors: pd.DataFrame, output_dir: Path) -> None:
    style()
    observed = (
        long.drop_duplicates("celltype")
        .set_index("celltype")
        .loc[list(LINEAGE_CELL_TYPES), "observed_delta_pp"]
    )
    method_values = {
        method: long[long["method"].eq(method)]
        .set_index("celltype")
        .loc[list(LINEAGE_CELL_TYPES), "predicted_delta_pp"]
        for method in METHOD_KEYS
    }
    error_values = errors.set_index("method")["cell_number_weighted_mae_pp"]

    rows: list[list[str]] = []
    for celltype in LINEAGE_CELL_TYPES:
        rows.append(
            [celltype, f"{observed.loc[celltype]:+.2f}"]
            + [f"{method_values[method].loc[celltype]:+.2f}" for method in METHOD_KEYS]
        )
    rows.append(
        ["Cell-number-weighted\nMAE", "—"]
        + [f"{error_values.loc[method]:.2f}" for method in METHOD_KEYS]
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
        METHOD_STYLES[STYLE_KEYS[method]].color for method in METHOD_KEYS
    ]
    method_colors[METHOD_KEYS.index("TrajectoryNet")] = "#7A7200"
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
        "E15.5→E16.5 full-trained composition change (pp)",
        fontsize=10,
        fontweight="normal",
        pad=1,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01)
    stem = output_dir / "pancreas_six_method_e155_e165_lineage_composition_table"
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
        args.output_dir / "pancreas_six_method_e155_e165_lineage_composition_table.png"
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
        "task": "Six-method E15.5-to-E16.5 full-trained composition-change comparison",
        "source": "same observed E15.5 cells for every method",
        "target_readout": f"{args.k}-NN soft voting against observed E16.5 RNA PCA50",
        "balanced_weighting": "uniform particles",
        "unbalanced_weighting": "native endpoint mass normalized over the fresh E15.5 cohort",
        "observed_change": "observed E16.5 fraction minus observed E15.5 fraction",
        "predicted_change": "predicted E16.5 fraction minus observed E15.5 fraction",
        "reported_celltypes": list(LINEAGE_CELL_TYPES),
        "mae_weighting": "pooled observed E15.5+E16.5 cell counts within the 12 reported cell types",
        "methods": list(METHOD_KEYS),
        "input": str(args.input.resolve()),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(errors.to_string(index=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
