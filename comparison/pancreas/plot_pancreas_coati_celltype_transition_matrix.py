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
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors


CELL_TYPE_ORDER = (
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

DISPLAY_NAMES = {
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

CELL_TYPE_COLORS = {
    "Mat. Acinar": "#b2df8a",
    "Imm. Acinar": "#ffd92f",
    "Prlf. Ductal": "#f781bf",
    "Ductal": "#e41a1c",
    "Ngn3 low": "#b3de69",
    "Ngn3 high cycling": "#66a61e",
    "Ngn3 high": "#fb9a99",
    "Eps. progenitors": "#c77cff",
    "Fev+": "#df65b0",
    "Fev+ Alpha": "#66c2a5",
    "Fev+ Beta": "#00bfc4",
    "Fev+ Delta": "#4dbbd5",
    "Alpha": "#377eb8",
    "Beta": "#ff7f00",
    "Delta": "#1b9e77",
    "Epsilon": "#8c564b",
}


def soft_label_probabilities(
    query: np.ndarray,
    reference: np.ndarray,
    reference_labels: np.ndarray,
    k: int,
) -> np.ndarray:
    k_effective = min(k, len(reference))
    neighbors = NearestNeighbors(n_neighbors=k_effective, n_jobs=-1)
    neighbors.fit(reference)
    indices = neighbors.kneighbors(query, return_distance=False)
    votes = reference_labels[indices]
    return np.column_stack(
        [(votes == cell_type).mean(axis=1) for cell_type in CELL_TYPE_ORDER]
    )


def aggregate_by_source(
    per_cell_probabilities: np.ndarray, source_labels: np.ndarray
) -> np.ndarray:
    matrix = np.zeros((len(CELL_TYPE_ORDER), len(CELL_TYPE_ORDER)), dtype=np.float64)
    for row, source_type in enumerate(CELL_TYPE_ORDER):
        mask = source_labels == source_type
        if not np.any(mask):
            raise ValueError(f"No source cells found for {source_type!r}")
        matrix[row] = per_cell_probabilities[mask].mean(axis=0)
    return matrix


def save_tables(
    matrix: np.ndarray,
    per_cell_probabilities: np.ndarray,
    source_labels: np.ndarray,
    stage_tag: str,
    output_dir: Path,
) -> None:
    wide = pd.DataFrame(matrix, index=CELL_TYPE_ORDER, columns=CELL_TYPE_ORDER)
    wide.index.name = "source_cell_type"
    wide.to_csv(output_dir / f"{stage_tag}_transition_matrix.csv")

    long = (
        wide.rename_axis(columns="target_cell_type")
        .stack()
        .rename("transition_probability")
        .reset_index()
    )
    long.to_csv(output_dir / f"{stage_tag}_transition_matrix_long.csv", index=False)

    top_rows = []
    for row, source_type in enumerate(CELL_TYPE_ORDER):
        order = np.argsort(matrix[row])[::-1]
        top_rows.append(
            {
                "source_cell_type": source_type,
                "n_source_cells": int(np.sum(source_labels == source_type)),
                "top_target": CELL_TYPE_ORDER[int(order[0])],
                "top_probability": float(matrix[row, order[0]]),
                "second_target": CELL_TYPE_ORDER[int(order[1])],
                "second_probability": float(matrix[row, order[1]]),
                "margin": float(matrix[row, order[0]] - matrix[row, order[1]]),
            }
        )
    pd.DataFrame(top_rows).to_csv(
        output_dir / f"{stage_tag}_top_transitions.csv", index=False
    )

    per_cell = pd.DataFrame(
        {
            "source_index": np.arange(len(source_labels)),
            "source_cell_type": source_labels,
            **{
                f"p_{cell_type}": per_cell_probabilities[:, col]
                for col, cell_type in enumerate(CELL_TYPE_ORDER)
            },
        }
    )
    per_cell.to_csv(
        output_dir / f"{stage_tag}_per_cell_soft_probabilities.csv.gz", index=False
    )


def plot_matrix(
    matrix: np.ndarray,
    source_stage: str,
    target_stage: str,
    stage_tag: str,
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.linewidth": 0.8,
        }
    )
    fig = plt.figure(figsize=(8.4, 8.8), constrained_layout=False)
    grid = fig.add_gridspec(
        4,
        3,
        width_ratios=[0.18, 1.0, 0.055],
        height_ratios=[0.055, 1.0, 0.34, 0.075],
        left=0.27,
        right=0.94,
        bottom=0.10,
        top=0.90,
        wspace=0.015,
        hspace=0.015,
    )
    top_ax = fig.add_subplot(grid[0, 1])
    left_ax = fig.add_subplot(grid[1, 0])
    ax = fig.add_subplot(grid[1, 1])
    cax = fig.add_subplot(grid[3, 1])

    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    labels = [DISPLAY_NAMES[cell_type] for cell_type in CELL_TYPE_ORDER]
    ticks = np.arange(len(CELL_TYPE_ORDER))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=58, ha="right", rotation_mode="anchor")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Target cell type ({target_stage})", labelpad=8)
    ax.set_ylabel(f"Source cell type ({source_stage})", labelpad=8)
    ax.tick_params(length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_color("#303030")
        spine.set_linewidth(0.8)

    strip_colors = [CELL_TYPE_COLORS[cell_type] for cell_type in CELL_TYPE_ORDER]
    strip_cmap = ListedColormap(strip_colors)
    top_ax.imshow(np.arange(len(CELL_TYPE_ORDER))[None, :], cmap=strip_cmap, aspect="auto")
    top_ax.set_axis_off()
    left_ax.imshow(np.arange(len(CELL_TYPE_ORDER))[:, None], cmap=strip_cmap, aspect="auto")
    left_ax.set_axis_off()

    colorbar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0.0, 0.5, 1.0])
    colorbar.set_label("Cell-type transition probability", labelpad=2)
    colorbar.ax.tick_params(length=2, pad=2)
    fig.suptitle(
        f"COATI balanced: {source_stage} → {target_stage}",
        x=0.61,
        y=0.955,
        fontsize=12,
        fontweight="bold",
    )

    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(
            output_dir / f"{stage_tag}_transition_matrix.{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build moscot-style cell-type transition matrices from a COATI trajectory."
    )
    parser.add_argument(
        "--coati-root",
        type=Path,
        default=Path("external/COATI/moscot"),
    )
    parser.add_argument(
        "--primary-trajectory",
        type=Path,
        default=Path(
            "results/coati_moscot_balanced_seed0_cy0p5_iter20000_fate_quality/"
            "primary_trajectory_s0_a0.5_iter20000.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/coati_moscot_balanced_seed0_cy0p5_iter20000_fate_quality/"
            "celltype_transition_matrix"
        ),
    )
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sync-weight", type=float, default=0.5)
    parser.add_argument("--training-iterations", type=int, default=20000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty {args.output_dir}; pass --overwrite explicitly."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_path = args.coati_root / "data/rna_time_data.npz"
    labels_path = args.coati_root / "data/celltype_sub_by_stage.npz"
    norm_path = args.coati_root / "data/primal_norm_params.pt"
    for path in (args.primary_trajectory, data_path, labels_path, norm_path):
        if not path.exists():
            raise FileNotFoundError(path)

    trajectory = np.asarray(
        torch.load(args.primary_trajectory, map_location="cpu"), dtype=np.float32
    )
    data = np.load(data_path)
    labels = np.load(labels_path, allow_pickle=True)
    scale = float(torch.load(norm_path, map_location="cpu")["scale"])
    normalized = {
        key: np.asarray(data[key], dtype=np.float32) / scale
        for key in ("time_0", "time_1", "time_2")
    }
    cell_types = {
        key: labels[key].astype(str) for key in ("time_0", "time_1", "time_2")
    }

    if trajectory.shape != (21, len(normalized["time_0"]), normalized["time_0"].shape[1]):
        raise ValueError(f"Unexpected trajectory shape: {trajectory.shape}")
    initial_error = float(np.max(np.abs(trajectory[0] - normalized["time_0"])))
    if initial_error > 1e-6:
        raise ValueError(f"Trajectory/data alignment failed: max abs error={initial_error}")
    observed_types = set().union(*(set(values) for values in cell_types.values()))
    if observed_types != set(CELL_TYPE_ORDER):
        raise ValueError(
            f"Cell-type mismatch: missing={set(CELL_TYPE_ORDER) - observed_types}, "
            f"extra={observed_types - set(CELL_TYPE_ORDER)}"
        )

    stage_specs = (
        ("e145_to_e155", "E14.5", "E15.5", 10, "time_1"),
        ("e145_to_e165", "E14.5", "E16.5", 20, "time_2"),
    )
    row_sum_errors: dict[str, float] = {}
    for stage_tag, source_stage, target_stage, trajectory_index, target_key in stage_specs:
        probabilities = soft_label_probabilities(
            trajectory[trajectory_index],
            normalized[target_key],
            cell_types[target_key],
            args.k,
        )
        matrix = aggregate_by_source(probabilities, cell_types["time_0"])
        row_sum_error = float(np.max(np.abs(matrix.sum(axis=1) - 1.0)))
        if row_sum_error > 1e-10:
            raise RuntimeError(f"{stage_tag} rows do not sum to one: {row_sum_error}")
        row_sum_errors[stage_tag] = row_sum_error
        save_tables(
            matrix,
            probabilities,
            cell_types["time_0"],
            stage_tag,
            args.output_dir,
        )
        plot_matrix(
            matrix,
            source_stage,
            target_stage,
            stage_tag,
            args.output_dir,
        )

    manifest = {
        "method": "COATI balanced",
        "seed": args.seed,
        "sync_weight_C_y": args.sync_weight,
        "training_iterations": args.training_iterations,
        "primary_trajectory": str(args.primary_trajectory.resolve()),
        "trajectory_shape": list(trajectory.shape),
        "trajectory_initial_frame_max_abs_error": initial_error,
        "space": "normalized RNA PCA50",
        "k": args.k,
        "cell_type_order": list(CELL_TYPE_ORDER),
        "matrices": {
            "e145_to_e155": {
                "source": "observed E14.5 cell-type labels",
                "target": "kNN soft labels of COATI trajectory at t=1.0 against observed E15.5",
            },
            "e145_to_e165": {
                "source": "observed E14.5 cell-type labels",
                "target": "kNN soft labels of COATI trajectory at t=2.0 against observed E16.5",
            },
        },
        "row_normalization": "Each source-cell-type row sums to 1.",
        "row_sum_max_abs_errors": row_sum_errors,
        "comparison_note": (
            "The moscot paper aggregates a probabilistic OT coupling. COATI instead "
            "produces deterministic continuous trajectories, so this comparable proxy "
            "assigns uniform kNN soft cell-type probabilities to each predicted endpoint "
            "and averages them within observed source cell types."
        ),
        "moscot_reference": {
            "paper": "https://www.nature.com/articles/s41586-024-08453-2",
            "tutorial": (
                "https://moscot.readthedocs.io/en/latest/notebooks/tutorials/"
                "200_temporal_problem.html"
            ),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote transition matrices to {args.output_dir}")
    for stage_tag, error in row_sum_errors.items():
        print(f"{stage_tag}: max row-sum error={error:.3e}")


if __name__ == "__main__":
    main()
