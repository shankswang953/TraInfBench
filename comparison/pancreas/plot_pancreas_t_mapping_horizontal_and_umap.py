#!/usr/bin/env python
"""Plot horizontal T cell-type AUROC and E15.5 RNA/ATAC UMAP panels."""

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

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import joblib
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from umap import UMAP

from analyze_pancreas_t_celltype_auroc import (
    DISPLAY,
    MODEL_ORDER,
    TRAIN_T,
    apply_model,
    load_inputs,
    load_model,
)
from plot_pancreas_coati_celltype_transition_matrix import (
    CELL_TYPE_COLORS,
    CELL_TYPE_ORDER,
    DISPLAY_NAMES,
)
from trainfbench_plot_style import NATURE_CUD, apply_nature_rc


ROOT = Path(__file__).resolve().parents[2]
MOSCOT_DATA = Path("external/COATI/moscot/data")
DEFAULT_AUROC = ROOT / "results/pancreas_t_celltype_auroc/t_celltype_auroc.csv"
DEFAULT_UMAP = ROOT / "results/pancreas_strict_loo_e155_dual_umap_5methods"
DEFAULT_OUTPUT = ROOT / "results/pancreas_t_mapping_horizontal_and_umap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auroc", type=Path, default=DEFAULT_AUROC)
    parser.add_argument("--umap-dir", type=Path, default=DEFAULT_UMAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_style() -> None:
    apply_nature_rc(font_size=10.0)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "font.weight": "normal",
            "axes.titlesize": 10.0,
            "axes.titleweight": "normal",
            "axes.labelsize": 10.0,
            "axes.labelweight": "normal",
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.0,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.02}
    fig.savefig(stem.with_suffix(".png"), dpi=600, **options)
    fig.savefig(stem.with_suffix(".pdf"), **options)
    fig.savefig(stem.with_suffix(".svg"), **options)
    plt.close(fig)


def plot_horizontal_auroc(detail: pd.DataFrame, output_dir: Path) -> None:
    loto = detail.loc[detail["model"].eq("Strict LOTO T")].sort_values(
        "one_vs_rest_auroc"
    )
    order = loto["celltype"].tolist()
    positions = np.arange(len(order), dtype=float)
    offsets = {
        "ATAC atlas reference": -0.18,
        "Full T": 0.0,
        "Strict LOTO T": 0.18,
    }
    styles = {
        "ATAC atlas reference": {
            "marker": "x",
            "color": "#111111",
            "facecolors": "none",
            "label": "ATAC reference",
        },
        "Full T": {
            "marker": "o",
            "color": NATURE_CUD["sky_blue"],
            "facecolors": "white",
            "label": "Full T",
        },
        "Strict LOTO T": {
            "marker": "D",
            "color": NATURE_CUD["vermillion"],
            "facecolors": NATURE_CUD["vermillion"],
            "label": "LOTO T",
        },
    }
    fig, ax = plt.subplots(figsize=(7.90, 3.35), facecolor="white")
    for model in MODEL_ORDER:
        values = (
            detail.loc[detail["model"].eq(model)]
            .set_index("celltype")
            .loc[order, "one_vs_rest_auroc"]
            .to_numpy(float)
        )
        style = styles[model]
        scatter_options = {
            "marker": style["marker"],
            "s": 30 if model != "ATAC atlas reference" else 34,
            "linewidths": 1.1,
            "label": style["label"],
            "zorder": 3,
        }
        if model == "ATAC atlas reference":
            scatter_options["color"] = style["color"]
        else:
            scatter_options["edgecolors"] = style["color"]
            scatter_options["facecolors"] = style["facecolors"]
        ax.scatter(positions + offsets[model], values, **scatter_options)
    macro = float(loto["one_vs_rest_auroc"].mean())
    ax.text(
        0.995,
        0.035,
        f"LOTO T macro AUROC = {macro:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.0,
    )
    ax.set_title("Cross-fitted RNA-to-ATAC mapping by cell type", pad=4)
    ax.set_ylabel("One-vs-rest AUROC")
    ax.set_xticks(
        positions,
        [DISPLAY.get(name, DISPLAY_NAMES.get(name, name)) for name in order],
        rotation=52,
        ha="right",
        rotation_mode="anchor",
    )
    observed = detail["one_vs_rest_auroc"].to_numpy(float)
    lower = max(0.5, float(observed.min()) - 0.025)
    ax.set_ylim(lower, 1.005)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.grid(axis="y", color="0.88", linewidth=0.55)
    ax.tick_params(width=0.8, length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=3,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.0,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.085, right=0.995, top=0.82, bottom=0.38)
    save_figure(fig, output_dir / "pancreas_t_celltype_auroc_horizontal")


def robust_limits(points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    lower = np.quantile(points, 0.001, axis=0)
    upper = np.quantile(points, 0.999, axis=0)
    span = np.maximum(upper - lower, 1e-6)
    return (
        (float(lower[0] - 0.04 * span[0]), float(upper[0] + 0.04 * span[0])),
        (float(lower[1] - 0.04 * span[1]), float(upper[1] + 0.04 * span[1])),
    )


def draw_umap(
    ax: plt.Axes,
    coordinates: np.ndarray,
    labels: np.ndarray,
    title: str,
    limits: tuple[tuple[float, float], tuple[float, float]],
    permutation: np.ndarray,
) -> None:
    colors = np.asarray([CELL_TYPE_COLORS[label] for label in labels], dtype=object)
    ax.scatter(
        coordinates[permutation, 0],
        coordinates[permutation, 1],
        s=0.75,
        c=colors[permutation],
        alpha=0.78,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, pad=2)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_umap_panels(
    umap_dir: Path,
    output_dir: Path,
    batch_size: int,
    seed: int,
) -> None:
    device = torch.device("cpu")
    rna, _, labels = load_inputs()
    loto_model = load_model(TRAIN_T / "T_FiLM_poissonvi_LOTO_holdout_1.pt", device)
    predicted_atac = apply_model(loto_model, rna, 1.0, device, batch_size)

    keys = ("time_0", "time_1", "time_2")
    rna_scale = float(
        torch.load(
            MOSCOT_DATA / "primal_norm_params.pt",
            map_location="cpu",
            weights_only=False,
        )["scale"]
    )
    atac_scale = float(
        torch.load(
            MOSCOT_DATA / "secondary_norm_params_poissonvi.pt",
            map_location="cpu",
            weights_only=False,
        )["scale"]
    )
    with np.load(MOSCOT_DATA / "rna_time_data.npz", allow_pickle=True) as archive:
        all_rna = np.concatenate(
            [np.asarray(archive[key], dtype=np.float32) for key in keys], axis=0
        ) / rna_scale
        rna_lengths = [len(archive[key]) for key in keys]
    with np.load(
        MOSCOT_DATA / "atac_poissonvi_time_data.npz", allow_pickle=True
    ) as archive:
        all_atac = np.concatenate(
            [np.asarray(archive[key], dtype=np.float32) for key in keys], axis=0
        ) / atac_scale
        atac_lengths = [len(archive[key]) for key in keys]
    if rna_lengths != atac_lengths:
        raise ValueError("RNA and ATAC stages are not paired")
    stage_index = np.concatenate(
        [np.full(length, index, dtype=np.int8) for index, length in enumerate(rna_lengths)]
    )
    heldout = stage_index == 1
    umap_options = {
        "n_components": 2,
        "n_neighbors": 15,
        "min_dist": 0.8,
        "metric": "euclidean",
        "random_state": seed,
        "transform_seed": seed,
        "transform_mode": "embedding",
        "low_memory": True,
    }
    print(f"[UMAP] fitting RNA atlas {all_rna.shape}", flush=True)
    rna_umap_model = UMAP(**umap_options)
    all_rna_coordinates = np.asarray(
        rna_umap_model.fit_transform(all_rna), dtype=np.float32
    )
    print(f"[UMAP] fitting ATAC atlas {all_atac.shape}", flush=True)
    atac_umap_model = UMAP(**umap_options)
    all_atac_coordinates = np.asarray(
        atac_umap_model.fit_transform(all_atac), dtype=np.float32
    )
    rna_coordinates = all_rna_coordinates[heldout]
    atac_coordinates = all_atac_coordinates[heldout]
    if len(rna_coordinates) != len(labels) or len(atac_coordinates) != len(labels):
        raise ValueError("Held-out UMAP coordinates do not align with E15.5 labels")
    predicted_coordinates = np.asarray(
        atac_umap_model.transform(predicted_atac), dtype=np.float32
    )

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(labels))
    rna_limits = robust_limits(rna_coordinates)
    atac_limits = robust_limits(
        np.concatenate([atac_coordinates, predicted_coordinates], axis=0)
    )
    fig = plt.figure(figsize=(7.90, 2.65), facecolor="white")
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=(1.0, 1.0, 1.0, 1.32),
        wspace=0.10,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    draw_umap(
        axes[0], rna_coordinates, labels, "RNA space", rna_limits, permutation
    )
    draw_umap(
        axes[1], atac_coordinates, labels, "ATAC space", atac_limits, permutation
    )
    draw_umap(
        axes[2],
        predicted_coordinates,
        labels,
        "Predicted ATAC",
        atac_limits,
        permutation,
    )
    legend_ax = fig.add_subplot(grid[0, 3])
    legend_ax.axis("off")
    handles = [
        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=3.6,
            markerfacecolor=CELL_TYPE_COLORS[celltype],
            markeredgewidth=0,
            label=DISPLAY_NAMES[celltype],
        )
        for celltype in CELL_TYPE_ORDER
    ]
    legend_ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        frameon=False,
        ncol=2,
        columnspacing=0.75,
        handletextpad=0.25,
        borderaxespad=0,
        labelspacing=0.35,
    )
    fig.subplots_adjust(left=0.01, right=0.995, top=0.95, bottom=0.03)
    save_figure(fig, output_dir / "pancreas_e155_rna_atac_predicted_atac_umap")
    np.savez_compressed(
        output_dir / "pancreas_e155_t_umap_coordinates.npz",
        labels=labels,
        rna_umap=rna_coordinates,
        atac_umap=atac_coordinates,
        predicted_atac_umap=predicted_coordinates,
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} is not empty; pass --overwrite to replace files"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    detail = pd.read_csv(args.auroc)
    plot_horizontal_auroc(detail, args.output_dir)
    plot_umap_panels(args.umap_dir, args.output_dir, args.batch_size, args.seed)
    manifest = {
        "task": "Pancreas cross-fitted T cell-type AUROC and UMAP display",
        "stage": "held-out E15.5",
        "T": str(TRAIN_T / "T_FiLM_poissonvi_LOTO_holdout_1.pt"),
        "rna_umap": "E15.5 coordinates from UMAP fitted on all observed RNA stages",
        "atac_umap": "E15.5 coordinates from UMAP fitted on all observed ATAC stages",
        "predicted_atac": "strict-LOTO T(RNA E15.5) transformed by the same ATAC UMAP",
        "celltype_color": "paired E15.5 cell-type annotation",
        "figure_width_inches": 7.90,
        "font": "Arial 10 pt, normal weight",
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
