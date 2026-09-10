"""Plot the paired Gastrulation RNA/ATAC atlases by canonical cell type.

The two panels use the exact fixed UMAP coordinates underlying the trajectory
and growth-distribution figures.  Cell-type colours come only from the shared
TraInfBench registry so the appendix reference remains consistent with every
other Gastrulation figure.
"""
from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
for cache_variable in ("MPLCONFIGDIR", "NUMBA_CACHE_DIR", "XDG_CACHE_HOME"):
    Path(os.environ[cache_variable]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
TRAINF_ROOT = ROOT / "external"
GASTRULATION = TRAINF_ROOT / "COATI" / "Gastrulation"
DATA = GASTRULATION / "data"
OUTPUT_DIR = (
    ROOT
    / "results"
    / "publication_figures"
    / "gastrulation_rna_atac_celltype_reference"
)

for path in (ROOT / "common",):
    if path.exists():
        sys.path.insert(0, str(path))

from plot_gastrulation_coati_dual_umap_flow import (  # noqa: E402
    TIME_KEYS,
    fixed_umap_interpolator,
    load_celltypes,
    load_concat_npz,
    load_scale,
    plot_limits,
)
from evaluate_gastrulation_full_cmcc import (  # noqa: E402
    _apply_t,
    _load_t_model,
)
from trainfbench_plot_style import (  # noqa: E402
    GASTRULATION_CELLTYPE_COLORS,
    GASTRULATION_CELLTYPE_GROUPS,
    apply_nature_rc,
    gastrulation_celltype_color,
)


def ordered_celltypes(observed: np.ndarray) -> list[str]:
    present = set(np.asarray(observed, dtype=str))
    order: list[str] = []
    for group in GASTRULATION_CELLTYPE_GROUPS.values():
        order.extend(label for label in group if label in present)
    order.extend(sorted(present.difference(order)))
    return order


def draw_atlas(
    ax: plt.Axes,
    embedding: np.ndarray,
    celltypes: np.ndarray,
    order: list[str],
    title: str,
    *,
    limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> None:
    counts = {label: int(np.sum(celltypes == label)) for label in order}
    # Large groups are drawn first so rare groups remain visible.
    plot_order = sorted(order, key=lambda label: counts[label], reverse=True)
    for label in plot_order:
        member = celltypes == label
        ax.scatter(
            embedding[member, 0],
            embedding[member, 1],
            s=0.75,
            color=gastrulation_celltype_color(label),
            alpha=0.78 if label != "Unannotated" else 0.20,
            linewidths=0,
            rasterized=True,
        )
    xlim, ylim = plot_limits(embedding) if limits is None else limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, fontweight="normal", pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def predict_atac_from_paired_rna(
    *,
    rna_path: Path,
    rna_scale: float,
    atac_scale: float,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    physical_times = (0.0, 1.0, 2.0, 2.5)
    t_model = _load_t_model(device)
    with np.load(rna_path, allow_pickle=True) as rna_by_time:
        mapped = [
            _apply_t(
                t_model,
                np.asarray(rna_by_time[key], dtype=np.float32) / rna_scale,
                physical_time,
                device,
                batch_size,
            )
            * atac_scale
            for key, physical_time in zip(TIME_KEYS, physical_times)
        ]
    return np.concatenate(mapped, axis=0).astype(np.float32, copy=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_nature_rc(font_size=10.0)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
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
            "svg.fonttype": "none",
        }
    )

    rna_latent = load_concat_npz(DATA / "rna_pca_by_time.npz")
    atac_latent = load_concat_npz(DATA / "atac_lsi_by_time_14D.npz")
    celltypes = load_celltypes(DATA / "celltype_sub_by_stage.npz")
    if not (len(rna_latent) == len(atac_latent) == len(celltypes)):
        raise RuntimeError(
            "RNA, ATAC, and cell-type arrays are not aligned: "
            f"{len(rna_latent)}, {len(atac_latent)}, {len(celltypes)}"
        )

    rna_embedding, _ = fixed_umap_interpolator(
        rna_latent,
        DATA / "rna_umap_model.joblib",
    )
    atac_embedding, atac_interpolator = fixed_umap_interpolator(
        atac_latent,
        DATA / "atac_umap_model_14D.joblib",
    )
    rna_scale = load_scale(DATA / "primal_norm_params.pt")
    atac_scale = load_scale(DATA / "secondary_norm_params.pt")
    predicted_atac = predict_atac_from_paired_rna(
        rna_path=DATA / "rna_pca_by_time.npz",
        rna_scale=rna_scale,
        atac_scale=atac_scale,
        device=torch.device("cpu"),
    )
    predicted_atac_embedding = atac_interpolator.predict(
        predicted_atac
    ).astype(np.float32, copy=False)
    if len(predicted_atac_embedding) != len(celltypes):
        raise RuntimeError(
            "Predicted ATAC and cell-type arrays are not aligned: "
            f"{len(predicted_atac_embedding)}, {len(celltypes)}"
        )
    order = ordered_celltypes(celltypes)

    # Match the Pancreas reference width and panel typography.  The 33-entry
    # legend is split by biological groups so the text stays at a true 10 pt:
    # progenitor/neural groups at right, remaining groups below.
    right_order = [
        label
        for group_name in ("Progenitor / primitive streak", "Neural / ectoderm")
        for label in GASTRULATION_CELLTYPE_GROUPS[group_name]
        if label in order
    ]
    bottom_order = [label for label in order if label not in set(right_order)]

    fig = plt.figure(figsize=(7.90, 4.15), facecolor="white")
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=(1.0, 1.0, 1.0, 1.32),
        height_ratios=(2.58, 1.22),
        wspace=0.10,
        hspace=0.02,
    )
    rna_axis = fig.add_subplot(grid[0, 0])
    atac_axis = fig.add_subplot(grid[0, 1])
    predicted_axis = fig.add_subplot(grid[0, 2])
    right_legend_axis = fig.add_subplot(grid[0, 3])
    bottom_legend_axis = fig.add_subplot(grid[1, :])
    atac_limits = plot_limits(
        np.concatenate(
            (atac_embedding, predicted_atac_embedding),
            axis=0,
        )
    )
    draw_atlas(rna_axis, rna_embedding, celltypes, order, "RNA space")
    draw_atlas(
        atac_axis,
        atac_embedding,
        celltypes,
        order,
        "ATAC space",
        limits=atac_limits,
    )
    draw_atlas(
        predicted_axis,
        predicted_atac_embedding,
        celltypes,
        order,
        "Predicted ATAC",
        limits=atac_limits,
    )
    right_legend_axis.set_axis_off()
    bottom_legend_axis.set_axis_off()

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=gastrulation_celltype_color(label),
            markeredgecolor="none",
            markersize=6.0,
            label=label,
        )
        for label in order
    ]
    handle_by_label = dict(zip(order, handles, strict=True))
    right_legend_axis.legend(
        handles=[handle_by_label[label] for label in right_order],
        labels=right_order,
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        ncol=1,
        frameon=False,
        fontsize=10.0,
        handletextpad=0.35,
        labelspacing=0.34,
        borderaxespad=0.0,
    )
    bottom_legend_axis.legend(
        handles=[handle_by_label[label] for label in bottom_order],
        labels=bottom_order,
        loc="center",
        bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
        mode="expand",
        ncol=4,
        frameon=False,
        fontsize=10.0,
        handletextpad=0.35,
        columnspacing=0.75,
        labelspacing=0.30,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(
        left=0.01,
        right=0.995,
        top=0.97,
        bottom=0.015,
    )

    stem = "gastrulation_rna_atac_celltype_reference_split_legend_10pt"
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    save_options = {
        "facecolor": "white",
        "bbox_inches": "tight",
        "pad_inches": 0.02,
    }
    fig.savefig(png_path, dpi=600, **save_options)
    fig.savefig(pdf_path, **save_options)
    fig.savefig(svg_path, **save_options)
    plt.close(fig)

    counts = {
        label: int(np.sum(celltypes == label))
        for label in order
    }
    metadata = {
        "analysis": (
            "paired Gastrulation RNA/ATAC cell-type reference atlas with "
            "T-predicted ATAC"
        ),
        "n_cells": int(len(celltypes)),
        "n_celltypes": int(len(order)),
        "celltype_order": order,
        "celltype_counts": counts,
        "celltype_colors": {
            label: GASTRULATION_CELLTYPE_COLORS[label]
            for label in order
        },
        "rna_umap": str(DATA / "rna_umap_model.joblib"),
        "atac_umap": str(DATA / "atac_umap_model_14D.joblib"),
        "t_model": str(DATA / "TrainT" / "T_FiLM.pt"),
        "predicted_atac_definition": (
            "Each observed RNA cell is divided by the RNA normalization scale, "
            "mapped through the frozen T model at its matched physical time "
            "(0.0, 1.0, 2.0, or 2.5), restored to raw ATAC-LSI scale, and "
            "projected with the fixed ATAC UMAP interpolator."
        ),
        "atac_and_predicted_atac_share_umap_limits": True,
        "celltype_source": str(DATA / "celltype_sub_by_stage.npz"),
        "page_size": "Pancreas-matched compact landscape (7.90 x 4.15 inches)",
        "legend_position": (
            "first 11 progenitor/neural labels in a dedicated right column; "
            "remaining 22 labels in four columns below"
        ),
        "legend_right_labels": right_order,
        "legend_bottom_labels": bottom_order,
        "raster_dpi": 600,
        "typography": (
            "Arial 10 pt throughout; normal weight; matched to Pancreas figure"
        ),
        "output_png": str(png_path),
        "output_pdf": str(pdf_path),
        "output_svg": str(svg_path),
    }
    metadata_path = OUTPUT_DIR / f"{stem}.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(png_path)
    print(pdf_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
