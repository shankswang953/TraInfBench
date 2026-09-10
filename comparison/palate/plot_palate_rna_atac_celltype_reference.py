#!/usr/bin/env python
"""Plot paired palate RNA, ATAC, and full-T-predicted ATAC by cell type."""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import anndata as ad
import joblib
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import torch

from trainfbench_plot_style import (
    PALATE_CELLTYPE_COLORS,
    PALATE_CELLTYPE_LABELS,
    PALATE_CELLTYPE_ORDER,
    apply_nature_rc,
)


ROOT = Path(__file__).resolve().parents[2]
TRAINF_ROOT = Path("external/COATI")
PALATE_ROOT = TRAINF_ROOT / "MouseBrain"
RNA_FILE = ROOT / "data/palate_rna_cytobridge.h5ad"
ATAC_FILE = ROOT / "data/palate_atac_benchmark.h5ad"
RNA_NORM = ROOT / "data/palate_rna_primal_norm_params.pt"
ATAC_NORM = ROOT / "data/palate_atac_secondary_norm_params_lsi15.pt"
UMAP_DIR = ROOT / "results/palate_strict_loo_dual_umap_tn_reversed"
T_SOURCE = PALATE_ROOT / "TrainMap/train_FiLM_MLP_lsi15.py"
T_CHECKPOINT = PALATE_ROOT / "TrainMap/T_FiLM_lsi15.pt"
DEFAULT_OUTPUT = ROOT / "results/palate_rna_atac_celltype_reference"
EXPECTED_T_SHA256 = "08587995268ef867af2fbe18b7b9c26030bdbeb65d4df3d08d088dcdb1968f2b"
UMAP_K = 5

sys.path.insert(0, str(TRAINF_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_python_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_full_t(device: torch.device) -> torch.nn.Module:
    observed_hash = sha256(T_CHECKPOINT)
    if observed_hash != EXPECTED_T_SHA256:
        raise RuntimeError(
            f"Full palate T version mismatch: {observed_hash} != {EXPECTED_T_SHA256}"
        )
    module = load_python_module(T_SOURCE, "palate_full_film_lsi15_for_umap")
    state = torch.load(T_CHECKPOINT, map_location=device, weights_only=False)
    model = module.FiLMMLP(**state["config"]).to(device)
    model.load_state_dict(state["state_dict"])
    return model.eval().requires_grad_(False)


def load_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rna = ad.read_h5ad(RNA_FILE)
    atac = ad.read_h5ad(ATAC_FILE)
    rna_names = rna.obs_names.to_numpy(str)
    atac_names = atac.obs_names.to_numpy(str)
    if not np.array_equal(rna_names, atac_names):
        raise ValueError("RNA and ATAC cells are not paired in the same order")
    labels = rna.obs["celltype_sub"].astype(str).to_numpy()
    atac_labels = atac.obs["celltype_sub"].astype(str).to_numpy()
    if not np.array_equal(labels, atac_labels):
        raise ValueError("RNA and ATAC cell-type annotations differ")
    unknown = set(labels).difference(PALATE_CELLTYPE_ORDER)
    if unknown:
        raise ValueError(f"Unregistered palate cell types: {sorted(unknown)}")
    times = rna.obs["time_point_processed"].to_numpy(float)
    atac_times = atac.obs["time_point_processed"].to_numpy(float)
    if not np.array_equal(times, atac_times):
        raise ValueError("RNA and ATAC physical times differ")
    rna_scale = float(
        np.asarray(
            torch.load(RNA_NORM, map_location="cpu", weights_only=False)["scale"]
        ).reshape(-1)[0]
    )
    atac_scale = float(
        np.asarray(
            torch.load(ATAC_NORM, map_location="cpu", weights_only=False)["scale"]
        ).reshape(-1)[0]
    )
    if rna_scale <= 0 or atac_scale <= 0:
        raise ValueError(f"Invalid normalization scales: RNA={rna_scale}, ATAC={atac_scale}")
    rna_norm = np.asarray(rna.obsm["X_latent"], dtype=np.float32) / rna_scale
    atac_norm = np.asarray(atac.obsm["X_lsi15"], dtype=np.float32) / atac_scale
    return rna_norm, atac_norm, times, labels


def fixed_umap_interpolator(
    latent: np.ndarray,
    model_path: Path,
) -> tuple[np.ndarray, KNeighborsRegressor]:
    model = joblib.load(model_path)
    embedding = np.asarray(model.embedding_, dtype=np.float32)
    raw_data = np.asarray(model._raw_data, dtype=np.float32)
    if raw_data.shape != latent.shape:
        raise RuntimeError(
            f"UMAP/raw latent shape mismatch for {model_path}: "
            f"{raw_data.shape} versus {latent.shape}"
        )
    max_abs = float(np.max(np.abs(raw_data - latent)))
    if max_abs > 1e-6:
        raise RuntimeError(
            f"UMAP reference mismatch for {model_path}: max |raw-latent|={max_abs:.3g}"
        )
    regressor = KNeighborsRegressor(
        n_neighbors=UMAP_K,
        weights="distance",
        algorithm="auto",
    )
    regressor.fit(latent, embedding)
    return embedding, regressor


def apply_t(
    model: torch.nn.Module,
    rna_norm: np.ndarray,
    times: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rna_norm), batch_size):
            stop = min(start + batch_size, len(rna_norm))
            x = torch.as_tensor(rna_norm[start:stop], dtype=torch.float32, device=device)
            t = torch.as_tensor(times[start:stop], dtype=torch.float32, device=device)
            output.append(model(x, t).detach().cpu().numpy().astype(np.float32))
    result = np.concatenate(output, axis=0)
    if result.shape != (len(rna_norm), 15):
        raise ValueError(f"Unexpected predicted ATAC shape: {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Predicted ATAC contains NaN or Inf")
    return result


def plot_limits(points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    lower = np.quantile(points, 0.0035, axis=0)
    upper = np.quantile(points, 0.9965, axis=0)
    span = np.maximum(upper - lower, 1e-6)
    return (
        (float(lower[0] - 0.035 * span[0]), float(upper[0] + 0.035 * span[0])),
        (float(lower[1] - 0.035 * span[1]), float(upper[1] + 0.035 * span[1])),
    )


def draw_atlas(
    ax: plt.Axes,
    embedding: np.ndarray,
    labels: np.ndarray,
    title: str,
    limits: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    present = [label for label in PALATE_CELLTYPE_ORDER if np.any(labels == label)]
    counts = {label: int(np.sum(labels == label)) for label in present}
    # Draw abundant groups first so rare populations remain visible.
    for label in sorted(present, key=lambda item: counts[item], reverse=True):
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=1.05,
            color=PALATE_CELLTYPE_COLORS[label],
            alpha=0.72,
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


def save_figure(fig: plt.Figure, stem: Path) -> None:
    options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.02}
    fig.savefig(stem.with_suffix(".png"), dpi=600, **options)
    fig.savefig(stem.with_suffix(".pdf"), **options)
    fig.savefig(stem.with_suffix(".svg"), **options)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_nature_rc(font_size=10.0)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "font.weight": "normal",
            "axes.titlesize": 10.0,
            "axes.titleweight": "normal",
            "legend.fontsize": 10.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    rna_norm, atac_norm, times, labels = load_inputs()
    rna_embedding, _ = fixed_umap_interpolator(
        rna_norm,
        UMAP_DIR / "rna_umap_model.joblib",
    )
    atac_embedding, atac_interpolator = fixed_umap_interpolator(
        atac_norm,
        UMAP_DIR / "atac_umap_model.joblib",
    )
    device = torch.device("cpu")
    full_t = load_full_t(device)
    predicted_atac = apply_t(
        full_t,
        rna_norm,
        times,
        device,
        args.batch_size,
    )
    predicted_embedding = atac_interpolator.predict(predicted_atac).astype(
        np.float32,
        copy=False,
    )

    rna_limits = plot_limits(rna_embedding)
    atac_limits = plot_limits(
        np.concatenate([atac_embedding, predicted_embedding], axis=0)
    )
    fig = plt.figure(figsize=(7.90, 2.45), facecolor="white")
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=(1.0, 1.0, 1.0, 1.35),
        wspace=0.08,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    draw_atlas(axes[0], rna_embedding, labels, "RNA space", rna_limits)
    draw_atlas(axes[1], atac_embedding, labels, "ATAC space", atac_limits)
    draw_atlas(
        axes[2],
        predicted_embedding,
        labels,
        "Predicted ATAC",
        atac_limits,
    )
    legend_axis = fig.add_subplot(grid[0, 3])
    legend_axis.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=PALATE_CELLTYPE_COLORS[label],
            markeredgecolor="none",
            markersize=4.4,
            label=PALATE_CELLTYPE_LABELS[label],
        )
        for label in PALATE_CELLTYPE_ORDER
    ]
    legend_axis.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        ncol=2,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.65,
        labelspacing=0.38,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.008, right=0.995, top=0.95, bottom=0.025)
    stem = args.output_dir / "palate_rna_atac_predicted_atac_celltype_umap"
    save_figure(fig, stem)

    np.savez_compressed(
        args.output_dir / "palate_t_umap_coordinates.npz",
        celltype=labels,
        physical_time=times.astype(np.float32),
        rna_umap=rna_embedding,
        atac_umap=atac_embedding,
        predicted_atac_umap=predicted_embedding,
        predicted_atac_norm=predicted_atac,
    )
    counts = {
        PALATE_CELLTYPE_LABELS[label]: int(np.sum(labels == label))
        for label in PALATE_CELLTYPE_ORDER
    }
    manifest = {
        "analysis": "Paired palate RNA/ATAC atlas and full-T-predicted ATAC",
        "n_cells": int(len(labels)),
        "n_celltypes": int(len(PALATE_CELLTYPE_ORDER)),
        "celltype_counts": counts,
        "celltype_colors": {
            PALATE_CELLTYPE_LABELS[label]: PALATE_CELLTYPE_COLORS[label]
            for label in PALATE_CELLTYPE_ORDER
        },
        "rna_input": "all observed cells in normalized RNA PCA40",
        "atac_input": "the paired observed cells in normalized ATAC LSI15",
        "predicted_atac": (
            "For every paired observed RNA cell i, compute full-data "
            "T(RNA_i, physical_time_i) in normalized ATAC LSI15."
        ),
        "umap_policy": (
            "RNA and ATAC coordinates use the fixed global real-atlas UMAPs. "
            "Predicted ATAC is projected into the real ATAC UMAP using the same "
            f"distance-weighted {UMAP_K}-NN interpolator as the Gastrulation figure."
        ),
        "shared_atac_axes": True,
        "T_checkpoint": str(T_CHECKPOINT),
        "T_sha256": sha256(T_CHECKPOINT),
        "rna_umap_model": str((UMAP_DIR / "rna_umap_model.joblib").resolve()),
        "atac_umap_model": str((UMAP_DIR / "atac_umap_model.joblib").resolve()),
        "font": "Arial 10 pt",
        "random_seed": args.seed,
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
