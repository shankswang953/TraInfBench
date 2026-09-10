"""Plot paired COATI RNA and ATAC trajectory flow on fixed atlas UMAPs.

The RNA and ATAC panels use the paired trajectories saved by
``Gastrulation/FilmSyncEnergy``.  The secondary trajectory is the frozen
time-dependent map T(primary, t), so consecutive secondary positions retain
both the RNA motion and the explicit temporal contribution of T.

Cell colour is imported from the project-wide Gastrulation registry.  Method
identity is encoded only by the panel titles and neutral streamlines.
"""
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import binned_statistic_2d
from sklearn.neighbors import KNeighborsRegressor


# The UMAP joblib files import numba functions whose installed package path is
# not writable in the shared environment.  Disable numba's on-disk cache before
# importing joblib/umap, matching the existing Gastrulation UMAP scripts.
try:
    import numba.core.dispatcher

    numba.core.dispatcher.Dispatcher.enable_caching = lambda self: None
    try:
        import numba.np.ufunc.ufuncbuilder

        numba.np.ufunc.ufuncbuilder.UFuncDispatcher.enable_caching = (
            lambda self: None
        )
    except Exception:
        pass
    try:
        import numba.np.ufunc.dufunc

        numba.np.ufunc.dufunc.DUFunc._enable_caching = lambda self: None
    except Exception:
        pass
except Exception:
    pass

import joblib

from trainfbench_plot_style import (
    GASTRULATION_CELLTYPE_ALIASES,
    gastrulation_celltype_color,
)


ROOT = Path(__file__).resolve().parents[2]
TRAINF_ROOT = ROOT / "external"
GASTRULATION = TRAINF_ROOT / "COATI" / "Gastrulation"
DATA = GASTRULATION / "data"
TRAJECTORY_DIR = GASTRULATION / "FilmSyncEnergy" / "trajectory"
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "publication_figures"
    / "gastrulation_coati_dual_umap_flow"
)

TIME_KEYS = ("time0", "time1", "time2", "time3")
UMAP_K = 5
GRID_BINS = 65
GRID_MIN_COUNT = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cy", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=50000)
    parser.add_argument("--max-particles", type=int, default=3000)
    parser.add_argument("--stream-density", type=float, default=1.30)
    parser.add_argument("--output-tag", type=str, default="dense")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def cy_token(value: float) -> str:
    return f"{value:.1f}"


def load_concat_npz(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    return np.concatenate([data[key] for key in TIME_KEYS], axis=0).astype(
        np.float32
    )


def load_celltypes(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    labels = np.concatenate(
        [data[key].astype(str) for key in TIME_KEYS], axis=0
    )
    return np.asarray(
        [GASTRULATION_CELLTYPE_ALIASES.get(label, label) for label in labels],
        dtype=object,
    )


def load_scale(path: Path) -> float:
    params = torch.load(path, map_location="cpu", weights_only=False)
    return float(params["scale"])


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
    if max_abs > 1e-5:
        raise RuntimeError(
            f"UMAP reference order mismatch for {model_path}: "
            f"max |raw-latent|={max_abs:.3g}"
        )
    regressor = KNeighborsRegressor(
        n_neighbors=UMAP_K,
        weights="distance",
        algorithm="auto",
    )
    regressor.fit(latent, embedding)
    return embedding, regressor


def trajectory_umap_pairs(
    interpolator: KNeighborsRegressor,
    trajectory_path: Path,
    scale: float,
    max_particles: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    trajectory = torch.load(
        trajectory_path,
        map_location="cpu",
        weights_only=False,
    )
    values = (
        trajectory.detach().cpu().numpy().astype(np.float32) * np.float32(scale)
    )
    n_steps, n_particles, dim = values.shape
    if n_particles > max_particles:
        rng = np.random.default_rng(seed)
        chosen = np.sort(
            rng.choice(n_particles, size=max_particles, replace=False)
        )
        values = values[:, chosen, :]
        n_particles = max_particles
    embedded = interpolator.predict(
        values.reshape(n_steps * n_particles, dim)
    ).astype(np.float32)
    embedded = embedded.reshape(n_steps, n_particles, 2)
    starts = embedded[:-1].reshape(-1, 2)
    displacements = (embedded[1:] - embedded[:-1]).reshape(-1, 2)
    return starts, displacements


def plot_limits(embedding: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    x0, x1 = np.percentile(embedding[:, 0], [0.35, 99.65])
    y0, y1 = np.percentile(embedding[:, 1], [0.35, 99.65])
    dx = 0.035 * (x1 - x0)
    dy = 0.035 * (y1 - y0)
    return (float(x0 - dx), float(x1 + dx)), (float(y0 - dy), float(y1 + dy))


def grid_field(
    points: np.ndarray,
    vectors: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_edges = np.linspace(xlim[0], xlim[1], GRID_BINS + 1)
    y_edges = np.linspace(ylim[0], ylim[1], GRID_BINS + 1)
    count, _, _, _ = binned_statistic_2d(
        points[:, 0],
        points[:, 1],
        points[:, 0],
        statistic="count",
        bins=(x_edges, y_edges),
    )
    u, _, _, _ = binned_statistic_2d(
        points[:, 0],
        points[:, 1],
        vectors[:, 0],
        statistic="mean",
        bins=(x_edges, y_edges),
    )
    v, _, _, _ = binned_statistic_2d(
        points[:, 0],
        points[:, 1],
        vectors[:, 1],
        statistic="mean",
        bins=(x_edges, y_edges),
    )
    u = np.nan_to_num(u, nan=0.0)
    v = np.nan_to_num(v, nan=0.0)
    sparse = count < GRID_MIN_COUNT
    u[sparse] = 0.0
    v[sparse] = 0.0
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    speed = np.sqrt(u.T**2 + v.T**2)
    return x_centers, y_centers, u.T, v.T, speed


def scatter_celltypes(
    ax: plt.Axes,
    embedding: np.ndarray,
    celltypes: np.ndarray,
) -> None:
    # Draw large populations first so small populations remain visible.
    labels, counts = np.unique(celltypes, return_counts=True)
    order = labels[np.argsort(counts)[::-1]]
    for label in order:
        mask = celltypes == label
        is_missing = label == "Unannotated"
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=0.75 if not is_missing else 0.45,
            c=gastrulation_celltype_color(label),
            alpha=0.42 if not is_missing else 0.18,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )


def add_streamlines(
    ax: plt.Axes,
    field: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    density: float,
) -> None:
    x_centers, y_centers, u, v, speed = field
    positive = speed[speed > 0]
    reference = (
        float(np.nanpercentile(positive, 90)) if positive.size else 1.0
    )
    linewidth = np.clip(0.36 + 0.76 * speed / (reference + 1e-8), 0.34, 1.05)
    streams = ax.streamplot(
        x_centers,
        y_centers,
        u,
        v,
        color="#262626",
        density=density,
        linewidth=linewidth,
        arrowsize=0.66,
        minlength=0.07,
        maxlength=4.5,
        integration_direction="forward",
        zorder=4,
    )
    halo = [
        path_effects.Stroke(linewidth=1.7, foreground="white", alpha=0.78),
        path_effects.Normal(),
    ]
    streams.lines.set_path_effects(halo)
    streams.arrows.set_path_effects(halo)


def style_panel(
    ax: plt.Axes,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    ax.set_title(title, fontsize=12, fontweight="normal", pad=2.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "font.weight": "normal",
            "axes.titlesize": 12,
            "axes.titleweight": "normal",
            "axes.labelsize": 12,
            "axes.labelweight": "normal",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    token = cy_token(args.cy)
    primary_path = (
        TRAJECTORY_DIR
        / f"primary_trajectory_s{args.seed}_a{token}_iter{args.iteration}.pt"
    )
    secondary_path = (
        TRAJECTORY_DIR
        / f"secondary_trajectory_s{args.seed}_a{token}_iter{args.iteration}.pt"
    )
    for path in (primary_path, secondary_path):
        if not path.exists():
            raise FileNotFoundError(path)

    rna_latent = load_concat_npz(DATA / "rna_pca_by_time.npz")
    atac_latent = load_concat_npz(DATA / "atac_lsi_by_time_14D.npz")
    celltypes = load_celltypes(DATA / "celltype_sub_by_stage.npz")
    if not (len(rna_latent) == len(atac_latent) == len(celltypes)):
        raise RuntimeError(
            "RNA, ATAC, and cell-type reference arrays are not aligned: "
            f"{len(rna_latent)}, {len(atac_latent)}, {len(celltypes)}"
        )

    rna_embedding, rna_interpolator = fixed_umap_interpolator(
        rna_latent,
        DATA / "rna_umap_model.joblib",
    )
    atac_embedding, atac_interpolator = fixed_umap_interpolator(
        atac_latent,
        DATA / "atac_umap_model_14D.joblib",
    )
    rna_xlim, rna_ylim = plot_limits(rna_embedding)
    atac_xlim, atac_ylim = plot_limits(atac_embedding)

    rna_points, rna_vectors = trajectory_umap_pairs(
        rna_interpolator,
        primary_path,
        load_scale(DATA / "primal_norm_params.pt"),
        args.max_particles,
        seed=104 + args.seed,
    )
    atac_points, atac_vectors = trajectory_umap_pairs(
        atac_interpolator,
        secondary_path,
        load_scale(DATA / "secondary_norm_params.pt"),
        args.max_particles,
        seed=104 + args.seed,
    )
    rna_field = grid_field(rna_points, rna_vectors, rna_xlim, rna_ylim)
    atac_field = grid_field(atac_points, atac_vectors, atac_xlim, atac_ylim)

    fig, axes = plt.subplots(1, 2, figsize=(4.13, 2.02))
    for ax, embedding, field, title, xlim, ylim in (
        (
            axes[0],
            rna_embedding,
            rna_field,
            "RNA space",
            rna_xlim,
            rna_ylim,
        ),
        (
            axes[1],
            atac_embedding,
            atac_field,
            "ATAC space",
            atac_xlim,
            atac_ylim,
        ),
    ):
        scatter_celltypes(ax, embedding, celltypes)
        add_streamlines(ax, field, args.stream_density)
        style_panel(ax, title, xlim, ylim)

    fig.subplots_adjust(left=0.006, right=0.994, bottom=0.008, top=0.91, wspace=0.055)
    stem = (
        f"gastrulation_coati_bal_cy{token.replace('.', 'p')}_"
        f"s{args.seed}_iter{args.iteration}_rna_atac_umap_flow"
    )
    if args.output_tag:
        stem = f"{stem}_{args.output_tag}"
    png_path = args.output_dir / f"{stem}.png"
    pdf_path = args.output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

    metadata = {
        "model": "COATI-BSOT",
        "C_y": args.cy,
        "seed": args.seed,
        "iteration": args.iteration,
        "stream_density": args.stream_density,
        "primary_trajectory": str(primary_path),
        "secondary_trajectory": str(secondary_path),
        "secondary_definition": "T_theta(primary_trajectory, physical_time)",
        "rna_umap": str(DATA / "rna_umap_model.joblib"),
        "atac_umap": str(DATA / "atac_umap_model_14D.joblib"),
        "celltype_colors": str(
            ROOT / "common" / "trainfbench_plot_style.py"
        ),
        "typography": "Arial; titles 12 pt; other text 10 pt; normal weight",
        "output_png": str(png_path),
        "output_pdf": str(pdf_path),
    }
    metadata_path = args.output_dir / f"{stem}.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(png_path)
    print(pdf_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
