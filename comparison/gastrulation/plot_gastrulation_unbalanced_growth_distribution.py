"""Plot trajectory-resolved growth localization for two unbalanced methods.

The figure follows the exact source-particle subset used by
``plot_gastrulation_unbalanced_dual_umap_flow.py``.  Instantaneous growth is
defined for both methods as d log(w) / dt and is evaluated at E7.5, E8.0,
E8.5, and E8.75 on each method's own RNA trajectory.

Raw magnitudes are not cross-method comparable because the methods use
different mass targets and growth penalties.  The visual therefore applies a
30-nearest-neighbour median within each method/time, subtracts the native-mass
weighted mean, and scales by the native-mass-weighted 95th percentile of the
absolute residual.  The same RNA-defined relative-growth scalar is displayed
in RNA and T-mapped ATAC coordinates.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/xdg-cache")
for cache_variable in ("MPLCONFIGDIR", "NUMBA_CACHE_DIR", "XDG_CACHE_HOME"):
    Path(os.environ[cache_variable]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TRAINF_ROOT = ROOT / "external"
GASTRULATION = TRAINF_ROOT / "COATI" / "Gastrulation"
DATA = GASTRULATION / "data"
COATI_DIR = (
    GASTRULATION
    / "UnbalancedSync_biological_num"
    / "trajectory_hpc_iter20000"
)
COATI_CHECKPOINT = (
    GASTRULATION
    / "UnbalancedSync_biological_num"
    / "checkpoint_hpc_iter20000"
    / "ckpt_s0_e0.01_m100.0_d0.01_a0.3_iter20000.pth"
)
CYTOBRIDGE_ADATA = (
    ROOT
    / "results"
    / "cytobridge_gastrulation_rna_20000_unbalanced"
    / "adata.h5ad"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "publication_figures"
    / "gastrulation_unbalanced_growth_distribution"
)

for path in (
    ROOT / "scripts",
    ROOT / "external" / "CytoBridge",
    TRAINF_ROOT / "TraInf",
):
    if path.exists():
        sys.path.insert(0, str(path))

from plot_gastrulation_coati_dual_umap_flow import (  # noqa: E402
    fixed_umap_interpolator,
    load_concat_npz,
    load_scale,
    plot_limits,
)

# Load the fixed UMAP objects before importing analysis dependencies that
# register additional numba functions.  This mirrors the working standalone
# UMAP plotting scripts and avoids fragile joblib/numba rebuild ordering.
RNA_ATLAS = load_concat_npz(DATA / "rna_pca_by_time.npz")
ATAC_ATLAS = load_concat_npz(DATA / "atac_lsi_by_time_14D.npz")
RNA_EMBEDDING, RNA_INTERPOLATOR = fixed_umap_interpolator(
    RNA_ATLAS,
    DATA / "rna_umap_model.joblib",
)
ATAC_EMBEDDING, ATAC_INTERPOLATOR = fixed_umap_interpolator(
    ATAC_ATLAS,
    DATA / "atac_umap_model_14D.joblib",
)

import pandas as pd  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from evaluate_gastrulation_flow_methods_normalized import (  # noqa: E402
    _cytobridge_velocity,
    _load_cytobridge_model,
)
from evaluate_gastrulation_full_cmcc import (  # noqa: E402
    _apply_t,
    _load_t_model,
)
from evaluate_gastrulation_loo_time1_paired_1nn_atac import (  # noqa: E402
    _load_own_model,
)
from plot_gastrulation_unbalanced_dual_umap_flow import (  # noqa: E402
    _chosen_indices,
    _load_tensor,
)


STAGES = ("E7.5", "E8.0", "E8.5", "E8.75")
STAGE_TIMES = np.asarray((0.0, 1.0, 2.0, 2.5), dtype=np.float32)
STAGE_INDICES = np.asarray((0, 10, 20, 25), dtype=np.int64)
INTERVALS = ((0.0, 1.0, 10), (1.0, 2.0, 10), (2.0, 2.5, 5))


def usot_growth(
    model,
    x_norm: torch.Tensor,
    time: float,
) -> torch.Tensor:
    t = torch.full(
        (len(x_norm), 1),
        float(time),
        dtype=x_norm.dtype,
        device=x_norm.device,
    )
    return (
        model.growth_net(torch.cat([x_norm, t], dim=1))[:, 0]
        * float(model.alpha_growth)
    )


def cytobridge_growth(
    model,
    x_norm: torch.Tensor,
    time: float,
    scale: float,
) -> torch.Tensor:
    t = torch.full(
        (len(x_norm), 1),
        float(time),
        dtype=x_norm.dtype,
        device=x_norm.device,
    )
    x_raw = x_norm * float(scale)
    return model.growth_net(torch.cat([x_raw, t], dim=1))[:, 0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cy", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=20000)
    parser.add_argument("--max-particles", type=int, default=3000)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--layout",
        choices=("a4", "compact"),
        default="a4",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help=(
            "Resolution for PNG output and rasterized point-cloud layers "
            "embedded in PDF output."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _evaluate_growth(
    method: str,
    model,
    trajectory_norm: np.ndarray,
    scale: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    values = np.empty((len(STAGE_INDICES), trajectory_norm.shape[1]), dtype=np.float64)
    for output_index, (step, time) in enumerate(zip(STAGE_INDICES, STAGE_TIMES)):
        pieces: list[np.ndarray] = []
        for start in range(0, trajectory_norm.shape[1], batch_size):
            stop = min(start + batch_size, trajectory_norm.shape[1])
            x = torch.as_tensor(
                trajectory_norm[step, start:stop],
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                if method == "COATI unbal.":
                    growth = usot_growth(model, x, float(time))
                elif method == "CytoBridge unbal.":
                    growth = cytobridge_growth(model, x, float(time), scale)
                else:
                    raise ValueError(method)
            pieces.append(growth.detach().cpu().numpy())
        values[output_index] = np.concatenate(pieces)
    return values


def _rank_percentile(values: np.ndarray) -> np.ndarray:
    n = len(values)
    return (rankdata(values, method="average") - 0.5) / float(n)


def _smooth_growth(
    positions_raw: np.ndarray,
    growth: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    smoothed = np.empty_like(growth, dtype=np.float64)
    for index in range(growth.shape[0]):
        k = min(int(neighbors), len(growth[index]))
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nn.fit(positions_raw[index])
        neighbor_index = nn.kneighbors(return_distance=False)
        smoothed[index] = np.median(growth[index][neighbor_index], axis=1)
    return smoothed


def _normalized_mass(log_mass: np.ndarray) -> np.ndarray:
    values = np.asarray(log_mass, dtype=np.float64)
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(sorted_weights)
    threshold = float(quantile) * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, threshold, side="left")), len(values) - 1)
    return float(sorted_values[index])


def _relative_growth(
    smoothed_growth: np.ndarray,
    log_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    normalized = np.empty_like(smoothed_growth, dtype=np.float64)
    percentiles = np.empty_like(smoothed_growth, dtype=np.float64)
    diagnostics: list[dict[str, float]] = []
    for index, stage in enumerate(STAGES):
        weights = _normalized_mass(log_mass[index])
        weighted_mean = float(np.sum(weights * smoothed_growth[index]))
        relative = smoothed_growth[index] - weighted_mean
        scale = _weighted_quantile(np.abs(relative), weights, 0.95)
        scale = max(scale, 1e-8)
        normalized[index] = np.clip(relative / scale, -1.0, 1.0)
        percentiles[index] = _rank_percentile(smoothed_growth[index])
        diagnostics.append(
            {
                "stage": stage,
                "mass_weighted_mean_smoothed_growth": weighted_mean,
                "mass_weighted_q95_absolute_relative_growth": scale,
                "native_total_mass_relative_to_initial": float(
                    np.exp(
                        logsumexp(log_mass[index])
                        - logsumexp(log_mass[0])
                    )
                ),
            }
        )
    return normalized, percentiles, diagnostics


def _rollout_cytobridge_with_mass(
    model,
    x0_norm: np.ndarray,
    scale: float,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    positions = [np.asarray(x0_norm, dtype=np.float32).copy()]
    log_mass = [np.zeros(len(x0_norm), dtype=np.float64)]
    current = positions[0]
    cumulative = log_mass[0]
    for t0, t1, steps in INTERVALS:
        dt = (float(t1) - float(t0)) / float(steps)
        for step in range(steps):
            next_position = np.empty_like(current)
            growth_values = np.empty(len(current), dtype=np.float64)
            time_mid = float(t0) + (step + 0.5) * dt
            for start in range(0, len(current), batch_size):
                stop = min(start + batch_size, len(current))
                x = torch.as_tensor(
                    current[start:stop],
                    dtype=torch.float32,
                    device=device,
                )
                t_mid = torch.tensor(
                    [time_mid],
                    dtype=torch.float32,
                    device=device,
                )
                with torch.no_grad():
                    velocity = _cytobridge_velocity(model, t_mid, x * float(scale))
                    growth = cytobridge_growth(
                        model,
                        x,
                        time_mid,
                        scale,
                    )
                    updated = x + dt * velocity / float(scale)
                next_position[start:stop] = updated.detach().cpu().numpy()
                growth_values[start:stop] = growth.detach().cpu().numpy()
            current = next_position
            cumulative = cumulative + dt * growth_values
            positions.append(current.copy())
            log_mass.append(cumulative.copy())
    return (
        np.stack(positions).astype(np.float32, copy=False),
        np.stack(log_mass).astype(np.float64, copy=False),
    )


def _map_stage_trajectory_with_t(
    model: torch.nn.Module,
    rna_stage_norm: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    mapped = [
        _apply_t(
            model,
            rna_stage_norm[index],
            float(time),
            device,
            batch_size,
        )
        for index, time in enumerate(STAGE_TIMES)
    ]
    return np.stack(mapped, axis=0).astype(np.float32, copy=False)


def _raw_summary(method: str, growth: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for stage, values in zip(STAGES, growth):
        quantiles = np.quantile(
            values,
            [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
        )
        rows.append(
            {
                "method": method,
                "stage": stage,
                "n_particles": len(values),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values)),
                "p01": float(quantiles[0]),
                "p05": float(quantiles[1]),
                "p25": float(quantiles[2]),
                "median": float(quantiles[3]),
                "p75": float(quantiles[4]),
                "p95": float(quantiles[5]),
                "p99": float(quantiles[6]),
                "fraction_positive": float(np.mean(values > 0)),
            }
        )
    return rows


def _agreement_rows(
    coati_percentile: np.ndarray,
    cytobridge_percentile: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, stage in enumerate(STAGES):
        a = coati_percentile[index]
        b = cytobridge_percentile[index]
        top_a = a >= 0.8
        top_b = b >= 0.8
        bottom_a = a <= 0.2
        bottom_b = b <= 0.2
        rows.append(
            {
                "stage": stage,
                "n_particles": len(a),
                "particlewise_spearman": float(spearmanr(a, b).statistic),
                "mean_absolute_percentile_difference": float(
                    np.mean(np.abs(a - b))
                ),
                "top20_jaccard": float(
                    np.sum(top_a & top_b) / max(np.sum(top_a | top_b), 1)
                ),
                "bottom20_jaccard": float(
                    np.sum(bottom_a & bottom_b)
                    / max(np.sum(bottom_a | bottom_b), 1)
                ),
            }
        )
    return rows


def _plot_stage_maps(
    output_dir: Path,
    modality: str,
    atlas_embedding: np.ndarray,
    method_embeddings: dict[str, np.ndarray],
    normalized_growth: dict[str, np.ndarray],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    cy: float,
    layout: str,
    dpi: int,
) -> tuple[Path, Path]:
    methods = ("COATI unbal.", "CytoBridge unbal.")
    is_a4 = layout == "a4"
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(11.69, 8.27) if is_a4 else (4.13, 2.35),
    )
    # Keep the atlas visible enough to orient each growth field without
    # competing with the colored trajectory particles.
    atlas_size = 0.44 if is_a4 else 0.19
    predicted_size = 3.0 if is_a4 else 1.25
    scatter = None
    for row, method in enumerate(methods):
        for column, stage in enumerate(STAGES):
            ax = axes[row, column]
            ax.scatter(
                atlas_embedding[:, 0],
                atlas_embedding[:, 1],
                s=atlas_size,
                c="#9E9E9E",
                alpha=0.24,
                linewidths=0,
                rasterized=True,
                zorder=1,
            )
            values = normalized_growth[method][column]
            order = np.argsort(np.abs(values))
            xy = method_embeddings[method][column][order]
            scatter = ax.scatter(
                xy[:, 0],
                xy[:, 1],
                s=predicted_size,
                c=values[order],
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
                alpha=0.80,
                linewidths=0,
                rasterized=True,
                zorder=3,
            )
            if row == 0:
                ax.set_title(stage, fontsize=12, fontweight="normal", pad=1.0)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[row, 0].text(
            -0.040 if is_a4 else -0.055,
            0.50,
            method,
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="normal",
        )

    if scatter is None:
        raise RuntimeError("No growth scatter was drawn")
    colorbar_axis = fig.add_axes(
        [0.31, 0.038, 0.48, 0.020]
        if is_a4
        else [0.31, 0.030, 0.48, 0.020]
    )
    colorbar = fig.colorbar(
        scatter,
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=[-1.0, 0.0, 1.0],
    )
    colorbar.ax.tick_params(labelsize=10, length=2.0, pad=1.0)
    colorbar.set_label(
        "Stage-normalized relative growth",
        fontsize=12 if is_a4 else 10,
        fontweight="normal",
        labelpad=1.5,
    )
    if is_a4:
        fig.subplots_adjust(
            left=0.045,
            right=0.995,
            bottom=0.105,
            top=0.965,
            wspace=0.018,
            hspace=0.018,
        )
    else:
        fig.subplots_adjust(
            left=0.075,
            right=0.997,
            bottom=0.105,
            top=0.945,
            wspace=0.025,
            hspace=0.025,
        )
    stem = (
        f"gastrulation_coati_unbal_cy{cy:.1f}_cytobridge_unbal_"
        f"growth_distribution_{modality.lower()}"
    ).replace(".", "p")
    if is_a4:
        stem = f"{stem}_a4_landscape"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    if is_a4:
        fig.savefig(png_path, dpi=dpi)
        fig.savefig(pdf_path, dpi=dpi)
    else:
        fig.savefig(
            png_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.010,
        )
        fig.savefig(
            pdf_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.010,
        )
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
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

    token = f"{args.cy:.1f}"
    primary_path = (
        COATI_DIR
        / f"primary_trajectory_s{args.seed}_a{token}_iter{args.iteration}.pt"
    )
    secondary_path = (
        COATI_DIR
        / f"secondary_trajectory_s{args.seed}_a{token}_iter{args.iteration}.pt"
    )
    mass_path = (
        COATI_DIR
        / f"mass_lnw_trajectory_s{args.seed}_a{token}_iter{args.iteration}.pt"
    )
    for path in (
        primary_path,
        secondary_path,
        mass_path,
        COATI_CHECKPOINT,
        CYTOBRIDGE_ADATA,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    coati_rna_norm = _load_tensor(primary_path)
    coati_atac_norm = _load_tensor(secondary_path)
    coati_log_mass = _load_tensor(mass_path).squeeze(-1).astype(np.float64)
    chosen = _chosen_indices(
        coati_rna_norm.shape[1],
        args.max_particles,
        args.seed,
    )

    rna_scale = load_scale(DATA / "primal_norm_params.pt")
    atac_scale = load_scale(DATA / "secondary_norm_params.pt")
    coati_model = _load_own_model(
        COATI_CHECKPOINT,
        coati_rna_norm.shape[-1],
        True,
        device,
    )
    cytobridge_model = _load_cytobridge_model(CYTOBRIDGE_ADATA, device)
    cytobridge_rna_norm, cytobridge_log_mass = _rollout_cytobridge_with_mass(
        cytobridge_model,
        coati_rna_norm[0],
        rna_scale,
        args.batch_size,
        device,
    )
    t_model = _load_t_model(device)
    cytobridge_atac_stage_norm = _map_stage_trajectory_with_t(
        t_model,
        cytobridge_rna_norm[STAGE_INDICES],
        args.batch_size,
        device,
    )

    growth = {
        "COATI unbal.": _evaluate_growth(
            "COATI unbal.",
            coati_model,
            coati_rna_norm,
            rna_scale,
            device,
            args.batch_size,
        ),
        "CytoBridge unbal.": _evaluate_growth(
            "CytoBridge unbal.",
            cytobridge_model,
            cytobridge_rna_norm,
            rna_scale,
            device,
            args.batch_size,
        ),
    }
    rna_raw = {
        "COATI unbal.": (
            coati_rna_norm[STAGE_INDICES] * np.float32(rna_scale)
        ),
        "CytoBridge unbal.": (
            cytobridge_rna_norm[STAGE_INDICES] * np.float32(rna_scale)
        ),
    }
    atac_raw = {
        "COATI unbal.": (
            coati_atac_norm[STAGE_INDICES] * np.float32(atac_scale)
        ),
        "CytoBridge unbal.": (
            cytobridge_atac_stage_norm * np.float32(atac_scale)
        ),
    }
    log_masses = {
        "COATI unbal.": coati_log_mass[STAGE_INDICES],
        "CytoBridge unbal.": cytobridge_log_mass[STAGE_INDICES],
    }
    normalized_growth: dict[str, np.ndarray] = {}
    percentiles: dict[str, np.ndarray] = {}
    smoothed: dict[str, np.ndarray] = {}
    normalization_rows: list[dict[str, object]] = []
    for method in growth:
        smoothed[method] = _smooth_growth(
            rna_raw[method],
            growth[method],
            args.neighbors,
        )
        (
            normalized_growth[method],
            percentiles[method],
            local_diagnostics,
        ) = _relative_growth(smoothed[method], log_masses[method])
        for row in local_diagnostics:
            normalization_rows.append({"method": method, **row})

    rna_xlim, rna_ylim = plot_limits(RNA_EMBEDDING)
    atac_xlim, atac_ylim = plot_limits(ATAC_EMBEDDING)

    rna_method_embedding = {
        method: np.stack(
            [RNA_INTERPOLATOR.predict(stage[chosen]) for stage in stages],
            axis=0,
        ).astype(np.float32)
        for method, stages in rna_raw.items()
    }
    atac_method_embedding = {
        method: np.stack(
            [ATAC_INTERPOLATOR.predict(stage[chosen]) for stage in stages],
            axis=0,
        ).astype(np.float32)
        for method, stages in atac_raw.items()
    }

    rna_png, rna_pdf = _plot_stage_maps(
        args.output_dir,
        "RNA",
        RNA_EMBEDDING,
        rna_method_embedding,
        {
            method: values[:, chosen]
            for method, values in normalized_growth.items()
        },
        rna_xlim,
        rna_ylim,
        args.cy,
        args.layout,
        args.dpi,
    )
    atac_png, atac_pdf = _plot_stage_maps(
        args.output_dir,
        "ATAC",
        ATAC_EMBEDDING,
        atac_method_embedding,
        {
            method: values[:, chosen]
            for method, values in normalized_growth.items()
        },
        atac_xlim,
        atac_ylim,
        args.cy,
        args.layout,
        args.dpi,
    )

    raw_rows = [
        row
        for method in ("COATI unbal.", "CytoBridge unbal.")
        for row in _raw_summary(method, growth[method])
    ]
    raw_path = args.output_dir / "raw_growth_summary.csv"
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
    normalization_path = args.output_dir / "relative_growth_normalization.csv"
    pd.DataFrame(normalization_rows).to_csv(normalization_path, index=False)
    agreement_path = args.output_dir / "growth_localization_agreement.csv"
    pd.DataFrame(
        _agreement_rows(
            percentiles["COATI unbal."],
            percentiles["CytoBridge unbal."],
        )
    ).to_csv(agreement_path, index=False)

    values_path = args.output_dir / "growth_distribution_values.npz"
    np.savez_compressed(
        values_path,
        source_particle_indices=chosen,
        stages=np.asarray(STAGES),
        stage_times=STAGE_TIMES,
        coati_growth_raw=growth["COATI unbal."].astype(np.float32),
        cytobridge_growth_raw=growth["CytoBridge unbal."].astype(np.float32),
        coati_log_mass=log_masses["COATI unbal."].astype(np.float32),
        cytobridge_log_mass=log_masses["CytoBridge unbal."].astype(np.float32),
        coati_growth_knn_median=smoothed["COATI unbal."].astype(np.float32),
        cytobridge_growth_knn_median=smoothed["CytoBridge unbal."].astype(
            np.float32
        ),
        coati_growth_percentile=percentiles["COATI unbal."].astype(np.float32),
        cytobridge_growth_percentile=percentiles[
            "CytoBridge unbal."
        ].astype(np.float32),
        coati_relative_growth_normalized=normalized_growth[
            "COATI unbal."
        ].astype(np.float32),
        cytobridge_relative_growth_normalized=normalized_growth[
            "CytoBridge unbal."
        ].astype(np.float32),
    )

    metadata = {
        "analysis": "trajectory-resolved instantaneous growth localization",
        "methods": ["COATI-USOT", "CytoBridge unbalanced"],
        "C_y": args.cy,
        "seed": args.seed,
        "iteration": args.iteration,
        "layout": args.layout,
        "raster_dpi": args.dpi,
        "page_size": (
            "A4 landscape (11.69 x 8.27 inches)"
            if args.layout == "a4"
            else "compact half-A4-width panel"
        ),
        "n_particles_analyzed": int(coati_rna_norm.shape[1]),
        "n_particles_rendered": int(len(chosen)),
        "display_sampling": (
            "the same deterministic source-particle subset used by the "
            "paired RNA/ATAC streamline comparison"
        ),
        "growth_definition": "instantaneous d log(w) / dt",
        "normalization": (
            f"RNA {args.neighbors}-NN median smoothing; subtract native-mass-"
            "weighted mean growth at each stage; divide by native-mass-weighted "
            "95th percentile of absolute relative growth; clip to [-1,1]"
        ),
        "raw_growth_warning": (
            "raw magnitudes are not cross-method comparable because mass "
            "targets and growth penalties differ"
        ),
        "atac_interpretation": (
            "the RNA-defined growth scalar is relocated to each method's "
            "T-mapped ATAC coordinates; there is no separate ATAC growth head"
        ),
        "coati_checkpoint": str(COATI_CHECKPOINT),
        "coati_primary_trajectory": str(primary_path),
        "coati_secondary_trajectory": str(secondary_path),
        "cytobridge_adata": str(CYTOBRIDGE_ADATA),
        "outputs": {
            "rna_png": str(rna_png),
            "rna_pdf": str(rna_pdf),
            "atac_png": str(atac_png),
            "atac_pdf": str(atac_pdf),
            "raw_summary": str(raw_path),
            "relative_growth_normalization": str(normalization_path),
            "agreement": str(agreement_path),
            "values": str(values_path),
        },
        "typography": (
            "Arial; stage titles 12 pt; method labels/colorbar 10 pt; "
            "normal weight"
        ),
    }
    metadata_path = args.output_dir / "analysis_manifest.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(rna_png)
    print(rna_pdf)
    print(atac_png)
    print(atac_pdf)
    print(raw_path)
    print(agreement_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
