"""Compare COATI and CytoBridge unbalanced flow in fixed RNA/ATAC UMAPs.

Both methods start from the same deterministic subset of the 9,018 E7.5
particles and are evaluated on the same physical time grid (0.0, ..., 2.5).
COATI uses its saved paired RNA/ATAC trajectories.  CytoBridge is integrated
in RNA space and then mapped at every physical time through the same frozen
FiLM map T used throughout the Gastrulation benchmark.

The streamlines summarize vector-field topology with uniform segment
contribution.  Native mass is intentionally not encoded here; mass-weighted
composition and coverage are reported in separate figures.
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
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

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
DEFAULT_CYTOBRIDGE = (
    ROOT
    / "results"
    / "cytobridge_gastrulation_rna_20000_unbalanced"
    / "adata.h5ad"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "publication_figures"
    / "gastrulation_unbalanced_dual_umap_flow"
)

for path in (ROOT / "scripts", ROOT / "external" / "CytoBridge"):
    if path.exists():
        sys.path.insert(0, str(path))

from evaluate_gastrulation_flow_methods_normalized import (  # noqa: E402
    _cytobridge_interval_path,
    _load_cytobridge_model,
)
from evaluate_gastrulation_full_cmcc import (  # noqa: E402
    _apply_t,
    _load_t_model,
)
from plot_gastrulation_coati_dual_umap_flow import (  # noqa: E402
    add_streamlines,
    fixed_umap_interpolator,
    grid_field,
    load_celltypes,
    load_concat_npz,
    load_scale,
    plot_limits,
    scatter_celltypes,
    style_panel,
)


PHYSICAL_TIMES = np.arange(0.0, 2.5 + 1e-8, 0.1, dtype=np.float32)
INTERVALS = ((0.0, 1.0, 10), (1.0, 2.0, 10), (2.0, 2.5, 5))
A4_WIDTH_IN = 210.0 / 25.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cy", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=20000)
    parser.add_argument("--max-particles", type=int, default=3000)
    parser.add_argument("--stream-density", type=float, default=1.30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--cytobridge-adata",
        type=Path,
        default=DEFAULT_CYTOBRIDGE,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default="cpu",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load_tensor(path: Path) -> np.ndarray:
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    return np.asarray(tensor, dtype=np.float32)


def _chosen_indices(n_particles: int, max_particles: int, seed: int) -> np.ndarray:
    if n_particles <= max_particles:
        return np.arange(n_particles, dtype=np.int64)
    rng = np.random.default_rng(104 + seed)
    return np.sort(
        rng.choice(n_particles, size=max_particles, replace=False)
    ).astype(np.int64)


def _embed_pairs(interpolator, trajectory_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_steps, n_particles, dim = trajectory_raw.shape
    embedded = interpolator.predict(
        trajectory_raw.reshape(n_steps * n_particles, dim)
    ).astype(np.float32)
    embedded = embedded.reshape(n_steps, n_particles, 2)
    starts = embedded[:-1].reshape(-1, 2)
    displacements = (embedded[1:] - embedded[:-1]).reshape(-1, 2)
    return starts, displacements


def _rollout_cytobridge(
    model,
    x0_norm: np.ndarray,
    rna_scale: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    current = np.asarray(x0_norm, dtype=np.float32)
    for interval_index, (t0, t1, steps) in enumerate(INTERVALS):
        path = _cytobridge_interval_path(
            model,
            current,
            t0,
            t1,
            steps,
            rna_scale,
            batch_size,
            device,
        )
        pieces.append(path if interval_index == 0 else path[1:])
        current = path[-1]
    trajectory = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
    if trajectory.shape[0] != len(PHYSICAL_TIMES):
        raise RuntimeError(
            f"Unexpected CytoBridge time grid length: {trajectory.shape[0]}"
        )
    return trajectory


def _map_trajectory_with_t(
    model: torch.nn.Module,
    rna_trajectory_norm: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    mapped = [
        _apply_t(
            model,
            rna_trajectory_norm[index],
            float(time),
            device,
            batch_size,
        )
        for index, time in enumerate(PHYSICAL_TIMES)
    ]
    return np.stack(mapped, axis=0).astype(np.float32, copy=False)


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
    time_path = COATI_DIR / f"t_grid_s{args.seed}_iter{args.iteration}.pt"
    for path in (primary_path, secondary_path, time_path, args.cytobridge_adata):
        if not path.exists():
            raise FileNotFoundError(path)

    coati_rna_norm = _load_tensor(primary_path)
    coati_atac_norm = _load_tensor(secondary_path)
    time_grid = _load_tensor(time_path).reshape(-1)
    if not np.allclose(time_grid, PHYSICAL_TIMES):
        raise RuntimeError("COATI time grid is not 0.0, 0.1, ..., 2.5")
    if coati_rna_norm.shape[:2] != coati_atac_norm.shape[:2]:
        raise RuntimeError("COATI RNA and ATAC trajectories are not aligned")

    chosen = _chosen_indices(
        coati_rna_norm.shape[1],
        args.max_particles,
        args.seed,
    )
    coati_rna_norm = coati_rna_norm[:, chosen]
    coati_atac_norm = coati_atac_norm[:, chosen]

    rna_scale = load_scale(DATA / "primal_norm_params.pt")
    atac_scale = load_scale(DATA / "secondary_norm_params.pt")
    cytobridge_model = _load_cytobridge_model(args.cytobridge_adata, device)
    cytobridge_rna_norm = _rollout_cytobridge(
        cytobridge_model,
        coati_rna_norm[0],
        rna_scale,
        args.batch_size,
        device,
    )
    t_model = _load_t_model(device)
    cytobridge_atac_norm = _map_trajectory_with_t(
        t_model,
        cytobridge_rna_norm,
        args.batch_size,
        device,
    )

    rna_latent = load_concat_npz(DATA / "rna_pca_by_time.npz")
    atac_latent = load_concat_npz(DATA / "atac_lsi_by_time_14D.npz")
    celltypes = load_celltypes(DATA / "celltype_sub_by_stage.npz")
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

    trajectory_pairs = {
        ("COATI unbal.", "RNA"): _embed_pairs(
            rna_interpolator,
            coati_rna_norm * np.float32(rna_scale),
        ),
        ("COATI unbal.", "ATAC"): _embed_pairs(
            atac_interpolator,
            coati_atac_norm * np.float32(atac_scale),
        ),
        ("CytoBridge unbal.", "RNA"): _embed_pairs(
            rna_interpolator,
            cytobridge_rna_norm * np.float32(rna_scale),
        ),
        ("CytoBridge unbal.", "ATAC"): _embed_pairs(
            atac_interpolator,
            cytobridge_atac_norm * np.float32(atac_scale),
        ),
    }

    fields = {}
    for (method, modality), (points, vectors) in trajectory_pairs.items():
        xlim, ylim = (
            (rna_xlim, rna_ylim)
            if modality == "RNA"
            else (atac_xlim, atac_ylim)
        )
        fields[(method, modality)] = grid_field(
            points,
            vectors,
            xlim,
            ylim,
        )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(A4_WIDTH_IN, 2.18),
        squeeze=False,
    )
    axes = axes.ravel()
    methods = ("COATI unbal.", "CytoBridge unbal.")
    modalities = ("RNA", "ATAC")
    panel_index = 0
    for method in methods:
        for modality in modalities:
            ax = axes[panel_index]
            embedding, xlim, ylim = (
                (rna_embedding, rna_xlim, rna_ylim)
                if modality == "RNA"
                else (atac_embedding, atac_xlim, atac_ylim)
            )
            scatter_celltypes(ax, embedding, celltypes)
            add_streamlines(
                ax,
                fields[(method, modality)],
                args.stream_density,
            )
            style_panel(
                ax,
                f"{modality} space",
                xlim,
                ylim,
            )
            panel_index += 1

    fig.text(
        0.25,
        0.982,
        methods[0],
        ha="center",
        va="top",
        fontsize=10,
        fontweight="normal",
    )
    fig.text(
        0.75,
        0.982,
        methods[1],
        ha="center",
        va="top",
        fontsize=10,
        fontweight="normal",
    )

    fig.subplots_adjust(
        left=0.004,
        right=0.996,
        bottom=0.008,
        top=0.84,
        wspace=0.025,
    )
    stem = (
        f"gastrulation_coati_unbal_cy{token.replace('.', 'p')}_"
        "cytobridge_unbal_rna_atac_umap_flow"
    )
    png_path = args.output_dir / f"{stem}.png"
    pdf_path = args.output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=400)
    fig.savefig(pdf_path)
    plt.close(fig)

    metadata = {
        "methods": ["COATI-USOT", "CytoBridge unbalanced"],
        "C_y": args.cy,
        "seed": args.seed,
        "iteration": args.iteration,
        "n_source_particles": int(len(chosen)),
        "source_particle_indices": chosen.tolist(),
        "physical_time_grid": PHYSICAL_TIMES.tolist(),
        "stream_density": args.stream_density,
        "stream_weighting": (
            "uniform segment contribution; native mass is intentionally "
            "not encoded in this topology figure"
        ),
        "coati_primary_trajectory": str(primary_path),
        "coati_secondary_trajectory": str(secondary_path),
        "cytobridge_adata": str(args.cytobridge_adata),
        "cytobridge_atac_definition": (
            "shared frozen T_FiLM(CytoBridge RNA trajectory, physical time)"
        ),
        "shared_t_checkpoint": str(DATA / "TrainT" / "T_FiLM.pt"),
        "rna_umap": str(DATA / "rna_umap_model.joblib"),
        "atac_umap": str(DATA / "atac_umap_model_14D.joblib"),
        "celltype_colors": str(
            ROOT / "scripts" / "trainfbench_plot_style.py"
        ),
        "typography": (
            "Arial; modality titles 12 pt; method labels 10 pt; normal weight"
        ),
        "layout": (
            f"1 x 4; A4 width ({A4_WIDTH_IN:.3f} inches); "
            "COATI RNA, COATI ATAC, CytoBridge RNA, CytoBridge ATAC"
        ),
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
