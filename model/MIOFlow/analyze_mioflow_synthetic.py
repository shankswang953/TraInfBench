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
import math
import shutil
import sys
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchdiffeq import odeint


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

from analyze_cytobridge_synthetic import (  # noqa: E402
    endpoint_metrics,
    load_film_model,
    load_scale,
    map_film_trajectory,
    observed_arrays,
    transform_shared_umap,
)
from evaluate_mioflow_terminal_push import (  # noqa: E402
    _as_tensor,
    _batched_slices,
    _decode_gaga_path,
    _encode_gaga,
    _load_gaga_model,
)
from mioflow.core.models.ode_model import ODEFunc  # noqa: E402


POPULATION_COLORS = {
    "4_1": "#4477AA",
    "4_5": "#EE7733",
    "5_2": "#228833",
    "5_3": "#CC3311",
}
SYNTHETIC_DIR = Path(
    "external/COATI/Synthetic/5scRNA"
)
TRAIN_T_DIR = Path(
    "external/COATI/Synthetic/TrainT"
)
CB_COMPARISON = ROOT / "results/cytobridge_synthetic_rna10_n128_i3000_comparison"
RNA10_UMAP_DIR = CB_COMPARISON / "rna10_umap"
TN_ANALYSIS = ROOT / "results/trajectorynet_synthetic_rna10_n128_i3000_analysis"


@torch.no_grad()
def rollout_mioflow_gaga(
    initial_normalized: np.ndarray,
    model_path: Path,
    device: torch.device,
    batch_size: int,
    steps_per_interval: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    """Return decoded RNA10, GAGA10, normalized model path, times, and GAGA."""
    state = torch.load(model_path, map_location=device, weights_only=False)
    if not bool(state.get("use_gaga", False)):
        raise ValueError(f"Expected a GAGA-enabled MIOFlow checkpoint: {model_path}")
    if str(state.get("input_space")) != "moscot-normalized":
        raise ValueError(
            "Synthetic GAGA evaluation expects checkpoint input_space=moscot-normalized"
        )

    gaga_path = Path(state["gaga_model_path"])
    if not gaga_path.is_file():
        gaga_path = model_path.parent / "gaga_model.pt"
    gaga = _load_gaga_model(gaga_path, device)
    latent0 = _encode_gaga(initial_normalized, gaga, device, batch_size)

    mean_vals = state["mean_vals"].detach().cpu().numpy().astype("float32")
    std_vals = state["std_vals"].detach().cpu().numpy().astype("float32")
    model0 = (latent0 - mean_vals) / std_vals
    model_times = np.asarray(state["time_points"], dtype=np.float32)
    if model_times.shape != (4,) or not np.allclose(model_times, [0, 1, 2, 3]):
        raise ValueError(f"Unexpected MIOFlow clock: {model_times.tolist()}")
    physical_times = np.linspace(
        0.0, 3.0, 3 * steps_per_interval + 1, dtype=np.float32
    )
    integration_times = torch.as_tensor(physical_times, device=device)

    ode_model = ODEFunc(
        input_dim=int(state["input_dim"]),
        hidden_dim=int(state["hidden_dim"]),
        momentum_beta=float(state.get("momentum_beta", 0.0)),
    ).to(device)
    ode_model.load_state_dict(state["model_state_dict"])
    ode_model.eval()

    paths = []
    for batch_slice in _batched_slices(model0.shape[0], batch_size):
        if hasattr(ode_model, "reset_momentum"):
            ode_model.reset_momentum()
        path = odeint(
            ode_model,
            _as_tensor(model0[batch_slice], device),
            integration_times,
        )
        paths.append(path.detach().cpu().numpy())
    model_path_normalized = np.concatenate(paths, axis=1).astype(
        "float32", copy=False
    )
    latent_path = model_path_normalized * std_vals + mean_vals
    decoded_normalized = _decode_gaga_path(
        latent_path, gaga, device, batch_size
    ).astype("float32", copy=False)
    return (
        decoded_normalized,
        latent_path.astype("float32", copy=False),
        model_path_normalized,
        physical_times,
        gaga,
    )


def add_background(axis: plt.Axes, points: np.ndarray, population: np.ndarray) -> None:
    for name in sorted(np.unique(population)):
        mask = population == name
        axis.scatter(
            points[mask, 0],
            points[mask, 1],
            s=6,
            alpha=0.22,
            color=POPULATION_COLORS[name],
            linewidths=0,
            rasterized=True,
            label=name,
        )


def add_paths(axis: plt.Axes, trajectory: np.ndarray, color: str) -> None:
    for index in range(trajectory.shape[1]):
        axis.plot(
            trajectory[:, index, 0],
            trajectory[:, index, 1],
            color=color,
            alpha=0.34,
            linewidth=0.75,
        )
    axis.scatter(
        trajectory[0, :, 0],
        trajectory[0, :, 1],
        s=8,
        color="#111111",
        alpha=0.65,
        linewidths=0,
    )
    axis.scatter(
        trajectory[-1, :, 0],
        trajectory[-1, :, 1],
        s=13,
        color=color,
        alpha=0.82,
        linewidths=0,
    )


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_standalone(
    observed_umap: np.ndarray,
    observed_atac: np.ndarray,
    population: np.ndarray,
    trajectory_umap: np.ndarray,
    trajectory_atac: np.ndarray,
    metrics: dict,
    iterations: int,
    num_samples: int,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14.8, 6.7), dpi=220)
    add_background(axes[0], observed_umap, population)
    add_paths(axes[0], trajectory_umap, "#009E73")
    axes[0].set_title(
        "MIOFlow GAGA10→decoder — shared RNA10 UMAP\n"
        f"route={metrics['rna']['route_consistent_fraction']:.1%}"
    )
    axes[0].set_xlabel("RNA10 UMAP1")
    axes[0].set_ylabel("RNA10 UMAP2")
    axes[0].legend(frameon=False, markerscale=2.5, loc="best")

    add_background(axes[1], observed_atac[:, :2], population)
    add_paths(axes[1], trajectory_atac[..., :2], "#009E73")
    axes[1].set_title(
        "MIOFlow GAGA10→decoder→FiLM — ATAC8\n"
        f"route={metrics['atac']['route_consistent_fraction']:.1%}"
    )
    axes[1].set_xlabel("Selected ATAC PC1")
    axes[1].set_ylabel("Selected ATAC PC3")
    for axis in axes:
        style_axis(axis)
    figure.suptitle(
        f"MIOFlow synthetic RNA10 — {iterations} epochs, num_samples={num_samples}",
        fontsize=16,
        y=1.01,
    )
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_four_model_comparison(
    observed_umap: np.ndarray,
    observed_atac: np.ndarray,
    population: np.ndarray,
    methods: list[dict],
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(27.0, 12.5), dpi=180)
    for column, method in enumerate(methods):
        add_background(axes[0, column], observed_umap, population)
        add_paths(axes[0, column], method["rna_umap"], method["color"])
        axes[0, column].set_title(
            f"{method['title']} — RNA10 UMAP\n"
            f"route={method['metrics']['rna']['route_consistent_fraction']:.1%}"
        )
        axes[0, column].set_xlabel("RNA10 UMAP1")
        axes[0, column].set_ylabel("RNA10 UMAP2")
        if column == 0:
            axes[0, column].legend(frameon=False, markerscale=2.3, loc="best")

        add_background(axes[1, column], observed_atac[:, :2], population)
        add_paths(axes[1, column], method["atac_raw"][..., :2], method["color"])
        axes[1, column].set_title(
            f"{method['title']} — FiLM ATAC8\n"
            f"route={method['metrics']['atac']['route_consistent_fraction']:.1%}"
        )
        axes[1, column].set_xlabel("Selected ATAC PC1")
        axes[1, column].set_ylabel("Selected ATAC PC3")
    for axis in axes.flat:
        style_axis(axis)
    figure.suptitle(
        "Synthetic RNA10 model comparison — 3000 iterations/epochs, 128 trajectories",
        fontsize=18,
        y=1.003,
    )
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_training_losses(result_dir: Path, output: Path) -> None:
    mioflow = pd.read_csv(result_dir / "losses.csv")
    gaga = pd.read_csv(result_dir / "gaga_losses.csv")
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), dpi=200)
    axes[0].plot(mioflow["epoch"], mioflow["total_loss"], label="total")
    axes[0].plot(mioflow["epoch"], mioflow["ot_loss"], label="OT", alpha=0.85)
    axes[0].plot(
        mioflow["epoch"], 0.01 * mioflow["energy_loss"], label="0.01 × energy", alpha=0.85
    )
    axes[0].set_title("MIOFlow training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False)

    for phase, frame in gaga.groupby("phase", sort=False):
        axes[1].plot(frame["epoch"], frame["train_loss"], label=str(phase))
    axes[1].set_title("GAGA two-phase training")
    axes[1].set_xlabel("Epoch within phase")
    axes[1].set_ylabel("Loss")
    axes[1].legend(frameon=False)
    for axis in axes:
        style_axis(axis)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze GAGA-enabled MIOFlow on the synthetic RNA10 example."
    )
    parser.add_argument(
        "--input-h5ad", type=Path, default=Path("data/synthetic_rna10_cytobridge.h5ad")
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("results/mioflow_synthetic_rna10_gaga10_n128_i3000/model.pt"),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/mioflow_synthetic_rna10_gaga10_n128_i3000"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mioflow_synthetic_rna10_gaga10_n128_i3000_analysis"),
    )
    parser.add_argument(
        "--rna-data", type=Path, default=SYNTHETIC_DIR / "all_time_scRNA_pca10.npz"
    )
    parser.add_argument(
        "--atac-data",
        type=Path,
        default=SYNTHETIC_DIR / "all_time_scATAC_pca8_no_pc2.npz",
    )
    parser.add_argument(
        "--labels", type=Path, default=SYNTHETIC_DIR / "all_time_scRNA_label.npz"
    )
    parser.add_argument(
        "--rna-norm", type=Path, default=SYNTHETIC_DIR / "primal_norm_params_rna10_w2.pt"
    )
    parser.add_argument(
        "--atac-norm",
        type=Path,
        default=SYNTHETIC_DIR / "secondary_norm_params_atac8_no_pc2_w2.pt",
    )
    parser.add_argument(
        "--film-model-file", type=Path, default=TRAIN_T_DIR / "film_model.py"
    )
    parser.add_argument(
        "--film-checkpoint",
        type=Path,
        default=TRAIN_T_DIR / "outputs/T_FiLM_rna10_atac8_no_pc2.pt",
    )
    parser.add_argument(
        "--umap-model", type=Path, default=RNA10_UMAP_DIR / "rna10_umap_model.joblib"
    )
    parser.add_argument(
        "--observed-umap", type=Path, default=RNA10_UMAP_DIR / "observed_rna10_umap.npz"
    )
    parser.add_argument(
        "--umap-python", type=Path, default=Path(_sys.executable)
    )
    parser.add_argument(
        "--umap-transform-script",
        type=Path,
        default=ROOT / "common/transform_joblib_umap.py",
    )
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--steps-per-interval", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    required = (
        args.input_h5ad,
        args.model,
        args.result_dir / "gaga_model.pt",
        args.result_dir / "losses.csv",
        args.result_dir / "gaga_losses.csv",
        args.rna_data,
        args.atac_data,
        args.labels,
        args.rna_norm,
        args.atac_norm,
        args.film_model_file,
        args.film_checkpoint,
        args.umap_model,
        args.observed_umap,
        args.umap_python,
        args.umap_transform_script,
        CB_COMPARISON / "cytobridge_trajectories.npz",
        CB_COMPARISON / "metrics.json",
        RNA10_UMAP_DIR / "cytobridge_rna10_umap_trajectories.npz",
        TN_ANALYSIS / "trajectorynet_trajectories.npz",
        TN_ANALYSIS / "metrics.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    device = torch.device(args.device)
    adata = ad.read_h5ad(args.input_h5ad)
    rna_scale = load_scale(args.rna_norm)
    atac_scale = load_scale(args.atac_norm)
    rna_raw, atac_raw, population, observed_time = observed_arrays(
        args.rna_data, args.atac_data, args.labels
    )
    normalized = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    source_raw = np.asarray(adata.obsm["X_pca_raw"], dtype=np.float32)
    if not np.allclose(source_raw, rna_raw, rtol=1e-6, atol=1e-6):
        raise ValueError("MIOFlow source X_pca_raw does not match RNA10 reference")
    if not np.allclose(normalized * rna_scale, rna_raw, rtol=1e-6, atol=1e-6):
        raise ValueError("MIOFlow normalized RNA10 does not match RNA10 reference")

    cb_archive = np.load(CB_COMPARISON / "cytobridge_trajectories.npz", allow_pickle=True)
    selected = np.asarray(cb_archive["selected_initial_indices_time1"], dtype=np.int64)
    first_mask = observed_time == 0
    initial_all = normalized[first_mask]
    initial_selected = initial_all[selected]
    first_population = population[first_mask]

    (
        selected_path_norm,
        selected_path_gaga,
        selected_path_model,
        physical_times,
        gaga,
    ) = rollout_mioflow_gaga(
        initial_selected, args.model, device, args.batch_size, args.steps_per_interval
    )
    all_path_norm, _, _, _, _ = rollout_mioflow_gaga(
        initial_all, args.model, device, args.batch_size, args.steps_per_interval
    )
    selected_path_raw = selected_path_norm * rna_scale
    all_endpoint_raw = all_path_norm[-1] * rna_scale

    selected_umap = transform_shared_umap(
        args.umap_python,
        args.umap_transform_script,
        args.umap_model,
        selected_path_raw.reshape(-1, 10),
    ).reshape(len(physical_times), len(selected), 2)
    observed_umap = np.asarray(
        np.load(args.observed_umap, allow_pickle=True)["observed_umap"], dtype=np.float32
    )
    film = load_film_model(args.film_model_file, args.film_checkpoint)
    selected_atac_norm = map_film_trajectory(film, selected_path_norm, physical_times)
    selected_atac_raw = selected_atac_norm * atac_scale
    all_atac_raw = (
        film(torch.as_tensor(all_path_norm[-1]), 3.0).detach().cpu().numpy()
        * atac_scale
    ).astype(np.float32)

    uniform_log_weights = np.full(
        len(initial_all), -math.log(len(initial_all)), dtype=np.float32
    )
    encoded_all = _encode_gaga(normalized, gaga, device, args.batch_size)
    decoded_all = _decode_gaga_path(
        encoded_all[None, ...], gaga, device, args.batch_size
    )[0]
    reconstruction_residual = decoded_all - normalized
    reconstruction_r2 = 1.0 - float(np.sum(reconstruction_residual**2)) / float(
        np.sum((normalized - normalized.mean(axis=0, keepdims=True)) ** 2)
    )
    losses = pd.read_csv(args.result_dir / "losses.csv")
    gaga_losses = pd.read_csv(args.result_dir / "gaga_losses.csv")
    method_metrics = {
        "rna": endpoint_metrics(
            all_endpoint_raw,
            rna_raw[observed_time == 3],
            population[observed_time == 3],
            first_population,
            uniform_log_weights,
        ),
        "atac": endpoint_metrics(
            all_atac_raw,
            atac_raw[observed_time == 3],
            population[observed_time == 3],
            first_population,
            uniform_log_weights,
        ),
        "components": ["GAGA10 encoder", "MIOFlow neural ODE", "GAGA decoder"],
        "n_all_initial_cells": int(len(initial_all)),
        "n_visualized_trajectories": int(len(selected)),
        "training_space": "GAGA10 learned from normalized RNA10",
        "returned_space": "GAGA decoder output in normalized RNA10",
        "decoder_reconstruction": {
            "normalized_rmse": float(np.sqrt(np.mean(reconstruction_residual**2))),
            "raw_rmse": float(np.sqrt(np.mean((reconstruction_residual * rna_scale) ** 2))),
            "r2": reconstruction_r2,
        },
        "final_training_loss": float(losses.iloc[-1]["total_loss"]),
        "final_ot_loss": float(losses.iloc[-1]["ot_loss"]),
        "final_energy_loss": float(losses.iloc[-1]["energy_loss"]),
        "gaga_final_encoder_loss": float(
            gaga_losses[gaga_losses["phase"] == "phase1"].iloc[-1]["train_loss"]
        ),
        "gaga_final_decoder_loss": float(
            gaga_losses[gaga_losses["phase"] == "phase2"].iloc[-1]["train_loss"]
        ),
    }

    arrays_to_check = (
        selected_path_norm,
        selected_path_gaga,
        selected_path_model,
        selected_path_raw,
        selected_umap,
        selected_atac_raw,
        all_endpoint_raw,
        all_atac_raw,
    )
    if not all(np.isfinite(values).all() for values in arrays_to_check):
        raise FloatingPointError("Non-finite values found in MIOFlow analysis")

    np.savez_compressed(
        args.output_dir / "mioflow_gaga10_trajectories.npz",
        physical_times=physical_times,
        selected_initial_indices_time1=selected,
        selected_initial_population=first_population[selected],
        trajectory_gaga10=selected_path_gaga,
        trajectory_gaga10_model_normalized=selected_path_model,
        trajectory_rna10_normalized=selected_path_norm,
        trajectory_rna10_raw=selected_path_raw,
        trajectory_rna10_umap=selected_umap,
        trajectory_atac8_normalized=selected_atac_norm,
        trajectory_atac8_raw=selected_atac_raw,
        all_initial_endpoint_rna10_raw=all_endpoint_raw,
        all_initial_endpoint_atac8_raw=all_atac_raw,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"mioflow_gaga10": method_metrics}, indent=2) + "\n"
    )
    plot_standalone(
        observed_umap,
        atac_raw,
        population,
        selected_umap,
        selected_atac_raw,
        method_metrics,
        args.iterations,
        args.num_samples,
        args.output_dir / "mioflow_gaga10_shared_rna10_umap_and_atac.png",
    )
    plot_training_losses(
        args.result_dir, args.output_dir / "mioflow_gaga10_training_losses.png"
    )

    cb_umap = np.load(
        RNA10_UMAP_DIR / "cytobridge_rna10_umap_trajectories.npz", allow_pickle=True
    )
    cb_metrics = json.loads((CB_COMPARISON / "metrics.json").read_text())
    tn_archive = np.load(TN_ANALYSIS / "trajectorynet_trajectories.npz", allow_pickle=True)
    tn_metrics = json.loads((TN_ANALYSIS / "metrics.json").read_text())["trajectorynet"]
    methods = [
        {
            "title": "CytoBridge balanced",
            "color": "#0072B2",
            "rna_umap": np.asarray(cb_umap["balanced_trajectory_rna10_umap"]),
            "atac_raw": np.asarray(cb_archive["balanced_trajectory_atac8_raw"]),
            "metrics": cb_metrics["balanced"],
        },
        {
            "title": "CytoBridge unbalanced",
            "color": "#D55E00",
            "rna_umap": np.asarray(cb_umap["unbalanced_trajectory_rna10_umap"]),
            "atac_raw": np.asarray(cb_archive["unbalanced_trajectory_atac8_raw"]),
            "metrics": cb_metrics["unbalanced"],
        },
        {
            "title": "TrajectoryNet",
            "color": "#7A5195",
            "rna_umap": np.asarray(tn_archive["trajectory_rna10_umap"]),
            "atac_raw": np.asarray(tn_archive["trajectory_atac8_raw"]),
            "metrics": tn_metrics,
        },
        {
            "title": "MIOFlow GAGA10→decoder",
            "color": "#009E73",
            "rna_umap": selected_umap,
            "atac_raw": selected_atac_raw,
            "metrics": method_metrics,
        },
    ]
    plot_four_model_comparison(
        observed_umap,
        atac_raw,
        population,
        methods,
        args.output_dir / "cytobridge_trajectorynet_mioflow_four_model_comparison.png",
    )
    manifest = {
        "input_h5ad": str(args.input_h5ad.resolve()),
        "model": str(args.model.resolve()),
        "gaga_model": str((args.result_dir / "gaga_model.pt").resolve()),
        "iterations": args.iterations,
        "num_samples": args.num_samples,
        "seed": 0,
        "pipeline": "normalized RNA10 -> GAGA10 -> MIOFlow -> GAGA decoder -> normalized RNA10",
        "gaga_encoder_epochs": int(
            gaga_losses[gaga_losses["phase"] == "phase1"]["epoch"].max()
        ),
        "gaga_decoder_epochs": int(
            gaga_losses[gaga_losses["phase"] == "phase2"]["epoch"].max()
        ),
        "rollout_steps_per_interval": args.steps_per_interval,
        "umap_model": str(args.umap_model.resolve()),
        "film_checkpoint": str(args.film_checkpoint.resolve()),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"mioflow_gaga10": method_metrics}, indent=2))


if __name__ == "__main__":
    main()
