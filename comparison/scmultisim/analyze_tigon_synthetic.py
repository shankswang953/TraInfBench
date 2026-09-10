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
from analyze_mioflow_synthetic import add_background, add_paths, style_axis  # noqa: E402
from evaluate_tigon_gastrulation_normalized import (  # noqa: E402
    _load_model,
    _load_tigon_module,
)


SYNTHETIC_DIR = Path(
    "external/COATI/Synthetic/5scRNA"
)
TRAIN_T_DIR = Path(
    "external/COATI/Synthetic/TrainT"
)
CB_COMPARISON = ROOT / "results/cytobridge_synthetic_rna10_n128_i3000_comparison"
RNA10_UMAP_DIR = CB_COMPARISON / "rna10_umap"
TN_ANALYSIS = ROOT / "results/trajectorynet_synthetic_rna10_n128_i3000_analysis"
MIO_ANALYSIS = ROOT / "results/mioflow_synthetic_rna10_gaga10_n128_i3000_analysis"


class TigonAugmentedDynamics(torch.nn.Module):
    def __init__(self, model, dimension: int):
        super().__init__()
        self.model = model
        self.dimension = dimension

    def forward(self, time, augmented):
        state = augmented[:, : self.dimension]
        velocity = self.model.velocity(time, state)
        growth = self.model.growth(time, state)
        return torch.cat([velocity, growth], dim=1)


@torch.no_grad()
def rollout_tigon(
    model,
    initial_normalized: np.ndarray,
    device: torch.device,
    batch_size: int,
    steps_per_interval: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dimension = initial_normalized.shape[1]
    physical_times = np.linspace(
        0.0, 3.0, 3 * steps_per_interval + 1, dtype=np.float32
    )
    integration_times = torch.as_tensor(physical_times, device=device)
    dynamics = TigonAugmentedDynamics(model, dimension)
    paths = []
    growth_paths = []
    for start in range(0, len(initial_normalized), batch_size):
        initial = torch.as_tensor(
            initial_normalized[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        augmented = torch.cat(
            [
                initial,
                torch.zeros(initial.shape[0], 1, dtype=initial.dtype, device=device),
            ],
            dim=1,
        )
        path = odeint(
            dynamics,
            augmented,
            integration_times,
            method="rk4",
            options={"step_size": 1.0 / steps_per_interval},
        )
        paths.append(path[..., :dimension].cpu().numpy())
        growth_paths.append(path[..., dimension].cpu().numpy())
    return (
        np.concatenate(paths, axis=1).astype("float32", copy=False),
        np.concatenate(growth_paths, axis=1).astype("float32", copy=False),
        physical_times,
    )


def plot_standalone(
    observed_umap: np.ndarray,
    observed_atac: np.ndarray,
    population: np.ndarray,
    trajectory_umap: np.ndarray,
    trajectory_atac: np.ndarray,
    metrics: dict,
    method_title: str,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14.8, 6.7), dpi=220)
    add_background(axes[0], observed_umap, population)
    add_paths(axes[0], trajectory_umap, "#CC79A7")
    axes[0].set_title(
        f"{method_title} — shared UMAP\n"
        f"route={metrics['rna']['route_consistent_fraction']:.1%}, "
        f"mass={metrics['rna']['mass_weighted_route_consistent_fraction']:.1%}"
    )
    axes[0].set_xlabel("RNA10 UMAP1")
    axes[0].set_ylabel("RNA10 UMAP2")
    axes[0].legend(frameon=False, markerscale=2.5, loc="best")

    add_background(axes[1], observed_atac[:, :2], population)
    add_paths(axes[1], trajectory_atac[..., :2], "#CC79A7")
    axes[1].set_title(
        "TIGON RNA10→FiLM — ATAC8\n"
        f"route={metrics['atac']['route_consistent_fraction']:.1%}, "
        f"mass={metrics['atac']['mass_weighted_route_consistent_fraction']:.1%}"
    )
    axes[1].set_xlabel("Selected ATAC PC1")
    axes[1].set_ylabel("Selected ATAC PC3")
    for axis in axes:
        style_axis(axis)
    figure.suptitle(
        f"{method_title} — 3000 iterations, num_samples=128",
        fontsize=16,
        y=1.01,
    )
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_five_model_comparison(
    observed_umap: np.ndarray,
    observed_atac: np.ndarray,
    population: np.ndarray,
    methods: list[dict],
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(32.0, 12.2), dpi=170)
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


def plot_growth_and_loss(
    history: pd.DataFrame,
    final_log_growth: np.ndarray,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), dpi=200)
    axes[0].plot(history["iter"], history["loss"], linewidth=1.0, alpha=0.55)
    smooth = history["loss"].rolling(51, center=True, min_periods=1).median()
    axes[0].plot(history["iter"], smooth, linewidth=2.0, color="#CC79A7")
    axes[0].set_title("TIGON density objective")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")

    axes[1].hist(final_log_growth, bins=50, color="#CC79A7", alpha=0.8)
    axes[1].set_title("Final cumulative log-growth")
    axes[1].set_xlabel("∫ growth dt")
    axes[1].set_ylabel("Particles")
    for axis in axes:
        style_axis(axis)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze direct-RNA10 TIGON on the synthetic example."
    )
    parser.add_argument(
        "--input-h5ad", type=Path, default=Path("data/synthetic_rna10_cytobridge.h5ad")
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/tigon_synthetic_rna10_direct_n128_i3000"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tigon_synthetic_rna10_direct_n128_i3000_analysis"),
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
    parser.add_argument("--steps-per-interval", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_path = args.result_dir / "tigon.pt"
    config_path = args.result_dir / "config.json"
    history_path = args.result_dir / "tigon_training_history.csv"
    required = (
        args.input_h5ad,
        model_path,
        config_path,
        history_path,
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
        MIO_ANALYSIS / "mioflow_gaga10_trajectories.npz",
        MIO_ANALYSIS / "metrics.json",
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
    normalized_w2 = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    raw_pca_input = np.asarray(adata.obsm["X_pca_raw"], dtype=np.float32)
    times = np.asarray(adata.obs["time_point_processed"], dtype=np.float32)
    rna_scale = load_scale(args.rna_norm)
    atac_scale = load_scale(args.atac_norm)
    rna_raw, atac_raw, population, observed_time = observed_arrays(
        args.rna_data, args.atac_data, args.labels
    )
    if not np.allclose(normalized_w2 * rna_scale, rna_raw, rtol=1e-6, atol=1e-6):
        raise ValueError("TIGON direct input does not match normalized RNA10 reference")
    if not np.allclose(raw_pca_input, rna_raw, rtol=1e-6, atol=1e-6):
        raise ValueError("X_pca_raw does not match the original RNA10 PCA reference")
    if not np.allclose(times, observed_time):
        raise ValueError("TIGON time labels do not match the RNA10 reference")

    tigon_module = _load_tigon_module()
    model, config, state = _load_model(tigon_module, model_path, config_path, device)
    if bool(config.get("use_ae", True)):
        raise ValueError("Expected direct RNA10 TIGON with use_ae=false")
    normalization_mode = str(config.get("embedding_normalization"))
    official_scaled = normalization_mode == "per-axis-minmax-minus2-2"
    if official_scaled:
        if str(config.get("embedding_key")) != "X_pca_raw":
            raise ValueError(
                "Official scaled-PCA run must read the original X_pca_raw embedding"
            )
        axis_min = np.asarray(
            config.get("resolved_embedding_normalization_axis_min"), dtype=np.float32
        )
        axis_max = np.asarray(
            config.get("resolved_embedding_normalization_axis_max"), dtype=np.float32
        )
        if axis_min.shape != (10,) or axis_max.shape != (10,):
            raise ValueError("Missing 10D per-axis scaling parameters in TIGON config")
        if not np.allclose(axis_min, raw_pca_input.min(axis=0), rtol=0, atol=1e-6):
            raise ValueError("Saved TIGON axis minima do not match full RNA10 PCA data")
        if not np.allclose(axis_max, raw_pca_input.max(axis=0), rtol=0, atol=1e-6):
            raise ValueError("Saved TIGON axis maxima do not match full RNA10 PCA data")
        axis_range = axis_max - axis_min
        model_space = (
            4.0 * (raw_pca_input - axis_min[None, :]) / axis_range[None, :] - 2.0
        ).astype(np.float32)
        method_title = "TIGON PCA10 scaled [-2,2]"
        training_space = (
            "original RNA PCA10 -> full-data per-axis min-max scaling to [-2,2] "
            "-> TIGON -> inverse scaling to RNA PCA10"
        )
    else:
        if str(config.get("embedding_key")) != "X_latent":
            raise ValueError(f"Unexpected TIGON embedding key: {config.get('embedding_key')}")
        if normalization_mode != "identity":
            raise ValueError("Expected identity normalization on normalized RNA10")
        axis_min = axis_max = axis_range = None
        model_space = normalized_w2
        method_title = "TIGON direct normalized RNA10"
        training_space = "direct normalized RNA10; no autoencoder or decoder"
    if int(state.get("model_dim", 0)) != 10:
        raise ValueError(f"Expected TIGON model_dim=10, got {state.get('model_dim')}")

    cb_archive = np.load(CB_COMPARISON / "cytobridge_trajectories.npz", allow_pickle=True)
    selected = np.asarray(cb_archive["selected_initial_indices_time1"], dtype=np.int64)
    first_mask = observed_time == 0
    initial_all = model_space[first_mask]
    initial_selected = initial_all[selected]
    first_population = population[first_mask]

    selected_path_model, selected_growth_path, physical_times = rollout_tigon(
        model, initial_selected, device, args.batch_size, args.steps_per_interval
    )
    all_path_model, all_growth_path, _ = rollout_tigon(
        model, initial_all, device, args.batch_size, args.steps_per_interval
    )
    selected_log_growth = np.clip(selected_growth_path, -12.0, 12.0)
    all_final_log_growth = np.clip(all_growth_path[-1], -12.0, 12.0)
    if official_scaled:
        selected_path_raw = (
            (selected_path_model + 2.0) * axis_range[None, None, :] / 4.0
            + axis_min[None, None, :]
        ).astype(np.float32)
        all_endpoint_raw = (
            (all_path_model[-1] + 2.0) * axis_range[None, :] / 4.0
            + axis_min[None, :]
        ).astype(np.float32)
    else:
        selected_path_raw = selected_path_model * rna_scale
        all_endpoint_raw = all_path_model[-1] * rna_scale
    selected_path_norm = (selected_path_raw / rna_scale).astype(np.float32)
    all_endpoint_norm = (all_endpoint_raw / rna_scale).astype(np.float32)

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
        film(torch.as_tensor(all_endpoint_norm), 3.0).detach().cpu().numpy()
        * atac_scale
    ).astype(np.float32)

    final_weights = np.exp(all_final_log_growth - np.max(all_final_log_growth))
    final_weights = final_weights / final_weights.sum()
    effective_sample_size = float(1.0 / np.sum(final_weights**2))
    method_metrics = {
        "rna": endpoint_metrics(
            all_endpoint_raw,
            rna_raw[observed_time == 3],
            population[observed_time == 3],
            first_population,
            all_final_log_growth,
        ),
        "atac": endpoint_metrics(
            all_atac_raw,
            atac_raw[observed_time == 3],
            population[observed_time == 3],
            first_population,
            all_final_log_growth,
        ),
        "components": ["TIGON velocity", "TIGON growth"],
        "n_all_initial_cells": int(len(initial_all)),
        "n_visualized_trajectories": int(len(selected)),
        "training_space": training_space,
        "movement": {
            "mean_endpoint_displacement_model_space": float(
                np.mean(np.linalg.norm(all_path_model[-1] - all_path_model[0], axis=1))
            ),
            "median_endpoint_displacement_model_space": float(
                np.median(np.linalg.norm(all_path_model[-1] - all_path_model[0], axis=1))
            ),
            "mean_endpoint_displacement_raw_rna10": float(
                np.mean(np.linalg.norm(all_endpoint_raw - rna_raw[first_mask], axis=1))
            ),
            "median_endpoint_displacement_raw_rna10": float(
                np.median(np.linalg.norm(all_endpoint_raw - rna_raw[first_mask], axis=1))
            ),
        },
        "growth": {
            "mean_cumulative_log_growth": float(np.mean(all_final_log_growth)),
            "std_cumulative_log_growth": float(np.std(all_final_log_growth)),
            "mean_mass_factor": float(np.mean(np.exp(all_final_log_growth))),
            "effective_sample_size": effective_sample_size,
        },
        "training_objective": state.get("training_objective", config.get("training_objective")),
        "ode_solver": state.get("ode_solver", config.get("ode_solver")),
    }
    history = pd.read_csv(history_path)
    method_metrics["final_training_loss"] = float(history.iloc[-1]["loss"])
    for key in ("transport_cost", "long_term_reconstruction", "short_term_reconstruction"):
        if key in history:
            method_metrics[f"final_{key}"] = float(history.iloc[-1][key])

    arrays_to_check = (
        selected_path_model,
        selected_path_norm,
        selected_growth_path,
        selected_path_raw,
        selected_umap,
        selected_atac_raw,
        all_endpoint_raw,
        all_atac_raw,
        all_growth_path,
    )
    if not all(np.isfinite(values).all() for values in arrays_to_check):
        raise FloatingPointError("Non-finite values found in TIGON analysis")

    np.savez_compressed(
        args.output_dir / "tigon_trajectories.npz",
        physical_times=physical_times,
        selected_initial_indices_time1=selected,
        selected_initial_population=first_population[selected],
        trajectory_tigon_model_space=selected_path_model,
        trajectory_rna10_normalized=selected_path_norm,
        trajectory_rna10_raw=selected_path_raw,
        trajectory_rna10_umap=selected_umap,
        trajectory_cumulative_log_growth=selected_growth_path,
        trajectory_clipped_log_weights=selected_log_growth,
        trajectory_atac8_normalized=selected_atac_norm,
        trajectory_atac8_raw=selected_atac_raw,
        all_initial_endpoint_rna10_raw=all_endpoint_raw,
        all_initial_endpoint_atac8_raw=all_atac_raw,
        all_initial_endpoint_cumulative_log_growth=all_growth_path[-1],
        all_initial_endpoint_clipped_log_weights=all_final_log_growth,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"tigon": method_metrics}, indent=2) + "\n"
    )
    plot_standalone(
        observed_umap,
        atac_raw,
        population,
        selected_umap,
        selected_atac_raw,
        method_metrics,
        method_title,
        args.output_dir / "tigon_shared_rna10_umap_and_atac.png",
    )
    plot_growth_and_loss(
        history,
        all_final_log_growth,
        args.output_dir / "tigon_training_loss_and_growth.png",
    )

    cb_umap = np.load(
        RNA10_UMAP_DIR / "cytobridge_rna10_umap_trajectories.npz", allow_pickle=True
    )
    cb_metrics = json.loads((CB_COMPARISON / "metrics.json").read_text())
    tn_archive = np.load(TN_ANALYSIS / "trajectorynet_trajectories.npz", allow_pickle=True)
    tn_metrics = json.loads((TN_ANALYSIS / "metrics.json").read_text())["trajectorynet"]
    mio_archive = np.load(MIO_ANALYSIS / "mioflow_gaga10_trajectories.npz", allow_pickle=True)
    mio_metrics = json.loads((MIO_ANALYSIS / "metrics.json").read_text())["mioflow_gaga10"]
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
            "rna_umap": np.asarray(mio_archive["trajectory_rna10_umap"]),
            "atac_raw": np.asarray(mio_archive["trajectory_atac8_raw"]),
            "metrics": mio_metrics,
        },
        {
            "title": method_title,
            "color": "#CC79A7",
            "rna_umap": selected_umap,
            "atac_raw": selected_atac_raw,
            "metrics": method_metrics,
        },
    ]
    plot_five_model_comparison(
        observed_umap,
        atac_raw,
        population,
        methods,
        args.output_dir / "five_model_shared_rna10_atac8_comparison.png",
    )
    manifest = {
        "input_h5ad": str(args.input_h5ad.resolve()),
        "model": str(model_path.resolve()),
        "iterations": 3000,
        "num_samples": 128,
        "seed": 0,
        "pipeline": training_space,
        "embedding_normalization": normalization_mode,
        "embedding_normalization_axis_min": (
            None if axis_min is None else axis_min.tolist()
        ),
        "embedding_normalization_axis_max": (
            None if axis_max is None else axis_max.tolist()
        ),
        "training_ode_solver": config.get("ode_solver"),
        "training_ode_steps": config.get("ode_steps"),
        "divergence_estimator": config.get("divergence_estimator"),
        "rollout_steps_per_interval": args.steps_per_interval,
        "growth_log_weight_clip": [-12.0, 12.0],
        "umap_model": str(args.umap_model.resolve()),
        "film_checkpoint": str(args.film_checkpoint.resolve()),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"tigon": method_metrics}, indent=2))


if __name__ == "__main__":
    main()
