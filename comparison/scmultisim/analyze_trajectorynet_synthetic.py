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
from evaluate_terminal_push import (  # noqa: E402
    _load_trajectorynet_model,
    _trajectorynet_diffeq,
)


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


@torch.no_grad()
def integrate_trajectorynet(
    model,
    initial: np.ndarray,
    device: torch.device,
    batch_size: int,
    steps_per_interval: int,
) -> tuple[np.ndarray, np.ndarray]:
    diffeq = _trajectorynet_diffeq(model)
    chunks = []
    time_scale = 0.5
    physical_times = np.linspace(0.0, 3.0, 3 * steps_per_interval + 1, dtype=np.float32)
    internal_step = time_scale / steps_per_interval
    for start in range(0, len(initial), batch_size):
        x = torch.as_tensor(
            initial[start : start + batch_size], dtype=torch.float32, device=device
        )
        states = [x.detach().cpu().numpy()]
        for interval in range(3):
            upper = (interval + 2.0) * time_scale
            lower = (interval + 1.0) * time_scale
            integration_times = torch.linspace(
                upper,
                lower,
                steps_per_interval + 1,
                dtype=torch.float32,
                device=device,
            )
            path = odeint(
                diffeq,
                x,
                integration_times,
                method="rk4",
                options={"step_size": internal_step},
            )
            states.extend(state.detach().cpu().numpy() for state in path[1:])
            x = path[-1]
        chunks.append(np.stack(states).astype(np.float32))
    return np.concatenate(chunks, axis=1), physical_times


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


def plot_standalone(
    observed_umap: np.ndarray,
    observed_atac: np.ndarray,
    population: np.ndarray,
    trajectory_umap: np.ndarray,
    trajectory_atac: np.ndarray,
    metrics: dict,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14.8, 6.7), dpi=220)
    add_background(axes[0], observed_umap, population)
    add_paths(axes[0], trajectory_umap, "#7A5195")
    axes[0].set_title(
        "TrajectoryNet — shared RNA10 UMAP\n"
        f"route={metrics['rna']['route_consistent_fraction']:.1%}"
    )
    axes[0].set_xlabel("RNA10 UMAP1")
    axes[0].set_ylabel("RNA10 UMAP2")
    axes[0].legend(frameon=False, markerscale=2.5, loc="best")

    add_background(axes[1], observed_atac[:, :2], population)
    add_paths(axes[1], trajectory_atac[..., :2], "#7A5195")
    axes[1].set_title(
        "TrajectoryNet — FiLM-mapped ATAC8\n"
        f"route={metrics['atac']['route_consistent_fraction']:.1%}"
    )
    axes[1].set_xlabel("Selected ATAC PC1")
    axes[1].set_ylabel("Selected ATAC PC3")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle(
        "TrajectoryNet synthetic RNA10 — 3000 iterations, num_samples=128",
        fontsize=16,
        y=1.01,
    )
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_three_model_comparison(
    observed_umap: np.ndarray,
    observed_atac: np.ndarray,
    population: np.ndarray,
    methods: list[dict],
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(21.0, 12.8), dpi=200)
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
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle(
        "Synthetic RNA10 model comparison — 3000 iterations, 128 trajectories",
        fontsize=18,
        y=1.003,
    )
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze TrajectoryNet on the synthetic RNA10 example."
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/synthetic_rna10_trajectorynet.npz")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/trajectorynet_synthetic_rna10_n128_i3000/checkpt-3000.pth"),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/trajectorynet_synthetic_rna10_n128_i3000"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/trajectorynet_synthetic_rna10_n128_i3000_analysis"),
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
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--steps-per-interval", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    required = (
        args.dataset,
        args.checkpoint,
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
    dataset = np.load(args.dataset, allow_pickle=True)
    normalized = np.asarray(dataset["pca"], dtype=np.float32)
    labels_rank = np.asarray(dataset["sample_labels"], dtype=np.int64)
    rna_scale = load_scale(args.rna_norm)
    atac_scale = load_scale(args.atac_norm)
    rna_raw, atac_raw, population, observed_time = observed_arrays(
        args.rna_data, args.atac_data, args.labels
    )
    if not np.allclose(normalized * rna_scale, rna_raw, rtol=1e-6, atol=1e-6):
        raise ValueError("TrajectoryNet input does not match normalized RNA10 reference")

    cb_archive = np.load(CB_COMPARISON / "cytobridge_trajectories.npz", allow_pickle=True)
    selected = np.asarray(cb_archive["selected_initial_indices_time1"], dtype=np.int64)
    first_mask = labels_rank == 0
    last_mask = labels_rank == 3
    initial_all = normalized[first_mask]
    initial_selected = initial_all[selected]
    first_population = population[observed_time == 0]

    model, model_args = _load_trajectorynet_model(
        args.checkpoint, 10, device, "rk4", 0.1
    )
    selected_path_norm, physical_times = integrate_trajectorynet(
        model,
        initial_selected,
        device,
        args.batch_size,
        args.steps_per_interval,
    )
    all_path_norm, _ = integrate_trajectorynet(
        model,
        initial_all,
        device,
        args.batch_size,
        args.steps_per_interval,
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
    all_endpoint_norm = all_path_norm[-1]
    all_atac_raw = (
        film(torch.as_tensor(all_endpoint_norm), 3.0).detach().cpu().numpy() * atac_scale
    ).astype(np.float32)
    uniform_log_weights = np.full(
        len(initial_all), -math.log(len(initial_all)), dtype=np.float32
    )
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
        "components": ["continuous_normalizing_flow"],
        "n_all_initial_cells": int(len(initial_all)),
        "n_visualized_trajectories": int(len(selected)),
        "time_scale": float(model_args.time_scale),
        "rollout": "piecewise reverse integration of the upstream TrajectoryNet clock",
    }
    history_path = args.result_dir / "loss_terms_train.csv"
    if history_path.is_file():
        history = pd.read_csv(history_path)
        method_metrics["final_training_loss"] = float(history.iloc[-1]["total_loss"])

    np.savez_compressed(
        args.output_dir / "trajectorynet_trajectories.npz",
        physical_times=physical_times,
        selected_initial_indices_time1=selected,
        selected_initial_population=first_population[selected],
        trajectory_rna10_normalized=selected_path_norm,
        trajectory_rna10_raw=selected_path_raw,
        trajectory_rna10_umap=selected_umap,
        trajectory_atac8_normalized=selected_atac_norm,
        trajectory_atac8_raw=selected_atac_raw,
        all_initial_endpoint_rna10_raw=all_endpoint_raw,
        all_initial_endpoint_atac8_raw=all_atac_raw,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"trajectorynet": method_metrics}, indent=2) + "\n"
    )
    plot_standalone(
        observed_umap,
        atac_raw,
        population,
        selected_umap,
        selected_atac_raw,
        method_metrics,
        args.output_dir / "trajectorynet_shared_rna10_umap_and_atac.png",
    )

    cb_umap = np.load(
        RNA10_UMAP_DIR / "cytobridge_rna10_umap_trajectories.npz", allow_pickle=True
    )
    cb_metrics = json.loads((CB_COMPARISON / "metrics.json").read_text())
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
            "rna_umap": selected_umap,
            "atac_raw": selected_atac_raw,
            "metrics": method_metrics,
        },
    ]
    plot_three_model_comparison(
        observed_umap,
        atac_raw,
        population,
        methods,
        args.output_dir / "cytobridge_trajectorynet_three_model_comparison.png",
    )
    manifest = {
        "dataset": str(args.dataset.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "iterations": 3000,
        "num_samples": 128,
        "seed": 0,
        "space": "RNA10 PCA divided by primal W2 scale",
        "whiten": False,
        "training_noise": 0.1,
        "divergence_fn": "approximate",
        "solver": "rk4",
        "training_step_size": 0.1,
        "rollout_steps_per_interval": args.steps_per_interval,
        "rollout_internal_step_size": 0.5 / args.steps_per_interval,
        "umap_model": str(args.umap_model.resolve()),
        "film_checkpoint": str(args.film_checkpoint.resolve()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"trajectorynet": method_metrics}, indent=2))


if __name__ == "__main__":
    main()
