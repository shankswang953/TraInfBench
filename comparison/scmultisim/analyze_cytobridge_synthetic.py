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
import importlib.util
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/trainfbench-cytobridge-synthetic-cache/numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/trainfbench-cytobridge-synthetic-cache/matplotlib")
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from torchdiffeq import odeint


TIME_LABELS = ("time1", "time2", "time3", "time4")
POPULATION_COLORS = {
    "4_1": "#4477AA",
    "4_5": "#EE7733",
    "5_2": "#228833",
    "5_3": "#CC3311",
}
EXPECTED_ENDPOINT = {"4_1": {"4_1"}, "4_5": {"5_2", "5_3"}}
DEFAULT_SYNTHETIC_DIR = Path(
    "external/COATI/Synthetic/5scRNA"
)
DEFAULT_SWEEP_DIR = Path(
    "external/COATI/Synthetic/BalancedSyncSweep"
)
DEFAULT_TRAIN_T_DIR = Path(
    "external/COATI/Synthetic/TrainT"
)


def load_scale(path: Path) -> float:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return float(checkpoint["scale"])


def load_film_model(model_file: Path, checkpoint_path: Path) -> torch.nn.Module:
    spec = importlib.util.spec_from_file_location("synthetic_film_model", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load FiLM model definition from {model_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = module.FiLMMLP(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def load_cytobridge(path: Path, device: torch.device):
    from CytoBridge.utils import load_model_from_adata

    adata = ad.read_h5ad(path)
    model = load_model_from_adata(adata).to(device).eval()
    return adata, model


def select_initial_indices(population: np.ndarray, num_samples: int, seed: int) -> np.ndarray:
    groups = ("4_1", "4_5")
    if num_samples % len(groups):
        raise ValueError("--num-samples must be even for balanced initial-population sampling")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for group in groups:
        candidates = np.flatnonzero(population == group)
        count = num_samples // len(groups)
        if len(candidates) < count:
            raise ValueError(f"Not enough {group} cells for {count} trajectories")
        selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
    return np.asarray(selected, dtype=np.int64)


def dynamics(model, t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
    x, log_weight = state
    t_column = t.reshape(1, 1).expand(x.shape[0], 1).to(dtype=x.dtype)
    network_input = torch.cat([x, t_column], dim=1)
    velocity = model.velocity_net(network_input)
    if "growth" in model.components:
        growth = model.growth_net(network_input)
    else:
        growth = torch.zeros_like(log_weight)
    return velocity, growth


@torch.no_grad()
def integrate(
    model,
    initial: np.ndarray,
    times: torch.Tensor,
    device: torch.device,
    batch_size: int,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    path_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    for start in range(0, len(initial), batch_size):
        block = torch.as_tensor(
            initial[start : start + batch_size], dtype=torch.float32, device=device
        )
        log_weight = torch.full(
            (len(block), 1), -math.log(len(initial)), dtype=torch.float32, device=device
        )
        path, log_weights = odeint(
            lambda t, state: dynamics(model, t, state),
            (block, log_weight),
            times,
            method="euler",
            options={"step_size": step_size},
        )
        path_chunks.append(path.detach().cpu().numpy())
        weight_chunks.append(log_weights.detach().cpu().numpy())
    return (
        np.concatenate(path_chunks, axis=1).astype(np.float32),
        np.concatenate(weight_chunks, axis=1).astype(np.float32)[..., 0],
    )


def stable_mass_fraction(log_weights: np.ndarray) -> np.ndarray:
    centered = np.asarray(log_weights, dtype=np.float64) - float(np.max(log_weights))
    weights = np.exp(centered)
    return weights / weights.sum()


def counts(values: np.ndarray) -> dict[str, int]:
    names, values_count = np.unique(values.astype(str), return_counts=True)
    return {name: int(count) for name, count in zip(names, values_count)}


def endpoint_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    target_population: np.ndarray,
    source_population: np.ndarray,
    log_weights: np.ndarray,
) -> dict:
    classifier = KNeighborsClassifier(n_neighbors=15, weights="distance")
    classifier.fit(target, target_population)
    predicted_population = classifier.predict(predicted).astype(str)
    route_consistent = np.asarray(
        [
            endpoint in EXPECTED_ENDPOINT.get(source, set())
            for source, endpoint in zip(source_population, predicted_population)
        ]
    )
    mass_fraction = stable_mass_fraction(log_weights)
    target_self = NearestNeighbors(n_neighbors=2).fit(target).kneighbors(target)[0][:, 1]
    predicted_distance = (
        NearestNeighbors(n_neighbors=1).fit(target).kneighbors(predicted)[0][:, 0]
    )
    mass_by_population = {
        name: float(mass_fraction[predicted_population == name].sum())
        for name in sorted(np.unique(predicted_population))
    }
    by_source = {}
    for source in sorted(np.unique(source_population)):
        mask = source_population == source
        source_mass = float(mass_fraction[mask].sum())
        by_source[source] = {
            "n_cells": int(mask.sum()),
            "route_consistent_fraction": float(route_consistent[mask].mean()),
            "mass_weighted_route_consistent_fraction": float(
                mass_fraction[mask & route_consistent].sum() / (source_mass + 1e-12)
            ),
            "predicted_population_counts": counts(predicted_population[mask]),
            "final_mass_fraction": source_mass,
        }
    return {
        "route_consistent_fraction": float(route_consistent.mean()),
        "mass_weighted_route_consistent_fraction": float(
            mass_fraction[route_consistent].sum()
        ),
        "endpoint_population_counts": counts(predicted_population),
        "endpoint_population_mass_fraction": mass_by_population,
        "by_source_population": by_source,
        "endpoint_to_target_nn_median_ratio": float(
            np.median(predicted_distance) / (np.median(target_self) + 1e-12)
        ),
    }


def transform_shared_umap(
    python: Path, helper: Path, model: Path, points: np.ndarray
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="cytobridge-synthetic-umap-") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "input.npy"
        output_path = temp_path / "output.npy"
        np.save(input_path, np.asarray(points, dtype=np.float32))
        transform_environment = os.environ.copy()
        # Serialized UMAP reducers contain Numba dispatchers. They cannot be
        # rebuilt when the parent CytoBridge launcher exports NUMBA_DISABLE_JIT=1.
        transform_environment.pop("NUMBA_DISABLE_JIT", None)
        subprocess.run(
            [
                str(python),
                str(helper),
                "--model",
                str(model),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ],
            check=True,
            env=transform_environment,
        )
        return np.asarray(np.load(output_path), dtype=np.float32)


@torch.no_grad()
def map_film_trajectory(model, trajectory: np.ndarray, physical_times: np.ndarray) -> np.ndarray:
    mapped = []
    for index, time_value in enumerate(physical_times):
        x = torch.as_tensor(trajectory[index], dtype=torch.float32)
        mapped.append(model(x, float(time_value)).cpu().numpy())
    return np.stack(mapped).astype(np.float32)


def observed_arrays(
    rna_path: Path, atac_path: Path, labels_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rna_archive = np.load(rna_path)
    atac_archive = np.load(atac_path)
    label_archive = np.load(labels_path, allow_pickle=True)
    rna = np.concatenate(
        [np.asarray(rna_archive[label], dtype=np.float32) for label in TIME_LABELS]
    )
    atac = np.concatenate(
        [np.asarray(atac_archive[label], dtype=np.float32) for label in TIME_LABELS]
    )
    population = np.concatenate([label_archive[label].astype(str) for label in TIME_LABELS])
    time = np.concatenate(
        [
            np.full(len(rna_archive[label]), index, dtype=np.float32)
            for index, label in enumerate(TIME_LABELS)
        ]
    )
    return rna, atac, population, time


def add_observed_background(
    axis: plt.Axes, coordinates: np.ndarray, population: np.ndarray, show_legend: bool
) -> None:
    for name in sorted(np.unique(population)):
        mask = population == name
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=6,
            alpha=0.22,
            color=POPULATION_COLORS.get(name, "#999999"),
            linewidths=0,
            rasterized=True,
            label=name,
        )
    if show_legend:
        axis.legend(frameon=False, markerscale=2.4, loc="best")


def add_trajectories(axis: plt.Axes, trajectory: np.ndarray, color: str) -> None:
    for cell_index in range(trajectory.shape[1]):
        axis.plot(
            trajectory[:, cell_index, 0],
            trajectory[:, cell_index, 1],
            color=color,
            linewidth=0.75,
            alpha=0.34,
        )
    axis.scatter(
        trajectory[0, :, 0], trajectory[0, :, 1], s=8, color="#111111", alpha=0.6
    )
    axis.scatter(
        trajectory[-1, :, 0], trajectory[-1, :, 1], s=11, color=color, alpha=0.72
    )


def plot_comparison(
    observed_umap: np.ndarray,
    observed_atac: np.ndarray,
    population: np.ndarray,
    results: list[dict],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14.8, 13.0), dpi=220)
    for column, result in enumerate(results):
        rna_axis = axes[0, column]
        add_observed_background(rna_axis, observed_umap, population, show_legend=column == 0)
        add_trajectories(rna_axis, result["trajectory_umap"], result["color"])
        rna_metrics = result["metrics"]["rna"]
        rna_axis.set_title(
            f"{result['title']} — shared RNA5 UMAP\n"
            f"route={rna_metrics['route_consistent_fraction']:.1%}, "
            f"mass-weighted={rna_metrics['mass_weighted_route_consistent_fraction']:.1%}"
        )
        rna_axis.set_xlabel("UMAP1")
        rna_axis.set_ylabel("UMAP2")

        atac_axis = axes[1, column]
        add_observed_background(atac_axis, observed_atac[:, :2], population, show_legend=False)
        add_trajectories(atac_axis, result["trajectory_atac_raw"][..., :2], result["color"])
        atac_metrics = result["metrics"]["atac"]
        atac_axis.set_title(
            f"{result['title']} — FiLM-mapped ATAC8\n"
            f"route={atac_metrics['route_consistent_fraction']:.1%}, "
            f"mass-weighted={atac_metrics['mass_weighted_route_consistent_fraction']:.1%}"
        )
        atac_axis.set_xlabel("Selected ATAC PC1")
        atac_axis.set_ylabel("Selected ATAC PC3")
    for axis in axes.flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle(
        "CytoBridge balanced vs unbalanced — 3000 train epochs, 128 sampled cells",
        fontsize=17,
        y=1.005,
    )
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project synthetic CytoBridge trajectories onto shared RNA UMAP and ATAC space."
    )
    parser.add_argument(
        "--balanced-adata",
        type=Path,
        default=Path(
            "results/cytobridge_synthetic_rna10_balanced_n128_i3000/adata.h5ad"
        ),
    )
    parser.add_argument(
        "--unbalanced-adata",
        type=Path,
        default=Path(
            "results/cytobridge_synthetic_rna10_unbalanced_n128_i3000/adata.h5ad"
        ),
    )
    parser.add_argument(
        "--rna-data",
        type=Path,
        default=DEFAULT_SYNTHETIC_DIR / "all_time_scRNA_pca10.npz",
    )
    parser.add_argument(
        "--atac-data",
        type=Path,
        default=DEFAULT_SYNTHETIC_DIR / "all_time_scATAC_pca8_no_pc2.npz",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_SYNTHETIC_DIR / "all_time_scRNA_label.npz",
    )
    parser.add_argument(
        "--rna-norm",
        type=Path,
        default=DEFAULT_SYNTHETIC_DIR / "primal_norm_params_rna10_w2.pt",
    )
    parser.add_argument(
        "--atac-norm",
        type=Path,
        default=DEFAULT_SYNTHETIC_DIR / "secondary_norm_params_atac8_no_pc2_w2.pt",
    )
    parser.add_argument(
        "--film-model-file", type=Path, default=DEFAULT_TRAIN_T_DIR / "film_model.py"
    )
    parser.add_argument(
        "--film-checkpoint",
        type=Path,
        default=DEFAULT_TRAIN_T_DIR / "outputs/T_FiLM_rna10_atac8_no_pc2.pt",
    )
    parser.add_argument(
        "--umap-model",
        type=Path,
        default=DEFAULT_SWEEP_DIR / "outputs/assets/rna5_umap_model.joblib",
    )
    parser.add_argument(
        "--observed-umap",
        type=Path,
        default=DEFAULT_SWEEP_DIR / "outputs/assets/observed_rna5_umap.npz",
    )
    parser.add_argument(
        "--umap-python",
        type=Path,
        default=Path(_sys.executable),
        help="Python runtime that created the shared serialized UMAP model.",
    )
    parser.add_argument(
        "--umap-transform-script",
        type=Path,
        default=Path(__file__).resolve().with_name("transform_joblib_umap.py"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cytobridge_synthetic_rna10_n128_i3000_comparison"),
    )
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=61)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    required = (
        args.balanced_adata,
        args.unbalanced_adata,
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
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    device = torch.device(args.device)
    rna_scale = load_scale(args.rna_norm)
    atac_scale = load_scale(args.atac_norm)
    rna_raw, atac_raw, population, observed_time = observed_arrays(
        args.rna_data, args.atac_data, args.labels
    )
    observed_archive = np.load(args.observed_umap, allow_pickle=True)
    observed_umap = np.asarray(observed_archive["observed_umap"], dtype=np.float32)
    if len(observed_umap) != len(rna_raw):
        raise ValueError("Observed UMAP rows do not match the supplied RNA archive")
    if not np.array_equal(observed_archive["observed_population"].astype(str), population):
        raise ValueError("Observed UMAP population ordering does not match labels archive")

    first_mask = observed_time == 0
    last_mask = observed_time == 3
    first_population = population[first_mask]
    selected_local = select_initial_indices(first_population, args.num_samples, args.seed)
    initial_normalized = (rna_raw[first_mask][selected_local] / rna_scale).astype(np.float32)
    all_initial_normalized = (rna_raw[first_mask] / rna_scale).astype(np.float32)
    selected_population = first_population[selected_local]
    times_np = np.linspace(0.0, 3.0, args.n_steps, dtype=np.float32)
    times = torch.as_tensor(times_np, device=device)
    integration_step_size = float(times_np[1] - times_np[0])
    film_model = load_film_model(args.film_model_file, args.film_checkpoint)

    method_specs = (
        ("balanced", "CytoBridge balanced", args.balanced_adata, "#0072B2"),
        ("unbalanced", "CytoBridge unbalanced", args.unbalanced_adata, "#D55E00"),
    )
    results: list[dict] = []
    output_arrays: dict[str, np.ndarray] = {
        "physical_times": times_np,
        "selected_initial_indices_time1": selected_local,
        "selected_initial_population": selected_population,
    }
    metrics: dict[str, dict] = {}
    for key, title, adata_path, color in method_specs:
        trained_adata, model = load_cytobridge(adata_path, device)
        trained_times = np.sort(trained_adata.obs["time_point_processed"].unique())
        if not np.allclose(trained_times, [0.0, 1.0, 2.0, 3.0]):
            raise ValueError(f"Unexpected time grid in {adata_path}: {trained_times}")
        trajectory_norm, trajectory_log_weights = integrate(
            model,
            initial_normalized,
            times,
            device,
            args.batch_size,
            integration_step_size,
        )
        all_path_norm, all_log_weights = integrate(
            model,
            all_initial_normalized,
            torch.as_tensor([0.0, 3.0], dtype=torch.float32, device=device),
            device,
            args.batch_size,
            integration_step_size,
        )
        trajectory_raw = trajectory_norm * rna_scale
        trajectory_umap = transform_shared_umap(
            args.umap_python,
            args.umap_transform_script,
            args.umap_model,
            trajectory_raw[..., :5].reshape(-1, 5),
        ).reshape(args.n_steps, args.num_samples, 2)
        trajectory_atac_norm = map_film_trajectory(film_model, trajectory_norm, times_np)
        trajectory_atac_raw = trajectory_atac_norm * atac_scale
        all_endpoint_norm = all_path_norm[-1]
        all_endpoint_raw = all_endpoint_norm * rna_scale
        all_atac_raw = (
            film_model(torch.as_tensor(all_endpoint_norm), 3.0).detach().cpu().numpy()
            * atac_scale
        ).astype(np.float32)
        final_log_weights = all_log_weights[-1]
        log_total_mass = float(
            np.logaddexp.reduce(final_log_weights.astype(np.float64))
        )
        method_metrics = {
            "rna": endpoint_metrics(
                all_endpoint_raw,
                rna_raw[last_mask],
                population[last_mask],
                first_population,
                final_log_weights,
            ),
            "atac": endpoint_metrics(
                all_atac_raw,
                atac_raw[last_mask],
                population[last_mask],
                first_population,
                final_log_weights,
            ),
            "final_log_total_mass": log_total_mass,
            "components": list(model.components),
            "n_all_initial_cells": int(len(all_initial_normalized)),
            "n_visualized_trajectories": int(args.num_samples),
        }
        metrics[key] = method_metrics
        results.append(
            {
                "key": key,
                "title": title,
                "color": color,
                "trajectory_umap": trajectory_umap,
                "trajectory_atac_raw": trajectory_atac_raw,
                "metrics": method_metrics,
            }
        )
        output_arrays[f"{key}_trajectory_rna10_normalized"] = trajectory_norm
        output_arrays[f"{key}_trajectory_rna10_raw"] = trajectory_raw
        output_arrays[f"{key}_trajectory_rna5_umap"] = trajectory_umap
        output_arrays[f"{key}_trajectory_atac8_normalized"] = trajectory_atac_norm
        output_arrays[f"{key}_trajectory_atac8_raw"] = trajectory_atac_raw
        output_arrays[f"{key}_trajectory_log_weights"] = trajectory_log_weights
        output_arrays[f"{key}_all_initial_endpoint_rna10_raw"] = all_endpoint_raw
        output_arrays[f"{key}_all_initial_endpoint_atac8_raw"] = all_atac_raw
        output_arrays[f"{key}_all_initial_endpoint_log_weights"] = final_log_weights

    plot_path = args.output_dir / "cytobridge_balanced_unbalanced_shared_projection.png"
    plot_comparison(observed_umap, atac_raw, population, results, plot_path)
    np.savez_compressed(args.output_dir / "cytobridge_trajectories.npz", **output_arrays)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    summary_rows = []
    for key, method_metrics in metrics.items():
        summary_rows.append(
            {
                "method": key,
                "rna_route_fraction": method_metrics["rna"]["route_consistent_fraction"],
                "rna_mass_weighted_route_fraction": method_metrics["rna"][
                    "mass_weighted_route_consistent_fraction"
                ],
                "rna_endpoint_nn_ratio": method_metrics["rna"][
                    "endpoint_to_target_nn_median_ratio"
                ],
                "atac_route_fraction": method_metrics["atac"]["route_consistent_fraction"],
                "atac_mass_weighted_route_fraction": method_metrics["atac"][
                    "mass_weighted_route_consistent_fraction"
                ],
                "atac_endpoint_nn_ratio": method_metrics["atac"][
                    "endpoint_to_target_nn_median_ratio"
                ],
                "final_log_total_mass": method_metrics["final_log_total_mass"],
            }
        )
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "summary.csv", index=False)
    manifest = {
        "balanced_adata": str(args.balanced_adata.resolve()),
        "unbalanced_adata": str(args.unbalanced_adata.resolve()),
        "rna_data": str(args.rna_data.resolve()),
        "atac_data": str(args.atac_data.resolve()),
        "labels": str(args.labels.resolve()),
        "rna_norm": str(args.rna_norm.resolve()),
        "atac_norm": str(args.atac_norm.resolve()),
        "rna_scale": rna_scale,
        "atac_scale": atac_scale,
        "film_checkpoint": str(args.film_checkpoint.resolve()),
        "umap_model": str(args.umap_model.resolve()),
        "umap_python": str(args.umap_python.resolve()),
        "observed_umap": str(args.observed_umap.resolve()),
        "num_samples": args.num_samples,
        "n_steps": args.n_steps,
        "seed": args.seed,
        "selected_initial_sampling": "equal 4_1/4_5 without replacement",
        "endpoint_metrics_use": "all 2006 time1 cells",
        "rna_projection": "normalized RNA10 multiplied by RNA scale, first five PCs transformed by shared UMAP",
        "atac_projection": "normalized RNA10 mapped by time-conditioned FiLM, then multiplied by ATAC scale",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Projection figure: {plot_path}")


if __name__ == "__main__":
    main()
