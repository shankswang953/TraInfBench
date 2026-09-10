#!/usr/bin/env python
"""Evaluate reversed-label TrajectoryNet in biological time1 -> time4 order."""

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
from geomloss import SamplesLoss
from sklearn.neighbors import KNeighborsClassifier
from torchdiffeq import odeint


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

from analyze_cytobridge_synthetic import transform_shared_umap  # noqa: E402
from evaluate_terminal_push import (  # noqa: E402
    _load_trajectorynet_model,
    _trajectorynet_diffeq,
)


SYNTHETIC_DIR = Path("external/COATI/Synthetic/5scRNA")
UMAP_DIR = (
    ROOT
    / "results/cytobridge_synthetic_rna10_n128_i3000_comparison/rna10_umap"
)
FIXED40 = Path(
    "external/COATI/Synthetic/"
    "BalancedSyncSweep/outputs/assets/fixed_initial_indices_40.npy"
)
SOURCE_COLORS = {"4_1": "#4477AA", "4_5": "#EE7733"}
TIME_NAMES = ("time1", "time2", "time3", "time4")


@torch.no_grad()
def integrate_native_backward(
    model,
    initial: np.ndarray,
    device: torch.device,
    batch_size: int,
    steps_per_interval: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Traverse reversed labels 3 -> 2 -> 1 -> 0 using density direction.

    At the raw ODE level, TrajectoryNet's ``reverse=False`` density transform
    integrates each model interval from its lower to upper internal time.
    Because the data labels were reversed, that direction is biological time1
    to time4.
    """

    diffeq = _trajectorynet_diffeq(model)
    time_scale = 0.5
    internal_step = time_scale / steps_per_interval
    chunks: list[np.ndarray] = []
    audit: dict | None = None

    for start in range(0, len(initial), batch_size):
        x = torch.as_tensor(
            initial[start : start + batch_size], dtype=torch.float32, device=device
        )
        states = [x.detach().cpu().numpy()]
        for biological_interval in range(3):
            source_reversed_rank = 3 - biological_interval
            lower = source_reversed_rank * time_scale
            upper = (source_reversed_rank + 1) * time_scale
            integration_times = torch.linspace(
                lower,
                upper,
                steps_per_interval + 1,
                dtype=torch.float32,
                device=device,
            )

            if audit is None:
                audit_n = min(32, len(x))
                short_times = torch.tensor(
                    [lower, upper], dtype=torch.float32, device=device
                )
                raw_endpoint = odeint(
                    diffeq,
                    x[:audit_n],
                    short_times,
                    method="rk4",
                    options={"step_size": 0.1},
                )[-1]
                zeros = torch.zeros(audit_n, 1, dtype=x.dtype, device=device)
                official_endpoint, _ = model(
                    x[:audit_n],
                    zeros,
                    integration_times=short_times,
                    reverse=False,
                )
                max_error = float(torch.max(torch.abs(raw_endpoint - official_endpoint)))
                audit = {
                    "direction": "TrajectoryNet density/backward transform (reverse=False)",
                    "first_model_interval": [lower, upper],
                    "raw_ode_vs_official_max_abs_error": max_error,
                    "audit_cells": audit_n,
                }

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

    physical_times = np.linspace(
        0.0, 3.0, 3 * steps_per_interval + 1, dtype=np.float32
    )
    if audit is None:
        raise ValueError("No initial cells were provided")
    return np.concatenate(chunks, axis=1), physical_times, audit


def sinkhorn_divergence(
    predicted: np.ndarray,
    observed: np.ndarray,
    dimensions: int,
    loss_fn: SamplesLoss,
    device: torch.device,
) -> float:
    x = torch.as_tensor(predicted[:, :dimensions], dtype=torch.float32, device=device)
    y = torch.as_tensor(observed[:, :dimensions], dtype=torch.float32, device=device)
    with torch.inference_mode():
        value = loss_fn(x, y)
    return max(float(value.detach().cpu()), 0.0)


def correct_rate(
    predicted_raw: np.ndarray,
    observed_raw: np.ndarray,
    observed_population: np.ndarray,
    source_population: np.ndarray,
) -> tuple[float, np.ndarray]:
    classifier = KNeighborsClassifier(n_neighbors=15, weights="distance")
    classifier.fit(observed_raw, observed_population)
    assigned = classifier.predict(predicted_raw).astype(str)
    is_correct = np.asarray(
        [
            label == "4_1" if source == "4_1" else label in {"4_5", "5_2", "5_3"}
            for source, label in zip(source_population, assigned)
        ],
        dtype=bool,
    )
    return float(is_correct.mean()), assigned


def make_umap_plot(
    observed_umap: np.ndarray,
    labels: np.ndarray,
    original_rank: np.ndarray,
    predicted_umap: np.ndarray,
    source_population: np.ndarray,
    metrics: dict,
    output_png: Path,
    output_pdf: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.65), sharex=True, sharey=True)
    for column, biological_rank in enumerate((1, 2, 3)):
        axis = axes[column]
        target = original_rank == biological_rank
        axis.scatter(
            observed_umap[target, 0],
            observed_umap[target, 1],
            s=8,
            color="#B8B8B8",
            alpha=0.32,
            linewidths=0,
            rasterized=True,
        )
        for population_name in ("4_1", "4_5"):
            source = source_population == population_name
            axis.scatter(
                predicted_umap[column, source, 0],
                predicted_umap[column, source, 1],
                s=8,
                color=SOURCE_COLORS[population_name],
                alpha=0.64,
                linewidths=0,
                rasterized=True,
                label=f"{population_name} source" if column == 0 else None,
            )
        item = metrics[TIME_NAMES[biological_rank]]
        axis.set_title(
            f"{TIME_NAMES[biological_rank]}\n"
            f"RNA $W_2$ = {math.sqrt(2.0 * item['sinkhorn_10d']):.3f}"
        )
        axis.set_xlabel("RNA10 UMAP1")
        if column == 0:
            axis.set_ylabel("RNA10 UMAP2")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.suptitle(
        "TrajectoryNet — reversed labels, native backward rollout from real time1",
        fontsize=14,
        y=0.995,
    )
    figure.subplots_adjust(left=0.07, right=0.995, top=0.80, bottom=0.22, wspace=0.08)
    figure.savefig(output_png, dpi=320, facecolor="white", bbox_inches="tight")
    figure.savefig(output_pdf, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/synthetic_rna10_trajectorynet_reversed.npz"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "results/trajectorynet_synthetic_rna10_reversed_n128_i3000/checkpt-3000.pth"
        ),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/trajectorynet_synthetic_rna10_reversed_n128_i3000"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/trajectorynet_synthetic_rna10_reversed_n128_i3000_analysis"
        ),
    )
    parser.add_argument(
        "--umap-model", type=Path, default=UMAP_DIR / "rna10_umap_model.joblib"
    )
    parser.add_argument(
        "--observed-umap", type=Path, default=UMAP_DIR / "observed_rna10_umap.npz"
    )
    parser.add_argument(
        "--umap-python", type=Path, default=Path(_sys.executable)
    )
    parser.add_argument(
        "--umap-transform-script",
        type=Path,
        default=ROOT / "common/transform_joblib_umap.py",
    )
    parser.add_argument("--fixed-indices", type=Path, default=FIXED40)
    parser.add_argument("--steps-per-interval", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sinkhorn-blur", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.dataset,
        args.checkpoint,
        args.umap_model,
        args.observed_umap,
        args.umap_python,
        args.umap_transform_script,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    archive = np.load(args.dataset, allow_pickle=True)
    normalized = np.asarray(archive["pca"], dtype=np.float32)
    raw = np.asarray(archive["pca_raw"], dtype=np.float32)
    reversed_rank = np.asarray(archive["sample_labels"], dtype=np.int64)
    original_rank = np.asarray(archive["original_sample_labels"], dtype=np.int64)
    population = np.asarray(archive["population"]).astype(str)
    rna_scale = float(np.asarray(archive["rna_scale"]))
    if not np.array_equal(reversed_rank, 3 - original_rank):
        raise ValueError("Dataset labels are not the expected exact reversal")
    if not np.allclose(normalized * rna_scale, raw, rtol=1e-6, atol=1e-6):
        raise ValueError("Normalized and raw RNA10 arrays do not match")

    device = torch.device(args.device)
    model, model_args = _load_trajectorynet_model(
        args.checkpoint, normalized.shape[1], device, "rk4", 0.1
    )
    initial_mask = original_rank == 0
    initial = normalized[initial_mask]
    source_population = population[initial_mask]
    trajectory, physical_times, direction_audit = integrate_native_backward(
        model,
        initial,
        device,
        args.batch_size,
        args.steps_per_interval,
    )
    observed_indices = np.asarray(
        [args.steps_per_interval, 2 * args.steps_per_interval, 3 * args.steps_per_interval]
    )
    predicted = trajectory[observed_indices]
    predicted_raw = predicted * rna_scale

    loss_fn = SamplesLoss(
        loss="sinkhorn", p=2, blur=args.sinkhorn_blur, backend="tensorized"
    ).to(device)
    metrics: dict[str, dict] = {}
    rows = []
    assigned_labels = []
    for output_index, biological_rank in enumerate((1, 2, 3)):
        target_mask = original_rank == biological_rank
        target = normalized[target_mask]
        target_raw = raw[target_mask]
        target_population = population[target_mask]
        rate, assigned = correct_rate(
            predicted_raw[output_index],
            target_raw,
            target_population,
            source_population,
        )
        assigned_labels.append(assigned)
        item = {
            "n_predicted": int(len(initial)),
            "n_observed": int(target_mask.sum()),
            "sinkhorn_2d": sinkhorn_divergence(
                predicted[output_index], target, 2, loss_fn, device
            ),
            "sinkhorn_10d": sinkhorn_divergence(
                predicted[output_index], target, 10, loss_fn, device
            ),
            "correct_rate_15nn": rate,
        }
        item["w2_rna10"] = math.sqrt(2.0 * item["sinkhorn_10d"])
        for dimensions in (2, 10):
            baseline = sinkhorn_divergence(initial, target, dimensions, loss_fn, device)
            item[f"no_movement_sinkhorn_{dimensions}d"] = baseline
            item[f"improvement_vs_no_movement_{dimensions}d"] = baseline - item[
                f"sinkhorn_{dimensions}d"
            ]
        metrics[TIME_NAMES[biological_rank]] = item
        rows.append({"time": TIME_NAMES[biological_rank], **item})
    mean_correct = float(np.mean([item["correct_rate_15nn"] for item in metrics.values()]))

    observed_umap = np.asarray(
        np.load(args.observed_umap, allow_pickle=True)["observed_umap"], dtype=np.float32
    )
    predicted_umap = transform_shared_umap(
        args.umap_python,
        args.umap_transform_script,
        args.umap_model,
        predicted_raw.reshape(-1, 10),
    ).reshape(3, len(initial), 2)

    if args.fixed_indices.is_file():
        fixed_indices = np.asarray(np.load(args.fixed_indices), dtype=np.int64)
    else:
        fixed_indices = np.arange(min(40, len(initial)), dtype=np.int64)
    if fixed_indices.ndim != 1 or np.any(fixed_indices < 0) or np.any(fixed_indices >= len(initial)):
        raise ValueError(f"Invalid fixed initial indices: {args.fixed_indices}")
    fixed_raw = trajectory[:, fixed_indices] * rna_scale
    fixed_umap = transform_shared_umap(
        args.umap_python,
        args.umap_transform_script,
        args.umap_model,
        fixed_raw.reshape(-1, 10),
    ).reshape(len(physical_times), len(fixed_indices), 2)

    np.savez_compressed(
        args.output_dir / "reversed_time_all_initial_trajectories.npz",
        physical_times=physical_times,
        trajectory_rna10_normalized=trajectory,
        rna_scale=np.asarray(rna_scale, dtype=np.float64),
        initial_population=source_population,
        observed_time_indices=observed_indices,
        predicted_observed_times_rna10_normalized=predicted,
        predicted_observed_times_rna10_raw=predicted_raw,
        predicted_observed_times_umap=predicted_umap,
        predicted_observed_times_15nn_labels=np.stack(assigned_labels),
        fixed_initial_indices=fixed_indices,
        fixed_trajectory_rna10_raw=fixed_raw,
        fixed_trajectory_umap=fixed_umap,
    )
    pd.DataFrame(rows).to_csv(args.output_dir / "observed_time_metrics.csv", index=False)
    result = {
        "training": {
            "time_label_mapping": {
                "original_time1": 3,
                "original_time2": 2,
                "original_time3": 1,
                "original_time4": 0,
            },
            "iterations": 3000,
            "batch_size": 128,
            "seed": 0,
            "time_scale": float(model_args.time_scale),
            "checkpoint": str(args.checkpoint.resolve()),
        },
        "rollout": {
            "start": "all 2,006 observed original-time1 cells",
            "direction": "native TrajectoryNet density/backward direction",
            "biological_order": "original time1 -> time2 -> time3 -> time4",
            "trajectory_shape": list(trajectory.shape),
            "direction_audit": direction_audit,
        },
        "metric": {
            "space": "normalized RNA10",
            "sinkhorn": f"GeomLoss debiased p=2, blur={args.sinkhorn_blur:g}",
            "w2_rna10": "sqrt(2 * debiased p=2 Sinkhorn divergence), computed in normalized 10D RNA PCA; UMAP is display only",
            "correctness": "distance-weighted 15-NN in raw RNA10",
        },
        "observed_times": metrics,
        "mean_correct_rate": mean_correct,
    }
    history_path = args.result_dir / "loss_terms_train.csv"
    if history_path.is_file():
        history = pd.read_csv(history_path)
        result["training"]["final_total_loss"] = float(history.iloc[-1]["total_loss"])
        iteration_column = "iteration" if "iteration" in history else "iter"
        if iteration_column in history:
            best_row = history.loc[history["total_loss"].idxmin()]
            result["training"]["best_total_loss"] = float(best_row["total_loss"])
            result["training"]["best_loss_iteration"] = int(
                best_row[iteration_column]
            )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    make_umap_plot(
        observed_umap,
        population,
        original_rank,
        predicted_umap,
        source_population,
        metrics,
        args.output_dir / "reversed_time_predicted_vs_observed_umap.png",
        args.output_dir / "reversed_time_predicted_vs_observed_umap.pdf",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
