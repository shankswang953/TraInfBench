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
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = Path("external/COATI/Synthetic")
DEFAULT_OUTPUT = (
    ROOT
    / "results/synthetic_rna10_coati_cytobridge_trajectorynet_reversed_method_columns"
)
METHODS = (
    "TrajectoryNet reversed",
    "COATI balanced",
    "COATI unbalanced",
    "CytoBridge balanced",
    "CytoBridge unbalanced",
)
TIME_KEYS = ("time2", "time3", "time4")
COLORS = {"4_1": "#4477AA", "4_5": "#EE7733"}


def normalized_positive_weights(values: np.ndarray) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float64).reshape(-1)
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("Particle weights must be finite and non-negative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Particle weights have zero total mass")
    return weights / total


def normalized_log_weights(values: np.ndarray) -> np.ndarray:
    log_weights = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.exp(log_weights - float(log_weights.max()))
    return weights / weights.sum()


def branch_correct_rate(
    predicted_raw: np.ndarray,
    target_raw: np.ndarray,
    target_population: np.ndarray,
    source_population: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[float, np.ndarray]:
    classifier = KNeighborsClassifier(n_neighbors=15, weights="distance")
    classifier.fit(target_raw, target_population)
    assigned = classifier.predict(predicted_raw).astype(str)
    correct = np.asarray(
        [
            label == "4_1" if source == "4_1" else label in {"4_5", "5_2", "5_3"}
            for source, label in zip(source_population, assigned)
        ],
        dtype=bool,
    )
    if weights is None:
        return float(correct.mean()), assigned
    return float(np.sum(normalized_positive_weights(weights) * correct)), assigned


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare COATI, CytoBridge, and reversed-time TrajectoryNet by observed time."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/synthetic_rna10_trajectorynet_reversed.npz"),
    )
    parser.add_argument(
        "--coati-cache",
        type=Path,
        default=(
            SYNTHETIC
            / "UnbalancedSyncSweep/outputs/bio_all1_alpha100_d10_e1/"
            "observed_time_distribution/all_initial_observed_time_predictions.npz"
        ),
    )
    parser.add_argument(
        "--cytobridge-cache",
        type=Path,
        default=(
            ROOT
            / "results/cytobridge_synthetic_rna10_officialcfg_sum_n128_i3000_observed_times/"
            "cytobridge_observed_time_cache.npz"
        ),
    )
    parser.add_argument(
        "--base-coordinates",
        type=Path,
        default=(
            ROOT
            / "results/coati_cytobridge_observed_time_umap_comparison_officialcfg_sum_i3000/"
            "comparison_coordinates.npz"
        ),
    )
    parser.add_argument(
        "--trajectorynet-cache",
        type=Path,
        default=(
            ROOT
            / "results/trajectorynet_synthetic_rna10_reversed_n128_i3000_analysis/"
            "reversed_time_all_initial_trajectories.npz"
        ),
    )
    parser.add_argument(
        "--trajectorynet-metrics",
        type=Path,
        default=(
            ROOT
            / "results/trajectorynet_synthetic_rna10_reversed_n128_i3000_analysis/metrics.json"
        ),
    )
    parser.add_argument(
        "--sinkhorn-csv",
        type=Path,
        default=(
            ROOT
            / "results/synthetic_rna10_same_space_predicted_time_sinkhorn_cytobridge_officialcfg_sum_i3000/"
            "predicted_time_sinkhorn.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    required = (
        args.dataset,
        args.coati_cache,
        args.cytobridge_cache,
        args.base_coordinates,
        args.trajectorynet_cache,
        args.trajectorynet_metrics,
        args.sinkhorn_csv,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    data = np.load(args.dataset, allow_pickle=True)
    coati = np.load(args.coati_cache, allow_pickle=False)
    cytobridge = np.load(args.cytobridge_cache, allow_pickle=False)
    coordinates = np.load(args.base_coordinates, allow_pickle=False)
    trajectorynet = np.load(args.trajectorynet_cache, allow_pickle=False)
    trajectorynet_metrics = json.loads(args.trajectorynet_metrics.read_text())
    sinkhorn = pd.read_csv(args.sinkhorn_csv)

    original_rank = np.asarray(data["original_sample_labels"], dtype=np.int64)
    raw = np.asarray(data["pca_raw"], dtype=np.float32)
    population = np.asarray(data["population"]).astype(str)
    source_population = population[original_rank == 0]
    if not np.array_equal(source_population, coordinates["initial_source"].astype(str)):
        raise ValueError("Initial source ordering differs between caches")
    if not np.array_equal(
        source_population, trajectorynet["initial_population"].astype(str)
    ):
        raise ValueError("TrajectoryNet source ordering differs from the shared time1 cells")

    predictions_raw = {
        "TrajectoryNet reversed": np.asarray(
            trajectorynet["predicted_observed_times_rna10_raw"], dtype=np.float32
        ),
        "COATI balanced": np.stack(
            [np.asarray(coati[f"coati_bal_{time}_rna"], dtype=np.float32) for time in TIME_KEYS]
        ),
        "COATI unbalanced": np.stack(
            [np.asarray(coati[f"coati_unbal_{time}_rna"], dtype=np.float32) for time in TIME_KEYS]
        ),
        "CytoBridge balanced": np.asarray(
            cytobridge["cytobridge_balanced_all_initial_selected_rna10_raw"],
            dtype=np.float32,
        ),
        "CytoBridge unbalanced": np.asarray(
            cytobridge["cytobridge_unbalanced_all_initial_selected_rna10_raw"],
            dtype=np.float32,
        ),
    }
    predicted_umap = {
        "TrajectoryNet reversed": np.asarray(
            trajectorynet["predicted_observed_times_umap"], dtype=np.float32
        ),
        "COATI balanced": np.stack(
            [np.asarray(coordinates[f"coati_bal_{time}"], dtype=np.float32) for time in TIME_KEYS]
        ),
        "COATI unbalanced": np.stack(
            [np.asarray(coordinates[f"coati_unbal_{time}"], dtype=np.float32) for time in TIME_KEYS]
        ),
        "CytoBridge balanced": np.stack(
            [np.asarray(coordinates[f"cytobridge_balanced_{time}"], dtype=np.float32) for time in TIME_KEYS]
        ),
        "CytoBridge unbalanced": np.stack(
            [np.asarray(coordinates[f"cytobridge_unbalanced_{time}"], dtype=np.float32) for time in TIME_KEYS]
        ),
    }
    observed_umap = [
        np.asarray(coordinates[f"observed_{time}"], dtype=np.float32)
        for time in TIME_KEYS
    ]

    native_weights: dict[str, list[np.ndarray | None]] = {
        "TrajectoryNet reversed": [None, None, None],
        "COATI balanced": [None, None, None],
        "COATI unbalanced": [
            normalized_positive_weights(coati[f"coati_unbal_{time}_weights"])
            for time in TIME_KEYS
        ],
        "CytoBridge balanced": [None, None, None],
        "CytoBridge unbalanced": [
            normalized_log_weights(
                cytobridge["cytobridge_unbalanced_all_initial_selected_log_weights"][index]
            )
            for index in range(3)
        ],
    }

    w2_lookup: dict[tuple[str, str], float] = {}
    native_rows = sinkhorn[sinkhorn["weighting"] == "native_mass"]
    for row in native_rows.itertuples(index=False):
        w2_lookup[(str(row.method), str(row.time_key))] = float(row.w2_like_sqrt_2s)
    for time in TIME_KEYS:
        item = trajectorynet_metrics["observed_times"][time]
        w2_lookup[("TrajectoryNet reversed", time)] = float(
            item.get("w2_rna10", np.sqrt(2.0 * item["sinkhorn_10d"]))
        )

    rows: list[dict] = []
    assigned_labels: dict[str, np.ndarray] = {}
    for method in METHODS:
        method_assigned = []
        for time_index, time_key in enumerate(TIME_KEYS, start=1):
            target_mask = original_rank == time_index
            rate, assigned = branch_correct_rate(
                predictions_raw[method][time_index - 1],
                raw[target_mask],
                population[target_mask],
                source_population,
                native_weights[method][time_index - 1],
            )
            method_assigned.append(assigned)
            rows.append(
                {
                    "method": method,
                    "time": time_key,
                    "rna_w2": w2_lookup[(method, time_key)],
                    "branch_correct_rate": rate,
                    "metric_weighting": (
                        "native particle mass"
                        if native_weights[method][time_index - 1] is not None
                        else "equal particles"
                    ),
                }
            )
        assigned_labels[method] = np.stack(method_assigned)
    metric_frame = pd.DataFrame(rows)
    metric_frame.to_csv(args.output_dir / "metrics.csv", index=False)

    permutation = np.random.default_rng(0).permutation(len(source_population))
    point_colors = np.asarray([COLORS[name] for name in source_population])
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.titlesize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(
        len(TIME_KEYS),
        len(METHODS),
        figsize=(17.0, 8.6),
        dpi=230,
        sharex=True,
        sharey=True,
    )
    for row_index, time_key in enumerate(TIME_KEYS):
        for column_index, method in enumerate(METHODS):
            axis = axes[row_index, column_index]
            axis.scatter(
                observed_umap[row_index][:, 0],
                observed_umap[row_index][:, 1],
                s=2.7,
                color="#A8A8A8",
                alpha=0.13,
                linewidths=0,
                rasterized=True,
            )
            axis.scatter(
                predicted_umap[method][row_index, permutation, 0],
                predicted_umap[method][row_index, permutation, 1],
                s=4.0,
                c=point_colors[permutation],
                alpha=0.40,
                linewidths=0,
                rasterized=True,
            )
            metric = metric_frame[
                (metric_frame["method"] == method) & (metric_frame["time"] == time_key)
            ].iloc[0]
            method_heading = f"{method}\n" if row_index == 0 else ""
            axis.set_title(
                method_heading
                + rf"RNA $W_2$ = {metric['rna_w2']:.3f}" + "\n"
                + f"Branch correct = {metric['branch_correct_rate']:.1%}",
                pad=3,
            )
            if column_index == 0:
                axis.set_ylabel(f"{time_key}\nRNA10 UMAP2", fontsize=10)
            if row_index == len(TIME_KEYS) - 1:
                axis.set_xlabel("RNA10 UMAP1")
            axis.tick_params(length=2.5, width=0.7)

    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5,
            markerfacecolor=COLORS["4_1"], markeredgewidth=0,
            label="4_1 source",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5,
            markerfacecolor=COLORS["4_5"], markeredgewidth=0,
            label="4_5 source",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5,
            markerfacecolor="#A8A8A8", markeredgewidth=0, alpha=0.5,
            label="observed target snapshot",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.006),
    )
    figure.suptitle(
        "Synthetic RNA10 snapshot fit and branch preservation",
        fontsize=15,
        y=0.998,
    )
    figure.text(
        0.5,
        0.038,
        "RNA W2 is computed in normalized 10D RNA PCA (not UMAP); unbalanced metrics use native particle mass.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    figure.subplots_adjust(
        left=0.075, right=0.995, top=0.91, bottom=0.125, wspace=0.08, hspace=0.28
    )
    png = args.output_dir / "method_columns_time2_time4_rna_w2_correct_umap.png"
    pdf = png.with_suffix(".pdf")
    figure.savefig(png, dpi=320, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    np.savez_compressed(
        args.output_dir / "comparison_coordinates.npz",
        initial_source=source_population,
        **{
            f"{method.lower().replace(' ', '_')}_predicted_umap": predicted_umap[method]
            for method in METHODS
        },
        **{f"observed_{time}": observed_umap[index] for index, time in enumerate(TIME_KEYS)},
    )
    manifest = {
        "methods_horizontal_order": list(METHODS),
        "times_vertical_order": list(TIME_KEYS),
        "predicted_cells": int(len(source_population)),
        "w2": "sqrt(2*S_epsilon) in normalized RNA10 PCA, blur=0.05; UMAP display only",
        "correctness": {
            "classifier": "distance-weighted 15-NN in raw RNA10 against each matching observed snapshot",
            "4_1_source": ["4_1"],
            "4_5_source": ["4_5", "5_2", "5_3"],
            "unbalanced_weighting": "native particle mass",
        },
        "plot": {
            "observed": "small gray points, alpha=0.13",
            "predicted": "single fixed-random-order colored scatter, size=4.0, alpha=0.40",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(metric_frame.to_string(index=False))
    print(png)


if __name__ == "__main__":
    main()
