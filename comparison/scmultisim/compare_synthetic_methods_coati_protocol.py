#!/usr/bin/env python
"""Compare external trajectory methods with the established COATI protocol."""

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
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/trainfbench-method-comparison-mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/trainfbench-method-comparison-numba")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/trainfbench-method-comparison-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

from analyze_cytobridge_synthetic import (  # noqa: E402
    integrate as integrate_cytobridge,
    load_cytobridge,
    load_scale,
    observed_arrays,
    transform_shared_umap,
)
from analyze_mioflow_synthetic import rollout_mioflow_gaga  # noqa: E402
from analyze_tigon_synthetic import rollout_tigon  # noqa: E402
from analyze_trajectorynet_synthetic import integrate_trajectorynet  # noqa: E402
from evaluate_terminal_push import _load_trajectorynet_model  # noqa: E402
from evaluate_tigon_gastrulation_normalized import (  # noqa: E402
    _load_model as load_tigon_model,
    _load_tigon_module,
)


SYNTHETIC = Path("external/COATI/Synthetic")
DATA_DIR = SYNTHETIC / "5scRNA"
COATI_METRIC_ROOT = (
    SYNTHETIC
    / "UnbalancedSyncSweep/outputs/bio_all1_alpha100_d10_e1"
)
FIXED40 = (
    SYNTHETIC
    / "BalancedSyncSweep/outputs/assets/fixed_initial_indices_40.npy"
)
ALLOWED = {"4_1": {"4_1"}, "4_5": {"4_5", "5_2", "5_3"}}
SOURCE_COLORS = {"4_1": "#4477AA", "4_5": "#EE7733"}
METHOD_SPECS = (
    ("cytobridge_balanced", "CytoBridge bal", False),
    ("cytobridge_unbalanced", "CytoBridge unbal", True),
    ("trajectorynet", "TrajectoryNet", False),
    ("mioflow", "MIOFlow", False),
    ("tigon", "TIGON", True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Roll out all 2,006 source cells and the shared fixed40 set, then "
            "compare methods using COATI's RNA10 15-NN protocol."
        )
    )
    parser.add_argument(
        "--cytobridge-balanced",
        type=Path,
        default=ROOT
        / "results/cytobridge_synthetic_rna10_balanced_d10_e1_n128_i3000/adata.h5ad",
    )
    parser.add_argument(
        "--cytobridge-unbalanced",
        type=Path,
        default=ROOT
        / "results/cytobridge_synthetic_rna10_unbalanced_d10_e1_n128_i3000/adata.h5ad",
    )
    parser.add_argument(
        "--trajectorynet-checkpoint",
        type=Path,
        default=ROOT
        / "results/trajectorynet_synthetic_rna10_n128_i3000/checkpt-3000.pth",
    )
    parser.add_argument(
        "--mioflow-model",
        type=Path,
        default=ROOT / "results/mioflow_synthetic_rna10_gaga10_n128_i3000/model.pt",
    )
    parser.add_argument(
        "--tigon-result-dir",
        type=Path,
        default=ROOT
        / "results/tigon_synthetic_rna10_pca_minus2_2_official_n128_i3000",
    )
    parser.add_argument(
        "--input-h5ad",
        type=Path,
        default=ROOT / "data/synthetic_rna10_cytobridge.h5ad",
    )
    parser.add_argument(
        "--rna-data",
        type=Path,
        default=DATA_DIR / "all_time_scRNA_pca10.npz",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DATA_DIR / "all_time_scRNA_label.npz",
    )
    parser.add_argument(
        "--rna-norm",
        type=Path,
        default=DATA_DIR / "primal_norm_params_rna10_w2.pt",
    )
    parser.add_argument("--fixed40", type=Path, default=FIXED40)
    parser.add_argument(
        "--umap-model",
        type=Path,
        default=ROOT
        / "results/cytobridge_synthetic_rna10_n128_i3000_comparison/rna10_umap/rna10_umap_model.joblib",
    )
    parser.add_argument(
        "--observed-umap",
        type=Path,
        default=ROOT
        / "results/cytobridge_synthetic_rna10_n128_i3000_comparison/rna10_umap/observed_rna10_umap.npz",
    )
    parser.add_argument(
        "--umap-python", type=Path, default=Path(_sys.executable)
    )
    parser.add_argument(
        "--umap-transform-script",
        type=Path,
        default=ROOT / "common/transform_joblib_umap.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/synthetic_rna10_all_method_coati_protocol_comparison",
    )
    parser.add_argument("--steps-per-interval", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=np.float64)
    values = np.exp(values - float(np.max(values)))
    return values / (values.sum() + 1e-300)


def correct_flags(
    predicted: np.ndarray,
    source: np.ndarray,
    classifier: KNeighborsClassifier,
) -> tuple[np.ndarray, np.ndarray]:
    predicted_population = classifier.predict(predicted).astype(str)
    correct = np.asarray(
        [target in ALLOWED[origin] for origin, target in zip(source, predicted_population)],
        dtype=bool,
    )
    return correct, predicted_population


def correct_rate(correct: np.ndarray, log_weights: np.ndarray, weighted: bool) -> float:
    if not weighted:
        return float(np.mean(correct))
    return float(np.sum(normalize_weights(log_weights)[correct]))


def rollout_cytobridge(
    adata_path: Path,
    initial_normalized: np.ndarray,
    device: torch.device,
    batch_size: int,
    steps_per_interval: int,
) -> tuple[np.ndarray, np.ndarray]:
    trained_adata, model = load_cytobridge(adata_path, device)
    trained_times = np.sort(
        trained_adata.obs["time_point_processed"].astype(float).unique()
    )
    if not np.allclose(trained_times, [0.0, 1.0, 2.0, 3.0]):
        raise ValueError(f"Unexpected CytoBridge time grid: {trained_times}")
    physical_times = np.linspace(
        0.0, 3.0, 3 * steps_per_interval + 1, dtype=np.float32
    )
    path, log_weights = integrate_cytobridge(
        model,
        initial_normalized,
        torch.as_tensor(physical_times, device=device),
        device,
        batch_size,
        1.0 / steps_per_interval,
    )
    return path, log_weights


def rollout_trajectorynet(
    checkpoint: Path,
    initial_normalized: np.ndarray,
    device: torch.device,
    batch_size: int,
    steps_per_interval: int,
) -> tuple[np.ndarray, np.ndarray]:
    model, _ = _load_trajectorynet_model(checkpoint, 10, device, "rk4", 0.1)
    path, _ = integrate_trajectorynet(
        model, initial_normalized, device, batch_size, steps_per_interval
    )
    return path, np.zeros(path.shape[:2], dtype=np.float32)


def rollout_mioflow(
    model_path: Path,
    initial_normalized: np.ndarray,
    device: torch.device,
    batch_size: int,
    steps_per_interval: int,
) -> tuple[np.ndarray, np.ndarray]:
    path, _, _, _, _ = rollout_mioflow_gaga(
        initial_normalized, model_path, device, batch_size, steps_per_interval
    )
    return path, np.zeros(path.shape[:2], dtype=np.float32)


def rollout_official_tigon(
    result_dir: Path,
    raw_all: np.ndarray,
    first_mask: np.ndarray,
    device: torch.device,
    batch_size: int,
    steps_per_interval: int,
) -> tuple[np.ndarray, np.ndarray]:
    model_path = result_dir / "tigon.pt"
    config_path = result_dir / "config.json"
    module = _load_tigon_module()
    model, config, state = load_tigon_model(module, model_path, config_path, device)
    if bool(config.get("use_ae", True)):
        raise ValueError("The comparison expects official direct PCA10 TIGON")
    if config.get("embedding_normalization") != "per-axis-minmax-minus2-2":
        raise ValueError("The comparison expects per-axis TIGON scaling to [-2,2]")
    if config.get("embedding_key") != "X_pca_raw":
        raise ValueError("Official TIGON must use X_pca_raw")
    if int(state.get("model_dim", 0)) != 10:
        raise ValueError("Expected a 10D TIGON model")
    axis_min = np.asarray(
        config["resolved_embedding_normalization_axis_min"], dtype=np.float32
    )
    axis_max = np.asarray(
        config["resolved_embedding_normalization_axis_max"], dtype=np.float32
    )
    if not np.allclose(axis_min, raw_all.min(axis=0), rtol=0, atol=1e-6):
        raise ValueError("TIGON scaling minima do not match the RNA10 data")
    if not np.allclose(axis_max, raw_all.max(axis=0), rtol=0, atol=1e-6):
        raise ValueError("TIGON scaling maxima do not match the RNA10 data")
    axis_range = axis_max - axis_min
    initial_model = (
        4.0 * (raw_all[first_mask] - axis_min[None, :]) / axis_range[None, :] - 2.0
    ).astype(np.float32)
    path_model, cumulative_growth, _ = rollout_tigon(
        model, initial_model, device, batch_size, steps_per_interval
    )
    path_raw = (
        (path_model + 2.0) * axis_range[None, None, :] / 4.0
        + axis_min[None, None, :]
    ).astype(np.float32)
    return path_raw, np.clip(cumulative_growth, -12.0, 12.0).astype(np.float32)


def coati_reference_metrics() -> list[dict]:
    terminal = json.loads(
        (COATI_METRIC_ROOT / "bio_all1_alpha100_metrics.json").read_text()
    )
    observed = json.loads(
        (
            COATI_METRIC_ROOT
            / "observed_time_correctness/observed_time_correctness.json"
        ).read_text()
    )
    balanced_sync = next(row for row in terminal["balanced_sync"] if row["cy"] == 0.1)
    unbalanced_sync = next(
        row for row in terminal["unbalanced_sync"] if row["cy"] == 0.1
    )
    return [
        {
            "method": "OT",
            "terminal_correct_rate": terminal["balanced_rna_only"]["count_route"],
            "fixed40_mean_correct_rate": observed["baseline"]["balanced"][
                "mean_over_time2_time3_time4"
            ],
            "metric_source": "COATI reference files",
        },
        {
            "method": "COATI bal (Cy=0.1)",
            "terminal_correct_rate": balanced_sync["route_all"],
            "fixed40_mean_correct_rate": observed["balanced"]["0.1"][
                "mean_over_time2_time3_time4"
            ],
            "metric_source": "COATI reference files",
        },
        {
            "method": "UOT",
            "terminal_correct_rate": terminal["unbalanced_rna_only"]["mass_route"],
            "fixed40_mean_correct_rate": observed["baseline"]["unbalanced"][
                "mean_over_time2_time3_time4"
            ],
            "metric_source": "COATI reference files",
        },
        {
            "method": "COATI unbal (Cy=0.1)",
            "terminal_correct_rate": unbalanced_sync["mass_route"],
            "fixed40_mean_correct_rate": observed["unbalanced"]["0.1"][
                "mean_over_time2_time3_time4"
            ],
            "metric_source": "COATI reference files",
        },
    ]


def plot_observed_time_comparison(
    method_results: dict[str, dict],
    observed_umap: np.ndarray,
    observed_time: np.ndarray,
    source: np.ndarray,
    output: Path,
) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 11})
    figure, axes = plt.subplots(
        3,
        len(METHOD_SPECS),
        figsize=(17.2, 10.0),
        sharex=True,
        sharey=True,
        dpi=240,
    )
    rng = np.random.default_rng(0)
    order = rng.permutation(len(source))
    colors = np.asarray([SOURCE_COLORS[value] for value in source])
    for column, (key, title, _) in enumerate(METHOD_SPECS):
        axes[0, column].set_title(title, fontsize=15, pad=8)
        for row, (time_value, time_name) in enumerate(((1, "time2"), (2, "time3"), (3, "time4"))):
            axis = axes[row, column]
            observed_mask = observed_time == time_value
            axis.scatter(
                observed_umap[observed_mask, 0],
                observed_umap[observed_mask, 1],
                s=5,
                color="#A7A7A7",
                alpha=0.18,
                linewidths=0,
                rasterized=True,
            )
            points = method_results[key]["observed_umap"][row]
            axis.scatter(
                points[order, 0],
                points[order, 1],
                s=7,
                c=colors[order],
                alpha=0.48,
                linewidths=0,
                rasterized=True,
            )
            rate = method_results[key]["metrics"]["all2006_per_time"][time_name]
            axis.text(
                0.03,
                0.04,
                f"Correct rate = {rate:.1%}",
                transform=axis.transAxes,
                fontsize=11,
                ha="left",
                va="bottom",
            )
            axis.set_aspect("equal", adjustable="box")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.8)
            if column == 0:
                axis.set_ylabel(time_name, fontsize=13)
    figure.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="", color=SOURCE_COLORS["4_1"], label="4_1 branch"
            ),
            Line2D(
                [0], [0], marker="o", linestyle="", color=SOURCE_COLORS["4_5"], label="4_5 branch"
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=12,
    )
    figure.suptitle(
        "External methods: all 2,006 initial cells in the shared RNA10 UMAP",
        fontsize=17,
        y=0.995,
    )
    figure.tight_layout(rect=[0.01, 0.045, 1.0, 0.965], w_pad=0.7, h_pad=0.8)
    figure.savefig(output, facecolor="white", bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(figure)


def plot_metric_summary(frame: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 10})
    metrics = (
        ("terminal_correct_rate", "Correct terminal rate ↑"),
        (
            "fixed40_mean_correct_rate",
            "Mean correct rate across observed times · fixed40 ↑",
        ),
    )
    colors = [
        "#9E9E9E",
        "#E69F00",
        "#7A7A7A",
        "#D55E00",
        "#56B4E9",
        "#0072B2",
        "#6A3D9A",
        "#009E73",
        "#CC79A7",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(15.8, 5.2), dpi=220)
    x = np.arange(len(frame))
    for axis, (column, title) in zip(axes.flat, metrics):
        values = frame[column].to_numpy(float)
        bars = axis.bar(x, values, color=colors, width=0.78)
        axis.set_title(title, fontsize=14)
        axis.set_xticks(x, frame["method"], rotation=32, ha="right")
        axis.set_ylim(0, max(values) * 1.19 + 1e-12)
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values) * 0.025,
                f"{value:.1%}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    figure.suptitle(
        "Synthetic RNA10 comparison under the COATI evaluation protocol",
        fontsize=17,
        y=0.995,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.965], h_pad=2.0, w_pad=1.0)
    figure.savefig(output, facecolor="white", bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    required = (
        args.cytobridge_balanced,
        args.cytobridge_unbalanced,
        args.trajectorynet_checkpoint,
        args.mioflow_model,
        args.tigon_result_dir / "tigon.pt",
        args.tigon_result_dir / "config.json",
        args.input_h5ad,
        args.rna_data,
        args.labels,
        args.rna_norm,
        args.fixed40,
        args.umap_model,
        args.observed_umap,
        args.umap_python,
        args.umap_transform_script,
        COATI_METRIC_ROOT / "bio_all1_alpha100_metrics.json",
        COATI_METRIC_ROOT
        / "observed_time_correctness/observed_time_correctness.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    if args.steps_per_interval != 20:
        raise ValueError("The formal comparison requires 20 steps per interval (61 total)")
    device = torch.device(args.device)
    rna_raw, _, population, observed_time = observed_arrays(
        args.rna_data,
        DATA_DIR / "all_time_scATAC_pca8_no_pc2.npz",
        args.labels,
    )
    rna_scale = load_scale(args.rna_norm)
    first_mask = observed_time == 0
    source = population[first_mask]
    initial_normalized = (rna_raw[first_mask] / rna_scale).astype(np.float32)
    fixed40 = np.asarray(np.load(args.fixed40), dtype=np.int64)
    if len(fixed40) != 40 or len(np.unique(fixed40)) != 40:
        raise ValueError("Expected 40 unique fixed source indices")
    fixed_source = source[fixed40]
    if {name: int(np.sum(fixed_source == name)) for name in np.unique(fixed_source)} != {
        "4_1": 20,
        "4_5": 20,
    }:
        raise ValueError("The fixed40 set is not 20 cells per source population")

    observed_umap = np.asarray(
        np.load(args.observed_umap, allow_pickle=True)["observed_umap"],
        dtype=np.float32,
    )
    if observed_umap.shape != (len(rna_raw), 2):
        raise ValueError("Observed RNA10 UMAP does not match the RNA data")
    classifiers = {
        step: KNeighborsClassifier(n_neighbors=15, weights="distance").fit(
            rna_raw[observed_time == time_value],
            population[observed_time == time_value],
        )
        for step, time_value in ((20, 1), (40, 2), (60, 3))
    }

    rollout_functions = {
        "cytobridge_balanced": lambda: rollout_cytobridge(
            args.cytobridge_balanced,
            initial_normalized,
            device,
            args.batch_size,
            args.steps_per_interval,
        ),
        "cytobridge_unbalanced": lambda: rollout_cytobridge(
            args.cytobridge_unbalanced,
            initial_normalized,
            device,
            args.batch_size,
            args.steps_per_interval,
        ),
        "trajectorynet": lambda: rollout_trajectorynet(
            args.trajectorynet_checkpoint,
            initial_normalized,
            device,
            args.batch_size,
            args.steps_per_interval,
        ),
        "mioflow": lambda: rollout_mioflow(
            args.mioflow_model,
            initial_normalized,
            device,
            args.batch_size,
            args.steps_per_interval,
        ),
        "tigon": lambda: rollout_official_tigon(
            args.tigon_result_dir,
            rna_raw,
            first_mask,
            device,
            args.batch_size,
            args.steps_per_interval,
        ),
    }

    method_results: dict[str, dict] = {}
    cache_arrays: dict[str, np.ndarray] = {
        "physical_times": np.linspace(0, 3, 61, dtype=np.float32),
        "fixed_initial_indices_40": fixed40,
        "fixed_initial_population": fixed_source,
        "all_initial_population": source,
    }
    metric_rows = []
    for key, title, weighted in METHOD_SPECS:
        print(f"Rolling out {title} ...", flush=True)
        path, log_weights = rollout_functions[key]()
        if key == "tigon":
            path_raw = path
        else:
            path_raw = (path * rna_scale).astype(np.float32)
        if path_raw.shape != (61, len(source), 10):
            raise ValueError(f"Unexpected {title} path shape: {path_raw.shape}")
        if log_weights.shape != (61, len(source)):
            raise ValueError(f"Unexpected {title} weight shape: {log_weights.shape}")

        all2006_per_time = {}
        fixed40_per_time = {}
        for step, time_name in ((20, "time2"), (40, "time3"), (60, "time4")):
            correct_all, _ = correct_flags(path_raw[step], source, classifiers[step])
            all2006_per_time[time_name] = correct_rate(
                correct_all, log_weights[step], weighted
            )
            correct_fixed, _ = correct_flags(
                path_raw[step, fixed40], fixed_source, classifiers[step]
            )
            fixed40_per_time[time_name] = correct_rate(
                correct_fixed, log_weights[step, fixed40], weighted
            )

        selected_raw = path_raw[[20, 40, 60]]
        selected_umap = transform_shared_umap(
            args.umap_python,
            args.umap_transform_script,
            args.umap_model,
            selected_raw.reshape(-1, 10),
        ).reshape(3, len(source), 2)
        metrics = {
            "weighted": weighted,
            "all2006_per_time": all2006_per_time,
            "all2006_mean_correct_rate": float(
                np.mean(list(all2006_per_time.values()))
            ),
            "all2006_terminal_correct_rate": all2006_per_time["time4"],
            "fixed40_per_time": fixed40_per_time,
            "fixed40_mean_correct_rate": float(np.mean(list(fixed40_per_time.values()))),
        }
        method_results[key] = {
            "title": title,
            "observed_umap": selected_umap,
            "metrics": metrics,
        }
        metric_rows.append(
            {
                "method": title,
                "terminal_correct_rate": all2006_per_time["time4"],
                "fixed40_mean_correct_rate": metrics["fixed40_mean_correct_rate"],
                "metric_source": "new aligned rollout",
            }
        )
        cache_arrays[f"{key}_fixed40_trajectory_rna10_raw"] = path_raw[:, fixed40]
        cache_arrays[f"{key}_fixed40_log_weights"] = log_weights[:, fixed40]
        cache_arrays[f"{key}_all_initial_selected_rna10_raw"] = path_raw[
            [20, 40, 60]
        ]
        cache_arrays[f"{key}_all_initial_selected_log_weights"] = log_weights[
            [20, 40, 60]
        ]
        cache_arrays[f"{key}_all_initial_observed_umap"] = selected_umap
        del path, path_raw, log_weights

    external_figure = args.output_dir / "external_methods_all2006_observed_times.png"
    plot_observed_time_comparison(
        method_results, observed_umap, observed_time, source, external_figure
    )
    np.savez_compressed(args.output_dir / "aligned_trajectory_cache.npz", **cache_arrays)
    (args.output_dir / "external_method_metrics.json").write_text(
        json.dumps(
            {
                "definition": {
                    "classifier": "distance-weighted 15-NN in raw RNA10",
                    "allowed": {key: sorted(value) for key, value in ALLOWED.items()},
                    "all2006": "rates at time2/time3/time4; native particle weights for unbalanced methods",
                    "fixed40": "same 20+20 source cells; mean over time2/time3/time4",
                },
                "methods": {
                    key: value["metrics"] for key, value in method_results.items()
                },
            },
            indent=2,
        )
        + "\n"
    )

    combined = pd.DataFrame(coati_reference_metrics() + metric_rows)
    combined.to_csv(args.output_dir / "all_method_metric_summary.csv", index=False)
    plot_metric_summary(
        combined, args.output_dir / "all_method_metric_summary.png"
    )
    manifest = {
        "rna_space": "raw PCA10 for kNN; shared full-RNA10 UMAP for display",
        "fixed40": str(args.fixed40.resolve()),
        "cytobridge_balanced": str(args.cytobridge_balanced.resolve()),
        "cytobridge_unbalanced": str(args.cytobridge_unbalanced.resolve()),
        "cytobridge_loss": {
            "lambda_ot": 10,
            "lambda_mass": 10,
            "lambda_density": 10,
            "lambda_energy": 1,
            "density_top_k": 5,
            "density_hinge_value": 0.01,
        },
        "trajectorynet_checkpoint": str(args.trajectorynet_checkpoint.resolve()),
        "mioflow_model": str(args.mioflow_model.resolve()),
        "tigon_result_dir": str(args.tigon_result_dir.resolve()),
        "coati_metric_root": str(COATI_METRIC_ROOT.resolve()),
        "umap_model": str(args.umap_model.resolve()),
        "n_initial": int(len(source)),
        "n_steps_fixed40": 61,
        "saved_all_initial_times": [1.0, 2.0, 3.0],
        "note": "Full 61-step paths are saved only for fixed40; all-2,006 caches contain selected evaluation times.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(combined.to_string(index=False), flush=True)
    print(external_figure, flush=True)


if __name__ == "__main__":
    main()
