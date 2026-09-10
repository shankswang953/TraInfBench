#!/usr/bin/env python
"""Plot the revised six-metric pancreas E15.5 strict-LOO benchmark."""

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

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from trainfbench_plot_style import apply_nature_rc, method_style


ROOT = Path(__file__).resolve().parents[2]
MOSCOT_ROOT = Path("external/COATI/moscot")
DEFAULT_INPUT = (
    ROOT
    / "results/pancreas_strict_loo_time1_six_metrics_20k_with_trajectorynet_mioflow"
)
DEFAULT_OUTPUT = ROOT / "results/pancreas_strict_loo_time1_six_metrics_revised"

METHODS = (
    ("COATI balanced 20k", "COATI bal.", "COATI balanced"),
    ("COATI unbalanced 20k C_y=0.5", "COATI unbal.", "COATI unbalanced"),
    ("CytoBridge balanced 20k", "CytoBridge bal.", "CytoBridge balanced"),
    (
        "CytoBridge unbalanced 20k",
        "CytoBridge unbal.",
        "CytoBridge unbalanced",
    ),
    ("TrajectoryNet 20k", "TrajectoryNet", "TrajectoryNet"),
    ("MIOFlow GAGA10 20k", "MIOFlow", "MIOFlow"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--distance-batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalized_weights(log_mass: np.ndarray) -> tuple[np.ndarray, str]:
    values = np.asarray(log_mass, dtype=np.float64).reshape(-1)
    if not np.any(np.isfinite(values)):
        return np.full(len(values), 1.0 / len(values)), "uniform"
    if not np.all(np.isfinite(values)):
        raise ValueError("Log mass contains a mixture of finite and non-finite values")
    relative = np.exp(values - values.max())
    return relative / relative.sum(), "native_mass"


def target_radii(points: np.ndarray, k: int) -> np.ndarray:
    if k < 1:
        raise ValueError("k must be positive")
    distances = (
        NearestNeighbors(n_neighbors=min(k + 1, len(points)), n_jobs=-1)
        .fit(points)
        .kneighbors(points, return_distance=True)[0]
    )
    return np.asarray(distances[:, -1], dtype=np.float32)


def local_predicted_mass(
    reference: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
    radii: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    prediction_norm = np.sum(np.square(prediction), axis=1)
    result = np.empty(len(reference), dtype=np.float64)
    for start in range(0, len(reference), batch_size):
        stop = min(start + batch_size, len(reference))
        batch = reference[start:stop]
        squared_distances = (
            np.sum(np.square(batch), axis=1)[:, None]
            + prediction_norm[None, :]
            - 2.0 * (batch @ prediction.T)
        )
        membership = squared_distances <= (
            np.square(radii[start:stop, None]) + 1e-7
        )
        result[start:stop] = membership @ weights
    return result


def expected_mass_aware_recall(local_mass: np.ndarray, effective_n: float) -> float:
    clipped = np.minimum(np.maximum(local_mass, 0.0), 1.0 - 1e-15)
    hit_probability = -np.expm1(effective_n * np.log1p(-clipped))
    return float(np.mean(hit_probability))


def load_references() -> dict[str, np.ndarray]:
    rna_scale = float(
        torch.load(
            MOSCOT_ROOT / "data/primal_norm_params.pt",
            map_location="cpu",
            weights_only=False,
        )["scale"]
    )
    atac_scale = float(
        torch.load(
            MOSCOT_ROOT / "data/secondary_norm_params_poissonvi.pt",
            map_location="cpu",
            weights_only=False,
        )["scale"]
    )
    rna = np.load(MOSCOT_ROOT / "data/rna_time_data.npz", allow_pickle=True)
    atac = np.load(
        MOSCOT_ROOT / "data/atac_poissonvi_time_data.npz", allow_pickle=True
    )
    return {
        "RNA": np.asarray(rna["time_1"], dtype=np.float32) / rna_scale,
        "ATAC": np.asarray(atac["time_1"], dtype=np.float32) / atac_scale,
    }


def build_revised_metrics(
    input_dir: Path,
    knn_k: int,
    distance_batch_size: int,
) -> pd.DataFrame:
    original = pd.read_csv(input_dir / "pancreas_strict_loo_time1_six_metric_scores.csv")
    references = load_references()
    radii = {
        modality: target_radii(points, knn_k)
        for modality, points in references.items()
    }
    rows: list[dict[str, object]] = []
    with np.load(input_dir / "predictions.npz", allow_pickle=True) as cache:
        cached = {
            str(cache[f"method{index}_name"]): index
            for index in range(len(METHODS))
        }
        for method, display, style_name in METHODS:
            if method not in cached:
                raise KeyError(f"Missing {method!r} from predictions.npz")
            index = cached[method]
            weights, weighting = normalized_weights(cache[f"method{index}_log_mass"])
            effective_n = float(1.0 / np.sum(np.square(weights)))
            original_row = original.loc[original["method"].eq(method)].iloc[0]
            row: dict[str, object] = {
                "method": method,
                "display": display,
                "style": style_name,
                "weighting": weighting,
                "effective_particle_count": effective_n,
                "rna_w2": float(original_row["rna_w2"]),
                "atac_w2": float(original_row["atac_w2"]),
                "rna_composition_similarity": float(
                    1.0 - np.sqrt(original_row["rna_celltype_composition_jsd"])
                ),
                "atac_composition_similarity": float(
                    1.0 - np.sqrt(original_row["atac_celltype_composition_jsd"])
                ),
                "rna_original_jsd": float(
                    original_row["rna_celltype_composition_jsd"]
                ),
                "atac_original_jsd": float(
                    original_row["atac_celltype_composition_jsd"]
                ),
            }
            for modality, cache_key in (("RNA", "rna"), ("ATAC", "atac")):
                prediction = np.asarray(
                    cache[f"method{index}_{cache_key}"], dtype=np.float32
                )
                local_mass = local_predicted_mass(
                    references[modality],
                    prediction,
                    weights,
                    radii[modality],
                    distance_batch_size,
                )
                row[f"{cache_key}_target_support_recall"] = (
                    expected_mass_aware_recall(local_mass, effective_n)
                )
                row[f"{cache_key}_hard_geometric_recall"] = float(
                    np.mean(local_mass > 0)
                )
            rows.append(row)
    return pd.DataFrame(rows)


def plot_metrics(
    metrics: pd.DataFrame,
    floor: pd.DataFrame,
    output_dir: Path,
) -> None:
    panels = (
        ("rna_target_support_recall", "RNA support ↑"),
        ("atac_target_support_recall", "ATAC support ↑"),
        ("rna_w2", r"RNA $W_2$ ↓"),
        ("atac_w2", r"ATAC $W_2$ ↓"),
        (
            "rna_composition_similarity",
            "RNA composition\nsimilarity ↑",
        ),
        (
            "atac_composition_similarity",
            "ATAC composition\nsimilarity ↑",
        ),
    )
    apply_nature_rc(font_size=10.0)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "font.weight": "normal",
            "axes.titlesize": 10.0,
            "axes.titleweight": "normal",
            "axes.labelsize": 10.0,
            "axes.labelweight": "normal",
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(4.2, 5.05), facecolor="white")
    positions = np.arange(len(metrics))
    for panel_index, (ax, (column, title)) in enumerate(
        zip(axes.ravel(), panels)
    ):
        values = metrics[column].to_numpy(dtype=np.float64)
        for position, (_, row), value in zip(positions, metrics.iterrows(), values):
            style = method_style(str(row["style"]))
            ax.barh(
                position,
                value,
                height=0.52,
                color=style.color,
                edgecolor=style.color,
                linewidth=0.7,
                zorder=2,
            )
            label = f"{value:.3f}"
            ax.text(
                0.985,
                position,
                label,
                ha="right",
                va="center",
                fontsize=10.0,
                fontweight="normal",
                color="#222222",
                clip_on=False,
                zorder=3,
                transform=ax.get_yaxis_transform(),
            )

        is_w2 = column.endswith("_w2")
        if is_w2:
            modality = "RNA" if column.startswith("rna") else "ATAC"
            floor_row = floor.loc[floor["modality"].eq(modality)].iloc[0]
            lower = float(floor_row["q025_w2"])
            upper = float(floor_row["q975_w2"])
            mean = float(floor_row["mean_w2"])
            ax.axvspan(lower, upper, color="#B3B3B3", alpha=0.34, lw=0, zorder=0)
            ax.axvline(mean, color="#666666", lw=0.8, ls="--", zorder=1)
            max_value = max(float(values.max()), upper)
            ax.set_xlim(0.0, max_value * 1.50)
        else:
            ax.set_xlim(0.0, 1.35)

        ax.plot(
            [0.75, 0.75],
            [-0.02, 1.02],
            transform=ax.transAxes,
            color="#C8C8C8",
            lw=0.6,
            ls=":",
            clip_on=False,
            zorder=0,
        )

        ax.set_ylim(len(metrics) - 0.38, -0.72)
        ax.set_xticks([])
        ax.set_yticks(
            positions,
            metrics["display"].astype(str).tolist() if panel_index % 2 == 0 else [],
        )
        ax.tick_params(length=0, pad=2)
        ax.set_title(title, fontsize=10.0, pad=2.0, fontweight="normal")
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "Held-out E15.5",
        fontsize=10.0,
        fontweight="normal",
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.39,
        right=0.985,
        top=0.870,
        bottom=0.035,
        hspace=0.43,
        wspace=0.15,
    )
    stem = output_dir / "pancreas_strict_loo_e155_six_metrics_revised_20k"
    options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.02}
    fig.savefig(stem.with_suffix(".png"), dpi=600, **options)
    fig.savefig(stem.with_suffix(".pdf"), **options)
    fig.savefig(stem.with_suffix(".svg"), **options)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} is not empty; pass --overwrite to replace its files"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = build_revised_metrics(
        args.input_dir,
        args.knn_k,
        args.distance_batch_size,
    )
    floor = pd.read_csv(args.input_dir / "sampling_floor_summary.csv")
    metrics.to_csv(args.output_dir / "revised_six_metric_scores.csv", index=False)
    plot_metrics(metrics, floor, args.output_dir)
    manifest = {
        "task": "Pancreas strict-LOO held-out E15.5 revised six-metric figure",
        "input_dir": str(args.input_dir),
        "knn_k": args.knn_k,
        "support_recall": (
            "For each real target cell, sum normalized predicted mass inside its "
            "real-data kNN radius; convert that mass to an expected hit probability "
            "using the method's weight ESS, then average over target cells."
        ),
        "composition_similarity": "1 - sqrt(base-2 composition JSD)",
        "w2": "Original W2 with the original random-half sampling-floor interval",
        "figure_size_inches": [4.2, 5.05],
        "font": "Arial 10 pt base",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
