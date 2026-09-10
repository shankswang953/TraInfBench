#!/usr/bin/env python
"""Plot pancreas strict-LOO predictions in fixed RNA and mapped-ATAC UMAPs."""

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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import joblib
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from umap import UMAP

from trainfbench_plot_style import apply_nature_rc, method_style


ROOT = Path(__file__).resolve().parents[2]
MOSCOT_DATA = Path("external/COATI/moscot/data")
DEFAULT_INPUT = (
    ROOT
    / "results/pancreas_strict_loo_time1_six_metrics_20k_with_trajectorynet_mioflow"
)
DEFAULT_METRICS = (
    ROOT
    / "results/pancreas_strict_loo_time1_six_metrics_revised/revised_six_metric_scores.csv"
)
DEFAULT_OUTPUT = ROOT / "results/pancreas_strict_loo_e155_dual_umap_5methods"

METHODS = (
    ("COATI balanced 20k", "COATI bal.", "COATI balanced"),
    ("COATI unbalanced 20k C_y=0.5", "COATI unbal.", "COATI unbalanced"),
    ("CytoBridge balanced 20k", "CytoBridge bal.", "CytoBridge balanced"),
    ("CytoBridge unbalanced 20k", "CytoBridge unbal.", "CytoBridge unbalanced"),
    ("TrajectoryNet 20k", "TrajectoryNet", "TrajectoryNet"),
)

PANEL_LAYOUT = (
    ("observed", "TrajectoryNet 20k"),
    ("CytoBridge balanced 20k", "CytoBridge unbalanced 20k"),
    ("COATI balanced 20k", "COATI unbalanced 20k C_y=0.5"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_real_atlases() -> tuple[dict[str, np.ndarray], np.ndarray]:
    rna = np.load(MOSCOT_DATA / "rna_time_data.npz", allow_pickle=True)
    atac = np.load(
        MOSCOT_DATA / "atac_poissonvi_time_data.npz", allow_pickle=True
    )
    keys = ("time_0", "time_1", "time_2")
    rna_scale = float(
        torch.load(
            MOSCOT_DATA / "primal_norm_params.pt",
            map_location="cpu",
            weights_only=False,
        )["scale"]
    )
    atac_scale = float(
        torch.load(
            MOSCOT_DATA / "secondary_norm_params_poissonvi.pt",
            map_location="cpu",
            weights_only=False,
        )["scale"]
    )
    stage_index = np.concatenate(
        [np.full(len(rna[key]), index, dtype=np.int8) for index, key in enumerate(keys)]
    )
    return (
        {
            "RNA": np.concatenate(
                [np.asarray(rna[key], dtype=np.float32) for key in keys], axis=0
            ) / rna_scale,
            "ATAC": np.concatenate(
                [np.asarray(atac[key], dtype=np.float32) for key in keys], axis=0
            ) / atac_scale,
        },
        stage_index,
    )


def load_predictions(input_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    records: dict[str, dict[str, np.ndarray]] = {}
    with np.load(input_dir / "predictions.npz", allow_pickle=True) as cache:
        for index in range(len(METHODS)):
            name = str(cache[f"method{index}_name"])
            records[name] = {
                "RNA": np.asarray(cache[f"method{index}_rna"], dtype=np.float32),
                "ATAC": np.asarray(cache[f"method{index}_atac"], dtype=np.float32),
                "log_mass": np.asarray(
                    cache[f"method{index}_log_mass"], dtype=np.float64
                ).reshape(-1),
            }
    expected = {item[0] for item in METHODS}
    if set(records) != expected:
        raise ValueError(
            f"Prediction methods differ: missing={expected - set(records)}, "
            f"extra={set(records) - expected}"
        )
    return records


def normalized_weights(log_mass: np.ndarray) -> np.ndarray:
    if not np.any(np.isfinite(log_mass)):
        return np.full(len(log_mass), 1.0 / len(log_mass), dtype=np.float64)
    if not np.all(np.isfinite(log_mass)):
        raise ValueError("Log mass contains mixed finite and non-finite values")
    relative = np.exp(log_mass - np.max(log_mass))
    return relative / np.sum(relative)


def point_sizes(log_mass: np.ndarray) -> np.ndarray:
    weights = normalized_weights(log_mass)
    relative = np.sqrt(weights * len(weights))
    return np.clip(0.72 * relative, 0.28, 2.60)


def fit_and_transform(
    real: np.ndarray,
    predictions: dict[str, dict[str, np.ndarray]],
    modality: str,
    *,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> tuple[UMAP, np.ndarray, dict[str, np.ndarray]]:
    model = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
        transform_seed=seed,
        transform_mode="embedding",
        low_memory=True,
    )
    real_umap = np.asarray(model.fit_transform(real), dtype=np.float32)
    ordered_names = [item[0] for item in METHODS]
    lengths = [len(predictions[name][modality]) for name in ordered_names]
    stacked = np.concatenate(
        [predictions[name][modality] for name in ordered_names], axis=0
    )
    transformed = np.asarray(model.transform(stacked), dtype=np.float32)
    endpoints = np.cumsum(lengths)[:-1]
    pieces = np.split(transformed, endpoints)
    return model, real_umap, dict(zip(ordered_names, pieces))


def robust_limits(
    real_umap: np.ndarray,
    predicted_umap: dict[str, np.ndarray],
) -> tuple[tuple[float, float], tuple[float, float]]:
    all_points = np.concatenate([real_umap, *predicted_umap.values()], axis=0)
    lower = np.quantile(all_points, 0.001, axis=0)
    upper = np.quantile(all_points, 0.999, axis=0)
    span = np.maximum(upper - lower, 1e-6)
    return (
        (float(lower[0] - 0.035 * span[0]), float(upper[0] + 0.035 * span[0])),
        (float(lower[1] - 0.035 * span[1]), float(upper[1] + 0.035 * span[1])),
    )


def draw_panel(
    ax: plt.Axes,
    *,
    modality: str,
    method: str,
    real_umap: np.ndarray,
    heldout_mask: np.ndarray,
    predicted_umap: dict[str, np.ndarray],
    predictions: dict[str, dict[str, np.ndarray]],
    metrics: pd.DataFrame,
    limits: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    ax.scatter(
        real_umap[:, 0],
        real_umap[:, 1],
        s=0.32,
        c="#D1D1D1",
        alpha=0.52,
        linewidths=0,
        rasterized=True,
        zorder=0,
    )
    if method == "observed":
        heldout = real_umap[heldout_mask]
        ax.scatter(
            heldout[:, 0],
            heldout[:, 1],
            s=0.52,
            c="#111111",
            alpha=0.82,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        title = f"Observed E15.5\nin global {modality} atlas"
    else:
        _, display, style_name = next(item for item in METHODS if item[0] == method)
        style = method_style(style_name)
        points = predicted_umap[method]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=point_sizes(predictions[method]["log_mass"]),
            c=style.color,
            alpha=0.78,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        metric_column = "rna_w2" if modality == "RNA" else "atac_w2"
        value = float(metrics.loc[metrics["method"].eq(method), metric_column].iloc[0])
        title = f"{display}\n{modality} $W_2$ = {value:.3f}"
    ax.set_title(title, fontsize=8.1, fontweight="normal", pad=1.0)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_figure(
    output_dir: Path,
    stage_index: np.ndarray,
    real_umaps: dict[str, np.ndarray],
    predicted_umaps: dict[str, dict[str, np.ndarray]],
    predictions: dict[str, dict[str, np.ndarray]],
    metrics: pd.DataFrame,
) -> None:
    apply_nature_rc(font_size=10.0)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 4, figsize=(7.15, 5.90), facecolor="white")
    heldout_mask = stage_index == 1
    limits_by_modality = {
        modality: robust_limits(real_umaps[modality], predicted_umaps[modality])
        for modality in ("RNA", "ATAC")
    }
    for group, modality in enumerate(("RNA", "ATAC")):
        column_offset = 2 * group
        for row, methods in enumerate(PANEL_LAYOUT):
            for local_column, method in enumerate(methods):
                ax = axes[row, column_offset + local_column]
                if method is None:
                    ax.axis("off")
                    continue
                draw_panel(
                    ax,
                    modality=modality,
                    method=method,
                    real_umap=real_umaps[modality],
                    heldout_mask=heldout_mask,
                    predicted_umap=predicted_umaps[modality],
                    predictions=predictions,
                    metrics=metrics,
                    limits=limits_by_modality[modality],
                )

    fig.text(
        0.255,
        0.992,
        "Held-out E15.5 RNA distribution",
        ha="center",
        va="top",
        fontsize=10.2,
        fontweight="normal",
    )
    fig.text(
        0.755,
        0.992,
        "Held-out E15.5 RNA-to-ATAC mapping",
        ha="center",
        va="top",
        fontsize=10.2,
        fontweight="normal",
    )
    legend_handles = [
        mpl.lines.Line2D(
            [], [], marker="o", linestyle="None", markersize=3.8,
            markerfacecolor="#D1D1D1", markeredgewidth=0, label="All observed stages"
        ),
        mpl.lines.Line2D(
            [], [], marker="*", linestyle="None", markersize=5.0,
            markerfacecolor="#111111", markeredgecolor="#111111", label="Real E15.5"
        ),
        mpl.lines.Line2D(
            [], [], marker="o", linestyle="None", markersize=3.8,
            markerfacecolor="#777777", markeredgewidth=0, label="Predicted"
        ),
    ]
    for x in (0.255, 0.755):
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(x, 0.955),
            ncol=3,
            frameon=False,
            fontsize=7.0,
            handletextpad=0.35,
            columnspacing=0.9,
        )
    fig.subplots_adjust(
        left=0.025,
        right=0.985,
        top=0.840,
        bottom=0.018,
        hspace=0.32,
        wspace=0.10,
    )
    stem = output_dir / "pancreas_strict_loo_e155_rna_atac_umap_5methods"
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
    real, stage_index = load_real_atlases()
    predictions = load_predictions(args.input_dir)
    metrics = pd.read_csv(args.metrics)

    models: dict[str, UMAP] = {}
    real_umaps: dict[str, np.ndarray] = {}
    predicted_umaps: dict[str, dict[str, np.ndarray]] = {}
    for modality in ("RNA", "ATAC"):
        print(f"[UMAP] fitting {modality} atlas {real[modality].shape}", flush=True)
        model, real_umap, predicted = fit_and_transform(
            real[modality],
            predictions,
            modality,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            seed=args.seed,
        )
        models[modality] = model
        real_umaps[modality] = real_umap
        predicted_umaps[modality] = predicted
        joblib.dump(model, args.output_dir / f"{modality.lower()}_umap_model.joblib")

    archive: dict[str, np.ndarray] = {
        "stage_index": stage_index,
        "real_rna_umap": real_umaps["RNA"],
        "real_atac_umap": real_umaps["ATAC"],
    }
    for index, (method, _, _) in enumerate(METHODS):
        archive[f"method{index}_name"] = np.asarray(method)
        archive[f"method{index}_rna_umap"] = predicted_umaps["RNA"][method]
        archive[f"method{index}_atac_umap"] = predicted_umaps["ATAC"][method]
    np.savez_compressed(args.output_dir / "umap_coordinates.npz", **archive)

    plot_figure(
        args.output_dir,
        stage_index,
        real_umaps,
        predicted_umaps,
        predictions,
        metrics,
    )
    manifest = {
        "task": "Pancreas strict-LOO E15.5 RNA and mapped-ATAC UMAP comparison",
        "heldout_stage": "E15.5",
        "rna_umap_input": "all observed stages in normalized RNA PCA50",
        "atac_umap_input": "all observed stages in normalized ATAC PoissonVI22",
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "random_state": args.seed,
        "prediction_weight_display": (
            "point area scales with sqrt(normalized native mass times particle count); "
            "uniform for methods without native mass"
        ),
        "metrics": str(args.metrics.resolve()),
        "predictions": str((args.input_dir / "predictions.npz").resolve()),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
