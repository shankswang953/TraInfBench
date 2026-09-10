#!/usr/bin/env python
"""Recompute and plot revised Gastrulation strict-LOO six-metric benchmarks."""

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
import re
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
from matplotlib.gridspec import GridSpec
from sklearn.neighbors import NearestNeighbors

from evaluate_gastrulation_full_cmcc import _load_references
from trainfbench_plot_style import apply_nature_rc, method_style


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT / "results/gastrulation_strict_loo_six_metrics_20k_cy0p3"
)
DEFAULT_FLOOR = (
    ROOT
    / "results/gastrulation_loo_shared_t_trajectorynet_terminal_floor"
    / "sampling_floor_summary.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "results/gastrulation_strict_loo_six_metrics_revised_cy0p3"
)

STAGES = (
    ("E8.0", 1, 1),
    ("E8.5", 2, 2),
)

METHODS = (
    ("COATI balanced", "COATI bal.", "COATI balanced", False),
    ("COATI unbalanced", "COATI unbal.", "COATI unbalanced", False),
    (
        "CytoBridge balanced 20k",
        "CytoBridge bal.",
        "CytoBridge balanced",
        False,
    ),
    (
        "CytoBridge unbalanced 20k",
        "CytoBridge unbal.",
        "CytoBridge unbalanced",
        False,
    ),
    ("MIOFlow 20k", "MIOFlow", "MIOFlow", False),
    ("TIGON 20k", "TIGON", "TIGON", False),
    ("TrajectoryNet 20k", "TrajectoryNet", "TrajectoryNet", False),
)

PANELS = (
    (
        ("rna_target_support_recall", "RNA support ↑"),
        ("atac_target_support_recall", "ATAC support ↑"),
    ),
    (
        ("rna_w2", r"RNA $W_2$ ↓"),
        ("atac_w2", r"ATAC $W_2$ ↓"),
    ),
    (
        ("rna_composition_similarity", "RNA composition\nsimilarity ↑"),
        ("atac_composition_similarity", "ATAC composition\nsimilarity ↑"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--floor", type=Path, default=DEFAULT_FLOOR)
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


def cache_method_indices(cache: np.lib.npyio.NpzFile, loo_index: int) -> dict[str, int]:
    pattern = re.compile(rf"loo{loo_index}_method(\d+)_name$")
    result: dict[str, int] = {}
    for key in cache.files:
        match = pattern.fullmatch(key)
        if match is None:
            continue
        index = int(match.group(1))
        result[str(cache[key].item())] = index
    return result


def build_revised_metrics(
    input_dir: Path,
    knn_k: int,
    distance_batch_size: int,
) -> pd.DataFrame:
    original = pd.read_csv(input_dir / "strict_loo_e80_e85_six_metric_scores.csv")
    rna_references, atac_references, _, _, _ = _load_references()
    cache_path = input_dir / "strict_loo_e80_e85_predictions.npz"
    rows: list[dict[str, object]] = []
    with np.load(cache_path, allow_pickle=False) as cache:
        for stage, loo_index, stage_index in STAGES:
            references = {
                "RNA": np.asarray(rna_references[stage_index], dtype=np.float32),
                "ATAC": np.asarray(atac_references[stage_index], dtype=np.float32),
            }
            radii = {
                modality: target_radii(points, knn_k)
                for modality, points in references.items()
            }
            indices = cache_method_indices(cache, loo_index)
            for method, display, style_name, hatched in METHODS:
                if method not in indices:
                    raise KeyError(f"Missing {stage} {method!r} from {cache_path}")
                index = indices[method]
                prefix = f"loo{loo_index}_method{index}"
                weights, weighting = normalized_weights(cache[f"{prefix}_log_mass"])
                effective_n = float(1.0 / np.sum(np.square(weights)))
                matches = original[
                    original["stage"].eq(stage) & original["method"].eq(method)
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected one original score for {stage} {method}; found {len(matches)}"
                    )
                original_row = matches.iloc[0]
                row: dict[str, object] = {
                    "stage": stage,
                    "method": method,
                    "display": display,
                    "style": style_name,
                    "hatched": hatched,
                    "weighting": weighting,
                    "effective_particle_count": effective_n,
                    "rna_w2": float(
                        np.sqrt(2.0 * original_row["rna_sinkhorn_divergence"])
                    ),
                    "atac_w2": float(
                        np.sqrt(2.0 * original_row["atac_sinkhorn_divergence"])
                    ),
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
                        cache[f"{prefix}_{cache_key}"], dtype=np.float32
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


def floor_row(floor: pd.DataFrame, stage: str, modality: str) -> pd.Series:
    matches = floor[floor["stage"].eq(stage) & floor["modality"].eq(modality)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one sampling-floor row for {stage} {modality}; found {len(matches)}"
        )
    return matches.iloc[0]


def plot_metrics(
    metrics: pd.DataFrame,
    floor: pd.DataFrame,
    output_dir: Path,
    *,
    stage_heading: str = "Held-out",
    stem_name: str = "gastrulation_strict_loo_e80_e85_six_metrics_revised_cy0p3",
) -> None:
    apply_nature_rc(font_size=10.0)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "font.weight": "normal",
            "axes.titlesize": 12.0,
            "axes.titleweight": "normal",
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(8.05, 4.75), facecolor="white")
    grid = GridSpec(
        5,
        5,
        figure=fig,
        width_ratios=(1.0, 1.0, 0.14, 1.0, 1.0),
        height_ratios=(1.0, 0.12, 1.0, 0.25, 1.0),
        hspace=0.0,
        wspace=0.14,
    )
    axes: dict[tuple[int, int], plt.Axes] = {}
    stage_columns = ((0, 1), (3, 4))
    panel_rows = (0, 2, 4)
    positions = np.arange(len(METHODS))
    for stage_position, (stage, _, _) in enumerate(STAGES):
        local = metrics[metrics["stage"].eq(stage)].set_index("method").loc[
            [method[0] for method in METHODS]
        ]
        for row_index, metric_pair in enumerate(PANELS):
            for modality_index, (column, title) in enumerate(metric_pair):
                grid_column = stage_columns[stage_position][modality_index]
                ax = fig.add_subplot(grid[panel_rows[row_index], grid_column])
                axes[(stage_position, modality_index)] = ax
                values = local[column].to_numpy(dtype=np.float64)
                for position, (_, record), value in zip(
                    positions, local.iterrows(), values
                ):
                    style = method_style(str(record["style"]))
                    hatched = bool(record["hatched"])
                    ax.barh(
                        position,
                        value,
                        height=0.52,
                        color="white" if hatched else style.color,
                        edgecolor=style.color,
                        hatch="///" if hatched else None,
                        linewidth=0.8 if hatched else 0.7,
                        zorder=2,
                    )
                    ax.text(
                        0.985,
                        position,
                        f"{value:.3f}",
                        ha="right",
                        va="center",
                        fontsize=10.0,
                        color="#222222",
                        clip_on=False,
                        zorder=3,
                        transform=ax.get_yaxis_transform(),
                    )

                is_w2 = column.endswith("_w2")
                if is_w2:
                    modality = "RNA" if column.startswith("rna") else "ATAC"
                    reference_floor = floor_row(floor, stage, modality)
                    lower = float(reference_floor["q025_w2"])
                    upper = float(reference_floor["q975_w2"])
                    mean = float(reference_floor["mean_w2"])
                    ax.axvspan(
                        lower,
                        upper,
                        color="#B3B3B3",
                        alpha=0.34,
                        lw=0,
                        zorder=0,
                    )
                    ax.axvline(mean, color="#666666", lw=0.8, ls="--", zorder=1)
                    ax.set_xlim(0.0, max(float(values.max()), upper) * 1.50)
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
                ax.set_ylim(len(METHODS) - 0.38, -0.72)
                show_labels = stage_position == 0 and modality_index == 0
                ax.set_yticks(
                    positions,
                    local["display"].astype(str).tolist() if show_labels else [],
                )
                ax.set_xticks([])
                ax.tick_params(length=0, pad=2)
                ax.set_title(title, pad=2.0, fontweight="normal")
                for spine in ax.spines.values():
                    spine.set_visible(False)

    fig.subplots_adjust(left=0.205, right=0.992, top=0.895, bottom=0.025)
    fig.canvas.draw()
    for stage_position, (stage, _, _) in enumerate(STAGES):
        left_axis = axes[(stage_position, 0)]
        right_axis = axes[(stage_position, 1)]
        center = (left_axis.get_position().x0 + right_axis.get_position().x1) / 2.0
        fig.text(
            center,
            0.985,
            f"{stage_heading} {stage}",
            ha="center",
            va="top",
            fontsize=12.0,
            fontweight="normal",
        )

    stem = output_dir / stem_name
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
    floor = pd.read_csv(args.floor)
    scores_path = args.output_dir / "revised_six_metric_scores.csv"
    metrics.to_csv(scores_path, index=False)
    plot_metrics(metrics, floor, args.output_dir)
    manifest = {
        "task": "Gastrulation strict-LOO E8.0/E8.5 revised six-metric benchmark",
        "input_dir": str(args.input_dir),
        "selected_cy": 0.3,
        "knn_k": args.knn_k,
        "support_recall": (
            "For each real target cell, sum normalized predicted mass inside its "
            "real-data kNN radius; convert that mass to an expected hit probability "
            "using the method's weight ESS, then average over target cells."
        ),
        "composition_similarity": "1 - sqrt(base-2 composition JSD)",
        "w2": "Original strict-LOO W2 with the reference-only random-half sampling-floor interval",
        "atac": (
            "Every RNA prediction was mapped by the corresponding strict LOTO T at "
            "the held-out physical time in the source benchmark."
        ),
        "mass": (
            "COATI unbalanced, CytoBridge unbalanced, and TIGON use native "
            "normalized particle mass; all other included predictions use uniform mass."
        ),
        "trajectorynet": "Gaussian-base strict-LOO prediction with the LOO-specific clock",
        "excluded_methods": (
            "OT(RNA) and UOT(RNA) were removed from the comparison by request."
        ),
        "sampling_floor": str(args.floor),
        "figure_size_inches": [8.05, 4.75],
        "font": "Arial; 12 pt titles and 10 pt method/value labels",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    columns = [
        "stage",
        "display",
        "rna_target_support_recall",
        "atac_target_support_recall",
        "rna_w2",
        "atac_w2",
        "rna_composition_similarity",
        "atac_composition_similarity",
    ]
    print(metrics[columns].to_string(index=False))


if __name__ == "__main__":
    main()
