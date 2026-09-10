#!/usr/bin/env python
"""Recompute and plot the revised palate strict-LOO six-metric benchmark.

This is the palate counterpart of plot_pancreas_strict_loo_six_metrics_revised.py.
It replaces 1NN effective coverage with 15NN mass-aware target support recall,
reports composition as 1-sqrt(JSD), and retains the existing normalized-space
W2 values with stage- and modality-matched random-half sampling floors.
"""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from geomloss import SamplesLoss
from sklearn.neighbors import NearestNeighbors

from evaluate_gastrulation_paired_knn_atac import (
    _microcluster_distribution,
    _weighted_support_sinkhorn,
)
from evaluate_palate_loo_same_space import _load_scale
from trainfbench_plot_style import apply_nature_rc, method_style


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results/palate_strict_sync_loo_full_t_gaga10"
DEFAULT_OUTPUT = ROOT / "results/palate_strict_loo_six_metrics_revised"
RNA_FILE = ROOT / "data/palate_rna_cytobridge.h5ad"
ATAC_FILE = ROOT / "data/palate_atac_benchmark.h5ad"
RNA_NORM = ROOT / "data/palate_rna_primal_norm_params.pt"
ATAC_NORM = ROOT / "data/palate_atac_secondary_norm_params_lsi15.pt"

METHODS = (
    ("COATI-B", "COATI bal.", "COATI balanced"),
    ("COATI-U", "COATI unbal.", "COATI unbalanced"),
    ("CytoBridge balanced", "CytoBridge bal.", "CytoBridge balanced"),
    ("CytoBridge unbalanced", "CytoBridge unbal.", "CytoBridge unbalanced"),
    ("TrajectoryNet", "TrajectoryNet", "TrajectoryNet"),
    ("MIOFlow", "MIOFlow", "MIOFlow"),
)
SCENARIOS = (
    ("loo_time1", "E13.5", "e13p5", 1.0),
    ("loo_time2", "E14.0", "e14p0", 1.5),
)
PANELS = (
    ("rna_target_support_recall", "RNA support\nrecall ↑"),
    ("atac_target_support_recall", "ATAC support\nrecall ↑"),
    ("rna_w2", r"RNA $W_2$ ↓"),
    ("atac_w2", r"ATAC $W_2$ ↓"),
    ("rna_composition_similarity", "RNA comp.\nsimilarity ↑"),
    ("atac_composition_similarity", "ATAC comp.\nsimilarity ↑"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--distance-batch-size", type=int, default=256)
    parser.add_argument("--sinkhorn-clusters", type=int, default=512)
    parser.add_argument("--sinkhorn-blur", type=float, default=0.05)
    parser.add_argument("--floor-repeats", type=int, default=50)
    parser.add_argument("--floor-seed", type=int, default=20260826)
    parser.add_argument(
        "--reuse-floor-dir",
        type=Path,
        help="Reuse method-independent sampling-floor CSVs from this result directory.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Reuse revised_six_metric_scores.csv and sampling_floor_summary.csv.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_weights(weight: np.ndarray) -> np.ndarray:
    values = np.asarray(weight, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Prediction weights must be finite and nonnegative")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Prediction weights have non-positive total mass")
    return values / total


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
    prediction = np.asarray(prediction, dtype=np.float32)
    prediction_norm = np.sum(np.square(prediction), axis=1)
    result = np.empty(len(reference), dtype=np.float64)
    for start in range(0, len(reference), batch_size):
        stop = min(start + batch_size, len(reference))
        batch = np.asarray(reference[start:stop], dtype=np.float32)
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


def load_references() -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    rna = ad.read_h5ad(RNA_FILE)
    atac = ad.read_h5ad(ATAC_FILE)
    if not np.array_equal(rna.obs_names.to_numpy(str), atac.obs_names.to_numpy(str)):
        raise ValueError("RNA and ATAC benchmark cells are not paired")
    time = pd.to_numeric(rna.obs["time_point_processed"], errors="raise").to_numpy(float)
    atac_time = pd.to_numeric(atac.obs["time_point_processed"], errors="raise").to_numpy(float)
    if not np.array_equal(time, atac_time):
        raise ValueError("RNA and ATAC time labels differ")
    rna_scale = _load_scale(RNA_NORM)
    atac_scale = _load_scale(ATAC_NORM)
    rna_norm = np.asarray(rna.obsm["X_latent"], dtype=np.float32) / rna_scale
    atac_norm = np.asarray(atac.obsm["X_lsi15"], dtype=np.float32) / atac_scale
    references: dict[str, dict[str, np.ndarray]] = {}
    for scenario, stage, _, heldout_time in SCENARIOS:
        subset = np.isclose(time, heldout_time)
        references[scenario] = {
            "RNA": np.asarray(rna_norm[subset], dtype=np.float32),
            "ATAC": np.asarray(atac_norm[subset], dtype=np.float32),
        }
        if len(references[scenario]["RNA"]) == 0:
            raise ValueError(f"No reference cells for {stage}")
    audit = {
        "rna_file": str(RNA_FILE),
        "atac_file": str(ATAC_FILE),
        "rna_scale": rna_scale,
        "atac_scale": atac_scale,
        "space": "RNA PCA40 and ATAC LSI15 divided by their benchmark W2 scales",
    }
    return references, audit


def select_original_row(
    original: pd.DataFrame,
    scenario: str,
    method: str,
) -> pd.Series:
    local = original[original["scenario"].eq(scenario) & original["method"].eq(method)]
    if method in {"COATI-B", "COATI-U"}:
        local = local[np.isclose(local["cy"], 0.5)]
    if len(local) != 1:
        raise ValueError(f"Expected one original score row for {scenario}, {method}; got {len(local)}")
    return local.iloc[0]


def select_composition_jsd(
    composition: pd.DataFrame,
    scenario: str,
    method: str,
    modality: str,
) -> float:
    local = composition[
        composition["scenario"].eq(scenario)
        & composition["method"].eq(method)
        & composition["modality"].eq(modality)
        & composition["weighting"].eq("native")
    ]
    if method in {"COATI-B", "COATI-U"}:
        local = local[np.isclose(local["cy"], 0.5)]
    if len(local) != 1:
        raise ValueError(
            f"Expected one native composition row for {scenario}, {method}, {modality}; got {len(local)}"
        )
    return float(local.iloc[0]["composition_jsd"])


def build_revised_metrics(
    input_dir: Path,
    references: dict[str, dict[str, np.ndarray]],
    knn_k: int,
    distance_batch_size: int,
) -> pd.DataFrame:
    original = pd.read_csv(input_dir / "palate_loo_four_metric_scores.csv")
    composition = pd.read_csv(input_dir / "palate_loo_celltype_composition_scores.csv")
    manifest = pd.read_csv(input_dir / "prediction_manifest.csv")
    rows: list[dict[str, object]] = []
    for scenario, stage, _, heldout_time in SCENARIOS:
        radii = {
            modality: target_radii(points, knn_k)
            for modality, points in references[scenario].items()
        }
        for method, display, style_name in METHODS:
            original_row = select_original_row(original, scenario, method)
            manifest_local = manifest[
                manifest["scenario"].eq(scenario) & manifest["method"].eq(method)
            ]
            if method in {"COATI-B", "COATI-U"}:
                manifest_local = manifest_local[np.isclose(manifest_local["cy"], 0.5)]
            if len(manifest_local) != 1:
                raise ValueError(f"Expected one manifest row for {scenario}, {method}")
            item = manifest_local.iloc[0]
            prediction_file = input_dir / str(item["prediction_file"])
            if sha256(prediction_file) != str(item["prediction_sha256"]):
                raise ValueError(f"Prediction checksum mismatch: {prediction_file}")
            with np.load(prediction_file, allow_pickle=True) as saved:
                prediction = {
                    "RNA": np.asarray(saved["rna_norm"], dtype=np.float32),
                    "ATAC": np.asarray(saved["atac_norm"], dtype=np.float32),
                }
                weights = normalized_weights(saved["weights"])
            effective_n = float(1.0 / np.sum(np.square(weights)))
            row: dict[str, object] = {
                "scenario": scenario,
                "heldout_stage": stage,
                "heldout_time": heldout_time,
                "method": method,
                "display": display,
                "style": style_name,
                "weighting": "native_mass" if effective_n < len(weights) - 1e-6 else "uniform",
                "effective_particle_count": effective_n,
                "prediction_file": str(prediction_file),
                "prediction_sha256": str(item["prediction_sha256"]),
                "rna_w2": float(original_row["rna_w2_distance"]),
                "atac_w2": float(original_row["atac_w2_distance"]),
            }
            for modality, prefix in (("RNA", "rna"), ("ATAC", "atac")):
                local_mass = local_predicted_mass(
                    references[scenario][modality],
                    prediction[modality],
                    weights,
                    radii[modality],
                    distance_batch_size,
                )
                row[f"{prefix}_target_support_recall"] = expected_mass_aware_recall(
                    local_mass, effective_n
                )
                row[f"{prefix}_hard_geometric_recall"] = float(np.mean(local_mass > 0))
                jsd = select_composition_jsd(composition, scenario, method, modality)
                row[f"{prefix}_composition_similarity"] = float(1.0 - np.sqrt(jsd))
                row[f"{prefix}_original_jsd"] = jsd
            rows.append(row)
            print(
                f"[{stage}] {display}: RNA recall={row['rna_target_support_recall']:.4f}, "
                f"ATAC recall={row['atac_target_support_recall']:.4f}",
                flush=True,
            )
    return pd.DataFrame(rows)


def sampling_floor(
    reference: np.ndarray,
    *,
    scenario: str,
    stage: str,
    modality: str,
    repeats: int,
    seed: int,
    sinkhorn_clusters: int,
    sinkhorn: SamplesLoss,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    points = np.asarray(reference, dtype=np.float32)
    half = len(points) // 2
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        order = rng.permutation(len(points))
        first = points[order[:half]]
        second = points[order[half : 2 * half]]
        first_uniform = np.full(len(first), 1.0 / len(first), dtype=np.float64)
        second_uniform = np.full(len(second), 1.0 / len(second), dtype=np.float64)
        weights_a, centers_a, _ = _microcluster_distribution(
            first, first_uniform, sinkhorn_clusters
        )
        weights_b, centers_b, _ = _microcluster_distribution(
            second, second_uniform, sinkhorn_clusters
        )
        divergence = _weighted_support_sinkhorn(
            weights_a, centers_a, weights_b, centers_b, sinkhorn
        )
        rows.append(
            {
                "scenario": scenario,
                "stage": stage,
                "modality": modality,
                "repeat": repeat,
                "batch_size": half,
                "sinkhorn_divergence": divergence,
                "w2": float(np.sqrt(2.0 * max(divergence, 0.0))),
            }
        )
        print(f"[floor {stage} {modality}] {repeat + 1}/{repeats}", flush=True)
    values = np.asarray([float(row["w2"]) for row in rows])
    return (
        {
            "scenario": scenario,
            "stage": stage,
            "modality": modality,
            "n_reference_cells": len(points),
            "batch_size": half,
            "repeats": repeats,
            "mean_w2": float(values.mean()),
            "sd_w2": float(values.std(ddof=1)) if repeats > 1 else 0.0,
            "median_w2": float(np.median(values)),
            "q025_w2": float(np.quantile(values, 0.025)),
            "q975_w2": float(np.quantile(values, 0.975)),
        },
        rows,
    )


def configure_style() -> None:
    apply_nature_rc(font_size=10.0)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "font.weight": "normal",
            "axes.titlesize": 10.5,
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 10.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def metric_axis_maxima(metrics: pd.DataFrame, floor: pd.DataFrame) -> dict[str, float]:
    maxima: dict[str, float] = {}
    for column, _ in PANELS:
        if column.endswith("_w2"):
            modality = "RNA" if column.startswith("rna") else "ATAC"
            floor_upper = float(floor.loc[floor["modality"].eq(modality), "q975_w2"].max())
            maxima[column] = 1.45 * max(float(metrics[column].max()), floor_upper)
        else:
            # Match the revised pancreas layout: leave a fixed right-hand
            # value column for bounded higher-is-better metrics.
            maxima[column] = 1.35
    return maxima


def draw_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    floor: pd.DataFrame,
    column: str,
    title: str,
    axis_maximum: float,
    show_ylabels: bool,
) -> None:
    positions = np.arange(len(frame))
    values = frame[column].to_numpy(float)
    for position, (_, row), value in zip(positions, frame.iterrows(), values):
        style = method_style(str(row["style"]))
        bar = ax.barh(
            position,
            value,
            height=0.52,
            color=style.color,
            edgecolor=style.markeredgecolor or style.color,
            linewidth=0.7,
            zorder=2,
        )[0]
        if str(row["method"]) == "TrajectoryNet":
            bar.set_edgecolor(style.color)
            bar.set_linewidth(0.0)
        ax.text(
            0.985,
            position,
            f"{value:.3f}",
            ha="right",
            va="center",
            fontsize=8.2,
            fontweight="normal",
            color="#222222",
            clip_on=False,
            zorder=3,
            transform=ax.get_yaxis_transform(),
        )

    if column.endswith("_w2"):
        modality = "RNA" if column.startswith("rna") else "ATAC"
        floor_row = floor[
            floor["scenario"].eq(str(frame.iloc[0]["scenario"]))
            & floor["modality"].eq(modality)
        ].iloc[0]
        ax.axvspan(
            float(floor_row["q025_w2"]),
            float(floor_row["q975_w2"]),
            color="#B3B3B3",
            alpha=0.34,
            lw=0,
            zorder=0,
        )
        ax.axvline(float(floor_row["mean_w2"]), color="#666666", lw=0.8, ls="--", zorder=1)

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
    ax.set_xlim(0.0, axis_maximum)
    ax.set_ylim(len(frame) - 0.38, -0.72)
    ax.set_xticks([])
    ax.set_yticks(positions, frame["display"].tolist() if show_ylabels else [])
    ax.tick_params(length=0, pad=2)
    ax.set_title(title, pad=2.0, fontweight="normal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_stage(
    metrics: pd.DataFrame,
    floor: pd.DataFrame,
    scenario: str,
    stage: str,
    stage_slug: str,
    maxima: dict[str, float],
    output_dir: Path,
) -> None:
    frame = metrics[metrics["scenario"].eq(scenario)].copy()
    frame["method"] = pd.Categorical(frame["method"], [method for method, _, _ in METHODS], ordered=True)
    frame = frame.sort_values("method")
    fig, axes = plt.subplots(3, 2, figsize=(4.13, 4.75), facecolor="white")
    for panel_index, (ax, (column, title)) in enumerate(zip(axes.ravel(), PANELS)):
        draw_panel(ax, frame, floor, column, title, maxima[column], panel_index % 2 == 0)
    fig.suptitle(
        f"Held-out {stage}", fontsize=12.0, fontweight="normal", y=0.985
    )
    fig.subplots_adjust(left=0.39, right=0.985, top=0.870, bottom=0.035, hspace=0.43, wspace=0.15)
    stem = output_dir / f"palate_strict_loo_{stage_slug}_six_metrics_revised_20k"
    options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.02}
    fig.savefig(stem.with_suffix(".png"), dpi=600, **options)
    fig.savefig(stem.with_suffix(".pdf"), **options)
    fig.savefig(stem.with_suffix(".svg"), **options)
    plt.close(fig)


def plot_combined(
    metrics: pd.DataFrame,
    floor: pd.DataFrame,
    maxima: dict[str, float],
    output_dir: Path,
) -> None:
    fig = plt.figure(figsize=(8.0, 4.75), facecolor="white")
    outer = fig.add_gridspec(1, 2, left=0.19, right=0.995, top=0.870, bottom=0.035, wspace=0.45)
    for stage_column, (scenario, stage, _, _) in enumerate(SCENARIOS):
        frame = metrics[metrics["scenario"].eq(scenario)].copy()
        frame["method"] = pd.Categorical(frame["method"], [method for method, _, _ in METHODS], ordered=True)
        frame = frame.sort_values("method")
        inner = outer[0, stage_column].subgridspec(3, 2, hspace=0.43, wspace=0.15)
        for panel_index, (column, title) in enumerate(PANELS):
            row, col = divmod(panel_index, 2)
            ax = fig.add_subplot(inner[row, col])
            draw_panel(ax, frame, floor, column, title, maxima[column], col == 0)
        center = 0.34 if stage_column == 0 else 0.83
        fig.text(
            center,
            0.985,
            f"Held-out {stage}",
            ha="center",
            va="top",
            fontsize=12.0,
            fontweight="normal",
        )
    stem = output_dir / "palate_strict_loo_e13p5_e14p0_six_metrics_revised_20k"
    options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.02}
    fig.savefig(stem.with_suffix(".png"), dpi=600, **options)
    fig.savefig(stem.with_suffix(".pdf"), **options)
    fig.savefig(stem.with_suffix(".svg"), **options)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    if args.plot_only:
        metrics = pd.read_csv(args.output_dir / "revised_six_metric_scores.csv")
        floor = pd.read_csv(args.output_dir / "sampling_floor_summary.csv")
        maxima = metric_axis_maxima(metrics, floor)
        pd.DataFrame(
            [
                {
                    "metric": column,
                    "shared_axis_minimum": 0.0,
                    "shared_axis_maximum": maximum,
                }
                for column, maximum in maxima.items()
            ]
        ).to_csv(args.output_dir / "shared_axis_scales.csv", index=False)
        for scenario, stage, stage_slug, _ in SCENARIOS:
            plot_stage(metrics, floor, scenario, stage, stage_slug, maxima, args.output_dir)
        plot_combined(metrics, floor, maxima, args.output_dir)
        print(f"Replotted existing results in {args.output_dir}", flush=True)
        return
    references, reference_audit = load_references()
    metrics = build_revised_metrics(
        args.input_dir,
        references,
        args.knn_k,
        args.distance_batch_size,
    )

    if args.reuse_floor_dir is not None:
        floor = pd.read_csv(args.reuse_floor_dir / "sampling_floor_summary.csv")
        expected = {(scenario, modality) for scenario, _, _, _ in SCENARIOS for modality in ("RNA", "ATAC")}
        observed = set(zip(floor["scenario"], floor["modality"]))
        if observed != expected:
            raise ValueError(f"Reused sampling floor has unexpected rows: {observed}")
        repeat_source = args.reuse_floor_dir / "sampling_floor_repeats.csv"
        floor_repeats_frame = pd.read_csv(repeat_source) if repeat_source.is_file() else pd.DataFrame()
    else:
        sinkhorn = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=args.sinkhorn_blur,
            debias=True,
            backend="tensorized",
        )
        floor_summaries: list[dict[str, object]] = []
        floor_repeats: list[dict[str, object]] = []
        for scenario_index, (scenario, stage, _, _) in enumerate(SCENARIOS):
            for modality_index, modality in enumerate(("RNA", "ATAC")):
                summary, repeats = sampling_floor(
                    references[scenario][modality],
                    scenario=scenario,
                    stage=stage,
                    modality=modality,
                    repeats=args.floor_repeats,
                    seed=args.floor_seed + 1000 * scenario_index + 100 * modality_index,
                    sinkhorn_clusters=args.sinkhorn_clusters,
                    sinkhorn=sinkhorn,
                )
                floor_summaries.append(summary)
                floor_repeats.extend(repeats)
        floor = pd.DataFrame(floor_summaries)
        floor_repeats_frame = pd.DataFrame(floor_repeats)
    metrics.to_csv(args.output_dir / "revised_six_metric_scores.csv", index=False)
    floor.to_csv(args.output_dir / "sampling_floor_summary.csv", index=False)
    floor_repeats_frame.to_csv(args.output_dir / "sampling_floor_repeats.csv", index=False)
    maxima = metric_axis_maxima(metrics, floor)
    pd.DataFrame(
        [{"metric": column, "shared_axis_minimum": 0.0, "shared_axis_maximum": maximum} for column, maximum in maxima.items()]
    ).to_csv(args.output_dir / "shared_axis_scales.csv", index=False)
    for scenario, stage, stage_slug, _ in SCENARIOS:
        plot_stage(metrics, floor, scenario, stage, stage_slug, maxima, args.output_dir)
    plot_combined(metrics, floor, maxima, args.output_dir)

    manifest = {
        "task": "Palate strict-LOO E13.5/E14.0 revised six-metric benchmark",
        "input_dir": str(args.input_dir),
        "knn_k": args.knn_k,
        "support_recall": (
            "For each real target cell, sum normalized predicted mass inside its "
            "real-data kNN radius; convert local mass to expected hit probability "
            "using the method's weight ESS; average over real target cells."
        ),
        "composition_similarity": "1 - sqrt(base-2 composition JSD)",
        "w2": "Existing common-normalized-space W2 with newly computed random-half sampling-floor interval",
        "floor_repeats": args.floor_repeats,
        "floor_seed": args.floor_seed,
        "floor_source": (
            str(args.reuse_floor_dir) if args.reuse_floor_dir is not None else "recomputed"
        ),
        "sinkhorn_clusters": args.sinkhorn_clusters,
        "sinkhorn_blur": args.sinkhorn_blur,
        "coati_training": "strict LOO trajectory model, Cy=0.5",
        "evaluation_T": "shared full-data RNA-to-ATAC T",
        "trajectorynet": "terminal-origin reverse-trained model",
        "mioflow": "GAGA10 model",
        "reference_audit": reference_audit,
        "shared_scales": "Each metric uses one identical raw x-axis scale across E13.5 and E14.0",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)
    print(f"Saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
