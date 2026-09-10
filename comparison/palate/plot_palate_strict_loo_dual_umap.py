#!/usr/bin/env python
"""Plot palate strict-LOO predictions in fixed RNA and mapped-ATAC UMAPs."""

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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import joblib
import matplotlib as mpl

mpl.use("Agg")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from umap import UMAP

from trainfbench_plot_style import apply_nature_rc, method_style


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results/palate_strict_sync_loo_full_t_gaga10_tn_reversed"
DEFAULT_METRICS = (
    ROOT
    / "results/palate_strict_loo_six_metrics_revised_tn_reversed/revised_six_metric_scores.csv"
)
DEFAULT_OUTPUT = ROOT / "results/palate_strict_loo_dual_umap_tn_reversed"
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
PANEL_LAYOUT = (
    ("observed", "TrajectoryNet"),
    ("CytoBridge balanced", "CytoBridge unbalanced"),
    ("COATI-B", "COATI-U"),
    ("MIOFlow", None),
)
SCENARIOS = (
    ("loo_time1", "E13.5", "e13p5", 1.0),
    ("loo_time2", "E14.0", "e14p0", 1.5),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-mioflow",
        action="store_true",
        help="Omit the MIOFlow panel and write a compact five-method figure.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_scale(path: Path) -> float:
    state = torch.load(path, map_location="cpu", weights_only=False)
    value = float(np.asarray(state["scale"]).reshape(-1)[0])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid normalization scale in {path}: {value}")
    return value


def load_real_atlases() -> tuple[dict[str, np.ndarray], np.ndarray]:
    rna = ad.read_h5ad(RNA_FILE)
    atac = ad.read_h5ad(ATAC_FILE)
    if not np.array_equal(rna.obs_names.to_numpy(str), atac.obs_names.to_numpy(str)):
        raise ValueError("RNA and ATAC cells are not paired")
    times = pd.to_numeric(rna.obs["time_point_processed"], errors="raise").to_numpy(float)
    atac_times = pd.to_numeric(atac.obs["time_point_processed"], errors="raise").to_numpy(float)
    if not np.array_equal(times, atac_times):
        raise ValueError("RNA and ATAC time labels differ")
    real = {
        "RNA": np.asarray(rna.obsm["X_latent"], dtype=np.float32) / load_scale(RNA_NORM),
        "ATAC": np.asarray(atac.obsm["X_lsi15"], dtype=np.float32) / load_scale(ATAC_NORM),
    }
    return real, times


def select_manifest_row(frame: pd.DataFrame, scenario: str, method: str) -> pd.Series:
    local = frame[frame["scenario"].eq(scenario) & frame["method"].eq(method)]
    if method in {"COATI-B", "COATI-U"}:
        local = local[np.isclose(local["cy"], 0.5)]
    if len(local) != 1:
        raise ValueError(f"Expected one prediction for {scenario}, {method}; got {len(local)}")
    return local.iloc[0]


def load_predictions(input_dir: Path) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    manifest = pd.read_csv(input_dir / "prediction_manifest.csv")
    result: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for scenario, _, _, _ in SCENARIOS:
        result[scenario] = {}
        for method, _, _ in METHODS:
            item = select_manifest_row(manifest, scenario, method)
            path = input_dir / str(item["prediction_file"])
            if sha256(path) != str(item["prediction_sha256"]):
                raise ValueError(f"Prediction checksum mismatch: {path}")
            with np.load(path, allow_pickle=True) as saved:
                weights = np.asarray(saved["weights"], dtype=np.float64).reshape(-1)
                weights = weights / weights.sum()
                result[scenario][method] = {
                    "RNA": np.asarray(saved["rna_norm"], dtype=np.float32),
                    "ATAC": np.asarray(saved["atac_norm"], dtype=np.float32),
                    "weights": weights,
                }
    return result


def point_sizes(weights: np.ndarray) -> np.ndarray:
    relative = np.sqrt(np.asarray(weights, dtype=np.float64) * len(weights))
    return np.clip(0.72 * relative, 0.28, 2.60)


def fit_and_transform(
    real: np.ndarray,
    predictions: dict[str, dict[str, dict[str, np.ndarray]]],
    modality: str,
    *,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> tuple[UMAP, np.ndarray, dict[str, dict[str, np.ndarray]]]:
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
    keys = [
        (scenario, method)
        for scenario, _, _, _ in SCENARIOS
        for method, _, _ in METHODS
    ]
    lengths = [len(predictions[scenario][method][modality]) for scenario, method in keys]
    stacked = np.concatenate(
        [predictions[scenario][method][modality] for scenario, method in keys], axis=0
    )
    transformed = np.asarray(model.transform(stacked), dtype=np.float32)
    pieces = np.split(transformed, np.cumsum(lengths)[:-1])
    output: dict[str, dict[str, np.ndarray]] = {scenario: {} for scenario, _, _, _ in SCENARIOS}
    for (scenario, method), values in zip(keys, pieces):
        output[scenario][method] = values
    return model, real_umap, output


def robust_limits(
    real_umap: np.ndarray,
    predicted_umap: dict[str, dict[str, np.ndarray]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    all_points = np.concatenate(
        [real_umap]
        + [values for scenario in predicted_umap.values() for values in scenario.values()],
        axis=0,
    )
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
    stage: str,
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
        s=0.30,
        c="#D1D1D1",
        alpha=0.50,
        linewidths=0,
        rasterized=True,
        zorder=0,
    )
    if method == "observed":
        heldout = real_umap[heldout_mask]
        ax.scatter(
            heldout[:, 0],
            heldout[:, 1],
            s=0.50,
            c="#111111",
            alpha=0.82,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        title = f"Observed {stage}\nin global {modality} atlas"
    else:
        _, display, style_name = next(item for item in METHODS if item[0] == method)
        style = method_style(style_name)
        points = predicted_umap[method]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=point_sizes(predictions[method]["weights"]),
            c=style.color,
            alpha=0.78,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        column = "rna_w2" if modality == "RNA" else "atac_w2"
        value = float(metrics.loc[metrics["method"].eq(method), column].iloc[0])
        title = f"{display}\n{modality} $W_2$ = {value:.3f}"
    ax.set_title(title, fontsize=8.0, fontweight="normal", pad=1.0)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_stage(
    output_dir: Path,
    *,
    scenario: str,
    stage: str,
    stage_slug: str,
    heldout_time: float,
    times: np.ndarray,
    real_umaps: dict[str, np.ndarray],
    predicted_umaps: dict[str, dict[str, dict[str, np.ndarray]]],
    predictions: dict[str, dict[str, dict[str, np.ndarray]]],
    metrics: pd.DataFrame,
    limits_by_modality: dict[str, tuple[tuple[float, float], tuple[float, float]]],
    exclude_mioflow: bool,
) -> None:
    apply_nature_rc(font_size=10.0)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "legend.fontsize": 10.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    panel_layout = PANEL_LAYOUT[:-1] if exclude_mioflow else PANEL_LAYOUT
    figure_height = 5.55 if exclude_mioflow else 7.05
    fig, axes = plt.subplots(
        len(panel_layout), 4, figsize=(7.15, figure_height), facecolor="white"
    )
    heldout_mask = np.isclose(times, heldout_time)
    scenario_metrics = metrics[metrics["scenario"].eq(scenario)]
    for group, modality in enumerate(("RNA", "ATAC")):
        column_offset = 2 * group
        for row, methods in enumerate(panel_layout):
            for local_column, method in enumerate(methods):
                ax = axes[row, column_offset + local_column]
                if method is None:
                    ax.axis("off")
                    continue
                draw_panel(
                    ax,
                    modality=modality,
                    method=method,
                    stage=stage,
                    real_umap=real_umaps[modality],
                    heldout_mask=heldout_mask,
                    predicted_umap=predicted_umaps[modality][scenario],
                    predictions=predictions[scenario],
                    metrics=scenario_metrics,
                    limits=limits_by_modality[modality],
                )

    fig.text(
        0.255,
        0.992,
        f"Held-out {stage} RNA distribution",
        ha="center",
        va="top",
        fontsize=10.2,
        fontweight="normal",
    )
    fig.text(
        0.755,
        0.992,
        f"Held-out {stage} RNA-to-ATAC mapping",
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
            markerfacecolor="#111111", markeredgecolor="#111111", label=f"Real {stage}"
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
            bbox_to_anchor=(x, 0.958),
            ncol=3,
            frameon=False,
            fontsize=7.0,
            handletextpad=0.35,
            columnspacing=0.9,
        )
    fig.subplots_adjust(
        left=0.025,
        right=0.985,
        top=0.855,
        bottom=0.018,
        hspace=0.30,
        wspace=0.10,
    )
    method_count = 5 if exclude_mioflow else 6
    stem = output_dir / f"palate_strict_loo_{stage_slug}_rna_atac_umap_{method_count}methods"
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
    real, times = load_real_atlases()
    predictions = load_predictions(args.input_dir)
    metrics = pd.read_csv(args.metrics)

    models: dict[str, UMAP] = {}
    real_umaps: dict[str, np.ndarray] = {}
    predicted_umaps: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for modality in ("RNA", "ATAC"):
        print(f"[UMAP] fitting global {modality} atlas {real[modality].shape}", flush=True)
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

    limits_by_modality = {
        modality: robust_limits(real_umaps[modality], predicted_umaps[modality])
        for modality in ("RNA", "ATAC")
    }
    archive: dict[str, np.ndarray] = {
        "physical_time": times.astype(np.float32),
        "real_rna_umap": real_umaps["RNA"],
        "real_atac_umap": real_umaps["ATAC"],
    }
    archive_methods = METHODS[:-1] if args.exclude_mioflow else METHODS
    for scenario, _, _, _ in SCENARIOS:
        for index, (method, _, _) in enumerate(archive_methods):
            archive[f"{scenario}_method{index}_name"] = np.asarray(method)
            archive[f"{scenario}_method{index}_rna_umap"] = predicted_umaps["RNA"][scenario][method]
            archive[f"{scenario}_method{index}_atac_umap"] = predicted_umaps["ATAC"][scenario][method]
    np.savez_compressed(args.output_dir / "umap_coordinates.npz", **archive)

    for scenario, stage, stage_slug, heldout_time in SCENARIOS:
        plot_stage(
            args.output_dir,
            scenario=scenario,
            stage=stage,
            stage_slug=stage_slug,
            heldout_time=heldout_time,
            times=times,
            real_umaps=real_umaps,
            predicted_umaps=predicted_umaps,
            predictions=predictions,
            metrics=metrics,
            limits_by_modality=limits_by_modality,
            exclude_mioflow=args.exclude_mioflow,
        )

    manifest = {
        "task": "Palate strict-LOO RNA and mapped-ATAC UMAP comparison",
        "scenarios": {scenario: stage for scenario, stage, _, _ in SCENARIOS},
        "rna_umap_input": "all observed stages in normalized RNA PCA40",
        "atac_umap_input": "all observed stages in normalized ATAC LSI15",
        "umap_policy": (
            "One fixed global UMAP per modality; fit only on real atlas cells, then "
            "transform every method and both LOO scenarios."
        ),
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "random_state": args.seed,
        "prediction_weight_display": (
            "point area scales with sqrt(normalized native mass times particle count); "
            "uniform for methods without native mass"
        ),
        "metrics": str(args.metrics.resolve()),
        "predictions": str((args.input_dir / "prediction_manifest.csv").resolve()),
        "trajectorynet": "reversed-rank 20k, official piecewise density path from common E12.5 cells",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
