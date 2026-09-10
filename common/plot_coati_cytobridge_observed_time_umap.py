#!/usr/bin/env python
"""Compare COATI and CytoBridge predictions with observed RNA10 UMAP snapshots."""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[1]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

from analyze_cytobridge_synthetic import transform_shared_umap  # noqa: E402


SYNTHETIC = Path("external/COATI/Synthetic")
DATA_DIR = SYNTHETIC / "5scRNA"
DEFAULT_COATI_CACHE = (
    SYNTHETIC
    / "UnbalancedSyncSweep/outputs/bio_all1_alpha100_d10_e1"
    / "observed_time_distribution/all_initial_observed_time_predictions.npz"
)
DEFAULT_CYTOBRIDGE_CACHE = (
    ROOT
    / "results/cytobridge_synthetic_rna10_d10_e0p1_n128_i3000_observed_times"
    / "cytobridge_observed_time_cache.npz"
)
DEFAULT_UMAP_DIR = (
    ROOT / "results/cytobridge_synthetic_rna10_n128_i3000_comparison/rna10_umap"
)
DEFAULT_SINKHORN = (
    ROOT
    / "results/synthetic_rna10_same_space_predicted_time_sinkhorn_e0p1_i3000"
    / "predicted_time_sinkhorn.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/coati_cytobridge_observed_time_umap_comparison_e0p1"
)

TIME_KEYS = ("time2", "time3", "time4")
METHODS = (
    ("COATI balanced", "coati_bal", "coati"),
    ("CytoBridge balanced", "cytobridge_balanced", "cytobridge"),
    ("COATI unbalanced", "coati_unbal", "coati"),
    ("CytoBridge unbalanced", "cytobridge_unbalanced", "cytobridge"),
)
SOURCE_COLORS = {"4_1": "#4477AA", "4_5": "#EE7733"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coati-cache", type=Path, default=DEFAULT_COATI_CACHE)
    parser.add_argument(
        "--cytobridge-cache", type=Path, default=DEFAULT_CYTOBRIDGE_CACHE
    )
    parser.add_argument(
        "--rna-data", type=Path, default=DATA_DIR / "all_time_scRNA_pca10.npz"
    )
    parser.add_argument(
        "--labels", type=Path, default=DATA_DIR / "all_time_scRNA_label.npz"
    )
    parser.add_argument(
        "--umap-model", type=Path, default=DEFAULT_UMAP_DIR / "rna10_umap_model.joblib"
    )
    parser.add_argument(
        "--observed-umap",
        type=Path,
        default=DEFAULT_UMAP_DIR / "observed_rna10_umap.npz",
    )
    parser.add_argument(
        "--umap-python", type=Path, default=Path(_sys.executable)
    )
    parser.add_argument(
        "--umap-helper", type=Path, default=ROOT / "common/transform_joblib_umap.py"
    )
    parser.add_argument("--sinkhorn-csv", type=Path, default=DEFAULT_SINKHORN)
    parser.add_argument(
        "--cytobridge-subtitle",
        default="energy = 0.1",
        help="Second-line label shown under both CytoBridge column titles.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    outputs = (
        output_dir / "coati_cytobridge_observed_time_umap.png",
        output_dir / "coati_cytobridge_observed_time_umap.pdf",
        output_dir / "comparison_coordinates.npz",
        output_dir / "manifest.json",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs; pass --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    rna_archive = np.load(args.rna_data, allow_pickle=False)
    label_archive = np.load(args.labels, allow_pickle=True)
    source = label_archive["time1"].astype(str)
    if source.shape != (2006,) or set(np.unique(source)) != set(SOURCE_COLORS):
        raise ValueError(f"Unexpected time1 source labels: {np.unique(source)}")

    observed_all = np.asarray(
        np.load(args.observed_umap, allow_pickle=False)["observed_umap"],
        dtype=np.float32,
    )
    observed_shapes = [len(rna_archive[f"time{i}"]) for i in range(1, 5)]
    offsets = np.cumsum([0] + observed_shapes)
    if observed_all.shape != (offsets[-1], 2):
        raise ValueError(f"Unexpected observed UMAP shape: {observed_all.shape}")
    observed = {
        f"time{i}": observed_all[offsets[i - 1] : offsets[i]]
        for i in range(1, 5)
    }

    coati = np.load(args.coati_cache, allow_pickle=False)
    cytobridge = np.load(args.cytobridge_cache, allow_pickle=False)
    coordinates: dict[tuple[str, str], np.ndarray] = {}
    for _, prefix, source_name in METHODS:
        if source_name == "coati":
            for time_key in TIME_KEYS:
                values = np.asarray(
                    coati[f"{prefix}_{time_key}_umap"], dtype=np.float32
                )
                if values.shape != (2006, 2):
                    raise ValueError(
                        f"Unexpected {prefix} {time_key} UMAP shape: {values.shape}"
                    )
                coordinates[(prefix, time_key)] = values
        else:
            raw_key = f"{prefix}_all_initial_selected_rna10_raw"
            raw = np.asarray(cytobridge[raw_key], dtype=np.float32)
            if raw.shape != (3, 2006, 10):
                raise ValueError(f"Unexpected {raw_key} shape: {raw.shape}")
            projected = transform_shared_umap(
                args.umap_python,
                args.umap_helper,
                args.umap_model,
                raw.reshape(-1, 10),
            ).reshape(3, 2006, 2)
            for time_index, time_key in enumerate(TIME_KEYS):
                coordinates[(prefix, time_key)] = projected[time_index]

    scores = pd.read_csv(args.sinkhorn_csv)
    equal_scores = scores[scores["weighting"] == "equal_particle"].set_index(
        ["method", "time_key"]
    )["sinkhorn_divergence"]
    for title, _, _ in METHODS:
        for time_key in TIME_KEYS:
            if (title, time_key) not in equal_scores.index:
                raise ValueError(f"Missing equal-particle Sinkhorn: {title} {time_key}")

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
        }
    )
    figure, axes = plt.subplots(3, 5, figsize=(14.5, 8.4), squeeze=False)
    figure.subplots_adjust(
        left=0.055,
        right=0.995,
        top=0.91,
        bottom=0.075,
        wspace=0.035,
        hspace=0.12,
    )
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(source))
    source_colors = np.asarray([SOURCE_COLORS[value] for value in source])

    for row, time_key in enumerate(TIME_KEYS):
        real = observed[time_key]
        all_row = np.concatenate(
            [real] + [coordinates[(prefix, time_key)] for _, prefix, _ in METHODS],
            axis=0,
        )
        lower = all_row.min(axis=0)
        upper = all_row.max(axis=0)
        padding = np.maximum((upper - lower) * 0.035, 0.25)
        lower -= padding
        upper += padding

        observed_axis = axes[row, 0]
        observed_axis.scatter(
            real[:, 0],
            real[:, 1],
            s=5,
            color="#707070",
            alpha=0.38,
            edgecolors="none",
            rasterized=True,
        )
        observed_axis.text(
            0.03,
            0.02,
            f"n = {len(real):,}",
            transform=observed_axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
        )

        for column, (title, prefix, _) in enumerate(METHODS, start=1):
            axis = axes[row, column]
            axis.scatter(
                real[:, 0],
                real[:, 1],
                s=4,
                color="#999999",
                alpha=0.14,
                edgecolors="none",
                rasterized=True,
            )
            predicted = coordinates[(prefix, time_key)]
            axis.scatter(
                predicted[order, 0],
                predicted[order, 1],
                s=8,
                c=source_colors[order],
                alpha=0.48,
                edgecolors="none",
                rasterized=True,
            )
            value = float(equal_scores.loc[(title, time_key)])
            axis.text(
                0.03,
                0.02,
                f"Equal-weight Sinkhorn = {value:.4f}",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
            )

        for column, axis in enumerate(axes[row]):
            axis.set_xlim(lower[0], upper[0])
            axis.set_ylim(lower[1], upper[1])
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#8A8A8A")
                spine.set_linewidth(0.8)
        axes[row, 0].set_ylabel(time_key, fontsize=12, fontweight="normal")

    column_titles = (
        "Observed",
        "COATI balanced",
        f"CytoBridge balanced\n{args.cytobridge_subtitle}",
        "COATI unbalanced",
        f"CytoBridge unbalanced\n{args.cytobridge_subtitle}",
    )
    for axis, title in zip(axes[0], column_titles):
        axis.set_title(title, pad=7, fontweight="normal")

    figure.suptitle(
        "Observed-time distributions in the shared RNA10 UMAP",
        fontsize=15,
        fontweight="normal",
        y=0.985,
    )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=SOURCE_COLORS["4_1"],
                label="predicted from 4_1",
                markersize=6,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=SOURCE_COLORS["4_5"],
                label="predicted from 4_5",
                markersize=6,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color="#999999",
                alpha=0.5,
                label="observed snapshot",
                markersize=6,
            ),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.008),
    )
    figure.savefig(outputs[0], dpi=300, facecolor="white")
    figure.savefig(outputs[1], facecolor="white")
    plt.close(figure)

    cache_arrays: dict[str, np.ndarray] = {
        "initial_source": source,
        **{f"observed_{key}": observed[key] for key in TIME_KEYS},
    }
    for _, prefix, _ in METHODS:
        for time_key in TIME_KEYS:
            cache_arrays[f"{prefix}_{time_key}"] = coordinates[(prefix, time_key)]
    np.savez_compressed(outputs[2], **cache_arrays)
    outputs[3].write_text(
        json.dumps(
            {
                "display": {
                    "times": list(TIME_KEYS),
                    "predicted_cells": 2006,
                    "prediction_weighting": "equal particle display",
                    "observed_background": "corresponding real snapshot",
                    "prediction_color": "initial time1 source identity",
                    "cytobridge_setting": "density=10, energy=0.1, 3000 main epochs",
                },
                "inputs": {
                    "coati_cache": str(args.coati_cache.resolve()),
                    "cytobridge_cache": str(args.cytobridge_cache.resolve()),
                    "umap_model": str(args.umap_model.resolve()),
                    "observed_umap": str(args.observed_umap.resolve()),
                    "sinkhorn_csv": str(args.sinkhorn_csv.resolve()),
                },
                "outputs": {path.name: str(path.resolve()) for path in outputs},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(outputs[0])


if __name__ == "__main__":
    main()
