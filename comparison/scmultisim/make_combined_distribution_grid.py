#!/usr/bin/env python3
"""The four-method grid stacked above the T-architecture grid, with one shared key.

Top block  : OT / COATI bal / UOT / COATI unbal  (the sync ablation).
Bottom block: MLP / temporal MLP / ridge / temporal ridge  (COATI bal at Cy=0.1,
              identical budget, only the frozen map T differs).

Both blocks are re-projected here with the same inverse-distance 15-NN onto the
shared observed embedding.  The two source caches disagree on this: the
four-method cache was embedded with the fitted UMAP reducer's `transform`, the
T-architecture cache with 15-NN, because that joblib was written under a newer
Python and no longer unpickles in any environment here.  Mixing the two in one
figure would put visually different projections in the same column layout, so
everything is recomputed from the cached RNA10 predictions instead.

Correct rates are read from the two metrics files unchanged -- they are computed
in raw RNA10 and do not depend on the embedding.
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
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from sklearn.neighbors import NearestNeighbors

HERE = (_REPO / "external/COATI/Synthetic/UnbalancedSyncSweep")
SYNTHETIC = HERE.parent

DATA = SYNTHETIC / "5scRNA/all_time_scRNA_pca10.npz"
LABELS = SYNTHETIC / "5scRNA/all_time_scRNA_label.npz"
UMAP_CACHE = Path(
    "results/"
    "cytobridge_synthetic_rna10_n128_i3000_comparison/rna10_umap/"
    "observed_rna10_umap.npz"
)
FOUR = HERE / "outputs/bio_all1_alpha100_d10_e1/observed_time_distribution"
ARCH = HERE / "outputs/t_architecture_distribution"

BLOCKS = (
    {
        "cache": FOUR / "all_initial_observed_time_predictions.npz",
        "metrics": FOUR / "observed_time_distribution_metrics.json",
        "keys": ("ot", "coati_bal", "uot", "coati_unbal"),
        "labels": ("OT", "COATI bal", "UOT", "COATI unbal"),
    },
    {
        "cache": ARCH / "all_initial_observed_time_predictions.npz",
        "metrics": ARCH / "observed_time_distribution_metrics.json",
        "keys": ("mlp", "tmlp", "ridge", "tridge"),
        "labels": ("COATI + MLP", "COATI + temporal MLP",
                   "COATI + ridge", "COATI + temporal ridge"),
    },
)
TIME_NAMES = ("time2", "time3", "time4")
# Colours come from the shared TraInfBench standard rather than being fixed
# here, so this figure matches the method-comparison strips instead of keeping
# the older hardcoded Paul Tol pair.
_STYLE_PATH = Path(
    "common/trainfbench_plot_style.py"
)
_style_spec = importlib.util.spec_from_file_location(
    "trainfbench_plot_style", _STYLE_PATH
)
_style = importlib.util.module_from_spec(_style_spec)
sys.modules["trainfbench_plot_style"] = _style
_style_spec.loader.exec_module(_style)
SOURCE_COLORS = {
    "4_1": _style.synthetic_population_color("4_1"),
    "4_5": _style.synthetic_population_color("4_5"),
}
OBSERVED_GREY = "#B3B3B3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width_mm", type=float, default=207.6)
    parser.add_argument("--height_mm", type=float, default=252.0)
    parser.add_argument("--font_size", type=float, default=12.0)
    parser.add_argument("--left", type=float, default=0.042,
                        help="Left margin as a figure fraction; it only has to\n                             hold the rotated row label.")
    parser.add_argument("--right", type=float, default=0.988)
    parser.add_argument("--out", type=Path,
                        default=HERE / "outputs/combined_distribution_grid/"
                                       "combined_distribution_grid")
    args = parser.parse_args()

    observed = {f"time{i}": np.asarray(np.load(DATA)[f"time{i}"], dtype=np.float32)
                for i in range(1, 5)}
    observed_all = np.concatenate(list(observed.values()))
    labels = np.load(LABELS, allow_pickle=True)
    initial_source = labels["time1"].astype(str)

    cache = np.load(UMAP_CACHE, allow_pickle=True)
    embedding = np.asarray(cache["observed_umap"], dtype=np.float32)
    expected = np.concatenate([labels[f"time{i}"] for i in range(1, 5)]).astype(str)
    if len(embedding) != len(observed_all) or not (
        cache["observed_population"].astype(str) == expected
    ).all():
        raise ValueError("The cached observed UMAP is not the time1..time4 concatenation")
    offsets = np.cumsum([0] + [len(observed[f"time{i}"]) for i in range(1, 5)])
    observed_umap = {f"time{i}": embedding[offsets[i - 1]:offsets[i]] for i in range(1, 5)}

    # One projector for every panel in the figure.
    projector = NearestNeighbors(n_neighbors=15, n_jobs=-1).fit(observed_all)

    def project(points: np.ndarray) -> np.ndarray:
        distances, indices = projector.kneighbors(points)
        inverse = 1.0 / np.maximum(distances, 1e-6)
        inverse /= inverse.sum(axis=1, keepdims=True)
        return np.sum(embedding[indices] * inverse[:, :, None], axis=1)

    for block in BLOCKS:
        for path in (block["cache"], block["metrics"]):
            if not path.is_file():
                raise FileNotFoundError(path)
        arrays = np.load(block["cache"])
        block["metric_data"] = json.loads(block["metrics"].read_text())["models"]
        block["umap"] = {
            (key, time): project(np.asarray(arrays[f"{key}_{time}_rna"], dtype=np.float32))
            for key in block["keys"] for time in TIME_NAMES
        }

    font_manager.findfont("Arial", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Arial", "font.size": args.font_size, "font.weight": "normal",
        "axes.titlesize": args.font_size, "axes.labelsize": args.font_size,
        "xtick.labelsize": args.font_size, "ytick.labelsize": args.font_size,
        "legend.fontsize": args.font_size,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "custom", "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    })
    figure = plt.figure(figsize=(args.width_mm / 25.4, args.height_mm / 25.4))
    figure.patch.set_facecolor("white")

    # Two grids rather than one 6-row grid: the blocks need a visible gap and
    # their own column headings, and each block keeps its own row limits.
    band = (0.955 - 0.075 - 0.055) / 2.0
    grids = [
        figure.add_gridspec(3, 4, left=args.left, right=args.right,
                            top=0.955, bottom=0.955 - band,
                            wspace=0.035, hspace=0.13),
        figure.add_gridspec(3, 4, left=args.left, right=args.right,
                            top=0.955 - band - 0.055, bottom=0.075,
                            wspace=0.035, hspace=0.13),
    ]

    rng = np.random.default_rng(0)
    order = rng.permutation(len(initial_source))
    point_colors = np.asarray([SOURCE_COLORS[label] for label in initial_source])

    for block, grid in zip(BLOCKS, grids):
        for row_index, time_name in enumerate(TIME_NAMES):
            background = observed_umap[time_name]
            row_points = np.concatenate(
                [background] + [block["umap"][(k, time_name)] for k in block["keys"]]
            )
            lower, upper = row_points.min(axis=0), row_points.max(axis=0)
            if row_index > 0:
                # Sparse upper-tail UMAP outliers otherwise create empty bands.
                upper[1] = np.quantile(row_points[:, 1], 0.995)
            padding = np.maximum((upper - lower) * 0.035, 0.25)
            plot_lower, plot_upper = lower - padding, upper + padding
            plot_lower[1] -= 0.10 * (plot_upper[1] - plot_lower[1])

            for column_index, (key, label) in enumerate(
                zip(block["keys"], block["labels"])
            ):
                axis = figure.add_subplot(grid[row_index, column_index])
                axis.scatter(background[:, 0], background[:, 1], s=4,
                             color=OBSERVED_GREY, alpha=0.18, edgecolors="none",
                             rasterized=True)
                predicted = block["umap"][(key, time_name)]
                axis.scatter(predicted[order, 0], predicted[order, 1], s=11,
                             c=point_colors[order], alpha=0.46, edgecolors="none",
                             rasterized=True)
                rate = block["metric_data"][key]["times"][time_name]["correct_rate"]
                axis.text(0.03, 0.015, f"Correct rate = {rate:.1%}",
                          transform=axis.transAxes, ha="left", va="bottom")
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_xlim(plot_lower[0], plot_upper[0])
                axis.set_ylim(plot_lower[1], plot_upper[1])
                if row_index == 0:
                    axis.set_title(label, pad=8)
                if column_index == 0:
                    axis.set_ylabel(time_name)

    figure.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", markersize=7,
                   color=SOURCE_COLORS["4_1"], label="4_1 branch"),
            Line2D([0], [0], marker="o", linestyle="", markersize=7,
                   color=SOURCE_COLORS["4_5"], label="4_5 branch"),
        ],
        loc="lower center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 0.008), columnspacing=2.4, handletextpad=0.4,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out.with_suffix(".pdf"), facecolor="white")
    figure.savefig(args.out.with_suffix(".png"), dpi=400, facecolor="white")
    plt.close(figure)
    print(f"wrote {args.out}.pdf/.png")
    print(f"page {args.width_mm} x {args.height_mm} mm, all text {args.font_size} pt")
    for block in BLOCKS:
        rates = [
            f"{label} {100 * block['metric_data'][key]['times']['time4']['correct_rate']:.1f}%"
            for key, label in zip(block["keys"], block["labels"])
        ]
        print("  time4:", ", ".join(rates))


if __name__ == "__main__":
    main()
