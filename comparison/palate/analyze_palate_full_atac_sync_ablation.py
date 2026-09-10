#!/usr/bin/env python
"""Compare full-data COATI with ATAC-only and RNA-only trajectories.

All six methods start from the same E12.5 paired cells.  The analysis keeps
only cells annotated as CNC-derived progenitors at E12.5, assigns terminal
anterior/posterior fate with one fixed E14.5 ATAC kNN classifier, and decodes
fixed external H3K27ac and TF-bound CRE programs using the same stage-specific
ATAC kNN readout for every method.

The peak sets are fixed independently of method output:

* anterior/posterior H3K27ac-specific peaks;
* SHOX2-bound anterior-active and MEOX2-bound posterior-active CREs.

This is a full-train synchronization ablation, not a held-out interpolation
benchmark.  It asks whether synchronization attaches the correct regulatory
identity to a developmental trajectory.  RNA-only trajectories are mapped to
ATAC with the same frozen full-data FiLM map used elsewhere in the benchmark.
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
import json
import os
from dataclasses import dataclass
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
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import torch

from analyze_palate_shox2_1nn_regulation import (
    ANTERIOR,
    ATAC_NORM,
    ATAC_RAW,
    ATAC_REFERENCE,
    CNC,
    POSTERIOR,
    RNA_NORM,
    RNA_REFERENCE,
)
from evaluate_palate_loo_same_space import _load_film, _load_scale, _map_to_atac
from trainfbench_plot_style import NATURE_CUD, apply_nature_rc


ROOT = Path(__file__).resolve().parents[2]
PALATE = Path("external/COATI/MouseBrain")
DEFAULT_OUTPUT = ROOT / "results/palate_full_atac_sync_ablation_with_rna_only"
PEAK_ANNOTATION = (
    ROOT
    / "results/palate_strict_loo_external_epigenetic_evidence"
    / "external_peak_annotations_with_meox2.csv.gz"
)
STAGES = ("E12.5", "E13.5", "E14.0", "E14.5")
TIMES = (0.0, 1.0, 1.5, 2.0)
TRAJECTORY_INDICES = (0, 10, 15, 20)


@dataclass(frozen=True)
class Method:
    name: str
    transport: str
    atac_path: Path | None
    mass_path: Path | None = None
    rna_path: Path | None = None


METHODS = (
    Method(
        "COATI-B",
        "balanced",
        PALATE
        / "FilmBalanced/trajectory/iter20000/secondary_trajectory_a0.5_iter20000.pt",
        rna_path=PALATE
        / "FilmBalanced/trajectory/iter20000/primary_trajectory_a0.5_iter20000.pt",
    ),
    Method(
        "OT(ATAC)",
        "balanced",
        PALATE
        / "BalancedATAC/trajectory/secondary_trajectory_balanced_ataconly_s0_iter20000.pt",
    ),
    Method(
        "OT(RNA)",
        "balanced",
        None,
        rna_path=PALATE
        / "Balanced/trajectory/primary_trajectory_balanced_rnaonly_s0_iter20000.pt",
    ),
    Method(
        "COATI-U",
        "unbalanced",
        PALATE
        / "FilmUnbalancedSync/trajectory/secondary_trajectory_filmunbalanced_s0_a0.5_iter20000.pt",
        mass_path=PALATE
        / "FilmUnbalancedSync/trajectory/mass_lnw_filmunbalanced_s0_a0.5_iter20000.pt",
        rna_path=PALATE
        / "FilmUnbalancedSync/trajectory/primary_trajectory_filmunbalanced_s0_a0.5_iter20000.pt",
    ),
    Method(
        "UOT(ATAC)",
        "unbalanced",
        PALATE
        / "UnbalancedATACOnly/trajectory/primary_trajectory_unbalanced_ataconly_s0_iter20000.pt",
        mass_path=PALATE
        / "UnbalancedATACOnly/trajectory/mass_lnw_trajectory_unbalanced_ataconly_s0_iter20000.pt",
    ),
    Method(
        "UOT(RNA)",
        "unbalanced",
        None,
        mass_path=PALATE
        / "UnbalancedRNAOnly/trajectory/mass_lnw_trajectory_unbalanced_rnaonly_s0_iter20000.pt",
        rna_path=PALATE
        / "UnbalancedRNAOnly/trajectory/primary_trajectory_unbalanced_rnaonly_s0_iter20000.pt",
    ),
)

PEAK_FAMILIES = {
    "H3K27ac": ("h3_anterior_specific", "h3_posterior_specific"),
    "TF-bound CRE": (
        "shox2_bound_anterior_active",
        "meox2_bound_posterior_active",
    ),
}

METHOD_COLORS = {
    "COATI-B": NATURE_CUD["blue"],
    "OT(ATAC)": NATURE_CUD["sky_blue"],
    "OT(RNA)": "#999999",
    "COATI-U": NATURE_CUD["vermillion"],
    "UOT(ATAC)": NATURE_CUD["black"],
    "UOT(RNA)": "#555555",
}
PLOT_LABELS = {
    "COATI-B": "COATI-B",
    "OT(ATAC)": "OT(ATAC)",
    "OT(RNA)": "OT(RNA)",
    "COATI-U": "COATI-U",
    "UOT(ATAC)": "UOT(ATAC)",
    "UOT(RNA)": "UOT(RNA)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_tensor(path: Path) -> np.ndarray:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 3 or result.shape[0] < 21 or result.shape[1] != 2570:
        raise ValueError(f"Unexpected trajectory shape at {path}: {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite trajectory at {path}")
    return result


def map_rna_trajectory(
    model: torch.nn.Module,
    trajectory: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    time_grid = np.linspace(0.0, 2.0, trajectory.shape[0])
    return np.stack(
        [
            _map_to_atac(model, state, float(time), device, batch_size)
            for state, time in zip(trajectory, time_grid)
        ]
    ).astype(np.float32, copy=False)


def normalized_weights(log_mass: np.ndarray | None, mask: np.ndarray) -> np.ndarray:
    n = int(mask.sum())
    if log_mass is None:
        return np.full(n, 1.0 / n)
    values = np.asarray(log_mass, dtype=float).reshape(-1)[mask]
    values -= np.max(values)
    weights = np.exp(np.clip(values, -80.0, 0.0))
    return weights / weights.sum()


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights)
    if not finite.any() or weights[finite].sum() <= 0:
        return float("nan")
    return float(np.average(values[finite], weights=weights[finite]))


def bootstrap_weighted_mean_interval(
    rng: np.random.Generator,
    values: np.ndarray,
    weights: np.ndarray,
    n_bootstrap: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[finite]
    weights = weights[finite]
    if len(values) == 0 or weights.sum() <= 0:
        return float("nan"), float("nan")
    probability = weights / weights.sum()
    samples = np.empty(n_bootstrap, dtype=float)
    for bootstrap_index in range(n_bootstrap):
        index = rng.choice(
            len(values), size=len(values), replace=True, p=probability
        )
        samples[bootstrap_index] = float(np.mean(values[index]))
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def fixed_fate_classifier(
    reference: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    k: int,
) -> tuple[NearestNeighbors, np.ndarray]:
    mask = np.isclose(times, 2.0) & np.isin(labels, [ANTERIOR, POSTERIOR])
    rows = np.flatnonzero(mask)
    model = NearestNeighbors(n_neighbors=min(k, len(rows)), n_jobs=-1).fit(reference[rows])
    return model, labels[rows]


def predict_fate(
    model: NearestNeighbors,
    training_labels: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    neighbours = model.kneighbors(values, return_distance=False)
    neighbour_labels = training_labels[neighbours]
    anterior_fraction = np.mean(neighbour_labels == ANTERIOR, axis=1)
    fate = np.where(anterior_fraction >= 0.5, "anterior", "posterior")
    confidence = np.maximum(anterior_fraction, 1.0 - anterior_fraction)
    return fate.astype(str), confidence


def cell_program_score(
    raw: ad.AnnData,
    positions: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    block = raw.X[:, np.asarray(positions, dtype=int)]
    if sparse.issparse(block):
        block = block.copy()
        block.data = np.ones_like(block.data)
        counts = np.asarray(block.sum(axis=1)).ravel()
    else:
        counts = np.sum(np.asarray(block) > 0, axis=1)
    rate = counts / np.maximum(depth, 1.0)
    transformed = np.log1p(rate * 1e5)
    return ((transformed - transformed.mean()) / max(transformed.std(), 1e-8)).astype(
        np.float32
    )


def decode_programs(
    predicted_atac: np.ndarray,
    atlas_atac: np.ndarray,
    atlas_times: np.ndarray,
    atlas_programs: dict[str, dict[str, np.ndarray]],
    k: int,
) -> dict[str, dict[str, np.ndarray]]:
    result = {
        family: {
            branch: np.empty((len(STAGES), predicted_atac.shape[1]), dtype=np.float32)
            for branch in ("anterior", "posterior")
        }
        for family in atlas_programs
    }
    for stage_index, (time, trajectory_index) in enumerate(
        zip(TIMES, TRAJECTORY_INDICES)
    ):
        rows = np.flatnonzero(np.isclose(atlas_times, time))
        nn = NearestNeighbors(n_neighbors=min(k, len(rows)), n_jobs=-1).fit(
            atlas_atac[rows]
        )
        local = nn.kneighbors(
            predicted_atac[trajectory_index], return_distance=False
        )
        neighbours = rows[local]
        for family, branch_scores in atlas_programs.items():
            for branch, values in branch_scores.items():
                result[family][branch][stage_index] = values[neighbours].mean(axis=1)
    return result


def bootstrap_interval(
    rng: np.random.Generator,
    values: dict[str, np.ndarray],
    weights: np.ndarray,
    n_bootstrap: int,
) -> dict[str, tuple[float, float]]:
    n = len(weights)
    probability = weights / weights.sum()
    samples: dict[str, list[float]] = {
        "terminal_margin": [],
        "margin_gain": [],
        "monotonic_rate": [],
        "initial_fate_auroc": [],
        "early_fate_auroc": [],
        "e140_fate_auroc": [],
    }
    fate_binary = values["fate_binary"].astype(int)
    for _ in range(n_bootstrap):
        index = rng.choice(n, size=n, replace=True, p=probability)
        margin = values["margin"][:, index]
        samples["terminal_margin"].append(float(np.mean(margin[-1])))
        samples["margin_gain"].append(float(np.mean(margin[-1] - margin[0])))
        samples["monotonic_rate"].append(float(np.mean(np.diff(margin, axis=0) >= 0)))
        local_y = fate_binary[index]
        if len(np.unique(local_y)) == 2:
            samples["initial_fate_auroc"].append(
                float(roc_auc_score(local_y, values["initial_score"][index]))
            )
            samples["early_fate_auroc"].append(
                float(roc_auc_score(local_y, values["early_score"][index]))
            )
            samples["e140_fate_auroc"].append(
                float(roc_auc_score(local_y, values["e140_score"][index]))
            )
    return {
        key: tuple(np.quantile(sample, [0.025, 0.975]))
        for key, sample in samples.items()
        if sample
    }


def save_plot(metrics: pd.DataFrame, output_dir: Path) -> None:
    apply_nature_rc(font_size=10)
    mpl.rcParams.update(
        {
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "figure.titleweight": "normal",
        }
    )
    panels = (
        ("H3K27ac", "terminal_margin", "H3K27ac margin ↑"),
        ("H3K27ac", "early_fate_auroc", "H3K27ac fate AUROC ↑"),
        ("TF-bound CRE", "terminal_margin", "TF CRE margin ↑"),
        ("TF-bound CRE", "early_fate_auroc", "TF CRE fate AUROC ↑"),
    )
    methods = [method.name for method in METHODS]
    fig, axes = plt.subplots(2, 2, figsize=(4.13, 3.25))
    for panel_index, (axis, (family, metric, title)) in enumerate(
        zip(axes.ravel(), panels)
    ):
        local = metrics[metrics["family"].eq(family)].set_index("method")
        x = np.arange(len(methods))
        values = np.asarray([local.loc[method, metric] for method in methods])
        lower = np.asarray([local.loc[method, f"{metric}_ci_low"] for method in methods])
        upper = np.asarray([local.loc[method, f"{metric}_ci_high"] for method in methods])
        errors = np.vstack([values - lower, upper - values])
        axis.bar(
            x,
            values,
            color=[METHOD_COLORS[method] for method in methods],
            width=0.72,
            yerr=errors,
            error_kw={"lw": 0.7, "capthick": 0.7, "capsize": 2},
        )
        axis.axhline(0.5 if metric == "early_fate_auroc" else 0.0, color="#777777", lw=0.7, ls="--")
        axis.set_title(title, fontsize=10, pad=3, weight="normal")
        if panel_index < 2:
            axis.set_xticks(x, [])
        else:
            axis.set_xticks(
                x,
                [PLOT_LABELS[method] for method in methods],
                rotation=43,
                ha="right",
            )
            axis.tick_params(axis="x", labelsize=8.0, pad=1)
        axis.spines[["top", "right"]].set_visible(False)
        axis.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
    fig.subplots_adjust(
        left=0.14, right=0.98, top=0.96, bottom=0.25, hspace=0.36, wspace=0.36
    )
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 450} if suffix == "png" else {}
        fig.savefig(
            output_dir / f"palate_full_atac_sync_ablation_with_rna_only.{suffix}",
            bbox_inches="tight",
            pad_inches=0.02,
            **kwargs,
        )
    plt.close(fig)


def save_stage_auroc_plot(
    metrics: pd.DataFrame,
    output_dir: Path,
    *,
    stage: str,
    metric: str,
    stem: str,
) -> None:
    apply_nature_rc(font_size=10)
    mpl.rcParams.update(
        {
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "figure.titleweight": "normal",
        }
    )
    methods = [method.name for method in METHODS]
    fig, axes = plt.subplots(1, 2, figsize=(4.13, 2.05), sharey=True)
    for axis, family in zip(axes, PEAK_FAMILIES):
        local = metrics[metrics["family"].eq(family)].set_index("method")
        x = np.arange(len(methods))
        values = np.asarray([local.loc[method, metric] for method in methods])
        lower = np.asarray(
            [local.loc[method, f"{metric}_ci_low"] for method in methods]
        )
        upper = np.asarray(
            [local.loc[method, f"{metric}_ci_high"] for method in methods]
        )
        axis.bar(
            x,
            values,
            color=[METHOD_COLORS[method] for method in methods],
            width=0.72,
            yerr=np.vstack([values - lower, upper - values]),
            error_kw={"lw": 0.7, "capthick": 0.7, "capsize": 2},
        )
        axis.axhline(0.5, color="#777777", lw=0.7, ls="--")
        display_family = "TF-CRE" if family == "TF-bound CRE" else family
        axis.set_title(
            f"{stage} {display_family}\nfate AUROC ↑",
            fontsize=10,
            pad=3,
            linespacing=0.95,
        )
        axis.set_xticks(
            x,
            [PLOT_LABELS[method] for method in methods],
            rotation=43,
            ha="right",
        )
        axis.tick_params(axis="x", labelsize=8.0, pad=1)
        axis.set_ylim(0.45, 1.0)
        axis.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
        axis.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(
        left=0.14, right=0.98, top=0.84, bottom=0.36, wspace=0.30
    )
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 450} if suffix == "png" else {}
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.02,
            **kwargs,
        )
    plt.close(fig)


def save_temporal_auroc_plot(metrics: pd.DataFrame, output_dir: Path) -> None:
    apply_nature_rc(font_size=10)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "figure.titleweight": "normal",
        }
    )
    stage_labels = ("E12.5", "E13.5", "E14.0")
    metric_names = (
        "initial_fate_auroc",
        "early_fate_auroc",
        "e140_fate_auroc",
    )
    line_styles = {
        "COATI-B": "-",
        "COATI-U": "-",
        "OT(ATAC)": ":",
        "UOT(ATAC)": ":",
        "OT(RNA)": "--",
        "UOT(RNA)": "--",
    }
    markers = {
        "COATI-B": "o",
        "OT(ATAC)": "s",
        "OT(RNA)": "^",
        "COATI-U": "D",
        "UOT(ATAC)": "P",
        "UOT(RNA)": "v",
    }
    x = np.arange(len(stage_labels), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(4.13, 2.42), sharex=True, sharey=True)
    for axis, family in zip(axes, PEAK_FAMILIES):
        local = metrics[metrics["family"].eq(family)].set_index("method")
        for method in [item.name for item in METHODS]:
            values = np.asarray(
                [float(local.loc[method, metric]) for metric in metric_names]
            )
            lower = np.asarray(
                [float(local.loc[method, f"{metric}_ci_low"]) for metric in metric_names]
            )
            upper = np.asarray(
                [float(local.loc[method, f"{metric}_ci_high"]) for metric in metric_names]
            )
            axis.errorbar(
                x,
                values,
                yerr=np.vstack([values - lower, upper - values]),
                color=METHOD_COLORS[method],
                linestyle=line_styles[method],
                marker=markers[method],
                markersize=3.2,
                markeredgewidth=0.55,
                linewidth=1.1,
                elinewidth=0.55,
                capsize=1.5,
                label=PLOT_LABELS[method],
            )
        axis.axhline(0.5, color="#888888", lw=0.65, ls="--", zorder=0)
        display_family = "TF-CRE" if family == "TF-bound CRE" else family
        axis.set_title(display_family, fontsize=10, pad=3, weight="normal")
        axis.set_xticks(x, stage_labels)
        axis.set_xlim(-0.12, 2.12)
        axis.set_ylim(0.48, 0.93)
        axis.set_yticks((0.5, 0.6, 0.7, 0.8, 0.9))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E5E5E5", lw=0.45, zorder=0)
    axes[0].set_ylabel("Fate AUROC")
    fig.supxlabel("Embryonic stage", y=0.035, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        columnspacing=0.75,
        labelspacing=0.25,
        handlelength=1.5,
        handletextpad=0.35,
        fontsize=8.2,
    )
    fig.subplots_adjust(
        left=0.14,
        right=0.985,
        top=0.76,
        bottom=0.20,
        wspace=0.24,
    )
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 450} if suffix == "png" else {}
        fig.savefig(
            output_dir / f"palate_full_atac_fate_auroc_temporal_curves.{suffix}",
            bbox_inches="tight",
            pad_inches=0.02,
            **kwargs,
        )
    plt.close(fig)


def save_split_temporal_auroc_plot(metrics: pd.DataFrame, output_dir: Path) -> None:
    apply_nature_rc(font_size=10)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "figure.titleweight": "normal",
        }
    )
    stage_labels = ("E12.5", "E13.5", "E14.0")
    metric_names = (
        "initial_fate_auroc",
        "early_fate_auroc",
        "e140_fate_auroc",
    )
    rows = (
        ("Balanced", ("COATI-B", "OT(ATAC)", "OT(RNA)")),
        ("Unbalanced", ("COATI-U", "UOT(ATAC)", "UOT(RNA)")),
    )
    line_styles = {
        "COATI-B": "-",
        "COATI-U": "-",
        "OT(ATAC)": ":",
        "UOT(ATAC)": ":",
        "OT(RNA)": "--",
        "UOT(RNA)": "--",
    }
    markers = {
        "COATI-B": "o",
        "OT(ATAC)": "s",
        "OT(RNA)": "^",
        "COATI-U": "D",
        "UOT(ATAC)": "P",
        "UOT(RNA)": "v",
    }
    x = np.arange(len(stage_labels), dtype=float)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(4.13, 3.30),
        sharex=True,
        sharey=True,
    )
    for row_index, (row_label, methods) in enumerate(rows):
        for column_index, family in enumerate(PEAK_FAMILIES):
            axis = axes[row_index, column_index]
            local = metrics[metrics["family"].eq(family)].set_index("method")
            for method in methods:
                values = np.asarray(
                    [float(local.loc[method, metric]) for metric in metric_names]
                )
                lower = np.asarray(
                    [
                        float(local.loc[method, f"{metric}_ci_low"])
                        for metric in metric_names
                    ]
                )
                upper = np.asarray(
                    [
                        float(local.loc[method, f"{metric}_ci_high"])
                        for metric in metric_names
                    ]
                )
                axis.errorbar(
                    x,
                    values,
                    yerr=np.vstack([values - lower, upper - values]),
                    color=METHOD_COLORS[method],
                    linestyle=line_styles[method],
                    marker=markers[method],
                    markersize=3.2,
                    markeredgewidth=0.55,
                    linewidth=1.1,
                    elinewidth=0.55,
                    capsize=1.5,
                    label=PLOT_LABELS[method],
                )
            axis.axhline(0.5, color="#888888", lw=0.65, ls="--", zorder=0)
            display_family = "TF-CRE" if family == "TF-bound CRE" else family
            axis.set_title(
                f"{row_label} · {display_family}",
                fontsize=10,
                pad=3,
                weight="normal",
            )
            axis.set_xlim(-0.12, 2.12)
            axis.set_ylim(0.48, 0.93)
            axis.set_yticks((0.5, 0.6, 0.7, 0.8, 0.9))
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#E5E5E5", lw=0.45, zorder=0)
            if row_index == 1:
                axis.set_xticks(x, stage_labels)
            else:
                axis.tick_params(axis="x", labelbottom=False)
            if column_index == 1:
                axis.legend(
                    frameon=False,
                    loc="upper left",
                    bbox_to_anchor=(0.01, 0.99),
                    borderaxespad=0,
                    handlelength=1.5,
                    handletextpad=0.35,
                    labelspacing=0.20,
                    fontsize=7.8,
                )
    fig.supylabel("Fate AUROC", x=0.02, fontsize=10)
    fig.supxlabel("Embryonic stage", y=0.025, fontsize=10)
    fig.subplots_adjust(
        left=0.15,
        right=0.985,
        top=0.96,
        bottom=0.17,
        hspace=0.42,
        wspace=0.24,
    )
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 450} if suffix == "png" else {}
        fig.savefig(
            output_dir
            / f"palate_full_atac_fate_auroc_temporal_balanced_unbalanced.{suffix}",
            bbox_inches="tight",
            pad_inches=0.02,
            **kwargs,
        )
    plt.close(fig)


def save_split_temporal_margin_plot(curves: pd.DataFrame, output_dir: Path) -> None:
    apply_nature_rc(font_size=10)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "figure.titleweight": "normal",
        }
    )
    rows = (
        ("Balanced", ("COATI-B", "OT(ATAC)", "OT(RNA)")),
        ("Unbalanced", ("COATI-U", "UOT(ATAC)", "UOT(RNA)")),
    )
    line_styles = {
        "COATI-B": "-",
        "COATI-U": "-",
        "OT(ATAC)": ":",
        "UOT(ATAC)": ":",
        "OT(RNA)": "--",
        "UOT(RNA)": "--",
    }
    markers = {
        "COATI-B": "o",
        "OT(ATAC)": "s",
        "OT(RNA)": "^",
        "COATI-U": "D",
        "UOT(ATAC)": "P",
        "UOT(RNA)": "v",
    }
    y_limits = {
        "H3K27ac": (0.0, 0.76),
        "TF-bound CRE": (0.0, 0.17),
    }
    y_ticks = {
        "H3K27ac": (0.0, 0.2, 0.4, 0.6),
        "TF-bound CRE": (0.0, 0.05, 0.10, 0.15),
    }
    x = np.arange(len(STAGES), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(4.13, 3.30), sharex=True)
    for row_index, (row_label, methods) in enumerate(rows):
        for column_index, family in enumerate(PEAK_FAMILIES):
            axis = axes[row_index, column_index]
            local = curves[curves["family"].eq(family)]
            for method in methods:
                method_rows = local[local["method"].eq(method)].set_index("stage")
                values = np.asarray(
                    [float(method_rows.loc[stage, "regulatory_margin"]) for stage in STAGES]
                )
                lower = np.asarray(
                    [
                        float(method_rows.loc[stage, "regulatory_margin_ci_low"])
                        for stage in STAGES
                    ]
                )
                upper = np.asarray(
                    [
                        float(method_rows.loc[stage, "regulatory_margin_ci_high"])
                        for stage in STAGES
                    ]
                )
                axis.errorbar(
                    x,
                    values,
                    yerr=np.vstack([values - lower, upper - values]),
                    color=METHOD_COLORS[method],
                    linestyle=line_styles[method],
                    marker=markers[method],
                    markersize=3.2,
                    markeredgewidth=0.55,
                    linewidth=1.1,
                    elinewidth=0.55,
                    capsize=1.5,
                    label=PLOT_LABELS[method],
                )
            axis.axhline(0.0, color="#888888", lw=0.65, ls="--", zorder=0)
            display_family = "TF-CRE" if family == "TF-bound CRE" else family
            axis.set_title(
                f"{row_label} · {display_family}",
                fontsize=10,
                pad=3,
                weight="normal",
            )
            axis.set_xlim(-0.14, len(STAGES) - 1 + 0.14)
            axis.set_ylim(*y_limits[family])
            axis.set_yticks(y_ticks[family])
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#E5E5E5", lw=0.45, zorder=0)
            if row_index == 1:
                axis.set_xticks(x, STAGES)
            else:
                axis.tick_params(axis="x", labelbottom=False)
            if column_index == 1:
                axis.legend(
                    frameon=False,
                    loc="upper left",
                    bbox_to_anchor=(0.01, 0.99),
                    borderaxespad=0,
                    handlelength=1.5,
                    handletextpad=0.35,
                    labelspacing=0.20,
                    fontsize=7.8,
                )
    fig.supylabel("Regulatory margin", x=0.02, fontsize=10)
    fig.supxlabel("Embryonic stage", y=0.025, fontsize=10)
    fig.subplots_adjust(
        left=0.15,
        right=0.985,
        top=0.96,
        bottom=0.17,
        hspace=0.42,
        wspace=0.27,
    )
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 450} if suffix == "png" else {}
        fig.savefig(
            output_dir
            / f"palate_full_atac_regulatory_margin_temporal_balanced_unbalanced.{suffix}",
            bbox_inches="tight",
            pad_inches=0.02,
            **kwargs,
        )
    plt.close(fig)


def save_h3_fate_and_separation_plot(
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    output_dir: Path,
) -> None:
    apply_nature_rc(font_size=10)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "figure.titleweight": "normal",
        }
    )
    methods = [item.name for item in METHODS]
    line_styles = {
        "COATI-B": "-",
        "COATI-U": "-",
        "OT(ATAC)": ":",
        "UOT(ATAC)": ":",
        "OT(RNA)": "--",
        "UOT(RNA)": "--",
    }
    markers = {
        "COATI-B": "o",
        "OT(ATAC)": "s",
        "OT(RNA)": "^",
        "COATI-U": "D",
        "UOT(ATAC)": "P",
        "UOT(RNA)": "v",
    }
    fig, axes = plt.subplots(1, 2, figsize=(4.13, 2.42))

    fate_axis = axes[0]
    fate_local = metrics[metrics["family"].eq("H3K27ac")].set_index("method")
    fate_metrics = (
        "initial_fate_auroc",
        "early_fate_auroc",
        "e140_fate_auroc",
    )
    fate_x = np.arange(len(fate_metrics), dtype=float)
    for method in methods:
        values = np.asarray(
            [float(fate_local.loc[method, metric]) for metric in fate_metrics]
        )
        lower = np.asarray(
            [float(fate_local.loc[method, f"{metric}_ci_low"]) for metric in fate_metrics]
        )
        upper = np.asarray(
            [float(fate_local.loc[method, f"{metric}_ci_high"]) for metric in fate_metrics]
        )
        fate_axis.errorbar(
            fate_x,
            values,
            yerr=np.vstack([values - lower, upper - values]),
            color=METHOD_COLORS[method],
            linestyle=line_styles[method],
            marker=markers[method],
            markersize=3.2,
            markeredgewidth=0.55,
            linewidth=1.1,
            elinewidth=0.55,
            capsize=1.5,
            label=PLOT_LABELS[method],
        )
    fate_axis.axhline(0.5, color="#888888", lw=0.65, ls="--", zorder=0)
    fate_axis.set_title("H3K27ac fate AUROC", fontsize=10, pad=3, weight="normal")
    fate_axis.set_ylabel("Fate AUROC")
    fate_axis.set_xticks(fate_x, ("E12.5", "E13.5", "E14.0"))
    fate_axis.set_xlim(-0.12, 2.12)
    fate_axis.set_ylim(0.48, 0.93)
    fate_axis.set_yticks((0.5, 0.6, 0.7, 0.8, 0.9))

    margin_axis = axes[1]
    margin_local = curves[curves["family"].eq("H3K27ac")]
    margin_x = np.arange(len(STAGES), dtype=float)
    for method in methods:
        method_rows = margin_local[margin_local["method"].eq(method)].set_index(
            "stage"
        )
        values = np.asarray(
            [float(method_rows.loc[stage, "regulatory_margin"]) for stage in STAGES]
        )
        lower = np.asarray(
            [
                float(method_rows.loc[stage, "regulatory_margin_ci_low"])
                for stage in STAGES
            ]
        )
        upper = np.asarray(
            [
                float(method_rows.loc[stage, "regulatory_margin_ci_high"])
                for stage in STAGES
            ]
        )
        margin_axis.errorbar(
            margin_x,
            values,
            yerr=np.vstack([values - lower, upper - values]),
            color=METHOD_COLORS[method],
            linestyle=line_styles[method],
            marker=markers[method],
            markersize=3.2,
            markeredgewidth=0.55,
            linewidth=1.1,
            elinewidth=0.55,
            capsize=1.5,
        )
    margin_axis.axhline(0.0, color="#888888", lw=0.65, ls="--", zorder=0)
    margin_axis.set_title(
        "H3K27ac separation margin", fontsize=10, pad=3, weight="normal"
    )
    margin_axis.set_ylabel("Separation margin")
    margin_axis.set_xticks(margin_x, STAGES)
    margin_axis.set_xlim(-0.15, 3.15)
    margin_axis.set_ylim(0.0, 0.76)
    margin_axis.set_yticks((0.0, 0.2, 0.4, 0.6))

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E5E5E5", lw=0.45, zorder=0)
        axis.set_xlabel("Embryonic stage")
    handles, labels = fate_axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        columnspacing=0.75,
        labelspacing=0.22,
        handlelength=1.5,
        handletextpad=0.35,
        fontsize=8.0,
    )
    fig.subplots_adjust(
        left=0.14,
        right=0.985,
        top=0.76,
        bottom=0.19,
        wspace=0.36,
    )
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 450} if suffix == "png" else {}
        fig.savefig(
            output_dir / f"palate_full_h3k27ac_fate_and_separation.{suffix}",
            bbox_inches="tight",
            pad_inches=0.02,
            **kwargs,
        )
    plt.close(fig)


def save_h3_balanced_unbalanced_plots(
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot H3K27ac fate and separation with balanced/unbalanced panels split."""
    apply_nature_rc(font_size=10)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "figure.titleweight": "normal",
        }
    )
    groups = {
        "Balanced": ("COATI-B", "OT(ATAC)", "OT(RNA)"),
        "Unbalanced": ("COATI-U", "UOT(ATAC)", "UOT(RNA)"),
    }
    line_styles = {
        "COATI-B": "-",
        "COATI-U": "-",
        "OT(ATAC)": ":",
        "UOT(ATAC)": ":",
        "OT(RNA)": "--",
        "UOT(RNA)": "--",
    }
    markers = {
        "COATI-B": "o",
        "OT(ATAC)": "s",
        "OT(RNA)": "^",
        "COATI-U": "D",
        "UOT(ATAC)": "P",
        "UOT(RNA)": "v",
    }

    fate_local = metrics[metrics["family"].eq("H3K27ac")].set_index("method")
    fate_metrics = (
        "initial_fate_auroc",
        "early_fate_auroc",
        "e140_fate_auroc",
    )
    fate_x = np.asarray(TIMES[: len(fate_metrics)], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(4.13, 2.22), sharey=True)
    for axis, (group_name, methods) in zip(axes, groups.items()):
        for method in methods:
            values = np.asarray(
                [float(fate_local.loc[method, metric]) for metric in fate_metrics]
            )
            lower = np.asarray(
                [
                    float(fate_local.loc[method, f"{metric}_ci_low"])
                    for metric in fate_metrics
                ]
            )
            upper = np.asarray(
                [
                    float(fate_local.loc[method, f"{metric}_ci_high"])
                    for metric in fate_metrics
                ]
            )
            axis.errorbar(
                fate_x,
                values,
                yerr=np.vstack([values - lower, upper - values]),
                color=METHOD_COLORS[method],
                linestyle=line_styles[method],
                marker=markers[method],
                markersize=3.2,
                markeredgewidth=0.55,
                linewidth=1.1,
                elinewidth=0.55,
                capsize=1.5,
                label=PLOT_LABELS[method],
            )
        axis.axhline(0.5, color="#888888", lw=0.65, ls="--", zorder=0)
        axis.set_title(group_name, fontsize=10, pad=3, weight="normal")
        axis.set_xticks(fate_x, ("t1", "t2", "t3"))
        axis.set_xlim(-0.06, 1.56)
        axis.set_ylim(0.48, 0.93)
        axis.set_yticks((0.5, 0.6, 0.7, 0.8, 0.9))
        axis.set_xlabel("Time", labelpad=2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E5E5E5", lw=0.45, zorder=0)
        axis.legend(
            frameon=False,
            loc="lower right",
            handlelength=1.45,
            handletextpad=0.35,
            labelspacing=0.2,
            fontsize=7.8,
        )
    axes[0].set_ylabel("Fate AUROC")
    fig.subplots_adjust(left=0.14, right=0.985, top=0.94, bottom=0.20, wspace=0.18)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 450} if suffix == "png" else {}
        fig.savefig(
            output_dir
            / f"palate_full_h3k27ac_fate_balanced_unbalanced.{suffix}",
            bbox_inches="tight",
            pad_inches=0.02,
            **kwargs,
        )
    plt.close(fig)

    margin_local = curves[curves["family"].eq("H3K27ac")]
    # Use the actual developmental intervals: E12.5→E13.5 is twice as long
    # as either E13.5→E14.0 or E14.0→E14.5.
    margin_x = np.asarray(TIMES, dtype=float)
    time_labels = ("t1", "t2", "t3", "t4")
    fig, axes = plt.subplots(1, 2, figsize=(4.23, 2.25), sharey=True)
    for axis, (group_name, methods) in zip(axes, groups.items()):
        for method in methods:
            method_rows = margin_local[margin_local["method"].eq(method)].set_index(
                "stage"
            )
            values = np.asarray(
                [
                    float(method_rows.loc[stage, "regulatory_margin"])
                    for stage in STAGES
                ]
            )
            lower = np.asarray(
                [
                    float(method_rows.loc[stage, "regulatory_margin_ci_low"])
                    for stage in STAGES
                ]
            )
            upper = np.asarray(
                [
                    float(method_rows.loc[stage, "regulatory_margin_ci_high"])
                    for stage in STAGES
                ]
            )
            axis.errorbar(
                margin_x,
                values,
                yerr=np.vstack([values - lower, upper - values]),
                color=METHOD_COLORS[method],
                linestyle=line_styles[method],
                marker=markers[method],
                markersize=3.2,
                markeredgewidth=0.55,
                linewidth=1.1,
                elinewidth=0.55,
                capsize=1.5,
                label=PLOT_LABELS[method],
            )
        axis.axhline(0.0, color="#888888", lw=0.65, ls="--", zorder=0)
        axis.set_title(group_name, fontsize=10, pad=3, weight="normal")
        axis.set_xticks(margin_x, time_labels)
        axis.set_xlim(-0.08, 2.08)
        axis.set_ylim(0.0, 0.76)
        axis.set_yticks((0.0, 0.2, 0.4, 0.6))
        axis.set_xlabel("Time", labelpad=2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E5E5E5", lw=0.45, zorder=0)
        axis.legend(
            frameon=False,
            loc="lower right",
            handlelength=1.45,
            handletextpad=0.35,
            labelspacing=0.2,
            fontsize=10,
        )
    axes[0].set_ylabel("Separation margin")
    fig.subplots_adjust(left=0.14, right=0.985, top=0.94, bottom=0.20, wspace=0.18)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 450} if suffix == "png" else {}
        fig.savefig(
            output_dir
            / f"palate_full_h3k27ac_separation_balanced_unbalanced.{suffix}",
            bbox_inches="tight",
            pad_inches=0.02,
            **kwargs,
        )
    plt.close(fig)


def save_curve_plot(curves: pd.DataFrame, output_dir: Path) -> None:
    apply_nature_rc(font_size=10)
    mpl.rcParams.update(
        {
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "figure.titleweight": "normal",
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(4.13, 3.25), sharex=True)
    for axis_index, (axis, family) in enumerate(zip(axes, PEAK_FAMILIES)):
        local = curves[curves["family"].eq(family)]
        for method in METHODS:
            row = local[local["method"].eq(method.name)].sort_values("time")
            axis.plot(
                row["time"],
                row["regulatory_margin"],
                color=METHOD_COLORS[method.name],
                marker="o",
                ms=3.0,
                lw=1.25,
                label=PLOT_LABELS[method.name],
            )
        axis.axhline(0.0, color="#888888", lw=0.65, ls="--")
        axis.set_title(f"{family} margin ↑", fontsize=10, pad=3, weight="normal")
        axis.set_xticks(TIMES, STAGES)
        if axis_index == 0:
            axis.tick_params(labelbottom=False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
    fig.supylabel("Correct − opposite program", x=0.015, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=4,
        columnspacing=0.75,
        labelspacing=0.25,
        handlelength=1.35,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.17, right=0.98, top=0.86, bottom=0.12, hspace=0.48)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 450} if suffix == "png" else {}
        fig.savefig(
            output_dir / f"palate_full_atac_regulatory_margin_curves.{suffix}",
            bbox_inches="tight",
            pad_inches=0.02,
            **kwargs,
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    metrics_path = output_dir / "full_atac_sync_ablation_metrics.csv"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {metrics_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    required = [PEAK_ANNOTATION, RNA_REFERENCE, ATAC_REFERENCE, ATAC_RAW]
    for method in METHODS:
        if method.atac_path is not None:
            required.append(method.atac_path)
        if method.mass_path is not None:
            required.append(method.mass_path)
        if method.rna_path is not None:
            required.append(method.rna_path)
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))

    rna = ad.read_h5ad(RNA_REFERENCE, backed="r")
    atac = ad.read_h5ad(ATAC_REFERENCE, backed="r")
    raw = ad.read_h5ad(ATAC_RAW, backed="r")
    if not np.array_equal(rna.obs_names, atac.obs_names) or not np.array_equal(
        rna.obs_names, raw.obs_names
    ):
        raise ValueError("Paired RNA/ATAC/raw cells are not aligned")
    labels = rna.obs["celltype_sub"].astype(str).to_numpy()
    atlas_times = pd.to_numeric(
        rna.obs["time_point_processed"], errors="raise"
    ).to_numpy(float)
    rna_norm = np.asarray(rna.obsm["X_latent"], dtype=np.float32) / _load_scale(
        RNA_NORM
    )
    atac_norm = np.asarray(atac.obsm["X_lsi15"], dtype=np.float32) / _load_scale(
        ATAC_NORM
    )
    device = torch.device(args.device)
    film, film_source, film_checkpoint, _ = _load_film(device)
    initial_rows = np.flatnonzero(np.isclose(atlas_times, 0.0))
    if len(initial_rows) != 2570:
        raise ValueError(f"Expected 2570 E12.5 cells, found {len(initial_rows)}")
    cnc_mask = labels[initial_rows] == CNC
    if int(cnc_mask.sum()) != 1335:
        raise ValueError(f"Expected 1335 E12.5 CNC cells, found {cnc_mask.sum()}")

    peak_table = pd.read_csv(
        PEAK_ANNOTATION,
        usecols=["peak", *[name for pair in PEAK_FAMILIES.values() for name in pair]],
    )
    if not np.array_equal(peak_table["peak"].astype(str), raw.var_names.astype(str)):
        raise ValueError("External peak annotations do not align to raw ATAC peaks")
    depth = pd.to_numeric(rna.obs["nFeature_peaks"], errors="coerce").to_numpy(float)
    atlas_programs: dict[str, dict[str, np.ndarray]] = {}
    peak_counts: dict[str, dict[str, int]] = {}
    for family, (anterior_key, posterior_key) in PEAK_FAMILIES.items():
        anterior_positions = np.flatnonzero(peak_table[anterior_key].astype(bool))
        posterior_positions = np.flatnonzero(peak_table[posterior_key].astype(bool))
        atlas_programs[family] = {
            "anterior": cell_program_score(raw, anterior_positions, depth),
            "posterior": cell_program_score(raw, posterior_positions, depth),
        }
        peak_counts[family] = {
            "anterior": int(len(anterior_positions)),
            "posterior": int(len(posterior_positions)),
        }

    atac_fate_model, atac_fate_labels = fixed_fate_classifier(
        atac_norm, labels, atlas_times, args.knn_k
    )
    rna_fate_model, rna_fate_labels = fixed_fate_classifier(
        rna_norm, labels, atlas_times, args.knn_k
    )

    rng = np.random.default_rng(args.seed)
    metric_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    fate_rows: list[dict[str, object]] = []
    for method in METHODS:
        if method.atac_path is not None:
            atac_trajectory = load_tensor(method.atac_path)
            atac_source = str(method.atac_path)
        elif method.rna_path is not None:
            atac_trajectory = map_rna_trajectory(
                film,
                load_tensor(method.rna_path),
                device,
                args.batch_size,
            )
            atac_source = f"FiLM({method.rna_path})"
        else:
            raise ValueError(f"{method.name} has neither ATAC nor RNA trajectory")
        decoded = decode_programs(
            atac_trajectory, atac_norm, atlas_times, atlas_programs, args.knn_k
        )
        terminal_atac = atac_trajectory[TRAJECTORY_INDICES[-1], cnc_mask]
        fate, confidence = predict_fate(
            atac_fate_model, atac_fate_labels, terminal_atac
        )
        terminal_log_mass = (
            None
            if method.mass_path is None
            else load_tensor(method.mass_path)[TRAJECTORY_INDICES[-1]]
        )
        terminal_weights = normalized_weights(terminal_log_mass, cnc_mask)
        fate_binary = (fate == "anterior").astype(int)

        rna_atac_agreement = float("nan")
        if method.rna_path is not None:
            rna_trajectory = load_tensor(method.rna_path)
            rna_fate, _ = predict_fate(
                rna_fate_model,
                rna_fate_labels,
                rna_trajectory[TRAJECTORY_INDICES[-1], cnc_mask],
            )
            rna_atac_agreement = weighted_mean(
                (rna_fate == fate).astype(float), terminal_weights
            )

        fate_rows.append(
            {
                "method": method.name,
                "transport": method.transport,
                "n_initial_cnc": int(cnc_mask.sum()),
                "terminal_anterior_fraction": weighted_mean(
                    (fate == "anterior").astype(float), terminal_weights
                ),
                "terminal_posterior_fraction": weighted_mean(
                    (fate == "posterior").astype(float), terminal_weights
                ),
                "terminal_classifier_confidence": weighted_mean(
                    confidence, terminal_weights
                ),
                "rna_atac_terminal_fate_agreement": rna_atac_agreement,
                "rna_fate_readout": (
                    "direct same-particle RNA trajectory"
                    if method.rna_path is not None
                    else "not available for ATAC-only model"
                ),
                "atac_trajectory_source": atac_source,
            }
        )

        for family in PEAK_FAMILIES:
            anterior = decoded[family]["anterior"][:, cnc_mask]
            posterior = decoded[family]["posterior"][:, cnc_mask]
            signed = anterior - posterior
            margin = np.where(fate[None, :] == "anterior", signed, -signed)
            correct = np.where(
                fate[None, :] == "anterior", anterior, posterior
            )
            opposite = np.where(
                fate[None, :] == "anterior", posterior, anterior
            )
            initial_score = signed[0]
            early_score = signed[1]
            e140_score = signed[2]
            point = {
                "terminal_margin": weighted_mean(margin[-1], terminal_weights),
                "margin_gain": weighted_mean(
                    margin[-1] - margin[0], terminal_weights
                ),
                "monotonic_rate": weighted_mean(
                    np.mean(np.diff(margin, axis=0) >= 0, axis=0), terminal_weights
                ),
                "initial_fate_auroc": float(
                    roc_auc_score(
                        fate_binary, initial_score, sample_weight=terminal_weights
                    )
                ),
                "early_fate_auroc": float(
                    roc_auc_score(
                        fate_binary, early_score, sample_weight=terminal_weights
                    )
                ),
                "e140_fate_auroc": float(
                    roc_auc_score(
                        fate_binary, e140_score, sample_weight=terminal_weights
                    )
                ),
            }
            intervals = bootstrap_interval(
                rng,
                {
                    "margin": margin,
                    "initial_score": initial_score,
                    "early_score": early_score,
                    "e140_score": e140_score,
                    "fate_binary": fate_binary,
                },
                terminal_weights,
                args.bootstrap,
            )
            row: dict[str, object] = {
                "method": method.name,
                "transport": method.transport,
                "family": family,
                "terminal_correct_activation": weighted_mean(
                    correct[-1], terminal_weights
                ),
                "terminal_opposite_leakage": weighted_mean(
                    opposite[-1], terminal_weights
                ),
                **point,
            }
            for metric, value in point.items():
                low, high = intervals[metric]
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            metric_rows.append(row)

            for stage_index, stage in enumerate(STAGES):
                stage_log_mass = (
                    None
                    if method.mass_path is None
                    else load_tensor(method.mass_path)[TRAJECTORY_INDICES[stage_index]]
                )
                stage_weights = normalized_weights(stage_log_mass, cnc_mask)
                stage_margin = margin[stage_index]
                margin_ci_low, margin_ci_high = bootstrap_weighted_mean_interval(
                    rng,
                    stage_margin,
                    stage_weights,
                    args.bootstrap,
                )
                curve_rows.append(
                    {
                        "method": method.name,
                        "transport": method.transport,
                        "family": family,
                        "stage": stage,
                        "time": TIMES[stage_index],
                        "correct_activation": weighted_mean(
                            correct[stage_index], stage_weights
                        ),
                        "opposite_leakage": weighted_mean(
                            opposite[stage_index], stage_weights
                        ),
                        "regulatory_margin": weighted_mean(stage_margin, stage_weights),
                        "regulatory_margin_ci_low": margin_ci_low,
                        "regulatory_margin_ci_high": margin_ci_high,
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    curves = pd.DataFrame(curve_rows)
    fate_audit = pd.DataFrame(fate_rows)
    metrics.to_csv(metrics_path, index=False)
    curves.to_csv(output_dir / "full_atac_program_curves.csv", index=False)
    fate_audit.to_csv(output_dir / "terminal_fate_audit.csv", index=False)
    save_plot(metrics, output_dir)
    save_stage_auroc_plot(
        metrics,
        output_dir,
        stage="E12.5",
        metric="initial_fate_auroc",
        stem="palate_full_atac_e125_fate_auroc",
    )
    save_stage_auroc_plot(
        metrics,
        output_dir,
        stage="E14.0",
        metric="e140_fate_auroc",
        stem="palate_full_atac_e140_fate_auroc",
    )
    save_temporal_auroc_plot(metrics, output_dir)
    save_split_temporal_auroc_plot(metrics, output_dir)
    save_curve_plot(curves, output_dir)
    save_split_temporal_margin_plot(curves, output_dir)
    save_h3_balanced_unbalanced_plots(metrics, curves, output_dir)

    manifest = {
        "analysis": "palate full-data ATAC synchronization ablation with RNA-only controls",
        "heldout": False,
        "cy": 0.5,
        "seed": 0,
        "iteration": 20000,
        "initial_cohort": {
            "stage": "E12.5",
            "cell_type": CNC,
            "n_particles": int(cnc_mask.sum()),
            "paired_initial_ids_shared_across_methods": True,
        },
        "terminal_fate": {
            "classifier": f"{args.knn_k}NN in normalized terminal ATAC LSI15",
            "training_labels": [ANTERIOR, POSTERIOR],
        },
        "program_readout": {
            "knn": args.knn_k,
            "space": "stage-specific normalized ATAC LSI15",
            "raw_accessibility": "binary peak accessibility normalized by nFeature_peaks, log1p, then globally z-scored per module",
            "peak_sets": peak_counts,
        },
        "unbalanced_weighting": "native terminal mass normalized within the E12.5 CNC cohort; stage curves use stage-specific normalized mass",
        "rna_only_atac_mapping": {
            "map": "same frozen full-data FiLM RNA-to-ATAC map used by the benchmark",
            "source": str(film_source),
            "checkpoint": str(film_checkpoint),
        },
        "methods": [method.name for method in METHODS],
        "bootstrap": args.bootstrap,
        "initial_fate_auroc": (
            "E12.5 anterior-minus-posterior program score versus the same method's "
            "E14.5 ATAC-defined terminal branch; measures routing consistency with "
            "initial regulatory priming, not clonal fate ground truth"
        ),
        "e140_fate_auroc": (
            "E14.0 anterior-minus-posterior program score versus the same method's "
            "E14.5 ATAC-defined terminal branch"
        ),
        "important_boundary": "Terminal fate is inferred from predicted ATAC, so early-fate AUROC measures within-trajectory regulatory commitment, not clonal ground truth. RNA-only ATAC states are inferred through the frozen full-data FiLM map.",
    }
    with (output_dir / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nTerminal fate audit")
    print(fate_audit.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
