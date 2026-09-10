#!/usr/bin/env python
"""Redraw the rostral TSS ablation and quantify exact HEP routing over C_y."""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/trainfbench-cache")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from trainfbench_plot_style import (
    apply_nature_rc,
    gastrulation_celltype_color,
    method_style,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_GASTRULATION = Path(
    "external/COATI/Gastrulation"
)
SOURCE_THREECOMPARE = (
    SOURCE_GASTRULATION / "ResultCompare" / "ThreeCompare"
)
DEFAULT_TSS_DIR = (
    SOURCE_THREECOMPARE
    / "rostral_reviewer_proof_sync03_suite"
    / "matched_background_floor0_tss_openness"
)
DEFAULT_TRAJECTORY_DIR = SOURCE_GASTRULATION / "FilmSyncEnergy" / "trajectory"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "gastrulation_rostral_tss_ablation"

SCORE_FILE = "rostral_matched_background_floor0_tss_openness_scores.csv"
CATEGORY_FILE = "matched_background_floor0_categories_by_margin.csv"
ROUTE_FILE = "matched_background_floor0_neural_open_wrong_blood_route_summary.csv"

METHOD_ORDER = ("RNA-only", "Sync 0.3", "ATAC-only")
METHOD_DISPLAY = {
    "RNA-only": "OT(RNA)",
    "Sync 0.3": "COATI bal.",
    "ATAC-only": "OT(ATAC)",
}
METHOD_STYLE_KEY = {
    "RNA-only": "OT baseline RNA",
    "Sync 0.3": "COATI balanced",
    "ATAC-only": "OT baseline ATAC",
}

NEURAL_PROGRAM_COLOR = gastrulation_celltype_color("Rostral neurectoderm")
HEP_COLOR = gastrulation_celltype_color("Haematoendothelial progenitors")
AMBIGUOUS_COLOR = gastrulation_celltype_color("Unannotated")
HEP_LABEL = "Haematoendothelial progenitors"
TIME_KEYS = ("time0", "time1", "time2", "time3")
CY_VALUES = tuple(round(value, 1) for value in np.arange(0.1, 1.01, 0.1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tss-dir", type=Path, default=DEFAULT_TSS_DIR)
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=DEFAULT_TRAJECTORY_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-sweep-recompute",
        action="store_true",
        help="Reuse hep_assignment_by_cy_and_step.csv when it already exists.",
    )
    return parser.parse_args()


def configure_style() -> None:
    apply_nature_rc(font_size=10.0)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "font.weight": "normal",
            "axes.titlesize": 12.0,
            "axes.titleweight": "normal",
            "axes.labelsize": 12.0,
            "axes.labelweight": "normal",
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.0,
            "figure.titlesize": 12.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial",
            "savefig.dpi": 400,
        }
    )


def ensure_writable_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing outputs. Pass --overwrite:\n"
            f"{joined}"
        )


def clean_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color="#D9DDE3", linewidth=0.55)
    ax.set_axisbelow(True)


def select_tss_inputs(
    tss_dir: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    scores = pd.read_csv(tss_dir / SCORE_FILE)
    categories = pd.read_csv(tss_dir / CATEGORY_FILE)
    routed = pd.read_csv(tss_dir / ROUTE_FILE)

    category_match = categories[
        np.isclose(categories["margin_open0"], 0.1)
        & np.isclose(categories["min_program_open0"], 0.0)
    ]
    route_match = routed[
        np.isclose(routed["margin_open0"], 0.1)
        & np.isclose(routed["min_program_open0"], 0.0)
        & np.isclose(routed["p_blood_threshold"], 0.5)
    ].copy()
    if len(category_match) != 1:
        raise RuntimeError("The margin=0.1 regulatory summary is not unique.")
    if set(route_match["method"]) != set(METHOD_ORDER):
        raise RuntimeError("The three requested ablation conditions are missing.")
    route_match["method"] = pd.Categorical(
        route_match["method"],
        METHOD_ORDER,
        ordered=True,
    )
    return scores, category_match.iloc[0], route_match.sort_values("method")


def plot_tss_ablation(
    scores: pd.DataFrame,
    category_row: pd.Series,
    routed_rows: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.85, 2.55),
        gridspec_kw={
            "width_ratios": (1.08, 1.00, 1.12),
            "left": 0.075,
            "right": 0.992,
            "bottom": 0.26,
            "top": 0.86,
            "wspace": 0.44,
        },
    )

    ax = axes[0]
    violin_data = (
        scores["neural_matched_bg_open0"].to_numpy(dtype=float),
        scores["blood_endothelial_matched_bg_open0"].to_numpy(dtype=float),
    )
    violin_colors = (NEURAL_PROGRAM_COLOR, HEP_COLOR)
    parts = ax.violinplot(
        violin_data,
        positions=(0, 1),
        widths=0.72,
        showmeans=False,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], violin_colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_linewidth(0.6)
        body.set_alpha(0.25)

    rng = np.random.default_rng(0)
    for position, values, color in zip(
        (0, 1),
        violin_data,
        violin_colors,
    ):
        sampled = rng.choice(values, size=min(250, len(values)), replace=False)
        ax.scatter(
            position + rng.normal(0.0, 0.045, size=len(sampled)),
            sampled,
            s=4.0,
            color=color,
            alpha=0.34,
            linewidths=0.0,
        )
        mean = float(np.mean(values))
        ax.scatter(
            position,
            mean,
            marker="D",
            s=24.0,
            color="#000000",
            zorder=4,
        )
        ax.annotate(
            f"{mean:.2f}",
            (position, mean),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title("Starting TSS accessibility", pad=5)
    ax.set_ylabel("Above-background\nTSS openness")
    ax.set_xticks((0, 1), ("Neural", "Blood/endo"))
    ax.set_ylim(-0.02, 1.50)
    ax.set_yticks((0.0, 0.5, 1.0, 1.5))
    clean_axis(ax)

    ax = axes[1]
    class_labels = ("Neural open", "Blood/endo open", "Ambiguous")
    class_values = 100.0 * np.asarray(
        (
            category_row["frac_neural_open"],
            category_row["frac_blood_open"],
            category_row["frac_ambiguous"],
        ),
        dtype=float,
    )
    class_counts = np.asarray(
        (
            category_row["n_neural_open"],
            category_row["n_blood_open"],
            category_row["n_ambiguous"],
        ),
        dtype=int,
    )
    class_x = np.arange(3)
    bars = ax.bar(
        class_x,
        class_values,
        width=0.64,
        color=(NEURAL_PROGRAM_COLOR, HEP_COLOR, AMBIGUOUS_COLOR),
        linewidth=0.0,
    )
    for bar, value, count in zip(bars, class_values, class_counts):
        ax.annotate(
            f"{value:.1f}%\n(n={count})",
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            linespacing=0.95,
        )
    ax.set_title("Starting regulatory state", pad=5)
    ax.set_ylabel("% of rostral starts")
    ax.set_xticks(class_x, class_labels, rotation=16, ha="right")
    ax.set_xlim(-0.70, 2.50)
    ax.set_ylim(0, 92)
    ax.set_yticks((0, 40, 80))
    clean_axis(ax)

    ax = axes[2]
    route_values = (
        100.0
        * routed_rows[
            "frac_neural_open_high_blood_among_neural_open"
        ].to_numpy(dtype=float)
    )
    route_counts = routed_rows["n_neural_open_high_blood"].to_numpy(dtype=int)
    n_neural_open = int(category_row["n_neural_open"])
    x = np.arange(len(METHOD_ORDER), dtype=float)
    method_colors = [
        method_style(METHOD_STYLE_KEY[str(method)]).color
        for method in routed_rows["method"]
    ]
    bars = ax.bar(
        x,
        route_values,
        width=0.64,
        color=method_colors,
        linewidth=0.0,
    )
    for bar, value, count in zip(bars, route_values, route_counts):
        ax.annotate(
            f"{value:.1f}%\n({count}/{n_neural_open})",
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            linespacing=0.95,
        )
    route_labels = [
        METHOD_DISPLAY[str(method)] for method in routed_rows["method"]
    ]
    ax.set_title("Assigned to blood/endothelium", pad=5)
    ax.set_ylabel("% of neural-open starts")
    ax.set_xticks(x, route_labels, rotation=16, ha="right")
    ax.set_xlim(-0.70, 2.50)
    ax.set_ylim(0, 21.5)
    ax.set_yticks((0, 10, 20))
    clean_axis(ax)
    return fig


def load_rna_reference() -> tuple[np.ndarray, np.ndarray, float]:
    rna_by_time = np.load(
        SOURCE_GASTRULATION / "data" / "rna_pca_by_time.npz",
        allow_pickle=True,
    )
    celltype_by_time = np.load(
        SOURCE_GASTRULATION / "data" / "celltype_sub_by_stage.npz",
        allow_pickle=True,
    )
    reference = np.concatenate(
        [np.asarray(rna_by_time[key], dtype=np.float32) for key in TIME_KEYS],
        axis=0,
    )
    labels = np.concatenate(
        [np.asarray(celltype_by_time[key]).astype(str) for key in TIME_KEYS],
        axis=0,
    )
    scale = float(
        torch.load(
            SOURCE_GASTRULATION / "data" / "primal_norm_params.pt",
            map_location="cpu",
            weights_only=False,
        )["scale"]
    )
    return reference, labels, scale


def neural_open_indices(scores: pd.DataFrame) -> np.ndarray:
    mask = (
        (scores["neural_matched_bg_open0"] >= 0.0)
        & (scores["delta_neural_minus_blood_open0"] >= 0.1)
    )
    indices = scores.loc[mask, "global_cell_index"].to_numpy(dtype=int)
    if len(indices) != 1107:
        raise RuntimeError(
            f"Expected 1,107 neural-open rostral starts, found {len(indices)}."
        )
    return indices


def trajectory_path(trajectory_dir: Path, cy: float) -> Path:
    return (
        trajectory_dir
        / f"primary_trajectory_s0_a{cy:.1f}_iter50000.pt"
    )


def compute_hep_sweep(
    scores: pd.DataFrame,
    trajectory_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference, labels, scale = load_rna_reference()
    nn = cKDTree(reference)
    cohort_indices = neural_open_indices(scores)

    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for cy in CY_VALUES:
        path = trajectory_path(trajectory_dir, cy)
        if not path.exists():
            raise FileNotFoundError(path)
        trajectory = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        if tuple(trajectory.shape[1:]) != (9018, 50):
            raise ValueError(f"Unexpected trajectory shape for {path}: {trajectory.shape}")

        states = (
            trajectory[:, cohort_indices, :] * scale
        ).detach().cpu().numpy().astype(np.float32)
        n_steps, n_cells, dimension = states.shape
        _, nearest = nn.query(
            states.reshape(n_steps * n_cells, dimension),
            k=1,
            workers=4,
        )
        assigned = labels[nearest].reshape(n_steps, n_cells)
        is_hep_by_step = assigned == HEP_LABEL
        ever_hep = np.any(is_hep_by_step, axis=0)
        step_counts: list[int] = []
        for step in range(n_steps):
            is_hep = is_hep_by_step[step]
            count = int(np.sum(is_hep))
            step_counts.append(count)
            rows.append(
                {
                    "C_y": cy,
                    "step": step,
                    "stage": 7.5 + 0.05 * step,
                    "n_neural_open_rostral_starts": len(cohort_indices),
                    "n_assigned_HEP": count,
                    "fraction_assigned_HEP": count / len(cohort_indices),
                    "percent_assigned_HEP": 100.0 * count / len(cohort_indices),
                    "trajectory": str(path),
                    "training_iteration": 50000,
                    "assignment": "1-NN to the observed all-stage RNA PCA atlas",
                }
            )
        peak_step = int(np.argmax(step_counts))
        peak_count = int(step_counts[peak_step])
        summary_rows.append(
            {
                "C_y": cy,
                "peak_step": peak_step,
                "peak_stage": 7.5 + 0.05 * peak_step,
                "n_neural_open_rostral_starts": len(cohort_indices),
                "peak_n_assigned_HEP": peak_count,
                "peak_fraction_assigned_HEP": peak_count / len(cohort_indices),
                "peak_percent_assigned_HEP": 100.0 * peak_count / len(cohort_indices),
                "terminal_n_assigned_HEP": int(step_counts[-1]),
                "terminal_percent_assigned_HEP": (
                    100.0 * step_counts[-1] / len(cohort_indices)
                ),
                "ever_n_assigned_HEP": int(np.sum(ever_hep)),
                "ever_percent_assigned_HEP": (
                    100.0 * np.mean(ever_hep)
                ),
                "trajectory": str(path),
                "training_iteration": 50000,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def plot_hep_sweep(summary: pd.DataFrame) -> plt.Figure:
    summary = summary.sort_values("C_y")
    x = summary["C_y"].to_numpy(dtype=float)
    y = summary["peak_percent_assigned_HEP"].to_numpy(dtype=float)
    counts = summary["peak_n_assigned_HEP"].to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(4.05, 2.75))
    fig.subplots_adjust(left=0.21, right=0.97, bottom=0.22, top=0.83)
    ax.plot(
        x,
        y,
        color=HEP_COLOR,
        marker="o",
        markersize=5.0,
        linewidth=1.8,
    )
    for cy, percent, count in zip(x, y, counts):
        ax.annotate(
            f"{percent:.2f}",
            (cy, percent),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    selected = np.isclose(x, 0.3)
    if selected.any():
        ax.scatter(
            x[selected],
            y[selected],
            s=66,
            facecolors="none",
            edgecolors=method_style("COATI balanced").color,
            linewidths=1.4,
            zorder=4,
        )
    upper = max(1.0, float(np.max(y)) + 0.28)
    ax.set_title("Peak HEP assignment", pad=6)
    ax.set_xlabel(r"$C_y$")
    ax.set_ylabel("% of neural-open\nrostral starts")
    ax.set_xlim(0.06, 1.04)
    ax.set_xticks(x)
    ax.set_ylim(0.0, upper)
    clean_axis(ax)
    return fig


def save_figure(fig: plt.Figure, prefix: Path) -> None:
    fig.savefig(
        prefix.with_suffix(".png"),
        dpi=400,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        prefix.with_suffix(".pdf"),
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation_prefix = output_dir / "rostral_tss_regulatory_ablation"
    sweep_prefix = output_dir / "rostral_hep_assignment_by_cy"
    step_csv = output_dir / "hep_assignment_by_cy_and_step.csv"
    summary_csv = output_dir / "hep_assignment_by_cy_summary.csv"
    manifest_csv = output_dir / "analysis_manifest.csv"
    outputs = [
        ablation_prefix.with_suffix(".png"),
        ablation_prefix.with_suffix(".pdf"),
        sweep_prefix.with_suffix(".png"),
        sweep_prefix.with_suffix(".pdf"),
        step_csv,
        summary_csv,
        manifest_csv,
    ]
    ensure_writable_outputs(outputs, args.overwrite)

    configure_style()
    scores, category_row, routed_rows = select_tss_inputs(args.tss_dir.resolve())
    ablation = plot_tss_ablation(scores, category_row, routed_rows)
    save_figure(ablation, ablation_prefix)

    if args.skip_sweep_recompute:
        if not step_csv.exists() or not summary_csv.exists():
            raise FileNotFoundError(
                "--skip-sweep-recompute requires the two existing sweep CSVs."
            )
        sweep_summary = pd.read_csv(summary_csv)
    else:
        sweep_steps, sweep_summary = compute_hep_sweep(
            scores,
            args.trajectory_dir.resolve(),
        )
        sweep_steps.to_csv(step_csv, index=False)
        sweep_summary.to_csv(summary_csv, index=False)
    sweep = plot_hep_sweep(sweep_summary)
    save_figure(sweep, sweep_prefix)

    pd.DataFrame(
        [
            {
                "analysis": "rostral TSS and regulatory-state ablation",
                "source_directory": str(args.tss_dir.resolve()),
                "conditions": (
                    "OT(RNA); COATI bal.; OT(ATAC)"
                ),
                "COATI_C_y": 0.3,
                "training_iteration": 20000,
                "regulatory_definition": (
                    "neural_open0 >= 0 and "
                    "neural_open0 - blood_endothelial_open0 >= 0.1"
                ),
                "route_definition": (
                    "terminal k=50 blood/endothelium probability > 0.5"
                ),
            },
            {
                "analysis": "exact HEP assignment across C_y",
                "source_directory": str(args.trajectory_dir.resolve()),
                "conditions": "COATI bal.; C_y=0.1,...,1.0; seed=0",
                "COATI_C_y": np.nan,
                "training_iteration": 50000,
                "regulatory_definition": (
                    "same 1,107 neural-open rostral-start cells"
                ),
                "route_definition": (
                    "peak instantaneous exact Haematoendothelial progenitors "
                    "assignment across 26 steps; 1-NN to all-stage RNA PCA atlas"
                ),
            },
        ]
    ).to_csv(manifest_csv, index=False)

    print(ablation_prefix.with_suffix(".png"))
    print(ablation_prefix.with_suffix(".pdf"))
    print(sweep_prefix.with_suffix(".png"))
    print(sweep_prefix.with_suffix(".pdf"))
    print(step_csv)
    print(summary_csv)
    print(manifest_csv)
    print("\nHEP C_y summary:")
    print(
        sweep_summary[
            [
                "C_y",
                "peak_step",
                "peak_percent_assigned_HEP",
                "terminal_percent_assigned_HEP",
                "ever_percent_assigned_HEP",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
