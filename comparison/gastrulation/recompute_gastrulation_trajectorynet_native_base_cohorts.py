#!/usr/bin/env python
"""Recompute TrajectoryNet cohorts from its native standard-normal base.

TrajectoryNet is a generative CNF.  Unlike the other benchmark methods, its
native use starts from the learned standard-normal base, generates the E7.5
marginal, and only then defines a biological cohort.  This script:

1. samples the native base and generates E7.5;
2. assigns E7.5 cell types by the shared global RNA 1NN readout;
3. retains the generated particles assigned to Rostral neurectoderm and, as a
   consistency control for the existing Caudal-epiblast figure, Caudal
   epiblast;
4. propagates the same particles through E8.0, E8.5, and E8.75;
5. replaces only the TrajectoryNet panel in the existing seven-method
   Caudal-epiblast composition figure.
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
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torchdiffeq import odeint


ROOT = Path(__file__).resolve().parents[2]
THREECOMPARE = Path(
    "external/COATI/Gastrulation/"
    "ResultCompare/ThreeCompare"
)
DEFAULT_CHECKPOINT = (
    ROOT
    / "results/trajectorynet_gastrulation_rna_20000/checkpt-20000.pth"
)
DEFAULT_OUTPUT_DIR = (
    THREECOMPARE / "trajectorynet_native_base_conditioned_composition"
)
BALANCED_COMPOSITION = (
    THREECOMPARE
    / "nmp_balanced_methods_main_analysis_gaga10"
    / "caudal_epiblast_group_composition.csv"
)
UNBALANCED_COMPOSITION = (
    THREECOMPARE
    / "nmp_unbalanced_methods_composition_ae10"
    / "caudal_epiblast_composition_by_step.csv"
)
ROSTRAL_OURS_COMPOSITION = (
    THREECOMPARE
    / "rostral_all_start_ours_composition_20000"
    / "all_rostral_ours_grouped_lineage_composition_by_step.csv"
)
ROSTRAL_EXTERNAL_COMPOSITION = (
    THREECOMPARE
    / "rostral_all_start_external_composition_20000"
    / "external_all_rostral_grouped_lineage_composition_by_step.csv"
)

sys.path.insert(0, str(ROOT / "common"))
from evaluate_gastrulation_full_cmcc import _load_references  # noqa: E402
from evaluate_terminal_push import (  # noqa: E402
    _load_trajectorynet_model,
    _trajectorynet_diffeq,
)
from trainfbench_plot_style import (  # noqa: E402
    GASTRULATION_FATE_STYLES,
    apply_nature_rc,
)


N_STATES = 26
INTEGRATION_STEPS = 30
PLOT_STEPS = (10, 10, 5)
STAGE_STEPS = (10, 20, 25)
FATE_GROUPS = (
    "Caudal epiblast",
    "NMP",
    "Posterior neural-spinal",
    "Paraxial-somitic",
    "Other",
)
METHOD_PANEL_ORDER = (
    "COATI balanced",
    "COATI unbalanced",
    "CytoBridge balanced",
    "CytoBridge unbalanced",
    "TrajectoryNet",
    "TIGON",
    "MIOFlow",
)
ROSTRAL_LINEAGES = (
    "Rostral neurectoderm",
    "Forebrain/Midbrain/Hindbrain",
    "Blood/erythroid lineage",
    "Other",
)
ROSTRAL_LINEAGE_COLORS = {
    "Rostral neurectoderm": "#2F6F9F",
    "Forebrain/Midbrain/Hindbrain": "#69A7C4",
    "Blood/erythroid lineage": "#D95F5F",
    "Other": "#6B7280",
}
ROSTRAL_METHOD_PANELS = (
    ("RNA-only", "BOT-RNA"),
    ("Sync 0.3", "COATI-BOT (sync 0.3)"),
    ("ATAC-only", "BOT-ATAC"),
    ("TrajectoryNet", "TrajectoryNet"),
    ("CytoBridge (balanced)", "CytoBridge (balanced)"),
    ("MIOFlow", "MIOFlow"),
)
BLOOD_ERYTHROID_CELLTYPES = {
    "Endothelium",
    "Haematoendothelial progenitors",
    "Blood progenitors 1",
    "Blood progenitors 2",
    "Erythroid1",
    "Erythroid2",
    "Erythroid3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-base", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda"), default="cpu"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fate_group(celltype: str) -> str:
    if celltype == "Caudal epiblast":
        return "Caudal epiblast"
    if celltype == "NMP":
        return "NMP"
    if celltype in {"Caudal neurectoderm", "Spinal cord", "Neural crest"}:
        return "Posterior neural-spinal"
    if celltype in {
        "Paraxial mesoderm",
        "Somitic mesoderm",
        "Caudal Mesoderm",
    }:
        return "Paraxial-somitic"
    return "Other"


def integrate_segment(
    values: np.ndarray,
    diffeq,
    upper: float,
    lower: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    times = torch.linspace(
        upper,
        lower,
        INTEGRATION_STEPS + 1,
        dtype=torch.float32,
        device=device,
    )
    batches: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        state = torch.as_tensor(
            values[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            path = odeint(
                diffeq,
                state,
                times,
                method="rk4",
                options={"step_size": 0.1},
            )
        batches.append(path.detach().cpu().numpy())
    return np.concatenate(batches, axis=1).astype(np.float32, copy=False)


def native_base_trajectory(
    checkpoint: Path,
    dimension: int,
    n_base: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    model, model_args = _load_trajectorynet_model(
        checkpoint,
        dimension,
        device,
        "rk4",
        0.1,
    )
    diffeq = _trajectorynet_diffeq(model)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    base = torch.randn(
        (n_base, dimension),
        generator=generator,
        dtype=torch.float32,
    ).numpy()

    # Native TrajectoryNet generation of the first observed marginal.
    base_to_e75 = integrate_segment(
        base,
        diffeq,
        float(model_args.time_scale),
        0.0,
        device,
        batch_size,
    )
    current = base_to_e75[-1]
    interval_paths: list[np.ndarray] = []
    for interval, plot_steps in enumerate(PLOT_STEPS):
        upper = (interval + 2.0) * float(model_args.time_scale)
        lower = (interval + 1.0) * float(model_args.time_scale)
        full_path = integrate_segment(
            current,
            diffeq,
            upper,
            lower,
            device,
            batch_size,
        )
        if INTEGRATION_STEPS % plot_steps != 0:
            raise ValueError("Integration steps must divide plotted steps")
        stride = INTEGRATION_STEPS // plot_steps
        sampled = full_path[::stride]
        interval_paths.append(sampled)
        current = full_path[-1]

    dense = np.concatenate(
        [
            interval_paths[0],
            interval_paths[1][1:],
            interval_paths[2][1:],
        ],
        axis=0,
    )
    if dense.shape != (N_STATES, n_base, dimension):
        raise RuntimeError(f"Unexpected dense trajectory shape: {dense.shape}")
    return dense, {
        "base_distribution": "standard normal",
        "n_base": n_base,
        "seed": seed,
        "time_scale": float(model_args.time_scale),
        "base_to_E7.5_internal_clock": [
            float(model_args.time_scale),
            0.0,
        ],
        "biological_forward_internal_clocks": [
            [
                (interval + 2.0) * float(model_args.time_scale),
                (interval + 1.0) * float(model_args.time_scale),
            ]
            for interval in range(3)
        ],
    }


def labels_from_trajectory(
    trajectory_raw: np.ndarray,
    reference_normalized: np.ndarray,
    reference_labels: np.ndarray,
    rna_scale: float,
) -> np.ndarray:
    neighbors = NearestNeighbors(
        n_neighbors=1,
        algorithm="brute",
        n_jobs=-1,
    ).fit(reference_normalized)
    n_steps, n_cells, dimension = trajectory_raw.shape
    indices = neighbors.kneighbors(
        (trajectory_raw / rna_scale).reshape(n_steps * n_cells, dimension),
        return_distance=False,
    )[:, 0]
    return reference_labels[indices].reshape(n_steps, n_cells)


def summarize_cohort(
    labels: np.ndarray,
    cohort_mask: np.ndarray,
    cohort: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = labels[:, cohort_mask]
    if selected.shape[1] == 0:
        raise RuntimeError(f"No generated E7.5 cells were assigned to {cohort}")
    fine_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    n_cells = selected.shape[1]
    all_celltypes = sorted(np.unique(labels.astype(str)))
    for step in range(N_STATES):
        values = selected[step].astype(str)
        counts = Counter(values)
        for celltype in all_celltypes:
            count = int(counts.get(celltype, 0))
            fine_rows.append(
                {
                    "cohort": cohort,
                    "method": "TrajectoryNet",
                    "step": step,
                    "celltype": celltype,
                    "group": fate_group(celltype),
                    "count": count,
                    "fraction": count / n_cells,
                    "percent": 100.0 * count / n_cells,
                    "n_cells": n_cells,
                }
            )
        grouped = Counter(fate_group(value) for value in values)
        for group in FATE_GROUPS:
            count = int(grouped.get(group, 0))
            group_rows.append(
                {
                    "cohort": cohort,
                    "method": "TrajectoryNet",
                    "step": step,
                    "group": group,
                    "count": count,
                    "fraction": count / n_cells,
                    "percent": 100.0 * count / n_cells,
                    "n_cells": n_cells,
                }
            )
    return pd.DataFrame(fine_rows), pd.DataFrame(group_rows)


def rostral_lineage(celltype: str) -> str:
    if celltype == "Rostral neurectoderm":
        return "Rostral neurectoderm"
    if celltype == "Forebrain/Midbrain/Hindbrain":
        return "Forebrain/Midbrain/Hindbrain"
    if celltype in BLOOD_ERYTHROID_CELLTYPES:
        return "Blood/erythroid lineage"
    return "Other"


def summarize_rostral_lineages(fine: pd.DataFrame) -> pd.DataFrame:
    table = fine.copy()
    table["lineage"] = table["celltype"].astype(str).map(rostral_lineage)
    grouped = (
        table.groupby(["method", "step", "lineage"], as_index=False)[
            ["count", "fraction", "percent"]
        ]
        .sum()
    )
    complete = pd.MultiIndex.from_product(
        [
            ["TrajectoryNet"],
            sorted(table["step"].unique()),
            ROSTRAL_LINEAGES,
        ],
        names=["method", "step", "lineage"],
    ).to_frame(index=False)
    return complete.merge(
        grouped,
        on=["method", "step", "lineage"],
        how="left",
    ).fillna({"count": 0, "fraction": 0.0, "percent": 0.0})


def plot_rostral_multi_method(
    trajectorynet_rostral: pd.DataFrame,
    prefix: Path,
) -> pd.DataFrame:
    ours = pd.read_csv(ROSTRAL_OURS_COMPOSITION)
    external = pd.read_csv(ROSTRAL_EXTERNAL_COMPOSITION)
    external = external[external["method"] != "TrajectoryNet"]
    combined = pd.concat(
        [ours, external, trajectorynet_rostral],
        ignore_index=True,
    )
    totals = combined.groupby(["method", "step"])["percent"].sum()
    if not np.allclose(totals.to_numpy(), 100.0, atol=1e-6):
        raise RuntimeError("Rostral lineage compositions do not sum to 100%")

    apply_nature_rc(font_size=10)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(9.2, 4.05),
        sharex=True,
        sharey=True,
    )
    for panel, (ax, (method, title)) in enumerate(
        zip(axes.ravel(), ROSTRAL_METHOD_PANELS)
    ):
        local = combined[combined["method"] == method]
        for lineage in ROSTRAL_LINEAGES:
            curve = local[local["lineage"] == lineage].sort_values("step")
            ax.plot(
                curve["step"],
                curve["percent"],
                color=ROSTRAL_LINEAGE_COLORS[lineage],
                linewidth=1.4,
                marker="o",
                markersize=2.5,
                markevery=2,
            )
        ax.set_title(
            title,
            fontsize=10,
            fontfamily="Arial",
            fontweight="normal",
            pad=2,
        )
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
        ax.set_xticks([0, 10, 20, 25])
        ax.set_xticklabels(["E7.5", "E8.0", "E8.5", "\nE8.75"])
        tick_labels = ax.get_xticklabels()
        if len(tick_labels) == 4:
            tick_labels[0].set_ha("left")
            tick_labels[1].set_ha("center")
            tick_labels[2].set_ha("center")
            tick_labels[3].set_ha("right")
        ax.tick_params(
            labelleft=(panel % 3 == 0),
            labelbottom=(panel >= 3),
            labelsize=10,
            pad=1,
        )
        ax.grid(axis="y", color="#E6E8EC", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = []
    for lineage in ROSTRAL_LINEAGES:
        handle, = axes[0, 0].plot(
            [],
            [],
            color=ROSTRAL_LINEAGE_COLORS[lineage],
            linewidth=1.4,
            marker="o",
            markersize=3.5,
        )
        handles.append(handle)
    fig.legend(
        handles,
        ROSTRAL_LINEAGES,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        frameon=False,
        fontsize=10,
        handlelength=1.5,
        columnspacing=1.0,
        labelspacing=0.2,
    )
    fig.supylabel(
        "Composition (%)",
        x=0.01,
        fontsize=12,
        fontfamily="Arial",
        fontweight="normal",
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.995,
        bottom=0.24,
        top=0.98,
        wspace=0.22,
        hspace=0.20,
    )
    fig.savefig(prefix.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return combined


def load_existing_seven_method_composition() -> pd.DataFrame:
    balanced = pd.read_csv(BALANCED_COMPOSITION).rename(
        columns={"group": "fate_group"}
    )
    balanced["method"] = balanced["method"].replace(
        {"COATI-BSOT": "COATI balanced"}
    )
    balanced["percent"] = 100.0 * balanced["fraction"]
    balanced = balanced[["method", "step", "fate_group", "percent"]]

    unbalanced = pd.read_csv(UNBALANCED_COMPOSITION)
    unbalanced = unbalanced[
        unbalanced["weighting"] == "normalized native mass"
    ][["method", "step", "fate_group", "percent"]]
    unbalanced["method"] = unbalanced["method"].replace(
        {
            "COATI unbal.": "COATI unbalanced",
            "CytoBridge unbal.": "CytoBridge unbalanced",
        }
    )
    return pd.concat([balanced, unbalanced], ignore_index=True)


def plot_seven_method_caudal(
    existing: pd.DataFrame,
    trajectorynet_caudal: pd.DataFrame,
    prefix: Path,
) -> pd.DataFrame:
    replacement = trajectorynet_caudal.rename(
        columns={"group": "fate_group"}
    )[["method", "step", "fate_group", "percent"]]
    combined = pd.concat(
        [existing[existing["method"] != "TrajectoryNet"], replacement],
        ignore_index=True,
    )
    apply_nature_rc(font_size=10)
    fig, axes_array = plt.subplots(
        4,
        2,
        figsize=(3.25, 5.3),
        sharex=True,
        sharey=True,
    )
    axes = axes_array.ravel()
    titles = {
        "COATI balanced": "COATI bal.",
        "COATI unbalanced": "COATI unbal.",
        "CytoBridge balanced": "CytoBridge bal.",
        "CytoBridge unbalanced": "CytoBridge unbal.",
        "TrajectoryNet": "TrajectoryNet",
        "TIGON": "TIGON",
        "MIOFlow": "MIOFlow",
    }
    for panel, (ax, method) in enumerate(zip(axes[:7], METHOD_PANEL_ORDER)):
        local = combined[combined["method"] == method]
        for group in FATE_GROUPS:
            style = GASTRULATION_FATE_STYLES[group]
            curve = local[local["fate_group"] == group].sort_values("step")
            ax.plot(
                curve["step"],
                curve["percent"],
                color=style.color,
                linewidth=1.3,
                marker=style.marker,
                markersize=2.7,
                markevery=2,
            )
        ax.set_title(
            titles[method],
            fontsize=10,
            fontfamily="Arial",
            fontweight="normal",
            pad=1,
        )
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 100)
        ax.set_xticks((0, *STAGE_STEPS))
        ax.set_xticklabels(("E7.5", "E8.0", "E8.5", "E8.75"))
        tick_labels = ax.get_xticklabels()
        if len(tick_labels) == 4:
            tick_labels[0].set_ha("left")
            tick_labels[2].set_ha("right")
            tick_labels[3].set_ha("left")
        ax.set_yticks([0, 50, 100])
        for stage_step in STAGE_STEPS:
            ax.axvline(
                stage_step,
                color="#B8B8B8",
                linewidth=0.7,
                linestyle=":",
                zorder=0,
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(
            labelleft=(panel % 2 == 0),
            labelbottom=(panel >= 6),
            labelsize=10,
            pad=1,
        )

    legend_ax = axes[7]
    legend_ax.axis("off")
    handles = []
    labels = []
    short = {
        "Caudal epiblast": "Caudal epi.",
        "NMP": "NMP",
        "Posterior neural-spinal": "Neural–spinal",
        "Paraxial-somitic": "Paraxial–somitic",
        "Other": "Other",
    }
    for group in FATE_GROUPS:
        style = GASTRULATION_FATE_STYLES[group]
        handle, = legend_ax.plot(
            [],
            [],
            color=style.color,
            linewidth=1.4,
            marker=style.marker,
            markersize=4,
        )
        handles.append(handle)
        labels.append(short[group])
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        handlelength=1.6,
        labelspacing=0.2,
        prop={"family": "Arial", "size": 10, "weight": "normal"},
    )
    fig.supylabel(
        "Composition (%)",
        x=0.035,
        fontsize=12,
        fontfamily="Arial",
        fontweight="normal",
    )
    fig.subplots_adjust(
        left=0.17,
        right=0.995,
        bottom=0.10,
        top=0.975,
        wspace=0.12,
        hspace=0.22,
    )
    fig.savefig(
        prefix.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        prefix.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    return combined


def plot_rostral_fine(fine: pd.DataFrame, prefix: Path) -> None:
    preferred = [
        "Rostral neurectoderm",
        "Forebrain/Midbrain/Hindbrain",
        "Caudal neurectoderm",
        "Spinal cord",
        "Surface ectoderm",
        "NMP",
        "Neural crest",
        "Paraxial mesoderm",
        "Somitic mesoderm",
    ]
    maxima = fine.groupby("celltype")["percent"].max()
    shown = [
        celltype
        for celltype in preferred
        if float(maxima.get(celltype, 0.0)) >= 1.0
    ]
    residual = (
        100.0
        - fine[fine["celltype"].isin(shown)]
        .groupby("step")["percent"]
        .sum()
    )
    colors = plt.get_cmap("tab10")
    apply_nature_rc(font_size=10)
    fig, ax = plt.subplots(figsize=(4.13, 2.65))
    for index, celltype in enumerate(shown):
        curve = fine[fine["celltype"] == celltype].sort_values("step")
        ax.plot(
            curve["step"],
            curve["percent"],
            linewidth=1.5,
            marker="o",
            markersize=2.7,
            markevery=2,
            color=colors(index % 10),
            label=celltype,
        )
    ax.plot(
        residual.index,
        residual.values,
        linewidth=1.4,
        color="#7A7F87",
        label="Other",
    )
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 100)
    ax.set_xticks((0, *STAGE_STEPS))
    ax.set_xticklabels(("E7.5", "E8.0", "E8.5", "E8.75"))
    tick_labels = ax.get_xticklabels()
    if len(tick_labels) == 4:
        tick_labels[0].set_ha("left")
        tick_labels[2].set_ha("right")
        tick_labels[3].set_ha("left")
    ax.set_ylabel("Composition (%)", fontsize=12)
    ax.set_title(
        "TrajectoryNet: native-base E7.5 rostral cohort",
        fontsize=12,
        fontweight="normal",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        fontsize=10,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )
    fig.subplots_adjust(left=0.15, right=0.99, top=0.88, bottom=0.35)
    fig.savefig(prefix.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected = [
        args.output_dir / "trajectorynet_native_base_labels.npz",
        args.output_dir / "trajectorynet_native_base_fine_composition.csv",
        args.output_dir / "trajectorynet_native_base_group_composition.csv",
        args.output_dir
        / "caudal_epiblast_seven_methods_trajectorynet_native_base.png",
        args.output_dir
        / "caudal_epiblast_seven_methods_trajectorynet_native_base.pdf",
        args.output_dir / "trajectorynet_native_base_rostral_composition.png",
        args.output_dir / "trajectorynet_native_base_rostral_composition.pdf",
        args.output_dir
        / "rostral_multi_method_trajectorynet_native_base.png",
        args.output_dir
        / "rostral_multi_method_trajectorynet_native_base.pdf",
        args.output_dir / "run_metadata.json",
    ]
    existing = [path for path in expected if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs; pass --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )

    rna_refs, _, stage_labels, rna_scale, _ = _load_references()
    reference = np.concatenate(rna_refs, axis=0).astype(
        np.float32, copy=False
    )
    reference_labels = np.concatenate(stage_labels).astype(str)
    trajectory_raw, clock = native_base_trajectory(
        args.checkpoint,
        reference.shape[1],
        args.n_base,
        args.seed,
        torch.device(args.device),
        args.batch_size,
    )
    labels = labels_from_trajectory(
        trajectory_raw,
        reference,
        reference_labels,
        rna_scale,
    )
    e75_labels = labels[0].astype(str)
    cohorts = ("Rostral neurectoderm", "Caudal epiblast")
    fine_frames: list[pd.DataFrame] = []
    group_frames: list[pd.DataFrame] = []
    masks: dict[str, np.ndarray] = {}
    for cohort in cohorts:
        masks[cohort] = e75_labels == cohort
        fine, group = summarize_cohort(labels, masks[cohort], cohort)
        fine_frames.append(fine)
        group_frames.append(group)

    fine_frame = pd.concat(fine_frames, ignore_index=True)
    group_frame = pd.concat(group_frames, ignore_index=True)
    fine_frame.to_csv(
        args.output_dir / "trajectorynet_native_base_fine_composition.csv",
        index=False,
    )
    group_frame.to_csv(
        args.output_dir / "trajectorynet_native_base_group_composition.csv",
        index=False,
    )
    np.savez_compressed(
        args.output_dir / "trajectorynet_native_base_labels.npz",
        labels=labels,
        e75_labels=e75_labels,
        rostral_mask=masks["Rostral neurectoderm"],
        caudal_epiblast_mask=masks["Caudal epiblast"],
        stage_steps=np.asarray([0, *STAGE_STEPS], dtype=int),
    )

    caudal = group_frame[
        group_frame["cohort"] == "Caudal epiblast"
    ].copy()
    combined = plot_seven_method_caudal(
        load_existing_seven_method_composition(),
        caudal,
        args.output_dir
        / "caudal_epiblast_seven_methods_trajectorynet_native_base",
    )
    combined.to_csv(
        args.output_dir
        / "caudal_epiblast_seven_methods_trajectorynet_native_base.csv",
        index=False,
    )
    rostral = fine_frame[
        fine_frame["cohort"] == "Rostral neurectoderm"
    ].copy()
    plot_rostral_fine(
        rostral,
        args.output_dir / "trajectorynet_native_base_rostral_composition",
    )
    rostral_grouped = summarize_rostral_lineages(rostral)
    rostral_grouped.to_csv(
        args.output_dir
        / "trajectorynet_native_base_rostral_grouped_lineages.csv",
        index=False,
    )
    rostral_combined = plot_rostral_multi_method(
        rostral_grouped,
        args.output_dir / "rostral_multi_method_trajectorynet_native_base",
    )
    rostral_combined.to_csv(
        args.output_dir
        / "rostral_multi_method_trajectorynet_native_base.csv",
        index=False,
    )

    initial_counts = Counter(e75_labels)
    metadata = {
        "analysis": (
            "TrajectoryNet native-base generation followed by E7.5 "
            "cell-type-conditioned trajectory composition"
        ),
        "checkpoint": str(args.checkpoint),
        "clock": clock,
        "rna_scale_for_shared_1NN": float(rna_scale),
        "classification": (
            "Global four-stage normalized RNA 1NN, matching the existing "
            "seven-method composition figure"
        ),
        "generated_E7.5_celltype_counts": dict(initial_counts),
        "selected_cohort_counts": {
            cohort: int(mask.sum()) for cohort, mask in masks.items()
        },
        "note": (
            "The seven-method Caudal-epiblast figure uses the generated E7.5 "
            "Caudal-epiblast subset so that all seven panels describe the same "
            "biological cohort. The separately requested generated E7.5 "
            "Rostral-neurectoderm subset is reported in its own plot."
        ),
    }
    (
        args.output_dir / "run_metadata.json"
    ).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    selected = fine_frame[
        fine_frame["step"].isin([0, *STAGE_STEPS])
    ].copy()
    selected.to_csv(
        args.output_dir
        / "trajectorynet_native_base_fine_composition_observed_stages.csv",
        index=False,
    )
    print(
        "Selected native-base E7.5 cohorts:",
        {
            cohort: int(mask.sum())
            for cohort, mask in masks.items()
        },
    )
    for cohort in cohorts:
        terminal = (
            fine_frame[
                (fine_frame["cohort"] == cohort)
                & (fine_frame["step"] == 25)
            ]
            .sort_values("percent", ascending=False)
            .head(10)
        )
        print(f"\n{cohort} terminal composition:")
        print(
            terminal[["celltype", "count", "percent"]].to_string(
                index=False,
                float_format=lambda value: f"{value:.2f}",
            )
        )
    print(f"\nOutputs: {args.output_dir}")


if __name__ == "__main__":
    main()
