#!/usr/bin/env python
"""Plot actual held-out RNA/ATAC cell-type compositions for selected methods."""

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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from evaluate_gastrulation_full_cmcc import _load_references
from evaluate_gastrulation_loo_time1_paired_1nn_atac import (
    _load_separate_modality_labels,
)
from trainfbench_plot_style import (
    GASTRULATION_CELLTYPE_ALIASES,
    GASTRULATION_CELLTYPE_GROUPS,
    apply_nature_rc,
    gastrulation_celltype_color,
)


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "results/gastrulation_loo_shared_t_trajectorynet_base"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/gastrulation_loo_composition_selected_trajectorynet_base"
)

METHODS = (
    ("Observed", "Observed"),
    ("BSOT C_y=0.3", "COATI bal"),
    ("USOT C_y=0.3", "COATI unbal"),
    ("CytoBridge balanced 20k", "CytoBridge bal"),
    ("CytoBridge unbalanced 20k", "CytoBridge unbal"),
    ("MIOFlow 20k", "MIOFlow"),
    ("TIGON 20k", "TIGON"),
    ("TrajectoryNet 20k", "TrajectoryNet"),
)

GROUP_ORDER = tuple(GASTRULATION_CELLTYPE_GROUPS)
GROUP_LABELS = {
    "Progenitor / primitive streak": "Progenitor / PS",
    "Neural / ectoderm": "Neural / ectoderm",
    "Axial / paraxial mesoderm": "Axial / paraxial",
    "Other mesoderm": "Other mesoderm",
    "Haemato-endothelial": "Haemato-endothelial",
    "Endoderm": "Endoderm",
    "Other / missing": "Other / missing",
}
GROUP_COLORS = {
    "Progenitor / primitive streak": gastrulation_celltype_color("Epiblast"),
    "Neural / ectoderm": gastrulation_celltype_color(
        "Forebrain/Midbrain/Hindbrain"
    ),
    "Axial / paraxial mesoderm": gastrulation_celltype_color("NMP"),
    "Other mesoderm": gastrulation_celltype_color("Mesenchyme"),
    "Haemato-endothelial": gastrulation_celltype_color("Endothelium"),
    "Endoderm": gastrulation_celltype_color("Gut"),
    "Other / missing": gastrulation_celltype_color("Unannotated"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def lineage_group(celltype: str) -> str:
    canonical = GASTRULATION_CELLTYPE_ALIASES.get(celltype, celltype)
    for group, members in GASTRULATION_CELLTYPE_GROUPS.items():
        if canonical in members:
            return group
    raise KeyError(f"Cell type is not registered in a lineage group: {celltype}")


def prediction_composition(
    prediction: np.ndarray,
    reference: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    indices = (
        NearestNeighbors(n_neighbors=1, n_jobs=-1)
        .fit(np.asarray(reference, dtype=np.float32))
        .kneighbors(
            np.asarray(prediction, dtype=np.float32),
            return_distance=False,
        )[:, 0]
    )
    assigned = np.asarray(labels, dtype=str)[indices]
    values, counts = np.unique(assigned, return_counts=True)
    total = float(counts.sum())
    return {
        str(celltype): float(count / total)
        for celltype, count in zip(values, counts)
    }


def selected_full_composition(
    *,
    stage_index: int,
    stage: str,
    rna_reference: np.ndarray,
    atac_reference: np.ndarray,
    rna_labels: np.ndarray,
    atac_labels: np.ndarray,
) -> pd.DataFrame:
    source_path = (
        ROOT
        / f"results/gastrulation_loo_time{stage_index}_shared_t_atac"
        / f"loo_time{stage_index}_shared_t_celltype_composition.csv"
    )
    source = pd.read_csv(source_path)
    rows: list[dict[str, object]] = []

    observed_source = source[
        source["method"].eq("BSOT C_y=0.3")
    ]
    for modality in ("RNA", "ATAC"):
        observed = observed_source[
            observed_source["modality"].eq(modality)
        ]
        for row in observed.itertuples(index=False):
            rows.append(
                {
                    "stage": stage,
                    "modality": modality,
                    "method": "Observed",
                    "display_name": "Observed",
                    "celltype": str(row.celltype),
                    "fraction": float(row.observed_fraction),
                    "origin": "held-out reference",
                }
            )

    for source_method, display_name in METHODS[1:-1]:
        selected = source[source["method"].eq(source_method)]
        for row in selected.itertuples(index=False):
            rows.append(
                {
                    "stage": stage,
                    "modality": str(row.modality),
                    "method": source_method,
                    "display_name": display_name,
                    "celltype": str(row.celltype),
                    "fraction": float(row.predicted_fraction),
                    "origin": "existing native prediction",
                }
            )

    base_path = (
        BASE_DIR
        / f"loo_time{stage_index}_trajectorynet_base_predictions.npz"
    )
    with np.load(base_path, allow_pickle=True) as base:
        particle_origin = str(np.asarray(base["particle_origin"]).item())
        if particle_origin != "standard_normal_base":
            raise ValueError(
                "TrajectoryNet prediction is not base-origin: "
                f"{particle_origin}"
            )
        predictions = {
            "RNA": np.asarray(base["prediction_rna_norm"], dtype=np.float32),
            "ATAC": np.asarray(base["prediction_atac_norm"], dtype=np.float32),
        }
    for modality, reference, labels in (
        ("RNA", rna_reference, rna_labels),
        ("ATAC", atac_reference, atac_labels),
    ):
        composition = prediction_composition(
            predictions[modality],
            reference,
            labels,
        )
        all_celltypes = sorted(
            set(np.asarray(labels, dtype=str).tolist())
            | set(composition)
        )
        for celltype in all_celltypes:
            rows.append(
                {
                    "stage": stage,
                    "modality": modality,
                    "method": "TrajectoryNet 20k",
                    "display_name": "TrajectoryNet",
                    "celltype": celltype,
                    "fraction": float(composition.get(celltype, 0.0)),
                    "origin": "standard-normal base prediction",
                }
            )

    result = pd.DataFrame(rows)
    expected = {display for _, display in METHODS}
    found = set(result["display_name"])
    if found != expected:
        raise ValueError(
            f"Composition methods mismatch; expected {expected}, found {found}"
        )
    return result


def grouped_composition(full: pd.DataFrame) -> pd.DataFrame:
    grouped = full.copy()
    grouped["lineage_group"] = grouped["celltype"].map(lineage_group)
    grouped = (
        grouped.groupby(
            [
                "stage",
                "modality",
                "method",
                "display_name",
                "lineage_group",
                "origin",
            ],
            as_index=False,
            sort=False,
        )["fraction"]
        .sum()
        .rename(columns={"fraction": "group_fraction"})
    )
    totals = grouped.groupby(
        ["stage", "modality", "display_name"]
    )["group_fraction"].transform("sum")
    grouped["group_fraction"] /= totals
    return grouped


def plot_stage(
    grouped: pd.DataFrame,
    *,
    stage: str,
    stage_slug: str,
    output_dir: Path,
) -> None:
    display_order = [display for _, display in METHODS]
    y = np.arange(len(display_order))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(3.55, 3.35),
        sharey=True,
        facecolor="white",
    )
    for column, (ax, modality) in enumerate(zip(axes, ("RNA", "ATAC"))):
        subset = grouped[
            grouped["stage"].eq(stage)
            & grouped["modality"].eq(modality)
        ]
        left = np.zeros(len(display_order), dtype=float)
        for group in GROUP_ORDER:
            values = np.asarray(
                [
                    subset[
                        subset["display_name"].eq(display)
                        & subset["lineage_group"].eq(group)
                    ]["group_fraction"].sum()
                    for display in display_order
                ],
                dtype=float,
            )
            ax.barh(
                y,
                values,
                left=left,
                height=0.68,
                color=GROUP_COLORS[group],
                edgecolor="white",
                linewidth=0.35,
            )
            left += values
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(len(display_order) - 0.35, -0.65)
        ax.set_title(modality, fontsize=12, fontweight="normal", pad=3)
        ax.set_xticks((0.0, 0.5, 1.0), ("0", "50%", "100%"))
        tick_labels = ax.get_xticklabels()
        tick_labels[0].set_horizontalalignment("left")
        tick_labels[-1].set_horizontalalignment("right")
        ax.tick_params(axis="x", labelsize=10, width=0.7, length=2.5)
        ax.tick_params(axis="y", labelsize=10, length=0, pad=3)
        if column == 0:
            ax.set_yticks(y, display_order)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.8)

    handles = [
        Patch(
            facecolor=GROUP_COLORS[group],
            edgecolor="none",
            label=GROUP_LABELS[group],
        )
        for group in GROUP_ORDER
    ]
    fig.suptitle(
        f"Held-out {stage} cell-type composition",
        fontsize=12,
        fontweight="normal",
        y=0.995,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handlelength=1.1,
        handletextpad=0.35,
        columnspacing=0.8,
        labelspacing=0.25,
    )
    fig.subplots_adjust(
        left=0.31,
        right=0.995,
        top=0.61,
        bottom=0.11,
        wspace=0.10,
    )
    stem = (
        output_dir
        / f"gastrulation_loo_{stage_slug}_celltype_composition_trajectorynet_base"
    )
    save_options = {
        "facecolor": "white",
        "bbox_inches": "tight",
        "pad_inches": 0.02,
    }
    fig.savefig(stem.with_suffix(".png"), dpi=600, **save_options)
    fig.savefig(stem.with_suffix(".pdf"), **save_options)
    fig.savefig(stem.with_suffix(".svg"), **save_options)
    plt.close(fig)


def plot_combined(
    grouped: pd.DataFrame,
    *,
    output_dir: Path,
) -> None:
    display_order = [display for _, display in METHODS]
    y = np.arange(len(display_order))
    fig = plt.figure(figsize=(7.1, 3.25), facecolor="white")
    outer = fig.add_gridspec(
        1,
        2,
        left=0.18,
        right=0.995,
        top=0.57,
        bottom=0.12,
        wspace=0.22,
    )
    bar_axes: list[plt.Axes] = []
    for stage_column in range(2):
        inner = outer[0, stage_column].subgridspec(
            1,
            2,
            wspace=0.10,
        )
        bar_axes.extend(
            [
                fig.add_subplot(inner[0, 0]),
                fig.add_subplot(inner[0, 1]),
            ]
        )

    for panel_index, (ax, stage, modality) in enumerate(
        zip(
            bar_axes,
            ("E8.0", "E8.0", "E8.5", "E8.5"),
            ("RNA", "ATAC", "RNA", "ATAC"),
        )
    ):
        subset = grouped[
            grouped["stage"].eq(stage)
            & grouped["modality"].eq(modality)
        ]
        left = np.zeros(len(display_order), dtype=float)
        for group in GROUP_ORDER:
            values = np.asarray(
                [
                    subset[
                        subset["display_name"].eq(display)
                        & subset["lineage_group"].eq(group)
                    ]["group_fraction"].sum()
                    for display in display_order
                ],
                dtype=float,
            )
            ax.barh(
                y,
                values,
                left=left,
                height=0.68,
                color=GROUP_COLORS[group],
                edgecolor="white",
                linewidth=0.35,
            )
            left += values
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(len(display_order) - 0.35, -0.65)
        ax.set_title(
            modality,
            fontsize=12,
            fontweight="normal",
            pad=3,
        )
        ax.set_xticks((0.0, 0.5, 1.0), ("0", "50%", "100%"))
        tick_labels = ax.get_xticklabels()
        tick_labels[0].set_horizontalalignment("left")
        tick_labels[-1].set_horizontalalignment("right")
        ax.tick_params(axis="x", labelsize=10, width=0.7, length=2.5)
        ax.tick_params(axis="y", labelsize=10, length=0, pad=3)
        if panel_index == 0:
            ax.set_yticks(y, display_order)
        else:
            ax.set_yticks(y, ())
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.8)

    fig.text(
        0.36,
        0.655,
        "Held-out E8.0 cell-type composition",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="normal",
    )
    fig.text(
        0.80,
        0.655,
        "Held-out E8.5 cell-type composition",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="normal",
    )
    handles = [
        Patch(
            facecolor=GROUP_COLORS[group],
            edgecolor="none",
            label=GROUP_LABELS[group],
        )
        for group in GROUP_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
        fontsize=9.0,
        handlelength=1.1,
        handletextpad=0.35,
        columnspacing=1.0,
        labelspacing=0.35,
    )
    stem = (
        output_dir
        / "gastrulation_loo_e80_e85_celltype_composition_trajectorynet_base"
    )
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        output_dir
        / f"gastrulation_loo_{stage_slug}_celltype_composition_trajectorynet_base.{extension}"
        for stage_slug in ("e80", "e85")
        for extension in ("png", "pdf", "svg")
    ]
    expected_outputs.extend(
        [
            output_dir
            / (
                "gastrulation_loo_e80_e85_celltype_composition_"
                f"trajectorynet_base.{extension}"
            )
            for extension in ("png", "pdf", "svg")
        ]
    )
    expected_outputs.extend(
        [
            output_dir / "gastrulation_loo_celltype_composition_full.csv",
            output_dir / "gastrulation_loo_lineage_group_composition.csv",
        ]
    )
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs; pass --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )

    apply_nature_rc(font_size=10.0)
    mpl.rcParams.update(
        {
            "axes.titlesize": 12.0,
            "axes.labelsize": 12.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.0,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "figure.titleweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    rna_ref, atac_ref, shared_labels, _, _ = _load_references()
    rna_labels, atac_labels, _ = _load_separate_modality_labels(
        shared_labels
    )
    full_frames: list[pd.DataFrame] = []
    for stage_index, stage in ((1, "E8.0"), (2, "E8.5")):
        full_frames.append(
            selected_full_composition(
                stage_index=stage_index,
                stage=stage,
                rna_reference=rna_ref[stage_index],
                atac_reference=atac_ref[stage_index],
                rna_labels=rna_labels[stage_index],
                atac_labels=atac_labels[stage_index],
            )
        )
    full = pd.concat(full_frames, ignore_index=True)
    grouped = grouped_composition(full)
    full.to_csv(
        output_dir / "gastrulation_loo_celltype_composition_full.csv",
        index=False,
    )
    grouped.to_csv(
        output_dir / "gastrulation_loo_lineage_group_composition.csv",
        index=False,
    )
    plot_stage(
        grouped,
        stage="E8.0",
        stage_slug="e80",
        output_dir=output_dir,
    )
    plot_stage(
        grouped,
        stage="E8.5",
        stage_slug="e85",
        output_dir=output_dir,
    )
    plot_combined(
        grouped,
        output_dir=output_dir,
    )
    for path in expected_outputs:
        print(path)


if __name__ == "__main__":
    main()
