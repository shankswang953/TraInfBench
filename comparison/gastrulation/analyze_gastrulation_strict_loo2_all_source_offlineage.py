#!/usr/bin/env python
"""Conservative strict-LOO2 E8.5 cross-germ leakage for committed E7.5 starts."""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import json
import os
from pathlib import Path
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

from analyze_gastrulation_loo_source_lineage_fidelity import (  # noqa: E402
    canonical_celltype,
    celltype_group,
    normalized_weights,
)
from analyze_gastrulation_strict_loo1_all_source_offlineage import (  # noqa: E402
    AMBIGUOUS_TARGETS,
    ECTODERM_TARGETS,
    ENDODERM_TARGETS,
    EVIDENCE_REFERENCES,
    MESODERM_TARGETS,
    SOURCE_ORDER,
    lineage_target_sets,
)
from evaluate_gastrulation_full_cmcc import _load_references  # noqa: E402
from trainfbench_plot_style import NATURE_CUD, apply_nature_rc  # noqa: E402


SOURCE_H5AD = ROOT / "data/gastrulation_rna_loo_time2_cytobridge.h5ad"
STRICT_CACHE = (
    ROOT
    / "results/gastrulation_strict_loo_six_metrics_20k_cy0p3"
    / "strict_loo_e80_e85_predictions.npz"
)
OT_TRAJECTORY = Path(
    "external/COATI/Gastrulation/"
    "balancedLOO/trajectory_time2/"
    "primary_trajectory_balanced_loo_time2_s0_iter20000.pt"
)
OUTPUT_DIR = ROOT / "results/gastrulation_strict_loo2_conservative_cross_germ_leakage"
MIN_SOURCE_CELLS_EXCLUSIVE = 200
K = 5
HELDOUT_STAGE_INDEX = 2
HELDOUT_TIME = 2.0
COATI_CY = 0.3
METHOD_ORDER = (
    "COATI bal.",
    "COATI unbal.",
    "OT(RNA)",
    "UOT(RNA)",
    "CytoBridge bal.",
    "CytoBridge unbal.",
    "MIOFlow (GAGA10D)",
)
METHOD_TAGS = tuple(f"M{index}" for index in range(1, len(METHOD_ORDER) + 1))
METHOD_RENAMES = {
    "COATI balanced": "COATI bal.",
    "COATI unbalanced": "COATI unbal.",
    "UOT(RNA)": "UOT(RNA)",
    "CytoBridge balanced 20k": "CytoBridge bal.",
    "CytoBridge unbalanced 20k": "CytoBridge unbal.",
    "MIOFlow 20k": "MIOFlow (GAGA10D)",
}


def load_array(path: Path) -> np.ndarray:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_source() -> tuple[np.ndarray, pd.Series]:
    source = ad.read_h5ad(SOURCE_H5AD, backed="r")
    try:
        times = pd.to_numeric(
            source.obs["time_point_processed"], errors="raise"
        ).to_numpy(float)
        obs = source.obs.loc[np.isclose(times, 0.0)].copy()
    finally:
        source.file.close()
    labels = obs["celltype"].map(canonical_celltype).to_numpy(dtype=str)
    counts = pd.Series(labels).value_counts()
    if len(labels) != 9018:
        raise ValueError(f"Expected 9018 E7.5 cells, found {len(labels)}")
    return labels, counts


def load_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with np.load(STRICT_CACHE, allow_pickle=False) as cache:
        index = 0
        while f"loo2_method{index}_name" in cache.files:
            source_name = str(cache[f"loo2_method{index}_name"])
            if source_name in METHOD_RENAMES:
                log_mass = np.asarray(
                    cache[f"loo2_method{index}_log_mass"], dtype=np.float64
                ).reshape(-1)
                has_mass = bool(np.all(np.isfinite(log_mass)))
                records.append(
                    {
                        "method": METHOD_RENAMES[source_name],
                        "rna": np.asarray(
                            cache[f"loo2_method{index}_rna"], dtype=np.float32
                        ),
                        "log_mass": (
                            log_mass
                            if has_mass
                            else np.zeros(len(log_mass), dtype=np.float64)
                        ),
                        "has_mass": has_mass,
                        "provenance": str(STRICT_CACHE),
                    }
                )
            index += 1

    # The newly completed balanced RNA-only LOO2 checkpoint was exported after
    # the shared benchmark cache was built. Its trajectory has dt=0.1 and the
    # held-out E8.5 frame is therefore t=2.0 / dt = 20.
    ot_path = load_array(OT_TRAJECTORY)
    if ot_path.ndim != 3 or ot_path.shape[1:] != (9018, 50):
        raise ValueError(f"Unexpected OT(RNA) trajectory shape: {ot_path.shape}")
    records.append(
        {
            "method": "OT(RNA)",
            "rna": ot_path[20].astype(np.float32, copy=False),
            "log_mass": np.zeros(9018, dtype=np.float64),
            "has_mass": False,
            "provenance": str(OT_TRAJECTORY),
        }
    )
    by_method = {str(record["method"]): record for record in records}
    missing = set(METHOD_ORDER) - set(by_method)
    if missing:
        raise ValueError(f"Missing LOO2 methods: {sorted(missing)}")
    return [by_method[method] for method in METHOD_ORDER]


def analyze() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[dict[str, object]]]:
    source_labels, source_counts = load_source()
    selected_celltypes = [
        celltype
        for celltype in SOURCE_ORDER
        if int(source_counts.get(celltype, 0)) > MIN_SOURCE_CELLS_EXCLUSIVE
    ]
    missing_sources = [
        celltype for celltype in SOURCE_ORDER if celltype not in selected_celltypes
    ]
    if missing_sources:
        raise ValueError(
            "Evidence-supported source states missing or below the cell threshold: "
            f"{missing_sources}"
        )

    rna_references, _, labels_by_stage, _, _ = _load_references()
    reference = np.asarray(rna_references[HELDOUT_STAGE_INDEX], dtype=np.float32)
    target_labels = np.asarray(
        [
            canonical_celltype(value)
            for value in labels_by_stage[HELDOUT_STAGE_INDEX]
        ],
        dtype=str,
    )
    neighbours = NearestNeighbors(
        n_neighbors=K,
        algorithm="brute",
        metric="euclidean",
        n_jobs=-1,
    ).fit(reference)

    rows: list[dict[str, object]] = []
    records = load_records()
    for record in records:
        method = str(record["method"])
        prediction = np.asarray(record["rna"], dtype=np.float32)
        log_mass = np.asarray(record["log_mass"], dtype=np.float64)
        native_mass = bool(record["has_mass"])
        if len(prediction) != len(source_labels):
            raise ValueError(
                f"{method}: prediction/source mismatch "
                f"{len(prediction)} != {len(source_labels)}"
            )
        assigned_labels = target_labels[
            neighbours.kneighbors(prediction, return_distance=False)
        ]
        for celltype in selected_celltypes:
            mask = source_labels == celltype
            on_lineage, ambiguous, off_lineage = lineage_target_sets(celltype)
            weights = normalized_weights(log_mass[mask], native_mass)
            local_assignments = assigned_labels[mask]
            on_scores = np.mean(
                np.isin(local_assignments, sorted(on_lineage)), axis=1
            )
            off_scores = np.mean(
                np.isin(local_assignments, sorted(off_lineage)), axis=1
            )
            ambiguous_scores = np.mean(
                np.isin(local_assignments, sorted(ambiguous)), axis=1
            )
            on_mass = float(np.sum(weights * on_scores))
            off_mass = float(np.sum(weights * off_scores))
            ambiguous_mass = float(np.sum(weights * ambiguous_scores))
            classified_mass = on_mass + off_mass
            rows.append(
                {
                    "stage": "E8.5",
                    "method": method,
                    "source_celltype": celltype,
                    "source_group": celltype_group(celltype),
                    "source_n": int(mask.sum()),
                    "k": K,
                    "weighting": "native mass" if native_mass else "uniform",
                    "on_lineage_target_celltypes": "; ".join(sorted(on_lineage)),
                    "ambiguous_target_celltypes": "; ".join(sorted(ambiguous)),
                    "off_lineage_target_celltypes": "; ".join(sorted(off_lineage)),
                    "estimable": True,
                    "on_lineage_mass": on_mass,
                    "off_lineage_mass": off_mass,
                    "ambiguous_mass": ambiguous_mass,
                    "classified_mass": classified_mass,
                    "conditional_off_lineage_fraction": (
                        off_mass / classified_mass if classified_mass > 0 else np.nan
                    ),
                }
            )

    detailed = pd.DataFrame(rows)
    mean_rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        local = detailed[detailed["method"].eq(method)].copy()
        denominator = int(local["source_n"].sum())
        mean_rows.append(
            {
                "stage": "E8.5",
                "method": method,
                "n_evaluable_source_celltypes": int(len(local)),
                "evaluable_source_cells": denominator,
                "cell_number_weighted_off_lineage_mass": float(
                    np.sum(
                        local["source_n"].to_numpy(float)
                        * local["off_lineage_mass"].to_numpy(float)
                    )
                    / denominator
                ),
                "cell_number_weighted_ambiguous_mass": float(
                    np.sum(
                        local["source_n"].to_numpy(float)
                        * local["ambiguous_mass"].to_numpy(float)
                    )
                    / denominator
                ),
                "cell_number_weighted_conditional_off_lineage_fraction": float(
                    np.sum(
                        local["source_n"].to_numpy(float)
                        * local["conditional_off_lineage_fraction"].to_numpy(float)
                    )
                    / denominator
                ),
            }
        )
    return detailed, pd.DataFrame(mean_rows), selected_celltypes, records


def plot(
    detailed: pd.DataFrame,
    means: pd.DataFrame,
    selected_celltypes: list[str],
) -> None:
    apply_nature_rc(font_size=10.0)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "axes.titlesize": 12.0,
            "axes.labelsize": 12.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.0,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    matrix = (
        detailed.pivot(
            index="source_celltype", columns="method", values="off_lineage_mass"
        )
        .loc[selected_celltypes, list(METHOD_ORDER)]
        .to_numpy(float)
    )
    weighted = (
        means.set_index("method")
        .loc[list(METHOD_ORDER), "cell_number_weighted_off_lineage_mass"]
        .to_numpy(float)
    )
    plot_matrix = np.vstack([matrix, weighted[None, :]]) * 100.0
    cmap = LinearSegmentedColormap.from_list(
        "off_lineage", ["#FFFFFF", "#F6D2C8", NATURE_CUD["vermillion"]]
    ).copy()
    vmax = float(max(10.0, np.ceil(np.nanmax(plot_matrix) / 10.0) * 10.0))
    fig, ax = plt.subplots(figsize=(6.65, 4.85))
    image = ax.imshow(
        plot_matrix,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        aspect="auto",
        interpolation="none",
    )
    display_source = {
        "Haematoendothelial progenitors": "Haemato-endothelial prog."
    }
    row_labels = [
        f"{display_source.get(celltype, celltype)}  (N={int(detailed.loc[detailed['source_celltype'].eq(celltype), 'source_n'].iloc[0]):,})"
        for celltype in selected_celltypes
    ] + ["Cell-number-weighted mean"]
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_xticks(np.arange(len(METHOD_ORDER)), METHOD_TAGS)
    ax.tick_params(axis="x", length=0, pad=5, labelrotation=0)
    ax.tick_params(axis="y", length=0, pad=5)
    fig.text(
        0.5,
        0.975,
        "Held-out E8.5 conservative cross-germ-layer leakage",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="normal",
    )
    fig.text(
        0.5,
        0.935,
        "Committed E7.5 starts (N>200); broad progenitor targets are not errors",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#555555",
    )
    ax.set_xticks(np.arange(-0.5, len(METHOD_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.axhline(len(selected_celltypes) - 0.5, color="#555555", linewidth=1.0)
    for row in range(plot_matrix.shape[0]):
        for column in range(plot_matrix.shape[1]):
            value = plot_matrix[row, column]
            ax.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if value >= 0.58 * vmax else "black",
                fontweight="normal",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Cross-germ-layer leakage (%)", fontsize=12)
    colorbar.set_ticks(np.linspace(0.0, vmax, 5))
    colorbar.ax.tick_params(labelsize=10)
    colorbar.outline.set_linewidth(0.5)
    fig.text(
        0.50,
        0.066,
        "M1, COATI bal.;  M2, COATI unbal.;  M3, OT(RNA);  M4, UOT(RNA)",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )
    fig.text(
        0.50,
        0.030,
        "M5, CytoBridge bal.;  M6, CytoBridge unbal.;  M7, MIOFlow (GAGA10D)",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.37, right=0.91, top=0.875, bottom=0.135)
    stem = OUTPUT_DIR / "strict_loo2_e85_conservative_cross_germ_leakage_k5"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches=None, facecolor="white")
    fig.savefig(
        stem.with_suffix(".png"), dpi=600, bbox_inches=None, facecolor="white"
    )
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detailed, means, selected_celltypes, records = analyze()
    detailed_path = OUTPUT_DIR / "strict_loo2_e85_conservative_cross_germ_leakage_k5.csv"
    means_path = OUTPUT_DIR / "strict_loo2_e85_conservative_cross_germ_weighted_mean_k5.csv"
    detailed.to_csv(detailed_path, index=False)
    means.to_csv(means_path, index=False)

    definition_rows: list[dict[str, object]] = []
    for celltype in selected_celltypes:
        on_lineage, ambiguous, off_lineage = lineage_target_sets(celltype)
        definition_rows.append(
            {
                "source_celltype": celltype,
                "source_n": int(
                    detailed.loc[
                        detailed["source_celltype"].eq(celltype), "source_n"
                    ].iloc[0]
                ),
                "evidence_grade": "Strong",
                "evidence_references": EVIDENCE_REFERENCES[celltype],
                "on_lineage_target_celltypes": "; ".join(sorted(on_lineage)),
                "ambiguous_excluded_target_celltypes": "; ".join(
                    sorted(ambiguous)
                ),
                "off_lineage_target_celltypes": "; ".join(sorted(off_lineage)),
            }
        )
    definition_path = OUTPUT_DIR / "conservative_cross_germ_definition.csv"
    pd.DataFrame(definition_rows).to_csv(definition_path, index=False)

    compact_rules = pd.DataFrame(
        [
            {
                "source_compartment": "Ectoderm",
                "included_sources": "Rostral neurectoderm; Surface ectoderm",
                "compatible_targets": "Any neural/ectoderm state",
                "unresolved_not_error": "Epiblast/primitive-streak states; NMP; PGC; Unannotated",
                "counted_as_error": "Mesoderm/haemato-endothelial or endoderm",
            },
            {
                "source_compartment": "Mesoderm/haemato-endothelial",
                "included_sources": "Paraxial mesoderm; Somitic mesoderm; Pharyngeal mesoderm; Haemato-endothelial progenitors",
                "compatible_targets": "Any mesoderm/haemato-endothelial state",
                "unresolved_not_error": "Epiblast/primitive-streak states; NMP; PGC; Unannotated",
                "counted_as_error": "Neural/ectoderm or endoderm",
            },
            {
                "source_compartment": "Endoderm",
                "included_sources": "Definitive endoderm; Gut",
                "compatible_targets": "Definitive endoderm or gut",
                "unresolved_not_error": "Epiblast/primitive-streak states; NMP; PGC; Unannotated",
                "counted_as_error": "Neural/ectoderm or mesoderm/haemato-endothelial",
            },
        ]
    )
    compact_rule_path = OUTPUT_DIR / "conservative_cross_germ_rule_table.csv"
    compact_rules.to_csv(compact_rule_path, index=False)
    plot(detailed, means, selected_celltypes)

    manifest = {
        "analysis": (
            "Strict-LOO2 conservative cross-germ-layer leakage from committed "
            "E7.5 source states to held-out E8.5"
        ),
        "source_cell_threshold": f"> {MIN_SOURCE_CELLS_EXCLUSIVE}",
        "selected_source_celltypes": selected_celltypes,
        "source_evidence_threshold": "strong; committed source states only",
        "k": K,
        "space": "common normalized RNA PCA50",
        "readout": "soft 5-NN against the real held-out E8.5 RNA atlas",
        "off_lineage_definition": (
            "Only unambiguous transitions between ectoderm, mesoderm (including "
            "haemato-endothelial), and endoderm compartments count as errors. "
            "Targets in the source germ layer are compatible. Epiblast, "
            "primitive-streak states, caudal epiblast, PGC, NMP, and Unannotated "
            "are unresolved and never count as errors. The denominator is the "
            "complete predicted mass."
        ),
        "coati_cy": COATI_CY,
        "coati_cy_note": (
            "C_y=0.4 is the shared strict-LOO2 20k setting; unbalanced C_y=0.3 "
            "does not have a 20k checkpoint."
        ),
        "weighted_mean": (
            "source-cell-number-weighted mean across committed, strongly "
            "supported source cell types with >200 E7.5 cells"
        ),
        "mass_weighting": (
            "native predicted mass normalized within each source cell type for "
            "unbalanced methods; uniform within source cell type for balanced methods"
        ),
        "methods": [
            {
                "name": str(record["method"]),
                "provenance": str(record["provenance"]),
            }
            for record in records
        ],
        "excluded_methods": {
            "TrajectoryNet": "Gaussian/terminal particles lack matched E7.5 source-cell identity",
            "TIGON": "excluded following the current figure specification",
        },
        "target_sets": {
            "ectoderm": sorted(ECTODERM_TARGETS),
            "mesoderm_haemato_endothelial": sorted(MESODERM_TARGETS),
            "endoderm": sorted(ENDODERM_TARGETS),
            "ambiguous": sorted(AMBIGUOUS_TARGETS),
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(detailed_path)
    print(means_path)
    print(definition_path)
    print(compact_rule_path)
    print(OUTPUT_DIR / "strict_loo2_e85_conservative_cross_germ_leakage_k5.pdf")
    print(OUTPUT_DIR / "strict_loo2_e85_conservative_cross_germ_leakage_k5.png")
    print("\nCell-number-weighted conservative cross-germ-layer leakage (%):")
    display = means[["method", "cell_number_weighted_off_lineage_mass"]].copy()
    display["cell_number_weighted_off_lineage_mass"] *= 100.0
    print(display.to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
