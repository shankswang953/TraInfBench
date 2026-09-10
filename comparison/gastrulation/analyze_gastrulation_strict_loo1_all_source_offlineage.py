#!/usr/bin/env python
"""Conservative strict-LOO1 E8.0 cross-germ leakage for committed E7.5 starts."""

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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

from analyze_gastrulation_loo_source_lineage_fidelity import (  # noqa: E402
    canonical_celltype,
    celltype_group,
    normalized_weights,
)
from analyze_gastrulation_strict_loo1_rostral_endpoint_offlineage import (  # noqa: E402
    SOURCE_H5AD,
    load_forward_records,
)
from evaluate_gastrulation_full_cmcc import _load_references  # noqa: E402
from trainfbench_plot_style import (  # noqa: E402
    GASTRULATION_CELLTYPE_GROUPS,
    NATURE_CUD,
    apply_nature_rc,
)


OUTPUT_DIR = ROOT / "results/gastrulation_strict_loo1_conservative_cross_germ_leakage"
MIN_SOURCE_CELLS_EXCLUSIVE = 200
K = 5
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

# The primary figure uses only committed source states with direct in-vivo
# fate-mapping, clonal-analysis, or genetic-lineage support. Broad or
# multipotent source annotations are excluded from the primary aggregate.
SOURCE_ORDER = (
    "Rostral neurectoderm",
    "Surface ectoderm",
    "Paraxial mesoderm",
    "Somitic mesoderm",
    "Pharyngeal mesoderm",
    "Haematoendothelial progenitors",
    "Def. endoderm",
    "Gut",
)

EVIDENCE_REFERENCES = {
    "Rostral neurectoderm": "Cajal et al. 2012, PMCID: PMC3243100",
    "Surface ectoderm": "Petit and Nicolas 2009, PMID: 19197371",
    "Paraxial mesoderm": "Guibentif et al. 2021, PMID: 33308481",
    "Somitic mesoderm": "Guibentif et al. 2021, PMID: 33308481",
    "Pharyngeal mesoderm": "Lescroart et al. 2015, PMID: 25605943",
    "Haematoendothelial progenitors": (
        "Wang et al. 2022, PMID: 35587592; Zovein et al. 2008, PMID: 19041779"
    ),
    "Def. endoderm": (
        "Probst et al. 2021, DOI: 10.1242/dev.193789; "
        "Nowotschin et al. 2019, PMID: 30959515"
    ),
    "Gut": "Nowotschin et al. 2019, PMID: 30959515",
}

ALL_TARGET_CELLTYPES = set().union(*map(set, GASTRULATION_CELLTYPE_GROUPS.values()))

# The most conservative primary endpoint resolves only cross-germ-layer
# incompatibility. NMP and the atlas's broad/multipotent progenitor labels are
# never called errors in this figure, regardless of source state.
AMBIGUOUS_TARGETS = (
    set(GASTRULATION_CELLTYPE_GROUPS["Progenitor / primitive streak"])
    | {"NMP", "Unannotated"}
)
ECTODERM_TARGETS = set(GASTRULATION_CELLTYPE_GROUPS["Neural / ectoderm"])
MESODERM_TARGETS = (
    set(GASTRULATION_CELLTYPE_GROUPS["Axial / paraxial mesoderm"])
    | set(GASTRULATION_CELLTYPE_GROUPS["Other mesoderm"])
    | set(GASTRULATION_CELLTYPE_GROUPS["Haemato-endothelial"])
) - {"NMP"}
ENDODERM_TARGETS = set(GASTRULATION_CELLTYPE_GROUPS["Endoderm"])

ECTODERM_SOURCES = {"Rostral neurectoderm", "Surface ectoderm"}
MESODERM_SOURCES = {
    "Paraxial mesoderm",
    "Somitic mesoderm",
    "Pharyngeal mesoderm",
    "Haematoendothelial progenitors",
}
ENDODERM_SOURCES = {"Def. endoderm", "Gut"}


def lineage_target_sets(source_celltype: str) -> tuple[set[str], set[str], set[str]]:
    """Return germ-layer-compatible, ambiguous, and incompatible targets."""

    if source_celltype in ECTODERM_SOURCES:
        on_lineage = set(ECTODERM_TARGETS)
    elif source_celltype in MESODERM_SOURCES:
        on_lineage = set(MESODERM_TARGETS)
    elif source_celltype in ENDODERM_SOURCES:
        on_lineage = set(ENDODERM_TARGETS)
    else:
        raise KeyError(f"No conservative source definition for {source_celltype}")
    ambiguous = set(AMBIGUOUS_TARGETS)
    off_lineage = ALL_TARGET_CELLTYPES - on_lineage - ambiguous
    return on_lineage, ambiguous, off_lineage


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


def analyze() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    source_labels, source_counts = load_source()
    selected_celltypes = [
        celltype
        for celltype in SOURCE_ORDER
        if int(source_counts.get(celltype, 0)) > MIN_SOURCE_CELLS_EXCLUSIVE
    ]
    missing = [celltype for celltype in SOURCE_ORDER if celltype not in selected_celltypes]
    if missing:
        raise ValueError(
            "Evidence-supported source states missing or below the cell threshold: "
            f"{missing}"
        )

    rna_references, _, labels_by_stage, _, _ = _load_references()
    reference = np.asarray(rna_references[1], dtype=np.float32)
    target_labels = np.asarray(
        [canonical_celltype(value) for value in labels_by_stage[1]], dtype=str
    )
    neighbours = NearestNeighbors(
        n_neighbors=K,
        algorithm="brute",
        metric="euclidean",
        n_jobs=-1,
    ).fit(reference)

    rows: list[dict[str, object]] = []
    for record in load_forward_records():
        method = str(record["display"])
        if method not in METHOD_ORDER:
            continue
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
            classified_scores = on_scores + off_scores
            classified_mass = float(np.sum(weights * classified_scores))
            if classified_mass <= 0.0:
                raise ValueError(f"{method}, {celltype}: no classified 5-NN mass")
            off_mass = float(np.sum(weights * off_scores))
            on_mass = float(np.sum(weights * on_scores))
            ambiguous_mass = float(np.sum(weights * ambiguous_scores))
            conditional_off_fraction = off_mass / classified_mass
            rows.append(
                {
                    "stage": "E8.0",
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
                    "conditional_off_lineage_fraction": conditional_off_fraction,
                }
            )

    detailed = pd.DataFrame(rows)
    mean_rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        local = detailed[
            detailed["method"].eq(method) & detailed["estimable"]
        ].copy()
        denominator = int(local["source_n"].sum())
        mean_rows.append(
            {
                "stage": "E8.0",
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
    means = pd.DataFrame(mean_rows)
    return detailed, means, selected_celltypes


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
            index="source_celltype",
            columns="method",
            values="off_lineage_mass",
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
    masked = np.ma.masked_invalid(plot_matrix)

    cmap = LinearSegmentedColormap.from_list(
        "off_lineage",
        ["#FFFFFF", "#F6D2C8", NATURE_CUD["vermillion"]],
    ).copy()
    cmap.set_bad("#E6E6E6")

    vmax = float(max(10.0, np.ceil(np.nanmax(plot_matrix) / 10.0) * 10.0))
    fig, ax = plt.subplots(figsize=(6.65, 4.85))
    image = ax.imshow(
        masked,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        aspect="auto",
        interpolation="none",
    )

    display_source = {
        "Haematoendothelial progenitors": "Haemato-endothelial prog.",
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
        "Held-out E8.0 conservative cross-germ-layer leakage",
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

    # White cell borders and a stronger separator above the weighted mean.
    ax.set_xticks(np.arange(-0.5, len(METHOD_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.axhline(len(selected_celltypes) - 0.5, color="#555555", linewidth=1.0)

    for row in range(plot_matrix.shape[0]):
        for column in range(plot_matrix.shape[1]):
            value = plot_matrix[row, column]
            if not np.isfinite(value):
                label = "—"
                color = "#666666"
            else:
                label = f"{value:.1f}"
                color = "white" if value >= 0.58 * vmax else "black"
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=10,
                color=color,
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

    stem = OUTPUT_DIR / "strict_loo1_e80_conservative_cross_germ_leakage_k5"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches=None, facecolor="white")
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=600,
        bbox_inches=None,
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detailed, means, selected_celltypes = analyze()
    detailed_path = OUTPUT_DIR / "strict_loo1_e80_conservative_cross_germ_leakage_k5.csv"
    means_path = OUTPUT_DIR / "strict_loo1_e80_conservative_cross_germ_weighted_mean_k5.csv"
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
                "ambiguous_excluded_target_celltypes": "; ".join(sorted(ambiguous)),
                "off_lineage_target_celltypes": "; ".join(sorted(off_lineage)),
            }
        )
    definition_path = OUTPUT_DIR / "conservative_cross_germ_definition.csv"
    pd.DataFrame(definition_rows).to_csv(definition_path, index=False)
    plot(detailed, means, selected_celltypes)

    manifest = {
        "analysis": (
            "Strict-LOO1 conservative cross-germ-layer leakage from committed "
            "E7.5 source states to held-out E8.0"
        ),
        "source_cell_threshold": f"> {MIN_SOURCE_CELLS_EXCLUSIVE}",
        "selected_source_celltypes": selected_celltypes,
        "source_evidence_threshold": "strong; committed source states only",
        "k": K,
        "space": "common normalized RNA PCA50",
        "readout": "soft 5-NN against the real held-out E8.0 RNA atlas",
        "off_lineage_definition": (
            "Only unambiguous transitions between ectoderm, mesoderm (including "
            "haemato-endothelial), and endoderm compartments count as errors. "
            "All targets in the source germ layer are compatible. Epiblast, "
            "primitive-streak states, caudal epiblast, PGC, NMP, and Unannotated "
            "are ambiguous for every source and never count as errors. The "
            "denominator is the complete predicted mass."
        ),
        "weighted_mean": (
            "source-cell-number-weighted mean across committed, strongly "
            "supported source cell types with >200 E7.5 cells"
        ),
        "mass_weighting": (
            "native predicted mass normalized within each source cell type for "
            "unbalanced methods; uniform within source cell type for balanced methods"
        ),
        "methods": list(METHOD_ORDER),
        "excluded_methods": {
            "TrajectoryNet": "terminal/base particles lack matched E7.5 source-cell identity",
            "TIGON": "excluded following the current figure specification",
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(detailed_path)
    print(means_path)
    print(definition_path)
    print(OUTPUT_DIR / "strict_loo1_e80_conservative_cross_germ_leakage_k5.pdf")
    print(OUTPUT_DIR / "strict_loo1_e80_conservative_cross_germ_leakage_k5.png")
    print("\nCell-number-weighted conservative cross-germ-layer leakage (%):")
    display = means[["method", "cell_number_weighted_off_lineage_mass"]].copy()
    display["cell_number_weighted_off_lineage_mass"] *= 100.0
    print(display.to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
