#!/usr/bin/env python
"""Compare conservative source-resolved off-lineage leakage across methods."""

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

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

from analyze_gastrulation_loo_source_lineage_fidelity import (  # noqa: E402
    canonical_celltype,
    normalized_weights,
)
from analyze_gastrulation_strict_loo1_all_source_offlineage import (  # noqa: E402
    ALL_TARGET_CELLTYPES,
    AMBIGUOUS_TARGETS,
    ECTODERM_TARGETS,
    ENDODERM_TARGETS,
    MESODERM_TARGETS,
    load_source as load_loo1_source,
)
from analyze_gastrulation_strict_loo1_rostral_endpoint_offlineage import (  # noqa: E402
    load_forward_records as load_loo1_records,
)
from analyze_gastrulation_strict_loo2_all_source_offlineage import (  # noqa: E402
    METHOD_ORDER as CORE_METHOD_ORDER,
    load_records as load_loo2_records,
    load_source as load_loo2_source,
)
from evaluate_gastrulation_full_cmcc import _load_references  # noqa: E402
from trainfbench_plot_style import apply_nature_rc  # noqa: E402


OUTPUT_DIR = ROOT / "results/gastrulation_strict_loo_all_source_method_offlineage"
K = int(os.environ.get("KNN_K", "5"))
if K <= 0:
    raise ValueError("KNN_K must be positive")
OUTPUT_TAG = (
    "gt200_ab_tigon_all_e75_cy0p3_counts_second_line"
    + ("" if K == 5 else f"_k{K}")
)
MIN_SOURCE_CELLS_EXCLUSIVE = 200
METHOD_ORDER = (*CORE_METHOD_ORDER, "TIGON")
TIGON_AE_CACHE = (
    ROOT
    / "results/tigon_official_ae_embeddings/"
    "gastrulation_ae10_upstream_public_exact_v2"
)
TIGON_RESULT_DIRS = {
    "E8.0": (
        ROOT
        / "results/"
        "tigon_gastrulation_loo_time1_upstream_public_exact_v2_rngexact_"
        "ae10_dopri5pack_n1024_20000_seed1"
    ),
    "E8.5": (
        ROOT
        / "results/"
        "tigon_gastrulation_loo_time2_upstream_public_exact_v2_rngexact_"
        "ae10_dopri5pack_n1024_20000_seed1"
    ),
}
TRAJECTORYNET_GAUSSIAN_CACHE = {
    "E8.0": (
        ROOT
        / "results/gastrulation_loo_rostral_gross_cross_lineage/"
        "trajectorynet_gaussian_rostral_e80.npz"
    ),
    "E8.5": (
        ROOT
        / "results/gastrulation_loo_rostral_gross_cross_lineage/"
        "trajectorynet_gaussian_rostral_e85.npz"
    ),
}
FULL_RNA_H5AD = ROOT / "data/gastrulation_rna_cytobridge.h5ad"
CAUDAL_POSTERIOR_CORRIDOR = {
    "NMP",
    "Caudal epiblast",
    "Caudal neurectoderm",
    "Caudal Mesoderm",
    "Paraxial mesoderm",
    "Somitic mesoderm",
    "Spinal cord",
}
MESENDODERM_SOURCES = {"Primitive Streak", "Anterior Primitive Streak"}
METHOD_LABELS = {
    method: ("MIOFlow" if method == "MIOFlow (GAGA10D)" else method)
    for method in METHOD_ORDER
}


def load_trajectorynet_gaussian_record(stage: str) -> dict[str, object]:
    """Load native Gaussian-forward TrajectoryNet and infer its E7.5 sources.

    The Gaussian particles are first generated to E7.5 using the strict-LOO
    model's physical clock. A particle receives a source label only when at
    least three of its five observed E7.5 neighbours agree. The same particle
    index is then followed to the held-out endpoint.
    """

    cache_path = TRAJECTORYNET_GAUSSIAN_CACHE[stage]
    with np.load(cache_path, allow_pickle=False) as cache:
        prediction = np.asarray(
            cache["heldout_prediction_rna_norm"], dtype=np.float32
        )
        neighbour_labels = np.asarray(
            cache["initial_neighbor_labels"], dtype=str
        )
    if neighbour_labels.ndim != 2 or neighbour_labels.shape[1] != 5:
        raise ValueError(
            f"{stage}: expected five E7.5 neighbours per Gaussian particle"
        )
    if len(prediction) != len(neighbour_labels):
        raise ValueError(f"{stage}: inconsistent TrajectoryNet cache lengths")

    source_labels = np.full(
        len(neighbour_labels), "Unassigned generated source", dtype="<U40"
    )
    for celltype in np.unique(neighbour_labels):
        majority = np.sum(neighbour_labels == celltype, axis=1) >= 3
        source_labels[majority] = celltype

    return {
        "method": "TrajectoryNet",
        "rna": prediction,
        "log_mass": np.zeros(len(prediction), dtype=np.float64),
        "has_mass": False,
        "source_labels": source_labels,
        "space": "shared normalized RNA PCA50",
        "initialization": (
            "standard Gaussian base; generated to E7.5 and source-labelled "
            "by 5-NN majority before propagation to the held-out stage"
        ),
        "provenance": str(cache_path),
    }

# Evidence class is intentionally separate from the downstream readout rule.
# A: literature-supported major descendant compartment.
# B: germ layer is supported, but exact descendants are broad/heterogeneous.
# M: multipotent/transitional source; excluded from single-layer leakage.
SOURCE_METADATA: dict[str, tuple[str, str]] = {
    "Rostral neurectoderm": ("A", "Ectoderm (anterior neural)"),
    "Nascent mesoderm": ("B", "Mesoderm (early/broad)"),
    "Mesenchyme": ("B", "Mesoderm-derived (broad)"),
    "Epiblast": ("M", "Pre-germ-layer/multilineage"),
    "Surface ectoderm": ("A", "Ectoderm (surface/non-neural)"),
    "Caudal epiblast": ("M", "Posterior neural-mesodermal progenitor"),
    "Haematoendothelial progenitors": (
        "A",
        "Mesoderm-derived (haemato-endothelial)",
    ),
    "Pharyngeal mesoderm": (
        "A",
        "Mesoderm (pharyngeal/cardiopharyngeal)",
    ),
    "Gut": ("A", "Endoderm (gut)"),
    "Primitive Streak": ("M", "Mesendoderm/multilineage"),
    "Somitic mesoderm": ("A", "Mesoderm (somitic)"),
    "Def. endoderm": ("A", "Endoderm (definitive)"),
    "Paraxial mesoderm": ("A", "Mesoderm (paraxial)"),
    "Mixed mesoderm": ("B", "Mesoderm (heterogeneous)"),
}


def classified_sources(source_counts: pd.Series) -> list[str]:
    """Return all >200 annotated sources in descending cell-count order."""

    sources = (
        source_counts[
            (source_counts > MIN_SOURCE_CELLS_EXCLUSIVE)
            & source_counts.index.to_series().ne("Unannotated")
        ]
        .sort_values(ascending=False)
        .index.tolist()
    )
    missing = [source for source in sources if source not in SOURCE_METADATA]
    if missing:
        raise KeyError(f"Missing A/B/M metadata for sources: {missing}")
    return sources


def analysis_sources(source_counts: pd.Series) -> list[str]:
    """Return >200 A/B sources used for off-lineage estimation."""

    return [
        source
        for source in classified_sources(source_counts)
        if SOURCE_METADATA[source][0] in {"A", "B"}
    ]


def load_tigon_native_record(stage: str) -> dict[str, object]:
    """Load source-conditioned TIGON LOO predictions in frozen native AE10."""

    result_dir = TIGON_RESULT_DIRS[stage]
    evaluation = json.loads(
        (result_dir / "selected_checkpoint_ae10_evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    if evaluation.get("selection_held_out_distribution_accessed") is not False:
        raise ValueError(f"{stage}: TIGON checkpoint selection accessed held-out data")
    prediction_path = (
        result_dir / "selected_checkpoint_ae10_all_e75_predictions.npz"
    )
    prediction_cache = np.load(
        prediction_path,
        allow_pickle=False,
    )
    time_value = 1.0 if stage == "E8.0" else 2.0
    time_tag = f"{time_value:g}".replace(".", "p")
    source_indices = np.asarray(
        prediction_cache["all_e75_source_indices"], dtype=np.int64
    )
    prediction = np.asarray(
        prediction_cache[f"all_e75_to_{time_tag}"], dtype=np.float32
    )
    log_mass = np.asarray(
        prediction_cache[f"all_e75_log_growth_to_{time_tag}"],
        dtype=np.float64,
    ).reshape(-1)

    full = ad.read_h5ad(FULL_RNA_H5AD, backed="r")
    try:
        obs_names = full.obs_names.to_numpy(dtype=str)
        times = pd.to_numeric(
            full.obs["time_point_processed"], errors="raise"
        ).to_numpy(float)
        labels = np.asarray(
            [canonical_celltype(value) for value in full.obs["celltype"]],
            dtype=str,
        )
    finally:
        full.file.close()
    cache_names = np.load(
        TIGON_AE_CACHE / "obs_names.npy", allow_pickle=False
    ).astype(str)
    if not np.array_equal(cache_names, obs_names):
        raise ValueError("Frozen TIGON AE10 cache does not match shared RNA order")
    latent = np.load(
        TIGON_AE_CACHE / "ae_latent_scaled_minus2_2.npy", mmap_mode="r"
    )
    source_global_indices = np.flatnonzero(np.isclose(times, 0.0))
    if np.max(source_indices) >= len(source_global_indices):
        raise ValueError(f"{stage}: TIGON source index exceeds E7.5 pool")
    sampled_global_indices = source_global_indices[source_indices]
    target_mask = np.isclose(times, time_value)
    if not (
        len(prediction) == len(log_mass) == len(sampled_global_indices)
    ):
        raise ValueError(f"{stage}: inconsistent TIGON prediction arrays")
    return {
        "method": "TIGON",
        "rna": prediction,
        "log_mass": log_mass,
        "has_mass": True,
        "source_labels": labels[sampled_global_indices],
        "reference": np.asarray(latent[target_mask], dtype=np.float32),
        "target_labels": labels[target_mask],
        "space": "frozen TIGON AE10 scaled to [-2, 2]",
        "initialization": "all 9,018 exact E7.5 cells; no subsampling",
        "provenance": str(prediction_path),
    }


def target_sets(source: str) -> tuple[set[str], set[str], set[str], str]:
    """Return compatible, ambiguous, incompatible targets and rule label."""

    if source == "Epiblast":
        # Epiblast is pluripotent at this resolution. The broadened compatible
        # set covers every annotated target fate, so off-lineage is not
        # identifiable rather than biologically zero.
        compatible = set(ALL_TARGET_CELLTYPES) - {"Unannotated"}
        ambiguous = {"Unannotated"}
        incompatible: set[str] = set()
        return (
            compatible,
            ambiguous,
            incompatible,
            "all annotated fates compatible; off-lineage not estimable",
        )
    if source == "Caudal epiblast":
        compatible = set(CAUDAL_POSTERIOR_CORRIDOR)
        ambiguous = (set(AMBIGUOUS_TARGETS) - compatible) | {
            "Intermediate mesoderm",
            "Surface ectoderm",
        }
        incompatible = ALL_TARGET_CELLTYPES - compatible - ambiguous
        return (
            compatible,
            ambiguous,
            incompatible,
            (
                "posterior axial corridor; intermediate mesoderm and surface "
                "ectoderm ambiguous"
            ),
        )
    if source in MESENDODERM_SOURCES:
        compatible = set(MESODERM_TARGETS) | set(ENDODERM_TARGETS)
        ambiguous = set(AMBIGUOUS_TARGETS) - compatible
        incompatible = ALL_TARGET_CELLTYPES - compatible - ambiguous
        return compatible, ambiguous, incompatible, "mesendoderm-compatible"
    if source == "PGC":
        compatible = {"PGC"}
        ambiguous = set(AMBIGUOUS_TARGETS) - compatible
        incompatible = ALL_TARGET_CELLTYPES - compatible - ambiguous
        return compatible, ambiguous, incompatible, "PGC-restricted"
    if source in ECTODERM_TARGETS:
        compatible = set(ECTODERM_TARGETS)
        rule = "ectoderm-compatible"
    elif source in MESODERM_TARGETS:
        compatible = set(MESODERM_TARGETS)
        rule = "mesoderm/haemato-endothelial-compatible"
    elif source in ENDODERM_TARGETS:
        compatible = set(ENDODERM_TARGETS)
        rule = "endoderm-compatible"
    else:
        raise KeyError(f"No conservative rule for source cell type: {source}")
    ambiguous = set(AMBIGUOUS_TARGETS)
    incompatible = ALL_TARGET_CELLTYPES - compatible - ambiguous
    return compatible, ambiguous, incompatible, rule


def normalized_records(stage: str) -> list[dict[str, object]]:
    if stage == "E8.0":
        records = [
            {
                "method": str(record["display"]),
                "rna": record["rna"],
                "log_mass": record["log_mass"],
                "has_mass": record["has_mass"],
                "provenance": (
                    "results/gastrulation_strict_loo1_benchmark/"
                    "strict_loo1_e80_benchmark_predictions.npz"
                ),
            }
            for record in load_loo1_records()
        ]
    elif stage == "E8.5":
        records = load_loo2_records()
    else:
        raise ValueError(stage)
    records.append(load_tigon_native_record(stage))
    by_method = {str(record["method"]): record for record in records}
    missing = set(METHOD_ORDER) - set(by_method)
    if missing:
        raise ValueError(f"{stage}: missing methods {sorted(missing)}")
    return [by_method[method] for method in METHOD_ORDER]


def analyze_stage(
    stage: str,
    stage_index: int,
    source_labels: np.ndarray,
    source_counts: pd.Series,
    references: list[np.ndarray],
    labels_by_stage: list[np.ndarray],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, object]],
]:
    reference = np.asarray(references[stage_index], dtype=np.float32)
    target_labels = np.asarray(
        [canonical_celltype(value) for value in labels_by_stage[stage_index]],
        dtype=str,
    )
    shared_neighbours = NearestNeighbors(
        n_neighbors=K,
        algorithm="brute",
        metric="euclidean",
        n_jobs=-1,
    ).fit(reference)
    source_order = analysis_sources(source_counts)

    records = normalized_records(stage)
    rows: list[dict[str, object]] = []
    destination_rows: list[dict[str, object]] = []
    for record in records:
        method = str(record["method"])
        prediction = np.asarray(record["rna"], dtype=np.float32)
        log_mass = np.asarray(record["log_mass"], dtype=np.float64).reshape(-1)
        native_mass = bool(record["has_mass"])
        record_source_labels = np.asarray(
            record.get("source_labels", source_labels), dtype=str
        )
        if "reference" in record:
            record_reference = np.asarray(record["reference"], dtype=np.float32)
            record_target_labels = np.asarray(record["target_labels"], dtype=str)
            neighbours = NearestNeighbors(
                n_neighbors=K,
                algorithm="brute",
                metric="euclidean",
                n_jobs=-1,
            ).fit(record_reference)
        else:
            record_target_labels = target_labels
            neighbours = shared_neighbours
        if len(prediction) != len(record_source_labels):
            raise ValueError(
                f"{stage}, {method}: {len(prediction)} predictions for "
                f"{len(record_source_labels)} source particles"
            )
        assignments = record_target_labels[
            neighbours.kneighbors(prediction, return_distance=False)
        ]
        for source in source_order:
            mask = record_source_labels == source
            if not np.any(mask):
                raise ValueError(f"{stage}, {method}: no sampled {source} particles")
            compatible, ambiguous, incompatible, rule = target_sets(source)
            evidence_class, layer = SOURCE_METADATA[source]
            weights = normalized_weights(log_mass[mask], native_mass)
            local = assignments[mask]
            estimable = source != "Epiblast"
            rows.append(
                {
                    "stage": stage,
                    "method": method,
                    "source_celltype": source,
                    "source_n": int(source_counts[source]),
                    "source_particle_n": int(mask.sum()),
                    "evidence_class": evidence_class,
                    "layer": layer,
                    "rule": rule,
                    "estimable": estimable,
                    "weighting": "native mass" if native_mass else "uniform",
                    "evaluation_space": str(
                        record.get("space", "shared normalized RNA PCA50")
                    ),
                    "on_lineage_mass": float(
                        np.sum(
                            weights
                            * np.mean(np.isin(local, sorted(compatible)), axis=1)
                        )
                    ),
                    "ambiguous_mass": float(
                        np.sum(
                            weights
                            * np.mean(np.isin(local, sorted(ambiguous)), axis=1)
                        )
                    ),
                    "off_lineage_mass": (
                        float(
                            np.sum(
                                weights
                                * np.mean(
                                    np.isin(local, sorted(incompatible)), axis=1
                                )
                            )
                        )
                        if estimable
                        else np.nan
                    ),
                }
            )
            for target in sorted(set(record_target_labels)):
                if target in compatible:
                    classification = "compatible"
                elif target in ambiguous:
                    classification = "ambiguous"
                elif target in incompatible:
                    classification = "off-lineage"
                else:
                    classification = "outside predefined atlas groups"
                destination_rows.append(
                    {
                        "stage": stage,
                        "method": method,
                        "source_celltype": source,
                        "source_n": int(source_counts[source]),
                        "source_particle_n": int(mask.sum()),
                        "evidence_class": evidence_class,
                        "layer": layer,
                        "evaluation_space": str(
                            record.get("space", "shared normalized RNA PCA50")
                        ),
                        "predicted_celltype": target,
                        "classification": classification,
                        "predicted_mass": float(
                            np.sum(weights * np.mean(local == target, axis=1))
                        ),
                    }
                )

    detailed = pd.DataFrame(rows)
    mean_rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        local = detailed[
            detailed["method"].eq(method) & detailed["estimable"]
        ]
        mean_rows.append(
            {
                "stage": stage,
                "method": method,
                "n_source_celltypes": int(len(local)),
                "n_source_cells": int(local["source_n"].sum()),
                "cell_number_weighted_off_lineage_mass": float(
                    np.average(
                        local["off_lineage_mass"], weights=local["source_n"]
                    )
                ),
            }
        )
    return detailed, pd.DataFrame(mean_rows), pd.DataFrame(destination_rows), records


def plot(
    detailed: pd.DataFrame,
    means: pd.DataFrame,
    source_counts: pd.Series,
) -> tuple[Path, Path]:
    apply_nature_rc(font_size=10.0)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "axes.titlesize": 12.0,
            "axes.labelsize": 12.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sources = analysis_sources(source_counts)
    matrices: dict[str, np.ndarray] = {}
    for stage in ("E8.0", "E8.5"):
        stage_data = detailed[detailed["stage"].eq(stage)]
        matrix = (
            stage_data.pivot(
                index="source_celltype",
                columns="method",
                values="off_lineage_mass",
            )
            .loc[sources, list(METHOD_ORDER)]
            .to_numpy(float)
            * 100.0
        )
        weighted = (
            means[means["stage"].eq(stage)]
            .set_index("method")
            .loc[list(METHOD_ORDER), "cell_number_weighted_off_lineage_mass"]
            .to_numpy(float)
            * 100.0
        )
        matrices[stage] = np.vstack([matrix, weighted[None, :]])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.20, 5.55),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    for ax, stage in zip(axes, ("E8.0", "E8.5")):
        matrix = matrices[stage]
        ax.set_facecolor("white")
        ax.set_xlim(-0.5, len(METHOD_ORDER) - 0.5)
        ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
        ax.set_title(f"Held-out {stage}", pad=5, fontweight="normal")
        ax.set_xticks(
            np.arange(len(METHOD_ORDER)),
            [METHOD_LABELS[method] for method in METHOD_ORDER],
        )
        ax.tick_params(axis="x", length=0, pad=4, labelrotation=52)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
            label.set_rotation_mode("anchor")
        ax.tick_params(axis="y", length=0, pad=4)
        ax.set_xticks(np.arange(-0.5, len(METHOD_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(sources) + 1, 1), minor=True)
        ax.grid(which="minor", color="#D8D8D8", linewidth=0.55)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.axhline(len(sources) - 0.5, color="#555555", linewidth=1.0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = float(matrix[row, column])
                if not np.isfinite(value):
                    ax.text(
                        column,
                        row,
                        "—",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="#666666",
                        fontweight="normal",
                    )
                    continue
                ax.text(
                    column,
                    row,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                    fontweight="normal",
                )

    display = {
        "Rostral neurectoderm": "Rostral neuroectoderm",
        "Haematoendothelial progenitors": "Haemato-endothelial prog.",
        "Primitive Streak": "Primitive streak",
        "Anterior Primitive Streak": "Anterior primitive streak",
        "Forebrain/Midbrain/Hindbrain": "Fore-/mid-/hindbrain",
    }
    row_labels = [
        f"{display.get(source, source)}\nn={int(source_counts[source]):,}"
        for source in sources
    ] + ["Cell-number-weighted mean"]
    axes[0].set_yticks(np.arange(len(row_labels)), row_labels)
    axes[0].tick_params(axis="y", labelleft=True)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.suptitle(
        "Conservative off-lineage leakage for A/B source states",
        x=0.55,
        y=0.986,
        fontsize=12,
        fontweight="normal",
    )
    fig.subplots_adjust(left=0.305, right=0.985, top=0.91, bottom=0.19)
    stem = OUTPUT_DIR / (
        "strict_loo_e80_e85_all_source_method_offlineage_" + OUTPUT_TAG
    )
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches=None, facecolor="white")
    fig.savefig(png_path, dpi=600, bbox_inches=None, facecolor="white")
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_labels, source_counts = load_loo1_source()
    loo2_labels, loo2_counts = load_loo2_source()
    if not np.array_equal(source_labels, loo2_labels):
        raise ValueError("LOO1 and LOO2 source-cell ordering differs")
    if not source_counts.sort_index().equals(loo2_counts.sort_index()):
        raise ValueError("LOO1 and LOO2 source-cell counts differ")

    references, _, labels_by_stage, _, _ = _load_references()
    detailed_frames = []
    mean_frames = []
    destination_frames = []
    provenance: dict[str, list[dict[str, str]]] = {}
    for stage, stage_index in (("E8.0", 1), ("E8.5", 2)):
        detailed, means, destinations, records = analyze_stage(
            stage,
            stage_index,
            source_labels,
            source_counts,
            references,
            labels_by_stage,
        )
        detailed_frames.append(detailed)
        mean_frames.append(means)
        destination_frames.append(destinations)
        provenance[stage] = [
            {
                "method": str(record["method"]),
                "source": str(record.get("provenance", "strict LOO cache")),
                "space": str(
                    record.get("space", "shared normalized RNA PCA50")
                ),
                "weighting": (
                    "native growth mass"
                    if bool(record["has_mass"])
                    else "uniform"
                ),
                "initialization": str(record.get("initialization", "all E7.5 cells")),
            }
            for record in records
        ]

    detailed = pd.concat(detailed_frames, ignore_index=True)
    means = pd.concat(mean_frames, ignore_index=True)
    destinations = pd.concat(destination_frames, ignore_index=True)
    detailed_path = OUTPUT_DIR / (
        "strict_loo_e80_e85_all_source_method_offlineage_" + OUTPUT_TAG + ".csv"
    )
    means_path = OUTPUT_DIR / (
        "strict_loo_e80_e85_method_weighted_means_" + OUTPUT_TAG + ".csv"
    )
    definitions_path = OUTPUT_DIR / (
        "source_lineage_definitions_" + OUTPUT_TAG + ".csv"
    )
    classification_path = OUTPUT_DIR / (
        "source_abm_classification_" + OUTPUT_TAG + ".csv"
    )
    destinations_path = OUTPUT_DIR / (
        "source_to_target_composition_" + OUTPUT_TAG + ".csv"
    )
    detailed.to_csv(detailed_path, index=False)
    means.to_csv(means_path, index=False)
    destinations.to_csv(destinations_path, index=False)
    definition_rows = [
        {
            "source_celltype": source,
            "source_n": int(source_counts[source]),
            "evidence_class": SOURCE_METADATA[source][0],
            "layer": SOURCE_METADATA[source][1],
            "included_in_off_lineage": SOURCE_METADATA[source][0] in {"A", "B"},
            "rule": target_sets(source)[3],
            "compatible_targets": "; ".join(sorted(target_sets(source)[0])),
            "ambiguous_targets": "; ".join(sorted(target_sets(source)[1])),
            "off_lineage_targets": "; ".join(sorted(target_sets(source)[2])),
        }
        for source in classified_sources(source_counts)
    ]
    definitions = pd.DataFrame(definition_rows)
    definitions.to_csv(definitions_path, index=False)
    definitions[
        [
            "source_celltype",
            "source_n",
            "evidence_class",
            "layer",
            "included_in_off_lineage",
        ]
    ].to_csv(classification_path, index=False)
    pdf_path, png_path = plot(detailed, means, source_counts)
    manifest = {
        "analysis": (
            "Strict-LOO method comparison of conservative off-lineage leakage "
            "for literature-classified A/B source states"
        ),
        "stages": ["E8.0", "E8.5"],
        "source": (
            "E7.5 source cell types with >200 cells; A/B included, M and "
            "Unannotated excluded"
        ),
        "source_evidence_classes": {
            "A": "literature-supported major descendant compartment",
            "B": "supported germ layer with broad or heterogeneous descendants",
            "M": "multipotent or transitional; excluded from single-layer leakage",
        },
        "readout": {
            "shared_methods": f"soft {K}-NN in normalized RNA PCA50",
            "TIGON": (
                f"soft {K}-NN in TIGON's frozen AE10 scaled to [-2, 2]; no "
                "cross-space coordinate comparison"
            ),
        },
        "mass": "native mass normalized within source for unbalanced methods; uniform otherwise",
        "TIGON": {
            "method_type": "unbalanced",
            "initialization": "all 9,018 exact E7.5 cells; no subsampling",
            "endpoint_weight": "normalized accumulated TIGON log-growth within each source type",
            "representation": (
                "full-data frozen transductive AE10; trajectory training and "
                "observed-only checkpoint selection exclude the held-out stage"
            ),
        },
        "TrajectoryNet": {
            "method_type": "balanced continuous normalizing flow",
            "initialization": (
                "standard Gaussian base; native Gaussian-to-data generation"
            ),
            "source_conditioning": (
                "generated E7.5 particles labelled by at least 3/5 observed "
                "E7.5 neighbours; identical particle indices followed to the "
                "held-out endpoint"
            ),
            "endpoint_weight": "uniform",
            "time_mapping": {
                "E8.0": (
                    "Gaussian-to-E7.5 [1.0, 0.5], followed by the cached "
                    "held-out endpoint using the strict-LOO physical clock"
                ),
                "E8.5": (
                    "Gaussian-to-E7.5 [0.5, 0.0], followed by the cached "
                    "rank/fractional-rank strict-LOO clock"
                ),
            },
        },
        "epiblast": (
            "The broadened compatible set contains all annotated target fates, "
            "so off-lineage is not estimable; the row is retained but excluded "
            "from the weighted mean."
        ),
        "caudal_epiblast": (
            "The original posterior axial corridor is retained, while "
            "Intermediate mesoderm and Surface ectoderm are treated as ambiguous "
            "rather than off-lineage. Caudal epiblast remains estimable and "
            "contributes to the weighted mean."
        ),
        "provenance": provenance,
    }
    (OUTPUT_DIR / f"manifest_{OUTPUT_TAG}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(detailed_path)
    print(means_path)
    print(definitions_path)
    print(classification_path)
    print(destinations_path)
    print(pdf_path)
    print(png_path)
    display_means = means.copy()
    display_means["cell_number_weighted_off_lineage_mass"] *= 100.0
    print("\nCell-number-weighted means (%):")
    print(
        display_means.to_string(
            index=False, float_format=lambda value: f"{value:.2f}"
        )
    )


if __name__ == "__main__":
    main()
