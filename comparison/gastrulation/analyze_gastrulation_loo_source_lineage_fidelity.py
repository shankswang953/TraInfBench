#!/usr/bin/env python
"""Biological lineage-fidelity audit for Gastrulation held-out predictions.

The primary analysis conditions each particle-preserving method on the same
E7.5 source cells. Predicted RNA particles are softly annotated against the
real held-out RNA atlas using k-nearest neighbours in the common normalized
PCA space. Unbalanced methods use their predicted native mass; balanced
methods use uniform particle weights.

TrajectoryNet is deliberately excluded: the benchmark's publication result is
generated from a Gaussian base, so its particles have no E7.5 source-cell
identity and cannot be conditioned on a source cell type.
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
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

from evaluate_gastrulation_full_cmcc import _load_references  # noqa: E402
from trainfbench_plot_style import (  # noqa: E402
    GASTRULATION_CELLTYPE_ALIASES,
    GASTRULATION_CELLTYPE_GROUPS,
    NATURE_CUD,
    apply_nature_rc,
    gastrulation_celltype_color,
)


DEFAULT_OUTPUT_DIR = ROOT / "results/gastrulation_loo_source_lineage_fidelity"
SOURCE_H5AD = ROOT / "data/gastrulation_rna_loo_time1_cytobridge.h5ad"

SCENARIOS = {
    "E8.0": {
        "stage_index": 1,
        "prediction": (
            ROOT
            / "results/gastrulation_loo_time1_coverage_validity"
            / "loo_time1_shared_t_atac_predictions.npz"
        ),
    },
    "E8.5": {
        "stage_index": 2,
        "prediction": (
            ROOT
            / "results/gastrulation_loo_time2_shared_t_atac"
            / "loo_time2_shared_t_predictions.npz"
        ),
    },
}

METHODS = (
    ("BSOT C_y=0.3", "COATI bal"),
    ("USOT C_y=0.3", "COATI unbal"),
    ("CytoBridge balanced 20k", "CytoBridge bal"),
    ("CytoBridge unbalanced 20k", "CytoBridge unbal"),
    ("MIOFlow 20k", "MIOFlow"),
    ("TIGON 20k", "TIGON"),
)

METHOD_COLORS = {
    "COATI bal": NATURE_CUD["blue"],
    "COATI unbal": NATURE_CUD["vermillion"],
    "CytoBridge bal": NATURE_CUD["bluish_green"],
    "CytoBridge unbal": NATURE_CUD["reddish_purple"],
    "MIOFlow": NATURE_CUD["orange"],
    "TIGON": NATURE_CUD["sky_blue"],
}

SOURCE_CELLTYPE = "Rostral neurectoderm"
ROSTRAL_CORRIDOR = {
    "Rostral neurectoderm",
    "Forebrain/Midbrain/Hindbrain",
}
OTHER_ECTODERM = {
    "Surface ectoderm",
    "Neural crest",
}
POSTERIOR_NEURAL = {
    "Caudal neurectoderm",
    "Spinal cord",
    "NMP",
}
NEURAL_COMPATIBLE = (
    ROSTRAL_CORRIDOR
    | POSTERIOR_NEURAL
    | {"Neural crest"}
)
HAEMATO_ENDOTHELIAL = {
    "Haematoendothelial progenitors",
    "Endothelium",
    "Blood progenitors 1",
    "Blood progenitors 2",
    "Erythroid1",
    "Erythroid2",
    "Erythroid3",
}
MESODERM_GROUPS = {
    "Axial / paraxial mesoderm",
    "Other mesoderm",
    "Haemato-endothelial",
}
ELIGIBLE_COMPARTMENTS = {
    "Neural / ectoderm",
    "Axial / paraxial mesoderm",
    "Other mesoderm",
    "Haemato-endothelial",
    "Endoderm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--sensitivity-k", type=int, nargs="*", default=[1])
    parser.add_argument("--min-source-cells", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def canonical_celltype(value: object) -> str:
    text = str(value)
    if text.lower() == "nan":
        text = "Unannotated"
    return GASTRULATION_CELLTYPE_ALIASES.get(text, text)


def celltype_group(value: str) -> str:
    canonical = canonical_celltype(value)
    for group, members in GASTRULATION_CELLTYPE_GROUPS.items():
        if canonical in members:
            return group
    raise KeyError(f"Unregistered Gastrulation cell type: {canonical}")


def allowed_groups(source_celltype: str) -> set[str]:
    group = celltype_group(source_celltype)
    if group == "Neural / ectoderm":
        return {"Neural / ectoderm"}
    if group in {"Axial / paraxial mesoderm", "Other mesoderm"}:
        return set(MESODERM_GROUPS)
    if group == "Haemato-endothelial":
        return {"Haemato-endothelial"}
    if group == "Endoderm":
        return {"Endoderm"}
    return set()


def normalized_weights(
    log_mass: np.ndarray,
    use_native_mass: bool,
) -> np.ndarray:
    if not use_native_mass:
        return np.full(len(log_mass), 1.0 / len(log_mass), dtype=np.float64)
    values = np.asarray(log_mass, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Native log mass contains non-finite values")
    weights = np.exp(values - values.max())
    return weights / weights.sum()


def soft_composition(
    neighbour_labels: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    if neighbour_labels.shape[0] != weights.shape[0]:
        raise ValueError("Particle and weight counts differ")
    k = neighbour_labels.shape[1]
    result: dict[str, float] = {}
    for celltype in np.unique(neighbour_labels):
        result[str(celltype)] = float(
            np.sum(weights[:, None] * (neighbour_labels == celltype)) / k
        )
    return result


def rostral_summary(composition: dict[str, float]) -> dict[str, float]:
    def total(celltypes: set[str]) -> float:
        return float(sum(composition.get(celltype, 0.0) for celltype in celltypes))

    all_neural = set(
        GASTRULATION_CELLTYPE_GROUPS["Neural / ectoderm"]
    )
    strict = total(ROSTRAL_CORRIDOR)
    other_ectoderm = total(OTHER_ECTODERM)
    posterior = total(POSTERIOR_NEURAL)
    haemato = total(HAEMATO_ENDOTHELIAL)
    broad_neural = total(all_neural)
    neural_compatible = total(NEURAL_COMPATIBLE)
    return {
        "strict_rostral_corridor_fraction": strict,
        "other_ectoderm_fraction": other_ectoderm,
        "posterior_neural_diversion_fraction": posterior,
        "haematoendothelial_leakage_fraction": haemato,
        "neural_compatible_fraction": neural_compatible,
        "offlineage_fraction": 1.0 - neural_compatible,
        "broad_neural_ectoderm_fraction": broad_neural,
        "strict_rostral_offlineage_fraction": 1.0 - strict,
        "broad_non_neural_fraction": 1.0 - broad_neural,
    }


def load_source_metadata() -> tuple[pd.DataFrame, np.ndarray]:
    adata = ad.read_h5ad(SOURCE_H5AD, backed="r")
    time = pd.to_numeric(
        adata.obs["time_point_processed"], errors="raise"
    ).to_numpy(float)
    mask = np.isclose(time, 0.0)
    source = adata.obs.loc[mask].copy()
    if len(source) != 9018:
        raise ValueError(f"Expected 9018 E7.5 source cells, found {len(source)}")
    source["celltype"] = source["celltype"].map(canonical_celltype)
    source_x_raw = np.asarray(adata.obsm["X_latent"][mask], dtype=np.float32)
    return source, source_x_raw


def analyze_k(
    *,
    k: int,
    source: pd.DataFrame,
    rna_references: list[np.ndarray],
    labels_by_stage: list[np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rostral_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    abundant_rows: list[dict[str, object]] = []
    source_labels = source["celltype"].to_numpy(dtype=str)
    source_counts = source["celltype"].value_counts()
    rostral_mask = source_labels == SOURCE_CELLTYPE

    for stage, spec in SCENARIOS.items():
        stage_index = int(spec["stage_index"])
        reference = np.asarray(rna_references[stage_index], dtype=np.float32)
        target_labels = np.asarray(
            [canonical_celltype(value) for value in labels_by_stage[stage_index]],
            dtype=str,
        )
        neighbours = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(reference)
        with np.load(Path(spec["prediction"]), allow_pickle=True) as cache:
            cache_methods = np.asarray(cache["methods"], dtype=str)
            for source_method, display_name in METHODS:
                matches = np.flatnonzero(cache_methods == source_method)
                if len(matches) != 1:
                    raise ValueError(
                        f"{stage}: expected one {source_method}, found {len(matches)}"
                    )
                method_index = int(matches[0])
                prediction = np.asarray(
                    cache["rna_predictions"][method_index],
                    dtype=np.float32,
                )
                if prediction.shape[0] != len(source):
                    raise ValueError(
                        f"{stage} {display_name}: prediction/source mismatch "
                        f"{prediction.shape[0]} != {len(source)}"
                    )
                log_mass = np.asarray(
                    cache["native_log_masses"][method_index],
                    dtype=np.float64,
                )
                use_native_mass = bool(cache["has_native_mass"][method_index])
                assigned = target_labels[
                    neighbours.kneighbors(prediction, return_distance=False)
                ]

                weights = normalized_weights(
                    log_mass[rostral_mask],
                    use_native_mass,
                )
                composition = soft_composition(
                    assigned[rostral_mask],
                    weights,
                )
                for celltype in sorted(set(target_labels)):
                    rostral_rows.append(
                        {
                            "stage": stage,
                            "method": display_name,
                            "source_celltype": SOURCE_CELLTYPE,
                            "source_n": int(rostral_mask.sum()),
                            "k": k,
                            "weighting": (
                                "native mass" if use_native_mass else "uniform"
                            ),
                            "predicted_celltype": celltype,
                            "predicted_fraction": composition.get(celltype, 0.0),
                        }
                    )
                summary_rows.append(
                    {
                        "stage": stage,
                        "method": display_name,
                        "source_celltype": SOURCE_CELLTYPE,
                        "source_n": int(rostral_mask.sum()),
                        "k": k,
                        "weighting": (
                            "native mass" if use_native_mass else "uniform"
                        ),
                        **rostral_summary(composition),
                    }
                )

                for source_celltype, count in source_counts.items():
                    allowed = allowed_groups(str(source_celltype))
                    if count < 1 or not allowed:
                        continue
                    mask = source_labels == source_celltype
                    local_weights = normalized_weights(
                        log_mass[mask],
                        use_native_mass,
                    )
                    target_groups = np.asarray(
                        [
                            celltype_group(celltype)
                            for celltype in assigned[mask].reshape(-1)
                        ],
                        dtype=str,
                    ).reshape(assigned[mask].shape)
                    on_compartment = np.isin(
                        target_groups,
                        sorted(allowed),
                    )
                    on_fraction = float(
                        np.sum(local_weights[:, None] * on_compartment) / k
                    )
                    abundant_rows.append(
                        {
                            "stage": stage,
                            "method": display_name,
                            "source_celltype": str(source_celltype),
                            "source_group": celltype_group(str(source_celltype)),
                            "source_n": int(count),
                            "k": k,
                            "weighting": (
                                "native mass" if use_native_mass else "uniform"
                            ),
                            "allowed_target_groups": "; ".join(sorted(allowed)),
                            "on_compartment_fraction": on_fraction,
                            "off_compartment_fraction": 1.0 - on_fraction,
                        }
                    )

    return (
        pd.DataFrame(rostral_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(abundant_rows),
    )


def abundance_association(
    abundant: pd.DataFrame,
    min_source_cells: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = abundant[abundant["source_n"] >= min_source_cells]
    for (stage, method, k), local in selected.groupby(
        ["stage", "method", "k"],
        sort=False,
    ):
        rho, p_value = spearmanr(
            local["source_n"].to_numpy(float),
            local["off_compartment_fraction"].to_numpy(float),
        )
        rows.append(
            {
                "stage": stage,
                "method": method,
                "k": int(k),
                "min_source_cells": min_source_cells,
                "n_source_celltypes": len(local),
                "spearman_rho_source_n_vs_off_compartment": float(rho),
                "spearman_p_value": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def plotted_celltypes(composition: pd.DataFrame, threshold: float = 0.02) -> list[str]:
    maxima = composition.groupby("predicted_celltype")[
        "predicted_fraction"
    ].max()
    selected = maxima[maxima >= threshold].sort_values(ascending=False).index
    return [str(value) for value in selected]


def plot_rostral_composition(
    composition: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    summary_column: str,
    summary_title: str,
    stem_name: str,
) -> None:
    method_order = [display for _, display in METHODS]
    celltypes = plotted_celltypes(composition)
    y = np.arange(len(method_order))
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.15, 5.0),
        gridspec_kw={"width_ratios": [3.6, 1.0], "hspace": 0.42, "wspace": 0.10},
    )

    for row, stage in enumerate(SCENARIOS):
        ax = axes[row, 0]
        local = composition[composition["stage"].eq(stage)]
        left = np.zeros(len(method_order), dtype=float)
        for celltype in celltypes:
            values = np.asarray(
                [
                    local[
                        local["method"].eq(method)
                        & local["predicted_celltype"].eq(celltype)
                    ]["predicted_fraction"].sum()
                    for method in method_order
                ]
            )
            ax.barh(
                y,
                values,
                left=left,
                height=0.72,
                color=gastrulation_celltype_color(celltype),
                edgecolor="white",
                linewidth=0.35,
            )
            left += values
        other = np.maximum(0.0, 1.0 - left)
        ax.barh(
            y,
            other,
            left=left,
            height=0.72,
            color=gastrulation_celltype_color("Unannotated"),
            edgecolor="white",
            linewidth=0.35,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_yticks(y, method_order)
        ax.set_ylim(len(method_order) - 0.35, -0.65)
        ax.set_xticks((0.0, 0.5, 1.0), ("0", "50%", "100%"))
        ax.set_title(f"Held-out {stage}", loc="left", fontweight="normal")
        ax.set_xlabel("Predicted cell-type fraction")
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)

        leak_ax = axes[row, 1]
        stage_summary = summary[summary["stage"].eq(stage)].set_index("method")
        metric_values = np.asarray(
            [
                stage_summary.loc[method, summary_column]
                for method in method_order
            ],
            dtype=float,
        )
        leak_ax.barh(
            y,
            metric_values,
            height=0.60,
            color=[METHOD_COLORS[method] for method in method_order],
            edgecolor="none",
        )
        for y_value, value in zip(y, metric_values):
            leak_ax.text(
                value + 0.008,
                y_value,
                f"{100.0 * value:.1f}%",
                va="center",
                ha="left",
            )
        upper = max(0.10, float(metric_values.max()) * 1.35)
        leak_ax.set_xlim(0.0, upper)
        leak_ax.set_ylim(len(method_order) - 0.35, -0.65)
        leak_ax.set_yticks([])
        leak_ax.set_xticks([])
        leak_ax.set_title(summary_title, loc="left", fontweight="normal")
        leak_ax.spines["bottom"].set_visible(False)
        leak_ax.spines["left"].set_visible(False)

    handles = [
        Patch(
            facecolor=gastrulation_celltype_color(celltype),
            edgecolor="none",
            label=celltype,
        )
        for celltype in celltypes
    ]
    handles.append(
        Patch(
            facecolor=gastrulation_celltype_color("Unannotated"),
            edgecolor="none",
            label="Other (<2% each)",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=4,
        frameon=False,
        handlelength=1.1,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.18, right=0.995, top=0.97, bottom=0.26)
    stem = output_dir / stem_name
    for extension in ("png", "pdf", "svg"):
        options = {"bbox_inches": "tight", "pad_inches": 0.02}
        if extension == "png":
            options["dpi"] = 600
        fig.savefig(stem.with_suffix(f".{extension}"), **options)
    plt.close(fig)


def plot_abundant_source_audit(
    abundant: pd.DataFrame,
    association: pd.DataFrame,
    min_source_cells: int,
    output_dir: Path,
) -> None:
    selected_methods = ["COATI bal", "COATI unbal"]
    local = abundant[
        abundant["method"].isin(selected_methods)
        & (abundant["source_n"] >= min_source_cells)
    ].copy()
    order = (
        local[local["stage"].eq("E8.0")]
        .groupby("source_celltype")["off_compartment_fraction"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    y = np.arange(len(order))
    offsets = {"COATI bal": -0.13, "COATI unbal": 0.13}
    markers = {"COATI bal": "o", "COATI unbal": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 4.15), sharey=True)
    for column, stage in enumerate(SCENARIOS):
        ax = axes[column]
        stage_local = local[local["stage"].eq(stage)]
        for method in selected_methods:
            values = stage_local[stage_local["method"].eq(method)].set_index(
                "source_celltype"
            )
            fraction = np.asarray(
                [values.loc[celltype, "off_compartment_fraction"] for celltype in order]
            )
            counts = np.asarray(
                [values.loc[celltype, "source_n"] for celltype in order]
            )
            sizes = 18.0 + 42.0 * np.sqrt(counts / counts.max())
            ax.scatter(
                fraction,
                y + offsets[method],
                s=sizes,
                color=METHOD_COLORS[method],
                marker=markers[method],
                linewidths=0.4,
                edgecolors="white",
                label=method,
                zorder=3,
            )
        assoc = association[
            association["stage"].eq(stage)
            & association["method"].eq("COATI unbal")
        ].iloc[0]
        ax.text(
            0.98,
            0.02,
            rf"$\rho_{{n,off}}$={assoc['spearman_rho_source_n_vs_off_compartment']:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )
        ax.set_xlim(left=0.0)
        ax.set_title(f"Held-out {stage}", fontweight="normal")
        ax.set_xlabel("Off-compartment fraction")
        ax.set_yticks(y, order)
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.66, 0.985),
        ncol=2,
        frameon=False,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.30, right=0.995, top=0.84, bottom=0.16, wspace=0.13)
    stem = output_dir / "abundant_source_celltype_off_compartment_audit"
    for extension in ("png", "pdf", "svg"):
        options = {"bbox_inches": "tight", "pad_inches": 0.02}
        if extension == "png":
            options["dpi"] = 600
        fig.savefig(stem.with_suffix(f".{extension}"), **options)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_dir / "rostral_heldout_celltype_composition.csv",
        output_dir / "rostral_heldout_lineage_summary.csv",
        output_dir / "source_celltype_off_compartment.csv",
        output_dir / "source_abundance_association.csv",
        output_dir / "analysis_manifest.json",
    ]
    output_paths.extend(
        output_dir / f"{stem}.{extension}"
        for stem in (
            "rostral_heldout_celltype_composition",
            "rostral_heldout_offlineage_composition",
            "abundant_source_celltype_off_compartment_audit",
        )
        for extension in ("png", "pdf", "svg")
    )
    existing = [path for path in output_paths if path.exists()]
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
            "axes.labelweight": "normal",
        }
    )
    source, source_x_raw = load_source_metadata()
    rna_references, _, labels_by_stage, rna_scale, _ = _load_references()
    source_x_norm = source_x_raw / float(rna_scale)
    if not np.allclose(
        source_x_norm,
        rna_references[0],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "E7.5 source rows are not exactly aligned to the prediction input"
        )
    k_values = list(dict.fromkeys([args.k, *args.sensitivity_k]))
    composition_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    abundant_frames: list[pd.DataFrame] = []
    for k in k_values:
        composition, summary, abundant = analyze_k(
            k=k,
            source=source,
            rna_references=rna_references,
            labels_by_stage=labels_by_stage,
        )
        composition_frames.append(composition)
        summary_frames.append(summary)
        abundant_frames.append(abundant)

    composition_all = pd.concat(composition_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)
    abundant_all = pd.concat(abundant_frames, ignore_index=True)
    association = abundance_association(
        abundant_all,
        args.min_source_cells,
    )
    composition_all.to_csv(output_paths[0], index=False)
    summary_all.to_csv(output_paths[1], index=False)
    abundant_all.to_csv(output_paths[2], index=False)
    association.to_csv(output_paths[3], index=False)

    primary_composition = composition_all[composition_all["k"].eq(args.k)]
    primary_summary = summary_all[summary_all["k"].eq(args.k)]
    primary_abundant = abundant_all[abundant_all["k"].eq(args.k)]
    primary_association = association[association["k"].eq(args.k)]
    plot_rostral_composition(
        primary_composition,
        primary_summary,
        output_dir,
        summary_column="haematoendothelial_leakage_fraction",
        summary_title="Haemato leakage",
        stem_name="rostral_heldout_celltype_composition",
    )
    plot_rostral_composition(
        primary_composition,
        primary_summary,
        output_dir,
        summary_column="offlineage_fraction",
        summary_title="Off-lineage",
        stem_name="rostral_heldout_offlineage_composition",
    )
    plot_abundant_source_audit(
        primary_abundant,
        primary_association,
        args.min_source_cells,
        output_dir,
    )

    manifest = {
        "analysis": "Gastrulation held-out source-conditioned lineage fidelity",
        "source_stage": "E7.5",
        "source_celltype_primary": SOURCE_CELLTYPE,
        "held_out_stages": list(SCENARIOS),
        "k_primary": args.k,
        "k_sensitivity": args.sensitivity_k,
        "min_source_cells": args.min_source_cells,
        "space": "common normalized RNA PCA50",
        "annotation": (
            "soft kNN cell-type assignment against the real held-out RNA atlas"
        ),
        "mass_weighting": (
            "native predicted mass for unbalanced methods; uniform particles "
            "for balanced methods"
        ),
        "off_lineage_definition": (
            "One minus the neural-compatible fraction. Neural-compatible "
            "contains Rostral neurectoderm, Forebrain/Midbrain/Hindbrain, "
            "Caudal neurectoderm, Spinal cord, NMP, and Neural crest. Surface "
            "ectoderm and all non-neural states are off-lineage. This matches "
            "the prior rostral analysis convention."
        ),
        "trajectorynet_exclusion": (
            "Publication held-out predictions originate from a Gaussian base "
            "and therefore lack E7.5 source-cell identity."
        ),
        "interpretation_limit": (
            "No clonal lineage-tracing labels are available. The analysis "
            "measures atlas-based lineage plausibility, not observed descendants "
            "of individual E7.5 cells."
        ),
        "methods": [display for _, display in METHODS],
        "inputs": {
            stage: str(spec["prediction"]) for stage, spec in SCENARIOS.items()
        },
    }
    output_paths[4].write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
