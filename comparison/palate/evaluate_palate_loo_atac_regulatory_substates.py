#!/usr/bin/env python
"""Evaluate within-cell-type ATAC regulatory-substate recovery in palate LOO.

Real held-out ATAC cells are first stratified by their published coarse cell
type (excluding the intermediate group).  Within every stage/cell-type stratum
the normalized ATAC LSI15 coordinates are standardized and clustered at fixed
K=2,3,4.  These method-independent clusters define regulatory substates.

Each method is then scored for mass-aware target support, support precision,
substate composition, rare-substate recovery, and within-cell-type covariance
preservation.  Native particle weights are used for every unbalanced method.
Biological programs are not used to define clusters; fixed external/author
programs are added only after scoring to annotate the recovered substates.
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
import hashlib
import json
import os
import zlib
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
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from analyze_palate_loo_atac_unique_support_biology import build_program_scores
from evaluate_palate_loo_same_space import _load_scale
from plot_palate_strict_loo_six_metrics_revised import (
    local_predicted_mass,
    normalized_weights,
    target_radii,
)
from trainfbench_plot_style import apply_nature_rc, method_style


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "results/palate_strict_sync_loo_full_t_gaga10_tn_forward_native_with_rna_only"
)
DEFAULT_OUTPUT = ROOT / "results/palate_loo_atac_regulatory_substates"
ATAC_FILE = ROOT / "data/palate_atac_benchmark.h5ad"
ATAC_NORM = ROOT / "data/palate_atac_secondary_norm_params_lsi15.pt"

SCENARIOS = {
    "loo_time1": {"stage": "E13.5", "time": 1.0},
    "loo_time2": {"stage": "E14.0", "time": 1.5},
}
INTERMEDIATE = "intermidiate cells"
METHOD_MAP = {
    "COATI-B": "COATI-B",
    "COATI-U": "COATI-U",
    "CytoBridge balanced": "CytoBridge-B",
    "CytoBridge unbalanced": "CytoBridge-U",
    "MIOFlow": "MIOFlow",
    "TrajectoryNet": "TrajectoryNet",
    "Balanced RNA-only": "OT(RNA)",
    "Unbalanced RNA-only": "UOT(RNA)",
}
METHOD_ORDER = (
    "COATI-B",
    "COATI-U",
    "CytoBridge-B",
    "CytoBridge-U",
    "MIOFlow",
    "TrajectoryNet",
    "OT(RNA)",
    "UOT(RNA)",
)
METHOD_STYLE = {
    "COATI-B": "COATI balanced",
    "COATI-U": "COATI unbalanced",
    "CytoBridge-B": "CytoBridge balanced",
    "CytoBridge-U": "CytoBridge unbalanced",
    "MIOFlow": "MIOFlow",
    "TrajectoryNet": "TrajectoryNet",
    "OT(RNA)": "Balanced RNA-only",
    "UOT(RNA)": "Unbalanced RNA-only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clusters", type=int, nargs="+", default=(2, 3, 4))
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--hit-threshold", type=float, default=0.5)
    parser.add_argument("--distance-batch-size", type=int, default=256)
    parser.add_argument("--stability-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_prediction(input_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else input_dir / path


def hit_probability(local_mass: np.ndarray, effective_n: float) -> np.ndarray:
    clipped = np.clip(np.asarray(local_mass, dtype=float), 0.0, 1.0 - 1e-15)
    return -np.expm1(effective_n * np.log1p(-clipped))


def standardized_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0, keepdims=True)
    sd = values.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return ((values - mean) / sd).astype(np.float32)


def weighted_covariance(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    centered = values - np.sum(values * weights[:, None], axis=0, keepdims=True)
    return (centered * weights[:, None]).T @ centered


def covariance_metrics(observed: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    if len(predicted) < 2 or float(np.sum(weights)) <= 0:
        return {"covariance_cosine": np.nan, "diversity_log_trace_error": np.nan}
    observed_cov = np.cov(np.asarray(observed, dtype=np.float64), rowvar=False, ddof=0)
    predicted_cov = weighted_covariance(predicted, weights)
    denominator = np.linalg.norm(observed_cov) * np.linalg.norm(predicted_cov)
    cosine = np.nan if denominator <= 1e-12 else float(np.sum(observed_cov * predicted_cov) / denominator)
    observed_trace = max(float(np.trace(observed_cov)), 1e-12)
    predicted_trace = max(float(np.trace(predicted_cov)), 1e-12)
    return {
        "covariance_cosine": cosine,
        "diversity_log_trace_error": abs(float(np.log(predicted_trace / observed_trace))),
    }


def safe_f1(recall: float, precision: float) -> float:
    if not np.isfinite(recall) or not np.isfinite(precision) or recall + precision <= 0:
        return np.nan
    return float(2.0 * recall * precision / (recall + precision))


def cluster_substates(
    coordinates: np.ndarray,
    celltypes: np.ndarray,
    clusters: tuple[int, ...],
    repeats: int,
    seed: int,
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    assignments = {k: np.full(len(coordinates), -1, dtype=int) for k in clusters}
    audit_rows: list[dict[str, object]] = []
    for celltype in pd.unique(celltypes):
        indices = np.flatnonzero(celltypes == celltype)
        standardized = standardized_rows(coordinates[indices])
        for k in clusters:
            local_seed = (seed + zlib.crc32(f"{celltype}|{k}".encode())) % (2**32)
            model = KMeans(n_clusters=k, n_init=50, random_state=local_seed).fit(standardized)
            labels = model.labels_.astype(int)
            assignments[k][indices] = labels
            stability = []
            for repeat in range(repeats):
                alternate = KMeans(
                    n_clusters=k,
                    n_init=10,
                    random_state=(local_seed + repeat + 1) % (2**32),
                ).fit_predict(standardized)
                stability.append(adjusted_rand_score(labels, alternate))
            counts = np.bincount(labels, minlength=k)
            audit_rows.append(
                {
                    "celltype": celltype,
                    "k": k,
                    "n_cells": len(indices),
                    "minimum_substate_size": int(counts.min()),
                    "maximum_substate_size": int(counts.max()),
                    "minimum_substate_fraction": float(counts.min() / counts.sum()),
                    "mean_seed_stability_ari": float(np.mean(stability)),
                    "minimum_seed_stability_ari": float(np.min(stability)),
                }
            )
    return assignments, pd.DataFrame(audit_rows)


def configure_plot() -> None:
    apply_nature_rc(font_size=10)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        kwargs: dict[str, object] = {"bbox_inches": "tight", "pad_inches": 0.025}
        if suffix == "png":
            kwargs["dpi"] = 400
        fig.savefig(output / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def plot_macro_summary(summary: pd.DataFrame, output: Path) -> None:
    configure_plot()
    averaged = (
        summary.groupby(["heldout_stage", "method"], as_index=False)
        .agg(
            recall=("macro_binary_recall", "mean"),
            precision=("macro_precision", "mean"),
            f1=("macro_f1", "mean"),
            rare=("macro_rarest_substate_recall", "mean"),
            composition=("macro_composition_similarity", "mean"),
            covariance=("macro_covariance_cosine", "mean"),
        )
    )
    panels = (
        ("recall", "Substate recall ↑"),
        ("precision", "Substate precision ↑"),
        ("f1", "Substate F1 ↑"),
        ("rare", "Rare-state recall ↑"),
        ("composition", "Composition sim. ↑"),
        ("covariance", "Covariance cosine ↑"),
    )
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.0), sharex=False)
    for stage_column, stage in enumerate(("E13.5", "E14.0")):
        local = averaged[averaged["heldout_stage"].eq(stage)].set_index("method")
        for metric_index, (metric, title) in enumerate(panels):
            row = metric_index // 2
            inner_column = metric_index % 2
            axis = axes[row, stage_column * 2 + inner_column]
            values = np.asarray([local.loc[method, metric] for method in METHOD_ORDER])
            for index, (method, value) in enumerate(zip(METHOD_ORDER, values)):
                axis.barh(index, value, height=0.56, color=method_style(METHOD_STYLE[method]).color)
                axis.text(value, index, f" {value:.3f}", va="center", ha="left", fontsize=8, color="#222222")
            axis.set_title(f"{stage}\n{title}", pad=3, fontsize=9)
            axis.set_xlim(0, max(1.0, float(np.nanmax(values)) * 1.15))
            axis.invert_yaxis()
            axis.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
            axis.grid(axis="x", color="#DDDDDD", linewidth=0.5)
            axis.set_axisbelow(True)
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.tick_params(axis="y", length=0, pad=2)
            if stage_column == 0 and inner_column == 0:
                axis.set_yticks(np.arange(len(METHOD_ORDER)), METHOD_ORDER, fontsize=8)
            else:
                axis.set_yticks(np.arange(len(METHOD_ORDER)), [])
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.08, top=0.97, hspace=0.56, wspace=0.40)
    save_figure(fig, output, "palate_atac_regulatory_substate_macro_summary")


def plot_k_stability(summary: pd.DataFrame, output: Path) -> None:
    configure_plot()
    fig, axes = plt.subplots(1, 2, figsize=(4.13, 2.25), sharex=True, sharey=True)
    for axis, stage in zip(axes, ("E13.5", "E14.0")):
        local = summary[summary["heldout_stage"].eq(stage)]
        for method in METHOD_ORDER:
            item = local[local["method"].eq(method)].sort_values("k")
            style = method_style(METHOD_STYLE[method])
            axis.plot(item["k"], item["macro_f1"], color=style.color, marker=style.marker, lw=1.2, ms=3.5, label=method)
        axis.set_title(stage)
        axis.set_xticks(sorted(local["k"].unique()))
        axis.set_ylim(0, 1)
        axis.grid(color="#DDDDDD", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Macro substate F1")
    fig.supxlabel("Number of ATAC substates per cell type", y=0.01)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=False, fontsize=8, handlelength=1.5)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.22, top=0.78, wspace=0.16)
    save_figure(fig, output, "palate_atac_regulatory_substate_k_stability")


def main() -> None:
    args = parse_args()
    clusters = tuple(sorted(set(int(value) for value in args.clusters)))
    if not clusters or min(clusters) < 2:
        raise ValueError("All cluster counts must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    required_outputs = (
        args.output_dir / "substate_metrics.csv",
        args.output_dir / "macro_summary.csv",
        args.output_dir / "common_support_extra.csv",
        args.output_dir / "analysis_manifest.json",
    )
    if not args.overwrite and any(path.exists() for path in required_outputs):
        raise FileExistsError("Refusing to overwrite existing regulatory-substate outputs")

    manifest_path = args.input_dir / "prediction_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    manifest["canonical_method"] = manifest["method"].map(METHOD_MAP)
    available = set(manifest["canonical_method"].dropna())
    if not set(METHOD_ORDER).issubset(available):
        raise ValueError(f"Missing methods: {sorted(set(METHOD_ORDER) - available)}")

    atac = ad.read_h5ad(ATAC_FILE)
    times = pd.to_numeric(atac.obs["time_point_processed"], errors="raise").to_numpy(float)
    all_celltypes = atac.obs["celltype_sub"].astype(str).to_numpy()
    cell_ids = atac.obs_names.to_numpy(str)
    scale = _load_scale(ATAC_NORM)
    coordinates = np.asarray(atac.obsm["X_lsi15"], dtype=np.float32) / scale
    program_frame, program_definitions = build_program_scores(atac)
    program_lookup = program_frame.set_index("cell_id")
    program_names = program_definitions["program"].astype(str).tolist()

    metric_rows: list[dict[str, object]] = []
    celltype_rows: list[dict[str, object]] = []
    macro_rows: list[dict[str, object]] = []
    common_rows: list[dict[str, object]] = []
    cluster_audits: list[pd.DataFrame] = []
    assignment_frames: list[pd.DataFrame] = []
    program_rows: list[dict[str, object]] = []

    for scenario, info in SCENARIOS.items():
        heldout_global = np.flatnonzero(
            np.isclose(times, float(info["time"])) & (all_celltypes != INTERMEDIATE)
        )
        heldout = coordinates[heldout_global]
        heldout_celltypes = all_celltypes[heldout_global]
        heldout_ids = cell_ids[heldout_global]
        assignments, cluster_audit = cluster_substates(
            heldout,
            heldout_celltypes,
            clusters,
            args.stability_repeats,
            args.seed + zlib.crc32(scenario.encode()),
        )
        cluster_audit.insert(0, "scenario", scenario)
        cluster_audit.insert(1, "heldout_stage", info["stage"])
        cluster_audits.append(cluster_audit)

        radii = np.empty(len(heldout), dtype=np.float32)
        for celltype in pd.unique(heldout_celltypes):
            indices = np.flatnonzero(heldout_celltypes == celltype)
            radii[indices] = target_radii(heldout[indices], args.knn_k)

        predictions: dict[str, dict[str, np.ndarray | float | str]] = {}
        local_manifest = manifest[manifest["scenario"].eq(scenario)]
        for method in METHOD_ORDER:
            selected = local_manifest[local_manifest["canonical_method"].eq(method)]
            if len(selected) != 1:
                raise ValueError(f"Expected one prediction for {scenario}, {method}; found {len(selected)}")
            item = selected.iloc[0]
            prediction_path = resolve_prediction(args.input_dir, str(item["prediction_file"]))
            with np.load(prediction_path, allow_pickle=False) as saved:
                predicted = np.asarray(saved["atac_norm"], dtype=np.float32)
                weights = normalized_weights(saved["weights"])
            effective_n = float(1.0 / np.sum(np.square(weights)))
            distance, nearest = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(heldout).kneighbors(predicted)
            distance = distance[:, 0]
            nearest = nearest[:, 0]
            supported = distance <= radii[nearest]
            local_mass = local_predicted_mass(
                heldout,
                predicted,
                weights,
                radii,
                args.distance_batch_size,
            )
            probability = hit_probability(local_mass, effective_n)
            predictions[method] = {
                "coordinates": predicted,
                "weights": weights,
                "nearest": nearest,
                "supported": supported,
                "hit_probability": probability,
                "effective_n": effective_n,
                "prediction_file": str(prediction_path),
                "prediction_sha256": sha256(prediction_path),
            }

        for k in clusters:
            substate = assignments[k]
            assignment_frames.append(
                pd.DataFrame(
                    {
                        "scenario": scenario,
                        "heldout_stage": info["stage"],
                        "cell_id": heldout_ids,
                        "celltype": heldout_celltypes,
                        "k": k,
                        "substate": substate,
                    }
                )
            )
            for celltype in pd.unique(heldout_celltypes):
                cell_mask = heldout_celltypes == celltype
                cell_indices = np.flatnonzero(cell_mask)
                counts = np.bincount(substate[cell_indices], minlength=k)
                rarest = int(np.argmin(counts))
                observed_composition = counts / counts.sum()

                local_programs = program_lookup.loc[heldout_ids[cell_indices], program_names]
                means = local_programs.mean(axis=0)
                sds = local_programs.std(axis=0, ddof=0).replace(0, 1.0)
                standardized_programs = (local_programs - means) / sds
                for cluster in range(k):
                    member = substate[cell_indices] == cluster
                    cluster_means = standardized_programs.loc[member].mean(axis=0).sort_values(ascending=False)
                    for rank, (program, value) in enumerate(cluster_means.head(5).items(), start=1):
                        program_rows.append(
                            {
                                "scenario": scenario,
                                "heldout_stage": info["stage"],
                                "celltype": celltype,
                                "k": k,
                                "substate": cluster,
                                "substate_fraction": float(counts[cluster] / counts.sum()),
                                "program_rank": rank,
                                "program": program,
                                "mean_within_celltype_z": float(value),
                            }
                        )

                for method in METHOD_ORDER:
                    record = predictions[method]
                    predicted = np.asarray(record["coordinates"])
                    weights = np.asarray(record["weights"])
                    nearest = np.asarray(record["nearest"], dtype=int)
                    supported = np.asarray(record["supported"], dtype=bool)
                    probability = np.asarray(record["hit_probability"], dtype=float)
                    assigned_celltype = heldout_celltypes[nearest]
                    assigned_substate = substate[nearest]
                    predicted_cell_mask = assigned_celltype == celltype
                    composition_mass = np.asarray(
                        [weights[predicted_cell_mask & (assigned_substate == cluster)].sum() for cluster in range(k)]
                    )
                    if composition_mass.sum() > 0:
                        predicted_composition = composition_mass / composition_mass.sum()
                        composition_jsd = float(jensenshannon(observed_composition, predicted_composition, base=2.0) ** 2)
                    else:
                        predicted_composition = np.full(k, np.nan)
                        composition_jsd = np.nan
                    covariance = covariance_metrics(
                        heldout[cell_indices],
                        predicted[predicted_cell_mask],
                        weights[predicted_cell_mask],
                    )
                    celltype_rows.append(
                        {
                            "scenario": scenario,
                            "heldout_stage": info["stage"],
                            "method": method,
                            "celltype": celltype,
                            "k": k,
                            "n_observed": len(cell_indices),
                            "predicted_celltype_mass": float(weights[predicted_cell_mask].sum()),
                            "composition_jsd": composition_jsd,
                            "composition_similarity": 1.0 - composition_jsd if np.isfinite(composition_jsd) else np.nan,
                            **covariance,
                        }
                    )
                    for cluster in range(k):
                        target_mask = cell_mask & (substate == cluster)
                        predicted_mask = predicted_cell_mask & (assigned_substate == cluster)
                        precision = (
                            float(weights[predicted_mask & supported].sum() / weights[predicted_mask].sum())
                            if weights[predicted_mask].sum() > 0
                            else np.nan
                        )
                        soft_recall = float(np.mean(probability[target_mask]))
                        binary_recall = float(np.mean(probability[target_mask] >= args.hit_threshold))
                        metric_rows.append(
                            {
                                "scenario": scenario,
                                "heldout_stage": info["stage"],
                                "method": method,
                                "celltype": celltype,
                                "k": k,
                                "substate": cluster,
                                "n_observed": int(target_mask.sum()),
                                "observed_fraction_within_celltype": float(target_mask.sum() / cell_mask.sum()),
                                "predicted_fraction_within_celltype": float(predicted_composition[cluster]),
                                "is_rarest_substate": cluster == rarest,
                                "soft_support_recall": soft_recall,
                                "binary_support_recall": binary_recall,
                                "support_precision": precision,
                                "support_f1": safe_f1(binary_recall, precision),
                                "effective_particle_count": float(record["effective_n"]),
                            }
                        )

            hit_matrix = np.column_stack(
                [np.asarray(predictions[method]["hit_probability"]) >= args.hit_threshold for method in METHOD_ORDER]
            )
            common = np.all(hit_matrix, axis=1)
            for method_index, method in enumerate(METHOD_ORDER):
                extra = hit_matrix[:, method_index] & ~common
                for celltype in pd.unique(heldout_celltypes):
                    cell_mask = heldout_celltypes == celltype
                    for cluster in range(k):
                        target_mask = cell_mask & (substate == cluster)
                        common_rows.append(
                            {
                                "scenario": scenario,
                                "heldout_stage": info["stage"],
                                "method": method,
                                "celltype": celltype,
                                "k": k,
                                "substate": cluster,
                                "n_observed": int(target_mask.sum()),
                                "n_common_supported": int(np.sum(common & target_mask)),
                                "n_method_extra_supported": int(np.sum(extra & target_mask)),
                                "method_extra_fraction_of_substate": float(np.mean(extra[target_mask])),
                            }
                        )

        print(f"[{scenario}] substates and support complete", flush=True)

    metrics = pd.DataFrame(metric_rows)
    celltype_metrics = pd.DataFrame(celltype_rows)
    for (scenario, stage, method, k), local in metrics.groupby(
        ["scenario", "heldout_stage", "method", "k"], sort=False
    ):
        cell_local = celltype_metrics[
            celltype_metrics["scenario"].eq(scenario)
            & celltype_metrics["method"].eq(method)
            & celltype_metrics["k"].eq(k)
        ]
        macro_rows.append(
            {
                "scenario": scenario,
                "heldout_stage": stage,
                "method": method,
                "k": k,
                "macro_soft_recall": float(local["soft_support_recall"].mean()),
                "macro_binary_recall": float(local["binary_support_recall"].mean()),
                "macro_precision": float(local["support_precision"].mean()),
                "macro_f1": float(local["support_f1"].mean()),
                "macro_rarest_substate_recall": float(local.loc[local["is_rarest_substate"], "binary_support_recall"].mean()),
                "macro_composition_jsd": float(cell_local["composition_jsd"].mean()),
                "macro_composition_similarity": float(cell_local["composition_similarity"].mean()),
                "macro_covariance_cosine": float(cell_local["covariance_cosine"].mean()),
                "macro_diversity_log_trace_error": float(cell_local["diversity_log_trace_error"].mean()),
            }
        )
    macro = pd.DataFrame(macro_rows)

    metrics.to_csv(args.output_dir / "substate_metrics.csv", index=False)
    celltype_metrics.to_csv(args.output_dir / "celltype_distribution_metrics.csv", index=False)
    macro.to_csv(args.output_dir / "macro_summary.csv", index=False)
    pd.DataFrame(common_rows).to_csv(args.output_dir / "common_support_extra.csv", index=False)
    pd.concat(cluster_audits, ignore_index=True).to_csv(args.output_dir / "cluster_stability_audit.csv", index=False)
    pd.concat(assignment_frames, ignore_index=True).to_csv(
        args.output_dir / "heldout_substate_assignments.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(program_rows).to_csv(args.output_dir / "substate_top_programs.csv", index=False)
    program_definitions.to_csv(args.output_dir / "program_definitions.csv", index=False)

    plot_macro_summary(macro, args.output_dir)
    plot_k_stability(macro, args.output_dir)

    manifest_output = {
        "analysis": "within-cell-type ATAC regulatory-substate recovery in strict palate LOO",
        "input_prediction_manifest": str(manifest_path.resolve()),
        "input_prediction_manifest_sha256": sha256(manifest_path),
        "methods": list(METHOD_ORDER),
        "coati_cy": 0.5,
        "clusters": list(clusters),
        "cluster_space": "within-stage, within-cell-type standardized normalized ATAC LSI15",
        "cluster_algorithm": "KMeans with 50 initializations; seed stability audited with repeated 10-initialization fits",
        "excluded_celltype": INTERMEDIATE,
        "target_support": f"cell-type-restricted {args.knn_k}NN radius; mass-aware hit probability",
        "hit_threshold": args.hit_threshold,
        "precision": "native predicted mass assigned to a substate whose nearest held-out target lies within that target's cell-type-restricted radius",
        "composition": "Jensen-Shannon divergence between observed and nearest-target-assigned predicted substate mass, normalized within cell type",
        "rare_substate": "smallest observed cluster within each stage/cell-type/K stratum",
        "biological_program_policy": "fixed programs annotate clusters after scoring and never define the substates",
        "native_mass_weights": True,
        "missing_fair_ablation": "Balanced and unbalanced ATAC-only models currently have full-train but no strict-LOO predictions and are not mixed into this held-out comparison",
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest_output, indent=2) + "\n", encoding="utf-8"
    )

    averaged = macro.groupby(["heldout_stage", "method"], as_index=False).mean(numeric_only=True)
    print(
        averaged[
            [
                "heldout_stage",
                "method",
                "macro_binary_recall",
                "macro_precision",
                "macro_f1",
                "macro_rarest_substate_recall",
                "macro_composition_similarity",
                "macro_covariance_cosine",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        flush=True,
    )


if __name__ == "__main__":
    main()
