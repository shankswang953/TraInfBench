#!/usr/bin/env python
"""Compare held-out NMP-to-somitic RNA and ATAC direction cosine."""

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

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "common"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_gastrulation_loo_regulatory_edges import (  # noqa: E402
    DATA,
    _candidate_feature_matrices,
    _cis_candidate_links,
    _cohort_mean,
    _cosine,
    _dense_standardized,
    _knn_aggregates,
    _memberships,
    _particle_weights,
    _select_training_links,
    _standardization,
    _standardized_mean,
    _training_active_links_for_branch,
)
from evaluate_gastrulation_full_cmcc import _load_references  # noqa: E402
from trainfbench_plot_style import apply_nature_rc, method_style  # noqa: E402


DEFAULT_GENE_LIST = (
    ROOT
    / "results/gastrulation_somitic_gene_module_selection"
    / "selected_somitic_core_genes.txt"
)
DEFAULT_CACHE_DIR = (
    ROOT
    / "results/gastrulation_loo_latest_external_predictions_with_trajectorynet_base"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/gastrulation_somitic_gene_module_selection"
    / "rna_atac_cosine_comparison"
)
DEFAULT_STRICT_LOO1_DIR = Path(
    "external/COATI/Gastrulation/"
    "USOT_strict_LOO1/trajectory_iter20000"
)
TSS_TABLE = DATA / "atac_tss_rowdata.csv"

TASKS = (
    ("time1", 1, 1.0),
    ("time2", 2, 2.0),
)
METHOD_NAMES = {
    "BSOT C_y=0.3": "COATI balanced",
    "USOT C_y=0.3": "COATI unbalanced",
    "CytoBridge balanced 20k": "CytoBridge balanced",
    "CytoBridge unbalanced 20k": "CytoBridge unbalanced",
    "MIOFlow 20k": "MIOFlow",
    "TrajectoryNet 20k": "TrajectoryNet",
}
METHOD_ORDER = tuple(METHOD_NAMES.values())


def _read_genes(path: Path) -> list[str]:
    genes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(genes) < 3 or len(genes) != len(set(genes)):
        raise ValueError(f"Expected at least three unique genes in {path}")
    return genes


def _gene_tss_seed(genes: list[str]) -> pd.DataFrame:
    annotations = pd.read_csv(TSS_TABLE)
    annotations = annotations[annotations["SYMBOL"].isin(genes)].copy()
    counts = annotations.groupby("SYMBOL").size()
    missing = [gene for gene in genes if gene not in counts]
    duplicated = counts[counts.ne(1)].index.tolist()
    if missing or duplicated:
        raise ValueError(
            f"TSS annotation problem; missing={missing}, duplicated={duplicated}"
        )
    annotations["gene"] = annotations["SYMBOL"].astype(str)
    annotations["chrom"] = annotations["seqnames"].astype(str)
    annotations["tss_pos"] = np.where(
        annotations["strand"].eq(1),
        annotations["end"],
        annotations["start"],
    ).astype(int)
    return annotations[["gene", "chrom", "tss_pos"]].sort_values("gene")


def _load_prediction_cache(
    cache_dir: Path, task_name: str, physical_time: float
) -> dict[str, np.ndarray]:
    path = cache_dir / f"loo_{task_name}_latest_external_predictions.npz"
    cache = np.load(path, allow_pickle=False)
    if not np.isclose(float(cache["physical_time"]), physical_time):
        raise ValueError(f"Physical-time mismatch in {path}")
    return {
        "path": np.asarray(str(path)),
        "methods": cache["methods"].astype(str),
        "rna_predictions": np.asarray(cache["rna_predictions"], dtype=np.float32),
        "atac_predictions": np.asarray(cache["atac_predictions"], dtype=np.float32),
        "native_log_masses": np.asarray(cache["native_log_masses"], dtype=np.float32),
        "has_native_mass": np.asarray(cache["has_native_mass"], dtype=bool),
    }


def _load_strict_usot_loo1(
    trajectory_dir: Path,
    cy: float,
) -> dict[str, object]:
    tag = f"{cy:.1f}"
    paths = {
        "rna": trajectory_dir
        / f"primary_trajectory_sync_all1_loo_time1_s0_a{tag}_iter20000.pt",
        "atac": trajectory_dir
        / f"secondary_trajectory_sync_all1_loo_time1_s0_a{tag}_iter20000.pt",
        "log_mass": trajectory_dir
        / f"mass_lnw_trajectory_sync_all1_loo_time1_s0_a{tag}_iter20000.pt",
        "t_grid": trajectory_dir / "t_grid_sync_all1_loo_time1_s0_iter20000.pt",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing strict LOO1 trajectories: " + ", ".join(missing))

    arrays: dict[str, np.ndarray] = {}
    for key, path in paths.items():
        value = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        arrays[key] = np.asarray(value)
    t_grid = arrays["t_grid"].reshape(-1)
    endpoint_index = int(np.argmin(np.abs(t_grid - 1.0)))
    if not np.isclose(t_grid[endpoint_index], 1.0, atol=1e-5):
        raise ValueError(
            f"Strict LOO1 trajectory has no E8.0/time=1 endpoint: "
            f"nearest={t_grid[endpoint_index]}"
        )
    rna = np.asarray(arrays["rna"][endpoint_index], dtype=np.float32)
    atac = np.asarray(arrays["atac"][endpoint_index], dtype=np.float32)
    log_mass = np.asarray(
        arrays["log_mass"][endpoint_index], dtype=np.float32
    ).reshape(-1)
    if (
        rna.shape[0] != atac.shape[0]
        or rna.shape[0] != log_mass.shape[0]
        or not np.isfinite(rna).all()
        or not np.isfinite(atac).all()
        or not np.isfinite(log_mass).all()
    ):
        raise ValueError("Invalid strict LOO1 endpoint arrays")
    return {
        "rna": rna,
        "atac": atac,
        "log_mass": log_mass,
        "initial_rna": np.asarray(arrays["rna"][0], dtype=np.float32),
        "cy": cy,
        "endpoint_index": endpoint_index,
        "endpoint_time": float(t_grid[endpoint_index]),
        "paths": {key: str(path) for key, path in paths.items()},
    }


def _evaluate(
    genes: list[str],
    cache_dir: Path,
    output_dir: Path,
    strict_loo1_dir: Path,
    strict_loo1_cy: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    processed = ad.read_h5ad(DATA / "gastrulation_rna_processed.h5ad", backed="r")
    raw_rna = ad.read_h5ad(DATA / "gastrulation_rna.h5ad", backed="r")
    raw_atac = ad.read_h5ad(DATA / "gastrulation_atac_peaks.h5ad", backed="r")

    candidates = _cis_candidate_links(
        _gene_tss_seed(genes), raw_atac.var_names, max_distance=250_000
    )
    links, rna, atac, gene_names, peak_names = _candidate_feature_matrices(
        processed, raw_rna, raw_atac, candidates
    )
    rna_references, atac_references, labels_by_stage, _, _ = _load_references()
    strict_loo1 = _load_strict_usot_loo1(strict_loo1_dir, strict_loo1_cy)
    strict_initial_max_abs = float(
        np.max(
            np.abs(
                np.asarray(strict_loo1["initial_rna"], dtype=np.float32)
                - rna_references[0]
            )
        )
    )
    if strict_initial_max_abs > 1e-5:
        raise ValueError(
            "Strict LOO1 trajectory does not start from the shared E7.5 RNA input; "
            f"max abs difference={strict_initial_max_abs:.6g}"
        )
    obs_times = processed.obs["processed_time"].to_numpy(float)
    obs_labels = (
        processed.obs["celltype"]
        .astype("string")
        .fillna("Unknown")
        .to_numpy(dtype=str)
    )

    metric_rows: list[dict[str, object]] = []
    vector_rows: list[dict[str, object]] = []
    active_frames: list[pd.DataFrame] = []
    manifest_tasks: dict[str, object] = {}

    for task_name, stage_index, physical_time in TASKS:
        aggregate_rna, aggregate_atac, aggregate_metadata, training_rows = (
            _knn_aggregates(
                processed,
                rna,
                atac,
                physical_time,
                k=50,
                iterations=500,
                overlap_cutoff=0.8,
                seed=10_000 + stage_index,
            )
        )
        selected_links = _select_training_links(
            links,
            aggregate_rna,
            aggregate_atac,
            min_correlation=0.45,
            max_fdr=1e-4,
            permutations=200,
            max_permutation_fdr=0.05,
            seed=20_000 + stage_index,
        )
        rna_center, rna_scale, atac_center, atac_scale = _standardization(
            aggregate_rna, aggregate_atac
        )
        heldout_rows = np.flatnonzero(np.isclose(obs_times, physical_time))
        heldout_labels = obs_labels[heldout_rows]
        if not np.array_equal(heldout_labels, labels_by_stage[stage_index]):
            raise ValueError(f"{task_name}: held-out labels do not align")
        heldout_rna = _dense_standardized(
            rna, heldout_rows, rna_center, rna_scale
        )
        heldout_atac = _dense_standardized(
            atac, heldout_rows, atac_center, atac_scale
        )

        training_nmp = training_rows[obs_labels[training_rows] == "NMP"]
        training_somitic = training_rows[
            obs_labels[training_rows] == "Somitic mesoderm"
        ]
        training_rna_delta = _standardized_mean(
            rna, training_somitic, rna_center, rna_scale
        ) - _standardized_mean(rna, training_nmp, rna_center, rna_scale)
        training_atac_delta = _standardized_mean(
            atac, training_somitic, atac_center, atac_scale
        ) - _standardized_mean(atac, training_nmp, atac_center, atac_scale)
        active = _training_active_links_for_branch(
            selected_links,
            training_rna_delta,
            training_atac_delta,
            minimum_effect=0.15,
        )
        if active.empty:
            raise ValueError(f"{task_name}: no training-selected active cis links")
        active.insert(0, "loo_task", task_name)
        active_frames.append(active)
        active_peak_indices = np.unique(active["peak_index"].to_numpy(int))
        active_gene_indices = np.unique(active["gene_index"].to_numpy(int))

        nmp = heldout_labels == "NMP"
        somitic = heldout_labels == "Somitic mesoderm"
        observed_rna = heldout_rna[somitic].mean(axis=0) - heldout_rna[nmp].mean(axis=0)
        observed_atac = (
            heldout_atac[somitic].mean(axis=0) - heldout_atac[nmp].mean(axis=0)
        )
        cache = _load_prediction_cache(cache_dir, task_name, physical_time)
        strict_replacement: dict[str, object] | None = None
        if task_name == "time1":
            method_indices = np.flatnonzero(cache["methods"] == "USOT C_y=0.3")
            if len(method_indices) != 1:
                raise ValueError(
                    "Expected exactly one original COATI-unbalanced entry in E8.0 cache"
                )
            method_index = int(method_indices[0])
            if cache["rna_predictions"][method_index].shape != np.asarray(
                strict_loo1["rna"]
            ).shape:
                raise ValueError("Strict LOO1 RNA endpoint shape mismatch")
            if cache["atac_predictions"][method_index].shape != np.asarray(
                strict_loo1["atac"]
            ).shape:
                raise ValueError("Strict LOO1 ATAC endpoint shape mismatch")
            cache["rna_predictions"][method_index] = np.asarray(
                strict_loo1["rna"], dtype=np.float32
            )
            cache["atac_predictions"][method_index] = np.asarray(
                strict_loo1["atac"], dtype=np.float32
            )
            cache["native_log_masses"][method_index] = np.asarray(
                strict_loo1["log_mass"], dtype=np.float32
            )
            cache["has_native_mass"][method_index] = True
            strict_replacement = {
                "replaced_method": "USOT C_y=0.3",
                "display_method": "COATI unbalanced",
                "cy": strict_loo1_cy,
                "endpoint_index": strict_loo1["endpoint_index"],
                "endpoint_time": strict_loo1["endpoint_time"],
                "initial_rna_max_abs_vs_shared_e75": strict_initial_max_abs,
                "trajectories": strict_loo1["paths"],
            }
        rna_neighbors = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(
            rna_references[stage_index]
        )
        atac_neighbors = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(
            atac_references[stage_index]
        )
        stage = str(processed.obs.iloc[heldout_rows[0]]["stage"])

        for cache_index, raw_method in enumerate(cache["methods"]):
            if raw_method not in METHOD_NAMES:
                continue
            method = METHOD_NAMES[raw_method]
            rna_indices = rna_neighbors.kneighbors(
                cache["rna_predictions"][cache_index], return_distance=False
            )
            atac_indices = atac_neighbors.kneighbors(
                cache["atac_predictions"][cache_index], return_distance=False
            )
            particle_rna = heldout_rna[rna_indices[:, 0]]
            particle_atac = heldout_atac[atac_indices[:, 0]]
            memberships = _memberships(rna_indices, heldout_labels)
            weights = _particle_weights(
                cache["native_log_masses"][cache_index]
                if cache["has_native_mass"][cache_index]
                else None,
                particle_rna.shape[0],
            )
            pred_nmp_rna, nmp_mass, nmp_effective_n = _cohort_mean(
                particle_rna, memberships["NMP"], weights
            )
            pred_somitic_rna, somitic_mass, somitic_effective_n = _cohort_mean(
                particle_rna, memberships["Somitic mesoderm"], weights
            )
            pred_nmp_atac, _, _ = _cohort_mean(
                particle_atac, memberships["NMP"], weights
            )
            pred_somitic_atac, _, _ = _cohort_mean(
                particle_atac, memberships["Somitic mesoderm"], weights
            )
            predicted_rna = pred_somitic_rna - pred_nmp_rna
            predicted_atac = pred_somitic_atac - pred_nmp_atac
            rna_cosine = _cosine(predicted_rna, observed_rna)
            atac_cosine = _cosine(
                predicted_atac[active_peak_indices],
                observed_atac[active_peak_indices],
            )
            metric_rows.append(
                {
                    "stage": stage,
                    "method": method,
                    "rna_cosine": rna_cosine,
                    "atac_cosine": atac_cosine,
                    "n_module_genes": len(gene_names),
                    "n_atac_linked_genes": len(active_gene_indices),
                    "n_atac_active_peaks": len(active_peak_indices),
                    "n_atac_active_links": len(active),
                    "native_mass_weighted": bool(cache["has_native_mass"][cache_index]),
                    "predicted_nmp_mass": nmp_mass,
                    "predicted_somitic_mass": somitic_mass,
                    "predicted_nmp_effective_n": nmp_effective_n,
                    "predicted_somitic_effective_n": somitic_effective_n,
                }
            )
            for modality, feature_names, observed, predicted, indices in (
                (
                    "RNA",
                    gene_names,
                    observed_rna,
                    predicted_rna,
                    np.arange(len(gene_names)),
                ),
                (
                    "ATAC",
                    peak_names,
                    observed_atac,
                    predicted_atac,
                    active_peak_indices,
                ),
            ):
                for index in indices:
                    vector_rows.append(
                        {
                            "stage": stage,
                            "method": method,
                            "modality": modality,
                            "feature": feature_names[index],
                            "observed_difference_z": float(observed[index]),
                            "predicted_difference_z": float(predicted[index]),
                        }
                    )
        manifest_tasks[task_name] = {
            "stage": stage,
            "prediction_cache": str(cache["path"].item()),
            "n_heldout_cells": int(len(heldout_rows)),
            "n_nmp_cells": int(nmp.sum()),
            "n_somitic_cells": int(somitic.sum()),
            "n_training_cells": int(len(training_rows)),
            "n_knn_aggregates": int(len(aggregate_metadata)),
            "n_cis_candidates": int(len(links)),
            "n_training_selected_links": int(
                selected_links["selected_training_only"].sum()
            ),
            "n_active_links": int(len(active)),
            "n_active_peaks": int(len(active_peak_indices)),
            "n_active_genes": int(len(active_gene_indices)),
            "active_genes": sorted(
                active["gene"].astype(str).unique().tolist()
            ),
            "strict_coati_unbalanced_replacement": strict_replacement,
        }

    processed.file.close()
    raw_rna.file.close()
    raw_atac.file.close()
    metrics = pd.DataFrame.from_records(metric_rows)
    vectors = pd.DataFrame.from_records(vector_rows)
    active_links = pd.concat(active_frames, ignore_index=True)
    manifest = {
        "genes": genes,
        "gene_tss_source": str(TSS_TABLE),
        "rna_definition": "All eight prespecified genes",
        "atac_definition": (
            "Unique cis peaks linked to the eight genes and selected without the "
            "held-out stage using low-overlap KNN aggregates; r>=0.45, Pearson "
            "BH FDR<=1e-4, permutation-null BH FDR<=0.05, and branch-coherent "
            "training effects."
        ),
        "cohort_definition": (
            "RNA 1NN defines particle NMP/somitic membership; the same weighted "
            "particles are read out with RNA and ATAC 1NN mappings."
        ),
        "excluded_methods": ["TIGON 20k"],
        "coati_unbalanced_protocol": (
            "E8.0 uses strict LOO1 C_y=0.3 primary, secondary, and native-mass "
            "trajectories supplied by the user; E8.5 retains the original LOO2 "
            "COATI-unbalanced cache entry."
        ),
        "tasks": manifest_tasks,
    }
    return metrics, vectors, active_links, manifest


def _plot(metrics: pd.DataFrame, output_dir: Path) -> None:
    apply_nature_rc(font_size=7.5)
    stages = metrics["stage"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        len(stages),
        2,
        figsize=(6.55, 5.05),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    y = np.arange(len(METHOD_ORDER))
    for row, stage in enumerate(stages):
        stage_data = metrics[metrics["stage"].eq(stage)].set_index("method")
        for column, (metric, modality) in enumerate(
            (("rna_cosine", "RNA"), ("atac_cosine", "ATAC"))
        ):
            axis = axes[row, column]
            frame = stage_data.loc[list(METHOD_ORDER)]
            values = frame[metric].to_numpy(float)
            colors = [method_style(method).color for method in METHOD_ORDER]
            bars = axis.barh(
                y,
                values,
                height=0.62,
                color=colors,
                edgecolor="#222222",
                linewidth=0.35,
                zorder=2,
            )
            axis.axvline(0.0, color="#444444", linewidth=0.7, zorder=1)
            axis.axvline(1.0, color="#777777", linestyle="--", linewidth=0.7)
            axis.set_xlim(-1.03, 1.20)
            axis.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            axis.grid(axis="x", color="#D9D9D9", linewidth=0.55, zorder=0)
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.tick_params(axis="y", length=0)
            axis.set_title(f"{stage} · {modality}", pad=4)
            axis.set_xlabel("Cosine similarity")
            for bar, value in zip(bars, values):
                axis.text(
                    1.055,
                    bar.get_y() + bar.get_height() / 2,
                    "NA" if not np.isfinite(value) else f"{value:.2f}",
                    ha="left",
                    va="center",
                    fontsize=7,
                )
    axes[0, 0].set_yticks(y, METHOD_ORDER)
    axes[0, 0].invert_yaxis()
    for axis in axes.flat[1:]:
        axis.tick_params(labelleft=False)
    fig.suptitle(
        "NMP-to-somitic direction recovery with the 8-gene core",
        fontsize=10,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.94,
        "E8.0 COATI-unbalanced: strict LOO1 (C_y=0.3) · RNA genes and linked ATAC peaks",
        ha="center",
        va="center",
        fontsize=7.2,
    )
    fig.subplots_adjust(left=0.255, right=0.985, bottom=0.105, top=0.88, hspace=0.39, wspace=0.12)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output_dir / f"somitic_core_8gene_rna_atac_cosine_k1.{suffix}",
            dpi=300 if suffix == "png" else None,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-list", type=Path, default=DEFAULT_GENE_LIST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--strict-loo1-dir", type=Path, default=DEFAULT_STRICT_LOO1_DIR
    )
    parser.add_argument("--strict-loo1-cy", type=float, default=0.3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    genes = _read_genes(args.gene_list.resolve())
    metrics, vectors, active_links, manifest = _evaluate(
        genes,
        args.cache_dir.resolve(),
        output_dir,
        args.strict_loo1_dir.resolve(),
        args.strict_loo1_cy,
    )
    metrics.to_csv(output_dir / "somitic_core_8gene_rna_atac_cosine_values.csv", index=False)
    vectors.to_csv(output_dir / "somitic_core_8gene_rna_atac_vectors.csv", index=False)
    active_links.to_csv(output_dir / "somitic_core_8gene_active_cis_links.csv", index=False)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _plot(metrics, output_dir)
    print(metrics.to_string(index=False))
    print(json.dumps(manifest["tasks"], indent=2))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
