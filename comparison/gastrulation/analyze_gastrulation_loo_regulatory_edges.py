#!/usr/bin/env python
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
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm, t as student_t
from sklearn.neighbors import NearestNeighbors
import torch


ROOT = Path(__file__).resolve().parents[2]
TRAINF_ROOT = Path("external/COATI")
GASTRULATION_ROOT = TRAINF_ROOT / "Gastrulation"
DATA = GASTRULATION_ROOT / "data"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/gastrulation_loo_regulatory_edges_archr_k50_weighted_latest"
)
ATAC_LSI = DATA / "atac_lsi_by_time_14D.npz"

for path in (ROOT / "common", TRAINF_ROOT):
    if path.exists():
        sys.path.insert(0, str(path))

from evaluate_gastrulation_full_cmcc import (  # noqa: E402
    _apply_t,
    _load_references,
    _load_t_model,
)


POSTERIOR_CELLTYPES = {
    "Primitive Streak",
    "Caudal epiblast",
    "Caudal neurectoderm",
    "Caudal Mesoderm",
    "Nascent mesoderm",
    "Mixed mesoderm",
    "NMP",
    "Notochord",
    "Paraxial mesoderm",
    "Somitic mesoderm",
    "Spinal cord",
}
BRANCHES = {
    "NMP to spinal cord": ("NMP", "Spinal cord"),
    "NMP to somitic mesoderm": ("NMP", "Somitic mesoderm"),
}


@dataclass(frozen=True)
class LooTask:
    name: str
    stage_key: str
    stage_index: int
    physical_time: float
    cache: Path


TASKS = (
    LooTask(
        name="time1",
        stage_key="time1",
        stage_index=1,
        physical_time=1.0,
        cache=(
            ROOT
            / "results/gastrulation_loo_latest_external_predictions"
            / "loo_time1_latest_external_predictions.npz"
        ),
    ),
    LooTask(
        name="time2",
        stage_key="time2",
        stage_index=2,
        physical_time=2.0,
        cache=(
            ROOT
            / "results/gastrulation_loo_latest_external_predictions"
            / "loo_time2_latest_external_predictions.npz"
        ),
    ),
)


def _load_selected_matrix(
    adata: ad.AnnData,
    row_positions: np.ndarray,
    feature_positions: np.ndarray,
    *,
    chunk_size: int = 512,
) -> sparse.csr_matrix:
    blocks: list[sparse.csr_matrix] = []
    feature_positions = np.asarray(feature_positions, dtype=int)
    for start in range(0, len(row_positions), chunk_size):
        rows = np.asarray(row_positions[start : start + chunk_size], dtype=int)
        block = adata.X[rows]
        if not sparse.issparse(block):
            block = sparse.csr_matrix(block)
        blocks.append(block[:, feature_positions].tocsr())
    return sparse.vstack(blocks, format="csr")


def _normalize_rna(
    matrix: sparse.csr_matrix, size_factors: np.ndarray
) -> sparse.csr_matrix:
    result = matrix.multiply(
        1.0 / np.maximum(np.asarray(size_factors, dtype=np.float64), 1e-12)[:, None]
    ).tocsr()
    result.data = np.log1p(result.data)
    return result


def _binarize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    result = matrix.astype(np.float32).tocsr(copy=True)
    result.data = np.ones_like(result.data, dtype=np.float32)
    return result


PEAK_PATTERN = re.compile(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")


def _cis_candidate_links(
    seed_links: pd.DataFrame,
    peak_names: pd.Index,
    max_distance: int,
) -> pd.DataFrame:
    """Expand the fixed NMP gene set to every cis peak within max_distance."""
    genes = (
        seed_links[["gene", "chrom", "tss_pos"]]
        .drop_duplicates()
        .sort_values("gene")
    )
    if genes["gene"].duplicated().any():
        raise ValueError("A candidate gene has multiple TSS definitions")

    peak_records: list[tuple[str, str, int]] = []
    for peak in peak_names.astype(str):
        match = PEAK_PATTERN.fullmatch(peak)
        if match is None:
            continue
        start = int(match.group("start"))
        end = int(match.group("end"))
        peak_records.append(
            (peak, match.group("chrom"), int(round((start + end) / 2)))
        )
    peak_frame = pd.DataFrame(
        peak_records, columns=["peak", "chrom", "peak_center"]
    )
    by_chrom = {
        chrom: frame.sort_values("peak_center").reset_index(drop=True)
        for chrom, frame in peak_frame.groupby("chrom", sort=False)
    }

    records: list[dict[str, object]] = []
    for gene in genes.itertuples(index=False):
        chrom_peaks = by_chrom.get(str(gene.chrom))
        if chrom_peaks is None:
            continue
        centers = chrom_peaks["peak_center"].to_numpy(np.int64)
        left = int(np.searchsorted(centers, int(gene.tss_pos) - max_distance))
        right = int(
            np.searchsorted(
                centers,
                int(gene.tss_pos) + max_distance,
                side="right",
            )
        )
        for peak in chrom_peaks.iloc[left:right].itertuples(index=False):
            signed_distance = int(peak.peak_center) - int(gene.tss_pos)
            absolute_distance = abs(signed_distance)
            records.append(
                {
                    "gene": str(gene.gene),
                    "peak": str(peak.peak),
                    "chrom": str(gene.chrom),
                    "tss_pos": int(gene.tss_pos),
                    "distance": signed_distance,
                    "absolute_distance": absolute_distance,
                    "is_promoter": absolute_distance <= 1_000,
                    "region_type": (
                        "promoter" if absolute_distance <= 1_000 else "distal"
                    ),
                }
            )
    result = pd.DataFrame.from_records(records)
    result = result.drop_duplicates(["gene", "peak"]).reset_index(drop=True)
    if result.empty:
        raise ValueError("No cis peak-gene candidate pairs were constructed")
    return result


def _candidate_feature_matrices(
    processed: ad.AnnData,
    raw_rna: ad.AnnData,
    raw_atac: ad.AnnData,
    links: pd.DataFrame,
) -> tuple[pd.DataFrame, sparse.csr_matrix, sparse.csr_matrix, list[str], list[str]]:
    links = links.copy()
    links["gene"] = links["gene"].astype(str)
    links["peak"] = links["peak"].astype(str)
    gene_names = sorted(links["gene"].unique().tolist())
    peak_names = sorted(links["peak"].unique().tolist())
    gene_positions = raw_rna.var_names.get_indexer(gene_names)
    peak_positions = raw_atac.var_names.get_indexer(peak_names)
    if np.any(gene_positions < 0) or np.any(peak_positions < 0):
        raise ValueError("Some candidate genes or peaks are absent from the raw matrices")

    raw_rna_rows = raw_rna.obs_names.get_indexer(processed.obs_names)
    raw_atac_rows = raw_atac.obs_names.get_indexer(processed.obs_names)
    if np.any(raw_rna_rows < 0) or np.any(raw_atac_rows < 0):
        raise ValueError("Processed cells are not all present in the paired raw matrices")

    print("[features] loading selected RNA genes", flush=True)
    rna = _load_selected_matrix(raw_rna, raw_rna_rows, gene_positions)
    rna = _normalize_rna(rna, processed.obs["sizeFactor"].to_numpy(float))
    print("[features] loading selected ATAC peaks", flush=True)
    atac = _load_selected_matrix(raw_atac, raw_atac_rows, peak_positions)
    atac = _binarize(atac)

    gene_lookup = {gene: idx for idx, gene in enumerate(gene_names)}
    peak_lookup = {peak: idx for idx, peak in enumerate(peak_names)}
    links["gene_index"] = links["gene"].map(gene_lookup).astype(int)
    links["peak_index"] = links["peak"].map(peak_lookup).astype(int)
    return links, rna, atac, gene_names, peak_names


def _load_joint_embedding(processed: ad.AnnData) -> np.ndarray:
    atac_npz = np.load(ATAC_LSI, allow_pickle=False)
    atac_lsi = np.concatenate(
        [
            np.asarray(atac_npz[key], dtype=np.float32)
            for key in ("time0", "time1", "time2", "time3")
        ],
        axis=0,
    )
    rna_pca = np.asarray(processed.obsm["X_pca"], dtype=np.float32)
    if len(atac_lsi) != processed.n_obs:
        raise ValueError(
            f"ATAC LSI rows {len(atac_lsi)} != processed cells {processed.n_obs}"
        )
    rna_pca = rna_pca[:, : min(30, rna_pca.shape[1])]
    return np.concatenate([rna_pca, atac_lsi], axis=1)


def _knn_aggregates(
    processed: ad.AnnData,
    rna: sparse.csr_matrix,
    atac: sparse.csr_matrix,
    heldout_time: float,
    k: int,
    iterations: int,
    overlap_cutoff: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Construct ArchR-style low-overlap KNN aggregates without label grouping."""
    obs = processed.obs
    mask = (
        ~np.isclose(obs["processed_time"].to_numpy(float), heldout_time)
        & obs["celltype"].astype(str).isin(POSTERIOR_CELLTYPES).to_numpy()
    )
    selected = np.flatnonzero(mask)
    if len(selected) < k:
        raise ValueError(f"Only {len(selected)} training cells for KNN k={k}")

    embedding = _load_joint_embedding(processed)[selected]
    n_rna = min(30, np.asarray(processed.obsm["X_pca"]).shape[1])
    rna_block = embedding[:, :n_rna]
    atac_block = embedding[:, n_rna:]
    for block in (rna_block, atac_block):
        block -= block.mean(axis=0, keepdims=True)
        block /= np.maximum(block.std(axis=0, keepdims=True), 1e-6)
    rna_block /= np.sqrt(max(rna_block.shape[1], 1))
    atac_block /= np.sqrt(max(atac_block.shape[1], 1))
    joint = np.concatenate([rna_block, atac_block], axis=1)

    neighbor_model = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(joint)
    rng = np.random.default_rng(seed)
    candidate_centers = rng.permutation(len(selected))[: min(iterations, len(selected))]
    neighbor_sets = neighbor_model.kneighbors(
        joint[candidate_centers], return_distance=False
    )
    accepted: list[np.ndarray] = []
    accepted_sets: list[set[int]] = []
    accepted_centers: list[int] = []
    for center, neighbors in zip(candidate_centers, neighbor_sets):
        neighbor_set = set(np.asarray(neighbors, dtype=int).tolist())
        if any(
            len(neighbor_set.intersection(previous)) / k > overlap_cutoff
            for previous in accepted_sets
        ):
            continue
        accepted.append(np.asarray(neighbors, dtype=int))
        accepted_sets.append(neighbor_set)
        accepted_centers.append(int(center))
    if len(accepted) < 30:
        raise ValueError(f"Only {len(accepted)} low-overlap KNN aggregates")

    group_count = len(accepted)
    assignment = sparse.csr_matrix(
        (
            np.full(group_count * k, 1.0 / k, dtype=np.float32),
            (
                np.repeat(np.arange(group_count), k),
                np.concatenate(accepted),
            ),
        ),
        shape=(group_count, len(selected)),
    )
    rna_profiles = np.asarray(
        (assignment @ rna[selected]).toarray(), dtype=np.float64
    )
    atac_profiles = np.asarray(
        (assignment @ atac[selected]).toarray(), dtype=np.float64
    )

    rows: list[dict[str, object]] = []
    for aggregate_index, (center, local_members) in enumerate(
        zip(accepted_centers, accepted)
    ):
        members = selected[local_members]
        celltypes = obs.iloc[members]["celltype"].astype(str)
        samples = obs.iloc[members]["sample_name"].astype(str)
        stages = obs.iloc[members]["stage"].astype(str)
        dominant_celltype = celltypes.value_counts().index[0]
        rows.append(
            {
                "aggregate": aggregate_index,
                "center_cell": str(obs.index[selected[center]]),
                "n_cells": int(len(members)),
                "dominant_celltype": str(dominant_celltype),
                "dominant_celltype_fraction": float(
                    np.mean(celltypes.to_numpy() == dominant_celltype)
                ),
                "n_celltypes": int(celltypes.nunique()),
                "n_samples": int(samples.nunique()),
                "n_stages": int(stages.nunique()),
            }
        )
    return rna_profiles, atac_profiles, pd.DataFrame(rows), selected


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=np.float64)
    result = np.full(pvalues.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(pvalues)
    if not finite.any():
        return result
    values = pvalues[finite]
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result[finite] = restored
    return result


def _select_training_links(
    links: pd.DataFrame,
    aggregate_rna: np.ndarray,
    aggregate_atac: np.ndarray,
    min_correlation: float,
    max_fdr: float,
    permutations: int,
    max_permutation_fdr: float,
    seed: int,
) -> pd.DataFrame:
    n_aggregates = aggregate_rna.shape[0]
    rna_centered = aggregate_rna - aggregate_rna.mean(axis=0, keepdims=True)
    atac_centered = aggregate_atac - aggregate_atac.mean(axis=0, keepdims=True)
    rna_z = rna_centered / np.maximum(
        aggregate_rna.std(axis=0, ddof=1, keepdims=True), 1e-12
    )
    atac_z = atac_centered / np.maximum(
        aggregate_atac.std(axis=0, ddof=1, keepdims=True), 1e-12
    )
    gene_indices = links["gene_index"].to_numpy(int)
    peak_indices = links["peak_index"].to_numpy(int)
    rna_edges = rna_z[:, gene_indices]
    atac_edges = atac_z[:, peak_indices]
    correlations = np.sum(rna_edges * atac_edges, axis=0) / (n_aggregates - 1)
    correlations = np.clip(correlations, -1.0, 1.0)
    statistic = correlations * np.sqrt(
        (n_aggregates - 2)
        / np.maximum(1.0 - np.square(correlations), 1e-15)
    )
    pvalues = 2.0 * student_t.sf(np.abs(statistic), df=n_aggregates - 2)

    rng = np.random.default_rng(seed)
    null = np.empty((permutations, len(links)), dtype=np.float32)
    for permutation in range(permutations):
        order = rng.permutation(n_aggregates)
        null[permutation] = np.sum(
            rna_edges * atac_edges[order], axis=0
        ) / (n_aggregates - 1)
    null_mean = null.mean(axis=0, dtype=np.float64)
    null_sd = np.maximum(null.std(axis=0, ddof=1, dtype=np.float64), 1e-12)
    permutation_z = (correlations - null_mean) / null_sd
    permutation_pvalues = 2.0 * norm.sf(np.abs(permutation_z))
    permutation_exceedance = (
        1.0
        + np.sum(np.abs(null) >= np.abs(correlations)[None, :], axis=0)
    ) / (permutations + 1.0)

    result = links.copy()
    result["training_knn_pearson_r"] = correlations
    result["training_knn_pearson_pvalue"] = pvalues
    result["training_knn_pearson_fdr"] = _benjamini_hochberg(pvalues)
    result["permutation_null_mean"] = null_mean
    result["permutation_null_sd"] = null_sd
    result["permutation_zscore"] = permutation_z
    result["permutation_parametric_pvalue"] = permutation_pvalues
    result["permutation_parametric_fdr"] = _benjamini_hochberg(
        permutation_pvalues
    )
    result["permutation_empirical_exceedance"] = permutation_exceedance
    result["selected_training_only"] = (
        result["training_knn_pearson_r"].ge(min_correlation)
        & result["training_knn_pearson_fdr"].le(max_fdr)
        & result["permutation_parametric_fdr"].le(max_permutation_fdr)
    )
    result["selection_fallback"] = False
    return result


def _standardization(
    aggregate_rna: np.ndarray,
    aggregate_atac: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rna_mean = aggregate_rna.mean(axis=0)
    rna_sd = aggregate_rna.std(axis=0, ddof=1)
    atac_mean = aggregate_atac.mean(axis=0)
    atac_sd = aggregate_atac.std(axis=0, ddof=1)
    rna_sd = np.maximum(rna_sd, 1e-3)
    atac_sd = np.maximum(atac_sd, 1e-3)
    return rna_mean, rna_sd, atac_mean, atac_sd


def _dense_standardized(
    matrix: sparse.csr_matrix,
    rows: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    values = np.asarray(matrix[rows].toarray(), dtype=np.float32)
    values -= np.asarray(center, dtype=np.float32)[None, :]
    values /= np.asarray(scale, dtype=np.float32)[None, :]
    return values


def _standardized_mean(
    matrix: sparse.csr_matrix,
    rows: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    if len(rows) == 0:
        return np.full(matrix.shape[1], np.nan, dtype=np.float64)
    mean = np.asarray(matrix[rows].mean(axis=0), dtype=np.float64).reshape(-1)
    return (mean - center) / scale


def _particle_weights(log_mass: np.ndarray | None, n: int) -> np.ndarray:
    if log_mass is None or not np.isfinite(log_mass).all():
        return np.full(n, 1.0 / n, dtype=np.float64)
    values = np.asarray(log_mass, dtype=np.float64).reshape(-1)
    values = np.exp(values - np.max(values))
    return values / values.sum()


def _load_predictions(
    task: LooTask,
    rna_reference_time0: np.ndarray,
    t_model: torch.nn.Module,
    batch_size: int,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    cache = np.load(task.cache, allow_pickle=False)
    methods = cache["methods"].astype(str).tolist()
    rna_predictions = np.asarray(cache["rna_predictions"], dtype=np.float32)
    atac_predictions = np.asarray(cache["atac_predictions"], dtype=np.float32)
    log_masses = np.asarray(cache["native_log_masses"], dtype=np.float32)
    has_mass = np.asarray(cache["has_native_mass"], dtype=bool)
    if not np.isclose(float(cache["physical_time"]), task.physical_time):
        raise ValueError(f"{task.name} prediction cache has the wrong physical time")

    provenance: dict[str, object] = {
        "prediction_cache": str(task.cache),
        "cache_methods": methods.copy(),
    }

    no_sync_specs = [
        (
            "USOT RNA-only (no Sync)",
            GASTRULATION_ROOT
            / f"unbalanced_LOO_result/{task.name}/RNA-only"
            / "primary_trajectory_unbalanced_rnaonly_bionum_s0_iter20000.pt",
            GASTRULATION_ROOT
            / f"unbalanced_LOO_result/{task.name}/RNA-only"
            / "mass_lnw_trajectory_unbalanced_rnaonly_bionum_s0_iter20000.pt",
        )
    ]
    if task.name == "time1":
        no_sync_specs.append(
            (
                "BSOT RNA-only (no Sync)",
                GASTRULATION_ROOT
                / "balancedLOO/trajectory/primary_trajectory_balanced_loo_s0_iter20000.pt",
                None,
            )
        )

    appended = []
    for name, trajectory_path, mass_path in no_sync_specs:
        if not trajectory_path.exists():
            continue
        trajectory = torch.load(trajectory_path, map_location="cpu", weights_only=False)
        trajectory = np.asarray(trajectory, dtype=np.float32)
        endpoint_index = int(round(task.physical_time * 10))
        if trajectory.shape[0] <= endpoint_index:
            raise ValueError(f"{trajectory_path} has no endpoint index {endpoint_index}")
        initial_max_abs = float(
            np.max(np.abs(trajectory[0] - np.asarray(rna_reference_time0)))
        )
        if initial_max_abs > 1e-4:
            raise ValueError(
                f"{name} does not start from the shared normalized E7.5 input: "
                f"max abs {initial_max_abs:.3g}"
            )
        endpoint = trajectory[endpoint_index]
        mapped = _apply_t(
            t_model,
            endpoint,
            task.physical_time,
            torch.device("cpu"),
            batch_size,
        )
        if mass_path is None:
            endpoint_mass = np.full(endpoint.shape[0], np.nan, dtype=np.float32)
            native = False
        else:
            mass_trajectory = torch.load(
                mass_path, map_location="cpu", weights_only=False
            )
            endpoint_mass = np.asarray(mass_trajectory, dtype=np.float32)[
                endpoint_index, :, 0
            ]
            native = True
        methods.append(name)
        rna_predictions = np.concatenate(
            [rna_predictions, endpoint[None, :, :]], axis=0
        )
        atac_predictions = np.concatenate(
            [atac_predictions, np.asarray(mapped, dtype=np.float32)[None, :, :]],
            axis=0,
        )
        log_masses = np.concatenate([log_masses, endpoint_mass[None, :]], axis=0)
        has_mass = np.concatenate([has_mass, np.asarray([native], dtype=bool)])
        appended.append(
            {
                "method": name,
                "trajectory": str(trajectory_path),
                "mass": None if mass_path is None else str(mass_path),
                "endpoint_index": endpoint_index,
                "initial_max_abs_vs_shared_input": initial_max_abs,
            }
        )
    provenance["same_architecture_no_sync"] = appended
    return methods, rna_predictions, atac_predictions, log_masses, has_mass, provenance


def _knn_readout(
    neighbor_indices: np.ndarray,
    reference_values: np.ndarray,
) -> np.ndarray:
    return np.asarray(reference_values[neighbor_indices].mean(axis=1), dtype=np.float32)


def _memberships(neighbor_indices: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    neighbor_labels = labels[neighbor_indices]
    return {
        celltype: np.mean(neighbor_labels == celltype, axis=1).astype(np.float64)
        for celltype in {value for pair in BRANCHES.values() for value in pair}
    }


def _cohort_mean(
    values: np.ndarray,
    membership: np.ndarray,
    particle_weights: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    weights = np.asarray(membership, dtype=np.float64) * particle_weights
    mass = float(weights.sum())
    if mass <= 1e-12:
        return np.full(values.shape[1], np.nan), 0.0, 0.0
    normalized = weights / mass
    effective_n = float(1.0 / np.sum(np.square(normalized)))
    return normalized @ values, mass, effective_n


def _cosine(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(x, y) / denom)


def _edge_metrics(
    predicted_gene_delta: np.ndarray,
    predicted_peak_delta: np.ndarray,
    true_gene_delta: np.ndarray,
    true_peak_delta: np.ndarray,
    active_links: pd.DataFrame,
) -> dict[str, float]:
    if (
        not np.isfinite(predicted_gene_delta).all()
        or not np.isfinite(predicted_peak_delta).all()
    ):
        return {
            "rna_transition_cosine": np.nan,
            "atac_transition_cosine": np.nan,
            "joint_transition_cosine": np.nan,
            "joint_sign_accuracy": np.nan,
            "predicted_peak_gene_alignment": np.nan,
            "real_peak_gene_alignment": np.nan,
            "peak_gene_alignment_abs_error": np.nan,
            "regulatory_recovery_score": np.nan,
            "joint_delta_mae": np.nan,
        }
    gene_indices = active_links["gene_index"].to_numpy(int)
    peak_indices = active_links["peak_index"].to_numpy(int)
    unique_genes = np.unique(gene_indices)
    unique_peaks = np.unique(peak_indices)
    pred_gene_edges = predicted_gene_delta[gene_indices]
    pred_peak_edges = predicted_peak_delta[peak_indices]
    true_gene_edges = true_gene_delta[gene_indices]
    true_peak_edges = true_peak_delta[peak_indices]

    rna_cosine = _cosine(
        predicted_gene_delta[unique_genes], true_gene_delta[unique_genes]
    )
    atac_cosine = _cosine(
        predicted_peak_delta[unique_peaks], true_peak_delta[unique_peaks]
    )
    joint_cosine = _cosine(
        np.concatenate([pred_gene_edges, pred_peak_edges]),
        np.concatenate([true_gene_edges, true_peak_edges]),
    )
    joint_sign = float(
        np.mean(
            (pred_gene_edges * true_gene_edges > 0)
            & (pred_peak_edges * true_peak_edges > 0)
        )
    )
    pred_crossmodal = _cosine(pred_gene_edges, pred_peak_edges)
    true_crossmodal = _cosine(true_gene_edges, true_peak_edges)
    regulatory_recovery = float(
        np.nanmean(
            [
                (rna_cosine + 1.0) / 2.0,
                (atac_cosine + 1.0) / 2.0,
                joint_sign,
            ]
        )
    )
    return {
        "rna_transition_cosine": rna_cosine,
        "atac_transition_cosine": atac_cosine,
        "joint_transition_cosine": joint_cosine,
        "joint_sign_accuracy": joint_sign,
        "predicted_peak_gene_alignment": pred_crossmodal,
        "real_peak_gene_alignment": true_crossmodal,
        "peak_gene_alignment_abs_error": abs(pred_crossmodal - true_crossmodal),
        "regulatory_recovery_score": regulatory_recovery,
        "joint_delta_mae": float(
            np.mean(
                np.abs(
                    np.concatenate([pred_gene_edges, pred_peak_edges])
                    - np.concatenate([true_gene_edges, true_peak_edges])
                )
            )
        ),
    }


def _training_active_links_for_branch(
    selected_links: pd.DataFrame,
    training_gene_delta: np.ndarray,
    training_peak_delta: np.ndarray,
    minimum_effect: float,
) -> pd.DataFrame:
    selected = selected_links[selected_links["selected_training_only"]].copy()
    gene_delta = training_gene_delta[selected["gene_index"].to_numpy(int)]
    peak_delta = training_peak_delta[selected["peak_index"].to_numpy(int)]
    active = (
        (gene_delta * peak_delta > 0)
        & (np.abs(gene_delta) >= minimum_effect)
        & (np.abs(peak_delta) >= minimum_effect)
    )
    result = selected.loc[active].copy()
    if len(result) < 15:
        coherent = gene_delta * peak_delta > 0
        result = selected.loc[coherent].copy()
        result["active_effect_fallback"] = True
    else:
        result["active_effect_fallback"] = False
    result["training_gene_delta_z"] = training_gene_delta[
        result["gene_index"].to_numpy(int)
    ]
    result["training_peak_delta_z"] = training_peak_delta[
        result["peak_index"].to_numpy(int)
    ]
    return result


def _evaluate_task(
    task: LooTask,
    processed: ad.AnnData,
    rna_matrix: sparse.csr_matrix,
    atac_matrix: sparse.csr_matrix,
    links: pd.DataFrame,
    gene_names: list[str],
    peak_names: list[str],
    rna_references: list[np.ndarray],
    atac_references: list[np.ndarray],
    labels_by_stage: list[np.ndarray],
    t_model: torch.nn.Module,
    output_dir: Path,
    ks: list[int],
    batch_size: int,
    knn_k: int,
    knn_iterations: int,
    overlap_cutoff: float,
    min_training_correlation: float,
    max_training_fdr: float,
    permutations: int,
    max_permutation_fdr: float,
    minimum_training_effect: float,
) -> dict[str, object]:
    task_dir = output_dir / f"loo_{task.name}"
    task_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[{task.name}] constructing training-only low-overlap KNN aggregates",
        flush=True,
    )
    aggregate_rna, aggregate_atac, aggregate_metadata, training_rows = (
        _knn_aggregates(
            processed,
            rna_matrix,
            atac_matrix,
            task.physical_time,
            knn_k,
            knn_iterations,
            overlap_cutoff,
            10_000 + task.stage_index,
        )
    )
    aggregate_metadata.to_csv(
        task_dir / "training_knn_aggregates.csv", index=False
    )
    selected_links = _select_training_links(
        links,
        aggregate_rna,
        aggregate_atac,
        min_training_correlation,
        max_training_fdr,
        permutations,
        max_permutation_fdr,
        20_000 + task.stage_index,
    )
    selected_links.to_csv(
        task_dir / "training_only_peak_gene_links.csv", index=False
    )
    print(
        f"[{task.name}] selected "
        f"{int(selected_links['selected_training_only'].sum())} "
        f"of {len(selected_links)} cis candidate links",
        flush=True,
    )

    rna_center, rna_scale, atac_center, atac_scale = _standardization(
        aggregate_rna, aggregate_atac
    )
    obs_times = processed.obs["processed_time"].to_numpy(float)
    obs_celltypes = (
        processed.obs["celltype"]
        .astype("string")
        .fillna("Unknown")
        .to_numpy(dtype=str)
    )
    heldout_rows = np.flatnonzero(np.isclose(obs_times, task.physical_time))
    heldout_labels = obs_celltypes[heldout_rows]
    if not np.array_equal(heldout_labels, labels_by_stage[task.stage_index]):
        raise ValueError(f"{task.name} processed labels do not align with latent references")
    heldout_rna_values = _dense_standardized(
        rna_matrix, heldout_rows, rna_center, rna_scale
    )
    heldout_atac_values = _dense_standardized(
        atac_matrix, heldout_rows, atac_center, atac_scale
    )

    true_contrast_rows: list[dict[str, object]] = []
    active_by_branch: dict[str, pd.DataFrame] = {}
    true_delta_by_branch: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for branch, (nmp_label, target_label) in BRANCHES.items():
        training_nmp_rows = training_rows[
            obs_celltypes[training_rows] == nmp_label
        ]
        training_target_rows = training_rows[
            obs_celltypes[training_rows] == target_label
        ]
        if len(training_nmp_rows) < 20 or len(training_target_rows) < 20:
            raise ValueError(
                f"{task.name}/{branch} has too few training cells: "
                f"NMP={len(training_nmp_rows)}, target={len(training_target_rows)}"
            )
        training_gene_delta = _standardized_mean(
            rna_matrix,
            training_target_rows,
            rna_center,
            rna_scale,
        ) - _standardized_mean(
            rna_matrix,
            training_nmp_rows,
            rna_center,
            rna_scale,
        )
        training_peak_delta = _standardized_mean(
            atac_matrix,
            training_target_rows,
            atac_center,
            atac_scale,
        ) - _standardized_mean(
            atac_matrix,
            training_nmp_rows,
            atac_center,
            atac_scale,
        )
        nmp = heldout_labels == nmp_label
        target = heldout_labels == target_label
        if nmp.sum() < 20 or target.sum() < 20:
            raise ValueError(
                f"{task.name}/{branch} has too few real held-out cells: "
                f"NMP={nmp.sum()}, target={target.sum()}"
            )
        true_gene_delta = heldout_rna_values[target].mean(axis=0) - heldout_rna_values[
            nmp
        ].mean(axis=0)
        true_peak_delta = heldout_atac_values[target].mean(axis=0) - heldout_atac_values[
            nmp
        ].mean(axis=0)
        active = _training_active_links_for_branch(
            selected_links,
            training_gene_delta,
            training_peak_delta,
            minimum_training_effect,
        )
        active["heldout_true_gene_delta_z"] = true_gene_delta[
            active["gene_index"].to_numpy(int)
        ]
        active["heldout_true_peak_delta_z"] = true_peak_delta[
            active["peak_index"].to_numpy(int)
        ]
        active.insert(0, "branch", branch)
        active.to_csv(
            task_dir / f"active_links_{branch.lower().replace(' ', '_')}.csv",
            index=False,
        )
        active_by_branch[branch] = active
        true_delta_by_branch[branch] = (true_gene_delta, true_peak_delta)
        true_contrast_rows.extend(
            {
                "branch": branch,
                "feature_type": "RNA gene",
                "feature": gene,
                "delta_z": float(true_gene_delta[idx]),
            }
            for idx, gene in enumerate(gene_names)
        )
        true_contrast_rows.extend(
            {
                "branch": branch,
                "feature_type": "ATAC peak",
                "feature": peak,
                "delta_z": float(true_peak_delta[idx]),
            }
            for idx, peak in enumerate(peak_names)
        )
        print(
            f"[{task.name}] {branch}: {len(active)} training-only active links",
            flush=True,
        )
    pd.DataFrame.from_records(true_contrast_rows).to_csv(
        task_dir / "heldout_true_branch_contrasts.csv", index=False
    )

    methods, predicted_rna, predicted_atac, log_masses, has_mass, provenance = (
        _load_predictions(
            task,
            rna_references[0],
            t_model,
            batch_size,
        )
    )
    max_k = max(ks)
    rna_neighbors = NearestNeighbors(n_neighbors=max_k, n_jobs=-1).fit(
        rna_references[task.stage_index]
    )
    atac_neighbors = NearestNeighbors(n_neighbors=max_k, n_jobs=-1).fit(
        atac_references[task.stage_index]
    )

    rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    for method_index, method in enumerate(methods):
        print(f"[{task.name}] readout {method}", flush=True)
        rna_neighbor_indices = rna_neighbors.kneighbors(
            predicted_rna[method_index], return_distance=False
        )
        atac_neighbor_indices = atac_neighbors.kneighbors(
            predicted_atac[method_index], return_distance=False
        )
        particle_weights = _particle_weights(
            log_masses[method_index] if has_mass[method_index] else None,
            predicted_rna.shape[1],
        )
        for k in ks:
            rna_indices_k = rna_neighbor_indices[:, :k]
            atac_indices_k = atac_neighbor_indices[:, :k]
            particle_rna_values = _knn_readout(rna_indices_k, heldout_rna_values)
            particle_atac_values = _knn_readout(atac_indices_k, heldout_atac_values)
            membership = _memberships(rna_indices_k, heldout_labels)
            for branch, (nmp_label, target_label) in BRANCHES.items():
                pred_nmp_rna, nmp_mass, nmp_effective = _cohort_mean(
                    particle_rna_values, membership[nmp_label], particle_weights
                )
                pred_target_rna, target_mass, target_effective = _cohort_mean(
                    particle_rna_values, membership[target_label], particle_weights
                )
                pred_nmp_atac, _, _ = _cohort_mean(
                    particle_atac_values, membership[nmp_label], particle_weights
                )
                pred_target_atac, _, _ = _cohort_mean(
                    particle_atac_values, membership[target_label], particle_weights
                )
                predicted_gene_delta = pred_target_rna - pred_nmp_rna
                predicted_peak_delta = pred_target_atac - pred_nmp_atac
                true_gene_delta, true_peak_delta = true_delta_by_branch[branch]
                active = active_by_branch[branch]
                metrics = _edge_metrics(
                    predicted_gene_delta,
                    predicted_peak_delta,
                    true_gene_delta,
                    true_peak_delta,
                    active,
                )
                row = {
                    "loo_task": task.name,
                    "heldout_stage": str(
                        processed.obs.iloc[heldout_rows[0]]["stage"]
                    ),
                    "method": method,
                    "k": k,
                    "branch": branch,
                    "n_active_links": int(len(active)),
                    "n_active_genes": int(active["gene"].nunique()),
                    "n_active_peaks": int(active["peak"].nunique()),
                    "native_mass_weighted": bool(has_mass[method_index]),
                    "predicted_nmp_mass": nmp_mass,
                    "predicted_target_mass": target_mass,
                    "predicted_nmp_effective_n": nmp_effective,
                    "predicted_target_effective_n": target_effective,
                    **metrics,
                }
                rows.append(row)

                if k == 1:
                    for link in active.itertuples(index=False):
                        edge_rows.append(
                            {
                                "loo_task": task.name,
                                "method": method,
                                "branch": branch,
                                "gene": link.gene,
                                "peak": link.peak,
                                "training_knn_pearson_r": link.training_knn_pearson_r,
                                "training_knn_pearson_fdr": link.training_knn_pearson_fdr,
                                "permutation_parametric_fdr": (
                                    link.permutation_parametric_fdr
                                ),
                                "true_gene_delta_z": float(
                                    true_gene_delta[link.gene_index]
                                ),
                                "predicted_gene_delta_z": float(
                                    predicted_gene_delta[link.gene_index]
                                ),
                                "true_peak_delta_z": float(
                                    true_peak_delta[link.peak_index]
                                ),
                                "predicted_peak_delta_z": float(
                                    predicted_peak_delta[link.peak_index]
                                ),
                                "both_sign_correct": bool(
                                    predicted_gene_delta[link.gene_index]
                                    * true_gene_delta[link.gene_index]
                                    > 0
                                    and predicted_peak_delta[link.peak_index]
                                    * true_peak_delta[link.peak_index]
                                    > 0
                                ),
                            }
                        )
    metrics = pd.DataFrame.from_records(rows)
    metrics.to_csv(task_dir / "regulatory_metrics.csv", index=False)
    pd.DataFrame.from_records(edge_rows).to_csv(
        task_dir / "regulatory_edge_details_k1.csv", index=False
    )
    provenance.update(
        {
            "heldout_time": task.physical_time,
            "heldout_stage_key": task.stage_key,
            "n_heldout_cells": int(len(heldout_rows)),
            "n_training_cells_for_linking": int(len(training_rows)),
            "n_training_knn_aggregates": int(len(aggregate_metadata)),
            "n_candidate_links": int(len(links)),
            "n_training_selected_links": int(
                selected_links["selected_training_only"].sum()
            ),
            "active_links_by_branch": {
                branch: int(len(frame)) for branch, frame in active_by_branch.items()
            },
            "k_values": ks,
            "uncertainty_intervals": "none",
            "aggregation_definition": (
                f"Low-overlap KNN groups in an equally block-scaled joint "
                f"RNA-PCA/ATAC-LSI space; k={knn_k}, "
                f"candidate centers={knn_iterations}, "
                f"maximum pairwise overlap fraction={overlap_cutoff:g}."
            ),
            "link_selection_definition": (
                f"Cis distance <= {int(links['absolute_distance'].max())} bp; "
                f"training KNN-aggregate Pearson r >= "
                f"{min_training_correlation:g}; BH FDR <= "
                f"{max_training_fdr:g}; permutation-null parametric BH FDR <= "
                f"{max_permutation_fdr:g}; {permutations} permutations. "
                "No rank fallback is applied."
            ),
            "readout_definition": (
                "RNA kNN defines particle lineage membership. The same particles are "
                "read out in RNA and shared-T ATAC spaces; ATAC does not independently "
                "select a lineage cohort. Every predicted particle is retained and "
                "weighted by its normalized native mass when available; all real "
                "held-out cells are retained with equal weight. Raw gene/peak values "
                "are reference-based kNN readouts, not direct decoded peak counts."
            ),
            "feature_selection_definition": (
                "The NMP-relevant gene set is fixed before LOO. Cis candidates, link "
                "correlation, permutation null, branch activity, direction, and "
                "effect-size filtering use only non-held-out training cells. "
                "Held-out cells are used only as the evaluation target."
            ),
        }
    )
    with (task_dir / "manifest.json").open("w") as handle:
        json.dump(provenance, handle, indent=2)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate training-only peak-gene regulatory branch recovery for "
            "gastrulation LOO time1/time2 predictions."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 10, 20])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-cis-distance", type=int, default=250_000)
    parser.add_argument("--knn-k", type=int, default=50)
    parser.add_argument("--knn-iterations", type=int, default=500)
    parser.add_argument("--overlap-cutoff", type=float, default=0.8)
    parser.add_argument("--min-training-correlation", type=float, default=0.45)
    parser.add_argument("--max-training-fdr", type=float, default=1e-4)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--max-permutation-fdr", type=float, default=0.05)
    parser.add_argument("--minimum-training-effect", type=float, default=0.15)
    parser.add_argument(
        "--prediction-cache-time1",
        type=Path,
        default=TASKS[0].cache,
        help="Prediction cache for the held-out E8.0 task.",
    )
    parser.add_argument(
        "--prediction-cache-time2",
        type=Path,
        default=TASKS[1].cache,
        help="Prediction cache for the held-out E8.5 task.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processed = ad.read_h5ad(DATA / "gastrulation_rna_processed.h5ad", backed="r")
    raw_rna = ad.read_h5ad(DATA / "gastrulation_rna.h5ad", backed="r")
    raw_atac = ad.read_h5ad(DATA / "gastrulation_atac_peaks.h5ad", backed="r")
    seed_links = pd.read_csv(DATA / "nmp_peak_gene_links.csv")
    candidate_links = _cis_candidate_links(
        seed_links,
        raw_atac.var_names,
        args.max_cis_distance,
    )
    print(
        f"[candidates] {candidate_links['gene'].nunique()} genes, "
        f"{candidate_links['peak'].nunique()} peaks, "
        f"{len(candidate_links)} cis pairs within "
        f"{args.max_cis_distance:,} bp",
        flush=True,
    )
    links, rna_matrix, atac_matrix, gene_names, peak_names = (
        _candidate_feature_matrices(
            processed, raw_rna, raw_atac, candidate_links
        )
    )
    rna_references, atac_references, labels_by_stage, _, _ = _load_references()
    t_model = _load_t_model(torch.device("cpu"))

    tasks = (
        LooTask(
            name="time1",
            stage_key="time1",
            stage_index=1,
            physical_time=1.0,
            cache=args.prediction_cache_time1,
        ),
        LooTask(
            name="time2",
            stage_key="time2",
            stage_index=2,
            physical_time=2.0,
            cache=args.prediction_cache_time2,
        ),
    )
    manifests = {}
    for task in tasks:
        manifests[task.name] = _evaluate_task(
            task,
            processed,
            rna_matrix,
            atac_matrix,
            links,
            gene_names,
            peak_names,
            rna_references,
            atac_references,
            labels_by_stage,
            t_model,
            args.output_dir,
            sorted(set(args.k)),
            args.batch_size,
            args.knn_k,
            args.knn_iterations,
            args.overlap_cutoff,
            args.min_training_correlation,
            args.max_training_fdr,
            args.permutations,
            args.max_permutation_fdr,
            args.minimum_training_effect,
        )
    with (args.output_dir / "manifest.json").open("w") as handle:
        json.dump(manifests, handle, indent=2)
    print(f"wrote: {args.output_dir}", flush=True)

    processed.file.close()
    raw_rna.file.close()
    raw_atac.file.close()


if __name__ == "__main__":
    main()
