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
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata, t as student_t


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/gastrulation_preserved_growth_genes"

PROLIFERATION_GENES = [
    "Mki67", "Top2a", "Pcna", "Mcm2", "Mcm3", "Mcm4", "Mcm5", "Mcm6",
    "Mcm7", "Mcm10", "Cdc6", "Cdc20", "Cdc45", "Ccna2", "Ccnb1",
    "Ccnb2", "Cdk1", "Ube2c", "Cenpf", "Tpx2", "Birc5", "Nusap1",
    "Pclaf", "Rrm2", "Tyms", "Aurka", "Aurkb", "Plk1",
]
DNA_REPLICATION_GENES = [
    "Pcna", "Mcm2", "Mcm3", "Mcm4", "Mcm5", "Mcm6", "Mcm7", "Mcm10",
    "Cdc6", "Cdc45", "Pclaf", "Rrm2", "Tyms",
]
G2M_MITOSIS_GENES = [
    "Mki67", "Top2a", "Cdc20", "Ccna2", "Ccnb1", "Ccnb2", "Cdk1",
    "Ube2c", "Cenpf", "Tpx2", "Birc5", "Nusap1", "Aurka", "Aurkb", "Plk1",
]
APOPTOSIS_GENES = [
    "Bax", "Bak1", "Bbc3", "Pmaip1", "Bcl2l11", "Casp3", "Casp7",
    "Casp8", "Casp9", "Apaf1", "Bid", "Fas", "Fadd", "Tnfrsf10b",
    "Dffa", "Dffb",
]
MODULES = {
    "Proliferation/cell cycle": PROLIFERATION_GENES,
    "DNA replication/S phase": DNA_REPLICATION_GENES,
    "G2/M and mitosis": G2M_MITOSIS_GENES,
    "Apoptosis/stress": APOPTOSIS_GENES,
}
BASELINES = ["USOT RNA-only", "CytoBridge unbalanced", "TIGON"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Associate source-cell RNA with integrated net growth among adjacent-interval "
            "endpoint-retained Gastrulation trajectories."
        )
    )
    parser.add_argument(
        "--particle-table",
        type=Path,
        default=DEFAULT_OUTPUT / "particle_preserved_growth.csv.gz",
    )
    parser.add_argument(
        "--expression-h5ad",
        type=Path,
        default=ROOT / "data/gastrulation_rna_full.h5ad",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k", type=int, choices=[1, 20], default=20)
    parser.add_argument("--min-source", type=int, default=200)
    parser.add_argument("--min-retained", type=int, default=100)
    parser.add_argument("--min-per-sample", type=int, default=20)
    parser.add_argument("--min-detected-fraction", type=float, default=0.05)
    parser.add_argument("--min-valid-c-y", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    out = np.full(len(p), np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return out
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.minimum(ranked, 1.0)
    return out


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sd = np.std(x)
    return (x - np.mean(x)) / sd if sd > 1e-12 else np.zeros_like(x)


def build_q(obs: pd.DataFrame, include_sample: bool = True) -> np.ndarray:
    pieces: list[np.ndarray] = [np.ones((len(obs), 1), dtype=np.float64)]
    if include_sample:
        sample = pd.get_dummies(obs["sample_name"].astype(str), drop_first=True, dtype=float)
        if sample.shape[1]:
            pieces.append(sample.to_numpy(np.float64))
    pieces.append(
        standardize(
            np.log1p(pd.to_numeric(obs["nCount_RNA"], errors="coerce").fillna(0).to_numpy(float))
        )[:, None]
    )
    pieces.append(
        standardize(
            pd.to_numeric(obs["mitochondrial_percent_RNA"], errors="coerce")
            .fillna(0)
            .to_numpy(float)
        )[:, None]
    )
    design = np.concatenate(pieces, axis=1)
    keep = np.std(design, axis=0) > 1e-12
    keep[0] = True
    q, _ = np.linalg.qr(design[:, keep], mode="reduced")
    return q


def residualize(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    return x - q @ (q.T @ x)


def dense_float(x) -> np.ndarray:
    if sp.issparse(x):
        return x.toarray().astype(np.float64, copy=False)
    return np.asarray(x, dtype=np.float64)


def prepare_expression(
    expression,
    global_indices: np.ndarray,
    obs: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    raw = dense_float(expression[global_indices])
    detected = np.mean(raw > 0, axis=0)
    mean_expression = np.mean(raw, axis=0)
    ranked = rankdata(raw, axis=0, method="average")
    q = build_q(obs.iloc[global_indices], include_sample=True)
    xr = residualize(ranked, q)
    xr -= np.mean(xr, axis=0, keepdims=True)
    xnorm = np.linalg.norm(xr, axis=0)
    return raw, xr, xnorm, detected, mean_expression, q.shape[1]


def correlate_prepared(
    growth: np.ndarray,
    xr: np.ndarray,
    xnorm: np.ndarray,
    obs: pd.DataFrame,
    global_indices: np.ndarray,
    covariate_rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = rankdata(np.asarray(growth, dtype=np.float64), method="average")
    q = build_q(obs.iloc[global_indices], include_sample=True)
    yr = residualize(y[:, None], q).ravel()
    yr -= np.mean(yr)
    ynorm = np.linalg.norm(yr)
    denom = ynorm * xnorm
    rho = np.full(xr.shape[1], np.nan, dtype=np.float64)
    valid = denom > 1e-12
    rho[valid] = yr @ xr[:, valid] / denom[valid]
    df = max(len(growth) - covariate_rank - 2, 1)
    clipped = np.clip(rho, -0.999999, 0.999999)
    stat = clipped * np.sqrt(df / np.maximum(1.0 - clipped * clipped, 1e-12))
    p = 2.0 * student_t.sf(np.abs(stat), df=df)
    p[~np.isfinite(rho)] = np.nan
    return rho, p


def module_score_all_cells(
    adata: ad.AnnData,
    stages: np.ndarray,
    requested: list[str],
) -> tuple[np.ndarray, list[str]]:
    present = [gene for gene in requested if gene in adata.var_names]
    if not present:
        return np.full(adata.n_obs, np.nan), []
    values = dense_float(adata[:, present].X)
    score = np.zeros(adata.n_obs, dtype=np.float64)
    for stage in np.unique(stages):
        idx = np.flatnonzero(stages == stage)
        local = values[idx]
        sd = np.std(local, axis=0)
        sd[sd < 1e-12] = 1.0
        score[idx] = np.mean((local - np.mean(local, axis=0)) / sd, axis=1)
    return score, present


def partial_spearman_one(
    growth: np.ndarray,
    score: np.ndarray,
    obs_local: pd.DataFrame,
    include_sample: bool,
) -> float:
    if len(growth) < 5 or np.std(growth) < 1e-12 or np.std(score) < 1e-12:
        return np.nan
    q = build_q(obs_local, include_sample=include_sample)
    x = rankdata(score, method="average")
    y = rankdata(growth, method="average")
    xr = residualize(x[:, None], q).ravel()
    yr = residualize(y[:, None], q).ravel()
    denom = np.linalg.norm(xr) * np.linalg.norm(yr)
    return float(xr @ yr / denom) if denom > 1e-12 else np.nan


def cohort_valid(
    frame: pd.DataFrame,
    obs: pd.DataFrame,
    min_retained: int,
    min_per_sample: int,
) -> bool:
    if len(frame) < min_retained:
        return False
    counts = obs.iloc[frame["source_index"].to_numpy(int)]["sample_name"].value_counts()
    counts = counts[counts > 0]
    return len(counts) >= 2 and bool((counts >= min_per_sample).all())


def module_rows(
    frame: pd.DataFrame,
    obs: pd.DataFrame,
    module_scores: dict[str, np.ndarray],
    *,
    cohort_type: str,
    comparator: str | None = None,
) -> list[dict[str, object]]:
    idx = frame["source_index"].to_numpy(int)
    growth = frame["native_log_growth_rate"].to_numpy(float)
    local_obs = obs.iloc[idx]
    order = np.argsort(growth)
    tail = max(1, int(np.floor(0.2 * len(growth))))
    low = order[:tail]
    high = order[-tail:]
    rows: list[dict[str, object]] = []
    for module, score_all in module_scores.items():
        score = score_all[idx]
        row: dict[str, object] = {
            "cohort_type": cohort_type,
            "comparator": comparator,
            "method": str(frame["method"].iloc[0]),
            "C_y": frame["C_y"].iloc[0],
            "interval": str(frame["interval"].iloc[0]),
            "celltype": str(frame["source_celltype"].iloc[0]),
            "module": module,
            "n_cells": int(len(frame)),
            "growth_sd": float(np.std(growth)),
            "partial_spearman": partial_spearman_one(growth, score, local_obs, True),
            "high_minus_low_module_score": float(np.mean(score[high]) - np.mean(score[low])),
        }
        sample_values: list[float] = []
        for sample in local_obs["sample_name"].astype(str).unique():
            mask = local_obs["sample_name"].astype(str).to_numpy() == sample
            value = partial_spearman_one(
                growth[mask], score[mask], local_obs.iloc[np.flatnonzero(mask)], False
            )
            row[f"rho_{sample}"] = value
            if np.isfinite(value):
                sample_values.append(value)
        row["replicate_sign_agreement"] = (
            bool(len(sample_values) >= 2 and np.all(np.sign(sample_values) == np.sign(sample_values[0])))
        )
        rows.append(row)
    return rows


def method_specific_associations(
    particles: pd.DataFrame,
    obs: pd.DataFrame,
    expression,
    genes: np.ndarray,
    module_scores: dict[str, np.ndarray],
    retained_col: str,
    eligible: set[tuple[str, str]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gene_frames: list[pd.DataFrame] = []
    module_output: list[dict[str, object]] = []
    cohort_rows: list[dict[str, object]] = []
    groups = particles.groupby(["method", "C_y", "interval", "source_celltype"], dropna=False, sort=False)
    for group_index, ((method, c_y, interval, celltype), group) in enumerate(groups, start=1):
        if (str(interval), str(celltype)) not in eligible:
            continue
        retained = group[group[retained_col].astype(bool)].copy()
        if not cohort_valid(retained, obs, args.min_retained, args.min_per_sample):
            continue
        idx = retained["source_index"].to_numpy(int)
        raw, xr, xnorm, detected, mean_expression, cov_rank = prepare_expression(
            expression, idx, obs
        )
        growth = retained["native_log_growth_rate"].to_numpy(float)
        if np.std(growth) < 1e-10:
            continue
        rho, p = correlate_prepared(growth, xr, xnorm, obs, idx, cov_rank)
        valid_gene = detected >= args.min_detected_fraction
        local = pd.DataFrame(
            {
                "cohort_type": "method-specific retained",
                "method": method,
                "C_y": c_y,
                "interval": interval,
                "celltype": celltype,
                "gene": genes,
                "partial_spearman": rho,
                "p_value": p,
                "detected_fraction": detected,
                "mean_expression": mean_expression,
                "n_cells": len(retained),
                "growth_sd": float(np.std(growth)),
            }
        )
        local = local[valid_gene & np.isfinite(rho)].copy()
        local["fdr"] = bh_fdr(local["p_value"].to_numpy(float))
        gene_frames.append(local)
        module_output.extend(
            module_rows(retained, obs, module_scores, cohort_type="method-specific retained")
        )
        counts = obs.iloc[idx]["sample_name"].value_counts()
        counts = counts[counts > 0]
        cohort_rows.append(
            {
                "cohort_type": "method-specific retained",
                "method": method,
                "C_y": c_y,
                "interval": interval,
                "celltype": celltype,
                "n_source": int(len(group)),
                "n_retained": int(len(retained)),
                "preserved_fraction": float(len(retained) / len(group)),
                "min_sample_n": int(counts.min()),
                "growth_sd": float(np.std(growth)),
                "growth_iqr": float(np.subtract(*np.percentile(growth, [75, 25]))),
            }
        )
        if group_index % 25 == 0:
            print(f"  method-specific groups scanned: {group_index}", flush=True)
    return (
        pd.concat(gene_frames, ignore_index=True) if gene_frames else pd.DataFrame(),
        pd.DataFrame(module_output),
        pd.DataFrame(cohort_rows),
    )


def summarize_sync_genes(frame: pd.DataFrame, min_valid_c_y: int) -> pd.DataFrame:
    sync = frame[frame["method"] == "USOT Sync"].copy()
    if sync.empty:
        return pd.DataFrame()
    keys = ["interval", "celltype", "gene"]
    result = (
        sync.groupby(keys, observed=True)["partial_spearman"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "n_C_y",
                "mean": "rho_mean",
                "median": "rho_median",
                "std": "rho_sd",
                "min": "rho_min",
                "max": "rho_max",
            }
        )
    )
    sign = (
        sync.assign(pos=lambda x: x["partial_spearman"] > 0)
        .groupby(keys, observed=True)["pos"]
        .mean()
        .rename("positive_fraction")
        .reset_index()
    )
    result = result.merge(sign, on=keys, validate="one_to_one")
    result["sign_consistency"] = np.maximum(
        result["positive_fraction"], 1.0 - result["positive_fraction"]
    )
    return result[result["n_C_y"] >= min_valid_c_y].copy()


def pairwise_common_associations(
    particles: pd.DataFrame,
    obs: pd.DataFrame,
    expression,
    genes: np.ndarray,
    module_scores: dict[str, np.ndarray],
    retained_col: str,
    eligible: set[tuple[str, str]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sync = particles[particles["method"] == "USOT Sync"].copy()
    gene_summaries: list[pd.DataFrame] = []
    module_output: list[dict[str, object]] = []
    cohort_rows: list[dict[str, object]] = []
    for baseline_name in BASELINES:
        print(f"[pairwise common] USOT Sync vs {baseline_name}", flush=True)
        baseline = particles[particles["method"] == baseline_name].copy()
        baseline = baseline.set_index(["interval", "source_index"], verify_integrity=True)
        accum: dict[tuple[str, str], dict[str, list]] = {}
        for (c_y, interval, celltype), sync_group in sync.groupby(
            ["C_y", "interval", "source_celltype"], observed=True, sort=False
        ):
            key = (str(interval), str(celltype))
            if key not in eligible:
                continue
            lookup = pd.MultiIndex.from_arrays(
                [sync_group["interval"].astype(str), sync_group["source_index"].to_numpy(int)]
            )
            base_group = baseline.loc[lookup].reset_index()
            if not np.array_equal(
                base_group["source_index"].to_numpy(int), sync_group["source_index"].to_numpy(int)
            ):
                raise ValueError("Pairwise common source order mismatch")
            common_mask = (
                sync_group[retained_col].astype(bool).to_numpy()
                & base_group[retained_col].astype(bool).to_numpy()
            )
            sync_common = sync_group.iloc[np.flatnonzero(common_mask)].copy()
            base_common = base_group.iloc[np.flatnonzero(common_mask)].copy()
            if not cohort_valid(sync_common, obs, args.min_retained, args.min_per_sample):
                continue
            idx = sync_common["source_index"].to_numpy(int)
            if not np.array_equal(idx, base_common["source_index"].to_numpy(int)):
                raise ValueError("Common-retained IDs differ between methods")
            _, xr, xnorm, detected, _, cov_rank = prepare_expression(expression, idx, obs)
            g_sync = sync_common["native_log_growth_rate"].to_numpy(float)
            g_base = base_common["native_log_growth_rate"].to_numpy(float)
            if min(np.std(g_sync), np.std(g_base)) < 1e-10:
                continue
            rho_sync, _ = correlate_prepared(g_sync, xr, xnorm, obs, idx, cov_rank)
            rho_base, _ = correlate_prepared(g_base, xr, xnorm, obs, idx, cov_rank)
            valid = detected >= args.min_detected_fraction
            store = accum.setdefault(
                key,
                {"sync": [], "base": [], "delta": [], "n": [], "C_y": [], "valid": []},
            )
            store["sync"].append(rho_sync)
            store["base"].append(rho_base)
            store["delta"].append(rho_sync - rho_base)
            store["n"].append(len(idx))
            store["C_y"].append(float(c_y))
            store["valid"].append(valid)

            sync_common["method"] = "USOT Sync"
            sync_common["C_y"] = float(c_y)
            base_common["method"] = baseline_name
            base_common["C_y"] = float(c_y)
            module_output.extend(
                module_rows(
                    sync_common,
                    obs,
                    module_scores,
                    cohort_type="pairwise common retained",
                    comparator=baseline_name,
                )
            )
            module_output.extend(
                module_rows(
                    base_common,
                    obs,
                    module_scores,
                    cohort_type="pairwise common retained",
                    comparator="USOT Sync",
                )
            )
            counts = obs.iloc[idx]["sample_name"].value_counts()
            counts = counts[counts > 0]
            cohort_rows.append(
                {
                    "cohort_type": "pairwise common retained",
                    "method_a": "USOT Sync",
                    "method_b": baseline_name,
                    "C_y": float(c_y),
                    "interval": interval,
                    "celltype": celltype,
                    "n_common": len(idx),
                    "min_sample_n": int(counts.min()),
                }
            )

        for (interval, celltype), store in accum.items():
            sync_rho = np.vstack(store["sync"])
            base_rho = np.vstack(store["base"])
            delta = np.vstack(store["delta"])
            valid_count = np.sum(np.vstack(store["valid"]), axis=0)
            n_cy = sync_rho.shape[0]
            if n_cy < args.min_valid_c_y:
                continue
            with np.errstate(invalid="ignore"):
                local = pd.DataFrame(
                    {
                        "baseline": baseline_name,
                        "interval": interval,
                        "celltype": celltype,
                        "gene": genes,
                        "n_C_y": n_cy,
                        "n_C_y_gene_detected": valid_count,
                        "median_n_common": float(np.median(store["n"])),
                        "min_n_common": int(np.min(store["n"])),
                        "sync_rho_median": np.nanmedian(sync_rho, axis=0),
                        "baseline_rho_median": np.nanmedian(base_rho, axis=0),
                        "delta_rho_median": np.nanmedian(delta, axis=0),
                        "delta_rho_mean": np.nanmean(delta, axis=0),
                        "delta_rho_sd": np.nanstd(delta, axis=0, ddof=1),
                        "delta_positive_fraction": np.mean(delta > 0, axis=0),
                        "sync_positive_fraction": np.mean(sync_rho > 0, axis=0),
                        "baseline_positive_fraction": np.mean(base_rho > 0, axis=0),
                    }
                )
            local = local[local["n_C_y_gene_detected"] >= args.min_valid_c_y].copy()
            gene_summaries.append(local)
    return (
        pd.concat(gene_summaries, ignore_index=True) if gene_summaries else pd.DataFrame(),
        pd.DataFrame(module_output),
        pd.DataFrame(cohort_rows),
    )


def write_top_genes(sync_summary: pd.DataFrame, output: Path) -> None:
    if sync_summary.empty:
        return
    clean = sync_summary[
        ~sync_summary["gene"].str.match(r"^(?:Rpl|Rps|Mrpl|Mrps|mt-|Gm\d+$)", case=False, na=False)
    ].copy()
    stable = clean[clean["sign_consistency"] >= 7 / 9].copy()
    top = []
    for _, group in stable.groupby(["interval", "celltype"], observed=True):
        top.append(group.nlargest(15, "rho_median"))
        top.append(group.nsmallest(15, "rho_median"))
    if top:
        pd.concat(top, ignore_index=True).to_csv(output, index=False)


def main() -> None:
    args = parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"k{args.k}"
    targets = [
        output / f"method_specific_gene_associations_{suffix}.csv.gz",
        output / f"method_specific_module_associations_{suffix}.csv",
        output / f"sync_gene_consensus_{suffix}.csv.gz",
        output / f"pairwise_common_gene_summary_{suffix}.csv.gz",
        output / f"pairwise_common_module_associations_{suffix}.csv",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")

    print("Loading particle cache and expression", flush=True)
    particles = pd.read_csv(args.particle_table)
    adata = ad.read_h5ad(args.expression_h5ad)
    source_ids = adata.obs_names.astype(str).to_numpy()
    if not np.array_equal(
        source_ids[particles["source_index"].to_numpy(int)],
        particles["source_cell_id"].astype(str).to_numpy(),
    ):
        raise ValueError("Particle source IDs are not aligned to RNA expression")
    retained_col = f"retained_k{args.k}"
    if retained_col not in particles:
        raise KeyError(retained_col)

    stages = adata.obs["stage"].astype(str).to_numpy()
    candidate_idx = np.flatnonzero(adata.var["highly_variable"].to_numpy(bool)).tolist()
    gene_lookup = {str(gene): i for i, gene in enumerate(adata.var_names.astype(str))}
    for requested in MODULES.values():
        candidate_idx.extend(gene_lookup[g] for g in requested if g in gene_lookup)
    candidate_idx = np.asarray(sorted(set(candidate_idx)), dtype=int)
    genes = np.asarray(adata.var_names.astype(str))[candidate_idx]
    expression = adata.X[:, candidate_idx]
    module_scores: dict[str, np.ndarray] = {}
    module_members: dict[str, list[str]] = {}
    for module, requested in MODULES.items():
        score, present = module_score_all_cells(adata, stages, requested)
        module_scores[module] = score
        module_members[module] = present

    source_reference = particles[particles["method"] == "USOT RNA-only"]
    source_counts = source_reference.groupby(["interval", "source_celltype"], observed=True).size()
    eligible = {
        (str(interval), str(celltype))
        for (interval, celltype), count in source_counts.items()
        if count >= args.min_source
        and str(celltype).lower() not in {"nan", "unknown", "unannotated"}
    }
    print(f"Candidate genes: {len(genes)}; source-eligible cohorts: {len(eligible)}", flush=True)

    print("Computing method-specific retained associations", flush=True)
    genes_method, modules_method, cohorts_method = method_specific_associations(
        particles,
        adata.obs,
        expression,
        genes,
        module_scores,
        retained_col,
        eligible,
        args,
    )
    genes_method.to_csv(targets[0], index=False, compression="gzip")
    modules_method.to_csv(targets[1], index=False)
    sync_summary = summarize_sync_genes(genes_method, args.min_valid_c_y)
    sync_summary.to_csv(targets[2], index=False, compression="gzip")
    write_top_genes(sync_summary, output / f"sync_stable_top_growth_genes_{suffix}.csv")

    print("Computing pairwise common-retained associations", flush=True)
    genes_common, modules_common, cohorts_common = pairwise_common_associations(
        particles,
        adata.obs,
        expression,
        genes,
        module_scores,
        retained_col,
        eligible,
        args,
    )
    genes_common.to_csv(targets[3], index=False, compression="gzip")
    modules_common.to_csv(targets[4], index=False)
    pd.concat([cohorts_method, cohorts_common], ignore_index=True, sort=False).to_csv(
        output / f"analysis_cohorts_{suffix}.csv", index=False
    )

    manifest = {
        "analysis": "Identity-preserving interval-specific integrated-growth gene association",
        "particle_table": str(args.particle_table),
        "expression": str(args.expression_h5ad),
        "primary_k": args.k,
        "source_expression": "Observed source cell log-normalized RNA; no trajectory-point expression lookup",
        "growth": "Integrated native log-mass gain divided by physical interval duration; rank association within interval and cell type",
        "gene_statistic": "Partial Spearman controlling sample_name, log1p(nCount_RNA), and mitochondrial_percent_RNA",
        "method_specific": "Each method's own endpoint-retained cells",
        "pairwise_common": "For every C_y and baseline, both methods evaluated on the exact intersection of endpoint-retained source cell IDs",
        "sync_summary": "Gene coefficients calculated separately for every C_y and then summarized; C_y settings are not biological replicates",
        "filters": {
            "min_source": args.min_source,
            "min_retained": args.min_retained,
            "min_per_sample": args.min_per_sample,
            "min_detected_fraction": args.min_detected_fraction,
            "min_valid_C_y": args.min_valid_c_y,
        },
        "candidate_genes": int(len(genes)),
        "module_members": module_members,
        "limitation": "Endpoint retention removes explicit cross-label differentiation but does not prove a constant identity at every intermediate point; associations are model-inferred and non-causal.",
    }
    (output / f"analysis_manifest_{suffix}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {len(genes_method):,} method-specific gene rows and "
        f"{len(genes_common):,} pairwise summary rows",
        flush=True,
    )


if __name__ == "__main__":
    main()
