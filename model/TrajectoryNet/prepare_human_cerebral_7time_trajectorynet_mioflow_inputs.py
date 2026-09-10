#!/usr/bin/env python3
"""Prepare seven-time TrajectoryNet and official-GAGA MIOFlow inputs.

TrajectoryNet is trained on the freshly normalized PCA30 coordinates with
reversed consecutive ranks, so its invertible density direction can be rolled
out from the observed D4 cells.  MIOFlow stores raw PCA30 as its shared source
space; the runner divides by the same seven-time scale before fitting the
official StandardScaler -> PHATE -> two-phase GAGA10 pipeline.
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
from pathlib import Path

import anndata as ad
import numpy as np


PREFIX = "human_cerebral_7time_d4_d21_no_d16"
DEFAULT_SOURCE = Path(f"data/{PREFIX}_rna_cytobridge_balanced.h5ad")
DEFAULT_TRAJECTORYNET = Path(f"data/{PREFIX}_rna_trajectorynet_reversed.npz")
DEFAULT_MIOFLOW = Path(f"data/{PREFIX}_rna_mioflow.h5ad")
TIME_KEYS = ("age_4", "age_7", "age_9", "age_11", "age_12", "age_18", "age_21")
AGES = (4, 7, 9, 11, 12, 18, 21)
PHYSICAL_TIMES = (0.0, 0.3, 0.5, 0.7, 0.8, 1.4, 1.7)
COUNTS = (501, 1309, 1630, 1560, 4100, 5842, 5721)
RNA_SCALE = 63.89521587795941


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5ad", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--trajectorynet-output", type=Path, default=DEFAULT_TRAJECTORYNET)
    parser.add_argument("--mioflow-output", type=Path, default=DEFAULT_MIOFLOW)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def protect(paths: tuple[Path, ...], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to replace existing prepared inputs; pass --overwrite only "
            "after inspection:\n" + "\n".join(str(path) for path in existing)
        )


def validate_source(adata: ad.AnnData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required_obs = {"time_index", "time_point_processed", "age_day", "time_key"}
    missing_obs = sorted(required_obs.difference(adata.obs.columns))
    if missing_obs:
        raise ValueError(f"Source is missing obs columns: {missing_obs}")
    for key in ("X_latent", "X_pca", "X_pca_raw"):
        if key not in adata.obsm:
            raise KeyError(f"Source is missing adata.obsm[{key!r}]")

    normalized = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    normalized_alias = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    raw = np.asarray(adata.obsm["X_pca_raw"], dtype=np.float32)
    ranks = adata.obs["time_index"].to_numpy(dtype=np.int64)
    if normalized.shape != (sum(COUNTS), 30) or raw.shape != normalized.shape:
        raise ValueError(f"Unexpected source matrix shapes: {normalized.shape}, {raw.shape}")
    if not np.array_equal(normalized, normalized_alias):
        raise ValueError("X_latent and X_pca are not identical normalized PCA30 arrays")
    if not np.allclose(normalized, raw / RNA_SCALE, atol=2e-6, rtol=2e-6):
        raise ValueError("Source coordinates do not use the expected seven-time RNA scale")
    if not np.isfinite(normalized).all() or not np.isfinite(raw).all():
        raise ValueError("Source PCA arrays contain non-finite values")

    observed_counts = tuple(int(np.sum(ranks == rank)) for rank in range(7))
    if observed_counts != COUNTS:
        raise ValueError(f"Unexpected chronological-rank counts: {observed_counts}")
    observed_times = tuple(
        float(adata.obs.loc[ranks == rank, "time_point_processed"].iloc[0])
        for rank in range(7)
    )
    if not np.allclose(observed_times, PHYSICAL_TIMES, atol=1e-7, rtol=0.0):
        raise ValueError(f"Unexpected physical times: {observed_times}")
    observed_keys = tuple(
        str(adata.obs.loc[ranks == rank, "time_key"].iloc[0]) for rank in range(7)
    )
    if observed_keys != TIME_KEYS:
        raise ValueError(f"Unexpected time keys: {observed_keys}")
    observed_ages = tuple(
        int(adata.obs.loc[ranks == rank, "age_day"].iloc[0]) for rank in range(7)
    )
    if observed_ages != AGES:
        raise ValueError(f"Unexpected ages: {observed_ages}")
    return normalized, raw, ranks


def write_trajectorynet(
    output: Path,
    source: Path,
    adata: ad.AnnData,
    normalized: np.ndarray,
    raw: np.ndarray,
    forward_ranks: np.ndarray,
) -> None:
    reversed_ranks = (6 - forward_ranks).astype(np.int64)
    payload = {
        "pca": normalized,
        "pca_normalized": normalized,
        "pca_raw": raw,
        "sample_labels": reversed_ranks,
        "original_sample_labels": forward_ranks,
        "reversed_time_index": reversed_ranks,
        "time_index": forward_ranks,
        "time_point_processed": adata.obs["time_point_processed"].to_numpy(dtype=np.float32),
        "age_day": adata.obs["age_day"].to_numpy(dtype=np.int64),
        "time_key": adata.obs["time_key"].astype(str).to_numpy(),
        "cell_ids": adata.obs_names.astype(str).to_numpy(),
        "time_keys": np.asarray(TIME_KEYS),
        "physical_time_values": np.asarray(PHYSICAL_TIMES, dtype=np.float32),
        "raw_to_model_scale": np.asarray(RNA_SCALE, dtype=np.float64),
        "time_label_mapping": np.asarray(
            [f"forward_rank_{rank}->reversed_rank_{6 - rank}" for rank in range(7)]
        ),
        "direction_protocol": np.asarray(
            "seven-time reversed labels; Gaussian generation D21 to D4; "
            "native density direction D4 to D21 from observed D4"
        ),
        "source_h5ad": np.asarray(str(source.resolve())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(output)


def write_mioflow(
    output: Path,
    source: Path,
    adata: ad.AnnData,
    normalized: np.ndarray,
    raw: np.ndarray,
) -> None:
    result = ad.AnnData(X=raw.copy(), obs=adata.obs.copy(), var=adata.var.copy())
    result.obsm["X_latent"] = raw.copy()
    result.obsm["X_pca"] = raw.copy()
    result.obsm["X_pca_normalized"] = normalized.copy()
    result.uns["dataset"] = "human_cerebral"
    result.uns["task"] = "full_7time_mioflow_official_gaga10_no_d16"
    result.uns["source_h5ad"] = str(source.resolve())
    result.uns["time_keys"] = np.asarray(TIME_KEYS)
    result.uns["physical_time_values"] = np.asarray(PHYSICAL_TIMES, dtype=np.float32)
    result.uns["excluded_time_keys"] = np.asarray(("age_16", "age_26", "age_31", "age_61"))
    result.uns["source_space"] = "raw RNA PCA30"
    result.uns["model_input_protocol"] = (
        "raw PCA30 divided by the seven-time scale, then StandardScaler, PHATE, "
        "two-phase GAGA10, and feature-wise GAGA10 z-score"
    )
    result.uns["rna_raw_to_normalized_scale"] = RNA_SCALE
    result.uns["mioflow_time_axis"] = "official consecutive ranks 0..6"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.h5ad")
    result.write_h5ad(temporary, compression="gzip")
    temporary.replace(output)


def main() -> None:
    args = parse_args()
    for path in (args.source_h5ad,):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    protect((args.trajectorynet_output, args.mioflow_output), args.overwrite)
    source = ad.read_h5ad(args.source_h5ad)
    normalized, raw, ranks = validate_source(source)
    write_trajectorynet(
        args.trajectorynet_output, args.source_h5ad, source, normalized, raw, ranks
    )
    write_mioflow(args.mioflow_output, args.source_h5ad, source, normalized, raw)
    print(f"[saved] TrajectoryNet reversed input: {args.trajectorynet_output}")
    print("  chronological D4..D21 ranks -> training labels: 6,5,4,3,2,1,0")
    print(f"[saved] MIOFlow official-GAGA input: {args.mioflow_output}")
    print(f"  raw PCA30 -> divide by {RNA_SCALE:.15g} -> StandardScaler/PHATE/GAGA10")


if __name__ == "__main__":
    main()
