#!/usr/bin/env python3
"""Audit the seven-time balanced/unbalanced CytoBridge inputs."""

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
from pathlib import Path

import anndata as ad
import numpy as np
import torch


PREFIX = "human_cerebral_7time_d4_d21_no_d16_rna_cytobridge"
TIME_KEYS = ("age_4", "age_7", "age_9", "age_11", "age_12", "age_18", "age_21")
TIMES = (0.0, 0.3, 0.5, 0.7, 0.8, 1.4, 1.7)
SOURCE_COUNTS = (501, 1309, 1630, 1560, 4100, 5842, 5721)
MASS_PRIOR = (1.00, 2.92, 4.15, 5.36, 5.96, 22.94, 33.99)
PRIOR_COUNTS = (1000, 2920, 4150, 5360, 5960, 22940, 33990)
RNA_SCALE = 63.89521587795941


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def counts_by_time(adata: ad.AnnData) -> tuple[int, ...]:
    times = adata.obs["time_point_processed"].to_numpy(dtype=float)
    return tuple(int(np.isclose(times, value, atol=1e-7).sum()) for value in TIMES)


def check_common(adata: ad.AnnData, expected_counts: tuple[int, ...], task: str) -> None:
    if adata.uns["task"] != task:
        raise ValueError(f"Unexpected task: {adata.uns['task']}")
    if tuple(str(value) for value in adata.uns["time_keys"]) != TIME_KEYS:
        raise ValueError("time_keys changed")
    if not np.allclose(adata.uns["physical_time_values"], TIMES, atol=1e-7, rtol=0.0):
        raise ValueError("physical times changed")
    if any(np.isclose(np.asarray(TIMES), 1.2)):
        raise ValueError("D16/t=1.2 must be absent")
    if counts_by_time(adata) != expected_counts:
        raise ValueError(
            f"row counts {counts_by_time(adata)} do not match expected {expected_counts}"
        )
    if adata.obsm["X_latent"].shape != (sum(expected_counts), 30):
        raise ValueError("X_latent has the wrong shape")
    if adata.obsm["X_pca_raw"].shape != (sum(expected_counts), 30):
        raise ValueError("X_pca_raw has the wrong shape")
    if not np.isfinite(adata.obsm["X_latent"]).all():
        raise ValueError("X_latent contains non-finite values")
    if not np.allclose(
        adata.obsm["X_latent"],
        adata.obsm["X_pca_raw"] / RNA_SCALE,
        atol=2e-6,
        rtol=2e-6,
    ):
        raise ValueError("X_latent is not the expected seven-time normalized PCA30")
    observed_times = tuple(sorted(np.unique(adata.obs["time_point_processed"].astype(float))))
    if not np.allclose(observed_times, TIMES, atol=1e-7, rtol=0.0):
        raise ValueError(f"Unexpected observed time grid: {observed_times}")
    if not adata.obs_names.is_unique:
        raise ValueError("AnnData observation names must be unique")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    balanced_path = data_dir / f"{PREFIX}_balanced.h5ad"
    unbalanced_path = data_dir / f"{PREFIX}_unbalanced_biological_prior.h5ad"
    metadata_path = data_dir / f"{PREFIX}_metadata.json"
    norm_path = data_dir / f"{PREFIX}_primal_norm_params.pt"
    for path in (balanced_path, unbalanced_path, metadata_path, norm_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    balanced = ad.read_h5ad(balanced_path)
    unbalanced = ad.read_h5ad(unbalanced_path)
    check_common(balanced, SOURCE_COUNTS, "full_7time_balanced_no_d16")
    check_common(
        unbalanced,
        PRIOR_COUNTS,
        "full_7time_unbalanced_biological_prior_no_d16",
    )

    realized = np.asarray(PRIOR_COUNTS, dtype=float) / PRIOR_COUNTS[0]
    if not np.allclose(realized, MASS_PRIOR, atol=1e-12, rtol=0.0):
        raise ValueError(f"Prior-encoded row ratios are wrong: {realized}")
    for key, source_count in zip(TIME_KEYS, SOURCE_COUNTS):
        block = unbalanced.obs.loc[unbalanced.obs["time_key"].astype(str).eq(key)]
        if block["source_cell_id"].astype(str).nunique() != source_count:
            raise ValueError(f"Not every source metacell is retained at {key}")
        multiplicity = block["source_cell_id"].astype(str).value_counts()
        if int(multiplicity.max() - multiplicity.min()) > 1:
            raise ValueError(f"Replication is not uniform at {key}")

    norm = torch.load(norm_path, map_location="cpu", weights_only=False)
    if not np.isclose(float(norm["scale"]), RNA_SCALE):
        raise ValueError("Copied RNA normalization scale changed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["excluded_time_keys"] != ["age_16", "age_26", "age_31", "age_61"]:
        raise ValueError("Metadata excluded-time list changed")
    if tuple(metadata["unbalanced"]["prior_encoded_row_counts"].values()) != PRIOR_COUNTS:
        raise ValueError("Metadata prior row counts changed")

    print("Seven-time CytoBridge setup passed")
    print(f"  times:              {list(TIMES)} (D16/t=1.2 absent)")
    print(f"  balanced rows:      {list(SOURCE_COUNTS)}; total={balanced.n_obs}")
    print(f"  unbalanced rows:    {list(PRIOR_COUNTS)}; total={unbalanced.n_obs}")
    print(f"  realized mass:      {realized.tolist()}")
    print(f"  RNA model space:    normalized PCA30; scale={RNA_SCALE}")
    print(f"  balanced input:     {balanced_path}")
    print(f"  unbalanced input:   {unbalanced_path}")


if __name__ == "__main__":
    main()
