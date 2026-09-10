#!/usr/bin/env python3
"""Audit seven-time TrajectoryNet and official-GAGA10 MIOFlow inputs."""

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
import torch


PREFIX = "human_cerebral_7time_d4_d21_no_d16"
TIMES = np.asarray((0.0, 0.3, 0.5, 0.7, 0.8, 1.4, 1.7), dtype=np.float32)
COUNTS = (501, 1309, 1630, 1560, 4100, 5842, 5721)
RNA_SCALE = 63.89521587795941


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectorynet-npz",
        type=Path,
        default=Path(f"data/{PREFIX}_rna_trajectorynet_reversed.npz"),
    )
    parser.add_argument(
        "--mioflow-h5ad",
        type=Path,
        default=Path(f"data/{PREFIX}_rna_mioflow.h5ad"),
    )
    parser.add_argument(
        "--trajectorynet-direction",
        choices=("forward", "reversed"),
        default="reversed",
    )
    parser.add_argument(
        "--norm-params",
        type=Path,
        default=Path(f"data/{PREFIX}_rna_cytobridge_primal_norm_params.pt"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.trajectorynet_npz, args.mioflow_h5ad, args.norm_params):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    norm = torch.load(args.norm_params, map_location="cpu", weights_only=False)
    if not np.isclose(float(norm["scale"]), RNA_SCALE):
        raise ValueError(f"Unexpected seven-time RNA scale: {norm['scale']}")

    with np.load(args.trajectorynet_npz, allow_pickle=True) as archive:
        pca = np.asarray(archive["pca"], dtype=np.float32)
        raw = np.asarray(archive["pca_raw"], dtype=np.float32)
        training_ranks = np.asarray(archive["sample_labels"], dtype=np.int64)
        forward_ranks = np.asarray(archive["original_sample_labels"], dtype=np.int64)
        physical_times = np.asarray(archive["physical_time_values"], dtype=np.float32)
    if pca.shape != (sum(COUNTS), 30) or raw.shape != pca.shape:
        raise ValueError(f"Unexpected TrajectoryNet shapes: {pca.shape}, {raw.shape}")
    expected_training_ranks = (
        forward_ranks if args.trajectorynet_direction == "forward" else 6 - forward_ranks
    )
    if not np.array_equal(training_ranks, expected_training_ranks):
        raise ValueError(
            f"TrajectoryNet labels do not match {args.trajectorynet_direction} direction"
        )
    if tuple(int(np.sum(forward_ranks == rank)) for rank in range(7)) != COUNTS:
        raise ValueError("TrajectoryNet chronological-rank counts changed")
    if not np.allclose(pca, raw / RNA_SCALE, atol=2e-6, rtol=2e-6):
        raise ValueError("TrajectoryNet pca is not the normalized seven-time PCA30")
    if not np.allclose(physical_times, TIMES, atol=1e-7, rtol=0.0):
        raise ValueError("TrajectoryNet physical-time metadata changed")

    mioflow = ad.read_h5ad(args.mioflow_h5ad)
    mio_raw = np.asarray(mioflow.obsm["X_latent"], dtype=np.float32)
    mio_pca = np.asarray(mioflow.obsm["X_pca"], dtype=np.float32)
    mio_normalized = np.asarray(mioflow.obsm["X_pca_normalized"], dtype=np.float32)
    if mioflow.shape != (sum(COUNTS), 30):
        raise ValueError(f"Unexpected MIOFlow input shape: {mioflow.shape}")
    if not np.array_equal(mio_raw, mio_pca):
        raise ValueError("MIOFlow X_latent and X_pca are not the same raw PCA30")
    if not np.allclose(mio_normalized, mio_raw / RNA_SCALE, atol=2e-6, rtol=2e-6):
        raise ValueError("MIOFlow normalized audit coordinates changed")
    if not np.array_equal(mio_normalized, pca):
        raise ValueError("TrajectoryNet and MIOFlow do not share identical normalized PCA30")
    mio_times = np.asarray(mioflow.uns["physical_time_values"], dtype=np.float32)
    if not np.allclose(mio_times, TIMES, atol=1e-7, rtol=0.0):
        raise ValueError("MIOFlow physical-time metadata changed")
    if any(np.isclose(TIMES, 1.2)):
        raise ValueError("D16/t=1.2 must be absent")

    print("Seven-time TrajectoryNet/MIOFlow setup passed")
    print(f"  stages: D4,D7,D9,D11,D12,D18,D21; counts={list(COUNTS)}")
    print(f"  physical times: {TIMES.tolist()}; D16 is absent")
    print(f"  common normalized PCA30 scale: {RNA_SCALE}")
    labels = "[0,1,2,3,4,5,6]" if args.trajectorynet_direction == "forward" else "[6,5,4,3,2,1,0]"
    print(
        f"  TrajectoryNet {args.trajectorynet_direction} labels by chronological stage: {labels}"
    )
    print("  MIOFlow labels: official consecutive ranks [0,1,2,3,4,5,6]")
    print(f"  TrajectoryNet input: {args.trajectorynet_npz.resolve()}")
    print(f"  MIOFlow input: {args.mioflow_h5ad.resolve()}")


if __name__ == "__main__":
    main()
