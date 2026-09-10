#!/usr/bin/env python
from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[1]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
from pathlib import Path

import anndata as ad
import numpy as np


def _labels_for_trajectorynet(times: np.ndarray, encoding: str) -> np.ndarray:
    if encoding == "rank":
        unique_times = np.asarray(sorted(np.unique(times)), dtype=float)
        return np.searchsorted(unique_times, times).astype(np.int64)
    if encoding != "physical":
        raise ValueError(f"Unknown TrajectoryNet time encoding: {encoding}")
    if np.allclose(times, np.round(times)):
        return np.round(times).astype(np.int64)
    return times.astype("float32", copy=False)


def _interpolate_rank_time(heldout_time: float, train_times: np.ndarray) -> float:
    insert_at = int(np.searchsorted(train_times, heldout_time))
    if insert_at == 0 or insert_at == train_times.size:
        raise ValueError(
            f"Held-out time {heldout_time:g} is outside train times {train_times.tolist()}"
        )
    left_index = insert_at - 1
    right_index = insert_at
    fraction = (heldout_time - train_times[left_index]) / (
        train_times[right_index] - train_times[left_index]
    )
    return float(left_index + fraction * (right_index - left_index))


def _check_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create leave-one-timepoint-out inputs for TrajectoryNet, "
            "CytoBridge, and MIOFlow. The train files keep the original "
            "time labels and remove only the held-out timepoint."
        )
    )
    parser.add_argument("--input-h5ad", type=Path, default=Path("data/moscot_rna_cytobridge.h5ad"))
    parser.add_argument("--trajectorynet-output", type=Path, default=Path("data/moscot_rna_loo_time1_trajectorynet.npz"))
    parser.add_argument("--h5ad-output", type=Path, default=Path("data/moscot_rna_loo_time1_cytobridge.h5ad"))
    parser.add_argument("--reference-output", type=Path, default=Path("data/moscot_rna_loo_time1_reference.npz"))
    parser.add_argument("--time-key", default="time_point_processed")
    parser.add_argument("--latent-key", default="X_latent")
    parser.add_argument("--embedding-name", default="pca")
    parser.add_argument("--heldout-time", type=float, default=1.0)
    parser.add_argument(
        "--trajectorynet-time-encoding",
        choices=("physical", "rank"),
        default="physical",
        help=(
            "physical preserves the source labels; rank encodes sorted training "
            "timepoints as consecutive integers 0..K-1."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for output in (args.trajectorynet_output, args.h5ad_output, args.reference_output):
        _check_output(output, args.overwrite)

    adata = ad.read_h5ad(args.input_h5ad)
    if args.time_key not in adata.obs:
        raise KeyError(f"{args.time_key!r} not found in adata.obs. Available: {list(adata.obs.columns)}")
    if args.latent_key not in adata.obsm:
        raise KeyError(f"{args.latent_key!r} not found in adata.obsm. Available: {list(adata.obsm.keys())}")

    times = np.asarray(adata.obs[args.time_key], dtype=float)
    x = np.asarray(adata.obsm[args.latent_key], dtype="float32")
    holdout_mask = np.isclose(times, args.heldout_time)
    train_mask = ~holdout_mask

    if not np.any(holdout_mask):
        raise ValueError(f"No cells found for held-out time {args.heldout_time:g}")
    train_times = np.array(sorted(np.unique(times[train_mask])), dtype=float)
    if train_times.size < 2:
        raise ValueError(f"Need at least two training timepoints after holdout, got {train_times.tolist()}")
    if args.heldout_time <= train_times.min() or args.heldout_time >= train_times.max():
        raise ValueError(
            "The held-out timepoint must sit between the two train endpoints for interpolation: "
            f"heldout={args.heldout_time:g}, train_times={train_times.tolist()}"
        )

    train_x = x[train_mask]
    train_labels = _labels_for_trajectorynet(
        times[train_mask], args.trajectorynet_time_encoding
    )
    train_model_times = np.asarray(sorted(np.unique(train_labels)), dtype=float)
    heldout_model_time = (
        _interpolate_rank_time(args.heldout_time, train_times)
        if args.trajectorynet_time_encoding == "rank"
        else float(args.heldout_time)
    )
    args.trajectorynet_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.trajectorynet_output,
        **{args.embedding_name: train_x, "sample_labels": train_labels},
    )

    train_adata = adata[train_mask].copy()
    train_adata.obs["sample_labels"] = train_labels
    train_adata.obs[args.time_key] = times[train_mask].astype("float32")
    train_adata.obsm[args.latent_key] = train_x
    train_adata.uns["loo_heldout_time"] = float(args.heldout_time)
    train_adata.uns["loo_train_times"] = train_times.astype("float32")
    train_adata.uns["loo_source_h5ad"] = str(args.input_h5ad)
    train_adata.uns["loo_time_key"] = args.time_key
    train_adata.uns["trajectorynet_time_encoding"] = args.trajectorynet_time_encoding
    train_adata.uns["trajectorynet_physical_train_times"] = train_times.astype("float32")
    train_adata.uns["trajectorynet_model_train_times"] = train_model_times.astype("float32")
    train_adata.uns["trajectorynet_heldout_model_time"] = heldout_model_time
    train_adata.write_h5ad(args.h5ad_output)

    initial_time = float(train_times.min())
    terminal_time = float(train_times.max())
    initial_mask = np.isclose(times, initial_time)
    terminal_mask = np.isclose(times, terminal_time)
    np.savez_compressed(
        args.reference_output,
        initial=x[initial_mask],
        heldout=x[holdout_mask],
        terminal=x[terminal_mask],
        initial_indices=np.flatnonzero(initial_mask).astype(np.int64),
        heldout_indices=np.flatnonzero(holdout_mask).astype(np.int64),
        terminal_indices=np.flatnonzero(terminal_mask).astype(np.int64),
        initial_time=np.asarray(initial_time, dtype="float32"),
        heldout_time=np.asarray(args.heldout_time, dtype="float32"),
        terminal_time=np.asarray(terminal_time, dtype="float32"),
        train_times=train_times.astype("float32"),
        trajectorynet_time_encoding=np.asarray(args.trajectorynet_time_encoding),
        trajectorynet_physical_train_times=train_times.astype("float32"),
        trajectorynet_model_train_times=train_model_times.astype("float32"),
        trajectorynet_heldout_model_time=np.asarray(heldout_model_time, dtype="float32"),
        input_h5ad=np.asarray(str(args.input_h5ad)),
        time_key=np.asarray(args.time_key),
        latent_key=np.asarray(args.latent_key),
    )

    print(f"input: {args.input_h5ad}")
    print(f"heldout_time: {args.heldout_time:g}")
    print(f"train_times: {train_times.tolist()}")
    print(f"trajectorynet_time_encoding: {args.trajectorynet_time_encoding}")
    print(f"trajectorynet_model_train_times: {train_model_times.tolist()}")
    print(f"trajectorynet_heldout_model_time: {heldout_model_time:.12g}")
    print(f"train_cells: {int(train_mask.sum())}, heldout_cells: {int(holdout_mask.sum())}")
    print(f"wrote: {args.trajectorynet_output}")
    print(f"wrote: {args.h5ad_output}")
    print(f"wrote: {args.reference_output}")


if __name__ == "__main__":
    main()
