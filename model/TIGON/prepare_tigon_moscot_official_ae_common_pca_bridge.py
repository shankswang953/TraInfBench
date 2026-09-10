#!/usr/bin/env python
"""Prepare the frozen expression-to-PCA50 bridge for moscot TIGON AE10."""

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
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from prepare_tigon_official_ae_embedding import load_full_expression


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT / "data/moscot_rna_full.h5ad"
DEFAULT_SHARED = ROOT / "data/moscot_rna_cytobridge.h5ad"
DEFAULT_CACHE = ROOT / "results/tigon_official_ae_embeddings/moscot_ae10"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-h5ad", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--shared-h5ad", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--components", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    bridge_path = cache_dir / "expression_to_shared_pca50.joblib"
    metrics_path = cache_dir / "expression_to_shared_pca50_metrics.json"
    ready_path = cache_dir / "common_pca_bridge_ready.json"
    outputs = (bridge_path, metrics_path, ready_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        if len(existing) == len(outputs):
            print(f"[reuse] moscot common-PCA bridge is ready: {bridge_path}")
            return
        raise FileExistsError(
            "Incomplete bridge outputs exist: "
            + ", ".join(str(path) for path in existing)
        )
    if args.components <= 0:
        raise ValueError("--components must be positive")

    expression, obs_names, _, genes = load_full_expression(
        "moscot", args.raw_h5ad.resolve(), args.shared_h5ad.resolve()
    )
    shared = ad.read_h5ad(args.shared_h5ad, backed="r")
    try:
        if not np.array_equal(
            shared.obs_names.astype(str).to_numpy(), obs_names
        ):
            raise ValueError("Bridge expression and shared PCA cells differ")
        shared_pca = np.asarray(shared.obsm["X_latent"], dtype=np.float32)
    finally:
        shared.file.close()
    if shared_pca.shape[1] != 50:
        raise ValueError(f"Expected shared PCA50, found {shared_pca.shape}")

    selected_genes = np.loadtxt(
        cache_dir / "selected_genes.csv",
        dtype=str,
        delimiter=",",
        skiprows=1,
    ).reshape(-1)
    if not np.array_equal(selected_genes, genes):
        raise ValueError("Frozen AE and bridge expression genes differ")

    scaler = StandardScaler()
    standardized = scaler.fit_transform(expression)
    components = min(
        int(args.components),
        standardized.shape[0] - 1,
        standardized.shape[1],
    )
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        random_state=0,
    )
    intermediate = pca.fit_transform(standardized)
    design = np.column_stack(
        [intermediate, np.ones(intermediate.shape[0], dtype=intermediate.dtype)]
    )
    affine = np.linalg.lstsq(design, shared_pca, rcond=None)[0].astype(
        np.float32
    )
    reconstructed = design @ affine
    residual = shared_pca - reconstructed
    centered = shared_pca - shared_pca.mean(axis=0, keepdims=True)
    r2 = 1.0 - float(np.square(residual).sum()) / float(
        np.square(centered).sum()
    )
    rmse = float(np.sqrt(np.square(residual).mean()))
    relative_rmse = rmse / float(np.sqrt(np.square(centered).mean()))
    joblib.dump(
        {
            "standard_scaler": scaler,
            "pca": pca,
            "affine": affine,
            "input_genes": genes,
            "input_space": "moscot 3000-HVG scran log-expression",
            "output_space": "exact shared raw PCA50",
            "fit_policy": "full-data unsupervised coordinate bridge",
        },
        bridge_path,
    )
    metrics = {
        "components": components,
        "cells": int(expression.shape[0]),
        "input_genes": int(expression.shape[1]),
        "output_dimension": int(shared_pca.shape[1]),
        "training_r2": r2,
        "training_rmse": rmse,
        "training_relative_rmse": relative_rmse,
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    ready = {
        "status": "ready",
        "bridge": str(bridge_path),
        "bridge_sha256": file_sha256(bridge_path),
        "metrics": str(metrics_path),
        **metrics,
    }
    ready_path.write_text(
        json.dumps(ready, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[done] bridge R2={r2:.6f}, relative_RMSE={relative_rmse:.6f}: "
        f"{bridge_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
