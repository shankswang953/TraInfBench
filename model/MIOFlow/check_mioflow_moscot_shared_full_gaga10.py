#!/usr/bin/env python
"""Audit the shared-Full-GAGA pancreas Full/LOO training contract."""

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
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FULL = ROOT / "data/moscot_rna_cytobridge.h5ad"
DEFAULT_LOO = ROOT / "data/moscot_rna_loo_time1_cytobridge.h5ad"
DEFAULT_RESULT = (
    ROOT / "results/mioflow_moscot_full_shared_gaga10_n1024_20000"
)


def _times(adata: ad.AnnData, key: str) -> list[float]:
    return sorted(
        pd.to_numeric(pd.Series(adata.obs[key].to_numpy()), errors="raise")
        .drop_duplicates()
        .astype(float)
        .tolist()
    )


def _shared_pca(adata: ad.AnnData, name: str) -> np.ndarray:
    for key in ("X_latent", "X_pca"):
        if key not in adata.obsm:
            raise KeyError(f"{name}: missing obsm[{key!r}]")
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    pca = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    if latent.shape != pca.shape:
        raise ValueError(f"{name}: X_latent and X_pca shapes differ")
    error = float(np.max(np.abs(latent - pca)))
    if error > 1e-6:
        raise ValueError(f"{name}: X_latent != X_pca; max abs error={error}")
    return latent


def _check_checkpoints(
    full: ad.AnnData,
    gaga_checkpoint: Path,
    model_checkpoint: Path,
) -> None:
    if not gaga_checkpoint.is_file():
        raise FileNotFoundError(gaga_checkpoint)
    if not model_checkpoint.is_file():
        raise FileNotFoundError(model_checkpoint)

    gaga = torch.load(gaga_checkpoint, map_location="cpu", weights_only=False)
    required_gaga = {
        "state_dict",
        "input_dim",
        "latent_dim",
        "hidden_dims",
        "input_scaler_mean",
        "input_scaler_scale",
        "input_scaler_n_samples_seen",
        "fit_indices",
        "source_checkpoint",
    }
    missing = required_gaga.difference(gaga)
    if missing:
        raise ValueError(f"Full GAGA checkpoint missing keys: {sorted(missing)}")
    if int(gaga["input_dim"]) != 50 or int(gaga["latent_dim"]) != 10:
        raise ValueError("Expected Full GAGA architecture PCA50 -> GAGA10")
    if gaga["source_checkpoint"] is not None:
        raise ValueError("Full GAGA must be fitted in the Full job, not loaded")
    if int(gaga["input_scaler_n_samples_seen"]) != full.n_obs:
        raise ValueError(
            "Full GAGA input scaler was not fitted on every Full-data cell"
        )
    fit_indices = np.asarray(gaga["fit_indices"], dtype=np.int64)
    if fit_indices.size == 0 or np.any(fit_indices < 0) or np.any(
        fit_indices >= full.n_obs
    ):
        raise ValueError("Full GAGA fit indices are invalid")

    model = torch.load(model_checkpoint, map_location="cpu", weights_only=False)
    if not bool(model.get("use_gaga")):
        raise ValueError("Full MIOFlow checkpoint does not use GAGA")
    if str(model.get("embedding_normalization", "")).replace("-", "_") != "zscore":
        raise ValueError("Full MIOFlow checkpoint must use GAGA-latent zscore")
    if int(model.get("input_dim", -1)) != 10:
        raise ValueError("Full MIOFlow checkpoint must operate in GAGA10")
    mean = np.asarray(model["mean_vals"], dtype=np.float64).reshape(-1)
    std = np.asarray(model["std_vals"], dtype=np.float64).reshape(-1)
    if mean.shape != (10,) or std.shape != (10,) or np.any(std <= 0):
        raise ValueError("Full GAGA-latent zscore parameters are invalid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-h5ad", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--loo-h5ad", type=Path, default=DEFAULT_LOO)
    parser.add_argument(
        "--full-gaga-checkpoint",
        type=Path,
        default=DEFAULT_RESULT / "gaga_model.pt",
    )
    parser.add_argument(
        "--full-model-checkpoint",
        type=Path,
        default=DEFAULT_RESULT / "model.pt",
    )
    parser.add_argument("--require-checkpoints", action="store_true")
    args = parser.parse_args()

    full = ad.read_h5ad(args.full_h5ad)
    loo = ad.read_h5ad(args.loo_h5ad)
    full_x = _shared_pca(full, "Full")
    loo_x = _shared_pca(loo, "LOO")
    if _times(full, "time_point_processed") != [0.0, 1.0, 2.0]:
        raise ValueError("Full input must contain times 0, 1, and 2")
    if _times(loo, "time_point_processed") != [0.0, 2.0]:
        raise ValueError("LOO input must contain only times 0 and 2")
    if not full.obs_names.is_unique or not loo.obs_names.is_unique:
        raise ValueError("Full and LOO cell identifiers must be unique")
    full_positions = full.obs_names.get_indexer(loo.obs_names)
    if np.any(full_positions < 0):
        raise ValueError("LOO contains cells that are absent from Full")
    subset_error = float(np.max(np.abs(full_x[full_positions] - loo_x)))
    if subset_error > 1e-6:
        raise ValueError(
            "LOO PCA coordinates differ from the corresponding Full cells; "
            f"max abs error={subset_error}"
        )

    if args.require_checkpoints:
        _check_checkpoints(
            full,
            args.full_gaga_checkpoint,
            args.full_model_checkpoint,
        )

    print("[OK] Full cells:", full.n_obs, "times:", _times(full, "time_point_processed"))
    print("[OK] LOO cells:", loo.n_obs, "times:", _times(loo, "time_point_processed"))
    print("[OK] LOO is an exact Full-data PCA subset; max abs error:", subset_error)
    if args.require_checkpoints:
        print("[OK] Frozen Full GAGA checkpoint:", args.full_gaga_checkpoint)
        print("[OK] Frozen Full latent zscore:", args.full_model_checkpoint)


if __name__ == "__main__":
    main()
