#!/usr/bin/env python
"""Read-only audit of the frozen moscot TIGON official-layout AE5 cache."""
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

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = (
    ROOT
    / "results/tigon_official_ae_embeddings/"
    "moscot_ae5_upstream_public_exact_v2"
)
EXPECTED_COUNTS = {0.0: 9029, 1.0: 10333, 2.0: 3242}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    cache = args.cache_dir.resolve()

    required = (
        "embedding_ready.json",
        "config.json",
        "ae.pt",
        "ae_latent_raw.npy",
        "ae_latent_scaled_minus2_2.npy",
        "ae_latent_scaling.npz",
        "obs_names.npy",
        "times_full.npy",
        "selected_genes.csv",
    )
    missing = [name for name in required if not (cache / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete AE5 cache; missing: {missing}")

    ready = json.loads((cache / "embedding_ready.json").read_text())
    config = json.loads((cache / "config.json").read_text())
    checkpoint = torch.load(cache / "ae.pt", map_location="cpu", weights_only=False)
    latent_raw = np.load(cache / "ae_latent_raw.npy", mmap_mode="r")
    latent_scaled = np.load(
        cache / "ae_latent_scaled_minus2_2.npy", mmap_mode="r"
    )
    obs_names = np.load(cache / "obs_names.npy", allow_pickle=False)
    times = np.load(cache / "times_full.npy", allow_pickle=False)
    scaling = np.load(cache / "ae_latent_scaling.npz")

    if ready.get("status") != "ready" or ready.get("dataset") != "moscot":
        raise ValueError("embedding_ready.json does not describe a ready moscot cache")
    if config.get("task") != "full" or bool(config.get("strict_loo")):
        raise ValueError("AE5 cache must be fitted once on the complete moscot data")
    if int(config.get("latent_dimension", -1)) != 5:
        raise ValueError(f"Expected config latent_dimension=5: {config}")
    if config.get("ae_architecture") != "3000-300-5-300-3000":
        raise ValueError(f"Unexpected AE architecture: {config.get('ae_architecture')}")
    if int(checkpoint.get("n_latent", -1)) != 5:
        raise ValueError("ae.pt does not contain a 5D latent model")
    if latent_raw.shape != latent_scaled.shape or latent_scaled.shape != (22604, 5):
        raise ValueError(f"Unexpected latent shape: {latent_scaled.shape}")
    if len(obs_names) != len(times) or len(times) != latent_scaled.shape[0]:
        raise ValueError("Cell IDs, times, and latent arrays are not aligned")
    if not np.isfinite(latent_scaled).all():
        raise ValueError("Scaled AE5 coordinates contain non-finite values")
    minimum = np.asarray(latent_scaled).min(axis=0)
    maximum = np.asarray(latent_scaled).max(axis=0)
    if not np.allclose(minimum, -2.0, atol=2e-5, rtol=0.0):
        raise ValueError(f"Scaled latent minima are not -2: {minimum}")
    if not np.allclose(maximum, 2.0, atol=2e-5, rtol=0.0):
        raise ValueError(f"Scaled latent maxima are not 2: {maximum}")
    if scaling["latent_min"].shape != (5,) or scaling["latent_max"].shape != (5,):
        raise ValueError("Latent scaling metadata is not five-dimensional")
    counts = {
        float(value): int(np.sum(np.isclose(times, value)))
        for value in np.unique(times)
    }
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected moscot stage counts: {counts}")

    print(f"AE5 cache passed: {cache}")
    print("architecture=3000-300-5-300-3000")
    print("latent scaling=per-axis min-max to [-2,2]")
    print(f"stage counts={counts}")
    print(f"reconstruction_mse={float(config['reconstruction_mse']):.8g}")


if __name__ == "__main__":
    main()
