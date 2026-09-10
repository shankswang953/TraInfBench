#!/usr/bin/env python
"""Roll out all synthetic time1 cells for two trained CytoBridge models."""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

from analyze_cytobridge_synthetic import load_scale  # noqa: E402
from compare_synthetic_methods_coati_protocol import rollout_cytobridge  # noqa: E402


SYNTHETIC_DATA = Path(
    "external/COATI/Synthetic/5scRNA"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balanced-adata", type=Path, required=True)
    parser.add_argument("--unbalanced-adata", type=Path, required=True)
    parser.add_argument(
        "--rna-data",
        type=Path,
        default=SYNTHETIC_DATA / "all_time_scRNA_pca10.npz",
    )
    parser.add_argument(
        "--rna-norm",
        type=Path,
        default=SYNTHETIC_DATA / "primal_norm_params_rna10_w2.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps-per-interval", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda"), default="cpu"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    if args.steps_per_interval != 20:
        raise ValueError("Formal observed times require 20 steps per interval")

    rna = np.load(args.rna_data, allow_pickle=False)
    initial_raw = np.asarray(rna["time1"], dtype=np.float32)
    if initial_raw.shape != (2006, 10):
        raise ValueError(f"Unexpected time1 shape: {initial_raw.shape}")
    scale = load_scale(args.rna_norm)
    initial_normalized = (initial_raw / scale).astype(np.float32)
    device = torch.device(args.device)

    arrays: dict[str, np.ndarray] = {
        "selected_physical_times": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "rna_scale": np.asarray(scale, dtype=np.float64),
    }
    for prefix, path in (
        ("cytobridge_balanced", args.balanced_adata),
        ("cytobridge_unbalanced", args.unbalanced_adata),
    ):
        print(f"Rolling out {prefix}", flush=True)
        trajectory_normalized, log_weights = rollout_cytobridge(
            path,
            initial_normalized,
            device,
            args.batch_size,
            args.steps_per_interval,
        )
        expected = (61, 2006, 10)
        if trajectory_normalized.shape != expected:
            raise ValueError(
                f"Unexpected {prefix} trajectory shape: {trajectory_normalized.shape}"
            )
        if log_weights.shape != (61, 2006):
            raise ValueError(
                f"Unexpected {prefix} log-weight shape: {log_weights.shape}"
            )
        selected = np.asarray([20, 40, 60], dtype=np.int64)
        arrays[f"{prefix}_all_initial_selected_rna10_raw"] = (
            trajectory_normalized[selected] * scale
        ).astype(np.float32)
        arrays[f"{prefix}_all_initial_selected_log_weights"] = np.asarray(
            log_weights[selected], dtype=np.float32
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
