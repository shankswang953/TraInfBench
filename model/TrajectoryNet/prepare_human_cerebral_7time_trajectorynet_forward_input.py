#!/usr/bin/env python3
"""Create the chronological D4-to-D21 TrajectoryNet input in the shared space."""

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

import numpy as np


PREFIX = "human_cerebral_7time_d4_d21_no_d16"
DEFAULT_SOURCE = Path(f"data/{PREFIX}_rna_trajectorynet_reversed.npz")
DEFAULT_OUTPUT = Path(f"data/{PREFIX}_rna_trajectorynet_forward.npz")
COUNTS = (501, 1309, 1630, 1560, 4100, 5842, 5721)
RNA_SCALE = 63.89521587795941


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.is_file() or args.source.stat().st_size == 0:
        raise FileNotFoundError(args.source)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")

    with np.load(args.source, allow_pickle=True) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    pca = np.asarray(payload["pca"], dtype=np.float32)
    raw = np.asarray(payload["pca_raw"], dtype=np.float32)
    forward = np.asarray(payload["original_sample_labels"], dtype=np.int64)
    if pca.shape != (sum(COUNTS), 30) or raw.shape != pca.shape:
        raise ValueError(f"Unexpected coordinate shapes: {pca.shape}, {raw.shape}")
    if tuple(int(np.sum(forward == rank)) for rank in range(7)) != COUNTS:
        raise ValueError("Chronological time-rank counts changed")
    if not np.allclose(pca, raw / RNA_SCALE, atol=2e-6, rtol=2e-6):
        raise ValueError("Input is not the shared normalized RNA PCA30 space")

    payload["sample_labels"] = forward
    payload["original_sample_labels"] = forward
    payload["forward_time_index"] = forward
    payload["time_label_mapping"] = np.asarray(
        [f"chronological_rank_{rank}->training_rank_{rank}" for rank in range(7)]
    )
    payload["direction_protocol"] = np.asarray(
        "normal chronological labels; TrajectoryNet generative direction D4 to D21"
    )
    payload["source_reversed_npz"] = np.asarray(str(args.source.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(args.output)
    print(f"[saved] {args.output}")
    print("  chronological stages: D4,D7,D9,D11,D12,D18,D21")
    print("  training labels:       0,1,2,3,4,5,6")
    print(f"  shared RNA scale:      {RNA_SCALE}")


if __name__ == "__main__":
    main()
