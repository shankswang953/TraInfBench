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
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reverse only the time labels of the synthetic RNA10 TrajectoryNet "
            "fixture, so native density/backward integration starts at biological time1."
        )
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=Path("data/synthetic_rna10_trajectorynet.npz"),
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("data/synthetic_rna10_trajectorynet_reversed.npz"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input_npz.is_file():
        raise FileNotFoundError(args.input_npz)
    if args.output_npz.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output_npz} exists; pass --overwrite to replace it"
        )

    source = np.load(args.input_npz, allow_pickle=True)
    original_labels = np.asarray(source["sample_labels"], dtype=np.int64)
    unique = np.unique(original_labels)
    if not np.array_equal(unique, np.arange(4)):
        raise ValueError(f"Expected rank labels [0,1,2,3], found {unique.tolist()}")
    reversed_labels = 3 - original_labels

    payload = {name: source[name] for name in source.files}
    payload["sample_labels"] = reversed_labels
    payload["original_sample_labels"] = original_labels
    payload["time_label_mapping"] = np.asarray(
        ["original_time1->3", "original_time2->2", "original_time3->1", "original_time4->0"]
    )
    payload["source_npz"] = np.asarray(str(args.input_npz.resolve()))
    payload["space"] = np.asarray(
        "RNA10 PCA divided by primal W2 normalization scale; time labels reversed only"
    )

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **payload)
    print(
        f"Wrote {args.output_npz}: pca={payload['pca'].shape}, "
        f"reversed_counts={[(int(t), int((reversed_labels == t).sum())) for t in unique]}"
    )


if __name__ == "__main__":
    main()
