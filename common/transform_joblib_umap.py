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
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["NUMBA_CACHE_DIR"] = os.environ.get(
    "UMAP_NUMBA_CACHE_DIR", "/tmp/numba-cache"
)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import joblib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform an NPY matrix with a serialized UMAP model."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    points = np.asarray(np.load(args.input), dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f"Expected a 2D input matrix, found {points.shape}")
    model = joblib.load(args.model)
    transformed = []
    for start in range(0, len(points), args.batch_size):
        transformed.append(model.transform(points[start : start + args.batch_size]))
    coordinates = np.concatenate(transformed).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, coordinates)
    print(f"Transformed {points.shape} -> {coordinates.shape}")


if __name__ == "__main__":
    main()
