#!/usr/bin/env python
"""Prepare and validate reversed-rank TrajectoryNet inputs for Palate.

Only the time labels are replaced. PCA coordinates, cell order, and strict-LOO
cell membership are unchanged. Reversed rank 0 is always observed E14.5, so
TrajectoryNet's ``reverse=True`` sampling path is Gaussian -> E14.5 -> earlier
stages. Its inverse/density path (``reverse=False`` in the upstream API) can
then start exactly from observed E12.5 cells and move in biological time.
"""

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


ROOT = Path(__file__).resolve().parents[2]
SPECS = {
    "full": {
        "input": ROOT / "data" / "palate_rna_trajectorynet.npz",
        "output": ROOT / "data" / "palate_rna_trajectorynet_reversed.npz",
        "physical_times": [0.0, 1.0, 1.5, 2.0],
        "stage_names": ["E12.5", "E13.5", "E14.0", "E14.5"],
        "heldout_time": None,
        "heldout_stage": None,
    },
    "loo_time1": {
        "input": ROOT / "data" / "palate_rna_loo_time1_trajectorynet.npz",
        "output": ROOT
        / "data"
        / "palate_rna_loo_time1_trajectorynet_reversed.npz",
        "physical_times": [0.0, 1.5, 2.0],
        "stage_names": ["E12.5", "E14.0", "E14.5"],
        "heldout_time": 1.0,
        "heldout_stage": "E13.5",
    },
    "loo_time2": {
        "input": ROOT / "data" / "palate_rna_loo_time2_trajectorynet.npz",
        "output": ROOT
        / "data"
        / "palate_rna_loo_time2_trajectorynet_reversed.npz",
        "physical_times": [0.0, 1.0, 2.0],
        "stage_names": ["E12.5", "E13.5", "E14.5"],
        "heldout_time": 1.5,
        "heldout_stage": "E14.0",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=tuple(SPECS))
    parser.add_argument("--input-npz", type=Path)
    parser.add_argument("--output-npz", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def forward_rank(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.sort(np.unique(labels))
    ranks = np.empty(len(labels), dtype=np.int64)
    for rank, label in enumerate(unique):
        ranks[np.isclose(labels, label)] = rank
    return ranks, unique


def interpolated_forward_rank(
    physical_training_times: np.ndarray, heldout_time: float
) -> float:
    stage_ranks = np.arange(len(physical_training_times), dtype=np.float64)
    return float(np.interp(heldout_time, physical_training_times, stage_ranks))


def load_source(path: Path, expected_stages: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as archive:
        if "pca" not in archive.files or "sample_labels" not in archive.files:
            raise ValueError(f"{path} must contain pca and sample_labels")
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    pca = np.asarray(payload["pca"], dtype=np.float32)
    labels = np.asarray(payload["sample_labels"], dtype=np.float64)
    ranks, unique = forward_rank(labels)
    if pca.ndim != 2 or pca.shape[1] != 40 or len(pca) != len(labels):
        raise ValueError(f"Unexpected Palate arrays: {pca.shape}, {labels.shape}")
    if len(unique) != expected_stages:
        raise ValueError(
            f"Expected {expected_stages} retained stages in {path}, found {unique.tolist()}"
        )
    if not np.array_equal(np.unique(ranks), np.arange(expected_stages)):
        raise ValueError("Could not convert source labels to consecutive forward ranks")
    if not np.isfinite(pca).all():
        raise ValueError("PCA contains non-finite values")
    return payload, pca, labels, ranks


def validate_output(
    path: Path,
    source_pca: np.ndarray,
    source_labels: np.ndarray,
    forward_ranks: np.ndarray,
    spec: dict,
) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    n_stages = len(spec["physical_times"])
    expected_reversed = n_stages - 1 - forward_ranks
    with np.load(path, allow_pickle=True) as output:
        pca = np.asarray(output["pca"], dtype=np.float32)
        reversed_ranks = np.asarray(output["sample_labels"], dtype=np.int64)
        stored_labels = np.asarray(output["original_sample_labels"], dtype=np.float64)
        stored_forward = np.asarray(output["forward_rank"], dtype=np.int64)
        physical_times = np.asarray(output["physical_training_times"], dtype=np.float32)
        stage_names = np.asarray(output["physical_training_stage_names"]).astype(str)
        heldout_reversed_rank = float(output["heldout_reversed_rank"])
    if not np.array_equal(pca, source_pca):
        raise ValueError("Reversed input changed PCA coordinates or cell order")
    if not np.array_equal(stored_labels, source_labels):
        raise ValueError("Stored original labels do not match the source")
    if not np.array_equal(stored_forward, forward_ranks):
        raise ValueError("Stored forward ranks do not match the source")
    if not np.array_equal(reversed_ranks, expected_reversed):
        raise ValueError("Reversed labels are not exactly max_rank - forward_rank")
    if not np.allclose(physical_times, spec["physical_times"], atol=1e-7, rtol=0):
        raise ValueError("Physical training times changed")
    if not np.array_equal(stage_names, np.asarray(spec["stage_names"])):
        raise ValueError("Physical stage names changed")

    heldout_time = spec["heldout_time"]
    if heldout_time is None:
        if not np.isnan(heldout_reversed_rank):
            raise ValueError("Full-data input should not have a held-out rank")
    else:
        forward = interpolated_forward_rank(physical_times, float(heldout_time))
        expected_heldout_reversed = (n_stages - 1) - forward
        if not np.isclose(heldout_reversed_rank, expected_heldout_reversed):
            raise ValueError("Incorrect held-out reversed fractional rank")

    rank_to_stage = list(reversed(spec["stage_names"]))
    counts = [int(np.sum(reversed_ranks == rank)) for rank in range(n_stages)]
    print(f"[OK] {path}")
    print(f"  cells={len(pca)}, PCA={pca.shape[1]}D, reversed-rank counts={counts}")
    print("  reverse=True sampling: Gaussian -> " + " -> ".join(rank_to_stage))
    print("  reverse=False observed-start path: " + " -> ".join(spec["stage_names"]))
    if heldout_time is not None:
        print(
            f"  held-out {spec['heldout_stage']} physical={heldout_time:g}, "
            f"reversed fractional rank={heldout_reversed_rank:g}"
        )


def main() -> None:
    args = parse_args()
    spec = dict(SPECS[args.task])
    input_path = args.input_npz or spec["input"]
    output_path = args.output_npz or spec["output"]
    payload, pca, labels, ranks = load_source(
        input_path, len(spec["physical_times"])
    )
    if args.check_only:
        validate_output(output_path, pca, labels, ranks, spec)
        return
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")

    n_stages = len(spec["physical_times"])
    reversed_ranks = (n_stages - 1 - ranks).astype(np.int64)
    heldout_reversed_rank = np.nan
    if spec["heldout_time"] is not None:
        forward = interpolated_forward_rank(
            np.asarray(spec["physical_times"], dtype=np.float64),
            float(spec["heldout_time"]),
        )
        heldout_reversed_rank = (n_stages - 1) - forward

    output_payload = dict(payload)
    output_payload["sample_labels"] = reversed_ranks
    output_payload["original_sample_labels"] = labels
    output_payload["forward_rank"] = ranks
    output_payload["reversed_rank"] = reversed_ranks
    output_payload["physical_training_times"] = np.asarray(
        spec["physical_times"], dtype=np.float32
    )
    output_payload["physical_training_stage_names"] = np.asarray(
        spec["stage_names"]
    )
    output_payload["reversed_rank_stage_names"] = np.asarray(
        list(reversed(spec["stage_names"]))
    )
    output_payload["heldout_physical_time"] = np.asarray(
        np.nan if spec["heldout_time"] is None else spec["heldout_time"],
        dtype=np.float32,
    )
    output_payload["heldout_stage"] = np.asarray(
        "none" if spec["heldout_stage"] is None else spec["heldout_stage"]
    )
    output_payload["heldout_reversed_rank"] = np.asarray(
        heldout_reversed_rank, dtype=np.float32
    )
    output_payload["direction_protocol"] = np.asarray(
        "reverse=True: Gaussian to E14.5 then earlier stages; "
        "reverse=False: observed E12.5 to later biological stages"
    )
    output_payload["source_forward_npz"] = np.asarray(str(input_path.resolve()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **output_payload)
    temporary.replace(output_path)
    validate_output(output_path, pca, labels, ranks, spec)


if __name__ == "__main__":
    main()
