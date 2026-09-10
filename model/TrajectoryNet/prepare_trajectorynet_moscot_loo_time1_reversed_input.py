#!/usr/bin/env python
"""Prepare strict-LOO reversed-rank TrajectoryNet input for pancreas E15.5."""

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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/moscot_rna_loo_time1_trajectorynet.npz"
DEFAULT_REFERENCE = ROOT / "data/moscot_rna_loo_time1_reference.npz"
DEFAULT_OUTPUT = ROOT / "data/moscot_rna_loo_time1_trajectorynet_reversed.npz"
EXPECTED_COUNTS = {0.0: 9029, 2.0: 3242}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference-npz", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-npz", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_source(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as archive:
        if "pca" not in archive.files or "sample_labels" not in archive.files:
            raise ValueError(f"{path} must contain pca and sample_labels")
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    pca = np.asarray(payload["pca"], dtype=np.float32)
    physical_labels = np.asarray(payload["sample_labels"], dtype=np.float64)
    if pca.shape != (sum(EXPECTED_COUNTS.values()), 50):
        raise ValueError(f"Expected retained PCA shape (12271, 50), found {pca.shape}")
    unique, counts = np.unique(physical_labels, return_counts=True)
    if not np.array_equal(unique, np.asarray([0.0, 2.0])):
        raise ValueError(f"Expected physical labels [0, 2], found {unique.tolist()}")
    expected_counts = np.asarray([EXPECTED_COUNTS[value] for value in unique])
    if not np.array_equal(counts, expected_counts):
        raise ValueError(
            f"Unexpected retained cell counts: {dict(zip(unique.tolist(), counts.tolist()))}"
        )
    return payload, pca, physical_labels


def validate_reference(path: Path, pca: np.ndarray, physical_labels: np.ndarray) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as reference:
        initial = np.asarray(reference["initial"], dtype=np.float32)
        terminal = np.asarray(reference["terminal"], dtype=np.float32)
        heldout = np.asarray(reference["heldout"], dtype=np.float32)
        initial_time = float(reference["initial_time"])
        heldout_time = float(reference["heldout_time"])
        terminal_time = float(reference["terminal_time"])
    if not np.isclose(initial_time, 0.0):
        raise ValueError(f"Expected E14.5 initial time 0, found {initial_time}")
    if not np.isclose(heldout_time, 1.0):
        raise ValueError(f"Expected E15.5 held-out time 1, found {heldout_time}")
    if not np.isclose(terminal_time, 2.0):
        raise ValueError(f"Expected E16.5 terminal time 2, found {terminal_time}")
    if initial.shape != (9029, 50) or terminal.shape != (3242, 50):
        raise ValueError(
            f"Unexpected reference endpoints: initial={initial.shape}, terminal={terminal.shape}"
        )
    if heldout.shape[1:] != (50,) or len(heldout) == 0:
        raise ValueError(f"Unexpected held-out reference shape: {heldout.shape}")
    if not np.array_equal(initial, pca[np.isclose(physical_labels, 0.0)]):
        raise ValueError("LOO initial reference is not aligned to the TrajectoryNet input")
    if not np.array_equal(terminal, pca[np.isclose(physical_labels, 2.0)]):
        raise ValueError("LOO terminal reference is not aligned to the TrajectoryNet input")


def reversed_labels(physical_labels: np.ndarray) -> np.ndarray:
    output = np.full(len(physical_labels), -1, dtype=np.int64)
    output[np.isclose(physical_labels, 0.0)] = 1
    output[np.isclose(physical_labels, 2.0)] = 0
    if np.any(output < 0):
        raise ValueError("Unexpected physical label while constructing reversed ranks")
    return output


def validate_output(
    path: Path,
    source_pca: np.ndarray,
    physical_labels: np.ndarray,
) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    expected = reversed_labels(physical_labels)
    with np.load(path, allow_pickle=True) as output:
        pca = np.asarray(output["pca"], dtype=np.float32)
        labels = np.asarray(output["sample_labels"], dtype=np.int64)
        stored_physical = np.asarray(output["physical_sample_labels"], dtype=float)
        heldout_physical_time = float(output["heldout_physical_time"])
        heldout_reversed_rank = float(output["heldout_reversed_rank"])
    if not np.array_equal(pca, source_pca):
        raise ValueError("Reversed input changed the PCA coordinates or cell order")
    if not np.array_equal(labels, expected):
        raise ValueError("Reversed labels are not E14.5->1 and E16.5->0")
    if not np.array_equal(stored_physical, physical_labels):
        raise ValueError("Stored physical labels do not match the source")
    if not np.isclose(heldout_physical_time, 1.0):
        raise ValueError("Held-out physical time is not E15.5/time 1")
    if not np.isclose(heldout_reversed_rank, 0.5):
        raise ValueError("Held-out reversed rank is not 0.5")
    unique, counts = np.unique(labels, return_counts=True)
    summary = {
        "pca_shape": list(pca.shape),
        "reversed_ranks": unique.tolist(),
        "reversed_rank_counts": counts.tolist(),
        "initial_e145_rank": 1,
        "terminal_e165_rank": 0,
        "heldout_e155_rank": 0.5,
        "heldout_cells_in_training": False,
        "coordinates_and_order_unchanged": True,
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    payload, pca, physical_labels = load_source(args.input_npz)
    validate_reference(args.reference_npz, pca, physical_labels)
    if args.check_only:
        validate_output(args.output_npz, pca, physical_labels)
        return
    if args.output_npz.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_npz} exists; pass --overwrite to replace it")

    output_payload = dict(payload)
    output_payload["sample_labels"] = reversed_labels(physical_labels)
    output_payload["physical_sample_labels"] = physical_labels.astype(np.float32)
    output_payload["retained_physical_times"] = np.asarray([0.0, 2.0], dtype=np.float32)
    output_payload["heldout_physical_time"] = np.asarray(1.0, dtype=np.float32)
    output_payload["heldout_reversed_rank"] = np.asarray(0.5, dtype=np.float32)
    output_payload["time_label_mapping"] = np.asarray(
        ["E14.5/time_0->reversed_rank_1", "E16.5/time_2->reversed_rank_0"]
    )
    output_payload["direction_protocol"] = np.asarray(
        "Gaussian generative direction E16.5 to E14.5; native density/inverse direction E14.5 to E16.5"
    )
    output_payload["source_npz"] = np.asarray(str(args.input_npz.resolve()))
    output_payload["reference_npz"] = np.asarray(str(args.reference_npz.resolve()))

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_npz.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **output_payload)
    temporary.replace(args.output_npz)
    validate_output(args.output_npz, pca, physical_labels)
    print(f"Wrote {args.output_npz}")


if __name__ == "__main__":
    main()
