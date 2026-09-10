#!/usr/bin/env python3
"""Attach physical ages/times to a Full or strict-LOO rank-clock trajectory."""

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


PHYSICAL_TIMES = np.asarray((0.0, 0.3, 0.5, 0.7, 0.8, 1.4, 1.7), dtype=np.float32)
AGES = np.asarray((4, 7, 9, 11, 12, 18, 21), dtype=np.int64)
HELD_OUT_AGES = (7, 9, 11, 12, 18)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--heldout-age",
        type=int,
        choices=HELD_OUT_AGES,
        help=(
            "Strict-LOO age. When supplied, map the six retained MIOFlow ranks "
            "to their unchanged physical times and record the held-out fractional rank."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.result_dir / "trajectories.npz"
    output = args.result_dir / "trajectory_time_mapping.npz"
    manifest = args.result_dir / "trajectory_time_mapping.json"
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    existing = [path for path in (output, manifest) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing time mapping:\n"
            + "\n".join(str(path) for path in existing)
        )
    with np.load(source, allow_pickle=True) as archive:
        rank_grid = np.asarray(archive["time_grid"], dtype=np.float32)
    if args.heldout_age is None:
        observed_ages = AGES
        observed_physical_times = PHYSICAL_TIMES
    else:
        keep = AGES != args.heldout_age
        observed_ages = AGES[keep]
        observed_physical_times = PHYSICAL_TIMES[keep]
    ranks = np.arange(len(observed_ages), dtype=np.float32)
    if (
        rank_grid.ndim != 1
        or not np.isclose(rank_grid[0], 0)
        or not np.isclose(rank_grid[-1], ranks[-1])
    ):
        raise ValueError(
            f"Expected a rank grid from 0 to {ranks[-1]:g}, "
            f"found {rank_grid[[0, -1]]}"
        )
    if not np.all(np.diff(rank_grid) > 0):
        raise ValueError("MIOFlow rank grid is not strictly increasing")
    physical_grid = np.interp(
        rank_grid, ranks, observed_physical_times
    ).astype(np.float32)
    observed_indices = []
    for rank in ranks:
        matches = np.flatnonzero(np.isclose(rank_grid, rank, atol=1e-6, rtol=0.0))
        if len(matches) != 1:
            raise ValueError(f"Rank grid does not contain observed rank {rank}")
        observed_indices.append(int(matches[0]))
    heldout_rank = np.nan
    heldout_physical_time = np.nan
    heldout_nearest_index = -1
    if args.heldout_age is not None:
        heldout_index = int(np.flatnonzero(AGES == args.heldout_age)[0])
        heldout_physical_time = float(PHYSICAL_TIMES[heldout_index])
        heldout_rank = float(
            np.interp(
                heldout_physical_time,
                observed_physical_times,
                ranks,
            )
        )
        heldout_nearest_index = int(np.argmin(np.abs(rank_grid - heldout_rank)))
    np.savez_compressed(
        output,
        rank_time_grid=rank_grid,
        physical_time_grid_piecewise=physical_grid,
        observed_ranks=ranks,
        observed_physical_times=observed_physical_times,
        observed_ages=observed_ages,
        observed_time_indices=np.asarray(observed_indices, dtype=np.int64),
        heldout_age=np.asarray(
            -1 if args.heldout_age is None else args.heldout_age, dtype=np.int64
        ),
        heldout_physical_time=np.asarray(heldout_physical_time, dtype=np.float32),
        heldout_interpolated_rank=np.asarray(heldout_rank, dtype=np.float32),
        heldout_nearest_time_index=np.asarray(heldout_nearest_index, dtype=np.int64),
    )
    manifest.write_text(
        json.dumps(
            {
                "method_clock": (
                    f"official MIOFlow consecutive ranks 0..{len(ranks) - 1}"
                ),
                "physical_clock": observed_physical_times.tolist(),
                "ages": observed_ages.tolist(),
                "mapping": "piecewise linear between observed rank/time pairs",
                "rank_grid_points": int(len(rank_grid)),
                "observed_time_indices": observed_indices,
                "heldout_age": args.heldout_age,
                "heldout_physical_time": (
                    None if args.heldout_age is None else heldout_physical_time
                ),
                "heldout_interpolated_rank": (
                    None if args.heldout_age is None else heldout_rank
                ),
                "heldout_nearest_time_index": (
                    None if args.heldout_age is None else heldout_nearest_index
                ),
                "important": (
                    "The ODE was trained and solved on rank time. Physical times are "
                    "metadata/mapping for comparison and must not be mistaken for the training clock."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[saved] {output}")
    print(f"  observed indices: {observed_indices}")
    if args.heldout_age is not None:
        print(
            f"  held-out D{args.heldout_age}: physical={heldout_physical_time:g}, "
            f"interpolated rank={heldout_rank:g}, nearest index={heldout_nearest_index}"
        )


if __name__ == "__main__":
    main()
