#!/usr/bin/env python3
"""Audit one or all strict seven-time CytoBridge LOO input pairs."""

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

import anndata as ad
import numpy as np


PREFIX = "human_cerebral_7time_d4_d21_no_d16_rna_cytobridge"
AGES = (4, 7, 9, 11, 12, 18, 21)
TIMES = (0.0, 0.3, 0.5, 0.7, 0.8, 1.4, 1.7)
SOURCE_COUNTS = (501, 1309, 1630, 1560, 4100, 5842, 5721)
PRIOR_COUNTS = (1000, 2920, 4150, 5360, 5960, 22940, 33990)
MASS_PRIOR = (1.00, 2.92, 4.15, 5.36, 5.96, 22.94, 33.99)
HELD_OUT_AGES = (7, 9, 11, 12, 18)
RNA_SCALE = 63.89521587795941

TIME_BY_AGE = dict(zip(AGES, TIMES))
SOURCE_COUNT_BY_AGE = dict(zip(AGES, SOURCE_COUNTS))
PRIOR_COUNT_BY_AGE = dict(zip(AGES, PRIOR_COUNTS))
MASS_BY_AGE = dict(zip(AGES, MASS_PRIOR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--age", default="all", help="D7, 7, or all")
    parser.add_argument(
        "--mode", choices=("balanced", "unbalanced", "all"), default="all"
    )
    return parser.parse_args()


def parse_ages(value: str) -> list[int]:
    if value.lower() == "all":
        return list(HELD_OUT_AGES)
    normalized = value.lower().removeprefix("d")
    if not normalized.isdigit() or int(normalized) not in HELD_OUT_AGES:
        raise ValueError(f"Age must be one of {HELD_OUT_AGES} or all; got {value}")
    return [int(normalized)]


def counts_by_age(adata: ad.AnnData) -> dict[int, int]:
    times = adata.obs["time_point_processed"].to_numpy(dtype=float)
    return {
        age: int(np.isclose(times, TIME_BY_AGE[age], atol=1e-7).sum())
        for age in AGES
    }


def audit_adata(
    adata: ad.AnnData, held_out_age: int, mode: str
) -> dict[str, object]:
    if not bool(adata.uns.get("strict_loo", False)):
        raise ValueError(f"D{held_out_age} {mode}: strict_loo flag is absent")
    if int(adata.uns["loo_heldout_age"]) != held_out_age:
        raise ValueError(f"D{held_out_age} {mode}: held-out age metadata changed")
    if not np.isclose(float(adata.uns["loo_heldout_time"]), TIME_BY_AGE[held_out_age]):
        raise ValueError(f"D{held_out_age} {mode}: held-out time metadata changed")
    retained_ages = [age for age in AGES if age != held_out_age]
    expected_counts = {
        age: (
            SOURCE_COUNT_BY_AGE[age]
            if mode == "balanced"
            else PRIOR_COUNT_BY_AGE[age]
        )
        for age in retained_ages
    }
    observed = counts_by_age(adata)
    if observed[held_out_age] != 0:
        raise ValueError(f"D{held_out_age} {mode}: held-out rows leaked into training")
    if {age: observed[age] for age in retained_ages} != expected_counts:
        raise ValueError(
            f"D{held_out_age} {mode}: retained counts changed: {observed}"
        )
    observed_times = sorted(
        adata.obs["time_point_processed"].astype(float).unique().tolist()
    )
    expected_times = [TIME_BY_AGE[age] for age in retained_ages]
    if not np.allclose(observed_times, expected_times, atol=1e-7, rtol=0.0):
        raise ValueError(f"D{held_out_age} {mode}: time grid changed")
    if tuple(int(age) for age in adata.uns["loo_train_ages"]) != tuple(retained_ages):
        raise ValueError(f"D{held_out_age} {mode}: loo_train_ages changed")
    if np.asarray(adata.obsm["X_latent"]).shape != (adata.n_obs, 30):
        raise ValueError(f"D{held_out_age} {mode}: wrong X_latent shape")
    if not np.allclose(
        np.asarray(adata.obsm["X_latent"]),
        np.asarray(adata.obsm["X_pca_raw"]) / RNA_SCALE,
        atol=2e-6,
        rtol=2e-6,
    ):
        raise ValueError(f"D{held_out_age} {mode}: common normalization changed")
    if mode == "unbalanced":
        realized = {
            age: observed[age] / observed[4]
            for age in retained_ages
        }
        expected_mass = {age: MASS_BY_AGE[age] for age in retained_ages}
        if not all(
            np.isclose(realized[age], expected_mass[age], atol=1e-12, rtol=0.0)
            for age in retained_ages
        ):
            raise ValueError(
                f"D{held_out_age} unbalanced: mass ratios changed: {realized}"
            )
    else:
        realized = None
    return {
        "rows": int(adata.n_obs),
        "retained_ages": retained_ages,
        "retained_times": expected_times,
        "counts": expected_counts,
        "realized_mass": realized,
        "heldout_rows_in_training": observed[held_out_age],
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    ages = parse_ages(args.age)
    modes = (
        ["balanced", "unbalanced"] if args.mode == "all" else [args.mode]
    )
    report: dict[str, object] = {}
    for held_out_age in ages:
        stem = f"{PREFIX}_strict_loo_D{held_out_age}"
        metadata_path = data_dir / f"{stem}_metadata.json"
        reference_path = data_dir / f"{stem}_reference.npz"
        if not metadata_path.is_file() or not reference_path.is_file():
            raise FileNotFoundError(f"Missing D{held_out_age} metadata/reference")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["held_out_age"] != held_out_age:
            raise ValueError(f"D{held_out_age}: metadata held-out age changed")
        with np.load(reference_path, allow_pickle=False) as reference:
            heldout_rows = len(reference["heldout"])
            heldout_raw_rows = len(reference["heldout_raw"])
            if heldout_rows != SOURCE_COUNT_BY_AGE[held_out_age]:
                raise ValueError(f"D{held_out_age}: reference row count changed")
            if heldout_raw_rows != heldout_rows:
                raise ValueError(f"D{held_out_age}: raw/normalized reference mismatch")
            if not np.allclose(
                reference["heldout"],
                reference["heldout_raw"] / RNA_SCALE,
                atol=2e-6,
                rtol=2e-6,
            ):
                raise ValueError(f"D{held_out_age}: reference normalization changed")
        report[f"D{held_out_age}"] = {"reference_rows": heldout_rows}
        for mode in modes:
            suffix = (
                "balanced.h5ad"
                if mode == "balanced"
                else "unbalanced_biological_prior.h5ad"
            )
            path = data_dir / f"{stem}_{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
            audit = audit_adata(ad.read_h5ad(path), held_out_age, mode)
            report[f"D{held_out_age}"][mode] = audit
            print(
                f"D{held_out_age} {mode}: passed; rows={audit['rows']} "
                f"times={audit['retained_times']}"
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
