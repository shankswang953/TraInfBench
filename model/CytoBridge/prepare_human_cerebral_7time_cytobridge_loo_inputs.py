#!/usr/bin/env python3
"""Prepare strict CytoBridge LOO inputs for the five internal seven-time ages."""

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
TIME_KEYS = ("age_4", "age_7", "age_9", "age_11", "age_12", "age_18", "age_21")
TIMES = (0.0, 0.3, 0.5, 0.7, 0.8, 1.4, 1.7)
SOURCE_COUNTS = (501, 1309, 1630, 1560, 4100, 5842, 5721)
PRIOR_COUNTS = (1000, 2920, 4150, 5360, 5960, 22940, 33990)
MASS_PRIOR = (1.00, 2.92, 4.15, 5.36, 5.96, 22.94, 33.99)
DEFAULT_HELD_OUT_AGES = (7, 9, 11, 12, 18)
RNA_SCALE = 63.89521587795941

TIME_BY_AGE = dict(zip(AGES, TIMES))
KEY_BY_AGE = dict(zip(AGES, TIME_KEYS))
SOURCE_COUNT_BY_AGE = dict(zip(AGES, SOURCE_COUNTS))
PRIOR_COUNT_BY_AGE = dict(zip(AGES, PRIOR_COUNTS))
MASS_BY_AGE = dict(zip(AGES, MASS_PRIOR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--ages", nargs="+", type=int, default=list(DEFAULT_HELD_OUT_AGES)
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)


def protect(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Outputs exist; pass --overwrite to replace them:\n"
            + "\n".join(str(path) for path in existing)
        )


def observed_counts(adata: ad.AnnData) -> dict[int, int]:
    times = adata.obs["time_point_processed"].to_numpy(dtype=float)
    return {
        age: int(np.isclose(times, TIME_BY_AGE[age], atol=1e-7).sum())
        for age in AGES
    }


def validate_source(adata: ad.AnnData, expected: tuple[int, ...], mode: str) -> None:
    if "X_latent" not in adata.obsm or "X_pca_raw" not in adata.obsm:
        raise KeyError(f"{mode} source lacks X_latent or X_pca_raw")
    if "time_point_processed" not in adata.obs:
        raise KeyError(f"{mode} source lacks time_point_processed")
    expected_by_age = dict(zip(AGES, expected))
    if observed_counts(adata) != expected_by_age:
        raise ValueError(
            f"{mode} source counts changed: {observed_counts(adata)} != {expected_by_age}"
        )
    if not np.allclose(
        np.asarray(adata.obsm["X_latent"]),
        np.asarray(adata.obsm["X_pca_raw"]) / RNA_SCALE,
        atol=2e-6,
        rtol=2e-6,
    ):
        raise ValueError(f"{mode} source is not in the seven-time normalized PCA30 space")


def strict_subset(
    source: ad.AnnData, held_out_age: int, mode: str
) -> ad.AnnData:
    held_out_time = TIME_BY_AGE[held_out_age]
    times = source.obs["time_point_processed"].to_numpy(dtype=float)
    keep = ~np.isclose(times, held_out_time, atol=1e-7)
    output = source[keep].copy()
    retained_ages = [age for age in AGES if age != held_out_age]
    retained_keys = [KEY_BY_AGE[age] for age in retained_ages]
    retained_times = [TIME_BY_AGE[age] for age in retained_ages]
    retained_mass = [MASS_BY_AGE[age] for age in retained_ages]

    output.uns["task"] = f"strict_loo_D{held_out_age}_{mode}_7time_no_d16"
    output.uns["strict_loo"] = True
    output.uns["loo_heldout_age"] = int(held_out_age)
    output.uns["loo_heldout_key"] = KEY_BY_AGE[held_out_age]
    output.uns["loo_heldout_time"] = float(held_out_time)
    output.uns["loo_train_ages"] = np.asarray(retained_ages, dtype=np.int64)
    output.uns["loo_train_times"] = np.asarray(retained_times, dtype=np.float32)
    output.uns["time_keys"] = np.asarray(retained_keys)
    output.uns["physical_time_values"] = np.asarray(retained_times, dtype=np.float32)
    output.uns["biological_mass_d4_normalized"] = np.asarray(
        retained_mass, dtype=np.float64
    )
    output.uns["heldout_rows_removed"] = int((~keep).sum())
    output.uns["model_space"] = "seven-time common-normalized RNA PCA30"
    output.uns["rna_raw_to_model_scale"] = RNA_SCALE
    if mode == "unbalanced":
        output.uns["prior_encoded_row_counts"] = np.asarray(
            [PRIOR_COUNT_BY_AGE[age] for age in retained_ages], dtype=np.int64
        )
        output.uns["cytobridge_mass_semantics"] = (
            "official rows_at_t/rows_at_D4 over retained training times; "
            "held-out time has no rows and no loss contribution"
        )
    else:
        output.uns["cytobridge_mass_semantics"] = (
            "balanced velocity-only strict LOO; mass loss disabled"
        )
    return output


def save_reference(
    balanced: ad.AnnData, held_out_age: int, output: Path
) -> None:
    times = balanced.obs["time_point_processed"].to_numpy(dtype=float)
    heldout_mask = np.isclose(times, TIME_BY_AGE[held_out_age], atol=1e-7)
    initial_mask = np.isclose(times, TIMES[0], atol=1e-7)
    terminal_mask = np.isclose(times, TIMES[-1], atol=1e-7)
    latent = np.asarray(balanced.obsm["X_latent"], dtype=np.float32)
    raw = np.asarray(balanced.obsm["X_pca_raw"], dtype=np.float32)
    retained_ages = [age for age in AGES if age != held_out_age]
    np.savez_compressed(
        output,
        initial=latent[initial_mask],
        heldout=latent[heldout_mask],
        terminal=latent[terminal_mask],
        initial_raw=raw[initial_mask],
        heldout_raw=raw[heldout_mask],
        terminal_raw=raw[terminal_mask],
        initial_ids=balanced.obs_names[initial_mask].astype(str).to_numpy(),
        heldout_ids=balanced.obs_names[heldout_mask].astype(str).to_numpy(),
        terminal_ids=balanced.obs_names[terminal_mask].astype(str).to_numpy(),
        initial_time=np.asarray(TIMES[0], dtype=np.float32),
        heldout_time=np.asarray(TIME_BY_AGE[held_out_age], dtype=np.float32),
        terminal_time=np.asarray(TIMES[-1], dtype=np.float32),
        train_times=np.asarray(
            [TIME_BY_AGE[age] for age in retained_ages], dtype=np.float32
        ),
        heldout_age=np.asarray(held_out_age, dtype=np.int64),
        rna_scale=np.asarray(RNA_SCALE, dtype=np.float64),
    )


def main() -> None:
    args = parse_args()
    ages = list(dict.fromkeys(args.ages))
    invalid = [age for age in ages if age not in DEFAULT_HELD_OUT_AGES]
    if invalid:
        raise ValueError(
            f"Only internal ages {list(DEFAULT_HELD_OUT_AGES)} are supported; got {invalid}"
        )

    data_dir = args.data_dir.resolve()
    balanced_source_path = data_dir / f"{PREFIX}_balanced.h5ad"
    unbalanced_source_path = (
        data_dir / f"{PREFIX}_unbalanced_biological_prior.h5ad"
    )
    suite_path = data_dir / f"{PREFIX}_strict_loo_suite_metadata.json"
    if suite_path.exists() and not args.overwrite:
        raise FileExistsError(f"{suite_path} exists; pass --overwrite to replace it")
    require(balanced_source_path)
    require(unbalanced_source_path)
    balanced_source = ad.read_h5ad(balanced_source_path)
    unbalanced_source = ad.read_h5ad(unbalanced_source_path)
    validate_source(balanced_source, SOURCE_COUNTS, "balanced")
    validate_source(unbalanced_source, PRIOR_COUNTS, "unbalanced")

    suite = {
        "dataset": "human_cerebral",
        "source_timeline": list(AGES),
        "held_out_ages": ages,
        "rna_scale": RNA_SCALE,
        "settings": {},
    }
    for held_out_age in ages:
        stem = f"{PREFIX}_strict_loo_D{held_out_age}"
        balanced_path = data_dir / f"{stem}_balanced.h5ad"
        unbalanced_path = data_dir / f"{stem}_unbalanced_biological_prior.h5ad"
        reference_path = data_dir / f"{stem}_reference.npz"
        metadata_path = data_dir / f"{stem}_metadata.json"
        protect(
            [balanced_path, unbalanced_path, reference_path, metadata_path],
            args.overwrite,
        )

        balanced = strict_subset(balanced_source, held_out_age, "balanced")
        unbalanced = strict_subset(unbalanced_source, held_out_age, "unbalanced")
        balanced.write_h5ad(balanced_path, compression="gzip")
        unbalanced.write_h5ad(unbalanced_path, compression="gzip")
        save_reference(balanced_source, held_out_age, reference_path)

        retained_ages = [age for age in AGES if age != held_out_age]
        metadata = {
            "strict_loo": True,
            "held_out_age": held_out_age,
            "held_out_key": KEY_BY_AGE[held_out_age],
            "held_out_time": TIME_BY_AGE[held_out_age],
            "held_out_balanced_reference_rows": SOURCE_COUNT_BY_AGE[held_out_age],
            "retained_ages": retained_ages,
            "retained_times": [TIME_BY_AGE[age] for age in retained_ages],
            "balanced_training_rows": {
                str(age): SOURCE_COUNT_BY_AGE[age] for age in retained_ages
            },
            "unbalanced_training_rows": {
                str(age): PRIOR_COUNT_BY_AGE[age] for age in retained_ages
            },
            "retained_biological_mass_d4_normalized": {
                str(age): MASS_BY_AGE[age] for age in retained_ages
            },
            "normalization": {
                "space": "common seven-time RNA PCA30",
                "scale": RNA_SCALE,
                "recomputed_after_holdout": False,
            },
            "balanced_input": str(balanced_path),
            "unbalanced_input": str(unbalanced_path),
            "reference": str(reference_path),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        suite["settings"][f"D{held_out_age}"] = metadata
        print(
            f"D{held_out_age}: balanced={balanced.n_obs} rows; "
            f"unbalanced={unbalanced.n_obs} rows; heldout={SOURCE_COUNT_BY_AGE[held_out_age]} rows"
        )

    suite_path.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    print(f"[save] {suite_path}")


if __name__ == "__main__":
    main()
