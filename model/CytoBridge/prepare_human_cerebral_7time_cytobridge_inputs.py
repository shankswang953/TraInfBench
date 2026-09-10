#!/usr/bin/env python3
"""Prepare balanced and biological-prior CytoBridge inputs for seven times.

CytoBridge's official neural-ODE trainer defines the target relative mass as
``n_t / n_0``, where ``n_t`` is the number of AnnData rows at a time point.
The balanced file therefore keeps each paired metacell once.  The unbalanced
file deterministically repeats metacells so its row-count ratios encode the
specified D4-normalized biological mass prior without changing CytoBridge.
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
import hashlib
import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


DEFAULT_SOURCE = Path(
    "external/COATI/humanCerebral/"
    "Data/selected_4_7_9_11_12_18_21"
)
PREFIX = "human_cerebral_7time_d4_d21_no_d16_rna_cytobridge"
TIME_KEYS = ("age_4", "age_7", "age_9", "age_11", "age_12", "age_18", "age_21")
AGES = (4, 7, 9, 11, 12, 18, 21)
TIMES = (0.0, 0.3, 0.5, 0.7, 0.8, 1.4, 1.7)
SOURCE_COUNTS = (501, 1309, 1630, 1560, 4100, 5842, 5721)
BIOLOGICAL_COUNTS = (20390, 59628, 84583, 109250, 121583, 467760, 693000)
MASS_PRIOR = (1.00, 2.92, 4.15, 5.36, 5.96, 22.94, 33.99)
RNA_SCALE = 63.89521587795941


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--prior-base-rows",
        type=int,
        default=1000,
        help=(
            "Rows used for D4 in the prior-encoded file. The default makes all "
            "two-decimal mass ratios exact integers and retains every source metacell."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)


def protected_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")


def load_blocks(path: Path, expected_dim: int) -> list[np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        if tuple(source.files) != TIME_KEYS:
            raise ValueError(f"Unexpected time keys in {path}: {source.files}")
        blocks = [np.asarray(source[key], dtype=np.float32) for key in TIME_KEYS]
    shapes = tuple(block.shape for block in blocks)
    expected = tuple((count, expected_dim) for count in SOURCE_COUNTS)
    if shapes != expected:
        raise ValueError(f"Unexpected shapes in {path}: {shapes}; expected {expected}")
    if not all(np.isfinite(block).all() for block in blocks):
        raise ValueError(f"Non-finite values in {path}")
    return blocks


def make_base_obs(source_obs: pd.DataFrame) -> pd.DataFrame:
    expected_keys = np.concatenate(
        [np.repeat(key, count) for key, count in zip(TIME_KEYS, SOURCE_COUNTS)]
    )
    expected_ages = np.concatenate(
        [np.repeat(age, count) for age, count in zip(AGES, SOURCE_COUNTS)]
    )
    if source_obs.shape[0] != sum(SOURCE_COUNTS):
        raise ValueError("paired_obs row count does not match the seven source blocks")
    if not np.array_equal(source_obs["time_key"].astype(str), expected_keys):
        raise ValueError("paired_obs time order differs from the source arrays")
    if not np.array_equal(
        source_obs["processed_age"].to_numpy(dtype=np.int64), expected_ages
    ):
        raise ValueError("paired_obs ages differ from the manifest")

    obs = source_obs.copy()
    obs["source_cell_id"] = obs["paired_metacell_id"].astype(str)
    obs.index = pd.Index(obs["source_cell_id"], name="cell_id")
    if not obs.index.is_unique:
        raise ValueError("Source paired metacell IDs are not unique")
    obs["time_index"] = np.concatenate(
        [np.repeat(index, count) for index, count in enumerate(SOURCE_COUNTS)]
    )
    obs["time_point_processed"] = np.concatenate(
        [np.repeat(time, count) for time, count in zip(TIMES, SOURCE_COUNTS)]
    ).astype(np.float32)
    obs["time_continuous"] = obs["time_point_processed"].to_numpy()
    obs["age_day"] = expected_ages
    obs["biological_cell_count"] = np.concatenate(
        [np.repeat(count, n) for count, n in zip(BIOLOGICAL_COUNTS, SOURCE_COUNTS)]
    )
    obs["biological_mass_d4_normalized"] = np.concatenate(
        [np.repeat(mass, n) for mass, n in zip(MASS_PRIOR, SOURCE_COUNTS)]
    )
    return obs


def make_adata(
    normalized: np.ndarray,
    raw: np.ndarray,
    obs: pd.DataFrame,
    *,
    task: str,
    source_dir: Path,
) -> ad.AnnData:
    var = pd.DataFrame(index=[f"PC{index + 1}" for index in range(normalized.shape[1])])
    adata = ad.AnnData(X=normalized.copy(), obs=obs.copy(), var=var)
    adata.obsm["X_latent"] = normalized.copy()
    adata.obsm["X_pca"] = normalized.copy()
    adata.obsm["X_pca_raw"] = raw.copy()
    adata.uns["dataset"] = "human_cerebral"
    adata.uns["task"] = task
    adata.uns["source_directory"] = str(source_dir)
    adata.uns["time_keys"] = np.asarray(TIME_KEYS)
    adata.uns["physical_time_values"] = np.asarray(TIMES, dtype=np.float32)
    adata.uns["excluded_time_keys"] = np.asarray(("age_16", "age_26", "age_31", "age_61"))
    adata.uns["model_space"] = "seven-time common-normalized RNA PCA30"
    adata.uns["rna_raw_to_model_scale"] = RNA_SCALE
    adata.uns["biological_mass_d4_normalized"] = np.asarray(MASS_PRIOR, dtype=np.float64)
    return adata


def deterministic_prior_replication(
    normalized_blocks: list[np.ndarray],
    raw_blocks: list[np.ndarray],
    source_obs: pd.DataFrame,
    target_counts: tuple[int, ...],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, dict[str, int]]]:
    rng = np.random.default_rng(seed)
    norm_outputs: list[np.ndarray] = []
    raw_outputs: list[np.ndarray] = []
    obs_outputs: list[pd.DataFrame] = []
    replication: dict[str, dict[str, int]] = {}
    offset = 0
    for time_index, (key, source_count, target_count) in enumerate(
        zip(TIME_KEYS, SOURCE_COUNTS, target_counts)
    ):
        if target_count < source_count:
            raise ValueError(
                f"prior-base-rows is too small: {key} target {target_count} < source {source_count}"
            )
        permutation = rng.permutation(source_count)
        selected = np.resize(permutation, target_count)
        norm_outputs.append(normalized_blocks[time_index][selected])
        raw_outputs.append(raw_blocks[time_index][selected])

        block_obs = source_obs.iloc[offset : offset + source_count].iloc[selected].copy()
        block_obs["source_cell_id"] = block_obs.index.astype(str)
        block_obs["prior_encoded_row"] = np.arange(target_count, dtype=np.int64)
        block_obs["time_index"] = time_index
        block_obs["time_point_processed"] = np.float32(TIMES[time_index])
        block_obs["time_continuous"] = np.float32(TIMES[time_index])
        block_obs["age_day"] = AGES[time_index]
        block_obs["biological_cell_count"] = BIOLOGICAL_COUNTS[time_index]
        block_obs["biological_mass_d4_normalized"] = MASS_PRIOR[time_index]
        block_obs.index = pd.Index(
            [f"{key}__cytoprior_{index:05d}" for index in range(target_count)],
            name="cell_id",
        )
        obs_outputs.append(block_obs)

        occurrence = np.bincount(selected, minlength=source_count)
        replication[key] = {
            "source_rows": source_count,
            "target_rows": target_count,
            "min_repeats": int(occurrence.min()),
            "max_repeats": int(occurrence.max()),
            "unique_source_rows_retained": int(np.count_nonzero(occurrence)),
        }
        offset += source_count

    return (
        np.vstack(norm_outputs).astype(np.float32, copy=False),
        np.vstack(raw_outputs).astype(np.float32, copy=False),
        pd.concat(obs_outputs, axis=0),
        replication,
    )


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    out_dir = args.out_dir.resolve()
    source_paths = {
        "raw": source_dir / "rna_pca30_by_time.npz",
        "normalized": source_dir / "rna_pca30_normalized_by_time.npz",
        "obs": source_dir / "paired_obs.csv",
        "manifest": source_dir / "manifest.json",
        "norm": source_dir / "primal_norm_params.pt",
        "norm_json": source_dir / "rna_normalization.json",
        "intervals": source_dir / "rna_exact_emd_intervals.csv",
        "prior": source_dir / "biological_prior.json",
    }
    for path in source_paths.values():
        require_file(path)

    if args.prior_base_rows <= 0:
        raise ValueError("--prior-base-rows must be positive")
    target_counts_float = np.asarray(MASS_PRIOR) * args.prior_base_rows
    if not np.allclose(target_counts_float, np.round(target_counts_float), atol=1e-10):
        raise ValueError("prior-base-rows does not make the mass prior integer-valued")
    target_counts = tuple(int(value) for value in np.round(target_counts_float))

    manifest = json.loads(source_paths["manifest"].read_text(encoding="utf-8"))
    if tuple(manifest["time_keys"]) != TIME_KEYS:
        raise ValueError("Source manifest does not contain the expected seven times")
    if tuple(manifest["paired_counts"][key] for key in TIME_KEYS) != SOURCE_COUNTS:
        raise ValueError("Source manifest paired counts changed")
    if not np.allclose(manifest["model_time_points"], TIMES, atol=1e-12, rtol=0.0):
        raise ValueError("Source physical times changed")
    if not np.allclose(
        manifest["biological_prior_d4_normalized"], MASS_PRIOR, atol=1e-12, rtol=0.0
    ):
        raise ValueError("Source biological prior changed")
    if not np.isclose(float(manifest["rna_scale"]), RNA_SCALE):
        raise ValueError("Source RNA normalization scale changed")

    raw_blocks = load_blocks(source_paths["raw"], 30)
    normalized_blocks = load_blocks(source_paths["normalized"], 30)
    for key, raw, normalized in zip(TIME_KEYS, raw_blocks, normalized_blocks):
        if not np.allclose(normalized, raw / RNA_SCALE, atol=2e-6, rtol=2e-6):
            raise ValueError(f"Normalized coordinates do not equal raw/RNA_SCALE at {key}")

    source_obs = make_base_obs(pd.read_csv(source_paths["obs"]))
    raw = np.vstack(raw_blocks).astype(np.float32, copy=False)
    normalized = np.vstack(normalized_blocks).astype(np.float32, copy=False)
    balanced = make_adata(
        normalized,
        raw,
        source_obs,
        task="full_7time_balanced_no_d16",
        source_dir=source_dir,
    )
    balanced.uns["cytobridge_mass_semantics"] = (
        "balanced velocity-only model; biological prior is recorded but mass loss is disabled"
    )

    prior_norm, prior_raw, prior_obs, replication = deterministic_prior_replication(
        normalized_blocks,
        raw_blocks,
        source_obs,
        target_counts,
        args.seed,
    )
    unbalanced = make_adata(
        prior_norm,
        prior_raw,
        prior_obs,
        task="full_7time_unbalanced_biological_prior_no_d16",
        source_dir=source_dir,
    )
    unbalanced.uns["cytobridge_mass_semantics"] = (
        "official trainer uses rows_at_t/rows_at_D4; deterministic uniform replication "
        "encodes the D4-normalized biological prior"
    )
    unbalanced.uns["prior_base_rows"] = args.prior_base_rows
    unbalanced.uns["prior_encoded_row_counts"] = np.asarray(target_counts, dtype=np.int64)
    unbalanced.uns["replication_seed"] = args.seed

    out_dir.mkdir(parents=True, exist_ok=True)
    balanced_path = out_dir / f"{PREFIX}_balanced.h5ad"
    unbalanced_path = out_dir / f"{PREFIX}_unbalanced_biological_prior.h5ad"
    metadata_path = out_dir / f"{PREFIX}_metadata.json"
    norm_path = out_dir / f"{PREFIX}_primal_norm_params.pt"
    norm_json_path = out_dir / f"{PREFIX}_rna_normalization.json"
    intervals_path = out_dir / f"{PREFIX}_rna_exact_emd_intervals.csv"
    prior_path = out_dir / f"{PREFIX}_biological_prior.json"
    outputs = (
        balanced_path,
        unbalanced_path,
        metadata_path,
        norm_path,
        norm_json_path,
        intervals_path,
        prior_path,
    )
    for path in outputs:
        protected_output(path, args.overwrite)

    balanced.write_h5ad(balanced_path, compression="gzip")
    unbalanced.write_h5ad(unbalanced_path, compression="gzip")
    shutil.copy2(source_paths["norm"], norm_path)
    shutil.copy2(source_paths["norm_json"], norm_json_path)
    shutil.copy2(source_paths["intervals"], intervals_path)
    shutil.copy2(source_paths["prior"], prior_path)

    metadata = {
        "dataset": "human_cerebral",
        "task": "CytoBridge full seven-time D4--D21 excluding D16",
        "time_keys": list(TIME_KEYS),
        "ages_days": list(AGES),
        "physical_time_values": list(TIMES),
        "excluded_time_keys": ["age_16", "age_26", "age_31", "age_61"],
        "source_paired_metacell_counts": dict(zip(TIME_KEYS, SOURCE_COUNTS)),
        "biological_cell_counts": dict(zip(TIME_KEYS, BIOLOGICAL_COUNTS)),
        "biological_mass_d4_normalized": dict(zip(TIME_KEYS, MASS_PRIOR)),
        "model_space": "seven-time common-normalized RNA PCA30",
        "rna_raw_to_model_scale": RNA_SCALE,
        "balanced": {
            "rows": int(balanced.n_obs),
            "mass_loss": False,
            "coordinate_sha256": sha256_array(normalized),
            "output": str(balanced_path),
        },
        "unbalanced": {
            "rows": int(unbalanced.n_obs),
            "mass_loss": True,
            "mass_encoding": "deterministic uniform row replication; official model unchanged",
            "prior_base_rows": args.prior_base_rows,
            "prior_encoded_row_counts": dict(zip(TIME_KEYS, target_counts)),
            "realized_relative_mass": dict(
                zip(TIME_KEYS, [count / target_counts[0] for count in target_counts])
            ),
            "replication": replication,
            "output": str(unbalanced_path),
        },
        "source_directory": str(source_dir),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"balanced:   {balanced_path} shape={balanced.shape}")
    print(f"unbalanced: {unbalanced_path} shape={unbalanced.shape}")
    print(f"times:      {list(zip(TIME_KEYS, TIMES))}")
    print(f"mass prior: {list(MASS_PRIOR)}")
    print(f"prior rows: {list(target_counts)}")
    print(f"metadata:   {metadata_path}")


if __name__ == "__main__":
    main()
