#!/usr/bin/env python3
"""Prepare matched fixed-space LOO inputs for TrajectoryNet and MIOFlow.

The five internal snapshots D7, D9, D11, D12, and D18 are held out one at a
time.  TrajectoryNet keeps the exact Full normalized RNA PCA30 coordinates and
uses consecutive forward ranks for the six retained snapshots.  MIOFlow keeps
the exact Full raw PCA30 coordinates; its runner freezes the Full GAGA10 model,
GAGA input scaler, and Full latent z-score instead of fitting a new embedding.
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
import json
from pathlib import Path

import anndata as ad
import numpy as np


PREFIX = "human_cerebral_7time_d4_d21_no_d16"
AGES = np.asarray((4, 7, 9, 11, 12, 18, 21), dtype=np.int64)
TIME_KEYS = np.asarray(
    ("age_4", "age_7", "age_9", "age_11", "age_12", "age_18", "age_21")
)
PHYSICAL_TIMES = np.asarray((0.0, 0.3, 0.5, 0.7, 0.8, 1.4, 1.7), dtype=np.float64)
COUNTS = np.asarray((501, 1309, 1630, 1560, 4100, 5842, 5721), dtype=np.int64)
HELD_OUT_AGES = (7, 9, 11, 12, 18)
RNA_SCALE = 63.89521587795941
TRAJECTORYNET_TIME_SCALE = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--trajectorynet-full",
        type=Path,
        default=Path(f"data/{PREFIX}_rna_trajectorynet_forward.npz"),
    )
    parser.add_argument(
        "--mioflow-full",
        type=Path,
        default=Path(f"data/{PREFIX}_rna_mioflow.h5ad"),
    )
    parser.add_argument("--ages", nargs="+", type=int, default=list(HELD_OUT_AGES))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)


def protect(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to replace prepared LOO inputs; pass --overwrite after inspection:\n"
            + "\n".join(str(path) for path in existing)
        )


def read_trajectorynet_full(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    required = {"pca", "pca_raw", "sample_labels", "age_day", "cell_ids"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"Full TrajectoryNet input is missing keys: {sorted(missing)}")
    pca = np.asarray(payload["pca"], dtype=np.float32)
    raw = np.asarray(payload["pca_raw"], dtype=np.float32)
    labels = np.asarray(payload["sample_labels"], dtype=np.int64)
    ages = np.asarray(payload["age_day"], dtype=np.int64)
    if pca.shape != (int(COUNTS.sum()), 30) or raw.shape != pca.shape:
        raise ValueError(f"Unexpected Full TrajectoryNet coordinates: {pca.shape}")
    if not np.array_equal(np.unique(labels), np.arange(len(AGES))):
        raise ValueError("Full TrajectoryNet labels must be chronological ranks 0..6")
    if tuple(int(np.sum(ages == age)) for age in AGES) != tuple(COUNTS):
        raise ValueError("Full TrajectoryNet age counts changed")
    if not np.allclose(pca, raw / RNA_SCALE, atol=2e-6, rtol=2e-6):
        raise ValueError("Full TrajectoryNet input is not in seven-time normalized PCA30")
    return payload


def validate_mioflow_full(adata: ad.AnnData) -> None:
    for key in ("X_latent", "X_pca", "X_pca_normalized"):
        if key not in adata.obsm:
            raise KeyError(f"Full MIOFlow input is missing obsm[{key!r}]")
    if "age_day" not in adata.obs or "time_point_processed" not in adata.obs:
        raise KeyError("Full MIOFlow input lacks age_day or time_point_processed")
    raw = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    pca = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    normalized = np.asarray(adata.obsm["X_pca_normalized"], dtype=np.float32)
    if adata.n_obs != int(COUNTS.sum()) or raw.shape != (adata.n_obs, 30):
        raise ValueError(f"Unexpected Full MIOFlow shape: {adata.shape}")
    if not np.array_equal(raw, pca):
        raise ValueError("Full MIOFlow X_latent and X_pca differ")
    if not np.allclose(normalized, raw / RNA_SCALE, atol=2e-6, rtol=2e-6):
        raise ValueError("Full MIOFlow coordinates use a different PCA normalization")
    ages = adata.obs["age_day"].to_numpy(dtype=np.int64)
    if tuple(int(np.sum(ages == age)) for age in AGES) != tuple(COUNTS):
        raise ValueError("Full MIOFlow age counts changed")


def retained_rank(full_rank: np.ndarray, heldout_index: int) -> np.ndarray:
    kept = full_rank != heldout_index
    rank = full_rank[kept].copy()
    rank[rank > heldout_index] -= 1
    return rank.astype(np.int64)


def heldout_rank(heldout_index: int) -> float:
    keep = np.arange(len(AGES)) != heldout_index
    return float(
        np.interp(
            PHYSICAL_TIMES[heldout_index],
            PHYSICAL_TIMES[keep],
            np.arange(len(AGES) - 1, dtype=np.float64),
        )
    )


def write_trajectorynet_loo(
    full: dict[str, np.ndarray], heldout_age: int, output: Path
) -> dict[str, object]:
    heldout_index = int(np.flatnonzero(AGES == heldout_age)[0])
    full_age = np.asarray(full["age_day"], dtype=np.int64)
    full_rank = np.asarray(full["sample_labels"], dtype=np.int64)
    keep = full_age != heldout_age
    retained_ages = AGES[AGES != heldout_age]
    retained_times = PHYSICAL_TIMES[AGES != heldout_age]
    labels = retained_rank(full_rank, heldout_index)
    fractional_rank = heldout_rank(heldout_index)

    payload = {
        "pca": np.asarray(full["pca"], dtype=np.float32)[keep],
        "pca_normalized": np.asarray(full["pca"], dtype=np.float32)[keep],
        "pca_raw": np.asarray(full["pca_raw"], dtype=np.float32)[keep],
        "sample_labels": labels,
        "training_rank": labels,
        "full_chronological_rank": full_rank[keep],
        "time_point_processed": np.asarray(
            full["time_point_processed"], dtype=np.float32
        )[keep],
        "age_day": full_age[keep],
        "time_key": np.asarray(full["time_key"])[keep],
        "cell_ids": np.asarray(full["cell_ids"])[keep],
        "retained_ages": retained_ages,
        "retained_physical_times": retained_times.astype(np.float32),
        "retained_training_ranks": np.arange(len(retained_ages), dtype=np.int64),
        "heldout_age": np.asarray(heldout_age, dtype=np.int64),
        "heldout_full_rank": np.asarray(heldout_index, dtype=np.int64),
        "heldout_physical_time": np.asarray(
            PHYSICAL_TIMES[heldout_index], dtype=np.float32
        ),
        "heldout_interpolated_training_rank": np.asarray(
            fractional_rank, dtype=np.float32
        ),
        "heldout_trajectorynet_cnf_time": np.asarray(
            (fractional_rank + 1.0) * TRAJECTORYNET_TIME_SCALE,
            dtype=np.float32,
        ),
        "raw_to_model_scale": np.asarray(RNA_SCALE, dtype=np.float64),
        "trajectorynet_time_scale": np.asarray(
            TRAJECTORYNET_TIME_SCALE, dtype=np.float32
        ),
        "strict_loo": np.asarray(True),
        "space_protocol": np.asarray(
            "exact subset of Full seven-time normalized RNA PCA30; no refit"
        ),
        "time_protocol": np.asarray(
            "six retained snapshots remapped to consecutive forward ranks 0..5; "
            "held-out rank obtained by physical-time interpolation"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(output)
    return {
        "rows": int(keep.sum()),
        "retained_ages": retained_ages.tolist(),
        "retained_physical_times": retained_times.tolist(),
        "heldout_interpolated_training_rank": fractional_rank,
        "heldout_trajectorynet_cnf_time": (
            fractional_rank + 1.0
        ) * TRAJECTORYNET_TIME_SCALE,
        "output": str(output.resolve()),
    }


def write_mioflow_loo(
    full: ad.AnnData, full_path: Path, heldout_age: int, output: Path
) -> dict[str, object]:
    heldout_index = int(np.flatnonzero(AGES == heldout_age)[0])
    keep = full.obs["age_day"].to_numpy(dtype=np.int64) != heldout_age
    result = full[keep].copy()
    retained_ages = AGES[AGES != heldout_age]
    retained_times = PHYSICAL_TIMES[AGES != heldout_age]
    fractional_rank = heldout_rank(heldout_index)
    result.uns["task"] = f"strict_loo_D{heldout_age}_fixed_full_gaga10"
    result.uns["strict_loo"] = True
    result.uns["loo_heldout_age"] = int(heldout_age)
    result.uns["loo_heldout_time"] = float(PHYSICAL_TIMES[heldout_index])
    result.uns["loo_train_ages"] = retained_ages
    result.uns["loo_train_physical_times"] = retained_times.astype(np.float32)
    result.uns["mioflow_training_ranks"] = np.arange(
        len(retained_ages), dtype=np.int64
    )
    result.uns["loo_heldout_interpolated_training_rank"] = fractional_rank
    result.uns["representation_protocol"] = (
        "reuse Full GAGA10 encoder, decoder, PCA input scaler, and Full GAGA10 "
        "feature-wise z-score; only the MIOFlow vector field is refitted"
    )
    result.uns["source_space"] = "exact subset of Full raw RNA PCA30"
    result.uns["rna_raw_to_normalized_scale"] = RNA_SCALE
    result.uns["source_full_h5ad"] = str(full_path.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.h5ad")
    result.write_h5ad(temporary, compression="gzip")
    temporary.replace(output)
    return {
        "rows": int(result.n_obs),
        "retained_ages": retained_ages.tolist(),
        "retained_physical_times": retained_times.tolist(),
        "heldout_interpolated_training_rank": fractional_rank,
        "output": str(output.resolve()),
    }


def main() -> None:
    args = parse_args()
    ages = list(dict.fromkeys(args.ages))
    invalid = [age for age in ages if age not in HELD_OUT_AGES]
    if invalid:
        raise ValueError(
            f"Only internal ages {list(HELD_OUT_AGES)} are supported; got {invalid}"
        )
    require(args.trajectorynet_full)
    require(args.mioflow_full)
    tn_full = read_trajectorynet_full(args.trajectorynet_full)
    mio_full = ad.read_h5ad(args.mioflow_full)
    validate_mioflow_full(mio_full)
    if not np.array_equal(
        np.asarray(tn_full["cell_ids"]).astype(str),
        mio_full.obs_names.astype(str).to_numpy(),
    ):
        raise ValueError("Full TrajectoryNet and MIOFlow cell order differs")
    if not np.array_equal(
        np.asarray(tn_full["pca"], dtype=np.float32),
        np.asarray(mio_full.obsm["X_pca_normalized"], dtype=np.float32),
    ):
        raise ValueError("Full TrajectoryNet and MIOFlow normalized PCA30 differs")

    data_dir = args.data_dir.resolve()
    suite_path = data_dir / f"{PREFIX}_rna_trajectorynet_mioflow_loo_suite.json"
    outputs: list[Path] = [suite_path]
    for age in ages:
        outputs.extend(
            [
                data_dir / f"{PREFIX}_rna_trajectorynet_strict_loo_D{age}_forward.npz",
                data_dir / f"{PREFIX}_rna_mioflow_strict_loo_D{age}.h5ad",
            ]
        )
    protect(outputs, args.overwrite)

    suite: dict[str, object] = {
        "dataset": "human_cerebral",
        "strict_loo": True,
        "full_ages": AGES.tolist(),
        "full_physical_times": PHYSICAL_TIMES.tolist(),
        "held_out_ages": ages,
        "rna_scale": RNA_SCALE,
        "trajectorynet_space": "fixed Full seven-time normalized RNA PCA30",
        "mioflow_space": (
            "fixed Full GAGA10 plus fixed Full GAGA-latent feature-wise z-score"
        ),
        "settings": {},
    }
    for age in ages:
        tn_path = data_dir / f"{PREFIX}_rna_trajectorynet_strict_loo_D{age}_forward.npz"
        mio_path = data_dir / f"{PREFIX}_rna_mioflow_strict_loo_D{age}.h5ad"
        tn_info = write_trajectorynet_loo(tn_full, age, tn_path)
        mio_info = write_mioflow_loo(mio_full, args.mioflow_full, age, mio_path)
        suite["settings"][f"D{age}"] = {
            "trajectorynet": tn_info,
            "mioflow": mio_info,
        }
        print(
            f"D{age}: retained={tn_info['rows']} cells; "
            f"held-out rank TN/MIOFlow={tn_info['heldout_interpolated_training_rank']:.6g}"
        )
    suite_path.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {suite_path}")


if __name__ == "__main__":
    main()
