#!/usr/bin/env python3
"""Audit fixed-space TrajectoryNet and Full-GAGA10 MIOFlow LOO inputs."""

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
from pathlib import Path

import anndata as ad
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
PREFIX = "human_cerebral_7time_d4_d21_no_d16"
AGES = np.asarray((4, 7, 9, 11, 12, 18, 21), dtype=np.int64)
PHYSICAL_TIMES = np.asarray((0.0, 0.3, 0.5, 0.7, 0.8, 1.4, 1.7), dtype=np.float64)
COUNTS = np.asarray((501, 1309, 1630, 1560, 4100, 5842, 5721), dtype=np.int64)
HELD_OUT_AGES = (7, 9, 11, 12, 18)
RNA_SCALE = 63.89521587795941
FULL_TN = ROOT / "data" / f"{PREFIX}_rna_trajectorynet_forward.npz"
FULL_MIO = ROOT / "data" / f"{PREFIX}_rna_mioflow.h5ad"
FULL_MIO_RESULT = (
    ROOT
    / "results"
    / "mioflow_human_cerebral_7time_d4_d21_no_d16_full_official_gaga10_n256_30000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--age", default="all", help="D7, 7, or all")
    parser.add_argument(
        "--method", choices=("trajectorynet", "mioflow", "all"), default="all"
    )
    parser.add_argument("--trajectorynet-full", type=Path, default=FULL_TN)
    parser.add_argument("--mioflow-full", type=Path, default=FULL_MIO)
    parser.add_argument(
        "--full-gaga-checkpoint",
        type=Path,
        default=FULL_MIO_RESULT / "gaga_model.pt",
    )
    parser.add_argument(
        "--full-mioflow-checkpoint",
        type=Path,
        default=FULL_MIO_RESULT / "model.pt",
    )
    return parser.parse_args()


def parse_ages(value: str) -> list[int]:
    if value.lower() == "all":
        return list(HELD_OUT_AGES)
    normalized = value.lower().removeprefix("d")
    if not normalized.isdigit() or int(normalized) not in HELD_OUT_AGES:
        raise ValueError(f"Age must be one of {HELD_OUT_AGES} or all; got {value}")
    return [int(normalized)]


def require(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_full_tn(path: Path) -> dict[str, np.ndarray]:
    require(path)
    with np.load(path, allow_pickle=True) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def expected_mask(full_ages: np.ndarray, heldout_age: int) -> np.ndarray:
    return full_ages != heldout_age


def expected_heldout_rank(heldout_age: int) -> float:
    index = int(np.flatnonzero(AGES == heldout_age)[0])
    keep = AGES != heldout_age
    return float(
        np.interp(
            PHYSICAL_TIMES[index],
            PHYSICAL_TIMES[keep],
            np.arange(len(AGES) - 1, dtype=np.float64),
        )
    )


def audit_trajectorynet(
    path: Path, full: dict[str, np.ndarray], heldout_age: int
) -> dict[str, object]:
    require(path)
    with np.load(path, allow_pickle=True) as archive:
        loo = {name: np.asarray(archive[name]) for name in archive.files}
    full_ages = np.asarray(full["age_day"], dtype=np.int64)
    keep = expected_mask(full_ages, heldout_age)
    expected_rows = int(keep.sum())
    for key in ("pca", "pca_raw", "sample_labels", "cell_ids", "age_day"):
        if key not in loo:
            raise KeyError(f"{path}: missing {key}")
    pca = np.asarray(loo["pca"], dtype=np.float32)
    raw = np.asarray(loo["pca_raw"], dtype=np.float32)
    labels = np.asarray(loo["sample_labels"], dtype=np.int64)
    if pca.shape != (expected_rows, 30) or raw.shape != pca.shape:
        raise ValueError(f"D{heldout_age} TrajectoryNet shape changed: {pca.shape}")
    if not np.array_equal(pca, np.asarray(full["pca"], dtype=np.float32)[keep]):
        raise ValueError(f"D{heldout_age} TrajectoryNet PCA is not an exact Full subset")
    if not np.array_equal(raw, np.asarray(full["pca_raw"], dtype=np.float32)[keep]):
        raise ValueError(f"D{heldout_age} TrajectoryNet raw PCA changed")
    if not np.array_equal(
        np.asarray(loo["cell_ids"]).astype(str),
        np.asarray(full["cell_ids"])[keep].astype(str),
    ):
        raise ValueError(f"D{heldout_age} TrajectoryNet cell identity/order changed")
    if heldout_age in np.asarray(loo["age_day"], dtype=np.int64):
        raise ValueError(f"D{heldout_age} leaked into TrajectoryNet training")
    if not np.array_equal(np.unique(labels), np.arange(6)):
        raise ValueError(f"D{heldout_age} TrajectoryNet ranks are not consecutive 0..5")
    retained_ages = AGES[AGES != heldout_age]
    expected_counts = COUNTS[AGES != heldout_age]
    observed_counts = tuple(
        int(np.sum(np.asarray(loo["age_day"], dtype=np.int64) == age))
        for age in retained_ages
    )
    if observed_counts != tuple(expected_counts):
        raise ValueError(f"D{heldout_age} TrajectoryNet retained counts changed")
    heldout_model_rank = float(loo["heldout_interpolated_training_rank"])
    if not np.isclose(heldout_model_rank, expected_heldout_rank(heldout_age)):
        raise ValueError(f"D{heldout_age} TrajectoryNet held-out rank changed")
    if not np.allclose(pca, raw / RNA_SCALE, atol=2e-6, rtol=2e-6):
        raise ValueError(f"D{heldout_age} TrajectoryNet PCA scale changed")
    return {
        "rows": expected_rows,
        "training_ranks": list(range(6)),
        "heldout_model_rank": heldout_model_rank,
    }


def audit_mioflow(
    path: Path, full: ad.AnnData, heldout_age: int
) -> dict[str, object]:
    require(path)
    loo = ad.read_h5ad(path)
    full_ages = full.obs["age_day"].to_numpy(dtype=np.int64)
    keep = expected_mask(full_ages, heldout_age)
    if loo.n_obs != int(keep.sum()) or loo.n_vars != 30:
        raise ValueError(f"D{heldout_age} MIOFlow shape changed: {loo.shape}")
    if not np.array_equal(
        loo.obs_names.astype(str).to_numpy(), full.obs_names[keep].astype(str).to_numpy()
    ):
        raise ValueError(f"D{heldout_age} MIOFlow cell identity/order changed")
    if heldout_age in loo.obs["age_day"].to_numpy(dtype=np.int64):
        raise ValueError(f"D{heldout_age} leaked into MIOFlow vector-field input")
    for key in ("X_latent", "X_pca", "X_pca_normalized"):
        actual = np.asarray(loo.obsm[key], dtype=np.float32)
        expected = np.asarray(full.obsm[key], dtype=np.float32)[keep]
        if not np.array_equal(actual, expected):
            raise ValueError(f"D{heldout_age} MIOFlow {key} is not an exact Full subset")
    raw = np.asarray(loo.obsm["X_latent"], dtype=np.float32)
    normalized = np.asarray(loo.obsm["X_pca_normalized"], dtype=np.float32)
    if not np.allclose(normalized, raw / RNA_SCALE, atol=2e-6, rtol=2e-6):
        raise ValueError(f"D{heldout_age} MIOFlow source normalization changed")
    if not bool(loo.uns["strict_loo"]):
        raise ValueError(f"D{heldout_age} MIOFlow strict_loo flag missing")
    if int(loo.uns["loo_heldout_age"]) != heldout_age:
        raise ValueError(f"D{heldout_age} MIOFlow held-out metadata changed")
    heldout_model_rank = float(loo.uns["loo_heldout_interpolated_training_rank"])
    if not np.isclose(heldout_model_rank, expected_heldout_rank(heldout_age)):
        raise ValueError(f"D{heldout_age} MIOFlow held-out rank changed")
    return {
        "rows": int(loo.n_obs),
        "training_ranks": list(range(6)),
        "heldout_model_rank": heldout_model_rank,
    }


def audit_full_gaga(gaga_path: Path, model_path: Path, full_n_obs: int) -> dict[str, str]:
    require(gaga_path)
    require(model_path)
    gaga = torch.load(gaga_path, map_location="cpu", weights_only=False)
    required = {
        "state_dict",
        "input_dim",
        "latent_dim",
        "hidden_dims",
        "input_scaler_mean",
        "input_scaler_scale",
        "fit_indices",
        "source_checkpoint",
    }
    missing = required.difference(gaga)
    if missing:
        raise ValueError(f"Full GAGA checkpoint missing keys: {sorted(missing)}")
    if int(gaga["input_dim"]) != 30 or int(gaga["latent_dim"]) != 10:
        raise ValueError("Full GAGA architecture must be PCA30 -> GAGA10")
    if [int(value) for value in gaga["hidden_dims"]] != [128, 64]:
        raise ValueError("Full GAGA hidden dimensions changed")
    if gaga["source_checkpoint"] is not None:
        raise ValueError("Expected a GAGA model fitted by the Full job")
    if int(gaga["input_scaler_n_samples_seen"]) != full_n_obs:
        raise ValueError("Full GAGA input scaler was not fitted on all Full cells")
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    if not bool(model.get("use_gaga", False)) or int(model.get("input_dim", -1)) != 10:
        raise ValueError("Full MIOFlow checkpoint is not a GAGA10 model")
    if str(model.get("embedding_normalization", "")).replace("-", "_") != "zscore":
        raise ValueError("Full MIOFlow checkpoint does not use latent z-score")
    mean = np.asarray(model["mean_vals"], dtype=np.float64).reshape(-1)
    std = np.asarray(model["std_vals"], dtype=np.float64).reshape(-1)
    if mean.shape != (10,) or std.shape != (10,) or np.any(std <= 0):
        raise ValueError("Full MIOFlow GAGA10 z-score is invalid")
    return {"gaga_sha256": sha256(gaga_path), "model_sha256": sha256(model_path)}


def main() -> None:
    args = parse_args()
    ages = parse_ages(args.age)
    methods = (
        ("trajectorynet", "mioflow") if args.method == "all" else (args.method,)
    )
    full_tn = load_full_tn(args.trajectorynet_full)
    require(args.mioflow_full)
    full_mio = ad.read_h5ad(args.mioflow_full)
    checkpoint_audit = None
    if "mioflow" in methods:
        checkpoint_audit = audit_full_gaga(
            args.full_gaga_checkpoint, args.full_mioflow_checkpoint, full_mio.n_obs
        )

    for age in ages:
        summaries: list[str] = []
        if "trajectorynet" in methods:
            path = (
                args.data_dir
                / f"{PREFIX}_rna_trajectorynet_strict_loo_D{age}_forward.npz"
            )
            audit = audit_trajectorynet(path, full_tn, age)
            summaries.append(
                f"TrajectoryNet rows={audit['rows']} ranks=0..5 "
                f"heldout_rank={audit['heldout_model_rank']:.6g}"
            )
        if "mioflow" in methods:
            path = args.data_dir / f"{PREFIX}_rna_mioflow_strict_loo_D{age}.h5ad"
            audit = audit_mioflow(path, full_mio, age)
            summaries.append(
                f"MIOFlow rows={audit['rows']} ranks=0..5 "
                f"heldout_rank={audit['heldout_model_rank']:.6g}"
            )
        print(f"[OK] D{age}: " + " | ".join(summaries))
    if checkpoint_audit is not None:
        print("[OK] frozen Full GAGA10:", args.full_gaga_checkpoint)
        print("[OK] frozen Full GAGA-latent z-score:", args.full_mioflow_checkpoint)
        print("  GAGA sha256:", checkpoint_audit["gaga_sha256"])
        print("  model sha256:", checkpoint_audit["model_sha256"])


if __name__ == "__main__":
    main()
