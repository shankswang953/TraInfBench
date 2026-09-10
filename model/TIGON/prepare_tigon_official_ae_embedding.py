#!/usr/bin/env python
"""Train a TIGON official-style AE embedding for a benchmark task.

The legacy default (``--task full``) preserves the dataset-level frozen cache.
For strict LOO benchmarks, ``--task loo_time1`` or ``loo_time2`` fits the AE,
validation split, and per-axis ``[-2, 2]`` scaling only on retained stages.
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
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from tigon_official_ae_common import (
    OFFICIAL_SOURCE_COMMIT,
    resolve_device,
    set_seed,
    train_official_ae,
)


ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "moscot": {
        "raw_h5ad": Path(
            "external/COATI/moscot/data/"
            "rna_adata.h5ad"
        ),
        "shared_h5ad": ROOT / "data" / "moscot_rna_cytobridge.h5ad",
        "cache_dir": ROOT / "results" / "tigon_official_ae_embeddings" / "moscot_ae10",
        "raw_time_key": "stage_num",
        "raw_stage_key": None,
        "stage_to_time": None,
        "hvg_policy": (
            "top 3000 full-data genes ranked by var['dispersions_norm'] "
            "(Seurat flavor)"
        ),
        "expression_layer": "scran_counts",
        "input_preprocessing": (
            "top-3000 Seurat-dispersion genes from the full dataset; "
            "scran-normalized log-expression layer without per-gene "
            "standardization"
        ),
    },
    "palate": {
        "raw_h5ad": Path(
            "external/COATI/MouseBrain/"
            "DataGen/Mouse/raw/CNC_RNA_HVG.h5ad"
        ),
        "shared_h5ad": ROOT / "data" / "palate_rna_cytobridge.h5ad",
        "cache_dir": ROOT / "results" / "tigon_official_ae_embeddings" / "palate_ae10",
        "raw_time_key": None,
        "raw_stage_key": "stage",
        "stage_to_time": {
            "E12.5": 0.0,
            "E13.5": 1.0,
            "E14.0": 1.5,
            "E14.5": 2.0,
        },
        "hvg_policy": "all variables in precomputed 3000-HVG file",
        "expression_layer": None,
        "input_preprocessing": (
            "3000-HVG log-expression without per-gene standardization"
        ),
    },
    "gastrulation": {
        "raw_h5ad": ROOT / "data" / "gastrulation_rna_full.h5ad",
        "shared_h5ad": ROOT / "data" / "gastrulation_rna_cytobridge.h5ad",
        "cache_dir": (
            ROOT / "results" / "tigon_official_ae_embeddings" / "gastrulation_ae10"
        ),
        "raw_time_key": "processed_time",
        "raw_stage_key": None,
        "stage_to_time": None,
        "hvg_policy": "var['highly_variable'] from the complete processed dataset",
        "expression_layer": None,
        "input_preprocessing": (
            "3000-HVG log-expression without per-gene standardization"
        ),
    },
    "human_cerebral_no_d61": {
        "raw_h5ad": Path(
            "external/COATI/humanCerebral/"
            "Data/humanBrainRNA.h5ad"
        ),
        "shared_h5ad": (
            ROOT / "data" / "human_cerebral_no_d61_rna_cytobridge.h5ad"
        ),
        "cache_dir": (
            ROOT
            / "results"
            / "tigon_official_ae_embeddings"
            / "human_cerebral_no_d61_full_ae10"
        ),
        "hvg_reference_h5ad": Path(
            "external/COATI/humanCerebral/"
            "Data/rna_dimReduced.h5ad"
        ),
        "raw_time_key": "age",
        "raw_stage_key": None,
        "stage_to_time": None,
        "age_to_time": {
            4: 0.0,
            7: 0.3,
            9: 0.5,
            11: 0.7,
            12: 0.8,
            16: 1.2,
            18: 1.4,
            21: 1.7,
            26: 2.2,
            31: 2.7,
        },
        "hvg_policy": (
            "the precomputed 3000-HVG gene set from rna_dimReduced.h5ad, "
            "applied to the original humanBrainRNA.h5ad expression matrix"
        ),
        "expression_layer": None,
        "input_preprocessing": (
            "original humanBrainRNA.h5ad X restricted to the benchmark's "
            "3000 precomputed HVGs; no per-gene standardization"
        ),
    },
}

TASK_HELD_OUT_TIMES = {
    "moscot": {"full": None, "loo_time1": 1.0},
    "palate": {"full": None, "loo_time1": 1.0, "loo_time2": 1.5},
    "gastrulation": {"full": None, "loo_time1": 1.0, "loo_time2": 2.0},
    "human_cerebral_no_d61": {"full": None, "loo_day7": 0.3},
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_full_expression(
    dataset: str,
    raw_h5ad: Path,
    shared_h5ad: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = ad.read_h5ad(
        raw_h5ad,
        backed=(
            "r"
            if dataset in {"moscot", "human_cerebral_no_d61"}
            else None
        ),
    )
    shared = ad.read_h5ad(shared_h5ad, backed="r")
    hvg_reference = None
    try:
        raw_names = pd.Index(raw.obs_names.astype(str))
        shared_names = pd.Index(shared.obs_names.astype(str))
        if not raw_names.is_unique or not shared_names.is_unique:
            raise ValueError("Cell IDs must be unique")

        if dataset == "moscot":
            raw_times_all = pd.to_numeric(
                raw.obs["stage_num"], errors="raise"
            ).to_numpy(np.float32)
            shared_times = pd.to_numeric(
                shared.obs["time_point_processed"], errors="raise"
            ).to_numpy(np.float32)
            time_order = np.unique(shared_times)
            indexer = np.concatenate(
                [
                    np.flatnonzero(np.isclose(raw_times_all, value))
                    for value in time_order
                ]
            )
            if len(indexer) != len(shared_names):
                raise ValueError(
                    "moscot raw-to-benchmark time-block alignment has the "
                    "wrong number of cells"
                )
            raw_pca = np.asarray(raw.obsm["X_pca"][indexer, :50], dtype=np.float32)
            shared_pca = np.asarray(shared.obsm["X_latent"], dtype=np.float32)
            if raw_pca.shape != shared_pca.shape:
                raise ValueError(
                    "moscot raw and shared PCA matrices have different shapes"
                )
            pca_max_abs_difference = float(
                np.max(np.abs(raw_pca - shared_pca))
            )
            if pca_max_abs_difference > 1e-6:
                raise ValueError(
                    "moscot raw rows do not match the benchmark's time-block "
                    f"cell order; PCA max_abs_difference={pca_max_abs_difference:.6g}"
                )
            if not np.allclose(raw_times_all[indexer], shared_times):
                raise ValueError(
                    "moscot raw and shared time labels are not aligned"
                )
            scores = pd.to_numeric(
                raw.var["dispersions_norm"], errors="coerce"
            ).to_numpy(np.float64)
            finite = np.flatnonzero(np.isfinite(scores))
            if len(finite) < 3000:
                raise ValueError(
                    "moscot needs at least 3000 genes with finite "
                    "var['dispersions_norm']"
                )
            ranked = finite[
                np.argsort(-scores[finite], kind="stable")[:3000]
            ]
            gene_indices = np.sort(ranked)
            expression = raw.layers["scran_counts"][indexer][:, gene_indices]
            if sp.issparse(expression):
                expression = expression.toarray()
            expression = np.asarray(expression, dtype=np.float32)
            genes = raw.var_names[gene_indices].astype(str).to_numpy()
            raw_times = raw_times_all[indexer]
        elif dataset == "human_cerebral_no_d61":
            if not shared_names.isin(raw_names).all():
                raise ValueError(
                    "Every human-cerebral benchmark cell must occur in "
                    "humanBrainRNA.h5ad"
                )
            indexer = raw_names.get_indexer(shared_names)
            if np.any(indexer < 0):
                raise AssertionError("Unexpected missing human-cerebral cells")

            raw_ages = pd.to_numeric(
                raw.obs.iloc[indexer]["age"], errors="raise"
            ).to_numpy(np.int64)
            shared_ages = pd.to_numeric(
                shared.obs["age_day"], errors="raise"
            ).to_numpy(np.int64)
            if not np.array_equal(raw_ages, shared_ages):
                raise ValueError(
                    "Original and shared human-cerebral age labels are not aligned"
                )
            age_to_time = DATASETS[dataset]["age_to_time"]
            unknown_ages = sorted(set(raw_ages.tolist()) - set(age_to_time))
            if unknown_ages:
                raise ValueError(
                    f"Unexpected human-cerebral ages: {unknown_ages}"
                )
            raw_times = np.asarray(
                [age_to_time[int(age)] for age in raw_ages], dtype=np.float32
            )
            shared_times = pd.to_numeric(
                shared.obs["time_point_processed"], errors="raise"
            ).to_numpy(np.float32)

            hvg_reference = ad.read_h5ad(
                DATASETS[dataset]["hvg_reference_h5ad"], backed="r"
            )
            hvg_genes = pd.Index(hvg_reference.var_names.astype(str))
            if len(hvg_genes) != 3000 or not hvg_genes.is_unique:
                raise ValueError(
                    "humanCerebral TIGON requires exactly 3000 unique "
                    "precomputed HVGs"
                )
            gene_indices = raw.var_names.get_indexer(hvg_genes)
            if np.any(gene_indices < 0):
                raise ValueError(
                    f"{int(np.sum(gene_indices < 0))} TIGON HVGs are absent "
                    "from humanBrainRNA.h5ad"
                )
            # Sorted columns make backed CSC slicing reliable. Gene order is
            # immaterial to the AE as long as it is frozen with the decoder.
            gene_indices = np.sort(gene_indices)
            genes = raw.var_names[gene_indices].astype(str).to_numpy()

            # Materialize the 30,743 x 3,000 benchmark matrix in small row
            # batches. This avoids loading the complete 34,088 x 33,538 source
            # AnnData or creating a second full dense expression copy.
            expression = np.empty(
                (len(indexer), len(gene_indices)), dtype=np.float32
            )
            for start in range(0, len(indexer), 2048):
                stop = min(start + 2048, len(indexer))
                rows = indexer[start:stop]
                row_order = np.argsort(rows)
                block = raw.X[rows[row_order]][:, gene_indices]
                if sp.issparse(block):
                    block = block.astype(np.float32).toarray()
                else:
                    block = np.asarray(block, dtype=np.float32)
                restored = np.empty_like(block)
                restored[row_order] = block
                expression[start:stop] = restored
        else:
            if set(raw_names) != set(shared_names):
                raise ValueError(
                    "The raw full dataset and shared benchmark full dataset "
                    "must contain exactly the same cells"
                )
            indexer = raw_names.get_indexer(shared_names)
            if np.any(indexer < 0):
                raise AssertionError("Unexpected missing full-data cells")
            if dataset == "gastrulation":
                if "highly_variable" not in raw.var:
                    raise KeyError(
                        "Gastrulation raw data lacks var['highly_variable']"
                    )
                gene_mask = raw.var["highly_variable"].fillna(False).to_numpy(bool)
            else:
                gene_mask = np.ones(raw.n_vars, dtype=bool)
            if int(gene_mask.sum()) != 3000:
                raise ValueError(
                    f"TIGON EMT-style AE expects exactly 3000 HVGs; "
                    f"found {int(gene_mask.sum())}"
                )

            expression = raw.X[indexer][:, gene_mask]
            if sp.issparse(expression):
                expression = expression.toarray()
            expression = np.asarray(expression, dtype=np.float32)
            config = DATASETS[dataset]
            if config["raw_time_key"] is not None:
                raw_times = pd.to_numeric(
                    raw.obs.iloc[indexer][config["raw_time_key"]], errors="raise"
                ).to_numpy(np.float32)
            else:
                stages = raw.obs.iloc[indexer][config["raw_stage_key"]].astype(str)
                unknown = sorted(set(stages) - set(config["stage_to_time"]))
                if unknown:
                    raise ValueError(f"Unknown {dataset} stages: {unknown}")
                raw_times = np.asarray(
                    [config["stage_to_time"][stage] for stage in stages],
                    dtype=np.float32,
                )
            shared_times = pd.to_numeric(
                shared.obs["time_point_processed"], errors="raise"
            ).to_numpy(np.float32)
            genes = raw.var_names[gene_mask].astype(str).to_numpy()

        if expression.shape[1] != 3000:
            raise ValueError(
                f"TIGON EMT-style AE expects exactly 3000 HVGs; "
                f"found {expression.shape[1]}"
            )
        if not np.isfinite(expression).all():
            raise ValueError("Expression matrix contains non-finite values")
        if not np.allclose(raw_times, shared_times):
            raise ValueError("Raw and shared full-data time labels are not aligned")
    finally:
        if hvg_reference is not None:
            hvg_reference.file.close()
        if getattr(raw, "isbacked", False):
            raw.file.close()
        shared.file.close()
    return expression, shared_names.to_numpy(str), shared_times, genes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare one frozen official-style TIGON AE embedding."
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument(
        "--task",
        default="full",
        choices=("full", "loo_time1", "loo_time2", "loo_day7"),
        help=(
            "Fit on all cells for full, or exclude the configured held-out "
            "stage before fitting the AE and latent scaling for strict LOO."
        ),
    )
    parser.add_argument("--raw-h5ad", type=Path)
    parser.add_argument("--shared-h5ad", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--ae-seed", type=int, default=4232)
    parser.add_argument("--ae-max-epochs", type=int, default=500)
    parser.add_argument(
        "--ae-latent-dim",
        type=int,
        default=10,
        help=(
            "Latent dimension in the public 3000-300-L-300-3000 layout. "
            "The upstream benchmark default is 10."
        ),
    )
    parser.add_argument(
        "--device", default="cpu", choices=("auto", "cpu", "mps", "cuda")
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Check full-data alignment, task filtering, and the 3000-HVG "
            "input without training or writing a cache."
        ),
    )
    args = parser.parse_args()

    if args.ae_latent_dim <= 0:
        raise ValueError("--ae-latent-dim must be positive")

    if args.task not in TASK_HELD_OUT_TIMES[args.dataset]:
        valid = sorted(TASK_HELD_OUT_TIMES[args.dataset])
        raise ValueError(
            f"Task {args.task!r} is not configured for {args.dataset}; "
            f"choose one of {valid}"
        )

    defaults = DATASETS[args.dataset]
    raw_h5ad = (args.raw_h5ad or defaults["raw_h5ad"]).resolve()
    shared_h5ad = (args.shared_h5ad or defaults["shared_h5ad"]).resolve()
    default_cache_dir = (
        defaults["cache_dir"]
        if args.task == "full"
        else ROOT
        / "results"
        / "tigon_official_ae_embeddings"
        / f"{args.dataset}_{args.task}_strict_ae10"
    )
    cache_dir = (args.cache_dir or default_cache_dir).resolve()
    ready_path = cache_dir / "embedding_ready.json"
    if (
        not args.validate_only
        and cache_dir.exists()
        and any(cache_dir.iterdir())
        and not args.overwrite
    ):
        raise FileExistsError(
            f"{cache_dir} is not empty; reuse it or pass --overwrite explicitly"
        )
    started = time.time()
    expression, obs_names, times, genes = load_full_expression(
        args.dataset,
        raw_h5ad,
        shared_h5ad,
    )
    held_out_time = TASK_HELD_OUT_TIMES[args.dataset][args.task]
    retained_mask = np.ones(len(times), dtype=bool)
    if held_out_time is not None:
        retained_mask = ~np.isclose(times, held_out_time)
        if not np.any(~retained_mask):
            raise ValueError(
                f"No cells matched held-out time {held_out_time:g} for "
                f"{args.dataset}/{args.task}"
            )
        expression = np.ascontiguousarray(expression[retained_mask])
        obs_names = obs_names[retained_mask]
        times = times[retained_mask]
    time_counts = {
        str(float(value)): int(np.sum(np.isclose(times, value)))
        for value in np.unique(times)
    }
    print(
        f"[data] dataset={args.dataset}, task={args.task}, "
        f"cells={expression.shape[0]}, "
        f"genes={expression.shape[1]}, times={sorted(np.unique(times).tolist())}",
        flush=True,
    )
    if args.validate_only:
        print(
            "[validate-only] alignment, strict task selection, and AE input "
            "checks passed",
            flush=True,
        )
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "obs_names.npy", obs_names)
    np.save(cache_dir / "times_full.npy", times)
    pd.Series(genes, name="gene").to_csv(
        cache_dir / "selected_genes.csv", index=False
    )

    device = resolve_device(args.device)
    set_seed(args.ae_seed)
    _, latent, _, _, reconstruction_mse = train_official_ae(
        expression,
        device=device,
        outdir=cache_dir,
        seed=args.ae_seed,
        max_epochs=args.ae_max_epochs,
        n_latent=args.ae_latent_dim,
    )
    latent_path = cache_dir / "ae_latent_scaled_minus2_2.npy"
    latent_sha256 = file_sha256(latent_path)
    config = {
        "dataset": args.dataset,
        "task": args.task,
        "strict_loo": bool(held_out_time is not None),
        "held_out_time": held_out_time,
        "raw_h5ad": str(raw_h5ad),
        "shared_full_h5ad": str(shared_h5ad),
        "cache_dir": str(cache_dir),
        "cells": int(expression.shape[0]),
        "retained_time_counts": time_counts,
        "genes": int(expression.shape[1]),
        "times": sorted(float(value) for value in np.unique(times)),
        "hvg_policy": defaults["hvg_policy"],
        "expression_layer": defaults["expression_layer"],
        "input_preprocessing": defaults["input_preprocessing"],
        "ae_architecture": (
            f"3000-300-{args.ae_latent_dim}-300-3000"
        ),
        "ae_activation": "relu",
        "ae_batch_norm": True,
        "ae_dropout": 0.2,
        "ae_seed": int(args.ae_seed),
        "ae_training_rng_policy": (
            "public Trainer reset after model and dataloader construction"
        ),
        "ae_batch_size": 128,
        "ae_learning_rate": 1e-3,
        "ae_weight_decay": 1e-4,
        "ae_validation_fraction": 0.1,
        "ae_max_epochs": int(args.ae_max_epochs),
        "ae_early_stopping_tolerance": 1e-2,
        "ae_early_stopping_patience": 30,
        "latent_dimension": int(latent.shape[1]),
        "latent_axis_scaling": (
            f"{args.task} retained-cell per-axis min-max scaling to [-2, 2]"
        ),
        "representation_fit_scope": (
            "complete dataset"
            if held_out_time is None
            else "strict LOO retained stages only"
        ),
        "latent_sha256": latent_sha256,
        "reconstruction_mse": float(reconstruction_mse),
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "elapsed_seconds": time.time() - started,
    }
    (cache_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    ready_path.write_text(
        json.dumps(
            {
                "status": "ready",
                "dataset": args.dataset,
                "task": args.task,
                "strict_loo": bool(held_out_time is not None),
                "held_out_time": held_out_time,
                "latent_sha256": latent_sha256,
                "cells": int(latent.shape[0]),
                "latent_dimension": int(latent.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[done] frozen {args.task} AE embedding ready: {cache_dir}", flush=True
    )


if __name__ == "__main__":
    main()
