#!/usr/bin/env python
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
import pandas as pd
import torch


TIME_LABELS = ("time1", "time2", "time3", "time4")
MODEL_TIMES = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
DEFAULT_SYNTHETIC_DIR = Path(
    "external/COATI/Synthetic/5scRNA"
)


def load_scale(path: Path) -> float:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "scale" not in checkpoint:
        raise ValueError(f"Expected a dictionary with a scalar 'scale' in {path}")
    scale = float(checkpoint["scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid normalization scale {scale!r} in {path}")
    return scale


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare normalized synthetic RNA10 AnnData for CytoBridge."
    )
    parser.add_argument(
        "--rna-data",
        type=Path,
        default=DEFAULT_SYNTHETIC_DIR / "all_time_scRNA_pca10.npz",
    )
    parser.add_argument(
        "--atac-data",
        type=Path,
        default=DEFAULT_SYNTHETIC_DIR / "all_time_scATAC_pca8_no_pc2.npz",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_SYNTHETIC_DIR / "all_time_scRNA_label.npz",
    )
    parser.add_argument(
        "--rna-norm",
        type=Path,
        default=DEFAULT_SYNTHETIC_DIR / "primal_norm_params_rna10_w2.pt",
    )
    parser.add_argument(
        "--atac-norm",
        type=Path,
        default=DEFAULT_SYNTHETIC_DIR / "secondary_norm_params_atac8_no_pc2_w2.pt",
    )
    parser.add_argument(
        "--film-checkpoint",
        type=Path,
        default=Path(
            "external/COATI/Synthetic/TrainT/outputs/"
            "T_FiLM_rna10_atac8_no_pc2.pt"
        ),
    )
    parser.add_argument(
        "--output-h5ad",
        type=Path,
        default=Path("data/synthetic_rna10_cytobridge.h5ad"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/synthetic_rna10_cytobridge_manifest.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in (
        args.rna_data,
        args.atac_data,
        args.labels,
        args.rna_norm,
        args.atac_norm,
        args.film_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    existing = [path for path in (args.output_h5ad, args.manifest) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite prepared inputs: " + ", ".join(map(str, existing))
        )

    rna_archive = np.load(args.rna_data)
    atac_archive = np.load(args.atac_data)
    label_archive = np.load(args.labels, allow_pickle=True)
    for label in TIME_LABELS:
        for name, archive in (
            ("RNA", rna_archive),
            ("ATAC", atac_archive),
            ("labels", label_archive),
        ):
            if label not in archive.files:
                raise KeyError(f"{label!r} missing from {name} archive")

    rna_blocks = [np.asarray(rna_archive[label], dtype=np.float32) for label in TIME_LABELS]
    atac_blocks = [np.asarray(atac_archive[label], dtype=np.float32) for label in TIME_LABELS]
    population_blocks = [label_archive[label].astype(str) for label in TIME_LABELS]
    if any(block.ndim != 2 or block.shape[1] != 10 for block in rna_blocks):
        raise ValueError("Every RNA block must have shape (n_cells, 10)")
    if any(block.ndim != 2 or block.shape[1] != 8 for block in atac_blocks):
        raise ValueError("Every selected ATAC block must have shape (n_cells, 8)")
    for label, rna, atac, population in zip(
        TIME_LABELS, rna_blocks, atac_blocks, population_blocks
    ):
        if not (len(rna) == len(atac) == len(population)):
            raise ValueError(
                f"Paired row mismatch at {label}: RNA={len(rna)}, "
                f"ATAC={len(atac)}, labels={len(population)}"
            )

    rna_scale = load_scale(args.rna_norm)
    atac_scale = load_scale(args.atac_norm)
    rna_raw = np.concatenate(rna_blocks).astype(np.float32, copy=False)
    rna_normalized = (rna_raw / rna_scale).astype(np.float32)
    counts = [len(block) for block in rna_blocks]
    time_index = np.concatenate(
        [np.full(count, index, dtype=np.int64) for index, count in enumerate(counts)]
    )
    physical_time = MODEL_TIMES[time_index]
    population = np.concatenate(population_blocks)
    time_label = np.concatenate(
        [np.full(count, label, dtype=object) for label, count in zip(TIME_LABELS, counts)]
    )
    cell_id = np.concatenate(
        [
            np.asarray([f"{label}_{index:05d}" for index in range(count)], dtype=object)
            for label, count in zip(TIME_LABELS, counts)
        ]
    )

    obs = pd.DataFrame(
        {
            "time_label": time_label.astype(str),
            "time_index": time_index,
            "time_point_processed": physical_time,
            "population": population.astype(str),
        },
        index=pd.Index(cell_id.astype(str), name="cell_id"),
    )
    var = pd.DataFrame(index=[f"RNA_PC{index}" for index in range(1, 11)])
    adata = ad.AnnData(X=rna_normalized.copy(), obs=obs, var=var)
    adata.obsm["X_latent"] = rna_normalized.copy()
    adata.obsm["X_pca"] = rna_normalized.copy()
    adata.obsm["X_pca_raw"] = rna_raw.copy()
    adata.uns["trainfbench"] = {
        "dataset": "synthetic_5scRNA_rna10_atac8_no_pc2",
        "rna_data": str(args.rna_data.resolve()),
        "atac_data": str(args.atac_data.resolve()),
        "population_labels": str(args.labels.resolve()),
        "rna_norm": str(args.rna_norm.resolve()),
        "atac_norm": str(args.atac_norm.resolve()),
        "film_checkpoint": str(args.film_checkpoint.resolve()),
        "rna_scale": rna_scale,
        "atac_scale": atac_scale,
        "time_labels": list(TIME_LABELS),
        "model_times": MODEL_TIMES.tolist(),
        "counts": counts,
        "latent_space": "raw RNA10 PCA divided by primal W2 scale",
        "selected_atac_pc_indices_1based": [1, 3, 4, 5, 6, 7, 8, 9],
    }

    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output_h5ad)
    manifest = {
        **adata.uns["trainfbench"],
        "output_h5ad": str(args.output_h5ad.resolve()),
        "shape": list(adata.shape),
        "X_latent_shape": list(adata.obsm["X_latent"].shape),
        "population_counts": {
            label: int(np.sum(population == label)) for label in sorted(np.unique(population))
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
