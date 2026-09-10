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
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

import numpy as np
import pandas as pd
import torch
from geomloss import SamplesLoss
from sklearn.neighbors import NearestNeighbors

from evaluate_gastrulation_full_cmcc import (
    STAGE_NAMES,
    UNKNOWN,
    _apply_t,
    _load_references,
    _load_t_model,
)
from evaluate_gastrulation_loo_time1_paired_1nn_atac import (
    HELDOUT_STAGE_IDX,
    HELDOUT_TIME,
    _build_predictions,
    _load_separate_modality_labels,
)
from evaluate_gastrulation_paired_knn_atac import (
    _composition,
    _js_divergence_base2,
    _microcluster_distribution,
    _normalized_particle_weights,
    _weighted_median,
    _weighted_support_sinkhorn,
)


ROOT = Path(__file__).resolve().parents[2]
TRAINF_ROOT = Path("external/COATI")
T_MODEL_FILE = TRAINF_ROOT / "Gastrulation/data/TrainT/train_FiLM_MLP.py"
T_CHECKPOINT = TRAINF_ROOT / "Gastrulation/data/TrainT/T_FiLM.pt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_reference(points: np.ndarray, sinkhorn_clusters: int) -> dict:
    points = np.asarray(points, dtype=np.float32)
    one_nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(points)
    leave_one_out = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(points)
    loo_distances = leave_one_out.kneighbors(points, return_distance=True)[0][:, 1]
    uniform = np.full(points.shape[0], 1.0 / points.shape[0], dtype=np.float64)
    weights, centers, inertia = _microcluster_distribution(
        points, uniform, sinkhorn_clusters
    )
    return {
        "neighbors": one_nn,
        "reference_loo_1nn_median": float(np.median(loo_distances)),
        "cluster_weights": weights,
        "cluster_centers": centers,
        "kmeans_inertia": inertia,
    }


def _evaluate_distribution(
    method: str,
    modality: str,
    points: np.ndarray,
    reference: np.ndarray,
    labels: np.ndarray,
    prepared: dict,
    classes: list[str],
    sinkhorn: SamplesLoss,
    query_log_mass: np.ndarray | None,
    sinkhorn_clusters: int,
    stage_name: str | None = None,
) -> tuple[dict, list[dict]]:
    points = np.asarray(points, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    query_weights, total_mass, effective_particles, max_particle_weight = (
        _normalized_particle_weights(points.shape[0], query_log_mass)
    )
    distances, indices = prepared["neighbors"].kneighbors(points, return_distance=True)
    distances = distances[:, 0]
    indices = indices[:, 0]
    transferred = np.zeros(reference.shape[0], dtype=np.float64)
    np.add.at(transferred, indices, query_weights)
    transferred /= transferred.sum()

    observed = np.full(reference.shape[0], 1.0 / reference.shape[0], dtype=np.float64)
    predicted_composition = _composition(transferred, labels, classes)
    observed_composition = _composition(observed, labels, classes)
    known = np.asarray([i for i, name in enumerate(classes) if name != UNKNOWN], dtype=int)
    composition_jsd = _js_divergence_base2(
        predicted_composition[known], observed_composition[known]
    )

    effective_cells = float(1.0 / np.sum(np.square(transferred)))
    coverage = effective_cells / reference.shape[0]
    projection_median = _weighted_median(distances, query_weights)
    projection_ratio = projection_median / prepared["reference_loo_1nn_median"]

    uniform_query = np.full(points.shape[0], 1.0 / points.shape[0], dtype=np.float64)
    if np.array_equal(points, reference) and np.allclose(
        query_weights, uniform_query, rtol=0.0, atol=1e-12
    ):
        query_cluster_weights = prepared["cluster_weights"]
        query_cluster_centers = prepared["cluster_centers"]
    else:
        query_cluster_weights, query_cluster_centers, _ = _microcluster_distribution(
            points, query_weights, sinkhorn_clusters
        )
    sinkhorn_divergence = _weighted_support_sinkhorn(
        query_cluster_weights,
        query_cluster_centers,
        prepared["cluster_weights"],
        prepared["cluster_centers"],
        sinkhorn,
    )

    prefix = modality.lower()
    unknown_idx = classes.index(UNKNOWN)
    metrics = {
        f"{prefix}_celltype_composition_jsd": composition_jsd,
        f"{prefix}_cjs_score": float(1.0 - composition_jsd),
        f"{prefix}_effective_coverage": coverage,
        f"{prefix}_effective_reference_cells": effective_cells,
        f"{prefix}_sinkhorn_divergence": sinkhorn_divergence,
        f"{prefix}_projection_1nn_median": projection_median,
        f"{prefix}_projection_distance_ratio": projection_ratio,
        f"{prefix}_predicted_unknown_mass": float(predicted_composition[unknown_idx]),
        f"{prefix}_observed_unknown_fraction": float(observed_composition[unknown_idx]),
        f"{prefix}_max_assigned_cell_mass": float(transferred.max()),
        f"{prefix}_query_total_mass": total_mass,
        f"{prefix}_query_effective_particle_count": effective_particles,
        f"{prefix}_query_max_normalized_particle_mass": max_particle_weight,
    }
    rows = [
        {
            "method": method,
            "stage": stage_name or STAGE_NAMES[HELDOUT_STAGE_IDX],
            "modality": modality.upper(),
            "celltype": celltype,
            "predicted_fraction": float(predicted_composition[class_idx]),
            "observed_fraction": float(observed_composition[class_idx]),
            "difference": float(
                predicted_composition[class_idx] - observed_composition[class_idx]
            ),
            "absolute_difference": float(
                abs(predicted_composition[class_idx] - observed_composition[class_idx])
            ),
        }
        for class_idx, celltype in enumerate(classes)
    ]
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Gastrulation E8.0 LOO RNA predictions after applying one shared "
            "frozen time-dependent RNA-to-ATAC map T."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/gastrulation_loo_time1_shared_t_atac"),
    )
    parser.add_argument("--sinkhorn-clusters", type=int, default=512)
    parser.add_argument("--sinkhorn-blur", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} is not empty; pass --overwrite to replace its files"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    rna_ref, atac_ref, shared_labels_by_stage, rna_scale, atac_scale = _load_references()
    rna_labels_by_stage, atac_labels_by_stage, label_audit = (
        _load_separate_modality_labels(shared_labels_by_stage)
    )
    x0 = rna_ref[0]
    heldout_rna = rna_ref[HELDOUT_STAGE_IDX]
    heldout_atac = atac_ref[HELDOUT_STAGE_IDX]
    heldout_rna_labels = rna_labels_by_stage[HELDOUT_STAGE_IDX]
    heldout_atac_labels = atac_labels_by_stage[HELDOUT_STAGE_IDX]
    classes = sorted(
        set(np.concatenate(rna_labels_by_stage).tolist())
        | set(np.concatenate(atac_labels_by_stage).tolist())
        | {UNKNOWN}
    )

    predictions, log_masses, provenance = _build_predictions(
        x0, rna_scale, device, args.batch_size, args.steps
    )
    t_model = _load_t_model(device)
    rna_prepared = _prepare_reference(heldout_rna, args.sinkhorn_clusters)
    atac_prepared = _prepare_reference(heldout_atac, args.sinkhorn_clusters)
    sinkhorn = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=args.sinkhorn_blur,
        debias=True,
        backend="tensorized",
    )

    summaries: list[dict] = []
    composition_rows: list[dict] = []
    for method, rna_prediction in predictions.items():
        native_mass = log_masses.get(method)
        atac_prediction = _apply_t(
            t_model, rna_prediction, HELDOUT_TIME, device, args.batch_size
        )
        rna_metrics, rna_rows = _evaluate_distribution(
            method,
            "RNA",
            rna_prediction,
            heldout_rna,
            heldout_rna_labels,
            rna_prepared,
            classes,
            sinkhorn,
            native_mass,
            args.sinkhorn_clusters,
        )
        atac_metrics, atac_rows = _evaluate_distribution(
            method,
            "ATAC",
            atac_prediction,
            heldout_atac,
            heldout_atac_labels,
            atac_prepared,
            classes,
            sinkhorn,
            native_mass,
            args.sinkhorn_clusters,
        )
        summary = {
            "method": method,
            "stage": STAGE_NAMES[HELDOUT_STAGE_IDX],
            "physical_time": HELDOUT_TIME,
            "query_weighting": "native_mass" if native_mass is not None else "uniform",
            **rna_metrics,
            **atac_metrics,
        }
        summaries.append(summary)
        composition_rows.extend(rna_rows)
        composition_rows.extend(atac_rows)
        print(
            f"[shared T, E8.0] {method}: "
            f"RNA-JSD={summary['rna_celltype_composition_jsd']:.6f} "
            f"ATAC-JSD={summary['atac_celltype_composition_jsd']:.6f} "
            f"ATAC-coverage={summary['atac_effective_coverage']:.6f} "
            f"ATAC-Sinkhorn={summary['atac_sinkhorn_divergence']:.6f}",
            flush=True,
        )

    # This isolates the mapping bottleneck of T itself at E8.0. It is a
    # diagnostic reference, not a mathematical upper/lower bound.
    real_rna_through_t = _apply_t(
        t_model, heldout_rna, HELDOUT_TIME, device, args.batch_size
    )
    t_diagnostic, t_rows = _evaluate_distribution(
        "T(real E8.0 RNA) diagnostic",
        "ATAC",
        real_rna_through_t,
        heldout_atac,
        heldout_atac_labels,
        atac_prepared,
        classes,
        sinkhorn,
        None,
        args.sinkhorn_clusters,
    )
    composition_rows.extend(t_rows)

    scores = pd.DataFrame(summaries).sort_values(
        [
            "atac_celltype_composition_jsd",
            "atac_sinkhorn_divergence",
            "atac_effective_coverage",
        ],
        ascending=[True, True, False],
        ignore_index=True,
    )
    scores.to_csv(args.output_dir / "loo_time1_shared_t_scores.csv", index=False)
    pd.DataFrame(composition_rows).to_csv(
        args.output_dir / "loo_time1_shared_t_celltype_composition.csv", index=False
    )
    pd.DataFrame([{"method": "T(real E8.0 RNA) diagnostic", **t_diagnostic}]).to_csv(
        args.output_dir / "shared_t_real_rna_diagnostic.csv", index=False
    )

    manifest = {
        "task": "Gastrulation LOO time1/E8.0 through one shared frozen T",
        "stage": STAGE_NAMES[HELDOUT_STAGE_IDX],
        "physical_time_given_to_T_for_every_method": HELDOUT_TIME,
        "training_physical_times": [0.0, 2.0, 2.5],
        "n_initial": int(x0.shape[0]),
        "n_heldout_rna": int(heldout_rna.shape[0]),
        "n_heldout_atac": int(heldout_atac.shape[0]),
        "shared_T": {
            "model_file": str(T_MODEL_FILE),
            "checkpoint": str(T_CHECKPOINT),
            "model_file_sha256": _sha256(T_MODEL_FILE),
            "checkpoint_sha256": _sha256(T_CHECKPOINT),
            "input_space": "normalized RNA PCA 50D",
            "output_space": "normalized ATAC LSI 14D",
            "time": "physical time 1.0 for every method",
        },
        "protocol": {
            "rna_celltype_jsd": (
                "Base-2 JSD between real E8.0 RNA cell-type composition and the "
                "composition induced by independent 1NN assignment in real RNA space."
            ),
            "atac_celltype_jsd": (
                "Each method's predicted RNA cells are first mapped by the same frozen "
                "T(x, physical_time=1.0). Base-2 JSD then compares the cell-type composition "
                "induced by independent 1NN assignment in real E8.0 ATAC space with the "
                "real ATAC composition. Unknown is excluded and known types renormalized."
            ),
            "atac_effective_coverage": (
                "After ATAC-space 1NN assignment, (1/sum_i q_i^2)/n_real_ATAC."
            ),
            "atac_sinkhorn": (
                "Debiased p=2 weighted Sinkhorn divergence directly between T outputs "
                "and real E8.0 ATAC, using independently constructed weighted "
                "microcluster supports."
            ),
            "sinkhorn_clusters": args.sinkhorn_clusters,
            "sinkhorn_blur": args.sinkhorn_blur,
            "mass": (
                "USOT, CytoBridge unbalanced, and TIGON use normalized native particle "
                "mass for RNA and ATAC metrics; balanced methods and MIOFlow use uniform mass."
            ),
            "mioflow_clock": (
                "MIOFlow predicts E8.0 at internal time 0.5 after ordinal factorization "
                "of observed stages; its prediction is passed to T at physical time 1.0."
            ),
            "trajectorynet": (
                "The existing labels [0,2,2.5] were paired by TrajectoryNet's upstream "
                "reverse-zip implementation with solver endpoints [1.0,1.5,2.0]. The "
                "held-out physical midpoint E8.0 was evaluated at solver time 1.25; its "
                "RNA prediction was passed to T at physical time 1.0."
            ),
            "t_diagnostic": (
                "T(real E8.0 RNA) is reported separately as a mapping bottleneck diagnostic "
                "and is never included in the method ranking."
            ),
        },
        "modality_label_audit": label_audit,
        "scales": {"rna": rna_scale, "atac": atac_scale},
        "methods": provenance,
    }
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\n[ranking by ATAC cell-type JSD after shared T]", flush=True)
    print(
        scores[
            [
                "method",
                "rna_celltype_composition_jsd",
                "atac_celltype_composition_jsd",
                "atac_effective_coverage",
                "atac_sinkhorn_divergence",
            ]
        ].round(6).to_string(index=False),
        flush=True,
    )
    print("\n[T(real held-out RNA) bottleneck diagnostic]", flush=True)
    print(pd.Series(t_diagnostic).round(6).to_string(), flush=True)
    print(f"\n[write] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
