#!/usr/bin/env python
"""Recompute and plot revised Gastrulation full-train six-metric benchmarks."""

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
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd
import torch

from evaluate_gastrulation_full_cmcc import (
    STAGE_NAMES,
    _apply_t,
    _load_references,
    _load_t_model,
)
from evaluate_gastrulation_full_train_fit_six_metrics import (
    DEFAULT_EXTERNAL_CACHE,
    NATIVE_MASS_METHODS,
    STAGES as FULL_STAGES,
    _load_coati,
    _load_external_cache,
)
from plot_gastrulation_strict_loo_six_metrics_revised import (
    METHODS,
    expected_mass_aware_recall,
    local_predicted_mass,
    normalized_weights,
    plot_metrics,
    target_radii,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results/gastrulation_full_train_fit_six_metrics"
DEFAULT_OUTPUT = ROOT / "results/gastrulation_full_train_fit_six_metrics_revised"

SOURCE_METHOD = {
    "COATI balanced": "BSOT C_y=0.3",
    "COATI unbalanced": "USOT C_y=0.3",
    "CytoBridge balanced 20k": "CytoBridge balanced 20k",
    "CytoBridge unbalanced 20k": "CytoBridge unbalanced 20k",
    "MIOFlow 20k": "MIOFlow 20k",
    "TIGON 20k": "TIGON 20k",
    "TrajectoryNet 20k": "TrajectoryNet 20k",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--external-cache", type=Path, default=DEFAULT_EXTERNAL_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--distance-batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_metrics(args: argparse.Namespace) -> pd.DataFrame:
    original = pd.read_csv(args.input_dir / "full_train_fit_all_scores.csv")
    rna_references, atac_references, _, _, _ = _load_references()
    references = {"RNA": rna_references, "ATAC": atac_references}
    external_predictions, external_log_masses = _load_external_cache(
        args.external_cache
    )
    coati_balanced, coati_unbalanced, coati_unbalanced_mass = _load_coati()
    with np.load(
        args.input_dir / "trajectorynet_terminal_predictions.npz",
        allow_pickle=False,
    ) as cached:
        trajectorynet = {
            "e80": np.asarray(cached["prediction_e80"], dtype=np.float32),
            "e85": np.asarray(cached["prediction_e85"], dtype=np.float32),
        }

    device = torch.device("cpu")
    t_model = _load_t_model(device)
    rows: list[dict[str, object]] = []
    for (
        stage_index,
        cache_index,
        trajectory_step,
        physical_time,
        stage_tag,
        _,
    ) in FULL_STAGES:
        stage = STAGE_NAMES[stage_index]
        predictions = {
            "COATI balanced": coati_balanced[trajectory_step]
            .numpy()
            .astype(np.float32, copy=False),
            "COATI unbalanced": coati_unbalanced[trajectory_step]
            .numpy()
            .astype(np.float32, copy=False),
            "CytoBridge balanced 20k": external_predictions[
                "CytoBridge balanced"
            ][cache_index],
            "CytoBridge unbalanced 20k": external_predictions[
                "CytoBridge unbalanced"
            ][cache_index],
            "MIOFlow 20k": external_predictions["MIOFlow"][cache_index],
            "TIGON 20k": external_predictions["TIGON"][cache_index],
            "TrajectoryNet 20k": trajectorynet[stage_tag],
        }
        log_masses = {
            "COATI unbalanced": coati_unbalanced_mass[
                trajectory_step, :, 0
            ]
            .numpy()
            .astype(np.float64, copy=False),
            "CytoBridge unbalanced 20k": external_log_masses[
                "CytoBridge unbalanced"
            ][cache_index].astype(np.float64, copy=False),
            "TIGON 20k": external_log_masses["TIGON"][cache_index].astype(
                np.float64, copy=False
            ),
        }
        radii = {
            modality: target_radii(
                np.asarray(values[stage_index], dtype=np.float32), args.knn_k
            )
            for modality, values in references.items()
        }
        for method, display, style_name, hatched in METHODS:
            prediction_rna = np.asarray(predictions[method], dtype=np.float32)
            if method in log_masses:
                weights, weighting = normalized_weights(log_masses[method])
            else:
                weights = np.full(
                    len(prediction_rna), 1.0 / len(prediction_rna), dtype=np.float64
                )
                weighting = "uniform"
            effective_n = float(1.0 / np.sum(np.square(weights)))
            prediction_atac = _apply_t(
                t_model,
                prediction_rna,
                float(physical_time),
                device,
                args.batch_size,
            ).astype(np.float32, copy=False)
            source_method = SOURCE_METHOD[method]
            match = original[
                original["stage"].eq(stage)
                & original["method"].eq(source_method)
            ]
            if len(match) != 1:
                raise ValueError(
                    f"Expected one full-train score for {stage} {source_method}; "
                    f"found {len(match)}"
                )
            score = match.iloc[0]
            row: dict[str, object] = {
                "stage": stage,
                "method": method,
                "display": display,
                "style": style_name,
                "hatched": hatched,
                "weighting": weighting,
                "effective_particle_count": effective_n,
                "rna_w2": float(
                    np.sqrt(2.0 * score["rna_sinkhorn_divergence"])
                ),
                "atac_w2": float(
                    np.sqrt(2.0 * score["atac_sinkhorn_divergence"])
                ),
                "rna_composition_similarity": float(
                    1.0 - np.sqrt(score["rna_celltype_composition_jsd"])
                ),
                "atac_composition_similarity": float(
                    1.0 - np.sqrt(score["atac_celltype_composition_jsd"])
                ),
                "rna_original_jsd": float(
                    score["rna_celltype_composition_jsd"]
                ),
                "atac_original_jsd": float(
                    score["atac_celltype_composition_jsd"]
                ),
            }
            for modality, prediction, key in (
                ("RNA", prediction_rna, "rna"),
                ("ATAC", prediction_atac, "atac"),
            ):
                reference = np.asarray(
                    references[modality][stage_index], dtype=np.float32
                )
                local_mass = local_predicted_mass(
                    reference,
                    prediction,
                    weights,
                    radii[modality],
                    args.distance_batch_size,
                )
                row[f"{key}_target_support_recall"] = expected_mass_aware_recall(
                    local_mass, effective_n
                )
                row[f"{key}_hard_geometric_recall"] = float(
                    np.mean(local_mass > 0)
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    outputs = (
        args.output_dir / "revised_six_metric_scores.csv",
        args.output_dir / "gastrulation_full_train_e80_e85_six_metrics_revised.png",
        args.output_dir / "gastrulation_full_train_e80_e85_six_metrics_revised.pdf",
        args.output_dir / "gastrulation_full_train_e80_e85_six_metrics_revised.svg",
        args.output_dir / "manifest.json",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite outputs; pass --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics(args)
    metrics.to_csv(outputs[0], index=False)
    floor = pd.read_csv(args.input_dir / "sampling_floor_summary.csv")
    plot_metrics(
        metrics,
        floor,
        args.output_dir,
        stage_heading="Full-train",
        stem_name="gastrulation_full_train_e80_e85_six_metrics_revised",
    )
    source_manifest = json.loads(
        (
            ROOT
            / "results/gastrulation_nonloo_all_rna_only_sinkhorn_pareto"
            / "analysis_manifest.json"
        ).read_text(encoding="utf-8")
    )
    manifest = {
        "task": "Gastrulation full-train E8.0/E8.5 revised six-metric benchmark",
        "training_regime": "all evaluated stages present during training",
        "knn_k": args.knn_k,
        "support": (
            "Expected mass-aware target support inside each real cell's 15-NN "
            "radius, using the prediction weight effective sample size."
        ),
        "composition_similarity": "1 - sqrt(base-2 composition JSD)",
        "w2": "sqrt(2 * debiased Sinkhorn divergence)",
        "atac": "Every predicted RNA distribution is mapped by the same full-data T.",
        "excluded_methods": ["OT(RNA)", "UOT(RNA)"],
        "included_methods": [record[1] for record in METHODS],
        "external_prediction_cache": str(args.external_cache.resolve()),
        "external_checkpoints": source_manifest["checkpoints"],
        "figure_size_inches": [8.05, 4.75],
        "font": "Arial; 12 pt titles and 10 pt method/value labels",
    }
    outputs[-1].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(metrics[
        [
            "stage",
            "display",
            "rna_target_support_recall",
            "atac_target_support_recall",
            "rna_w2",
            "atac_w2",
            "rna_composition_similarity",
            "atac_composition_similarity",
        ]
    ].to_string(index=False))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
