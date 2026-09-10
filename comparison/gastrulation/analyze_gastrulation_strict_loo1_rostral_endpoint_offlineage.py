#!/usr/bin/env python
"""Endpoint-only E7.5 Rostral to held-out E8.0 off-lineage analysis."""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from analyze_gastrulation_loo_rostral_gross_cross_lineage import (
    BLOOD_ERYTHROID,
    ENDODERM,
    GROSS_CROSS_LINEAGE,
    K_NEIGHBORS,
    MESODERM,
    SOURCE_CELLTYPE,
    clean_label,
    summarize_prediction,
    trajectorynet_terminal_rostral_prediction,
)
from evaluate_gastrulation_full_cmcc import _load_references


ROOT = Path(__file__).resolve().parents[2]
SOURCE_H5AD = ROOT / "data/gastrulation_rna_loo_time1_cytobridge.h5ad"
PREDICTION_CACHE = (
    ROOT
    / "results/gastrulation_strict_loo1_benchmark"
    / "strict_loo1_e80_benchmark_predictions.npz"
)
TRAJECTORYNET_CHECKPOINT = (
    ROOT
    / "results/trajectorynet_gastrulation_loo_time1_20000"
    / "checkpt-20000.pth"
)
OUTPUT_DIR = ROOT / "results/gastrulation_strict_loo1_rostral_endpoint_offlineage"


DISPLAY_NAMES = {
    "BSOT strict LOO1 C_y=0.3": "COATI bal.",
    "USOT strict LOO1 C_y=0.3": "COATI unbal.",
    "BSOT RNA-only (no Sync)": "OT(RNA)",
    "USOT RNA-only (no Sync)": "UOT(RNA)",
    "CytoBridge balanced 20k": "CytoBridge bal.",
    "CytoBridge unbalanced 20k": "CytoBridge unbal.",
    "MIOFlow 20k": "MIOFlow (GAGA10D)",
}
METHOD_ORDER = tuple(DISPLAY_NAMES.values()) + (
    "TrajectoryNet (terminal backward)",
)


def gross_group(celltype: str) -> str:
    if celltype in MESODERM:
        return "Mesoderm"
    if celltype in BLOOD_ERYTHROID:
        return "Blood/endothelium/erythroid"
    if celltype in ENDODERM:
        return "Endoderm"
    raise ValueError(f"{celltype} is not a predefined gross off-lineage type")


def load_forward_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with np.load(PREDICTION_CACHE, allow_pickle=False) as cache:
        index = 0
        while f"method_{index}" in cache.files:
            source = str(cache[f"method_{index}"])
            if source in DISPLAY_NAMES:
                records.append(
                    {
                        "source": source,
                        "display": DISPLAY_NAMES[source],
                        "rna": np.asarray(cache[f"rna_{index}"], dtype=np.float32),
                        "log_mass": np.asarray(
                            cache[f"log_mass_{index}"], dtype=np.float64
                        ),
                        "has_mass": bool(cache[f"has_mass_{index}"]),
                    }
                )
            index += 1
    missing = set(DISPLAY_NAMES) - {str(record["source"]) for record in records}
    if missing:
        raise ValueError(f"Missing expected forward methods: {sorted(missing)}")
    return records


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = ad.read_h5ad(SOURCE_H5AD, backed="r")
    try:
        source_times = pd.to_numeric(
            source.obs["time_point_processed"], errors="raise"
        ).to_numpy(float)
        source_obs = source.obs.loc[np.isclose(source_times, 0.0)].copy()
    finally:
        source.file.close()
    source_labels = np.asarray(
        [clean_label(value) for value in source_obs["celltype"]], dtype=str
    )
    source_mask = source_labels == SOURCE_CELLTYPE
    if len(source_labels) != 9018 or int(source_mask.sum()) != 1568:
        raise ValueError(
            f"Unexpected E7.5 source: total={len(source_labels)}, "
            f"Rostral={int(source_mask.sum())}"
        )

    rna_references, _, labels_by_stage, rna_scale, _ = _load_references()
    heldout_reference = np.asarray(rna_references[1], dtype=np.float32)
    heldout_labels = np.asarray(
        [clean_label(value) for value in labels_by_stage[1]], dtype=str
    )
    neighbors = NearestNeighbors(
        n_neighbors=K_NEIGHBORS,
        algorithm="brute",
        metric="euclidean",
        n_jobs=-1,
    ).fit(heldout_reference)

    summary_rows: list[dict[str, object]] = []
    composition_rows: list[dict[str, object]] = []
    for record in load_forward_records():
        summary, composition = summarize_prediction(
            stage="E8.0",
            method=str(record["display"]),
            prediction=np.asarray(record["rna"], dtype=np.float32),
            log_mass=np.asarray(record["log_mass"], dtype=np.float64),
            native_mass=bool(record["has_mass"]),
            source_mask=source_mask,
            neighbors=neighbors,
            target_labels=heldout_labels,
            provenance=str(PREDICTION_CACHE),
        )
        summary_rows.append(summary)
        composition_rows.extend(composition)

    trajectorynet_cache = OUTPUT_DIR / "trajectorynet_terminal_backward_e80.npz"
    tn_prediction, tn_mask, tn_audit = trajectorynet_terminal_rostral_prediction(
        stage="E8.0",
        checkpoint=TRAJECTORYNET_CHECKPOINT,
        rna_references=rna_references,
        labels_by_stage=labels_by_stage,
        rna_scale=rna_scale,
        output_path=trajectorynet_cache,
    )
    tn_summary, tn_composition = summarize_prediction(
        stage="E8.0",
        method="TrajectoryNet (terminal backward)",
        prediction=tn_prediction,
        log_mass=np.zeros(len(tn_prediction), dtype=np.float32),
        native_mass=False,
        source_mask=np.ones(len(tn_prediction), dtype=bool),
        neighbors=neighbors,
        target_labels=heldout_labels,
        provenance=str(trajectorynet_cache),
    )
    tn_summary["terminal_particle_count"] = int(len(tn_mask))
    summary_rows.append(tn_summary)
    composition_rows.extend(tn_composition)

    summary = pd.DataFrame(summary_rows)
    composition = pd.DataFrame(composition_rows)
    destination_rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        method_summary = summary.loc[summary["method"].eq(method)].iloc[0]
        gross_total = float(method_summary["gross_cross_lineage_leakage_fraction"])
        method_composition = composition.loc[
            composition["method"].eq(method)
            & composition["predicted_celltype"].isin(GROSS_CROSS_LINEAGE)
        ].copy()
        method_composition = method_composition.sort_values(
            "predicted_fraction", ascending=False, ignore_index=True
        )
        for rank, row in method_composition.iterrows():
            absolute = float(row["predicted_fraction"])
            destination_rows.append(
                {
                    "stage": "E8.0",
                    "method": method,
                    "source_celltype": SOURCE_CELLTYPE,
                    "destination_rank": rank + 1,
                    "off_lineage_group": gross_group(
                        str(row["predicted_celltype"])
                    ),
                    "predicted_celltype": str(row["predicted_celltype"]),
                    "absolute_fraction_of_rostral_cohort": absolute,
                    "share_of_gross_off_lineage": (
                        absolute / gross_total if gross_total > 0 else np.nan
                    ),
                }
            )
        if len(method_composition):
            top = method_composition.iloc[0]
            mask = summary["method"].eq(method)
            summary.loc[mask, "top_off_lineage_destination"] = str(
                top["predicted_celltype"]
            )
            summary.loc[mask, "top_off_lineage_destination_fraction"] = float(
                top["predicted_fraction"]
            )
            summary.loc[mask, "top_destination_share_of_gross_off_lineage"] = (
                float(top["predicted_fraction"]) / gross_total
                if gross_total > 0
                else np.nan
            )

    summary["method_order"] = summary["method"].map(
        {method: index for index, method in enumerate(METHOD_ORDER)}
    )
    summary = summary.sort_values("method_order").drop(columns="method_order")
    destinations = pd.DataFrame(destination_rows)
    summary_path = OUTPUT_DIR / "strict_loo1_rostral_to_e80_offlineage_k5.csv"
    destination_path = OUTPUT_DIR / "strict_loo1_rostral_to_e80_wrong_destinations_k5.csv"
    composition_path = OUTPUT_DIR / "strict_loo1_rostral_to_e80_full_composition_k5.csv"
    summary.to_csv(summary_path, index=False)
    destinations.to_csv(destination_path, index=False)
    composition.to_csv(composition_path, index=False)
    manifest = {
        "analysis": "Endpoint-only E7.5 Rostral to held-out E8.0 off-lineage",
        "no_temporal_auc": True,
        "source": {
            "stage": "E7.5",
            "celltype": SOURCE_CELLTYPE,
            "cells": int(source_mask.sum()),
        },
        "endpoint": "held-out E8.0 RNA reference",
        "classifier": "soft 5-NN in normalized shared RNA PCA50",
        "gross_off_lineage_definition": {
            "mesoderm": sorted(MESODERM),
            "blood_endothelium_erythroid": sorted(BLOOD_ERYTHROID),
            "endoderm": sorted(ENDODERM),
        },
        "unbalanced_weighting": "native mass normalized within source Rostral cohort",
        "balanced_weighting": "uniform",
        "strict_coati_balanced_and_unbalanced_source": str(PREDICTION_CACHE),
        "trajectorynet": tn_audit,
        "excluded_methods": ["TIGON"],
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    display = summary[
        [
            "method",
            "gross_cross_lineage_leakage_fraction",
            "mesoderm_leakage_fraction",
            "blood_erythroid_leakage_fraction",
            "endoderm_leakage_fraction",
            "top_off_lineage_destination",
            "top_off_lineage_destination_fraction",
        ]
    ].copy()
    for column in display.columns[1:5].tolist() + [
        "top_off_lineage_destination_fraction"
    ]:
        display[column] *= 100.0
    print(display.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nSummary: {summary_path}")
    print(f"Wrong destinations: {destination_path}")


if __name__ == "__main__":
    main()
