#!/usr/bin/env python3
"""Evaluate the human-cerebral COATI-balanced C_y sweep at all observed ages.

This is the missing balanced counterpart to the existing all-time
synchronization-ablation analysis.  It deliberately imports and reuses the
original analysis functions so normalization, RK4 integration, Sinkhorn blur,
sampling, and random-number conventions remain identical.
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
import importlib.util
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
from geomloss import SamplesLoss


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(
    "external/COATI/humanCerebral/ResultCompare/"
    "synchronization_ablation_7time_iter30000/make_synchronization_ablation.py"
)
DEFAULT_OUTPUT_DIR = ROOT / "results/combined_synchronization_ablation"
CY_VALUES = tuple(round(value / 10.0, 1) for value in range(1, 10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--median-sample", type=int, default=2000)
    parser.add_argument("--integration-step", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_source_module():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    spec = importlib.util.spec_from_file_location("human_sync_ablation_source", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "human_cerebral_balanced_cy_sweep_all_time.csv"
    detailed_csv = args.output_dir / "human_cerebral_balanced_cy_sweep_by_time.csv"
    manifest_path = args.output_dir / "human_cerebral_balanced_cy_sweep_manifest.json"
    if not args.overwrite and any(
        path.exists() for path in (summary_csv, detailed_csv, manifest_path)
    ):
        raise FileExistsError(
            f"Outputs already exist in {args.output_dir}; pass --overwrite"
        )

    source = load_source_module()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rna_scale = source.load_scale(source.DATA_DIR / "primal_norm_params.pt")
    atac_scale = source.load_scale(
        source.DATA_DIR / "secondary_norm_params_lsi12.pt"
    )
    observed = {
        "RNA": source.load_observed(
            source.DATA_DIR / "rna_pca30_by_time.npz", rna_scale
        ),
        "ATAC": source.load_observed(
            source.DATA_DIR / "atac_lsi12_by_time.npz", atac_scale
        ),
    }
    forward_map = source.load_map(source.FORWARD_MAP_PATH)
    medians = {
        modality: source.stable_medians(values, args.median_sample, args.seed + index)
        for index, (modality, values) in enumerate(observed.items())
    }
    blurs = {
        modality: [max(0.05 * value, 1e-6) for value in values]
        for modality, values in medians.items()
    }

    detailed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for cy in CY_VALUES:
        checkpoint = (
            source.ARCHIVE_DIR
            / "COATI_balanced/checkpoint"
            / f"ckpt_s0_e0.1_m100.0_d0.1_a{cy:.1f}_iter30000.pth"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = source.load_vector_field(
            checkpoint,
            dim=30,
            unbalanced=False,
            expected_iteration=30000,
        )
        rna_prediction, weights = source.integrate(
            model,
            observed["RNA"][0],
            unbalanced=False,
            step=args.integration_step,
        )
        atac_prediction = source.map_at_observed_times(forward_map, rna_prediction)
        predictions = {"RNA": rna_prediction, "ATAC": atac_prediction}

        modality_sums: dict[str, float] = {}
        for modality_index, modality in enumerate(("RNA", "ATAC")):
            values_by_time: list[float] = []
            for time_index, (age, model_time) in enumerate(
                zip(source.AGES, source.TIME_POINTS)
            ):
                loss = SamplesLoss(
                    loss="sinkhorn",
                    p=2,
                    blur=blurs[modality][time_index],
                    debias=True,
                )
                # Offset 3 is the original script's COATI-balanced stream.
                rng = np.random.default_rng(
                    args.seed
                    + 100000 * modality_index
                    + 1000 * time_index
                    + 3
                )
                replicate_values = source.repeated_sinkhorn(
                    loss,
                    predictions[modality][time_index],
                    observed[modality][time_index],
                    weights[time_index],
                    args.batch_size,
                    args.repeats,
                    rng,
                )
                stats = source.summarize(replicate_values)
                values_by_time.append(stats["mean"])
                detailed_rows.append(
                    {
                        "method": "COATI bal",
                        "cy": cy,
                        "modality": modality,
                        "age_day": age,
                        "model_time": model_time,
                        "sinkhorn_mean": stats["mean"],
                        "sinkhorn_std": stats["std"],
                        "sinkhorn_q025": stats["q025"],
                        "sinkhorn_q975": stats["q975"],
                        "blur": blurs[modality][time_index],
                        "checkpoint": str(checkpoint),
                    }
                )
            modality_sums[modality] = float(np.sum(values_by_time))
        summary_rows.append(
            {
                "method": "COATI bal",
                "family": "balanced",
                "native_modality": "RNA",
                "cy": cy,
                "D_RNA": modality_sums["RNA"],
                "D_ATAC": modality_sums["ATAC"],
                "checkpoint": str(checkpoint),
            }
        )
        print(
            f"[C_y={cy:.1f}] sum RNA={modality_sums['RNA']:.8f}; "
            f"ATAC={modality_sums['ATAC']:.8f}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    detailed = pd.DataFrame(detailed_rows)
    summary.to_csv(summary_csv, index=False)
    detailed.to_csv(detailed_csv, index=False)

    # The original published-in-workspace table used C_y=0.5.  This audit makes
    # any accidental divergence in preprocessing or random sampling explicit.
    original_sum_csv = (
        SOURCE.parent
        / "scatter_points_sum_all_times_coati_unbal_all_Cy_iter40000_uot_iter40000.csv"
    )
    original = pd.read_csv(original_sum_csv)
    original_cy05 = original.loc[original["method"] == "COATI bal"].iloc[0]
    recomputed_cy05 = summary.loc[np.isclose(summary["cy"], 0.5)].iloc[0]
    audit_difference = {
        "D_RNA": float(recomputed_cy05["D_RNA"] - original_cy05["D_RNA"]),
        "D_ATAC": float(recomputed_cy05["D_ATAC"] - original_cy05["D_ATAC"]),
    }
    if max(abs(value) for value in audit_difference.values()) > 1e-8:
        raise RuntimeError(
            "C_y=0.5 does not reproduce the original all-time summary: "
            f"{audit_difference}"
        )

    manifest = {
        "analysis": "human cerebral COATI-balanced all-time C_y sweep",
        "source_script": str(SOURCE),
        "source_all_time_csv": str(original_sum_csv),
        "cy_values": list(CY_VALUES),
        "checkpoint_iteration": 30000,
        "ages": list(source.AGES),
        "time_points": list(source.TIME_POINTS),
        "integration": {"method": "rk4", "step_size": args.integration_step},
        "sinkhorn": {
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "p": 2,
            "debias": True,
            "blur": "0.05 * within-time median pairwise distance",
        },
        "normalization": {"rna_scale": rna_scale, "atac_scale": atac_scale},
        "cy_0.5_reproduction_difference": audit_difference,
        "outputs": {"summary": str(summary_csv), "detailed": str(detailed_csv)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
