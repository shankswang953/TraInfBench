#!/usr/bin/env python
"""Evaluate a locked observed-selected TIGON checkpoint with two initializations."""

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

from geomloss import SamplesLoss
import numpy as np
import pandas as pd
import torch

from run_tigon_frozen_official_ae_embedding import TASKS, load_frozen_task
from tigon_checkpoint_evaluation_common import (
    NATIVE_INITIAL_COVARIANCE,
    deterministic_target_sample,
    distribution_metrics,
    file_sha256,
    initialize_particles,
    load_checkpoint_model,
    rollout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("moscot", "palate", "gastrulation"))
    parser.add_argument("--task", required=True, choices=("full", "loo_time1", "loo_time2"))
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--sinkhorn-blur", type=float, default=0.05)
    parser.add_argument("--evaluation-seed", type=int, default=20260728)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_key = (args.dataset, args.task)
    if task_key not in TASKS:
        raise ValueError(f"Task is not configured: {task_key}")
    result_dir = args.result_dir.resolve()
    config = json.loads((result_dir / "config.json").read_text(encoding="utf-8"))
    cache_dir = (
        args.cache_dir.resolve()
        if args.cache_dir is not None
        else Path(config["frozen_full_ae_cache"]).resolve()
    )
    selection_path = result_dir / "observed_checkpoint_selection.json"
    selected_checkpoint = result_dir / "tigon_checkpoint_selected_observed.pt"
    if not selection_path.exists() or not selected_checkpoint.exists():
        raise FileNotFoundError(
            "Run select_tigon_observed_checkpoint.py before evaluation"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    held_out_accessed = selection.get(
        "held_out_distribution_accessed_for_selection",
        selection.get("held_out_data_accessed"),
    )
    if held_out_accessed is not False:
        raise ValueError("Checkpoint selection is not certified observed-only")
    if (
        selection.get("selected_checkpoint_sha256")
        != file_sha256(selected_checkpoint)
    ):
        raise ValueError("Selected checkpoint checksum does not match manifest")

    metrics_path = result_dir / "selected_checkpoint_ae10_metrics.csv"
    predictions_path = result_dir / "selected_checkpoint_ae10_predictions.npz"
    manifest_path = result_dir / "selected_checkpoint_ae10_evaluation.json"
    existing = [
        path
        for path in (metrics_path, predictions_path, manifest_path)
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Selected-checkpoint evaluation already exists: "
            + ", ".join(str(path) for path in existing)
        )

    full_defaults = TASKS[(args.dataset, "full")]
    latent, times, unique_times, full_audit = load_frozen_task(
        args.dataset,
        "full",
        Path(full_defaults["h5ad"]).resolve(),
        cache_dir,
        latent_source=str(config.get("latent_source", "scaled_minus2_2")),
        latent_standardization=str(
            config.get("latent_standardization", "none")
        ),
    )
    if full_audit["frozen_latent_sha256"] != config.get(
        "frozen_latent_sha256"
    ):
        raise ValueError("Full evaluation cache does not match training")
    if (
        full_audit["frozen_latent_sha256"]
        != selection["frozen_latent_sha256"]
    ):
        raise ValueError("Full evaluation and selected model use different AEs")

    model, saved_args, state = load_checkpoint_model(
        selected_checkpoint, torch.device(args.device)
    )
    required_protocol = selection.get("required_checkpoint_protocol")
    if required_protocol is not None:
        actual_protocol = (
            saved_args.get("safe_reproduction_protocol")
            if isinstance(saved_args, dict)
            else getattr(saved_args, "safe_reproduction_protocol", None)
        )
        if actual_protocol != required_protocol:
            raise ValueError("Selected checkpoint protocol does not match manifest")
    if int(state["iter"]) != int(selection["selected_iteration"]):
        raise ValueError("Selected checkpoint iteration does not match manifest")
    source_centers = latent[np.isclose(times, unique_times[0])]
    target_by_time = {
        float(time_value): deterministic_target_sample(
            latent[np.isclose(times, time_value)], args.eval_samples
        )
        for time_value in unique_times[1:]
    }
    device = torch.device(args.device)
    loss_fn = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=args.sinkhorn_blur,
        debias=True,
        backend="tensorized",
    )
    held_out_time = TASKS[task_key]["held_out_time"]
    rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    for initialization in ("exact_cells", "native_cov0.02"):
        source, source_indices = initialize_particles(
            source_centers,
            args.eval_samples,
            initialization=initialization,
            seed=args.evaluation_seed,
            covariance=NATIVE_INITIAL_COVARIANCE,
        )
        predictions, log_growths = rollout(
            model=model,
            saved_args=saved_args,
            source=source,
            evaluation_times=unique_times,
            device=device,
            batch_size=args.batch_size,
        )
        arrays[f"{initialization}_source_indices"] = source_indices
        arrays[f"{initialization}_source"] = source
        for stage_index, time_value in enumerate(unique_times[1:], start=1):
            metrics = distribution_metrics(
                predicted=predictions[stage_index],
                log_growth=log_growths[stage_index],
                target=target_by_time[float(time_value)],
                loss_fn=loss_fn,
                device=device,
            )
            role = (
                "heldout"
                if held_out_time is not None
                and np.isclose(float(time_value), float(held_out_time))
                else "observed_train"
            )
            rows.append(
                {
                    "dataset": args.dataset,
                    "task": args.task,
                    "selected_iteration": int(state["iter"]),
                    "initialization": initialization,
                    "initialization_covariance": (
                        NATIVE_INITIAL_COVARIANCE
                        if initialization == "native_cov0.02"
                        else 0.0
                    ),
                    "stage_time": float(time_value),
                    "stage_role": role,
                    "density_sigma_next": float(state["density_sigma"]),
                    **metrics,
                }
            )
            time_tag = f"{float(time_value):g}".replace(".", "p")
            arrays[f"{initialization}_to_{time_tag}"] = predictions[
                stage_index
            ]
            arrays[f"{initialization}_log_growth_to_{time_tag}"] = (
                log_growths[stage_index]
            )
        print(f"[evaluate] {initialization} complete", flush=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_path, index=False)
    np.savez_compressed(predictions_path, **arrays)
    manifest = {
        "dataset": args.dataset,
        "task": args.task,
        "selection_manifest": str(selection_path),
        "selection_scope": selection["selection_scope"],
        "selection_held_out_distribution_accessed": held_out_accessed,
        "frozen_representation_is_transductive": selection.get(
            "frozen_representation_is_transductive"
        ),
        "selected_iteration": int(state["iter"]),
        "selected_checkpoint": str(selected_checkpoint),
        "selected_checkpoint_sha256": file_sha256(selected_checkpoint),
        "evaluation_space": (
            "frozen TIGON AE10 scaled to [-2, 2]; this is method-space QC, "
            "not the final cross-method normalized-PCA metric"
        ),
        "evaluation_initializations": {
            "exact_cells": "common conditional-push initialization",
            "native_cov0.02": (
                "TIGON-native Gaussian-mixture sampling covariance 0.02"
            ),
        },
        "eval_samples": int(args.eval_samples),
        "evaluation_seed": int(args.evaluation_seed),
        "sinkhorn_metric": (
            "growth-weighted and unweighted 2x debiased p=2 "
            "Sinkhorn divergence"
        ),
        "sinkhorn_blur": float(args.sinkhorn_blur),
        "held_out_time": held_out_time,
        "metrics_csv": str(metrics_path),
        "predictions_npz": str(predictions_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] wrote {metrics_path}", flush=True)
    print(f"[done] wrote {predictions_path}", flush=True)


if __name__ == "__main__":
    main()
