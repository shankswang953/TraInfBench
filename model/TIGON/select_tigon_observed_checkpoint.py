#!/usr/bin/env python
"""Select a TIGON checkpoint using observed training stages only."""

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
    parser.add_argument(
        "--selection-initialization",
        choices=("native_cov0.02", "exact_cells"),
        default="native_cov0.02",
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--require-protocol",
        help="Reject candidate checkpoints without this serialized protocol tag.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_key = (args.dataset, args.task)
    if task_key not in TASKS:
        raise ValueError(f"Task is not configured: {task_key}")
    if args.eval_samples <= 0:
        raise ValueError("--eval-samples must be positive")

    result_dir = args.result_dir.resolve()
    config = json.loads((result_dir / "config.json").read_text(encoding="utf-8"))
    cache_dir = (
        args.cache_dir.resolve()
        if args.cache_dir is not None
        else Path(config["frozen_full_ae_cache"]).resolve()
    )
    metrics_path = result_dir / "observed_checkpoint_metrics.csv"
    selection_path = result_dir / "observed_checkpoint_selection.json"
    selected_checkpoint = result_dir / "tigon_checkpoint_selected_observed.pt"
    protected = (metrics_path, selection_path, selected_checkpoint)
    existing = [path for path in protected if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Observed checkpoint selection already exists: "
            + ", ".join(str(path) for path in existing)
        )

    defaults = TASKS[task_key]
    latent, times, unique_times, audit = load_frozen_task(
        args.dataset,
        args.task,
        Path(defaults["h5ad"]).resolve(),
        cache_dir,
        latent_source=str(config.get("latent_source", "scaled_minus2_2")),
        latent_standardization=str(
            config.get("latent_standardization", "none")
        ),
    )
    if audit["frozen_latent_sha256"] != config.get("frozen_latent_sha256"):
        raise ValueError("Checkpoint selection cache does not match training")
    source_centers = latent[np.isclose(times, unique_times[0])]
    source, source_indices = initialize_particles(
        source_centers,
        args.eval_samples,
        initialization=args.selection_initialization,
        seed=args.evaluation_seed,
        covariance=NATIVE_INITIAL_COVARIANCE,
    )
    target_by_time = {
        float(time_value): deterministic_target_sample(
            latent[np.isclose(times, time_value)], args.eval_samples
        )
        for time_value in unique_times[1:]
    }

    checkpoints = sorted(result_dir.glob("tigon_checkpoint_iter*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No periodic TIGON checkpoints found in {result_dir}"
        )
    device = torch.device(args.device)
    loss_fn = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=args.sinkhorn_blur,
        debias=True,
        backend="tensorized",
    )

    rows: list[dict[str, object]] = []
    checkpoint_scores: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        model, saved_args, state = load_checkpoint_model(checkpoint, device)
        if args.require_protocol is not None:
            actual_protocol = (
                saved_args.get("safe_reproduction_protocol")
                if isinstance(saved_args, dict)
                else getattr(saved_args, "safe_reproduction_protocol", None)
            )
            if actual_protocol != args.require_protocol:
                raise ValueError(
                    f"{checkpoint} protocol={actual_protocol!r}, "
                    f"required={args.require_protocol!r}"
                )
            actual_action_mode = (
                saved_args.get("action_state_mode")
                if isinstance(saved_args, dict)
                else getattr(saved_args, "action_state_mode", None)
            )
            expected_action_mode = (
                "upstream_public_exact"
                if args.require_protocol
                == "gastrulation_tigon_upstream_public_exact_v2"
                else "official_stationary"
            )
            if actual_action_mode != expected_action_mode:
                raise ValueError(
                    f"{checkpoint} action_state_mode={actual_action_mode!r}, "
                    f"required={expected_action_mode!r}"
                )
            if (
                args.require_protocol
                == "gastrulation_tigon_upstream_public_exact_v2"
            ):
                actual_backend = (
                    saved_args.get("density_ode_backend")
                    if isinstance(saved_args, dict)
                    else getattr(saved_args, "density_ode_backend", None)
                )
                if actual_backend != "torchdiffeqpack":
                    raise ValueError(
                        f"{checkpoint} density_ode_backend={actual_backend!r}, "
                        "required='torchdiffeqpack'"
                    )
        saved_times = np.asarray(state["unique_times"], dtype=np.float32)
        if not np.array_equal(saved_times, unique_times):
            raise ValueError(
                f"{checkpoint} was trained on {saved_times.tolist()}, "
                f"not {unique_times.tolist()}"
            )
        predictions, log_growths = rollout(
            model=model,
            saved_args=saved_args,
            source=source,
            evaluation_times=unique_times,
            device=device,
            batch_size=args.batch_size,
        )
        iteration = int(state["iter"])
        stage_scores: list[float] = []
        for stage_index, time_value in enumerate(unique_times[1:], start=1):
            metrics = distribution_metrics(
                predicted=predictions[stage_index],
                log_growth=log_growths[stage_index],
                target=target_by_time[float(time_value)],
                loss_fn=loss_fn,
                device=device,
            )
            stage_scores.append(metrics["weighted_w2_squared"])
            rows.append(
                {
                    "iteration": iteration,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "stage_time": float(time_value),
                    "stage_role": "observed_train",
                    "selection_initialization": args.selection_initialization,
                    "density_sample_covariance": (
                        NATIVE_INITIAL_COVARIANCE
                        if args.selection_initialization == "native_cov0.02"
                        else 0.0
                    ),
                    "density_sigma_next": float(state["density_sigma"]),
                    **metrics,
                }
            )
        mean_score = float(np.mean(stage_scores))
        checkpoint_scores.append(
            {
                "iteration": iteration,
                "checkpoint": checkpoint,
                "checkpoint_sha256": file_sha256(checkpoint),
                "mean_observed_weighted_w2_squared": mean_score,
            }
        )
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
        print(
            f"[select] iter={iteration}, "
            f"observed_mean_W2^2={mean_score:.6g}",
            flush=True,
        )

    selected = min(
        checkpoint_scores,
        key=lambda row: (
            float(row["mean_observed_weighted_w2_squared"]),
            int(row["iteration"]),
        ),
    )
    temporary = selected_checkpoint.with_suffix(".pt.tmp")
    shutil.copy2(Path(selected["checkpoint"]), temporary)
    temporary.replace(selected_checkpoint)
    manifest = {
        "dataset": args.dataset,
        "task": args.task,
        "result_dir": str(result_dir),
        "selection_scope": "observed_training_stages_only",
        "required_checkpoint_protocol": args.require_protocol,
        "held_out_distribution_accessed_for_selection": False,
        "frozen_representation_is_transductive": args.task != "full",
        "representation_policy": (
            "one frozen full-data AE embedding; LOO trajectory training and "
            "checkpoint scoring use only task-observed stages"
        ),
        "held_out_time": audit["held_out_time"],
        "observed_training_times": [float(value) for value in unique_times],
        "source_time": float(unique_times[0]),
        "scored_times": [float(value) for value in unique_times[1:]],
        "selection_metric": (
            "equal-time mean of growth-weighted 2x debiased p=2 "
            "Sinkhorn divergence in frozen AE10"
        ),
        "sinkhorn_blur": float(args.sinkhorn_blur),
        "selection_initialization": args.selection_initialization,
        "density_sample_covariance": (
            NATIVE_INITIAL_COVARIANCE
            if args.selection_initialization == "native_cov0.02"
            else 0.0
        ),
        "evaluation_seed": int(args.evaluation_seed),
        "source_center_indices_sha256": hashlib.sha256(
            source_indices.tobytes()
        ).hexdigest(),
        "eval_samples": int(args.eval_samples),
        "candidate_checkpoints": len(checkpoint_scores),
        "selected_iteration": int(selected["iteration"]),
        "selected_source_checkpoint": str(selected["checkpoint"]),
        "selected_source_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_checkpoint": str(selected_checkpoint),
        "selected_checkpoint_sha256": file_sha256(selected_checkpoint),
        "selected_mean_observed_weighted_w2_squared": float(
            selected["mean_observed_weighted_w2_squared"]
        ),
        "frozen_latent_sha256": audit["frozen_latent_sha256"],
        "metrics_csv": str(metrics_path),
    }
    selection_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[selected] iter={manifest['selected_iteration']}, "
        f"observed_mean_W2^2="
        f"{manifest['selected_mean_observed_weighted_w2_squared']:.6g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
