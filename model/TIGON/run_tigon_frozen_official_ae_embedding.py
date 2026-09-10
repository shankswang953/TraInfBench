#!/usr/bin/env python
"""Train TIGON Full/LOO using a frozen full-data official AE embedding."""

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
import torch

from tigon_official_ae_common import (
    OFFICIAL_SOURCE_COMMIT,
    OFFICIAL_SOURCE_URL,
    TIGON_DENSITY_SIGMA_CLIP,
    TIGON_SAFE_REPRODUCTION_PROTOCOL,
    TIGON_UPSTREAM_PUBLIC_EXACT_PROTOCOL,
    build_tigon_density_args,
    load_tigon_module,
    resolve_device,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[2]
TASKS = {
    ("moscot", "full"): {
        "h5ad": ROOT / "data" / "moscot_rna_cytobridge.h5ad",
        "cache": ROOT / "results" / "tigon_official_ae_embeddings" / "moscot_ae10",
        "times": [0.0, 1.0, 2.0],
        "held_out_time": None,
        "action_steps": 2,
    },
    ("moscot", "loo_time1"): {
        "h5ad": ROOT / "data" / "moscot_rna_loo_time1_cytobridge.h5ad",
        "cache": ROOT / "results" / "tigon_official_ae_embeddings" / "moscot_ae10",
        "times": [0.0, 2.0],
        "held_out_time": 1.0,
        "action_steps": 1,
    },
    ("palate", "full"): {
        "h5ad": ROOT / "data" / "palate_rna_cytobridge.h5ad",
        "cache": ROOT / "results" / "tigon_official_ae_embeddings" / "palate_ae10",
        "times": [0.0, 1.0, 1.5, 2.0],
        "held_out_time": None,
        "action_steps": 4,
    },
    ("palate", "loo_time1"): {
        "h5ad": ROOT / "data" / "palate_rna_loo_time1_cytobridge.h5ad",
        "cache": ROOT / "results" / "tigon_official_ae_embeddings" / "palate_ae10",
        "times": [0.0, 1.5, 2.0],
        "held_out_time": 1.0,
        "action_steps": 4,
    },
    ("palate", "loo_time2"): {
        "h5ad": ROOT / "data" / "palate_rna_loo_time2_cytobridge.h5ad",
        "cache": ROOT / "results" / "tigon_official_ae_embeddings" / "palate_ae10",
        "times": [0.0, 1.0, 2.0],
        "held_out_time": 1.5,
        "action_steps": 2,
    },
    ("gastrulation", "full"): {
        "h5ad": ROOT / "data" / "gastrulation_rna_cytobridge.h5ad",
        "cache": (
            ROOT / "results" / "tigon_official_ae_embeddings" / "gastrulation_ae10"
        ),
        "times": [0.0, 1.0, 2.0, 2.5],
        "held_out_time": None,
        "action_steps": 5,
    },
    ("gastrulation", "loo_time1"): {
        "h5ad": ROOT / "data" / "gastrulation_rna_loo_time1_cytobridge.h5ad",
        "cache": (
            ROOT / "results" / "tigon_official_ae_embeddings" / "gastrulation_ae10"
        ),
        "times": [0.0, 2.0, 2.5],
        "held_out_time": 1.0,
        "action_steps": 5,
    },
    ("gastrulation", "loo_time2"): {
        "h5ad": ROOT / "data" / "gastrulation_rna_loo_time2_cytobridge.h5ad",
        "cache": (
            ROOT / "results" / "tigon_official_ae_embeddings" / "gastrulation_ae10"
        ),
        "times": [0.0, 1.0, 2.5],
        "held_out_time": 2.0,
        "action_steps": 5,
    },
    ("human_cerebral_no_d61", "full"): {
        "h5ad": (
            ROOT / "data" / "human_cerebral_no_d61_rna_cytobridge.h5ad"
        ),
        "cache": (
            ROOT
            / "results"
            / "tigon_official_ae_embeddings"
            / "human_cerebral_no_d61_full_ae10"
        ),
        "times": [0.0, 0.3, 0.5, 0.7, 0.8, 1.2, 1.4, 1.7, 2.2, 2.7],
        "held_out_time": None,
        "action_steps": 6,
    },
    ("human_cerebral_no_d61", "loo_day7"): {
        "h5ad": (
            ROOT
            / "data"
            / "human_cerebral_no_d61_rna_loo_day7_cytobridge.h5ad"
        ),
        "cache": (
            ROOT
            / "results"
            / "tigon_official_ae_embeddings"
            / "human_cerebral_no_d61_loo_day7_strict_ae10"
        ),
        "times": [0.0, 0.5, 0.7, 0.8, 1.2, 1.4, 1.7, 2.2, 2.7],
        "held_out_time": 0.3,
        "action_steps": 6,
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def default_outdir(dataset: str, task: str) -> Path:
    return (
        ROOT
        / "results"
        / f"tigon_{dataset}_{task}_official_ae10_dopri5_exact_n1024_20000"
    )


def _checkpoint_argument(saved_args: object, name: str) -> object:
    if isinstance(saved_args, dict):
        return saved_args.get(name)
    return getattr(saved_args, name, None)


def _optional_float_close(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return bool(np.isclose(float(left), float(right), rtol=1e-7, atol=1e-10))


def _first_nonfinite_tensor_path(value: object, prefix: str) -> str | None:
    if torch.is_tensor(value):
        return None if torch.isfinite(value).all() else prefix
    if isinstance(value, dict):
        for name, nested in value.items():
            found = _first_nonfinite_tensor_path(nested, f"{prefix}.{name}")
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found = _first_nonfinite_tensor_path(nested, f"{prefix}[{index}]")
            if found is not None:
                return found
    return None


def validate_safe_resume_checkpoint(
    checkpoint_path: Path,
    *,
    model_dim: int,
    unique_times: np.ndarray,
    counts: list[int],
    tigon_iters: int,
    num_samples: int,
    checkpoint_every: int,
    tigon_seed: int,
    action_steps: int,
    action_state_mode: str,
    density_ode_backend: str,
    safe_reproduction_protocol: str,
    density_sigma_initial: float,
    density_sigma_min: float | None,
    density_sigma_hard_clip: float | None,
    density_sigma_anneal_every: int,
) -> dict:
    """Validate the actual checkpoint payload before permitting a resume.

    Checking only ``config.json`` is insufficient: legacy runs could have a
    newer config next to a checkpoint whose serialized Namespace lacked the
    official stationary-action switch.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Cannot resume because checkpoint is missing: {checkpoint_path}"
        )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = state.get("args")
    if saved_args is None:
        raise ValueError("Cannot resume a checkpoint without serialized args")

    expected = {
        "safe_reproduction_protocol": safe_reproduction_protocol,
        "action_state_mode": action_state_mode,
        "density_stabilization": "official",
        "divergence_estimator": "exact",
        "ode_solver": "dopri5",
        "density_ode_backend": density_ode_backend,
        "action_ode_solver": "midpoint",
        "density_weight": 1e4,
        "density_sample_sigma": 0.02,
        "tigon_hidden_dim": 16,
        "tigon_hidden_layers": 4,
        "tigon_lr": 3e-3,
        "num_samples": int(num_samples),
        "tigon_iters": int(tigon_iters),
        "checkpoint_every": int(checkpoint_every),
        "seed": int(tigon_seed),
        "action_ode_steps": int(action_steps),
        "density_sigma_initial": float(density_sigma_initial),
        "density_sigma_min": density_sigma_min,
        "density_sigma_hard_clip": density_sigma_hard_clip,
        "official_density_sigma_schedule": (
            density_sigma_min is None and density_sigma_hard_clip is None
        ),
        "density_sigma_official_stop_threshold": (
            0.02
            if density_sigma_min is None and density_sigma_hard_clip is None
            else None
        ),
        "density_sigma_anneal_every": int(density_sigma_anneal_every),
        "density_sigma_anneal_factor": 0.5,
        "density_sigma_anneal_threshold": 3e-4,
        "density_sigma_anneal_stop_before_end": 400,
    }
    mismatches: list[str] = []
    for name, wanted in expected.items():
        actual = _checkpoint_argument(saved_args, name)
        if isinstance(wanted, float):
            matches = actual is not None and np.isclose(
                float(actual), wanted, rtol=1e-7, atol=1e-10
            )
        else:
            matches = actual == wanted
        if not matches:
            mismatches.append(f"{name}: saved={actual!r}, expected={wanted!r}")

    saved_times = np.asarray(state.get("unique_times", []), dtype=np.float32)
    if not np.array_equal(saved_times, np.asarray(unique_times, dtype=np.float32)):
        mismatches.append(
            f"unique_times: saved={saved_times.tolist()}, "
            f"expected={np.asarray(unique_times).tolist()}"
        )
    if int(state.get("model_dim", -1)) != int(model_dim):
        mismatches.append(
            f"model_dim: saved={state.get('model_dim')!r}, expected={model_dim}"
        )
    saved_counts = [int(value) for value in state.get("counts", [])]
    if saved_counts != [int(value) for value in counts]:
        mismatches.append(f"counts: saved={saved_counts}, expected={counts}")
    if state.get("training_objective") != "tigon-density":
        mismatches.append(
            "training_objective: "
            f"saved={state.get('training_objective')!r}, expected='tigon-density'"
        )
    saved_iteration = int(state.get("iter", -1))
    if not 0 < saved_iteration <= tigon_iters:
        mismatches.append(
            f"iter: saved={saved_iteration}, expected within [1, {tigon_iters}]"
        )
    saved_sigma = float(state.get("density_sigma", float("nan")))
    if not np.isfinite(saved_sigma) or (
        density_sigma_min is not None
        and saved_sigma < density_sigma_min - 1e-10
    ):
        mismatches.append(
            f"density_sigma: saved={saved_sigma!r}, minimum={density_sigma_min}"
        )

    for name in ("model_state_dict", "optimizer_state_dict"):
        payload = state.get(name)
        if not isinstance(payload, dict) or not payload:
            mismatches.append(f"{name}: missing or empty")
            continue
        nonfinite_path = _first_nonfinite_tensor_path(payload, name)
        if nonfinite_path is not None:
            mismatches.append(f"{nonfinite_path}: contains non-finite values")

    if safe_reproduction_protocol == TIGON_UPSTREAM_PUBLIC_EXACT_PROTOCOL:
        for rng_name in (
            "torch_rng_state",
            "python_random_state",
            "numpy_rng_state",
        ):
            if rng_name not in state:
                mismatches.append(f"{rng_name}: missing from exact v2 checkpoint")

    history = state.get("history", [])
    if not history or int(history[-1].get("iter", -1)) != saved_iteration:
        mismatches.append("history: missing or inconsistent with saved iteration")
    elif not all(
        np.isfinite(float(history[-1].get(key, float("nan"))))
        for key in (
            "loss",
            "transport_cost",
            "long_term_reconstruction",
            "short_term_reconstruction",
        )
    ):
        mismatches.append("history: latest loss components are non-finite")

    if mismatches:
        raise ValueError(
            "Unsafe or incompatible TIGON resume checkpoint. Start a new "
            "safe-reproduction output directory.\n- " + "\n- ".join(mismatches)
        )
    return state


def load_frozen_task(
    dataset: str,
    task: str,
    h5ad_path: Path,
    cache_dir: Path,
    *,
    latent_source: str = "scaled_minus2_2",
    latent_standardization: str = "none",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    ready_path = cache_dir / "embedding_ready.json"
    config_path = cache_dir / "config.json"
    if not ready_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            f"Frozen embedding cache is incomplete under {cache_dir}"
        )
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    cache_config = json.loads(config_path.read_text(encoding="utf-8"))
    cache_task = str(cache_config.get("task", "full"))
    if cache_task not in {"full", task}:
        raise ValueError(
            f"AE cache task={cache_task!r} cannot be used for requested "
            f"task={task!r}"
        )
    configured_latent_files = cache_config.get("latent_files", {})
    latent_paths = {
        "scaled_minus2_2": cache_dir
        / configured_latent_files.get(
            "scaled_minus2_2", "ae_latent_scaled_minus2_2.npy"
        ),
        "raw": cache_dir
        / configured_latent_files.get("raw", "ae_latent_raw.npy"),
    }
    if latent_source not in latent_paths:
        raise ValueError(f"Unknown latent source: {latent_source}")
    if latent_standardization not in {"none", "zscore"}:
        raise ValueError(
            f"Unknown latent standardization: {latent_standardization}"
        )
    latent_path = latent_paths[latent_source]
    obs_path = cache_dir / "obs_names.npy"
    times_path = cache_dir / "times_full.npy"
    for required in (ready_path, config_path, latent_path, obs_path, times_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Frozen AE cache is incomplete; missing {required}. "
                "Run prepare_tigon_official_ae_embedding.py first."
            )
    if ready.get("status") != "ready" or cache_config.get("dataset") != dataset:
        raise ValueError(f"Frozen embedding cache does not match dataset={dataset}")
    latent_sha256 = file_sha256(latent_path)
    if (
        latent_source == "scaled_minus2_2"
        and latent_sha256 != ready.get("latent_sha256")
    ):
        raise ValueError("Frozen AE latent checksum does not match ready marker")

    full_obs_names = pd.Index(np.load(obs_path, allow_pickle=False).astype(str))
    latent_full = np.load(latent_path, mmap_mode="r")
    times_full = np.load(times_path, allow_pickle=False).astype(np.float32)
    if (
        latent_full.ndim != 2
        or latent_full.shape[0] != len(full_obs_names)
        or times_full.shape != (len(full_obs_names),)
    ):
        raise ValueError("Frozen embedding cache arrays have inconsistent shapes")

    task_data = ad.read_h5ad(h5ad_path, backed="r")
    try:
        task_obs_names = pd.Index(task_data.obs_names.astype(str))
        task_times = pd.to_numeric(
            task_data.obs["time_point_processed"], errors="raise"
        ).to_numpy(np.float32)
    finally:
        task_data.file.close()
    indexer = full_obs_names.get_indexer(task_obs_names)
    if np.any(indexer < 0):
        raise ValueError(
            f"{int(np.sum(indexer < 0))} {dataset}/{task} cells are absent "
            "from the frozen full-data AE embedding"
        )
    if (task == "full" or cache_task == task) and (
        len(indexer) != len(full_obs_names)
        or not np.array_equal(indexer, np.arange(len(full_obs_names)))
    ):
        raise ValueError(
            "A task-specific AE cache must exactly match the task cell order"
        )
    if not np.allclose(times_full[indexer], task_times):
        raise ValueError("Task times do not match frozen full-data cache")

    expected_times = np.asarray(TASKS[(dataset, task)]["times"], dtype=np.float32)
    unique_times = np.unique(task_times)
    if not np.array_equal(unique_times, expected_times):
        raise ValueError(
            f"Unexpected {dataset}/{task} times: {unique_times.tolist()}, "
            f"expected {expected_times.tolist()}"
        )
    standardization_mean = np.asarray(
        np.mean(latent_full, axis=0), dtype=np.float32
    )
    standardization_sd = np.asarray(
        np.std(latent_full, axis=0), dtype=np.float32
    )
    if np.any(standardization_sd <= 1e-8):
        raise ValueError("Frozen AE latent contains a near-constant coordinate")
    latent_task = np.asarray(latent_full[indexer], dtype=np.float32)
    if latent_standardization == "zscore":
        latent_task = (
            latent_task - standardization_mean[None, :]
        ) / standardization_sd[None, :]
    latent_task = np.ascontiguousarray(latent_task, dtype=np.float32)
    embedding_kind = str(cache_config.get("embedding_kind", "ae")).lower()
    embedding_description = cache_config.get(
        "embedding_source",
        (
            f"AE trained for cache task={cache_task}; "
            f"latent={latent_source}; standardization={latent_standardization}"
        ),
    )
    strict_task_cache = cache_task == task and task != "full"
    cache_fit_scope = (
        "strict LOO retained stages"
        if strict_task_cache
        else "complete frozen dataset"
    )
    audit = {
        "dataset": dataset,
        "task": task,
        "task_h5ad": str(h5ad_path),
        "frozen_full_embedding_cache": str(cache_dir),
        "frozen_full_ae_cache": str(cache_dir),
        "frozen_latent_sha256": latent_sha256,
        "latent_source": latent_source,
        "latent_source_path": str(latent_path),
        "latent_standardization": latent_standardization,
        "latent_standardization_fit": cache_fit_scope,
        "latent_source_full_mean": standardization_mean.tolist(),
        "latent_source_full_sd": standardization_sd.tolist(),
        "embedding_kind": embedding_kind,
        "embedding_source": embedding_description,
        "frozen_ae_cache_recorded_official_source_commit": cache_config.get(
            "official_source_commit"
        ),
        "runtime_official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "frozen_ae_cache_source_commit_metadata_corrected_at_runtime": (
            cache_config.get("official_source_commit") != OFFICIAL_SOURCE_COMMIT
        ),
        "embedding_cache_task": cache_task,
        "embedding_retrained_for_loo": strict_task_cache,
        "latent_scaling_refit_for_loo": strict_task_cache,
        "task_selection": (
            "exact task-specific strict-LOO latent array"
            if strict_task_cache
            else "cell-ID subset of frozen full-data latent array"
        ),
        "task_cells": int(len(task_obs_names)),
        "embedding_cache_cells": int(len(full_obs_names)),
        "full_cache_cells": int(len(full_obs_names)),
        "latent_dimension": int(latent_task.shape[1]),
        "training_times": [float(value) for value in unique_times],
        "held_out_time": TASKS[(dataset, task)]["held_out_time"],
        "cell_alignment_missing": int(np.sum(indexer < 0)),
        "time_alignment_mismatches": int(
            np.sum(times_full[indexer] != task_times)
        ),
    }
    return latent_task, task_times, unique_times, audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train TIGON on Full/LOO subsets of one frozen AE embedding."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=(
            "moscot",
            "palate",
            "gastrulation",
            "human_cerebral_no_d61",
        ),
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=("full", "loo_time1", "loo_time2", "loo_day7"),
    )
    parser.add_argument("--h5ad", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--tigon-iters", type=int, default=20000)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--tigon-seed", type=int, default=1)
    parser.add_argument(
        "--action-state-mode",
        choices=("upstream_public_exact", "paper_fixed_sample"),
        default="upstream_public_exact",
        help=(
            "upstream_public_exact literally reproduces the public trans_loss "
            "zero-state nested growth integral and is the publication default; "
            "paper_fixed_sample evaluates that integral at each sampled state."
        ),
    )
    parser.add_argument(
        "--density-ode-backend",
        choices=("torchdiffeqpack", "torchdiffeq"),
        default="torchdiffeqpack",
        help=(
            "Use the exact TorchDiffEqPack Dopri5 backend bundled with the "
            "frozen public TIGON commit, or the compatible torchdiffeq port."
        ),
    )
    parser.add_argument(
        "--latent-source",
        choices=("scaled_minus2_2", "raw"),
        default="scaled_minus2_2",
        help="Choose the frozen AE10 coordinate array used for TIGON.",
    )
    parser.add_argument(
        "--latent-standardization",
        choices=("none", "zscore"),
        default="none",
        help=(
            "Optionally z-score every latent coordinate using statistics "
            "computed once on the complete frozen dataset."
        ),
    )
    parser.add_argument(
        "--latent-first-dims",
        type=int,
        help=(
            "Train on the first N coordinates of the frozen AE10 embedding. "
            "The default uses all 10 coordinates."
        ),
    )
    parser.add_argument(
        "--fixed-density-sigma",
        type=float,
        help=(
            "Use one fixed isotropic KDE covariance throughout training. "
            "This disables TIGON's bandwidth annealing."
        ),
    )
    parser.add_argument(
        "--density-sigma-initial",
        type=float,
        help=(
            "Override the starting covariance for TIGON's conditional "
            "bandwidth annealing. The upstream default is 1."
        ),
    )
    parser.add_argument(
        "--official-density-sigma-schedule",
        action="store_true",
        help=(
            "Use upstream TIGON's exact no-floor annealing rule: start at 1, "
            "test current sigma >0.02, then halve without clamping."
        ),
    )
    parser.add_argument(
        "--allow-density-sigma-below-clip",
        "--allow-fixed-density-sigma-below-clip",
        dest="allow_density_sigma_below_clip",
        action="store_true",
        help=(
            "Diagnostic-only: allow a fixed covariance or adaptive minimum "
            f"below the default safety clip {TIGON_DENSITY_SIGMA_CLIP:g}."
        ),
    )
    parser.add_argument(
        "--density-sigma-min",
        type=float,
        default=TIGON_DENSITY_SIGMA_CLIP,
        help=(
            "Lower covariance bound for adaptive bandwidth annealing. "
            "The publication-stable AE10 protocol uses 0.125."
        ),
    )
    parser.add_argument(
        "--device", default="cpu", choices=("auto", "cpu", "mps", "cuda")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate shared frozen embedding selection without training.",
    )
    args = parser.parse_args()

    if (args.dataset, args.task) not in TASKS:
        valid = sorted(task for dataset, task in TASKS if dataset == args.dataset)
        raise ValueError(
            f"Task {args.task!r} is not configured for {args.dataset}; "
            f"choose one of {valid}"
        )
    task_defaults = TASKS[(args.dataset, args.task)]
    safe_reproduction_protocol = (
        TIGON_UPSTREAM_PUBLIC_EXACT_PROTOCOL
        if args.action_state_mode == "upstream_public_exact"
        else TIGON_SAFE_REPRODUCTION_PROTOCOL
    )
    h5ad_path = (args.h5ad or task_defaults["h5ad"]).resolve()
    cache_dir = (args.cache_dir or task_defaults["cache"]).resolve()
    outdir = (
        args.outdir or default_outdir(args.dataset, args.task)
    ).resolve()
    if args.tigon_iters <= 0 or args.num_samples <= 0:
        raise ValueError("TIGON iterations and num_samples must be positive")
    if args.latent_first_dims is not None and args.latent_first_dims < 1:
        raise ValueError("--latent-first-dims must be positive")
    if args.fixed_density_sigma is not None and args.fixed_density_sigma <= 0:
        raise ValueError("--fixed-density-sigma must be positive")
    if args.density_sigma_initial is not None and args.density_sigma_initial <= 0:
        raise ValueError(
            "--density-sigma-initial must be positive"
        )
    if (
        args.fixed_density_sigma is not None
        and args.density_sigma_initial is not None
    ):
        raise ValueError(
            "--fixed-density-sigma and --density-sigma-initial are "
            "mutually exclusive"
        )
    if args.official_density_sigma_schedule:
        if args.fixed_density_sigma is not None:
            raise ValueError(
                "--official-density-sigma-schedule cannot be combined with "
                "--fixed-density-sigma"
            )
        if args.density_sigma_initial is not None and not np.isclose(
            args.density_sigma_initial, 1.0
        ):
            raise ValueError(
                "The upstream TIGON sigma schedule must start at sigma=1"
            )
    if args.density_sigma_min <= 0:
        raise ValueError("--density-sigma-min must be positive")
    if (
        args.fixed_density_sigma is not None
        and not np.isclose(args.density_sigma_min, TIGON_DENSITY_SIGMA_CLIP)
    ):
        raise ValueError(
            "--fixed-density-sigma cannot be combined with a non-default "
            "--density-sigma-min"
        )
    if args.official_density_sigma_schedule:
        runtime_sigma_hard_clip = None
        effective_fixed_density_sigma = None
        effective_density_sigma_min = None
        density_sigma_initial = 1.0
        density_sigma_min = None
    else:
        runtime_sigma_hard_clip = (
            (
                float(args.fixed_density_sigma)
                if args.fixed_density_sigma is not None
                else float(args.density_sigma_min)
            )
            if args.allow_density_sigma_below_clip
            else TIGON_DENSITY_SIGMA_CLIP
        )
        effective_fixed_density_sigma = (
            None
            if args.fixed_density_sigma is None
            else max(float(args.fixed_density_sigma), runtime_sigma_hard_clip)
        )
        effective_density_sigma_min = (
            max(float(args.density_sigma_min), runtime_sigma_hard_clip)
            if effective_fixed_density_sigma is None
            else effective_fixed_density_sigma
        )
        density_sigma_initial = (
            (
                1.0
                if args.density_sigma_initial is None
                else max(
                    float(args.density_sigma_initial),
                    effective_density_sigma_min,
                )
            )
            if effective_fixed_density_sigma is None
            else effective_fixed_density_sigma
        )
        density_sigma_min = (
            effective_density_sigma_min
            if effective_fixed_density_sigma is None
            else effective_fixed_density_sigma
        )
    density_sigma_anneal_every = (
        100 if effective_fixed_density_sigma is None else 0
    )
    if outdir.exists() and any(outdir.iterdir()) and not (
        args.resume or args.audit_only
    ):
        raise FileExistsError(
            f"{outdir} is not empty; use a protected runner with OVERWRITE=1 "
            "or pass --resume for a compatible checkpoint"
        )
    outdir.mkdir(parents=True, exist_ok=True)

    latent, times, unique_times, audit = load_frozen_task(
        args.dataset,
        args.task,
        h5ad_path,
        cache_dir,
        latent_source=args.latent_source,
        latent_standardization=args.latent_standardization,
    )
    full_latent_dimension = int(latent.shape[1])
    if (
        args.latent_first_dims is not None
        and args.latent_first_dims > full_latent_dimension
    ):
        raise ValueError(
            "--latent-first-dims exceeds the frozen embedding dimension: "
            f"{args.latent_first_dims} > {full_latent_dimension}"
        )
    selected_dimensions = (
        full_latent_dimension
        if args.latent_first_dims is None
        else int(args.latent_first_dims)
    )
    coordinate_indices = list(range(selected_dimensions))
    latent = np.ascontiguousarray(
        latent[:, :selected_dimensions], dtype=np.float32
    )
    audit.update(
        {
            "frozen_full_latent_dimension": full_latent_dimension,
            "latent_dimension": selected_dimensions,
            "latent_coordinate_selection": (
                "all frozen embedding coordinates"
                if selected_dimensions == full_latent_dimension
                else f"first {selected_dimensions} frozen embedding coordinates"
            ),
            "latent_coordinate_indices": coordinate_indices,
            "selected_latent_sha256": array_sha256(latent),
        }
    )
    audit_path = outdir / (
        "frozen_ae_embedding_audit.json"
        if audit.get("embedding_kind") == "ae"
        else "frozen_embedding_audit.json"
    )
    if args.resume and audit_path.exists():
        previous = json.loads(audit_path.read_text(encoding="utf-8"))
        if previous.get("frozen_latent_sha256") != audit["frozen_latent_sha256"]:
            raise ValueError("Cannot resume with a different frozen AE embedding")
        previous_selected_sha256 = previous.get("selected_latent_sha256")
        if (
            previous_selected_sha256 is not None
            and previous_selected_sha256 != audit["selected_latent_sha256"]
        ):
            raise ValueError(
                "Cannot resume with a different frozen AE coordinate selection"
            )
        if (
            previous_selected_sha256 is None
            and selected_dimensions != full_latent_dimension
        ):
            raise ValueError(
                "Legacy checkpoints without coordinate-selection metadata "
                "can only resume in the original full AE10 space"
            )
    config_path = outdir / "config.json"
    if args.resume and config_path.exists():
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        previous_action_state_mode = previous_config.get("action_state_mode")
        if previous_action_state_mode != args.action_state_mode:
            raise ValueError(
                "Cannot resume with a different TIGON action-state policy: "
                f"previous={previous_action_state_mode!r}, "
                f"requested={args.action_state_mode!r}."
            )
        if previous_config.get("safe_reproduction_protocol") != safe_reproduction_protocol:
            raise ValueError(
                "Cannot resume with a different safe-reproduction protocol"
            )
        previous_fixed_sigma = previous_config.get("fixed_density_sigma")
        previous_effective_fixed_sigma = (
            None
            if previous_fixed_sigma is None
            else max(float(previous_fixed_sigma), TIGON_DENSITY_SIGMA_CLIP)
        )
        if previous_effective_fixed_sigma != effective_fixed_density_sigma:
            raise ValueError(
                "Cannot resume with a different fixed-density-sigma policy: "
                f"previous={previous_effective_fixed_sigma}, "
                f"requested={effective_fixed_density_sigma}"
            )
        previous_sigma_min = previous_config.get(
            "density_sigma_min", TIGON_DENSITY_SIGMA_CLIP
        )
        if not _optional_float_close(
            previous_sigma_min, effective_density_sigma_min
        ):
            raise ValueError(
                "Cannot resume with a different density-sigma-min policy: "
                f"previous={previous_sigma_min}, "
                f"requested={effective_density_sigma_min}"
            )
        previous_hard_clip = previous_config.get(
            "density_sigma_hard_clip", TIGON_DENSITY_SIGMA_CLIP
        )
        if not _optional_float_close(previous_hard_clip, runtime_sigma_hard_clip):
            raise ValueError(
                "Cannot resume with a different density-sigma hard clip: "
                f"previous={previous_hard_clip}, "
                f"requested={runtime_sigma_hard_clip}"
            )
    if args.resume:
        counts = [
            int(np.sum(np.isclose(times, time_value)))
            for time_value in unique_times
        ]
        resumed_state = validate_safe_resume_checkpoint(
            outdir / "tigon_checkpoint_latest.pt",
            model_dim=selected_dimensions,
            unique_times=unique_times,
            counts=counts,
            tigon_iters=args.tigon_iters,
            num_samples=args.num_samples,
            checkpoint_every=args.checkpoint_every,
            tigon_seed=args.tigon_seed,
            action_steps=int(task_defaults["action_steps"]),
            action_state_mode=args.action_state_mode,
            density_ode_backend=args.density_ode_backend,
            safe_reproduction_protocol=safe_reproduction_protocol,
            density_sigma_initial=float(density_sigma_initial),
            density_sigma_min=density_sigma_min,
            density_sigma_hard_clip=runtime_sigma_hard_clip,
            density_sigma_anneal_every=int(density_sigma_anneal_every),
        )
        print(
            "[resume-audit] checkpoint payload passed safe reproduction "
            f"validation at iter={int(resumed_state['iter'])}",
            flush=True,
        )
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(
        f"[audit] {args.dataset}/{args.task}: cells={latent.shape[0]}, "
        f"cache_task={audit['embedding_cache_task']}, "
        f"embedding={audit['frozen_latent_sha256'][:12]}, "
        f"times={unique_times.tolist()}",
        flush=True,
    )
    if args.audit_only:
        print("[audit-only] frozen embedding checks passed", flush=True)
        return

    started = time.time()
    embedding_kind = str(audit.get("embedding_kind", "ae"))
    embedding_scope_statement = (
        f"LOO refits the {embedding_kind} and per-axis latent scaling using "
        "retained stages only; TIGON trajectory training and checkpoint "
        "selection also exclude the held-out stage"
        if audit.get("embedding_retrained_for_loo")
        else (
            f"LOO uses one full-data frozen {embedding_kind} embedding; TIGON "
            "trajectory training and checkpoint selection exclude the "
            "held-out stage"
        )
    )
    declared_deviations = [
        (
            "the reported checkpoint is selected using observed training "
            "stages only instead of always using the final iteration"
        ),
        embedding_scope_statement,
        (
            "decoded predictions are mapped through a frozen bridge only "
            "for common-space benchmark evaluation"
        ),
    ]
    if args.latent_source == "scaled_minus2_2":
        declared_deviations.insert(
            0,
            (
                f"the frozen {embedding_kind.upper()} coordinates are scaled per axis to [-2,2] "
                "before TIGON trajectory training"
            ),
        )
    if args.density_ode_backend == "torchdiffeq":
        declared_deviations.insert(
            0,
            (
                "torchdiffeq Dopri5 replaces the public TorchDiffEqPack "
                "Dopri5 backend; equations and tolerances are unchanged"
            ),
        )
    declared_deviations.insert(
        0,
        (
            "the public Python-loop KDE is evaluated with a vectorized "
            "log-sum-exp implementation verified numerically equivalent"
        ),
    )
    if not args.official_density_sigma_schedule:
        declared_deviations.insert(
            0,
            (
                "10D raw-density covariance annealing uses an explicit lower "
                f"bound of {density_sigma_min:g}"
            ),
        )
    if args.action_state_mode != "upstream_public_exact":
        declared_deviations.insert(
            0,
            (
                "the action growth integral is evaluated at fixed sampled "
                "coordinates rather than at the public code's zero state"
            ),
        )
    config = {
        **audit,
        "outdir": str(outdir),
        "tigon_iters": int(args.tigon_iters),
        "num_samples": int(args.num_samples),
        "checkpoint_every": int(args.checkpoint_every),
        "tigon_seed": int(args.tigon_seed),
        "training_objective": "tigon-density",
        "safe_reproduction_protocol": safe_reproduction_protocol,
        "ode_solver": "dopri5",
        "ode_backend": args.density_ode_backend,
        "ode_rtol": 1e-3,
        "ode_atol": 1e-5,
        "divergence_estimator": "exact",
        "density_loss": "short+long raw-density MSE",
        "density_weight": 1e4,
        "density_stabilization": "official",
        "density_bandwidth_policy": (
            (
                "upstream TIGON: halve while current sigma >0.02; "
                "no post-update floor"
            )
            if args.official_density_sigma_schedule
            else f"adaptive annealing clipped at sigma={density_sigma_min:g}"
            if effective_fixed_density_sigma is None
            else "fixed covariance; annealing disabled"
        ),
        "official_density_sigma_schedule": bool(
            args.official_density_sigma_schedule
        ),
        "density_sigma_official_stop_threshold": (
            0.02 if args.official_density_sigma_schedule else None
        ),
        "fixed_density_sigma_requested": args.fixed_density_sigma,
        "fixed_density_sigma": effective_fixed_density_sigma,
        "density_sigma_initial_requested": args.density_sigma_initial,
        "allow_fixed_density_sigma_below_clip": bool(
            args.allow_density_sigma_below_clip
        ),
        "allow_density_sigma_below_clip": bool(
            args.allow_density_sigma_below_clip
        ),
        "density_sigma_hard_clip": runtime_sigma_hard_clip,
        "density_sigma_initial": density_sigma_initial,
        "density_sigma_min": density_sigma_min,
        "density_sigma_anneal_every": density_sigma_anneal_every,
        "density_sigma_anneal_factor": 0.5,
        "density_sigma_anneal_threshold": 3e-4,
        "density_sigma_anneal_stop_before_end": 400,
        "density_sample_sigma": 0.02,
        "tigon_hidden_dim": 16,
        "tigon_hidden_layers": 4,
        "tigon_learning_rate": 3e-3,
        "tigon_weight_decay": 0.01,
        "action_solver": "midpoint",
        "action_ode_solver": "midpoint",
        "action_steps": int(task_defaults["action_steps"]),
        "action_state_mode": args.action_state_mode,
        "action_spatial_state": "fixed sampled coordinates in outer action ODE",
        "action_growth_integral": (
            "nested midpoint integral at zero spatial state, literally matching "
            "the public trans_loss implementation"
            if args.action_state_mode == "upstream_public_exact"
            else "nested midpoint integral at each fixed sampled coordinate"
        ),
        "official_source_url": OFFICIAL_SOURCE_URL,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "benchmark_implementation_file": str(
            ROOT / "model" / "TIGON" / "run_tigon_moscot_ae.py"
        ),
        "benchmark_implementation_sha256": file_sha256(
            ROOT / "model" / "TIGON" / "run_tigon_moscot_ae.py"
        ),
        "declared_deviations_from_upstream_examples": declared_deviations,
    }
    (outdir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    device = resolve_device(args.device)
    set_seed(args.tigon_seed)
    tigon_args = build_tigon_density_args(
        outdir=outdir,
        tigon_iters=args.tigon_iters,
        num_samples=args.num_samples,
        checkpoint_every=args.checkpoint_every,
        action_steps=task_defaults["action_steps"],
        seed=args.tigon_seed,
        resume=args.resume,
        fixed_density_sigma=effective_fixed_density_sigma,
        density_sigma_initial=(
            density_sigma_initial
            if effective_fixed_density_sigma is None
            else None
        ),
        density_sigma_min=(
            effective_density_sigma_min
            if effective_fixed_density_sigma is None
            else None
        ),
        density_sigma_hard_clip=runtime_sigma_hard_clip,
        official_density_sigma_schedule=bool(
            args.official_density_sigma_schedule
        ),
        density_sigma_official_stop_threshold=0.02,
        action_state_mode=args.action_state_mode,
        density_ode_backend=args.density_ode_backend,
    )
    sigma_description = (
        "upstream sigma schedule with guard >0.02 and no floor"
        if args.official_density_sigma_schedule
        else (
            f"adaptive sigma starting at {density_sigma_initial:g}, "
            f"clipped at {density_sigma_min:g}"
        )
        if effective_fixed_density_sigma is None
        else f"fixed sigma={effective_fixed_density_sigma:g}"
    )
    print(
        "[tigon] official raw-density MSE x1e4, short+long, exact divergence, "
        f"Dopri5, action={args.action_state_mode}, midpoint integration, "
        f"hidden=16x4, {sigma_description}",
        flush=True,
    )
    load_tigon_module().train_tigon_density(
        latent,
        times,
        unique_times,
        tigon_args,
        device,
        outdir,
    )
    config["elapsed_seconds"] = time.time() - started
    config["status"] = "complete"
    (outdir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(
        f"[done] {args.dataset}/{args.task}: {outdir}, "
        f"elapsed={config['elapsed_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
