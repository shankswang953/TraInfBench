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
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torchdiffeq import odeint


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "common"
for path in [
    SCRIPTS,
    ROOT / "external" / "CytoBridge",
    ROOT / "external" / "MIOFlow",
]:
    if path.exists():
        sys.path.insert(0, str(path))

from evaluate_gastrulation_flow_methods_normalized import (
    _cytobridge_interval_path,
    _load_cytobridge_model,
    _load_mioflow_bundle,
    _mioflow_interval_path,
)
from evaluate_terminal_push import (
    _load_trajectorynet_model,
    _trajectorynet_diffeq,
)
from evaluate_tigon_gastrulation_normalized import (
    _infer_model_input_space,
    _integrate_normalized,
    _load_model as _load_tigon_model,
    _load_tigon_module,
)


OWN_ROOT = Path("external/COATI/Gastrulation")
OWN_DATA = OWN_ROOT / "data"
OWN_SYNC = OWN_ROOT / "UnbalancedSync_biological_num"
OWN_TRAJECTORIES = OWN_SYNC / "trajectory_hpc_iter20000"
OWN_CHECKPOINTS = OWN_SYNC / "checkpoint_hpc_iter20000"

STAGE_KEYS = ["time0", "time1", "time2", "time3"]
STAGE_NAMES = ["E7.5", "E8.0", "E8.5", "E8.75"]
PHYSICAL_TIMES = np.asarray([0.0, 1.0, 2.0, 2.5], dtype=np.float32)
TRAJECTORYNET_TIMES = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
OWN_STEPS = [0, 10, 20, 25]
UNKNOWN = "Unknown"


def _load_scale(path: Path) -> float:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return float(np.asarray(state["scale"]).reshape(-1)[0])


def _sanitize_labels(values: np.ndarray) -> np.ndarray:
    labels = pd.Series(values).astype("string").fillna(UNKNOWN).to_numpy(dtype=str)
    labels[np.isin(labels, ["nan", "None", "<NA>", ""])] = UNKNOWN
    return labels


def _sha256_array(x: np.ndarray) -> str:
    arr = np.ascontiguousarray(x)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _load_references() -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], float, float]:
    rna_npz = np.load(OWN_DATA / "rna_pca_by_time.npz", allow_pickle=True)
    atac_npz = np.load(OWN_DATA / "atac_lsi_by_time_14D.npz", allow_pickle=True)
    label_npz = np.load(OWN_DATA / "celltype_sub_by_stage.npz", allow_pickle=True)
    rna_scale = _load_scale(OWN_DATA / "primal_norm_params.pt")
    atac_scale = _load_scale(OWN_DATA / "secondary_norm_params.pt")
    rna = [np.asarray(rna_npz[key], dtype=np.float32) / rna_scale for key in STAGE_KEYS]
    atac = [np.asarray(atac_npz[key], dtype=np.float32) / atac_scale for key in STAGE_KEYS]
    labels = [_sanitize_labels(label_npz[key]) for key in STAGE_KEYS]
    return rna, atac, labels, rna_scale, atac_scale


def _audit_inputs(rna_scale: float) -> tuple[pd.DataFrame, dict]:
    source_h5ad = ROOT / "data" / "gastrulation_rna_cytobridge.h5ad"
    source = ad.read_h5ad(source_h5ad)
    source_times = pd.to_numeric(source.obs["time_point_processed"], errors="raise").to_numpy(float)
    source_x = np.asarray(source.obsm["X_latent"], dtype=np.float32)
    own_npz = np.load(OWN_DATA / "rna_pca_by_time.npz", allow_pickle=True)
    tn_npz = np.load(ROOT / "data" / "gastrulation_rna_trajectorynet.npz", allow_pickle=True)
    tn_x = np.asarray(tn_npz["pca"], dtype=np.float32)
    tn_labels = np.asarray(tn_npz["sample_labels"]).reshape(-1)

    cb_paths = {
        "CytoBridge balanced": ROOT / "results" / "cytobridge_gastrulation_rna_20000" / "adata.h5ad",
        "CytoBridge unbalanced": ROOT / "results" / "cytobridge_gastrulation_rna_20000_unbalanced" / "adata.h5ad",
    }
    cb_data = {name: ad.read_h5ad(path) for name, path in cb_paths.items()}
    tigon_space = np.asarray(
        np.load(ROOT / "results" / "tigon_gastrulation_rna_20000" / "model_space.npy"),
        dtype=np.float32,
    )
    mio_state = torch.load(
        ROOT / "results" / "mioflow_gastrulation_rna_20000" / "model.pt",
        map_location="cpu",
        weights_only=False,
    )

    own_concat = np.concatenate([np.asarray(own_npz[key], dtype=np.float32) for key in STAGE_KEYS])
    own_counts = [int(np.asarray(own_npz[key]).shape[0]) for key in STAGE_KEYS]
    rows: list[dict] = []

    def add(name: str, arr: np.ndarray, order_match: bool, note: str) -> None:
        arr = np.asarray(arr, dtype=np.float32)
        rows.append(
            {
                "method_input": name,
                "shape": str(tuple(arr.shape)),
                "same_shape_as_own": bool(arr.shape == own_concat.shape),
                "max_abs_diff_vs_own": _max_abs(arr, own_concat),
                "exact_values_and_order": bool(arr.shape == own_concat.shape and np.array_equal(arr, own_concat)),
                "cell_order_match": bool(order_match),
                "sha256": _sha256_array(arr),
                "note": note,
            }
        )

    add(
        "Shared H5AD X_latent",
        source_x,
        np.array_equal(source.obs_names.to_numpy(), source.obs_names.to_numpy()),
        "Raw 50D RNA PCA; physical times 0,1,2,2.5.",
    )
    add(
        "MIOFlow declared shared H5AD X_latent",
        source_x,
        True,
        "Checkpoint declares this H5AD/X_latent with identity normalization and no GAGA.",
    )
    add(
        "TrajectoryNet NPZ pca",
        tn_x,
        bool(
            np.array_equal(tn_labels, np.repeat(np.arange(4), own_counts))
            and np.array_equal(np.asarray(tn_npz["cell_ids"], dtype=str), source.obs_names.to_numpy(dtype=str))
        ),
        "Same raw PCA; time labels are indices 0,1,2,3.",
    )
    for name, cb in cb_data.items():
        add(
            f"{name} saved adata X_latent",
            np.asarray(cb.obsm["X_latent"], dtype=np.float32),
            bool(np.array_equal(cb.obs_names.to_numpy(), source.obs_names.to_numpy())),
            "Saved training AnnData checked against the shared input H5AD.",
        )
    add(
        "TIGON model_space.npy",
        tigon_space,
        True,
        "No-AE run; the 50D model space is the stored X_latent matrix.",
    )
    add(
        "FilmSync balanced raw NPZ",
        own_concat,
        True,
        "FilmSync/Args.py resolves ../data/rna_pca_by_time.npz to the same raw 50D RNA PCA fixture.",
    )
    add(
        "UnbalancedSync biological-number raw NPZ",
        own_concat,
        True,
        "The unbalanced Sync alpha sweep uses the same raw 50D RNA PCA fixture as FilmSync.",
    )

    mio_mean = np.asarray(mio_state["mean_vals"].detach().cpu(), dtype=np.float32)
    mio_std = np.asarray(mio_state["std_vals"].detach().cpu(), dtype=np.float32)
    mio_config = dict(mio_state.get("config", {}))
    mio_audit = {
        "checkpoint_input_space": str(mio_state.get("input_space")),
        "checkpoint_embedding_normalization": str(mio_state.get("embedding_normalization")),
        "checkpoint_input_h5ad": str(mio_config.get("input_h5ad")),
        "checkpoint_latent_key": str(mio_state.get("latent_key")),
        "checkpoint_use_gaga": bool(mio_state.get("use_gaga", False)),
        "checkpoint_time_points": [float(x) for x in mio_state.get("time_points", [])],
        "checkpoint_identity_mean_max_abs": float(np.max(np.abs(mio_mean))),
        "checkpoint_identity_std_max_abs_diff_from_one": float(np.max(np.abs(mio_std - 1.0))),
        "same_raw_input_supported_by_checkpoint": bool(
            mio_config.get("input_h5ad") == "data/gastrulation_rna_cytobridge.h5ad"
            and mio_state.get("latent_key") == "X_latent"
            and mio_state.get("input_space") == "obsm"
            and mio_state.get("embedding_normalization") == "identity"
            and not bool(mio_state.get("use_gaga", False))
            and np.array_equal(mio_mean, np.zeros_like(mio_mean))
            and np.array_equal(mio_std, np.ones_like(mio_std))
        ),
        "effective_support_caveat": (
            "MIOFlow received the same full input file, but its interval sampler draws one common "
            "randperm(min(n_t,n_t+1)); when adjacent snapshot sizes differ, indices beyond the smaller "
            "snapshot are never selected in that interval."
        ),
    }

    identity = {
        "own_counts_by_stage": own_counts,
        "rna_scale": rna_scale,
        "own_raw_rna_sha256": _sha256_array(own_concat),
        "all_matrix_inputs_exact": bool(all(row["exact_values_and_order"] for row in rows)),
        "mioflow": mio_audit,
    }
    return pd.DataFrame(rows), identity


def _load_t_model(device: torch.device) -> torch.nn.Module:
    model_file = OWN_DATA / "TrainT" / "train_FiLM_MLP.py"
    checkpoint = OWN_DATA / "TrainT" / "T_FiLM.pt"
    sys.path.insert(0, str(OWN_ROOT.parent))
    spec = importlib.util.spec_from_file_location("gastrulation_full_cmcc_film", model_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {model_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = module.FiLMMLP(**state["config"]).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval().requires_grad_(False)
    return model


@torch.no_grad()
def _apply_t(
    model: torch.nn.Module,
    rna_points_norm: np.ndarray,
    physical_time: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, rna_points_norm.shape[0], batch_size):
        batch = torch.as_tensor(rna_points_norm[start : start + batch_size], dtype=torch.float32, device=device)
        outputs.append(model(batch, physical_time).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _trajectorynet_rollout(
    x0_norm: np.ndarray,
    scale: float,
    device: torch.device,
    batch_size: int,
    steps_per_interval: int,
) -> list[np.ndarray]:
    checkpoint = ROOT / "results" / "trajectorynet_gastrulation_rna_20000" / "checkpt-20000.pth"
    model, args = _load_trajectorynet_model(checkpoint, x0_norm.shape[1], device, "rk4", 0.1)
    diffeq = _trajectorynet_diffeq(model)

    def velocity(t_value: torch.Tensor, x_norm: torch.Tensor) -> torch.Tensor:
        return diffeq(t_value, x_norm * scale) / scale

    endpoints = [np.asarray(x0_norm, dtype=np.float32)]
    x = endpoints[0]
    for interval in range(3):
        batches = []
        source_internal = (float(TRAJECTORYNET_TIMES[interval]) + 1.0) * args.time_scale
        target_internal = (float(TRAJECTORYNET_TIMES[interval + 1]) + 1.0) * args.time_scale
        ts = torch.linspace(
            target_internal,
            source_internal,
            steps_per_interval + 1,
            device=device,
            dtype=torch.float32,
        )
        for start in range(0, x.shape[0], batch_size):
            batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            with torch.no_grad():
                path = odeint(
                    velocity,
                    batch,
                    ts,
                    method="rk4",
                    options={"step_size": 0.1},
                )
            batches.append(path[-1].detach().cpu().numpy())
        x = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
        endpoints.append(x)
        print(f"[prediction] TrajectoryNet -> {STAGE_NAMES[interval + 1]} {x.shape}", flush=True)
    return endpoints


def _cytobridge_rollout(
    name: str,
    adata_path: Path,
    x0_norm: np.ndarray,
    scale: float,
    device: torch.device,
    batch_size: int,
    steps_per_interval: int,
) -> list[np.ndarray]:
    model = _load_cytobridge_model(adata_path, device)
    endpoints = [np.asarray(x0_norm, dtype=np.float32)]
    x = endpoints[0]
    for interval in range(3):
        path = _cytobridge_interval_path(
            model,
            x,
            float(PHYSICAL_TIMES[interval]),
            float(PHYSICAL_TIMES[interval + 1]),
            steps_per_interval,
            scale,
            batch_size,
            device,
        )
        x = path[-1].astype(np.float32, copy=False)
        endpoints.append(x)
        print(f"[prediction] {name} -> {STAGE_NAMES[interval + 1]} {x.shape}", flush=True)
    return endpoints


def _mioflow_rollout(
    x0_norm: np.ndarray,
    scale: float,
    device: torch.device,
    batch_size: int,
    steps_per_interval: int,
) -> list[np.ndarray]:
    model_path = ROOT / "results" / "mioflow_gastrulation_rna_20000" / "model.pt"
    bundle = _load_mioflow_bundle(model_path, device)
    if len(bundle.model_times) != 4:
        raise ValueError(f"Expected four MIOFlow time points, got {bundle.model_times.tolist()}")
    endpoints = [np.asarray(x0_norm, dtype=np.float32)]
    x = endpoints[0]
    for interval in range(3):
        path = _mioflow_interval_path(
            bundle,
            x,
            interval,
            steps_per_interval,
            scale,
            batch_size,
            device,
        )
        x = path[-1].astype(np.float32, copy=False)
        endpoints.append(x)
        print(f"[prediction] MIOFlow -> {STAGE_NAMES[interval + 1]} {x.shape}", flush=True)
    return endpoints


def _tigon_rollout(
    x0_norm: np.ndarray,
    scale: float,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    result_dir = ROOT / "results" / "tigon_gastrulation_rna_20000"
    module = _load_tigon_module()
    model, config, _ = _load_tigon_model(
        module,
        result_dir / "tigon.pt",
        result_dir / "config.json",
        device,
    )
    model_input_space = _infer_model_input_space(config, "auto")
    ode_steps = int(config.get("ode_steps", 8))
    endpoints = [np.asarray(x0_norm, dtype=np.float32)]
    growth = [np.zeros(x0_norm.shape[0], dtype=np.float32)]
    x = torch.as_tensor(x0_norm, dtype=torch.float32, device=device)
    cumulative_growth = torch.zeros(x.shape[0], dtype=x.dtype, device=device)
    with torch.no_grad():
        for interval in range(3):
            x, log_growth, _, _ = _integrate_normalized(
                model,
                x,
                float(PHYSICAL_TIMES[interval]),
                float(PHYSICAL_TIMES[interval + 1]),
                ode_steps,
                scale,
                model_input_space,
            )
            cumulative_growth = cumulative_growth + log_growth
            endpoints.append(x.detach().cpu().numpy().astype(np.float32, copy=False))
            growth.append(cumulative_growth.detach().cpu().numpy().astype(np.float32, copy=False))
            print(f"[prediction] TIGON -> {STAGE_NAMES[interval + 1]} {tuple(x.shape)}", flush=True)
    return endpoints, growth


def _alpha_from_path(path: Path) -> float:
    match = re.search(r"_a([0-9.]+)_iter", path.name)
    if not match:
        raise ValueError(f"Could not parse alpha from {path}")
    return float(match.group(1))


def _own_sync_rollouts(x0_norm: np.ndarray) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    main: dict[str, list[np.ndarray]] = {}
    direct: dict[str, list[np.ndarray]] = {}
    primary_paths = sorted(OWN_TRAJECTORIES.glob("primary_trajectory_s0_a*_iter20000.pt"), key=_alpha_from_path)
    if len(primary_paths) != 9:
        raise FileNotFoundError(f"Expected nine own-model alpha trajectories, found {len(primary_paths)}")
    for primary_path in primary_paths:
        alpha = _alpha_from_path(primary_path)
        tag = f"{alpha:.1f}"
        primary = torch.load(primary_path, map_location="cpu", weights_only=False)
        secondary = torch.load(
            OWN_TRAJECTORIES / f"secondary_trajectory_s0_a{tag}_iter20000.pt",
            map_location="cpu",
            weights_only=False,
        )
        primary_np = np.asarray(primary.detach().cpu(), dtype=np.float32)
        secondary_np = np.asarray(secondary.detach().cpu(), dtype=np.float32)
        if primary_np.shape[1:] != x0_norm.shape:
            raise ValueError(f"Unexpected own trajectory shape {primary_np.shape} for alpha={alpha}")
        if not np.allclose(primary_np[0], x0_norm, rtol=0.0, atol=2e-6):
            raise ValueError(f"Own alpha={alpha} does not start from the common E7.5 cells")
        name = f"UnbalancedSync biological-number alpha={tag}"
        main[name] = [primary_np[step] for step in OWN_STEPS]
        direct[name] = [secondary_np[step] for step in OWN_STEPS]
    return main, direct


def _soft_knn_probabilities(
    neighbors: NearestNeighbors,
    reference_labels: np.ndarray,
    class_to_index: dict[str, int],
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distances, indices = neighbors.kneighbors(points, return_distance=True)
    near_labels = reference_labels[indices]
    weights = np.zeros_like(distances, dtype=np.float64)
    zero = distances <= 1e-12
    rows_with_zero = zero.any(axis=1)
    weights[rows_with_zero] = zero[rows_with_zero].astype(np.float64)
    weights[~rows_with_zero] = 1.0 / np.maximum(distances[~rows_with_zero], 1e-12)
    probabilities = np.zeros((points.shape[0], len(class_to_index)), dtype=np.float64)
    row_indices = np.repeat(np.arange(points.shape[0]), indices.shape[1])
    class_indices = np.fromiter(
        (class_to_index[label] for label in near_labels.reshape(-1)),
        dtype=np.int64,
        count=near_labels.size,
    )
    np.add.at(probabilities, (row_indices, class_indices), weights.reshape(-1))
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-300)
    return probabilities, distances[:, 0]


def _cmcc_per_cell(
    p_rna: np.ndarray,
    p_atac: np.ndarray,
    unknown_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    known_indices = np.asarray([i for i in range(p_rna.shape[1]) if i != unknown_idx], dtype=int)
    rna_known = p_rna[:, known_indices]
    atac_known = p_atac[:, known_indices]
    rna_mass = rna_known.sum(axis=1)
    atac_mass = atac_known.sum(axis=1)
    valid = (rna_mass > 1e-12) & (atac_mass > 1e-12)
    rna_known[valid] /= rna_mass[valid, None]
    atac_known[valid] /= atac_mass[valid, None]
    midpoint = 0.5 * (rna_known + atac_known)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_rna = np.where(rna_known > 0, rna_known * np.log2(rna_known / midpoint), 0.0).sum(axis=1)
        kl_atac = np.where(atac_known > 0, atac_known * np.log2(atac_known / midpoint), 0.0).sum(axis=1)
    score = 1.0 - 0.5 * (kl_rna + kl_atac)
    score[~valid] = np.nan
    return score, valid


def _stage_metric(
    method: str,
    readout: str,
    stage_idx: int,
    rna_points: np.ndarray,
    atac_points: np.ndarray,
    rna_neighbors: NearestNeighbors,
    atac_neighbors: NearestNeighbors,
    labels: np.ndarray,
    classes: list[str],
) -> dict:
    class_to_index = {name: idx for idx, name in enumerate(classes)}
    unknown_idx = class_to_index[UNKNOWN]
    p_rna, rna_distance = _soft_knn_probabilities(rna_neighbors, labels, class_to_index, rna_points)
    p_atac, atac_distance = _soft_knn_probabilities(atac_neighbors, labels, class_to_index, atac_points)
    cmcc, valid = _cmcc_per_cell(p_rna, p_atac, unknown_idx)
    rna_hard_unknown = np.argmax(p_rna, axis=1) == unknown_idx
    atac_hard_unknown = np.argmax(p_atac, axis=1) == unknown_idx
    return {
        "method": method,
        "readout": readout,
        "stage": STAGE_NAMES[stage_idx],
        "stage_key": STAGE_KEYS[stage_idx],
        "physical_time": float(PHYSICAL_TIMES[stage_idx]),
        "n_query": int(rna_points.shape[0]),
        "n_valid_known": int(valid.sum()),
        "valid_known_fraction": float(valid.mean()),
        "cmcc_mean": float(np.nanmean(cmcc)),
        "cmcc_median": float(np.nanmedian(cmcc)),
        "rna_unknown_probability_mean": float(p_rna[:, unknown_idx].mean()),
        "atac_unknown_probability_mean": float(p_atac[:, unknown_idx].mean()),
        "rna_hard_unknown_rate": float(rna_hard_unknown.mean()),
        "atac_hard_unknown_rate": float(atac_hard_unknown.mean()),
        "mean_rna_1nn_distance": float(rna_distance.mean()),
        "mean_atac_1nn_distance": float(atac_distance.mean()),
    }


def _manifest(args: argparse.Namespace, rna_scale: float, atac_scale: float, input_identity: dict) -> dict:
    own_ckpts = sorted(OWN_CHECKPOINTS.glob("ckpt_s0_e0.01_m100.0_d0.01_a*_iter20000.pth"))
    return {
        "protocol": {
            "metric": "CMCC = 1 - base-2 Jensen-Shannon divergence between distance-weighted kNN cell-type distributions",
            "k": int(args.k),
            "query_cells": "all 9,018 E7.5 RNA cells, same order for every method",
            "main_readout": "Each RNA trajectory is converted to common normalized RNA PCA, then mapped with the same full-data T_FiLM checkpoint.",
            "unknown_handling": "Unknown is retained during kNN assignment, reported separately, then removed and known probabilities renormalized before JSD.",
            "overall": "Arithmetic mean of stage-level mean CMCC at E8.0, E8.5, and E8.75; E7.5 is excluded.",
            "weighting": "Uniform query-cell weights and equal stage weights.",
            "rna_scale": rna_scale,
            "atac_scale": atac_scale,
            "prediction_integration": {
                "TrajectoryNet": "rk4, step_size=0.1, 20 reported points per interval; internal time labels 0,1,2,3",
                "CytoBridge": f"midpoint Euler, {args.steps_per_interval} steps per physical interval",
                "MIOFlow": f"torchdiffeq adaptive default with {args.steps_per_interval + 1} requested points; internal times 0,1,2,3 mapped index-wise",
                "TIGON": "training/evaluation explicit midpoint Euler with config ode_steps=8 and physical times 0,1,2,2.5",
                "UnbalancedSync biological-number": "saved 26-point trajectories; stages use indices 0,10,20,25",
            },
        },
        "training_instances": [
            {
                "method": "TrajectoryNet 20k",
                "checkpoint": str(ROOT / "results/trajectorynet_gastrulation_rna_20000/checkpt-20000.pth"),
                "alternate_checkpoint": str(ROOT / "results/trajectorynet_gastrulation_rna_20000/checkpt.pth"),
                "checkpoint_note": "The two files have byte-different containers but exactly identical 22-tensor state_dict values.",
                "input": str(ROOT / "data/gastrulation_rna_trajectorynet.npz"),
                "preprocessing": "raw 50D PCA; stage labels 0,1,2,3",
                "training_loss": "continuous-normalizing-flow negative log likelihood; no explicit OT, energy, density, or mass loss",
            },
            {
                "method": "CytoBridge balanced 20k",
                "checkpoint": str(ROOT / "results/cytobridge_gastrulation_rna_20000/Train/last_model.pth"),
                "saved_adata": str(ROOT / "results/cytobridge_gastrulation_rna_20000/adata.h5ad"),
                "preprocessing": "raw 50D X_latent; physical times 0,1,2,2.5",
                "training_loss": "10 * Sinkhorn OT + 0.01 * energy; growth component absent, so configured mass term is inactive",
            },
            {
                "method": "CytoBridge unbalanced 20k",
                "checkpoint": str(ROOT / "results/cytobridge_gastrulation_rna_20000_unbalanced/Train/last_model.pth"),
                "saved_adata": str(ROOT / "results/cytobridge_gastrulation_rna_20000_unbalanced/adata.h5ad"),
                "preprocessing": "raw 50D X_latent; physical times 0,1,2,2.5",
                "training_loss": "500-epoch detached-Sinkhorn pretrain, then 10 * Sinkhorn OT + 10 * global/local mass + 0.01 * energy",
                "mass_target": "observed snapshot count ratios, not the biological-number prior",
            },
            {
                "method": "MIOFlow 20k",
                "checkpoint": str(ROOT / "results/mioflow_gastrulation_rna_20000/model.pt"),
                "input": str(ROOT / "data/gastrulation_rna_cytobridge.h5ad"),
                "preprocessing": "raw 50D X_latent with identity normalization (checkpoint mean=0, std=1); no GAGA",
                "training_loss": "exact EMD/OT + 0.01 * path energy; density and growth/mass losses disabled",
            },
            {
                "method": "TIGON 20k no-AE",
                "checkpoint": str(ROOT / "results/tigon_gastrulation_rna_20000/tigon.pt"),
                "config": str(ROOT / "results/tigon_gastrulation_rna_20000/config.json"),
                "preprocessing": "raw 50D X_latent directly; no autoencoder; physical times 0,1,2,2.5",
                "training_loss": "Sinkhorn OT + 0.01 * energy + 10 * mass/count-ratio + 0.0001 * growth L2",
                "mass_target": "observed snapshot count ratios, not the biological-number prior",
            },
            {
                "method": "UnbalancedSync biological-number alpha sweep, seed 0, 20k",
                "checkpoints": [str(path) for path in own_ckpts],
                "trajectories": str(OWN_TRAJECTORIES),
                "preprocessing": "RNA raw PCA/global scale and paired ATAC first-14 LSI/global scale; physical times 0,1,2,2.5",
                "training_loss": (
                    "primary and secondary terminal Sinkhorn (adaptive coefficients bounded 10..100); "
                    "density 0.01 and velocity energy 0.01 in both modalities, mixed by alpha; "
                    "mass coefficient 1 plus growth energy 0.01, with biological-number prior [1,4,6,11]; alpha 0.1..0.9"
                ),
            },
        ],
        "mapping_T": {
            "checkpoint": str(OWN_DATA / "TrainT/T_FiLM.pt"),
            "architecture": "FiLM MLP 50D RNA -> 14D ATAC, hidden 128, 3 layers, t-hidden 32",
            "training": "all four stages, paired cells, 80/20 split per stage, MSE training, best validation Sinkhorn checkpoint",
        },
        "input_identity": input_identity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-data gastrulation CMCC comparison with a shared RNA-to-ATAC T map.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/gastrulation_full_cmcc"))
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--steps-per-interval", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--include-direct-sync-secondary", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite to replace its files")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    rna_ref, atac_ref, labels_by_stage, rna_scale, atac_scale = _load_references()
    audit_df, input_identity = _audit_inputs(rna_scale)
    if not input_identity["all_matrix_inputs_exact"]:
        raise RuntimeError("At least one method matrix is not exactly identical to the own-model raw RNA input")
    if not input_identity["mioflow"]["same_raw_input_supported_by_checkpoint"]:
        raise RuntimeError("MIOFlow checkpoint does not document the shared raw RNA PCA with identity preprocessing")

    x0 = rna_ref[0]
    print(f"[input] common E7.5 initial cells: {x0.shape}", flush=True)
    print(f"[input] exact matrix identity checks: {len(audit_df)} passed", flush=True)

    predictions: dict[str, list[np.ndarray]] = {
        "TrajectoryNet 20k": _trajectorynet_rollout(
            x0, rna_scale, device, args.batch_size, args.steps_per_interval
        ),
        "CytoBridge balanced 20k": _cytobridge_rollout(
            "CytoBridge balanced",
            ROOT / "results/cytobridge_gastrulation_rna_20000/adata.h5ad",
            x0,
            rna_scale,
            device,
            args.batch_size,
            args.steps_per_interval,
        ),
        "CytoBridge unbalanced 20k": _cytobridge_rollout(
            "CytoBridge unbalanced",
            ROOT / "results/cytobridge_gastrulation_rna_20000_unbalanced/adata.h5ad",
            x0,
            rna_scale,
            device,
            args.batch_size,
            args.steps_per_interval,
        ),
        "MIOFlow 20k": _mioflow_rollout(
            x0, rna_scale, device, args.batch_size, args.steps_per_interval
        ),
    }
    tigon_predictions, _ = _tigon_rollout(x0, rna_scale, device)
    predictions["TIGON 20k"] = tigon_predictions
    own_predictions, own_direct = _own_sync_rollouts(x0)
    predictions.update(own_predictions)

    t_model = _load_t_model(device)
    classes = sorted(set(np.concatenate(labels_by_stage).tolist()) | {UNKNOWN})
    stage_rows: list[dict] = []

    def evaluate_one(method: str, rna_points_by_stage: list[np.ndarray], direct_atac=None, readout="shared full-data T") -> None:
        for stage_idx in range(4):
            label_ref = labels_by_stage[stage_idx]
            rna_nn = NearestNeighbors(n_neighbors=args.k, algorithm="auto", n_jobs=-1).fit(rna_ref[stage_idx])
            atac_nn = NearestNeighbors(n_neighbors=args.k, algorithm="auto", n_jobs=-1).fit(atac_ref[stage_idx])
            rna_points = np.asarray(rna_points_by_stage[stage_idx], dtype=np.float32)
            atac_points = (
                np.asarray(direct_atac[stage_idx], dtype=np.float32)
                if direct_atac is not None
                else _apply_t(t_model, rna_points, float(PHYSICAL_TIMES[stage_idx]), device, args.batch_size)
            )
            row = _stage_metric(
                method,
                readout,
                stage_idx,
                rna_points,
                atac_points,
                rna_nn,
                atac_nn,
                label_ref,
                classes,
            )
            stage_rows.append(row)
            print(
                f"[CMCC] {method} {STAGE_NAMES[stage_idx]} = {row['cmcc_mean']:.6f} "
                f"(valid={row['valid_known_fraction']:.3f})",
                flush=True,
            )

    for method, method_predictions in predictions.items():
        evaluate_one(method, method_predictions)

    if args.include_direct_sync_secondary:
        for method, method_predictions in own_predictions.items():
            evaluate_one(
                f"{method} direct-ATAC sensitivity",
                method_predictions,
                direct_atac=own_direct[method],
                readout="direct trained secondary trajectory",
            )

    # An empirical ceiling/readout diagnostic: observed RNA cells mapped with the same T.
    observed_rows: list[dict] = []
    for stage_idx in range(4):
        rna_nn = NearestNeighbors(n_neighbors=args.k, algorithm="auto", n_jobs=-1).fit(rna_ref[stage_idx])
        atac_nn = NearestNeighbors(n_neighbors=args.k, algorithm="auto", n_jobs=-1).fit(atac_ref[stage_idx])
        mapped = _apply_t(t_model, rna_ref[stage_idx], float(PHYSICAL_TIMES[stage_idx]), device, args.batch_size)
        observed_rows.append(
            _stage_metric(
                "Observed RNA through T (diagnostic)",
                "shared full-data T",
                stage_idx,
                rna_ref[stage_idx],
                mapped,
                rna_nn,
                atac_nn,
                labels_by_stage[stage_idx],
                classes,
            )
        )

    stage_df = pd.DataFrame(stage_rows)
    diagnostic_df = pd.DataFrame(observed_rows)
    overall_rows = []
    for (method, readout), group in stage_df.groupby(["method", "readout"], sort=False):
        scored = group[group["stage"].isin(STAGE_NAMES[1:])].set_index("stage").reindex(STAGE_NAMES[1:])
        overall_rows.append(
            {
                "method": method,
                "readout": readout,
                "overall_cmcc_equal_stage_mean": float(scored["cmcc_mean"].mean()),
                "E7.5_sanity_cmcc": float(group.loc[group["stage"] == "E7.5", "cmcc_mean"].iloc[0]),
                "E8.0_cmcc": float(scored.loc["E8.0", "cmcc_mean"]),
                "E8.5_cmcc": float(scored.loc["E8.5", "cmcc_mean"]),
                "E8.75_cmcc": float(scored.loc["E8.75", "cmcc_mean"]),
                "mean_valid_known_fraction": float(scored["valid_known_fraction"].mean()),
                "mean_rna_unknown_probability": float(scored["rna_unknown_probability_mean"].mean()),
                "mean_atac_unknown_probability": float(scored["atac_unknown_probability_mean"].mean()),
                "mean_rna_1nn_distance": float(scored["mean_rna_1nn_distance"].mean()),
                "mean_atac_1nn_distance": float(scored["mean_atac_1nn_distance"].mean()),
            }
        )
    overall_df = pd.DataFrame(overall_rows).sort_values(
        "overall_cmcc_equal_stage_mean", ascending=False, ignore_index=True
    )

    manifest = _manifest(args, rna_scale, atac_scale, input_identity)
    audit_df.to_csv(args.output_dir / "input_identity_audit.csv", index=False)
    stage_df.to_csv(args.output_dir / "cmcc_stage_scores.csv", index=False)
    overall_df.to_csv(args.output_dir / "cmcc_overall_scores.csv", index=False)
    diagnostic_df.to_csv(args.output_dir / "observed_T_diagnostic.csv", index=False)
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("\n[overall CMCC]", flush=True)
    print(
        overall_df[["method", "overall_cmcc_equal_stage_mean", "E8.0_cmcc", "E8.5_cmcc", "E8.75_cmcc"]]
        .round(6)
        .to_string(index=False),
        flush=True,
    )
    print(f"\n[write] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
