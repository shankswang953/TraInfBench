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
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

import h5py
import numpy as np
import pandas as pd
import torch
from geomloss import SamplesLoss
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from torchdiffeq import odeint


ROOT = Path(__file__).resolve().parents[2]
TRAINF_ROOT = Path("external/COATI")
GASTRULATION_ROOT = TRAINF_ROOT / "Gastrulation"
for path in [
    ROOT / "common",
    ROOT / "external" / "CytoBridge",
    ROOT / "external" / "MIOFlow",
    TRAINF_ROOT,
]:
    if path.exists():
        sys.path.insert(0, str(path))

from evaluate_gastrulation_flow_methods_normalized import (
    _cytobridge_interval_path,
    _cytobridge_velocity,
    _load_cytobridge_model,
    _load_mioflow_bundle,
    _mioflow_model_path_to_xnorm,
    _mioflow_xnorm_to_model_space,
)
from evaluate_gastrulation_full_cmcc import (
    PHYSICAL_TIMES,
    STAGE_KEYS,
    STAGE_NAMES,
    UNKNOWN,
    _load_references,
)
from evaluate_gastrulation_paired_knn_atac import (
    _composition,
    _evaluate_stage,
    _js_divergence_base2,
    _microcluster_distribution,
    _normalized_particle_weights,
    _soft_knn_transfer,
)
from evaluate_tigon_gastrulation_normalized import (
    _infer_model_input_space,
    _integrate_normalized,
    _load_model as _load_tigon_model,
    _load_tigon_module,
)
from evaluate_terminal_push import (
    _load_trajectorynet_model,
    _trajectorynet_diffeq,
)
from src.Neural import MLPVectorField


HELDOUT_STAGE_IDX = 1
HELDOUT_TIME = 1.0
BSOT_CHECKPOINTS = GASTRULATION_ROOT / "FilmSyncLOO" / "checkpoint"
USOT_LOCAL_CHECKPOINTS = (
    GASTRULATION_ROOT
    / "unbalanced_LOO_result"
    / "time1"
    / "Sync_bio_num"
    / "checkpoint"
)
USOT_EXTRA_CHECKPOINTS = ROOT / "results" / "usot_loo_seed0_20000_checkpoints"
TRAJECTORYNET_LOO_DATASET = ROOT / "data" / "gastrulation_rna_loo_time1_trajectorynet.npz"
TRAJECTORYNET_LOO_CHECKPOINT = (
    ROOT / "results" / "trajectorynet_gastrulation_loo_time1_20000" / "checkpt-20000.pth"
)


def _decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values],
        dtype=str,
    )


def _read_categorical(group: h5py.Group) -> np.ndarray:
    categories = _decode_strings(np.asarray(group["categories"]))
    codes = np.asarray(group["codes"], dtype=np.int64)
    values = np.full(codes.shape, UNKNOWN, dtype=object)
    known = codes >= 0
    values[known] = categories[codes[known]]
    return np.asarray(values, dtype=str)


def _load_separate_modality_labels(
    shared_labels_by_stage: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    data_root = GASTRULATION_ROOT / "data"
    records = {}
    for modality, filename in {
        "RNA": "gastrulation_rna_processed.h5ad",
        "ATAC": "gastrulation_atac_processed.h5ad",
    }.items():
        path = data_root / filename
        with h5py.File(path, "r") as handle:
            obs = handle["obs"]
            records[modality] = {
                "path": str(path),
                "cell_ids": _decode_strings(np.asarray(obs["_index"])),
                "times": _read_categorical(obs["processed_time"]).astype(float),
                "labels": _read_categorical(obs["celltype"]),
            }

    rna = records["RNA"]
    atac = records["ATAC"]
    if not np.array_equal(rna["cell_ids"], atac["cell_ids"]):
        raise ValueError("Processed RNA and ATAC cell IDs are not exactly paired and ordered")
    if not np.array_equal(rna["times"], atac["times"]):
        raise ValueError("Processed RNA and ATAC time labels differ")

    labels_by_modality: dict[str, list[np.ndarray]] = {"RNA": [], "ATAC": []}
    shared_matches = {}
    for stage_idx, physical_time in enumerate(PHYSICAL_TIMES):
        for modality in ("RNA", "ATAC"):
            record = records[modality]
            labels = record["labels"][np.isclose(record["times"], physical_time)]
            if labels.shape != shared_labels_by_stage[stage_idx].shape:
                raise ValueError(
                    f"{modality} {STAGE_NAMES[stage_idx]} label count {labels.shape[0]} "
                    f"does not match shared reference {shared_labels_by_stage[stage_idx].shape[0]}"
                )
            labels_by_modality[modality].append(labels)
            shared_matches[f"{modality}_{STAGE_KEYS[stage_idx]}"] = bool(
                np.array_equal(labels, shared_labels_by_stage[stage_idx])
            )

    label_mismatches = int(np.sum(rna["labels"] != atac["labels"]))
    audit = {
        "rna_source": rna["path"],
        "atac_source": atac["path"],
        "n_paired_cells": int(rna["cell_ids"].shape[0]),
        "cell_ids_exact": True,
        "times_exact": True,
        "rna_atac_celltype_mismatches": label_mismatches,
        "rna_atac_celltypes_exact": label_mismatches == 0,
        "shared_npz_label_matches": shared_matches,
    }
    return labels_by_modality["RNA"], labels_by_modality["ATAC"], audit


def _c_y(path: Path) -> float:
    match = re.search(r"_a([0-9.]+)_iter20000\.pth$", path.name)
    if match is None:
        raise ValueError(f"Could not parse C_y from {path}")
    return float(match.group(1))


def _expected_sweep(paths: list[Path], method: str) -> list[Path]:
    paths = sorted(paths, key=_c_y)
    observed = np.asarray([_c_y(path) for path in paths], dtype=float)
    expected = np.arange(0.1, 1.0, 0.1)
    if len(paths) != len(expected) or not np.allclose(observed, expected, rtol=0.0, atol=1e-12):
        raise FileNotFoundError(
            f"Expected {method} C_y=0.1..0.9, found {observed.tolist()}"
        )
    return paths


def _bsot_checkpoints() -> list[Path]:
    return _expected_sweep(
        list(BSOT_CHECKPOINTS.glob("ckpt_s0_e0.1_m100.0_d0.1_a*_iter20000.pth")),
        "BSOT",
    )


def _usot_checkpoints() -> list[Path]:
    candidates = list(
        USOT_LOCAL_CHECKPOINTS.glob("ckpt_s0_e0.01_m100.0_d0.01_a*_iter20000.pth")
    )
    candidates.extend(
        USOT_EXTRA_CHECKPOINTS.glob("ckpt_s0_e0.01_m100.0_d0.01_a*_iter20000.pth")
    )
    by_c_y = {_c_y(path): path for path in candidates}
    return _expected_sweep(list(by_c_y.values()), "USOT")


def _load_own_model(checkpoint_path: Path, dim: int, unbalanced: bool, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if int(checkpoint.get("iteration", -1)) != 20_000:
        raise ValueError(
            f"Expected iteration=20000 in {checkpoint_path}, got {checkpoint.get('iteration')}"
        )
    model = MLPVectorField(
        dim=dim,
        hidden_dim=800,
        n_layers=3,
        activation="leaky_relu",
        unbalanced=unbalanced,
        alpha_growth=1.0 if unbalanced else 0.0,
    ).to(device)
    model.load_state_dict(checkpoint["func_state_dict"])
    model.eval().requires_grad_(False)
    return model


def _bsot_prediction(
    x0_norm: np.ndarray,
    checkpoint_path: Path,
    device: torch.device,
) -> np.ndarray:
    model = _load_own_model(checkpoint_path, x0_norm.shape[1], False, device)
    x0 = torch.as_tensor(x0_norm, dtype=torch.float32, device=device)
    e0 = torch.zeros(x0.shape[0], 1, dtype=x0.dtype, device=device)
    times = torch.tensor([0.0, HELDOUT_TIME], dtype=x0.dtype, device=device)
    with torch.no_grad():
        positions, _ = odeint(
            model,
            (x0, e0),
            times,
            method="rk4",
            options={"step_size": 0.1},
        )
    return positions[-1].detach().cpu().numpy().astype(np.float32, copy=False)


def _usot_prediction(
    x0_norm: np.ndarray,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model = _load_own_model(checkpoint_path, x0_norm.shape[1], True, device)
    x0 = torch.as_tensor(x0_norm, dtype=torch.float32, device=device)
    n = x0.shape[0]
    log_mass0 = torch.full(
        (n, 1),
        -float(np.log(n)),
        dtype=x0.dtype,
        device=device,
    )
    e0 = torch.zeros(n, 1, dtype=x0.dtype, device=device)
    times = torch.tensor([0.0, HELDOUT_TIME], dtype=x0.dtype, device=device)
    with torch.no_grad():
        positions, log_mass, _, _ = odeint(
            model,
            (x0, log_mass0, e0, e0),
            times,
            method="rk4",
            options={"step_size": 0.1},
        )
    return (
        positions[-1].detach().cpu().numpy().astype(np.float32, copy=False),
        log_mass[-1, :, 0].detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _cytobridge_balanced_prediction(
    x0_norm: np.ndarray,
    scale: float,
    device: torch.device,
    batch_size: int,
    steps: int,
) -> np.ndarray:
    path = ROOT / "results" / "cytobridge_gastrulation_loo_time1_20000" / "adata.h5ad"
    model = _load_cytobridge_model(path, device)
    prediction = _cytobridge_interval_path(
        model,
        x0_norm,
        0.0,
        HELDOUT_TIME,
        steps,
        scale,
        batch_size,
        device,
    )[-1]
    return prediction.astype(np.float32, copy=False)


def _cytobridge_unbalanced_prediction(
    x0_norm: np.ndarray,
    scale: float,
    device: torch.device,
    batch_size: int,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    path = (
        ROOT
        / "results"
        / "cytobridge_gastrulation_loo_time1_20000_unbalanced"
        / "adata.h5ad"
    )
    model = _load_cytobridge_model(path, device)
    if "growth" not in model.components or not hasattr(model, "growth_net"):
        raise ValueError("CytoBridge unbalanced checkpoint has no growth network")

    dt = HELDOUT_TIME / float(steps)
    position_batches = []
    log_mass_batches = []
    initial_log_mass = -float(np.log(x0_norm.shape[0]))
    for start in range(0, x0_norm.shape[0], batch_size):
        stop = min(start + batch_size, x0_norm.shape[0])
        x = torch.as_tensor(x0_norm[start:stop], dtype=torch.float32, device=device)
        log_mass = torch.full(
            (x.shape[0], 1),
            initial_log_mass,
            dtype=x.dtype,
            device=device,
        )
        with torch.no_grad():
            for step in range(steps):
                t_mid = torch.tensor(
                    [(step + 0.5) * dt],
                    dtype=torch.float32,
                    device=device,
                )
                x_raw = x * scale
                velocity_raw = _cytobridge_velocity(model, t_mid, x_raw)
                growth = model.growth_net(
                    torch.cat([x_raw, t_mid.expand(x.shape[0], 1)], dim=1)
                )
                x = x + dt * (velocity_raw / scale)
                log_mass = log_mass + dt * growth
        position_batches.append(x.detach().cpu().numpy())
        log_mass_batches.append(log_mass[:, 0].detach().cpu().numpy())
    return (
        np.concatenate(position_batches).astype(np.float32, copy=False),
        np.concatenate(log_mass_batches).astype(np.float32, copy=False),
    )


def _tigon_prediction(
    x0_norm: np.ndarray,
    scale: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    result_dir = ROOT / "results" / "tigon_gastrulation_loo_time1_20000"
    module = _load_tigon_module()
    model, config, _ = _load_tigon_model(
        module,
        result_dir / "tigon.pt",
        result_dir / "config.json",
        device,
    )
    model_input_space = _infer_model_input_space(config, "auto")
    x = torch.as_tensor(x0_norm, dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction, log_growth, _, _ = _integrate_normalized(
            model,
            x,
            0.0,
            HELDOUT_TIME,
            int(config.get("ode_steps", 8)),
            scale,
            model_input_space,
        )
    initial_log_mass = -float(np.log(x0_norm.shape[0]))
    return (
        prediction.detach().cpu().numpy().astype(np.float32, copy=False),
        (log_growth.detach().cpu().numpy() + initial_log_mass).astype(
            np.float32, copy=False
        ),
    )


def _mioflow_prediction(
    x0_norm: np.ndarray,
    scale: float,
    device: torch.device,
    batch_size: int,
    steps: int,
) -> np.ndarray:
    model_path = ROOT / "results" / "mioflow_gastrulation_loo_time1_20000" / "model.pt"
    bundle = _load_mioflow_bundle(model_path, device)
    if not np.allclose(bundle.model_times, np.asarray([0.0, 1.0, 2.0])):
        raise ValueError(f"Unexpected MIOFlow LOO times: {bundle.model_times.tolist()}")

    # MIOFlow factorizes the observed [E7.5, E8.5, E8.75] stages to [0, 1, 2].
    # E8.0 is halfway between E7.5 and E8.5, so its native model time is 0.5.
    z0 = _mioflow_xnorm_to_model_space(x0_norm, bundle, scale, device, batch_size)
    times = torch.linspace(0.0, 0.5, steps + 1, dtype=torch.float32, device=device)
    paths = []
    for start in range(0, z0.shape[0], batch_size):
        stop = min(start + batch_size, z0.shape[0])
        batch = torch.as_tensor(z0[start:stop], dtype=torch.float32, device=device)
        if hasattr(bundle.model, "reset_momentum"):
            bundle.model.reset_momentum()
        with torch.no_grad():
            path = odeint(bundle.model, batch, times)
        paths.append(path.detach().cpu().numpy())
    model_path_array = np.concatenate(paths, axis=1).astype(np.float32, copy=False)
    return _mioflow_model_path_to_xnorm(
        model_path_array,
        bundle,
        scale,
        device,
        batch_size,
    )[-1].astype(np.float32, copy=False)


def _trajectorynet_prediction(
    x0_norm: np.ndarray,
    scale: float,
    device: torch.device,
    batch_size: int,
    steps: int,
) -> tuple[np.ndarray, dict]:
    """Evaluate the existing LOO checkpoint on its actual as-trained solver clock."""
    data = np.load(TRAJECTORYNET_LOO_DATASET, allow_pickle=True)
    labels = np.unique(np.asarray(data["sample_labels"], dtype=float))
    expected_labels = np.asarray([0.0, 2.0, 2.5])
    if not np.array_equal(labels, expected_labels):
        raise ValueError(
            f"Unexpected TrajectoryNet LOO labels {labels.tolist()}; "
            f"expected {expected_labels.tolist()}"
        )

    time_scale = 0.5
    int_tps = (np.arange(max(labels) + 1) + 1.0) * time_scale
    if int_tps.shape[0] < labels.shape[0]:
        raise ValueError(
            f"TrajectoryNet grid has {int_tps.shape[0]} endpoints for {labels.shape[0]} labels"
        )
    # The upstream reverse zip pairs sorted labels with the final len(labels)
    # endpoints of int_tps. For [0,2,2.5], this is [1.0,1.5,2.0].
    assigned_endpoints = int_tps[-labels.shape[0] :]
    label_to_endpoint = {
        float(label): float(endpoint)
        for label, endpoint in zip(labels, assigned_endpoints)
    }
    source_physical = 0.0
    next_physical = 2.0
    fraction = (HELDOUT_TIME - source_physical) / (next_physical - source_physical)
    source_endpoint = label_to_endpoint[source_physical]
    next_endpoint = label_to_endpoint[next_physical]
    heldout_endpoint = next_endpoint - fraction * (
        next_endpoint - source_endpoint
    )
    if not np.isclose(heldout_endpoint, 1.25):
        raise ValueError(f"Unexpected TrajectoryNet held-out solver time {heldout_endpoint}")

    model, _ = _load_trajectorynet_model(
        TRAJECTORYNET_LOO_CHECKPOINT,
        x0_norm.shape[1],
        device,
        "rk4",
        0.1,
    )
    raw_diffeq = _trajectorynet_diffeq(model)

    def normalized_diffeq(t_value: torch.Tensor, y_norm: torch.Tensor) -> torch.Tensor:
        return raw_diffeq(t_value, y_norm * scale) / scale

    # The source snapshot is the output at the upper end of the *next*
    # interval. A fractional biological-forward prediction therefore starts
    # at next_endpoint and traverses only the requested prefix in reverse.
    times = torch.linspace(
        next_endpoint,
        heldout_endpoint,
        steps + 1,
        dtype=torch.float32,
        device=device,
    )
    batches = []
    for start in range(0, x0_norm.shape[0], batch_size):
        stop = min(start + batch_size, x0_norm.shape[0])
        batch = torch.as_tensor(
            x0_norm[start:stop], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            path = odeint(
                normalized_diffeq,
                batch,
                times,
                method="rk4",
                options={"step_size": 0.1},
            )
        batches.append(path[-1].detach().cpu().numpy())
    prediction = np.concatenate(batches).astype(np.float32, copy=False)
    clock = {
        "input_labels": labels.tolist(),
        "upstream_uniform_grid": int_tps.tolist(),
        "label_to_solver_endpoint": {
            str(label): endpoint for label, endpoint in label_to_endpoint.items()
        },
        "heldout_physical_time": HELDOUT_TIME,
        "heldout_solver_time": float(heldout_endpoint),
        "source_state_solver_time": float(next_endpoint),
        "lower_interval_boundary": float(source_endpoint),
        "inference_direction": (
            "next_interval_upper_endpoint_to_fractional_reverse_endpoint"
        ),
    }
    return prediction, clock


def _prepare_heldout_reference(
    rna: np.ndarray,
    atac: np.ndarray,
    sinkhorn_clusters: int,
) -> dict:
    neighbors = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1).fit(rna)
    loo_neighbors = NearestNeighbors(n_neighbors=2, algorithm="auto", n_jobs=-1).fit(rna)
    loo_distances = loo_neighbors.kneighbors(rna, return_distance=True)[0][:, 1]
    uniform = np.full(rna.shape[0], 1.0 / rna.shape[0], dtype=np.float64)
    rna_weights, rna_centers, rna_inertia = _microcluster_distribution(
        rna,
        uniform,
        sinkhorn_clusters,
    )
    n_clusters = min(int(sinkhorn_clusters), atac.shape[0])
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=0,
        batch_size=2048,
        n_init=3,
        max_iter=200,
    )
    cluster_ids = kmeans.fit_predict(atac)
    return {
        "neighbors": neighbors,
        "reference_loo_1nn_median": float(np.median(loo_distances)),
        "rna_cluster_weights": rna_weights,
        "rna_cluster_centers": rna_centers,
        "rna_kmeans_inertia": rna_inertia,
        "cluster_ids": cluster_ids.astype(np.int64, copy=False),
        "cluster_centers": np.asarray(kmeans.cluster_centers_, dtype=np.float32),
        "kmeans_inertia": float(kmeans.inertia_),
    }


def _evaluate_modality_celltype_compositions(
    method: str,
    points: np.ndarray,
    rna_reference: np.ndarray,
    rna_labels: np.ndarray,
    atac_labels: np.ndarray,
    prepared: dict,
    classes: list[str],
    query_log_mass: np.ndarray | None,
) -> tuple[dict, list[dict]]:
    query_weights, _, _, _ = _normalized_particle_weights(
        points.shape[0],
        query_log_mass,
    )
    transferred, _, _, _ = _soft_knn_transfer(
        prepared["neighbors"],
        np.asarray(points, dtype=np.float32),
        rna_reference.shape[0],
        query_weights,
    )
    observed = np.full(
        rna_reference.shape[0],
        1.0 / rna_reference.shape[0],
        dtype=np.float64,
    )
    unknown_idx = classes.index(UNKNOWN)
    known_indices = np.asarray(
        [class_idx for class_idx in range(len(classes)) if class_idx != unknown_idx],
        dtype=int,
    )

    metrics = {}
    rows = []
    for modality, labels in (("rna", rna_labels), ("atac", atac_labels)):
        transferred_composition = _composition(transferred, labels, classes)
        observed_composition = _composition(observed, labels, classes)
        composition_jsd = _js_divergence_base2(
            transferred_composition[known_indices],
            observed_composition[known_indices],
        )
        metrics[f"{modality}_cjs_score"] = float(1.0 - composition_jsd)
        metrics[f"{modality}_celltype_composition_jsd"] = composition_jsd
        metrics[f"{modality}_transferred_unknown_mass"] = float(
            transferred_composition[unknown_idx]
        )
        metrics[f"{modality}_observed_unknown_fraction"] = float(
            observed_composition[unknown_idx]
        )
        rows.extend(
            {
                "method": method,
                "stage": STAGE_NAMES[HELDOUT_STAGE_IDX],
                "modality": modality.upper(),
                "celltype": celltype,
                "transferred_fraction": float(transferred_composition[class_idx]),
                "observed_fraction": float(observed_composition[class_idx]),
                "difference": float(
                    transferred_composition[class_idx] - observed_composition[class_idx]
                ),
                "absolute_difference": float(
                    abs(
                        transferred_composition[class_idx]
                        - observed_composition[class_idx]
                    )
                ),
            }
            for class_idx, celltype in enumerate(classes)
        )
    return metrics, rows


def _build_predictions(
    x0_norm: np.ndarray,
    scale: float,
    device: torch.device,
    batch_size: int,
    steps: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict]]:
    predictions: dict[str, np.ndarray] = {}
    log_masses: dict[str, np.ndarray] = {}
    provenance: dict[str, dict] = {}

    for checkpoint in _bsot_checkpoints():
        value = _c_y(checkpoint)
        name = f"BSOT C_y={value:.1f}"
        predictions[name] = _bsot_prediction(x0_norm, checkpoint, device)
        provenance[name] = {"checkpoint": str(checkpoint), "weighting": "uniform"}
        print(f"[prediction] {name}", flush=True)

    for checkpoint in _usot_checkpoints():
        value = _c_y(checkpoint)
        name = f"USOT C_y={value:.1f}"
        predictions[name], log_masses[name] = _usot_prediction(
            x0_norm, checkpoint, device
        )
        provenance[name] = {"checkpoint": str(checkpoint), "weighting": "native_mass"}
        print(f"[prediction] {name}", flush=True)

    name = "CytoBridge balanced 20k"
    predictions[name] = _cytobridge_balanced_prediction(
        x0_norm, scale, device, batch_size, steps
    )
    provenance[name] = {
        "checkpoint": str(
            ROOT
            / "results/cytobridge_gastrulation_loo_time1_20000/Train/last_model.pth"
        ),
        "weighting": "uniform",
    }
    print(f"[prediction] {name}", flush=True)

    name = "CytoBridge unbalanced 20k"
    predictions[name], log_masses[name] = _cytobridge_unbalanced_prediction(
        x0_norm, scale, device, batch_size, steps
    )
    provenance[name] = {
        "checkpoint": str(
            ROOT
            / "results/cytobridge_gastrulation_loo_time1_20000_unbalanced/Train/last_model.pth"
        ),
        "weighting": "native_mass",
    }
    print(f"[prediction] {name}", flush=True)

    name = "TIGON 20k"
    predictions[name], log_masses[name] = _tigon_prediction(x0_norm, scale, device)
    provenance[name] = {
        "checkpoint": str(ROOT / "results/tigon_gastrulation_loo_time1_20000/tigon.pt"),
        "weighting": "native_mass",
    }
    print(f"[prediction] {name}", flush=True)

    name = "MIOFlow 20k"
    predictions[name] = _mioflow_prediction(
        x0_norm, scale, device, batch_size, steps
    )
    provenance[name] = {
        "checkpoint": str(ROOT / "results/mioflow_gastrulation_loo_time1_20000/model.pt"),
        "weighting": "uniform",
        "heldout_model_time": 0.5,
    }
    print(f"[prediction] {name}", flush=True)

    name = "TrajectoryNet 20k"
    predictions[name], trajectorynet_clock = _trajectorynet_prediction(
        x0_norm, scale, device, batch_size, steps
    )
    provenance[name] = {
        "checkpoint": str(TRAJECTORYNET_LOO_CHECKPOINT),
        "dataset": str(TRAJECTORYNET_LOO_DATASET),
        "weighting": "uniform",
        "clock": trajectorynet_clock,
    }
    print(f"[prediction] {name}", flush=True)
    return predictions, log_masses, provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate gastrulation time1 LOO predictions by cell-type composition and "
            "paired RNA-to-ATAC 1NN transfer."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/gastrulation_loo_time1_paired_1nn_atac"),
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
    rna_ref, atac_ref, labels_by_stage, rna_scale, atac_scale = _load_references()
    rna_labels_by_stage, atac_labels_by_stage, label_audit = (
        _load_separate_modality_labels(labels_by_stage)
    )
    x0 = rna_ref[0]
    heldout_rna = rna_ref[HELDOUT_STAGE_IDX]
    heldout_atac = atac_ref[HELDOUT_STAGE_IDX]
    heldout_rna_labels = rna_labels_by_stage[HELDOUT_STAGE_IDX]
    heldout_atac_labels = atac_labels_by_stage[HELDOUT_STAGE_IDX]
    if (
        heldout_rna.shape[0] != heldout_atac.shape[0]
        or heldout_rna_labels.shape[0] != heldout_rna.shape[0]
        or heldout_atac_labels.shape[0] != heldout_rna.shape[0]
    ):
        raise ValueError(
            "E8.0 paired RNA/ATAC/RNA-label/ATAC-label reference lengths do not match: "
            f"{heldout_rna.shape[0]}, {heldout_atac.shape[0]}, "
            f"{heldout_rna_labels.shape[0]}, {heldout_atac_labels.shape[0]}"
        )

    predictions, log_masses, provenance = _build_predictions(
        x0,
        rna_scale,
        device,
        args.batch_size,
        args.steps,
    )
    prepared = _prepare_heldout_reference(
        heldout_rna,
        heldout_atac,
        args.sinkhorn_clusters,
    )
    classes = sorted(
        set(np.concatenate(rna_labels_by_stage).tolist())
        | set(np.concatenate(atac_labels_by_stage).tolist())
        | {UNKNOWN}
    )
    sinkhorn = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=args.sinkhorn_blur,
        debias=True,
        backend="tensorized",
    )

    diagnostic, _ = _evaluate_stage(
        "Observed E8.0 self-transfer",
        HELDOUT_STAGE_IDX,
        heldout_rna,
        heldout_rna,
        heldout_rna_labels,
        prepared,
        classes,
        sinkhorn,
        query_log_mass=None,
        query_weighting="uniform",
    )
    diagnostic.pop("atac_cjs_score")
    diagnostic.pop("atac_composition_jsd")
    diagnostic.pop("transferred_unknown_mass")
    diagnostic.pop("observed_unknown_fraction")
    diagnostic_celltype, _ = _evaluate_modality_celltype_compositions(
        "Observed E8.0 self-transfer",
        heldout_rna,
        heldout_rna,
        heldout_rna_labels,
        heldout_atac_labels,
        prepared,
        classes,
        query_log_mass=None,
    )
    diagnostic.update(diagnostic_celltype)

    summaries = []
    composition_rows = []
    for method, prediction in predictions.items():
        native_mass = log_masses.get(method)
        summary, _ = _evaluate_stage(
            method,
            HELDOUT_STAGE_IDX,
            prediction,
            heldout_rna,
            heldout_rna_labels,
            prepared,
            classes,
            sinkhorn,
            query_log_mass=native_mass,
            query_weighting="native_mass" if native_mass is not None else "uniform",
        )
        summary.pop("atac_cjs_score")
        summary.pop("atac_composition_jsd")
        summary.pop("transferred_unknown_mass")
        summary.pop("observed_unknown_fraction")
        celltype_metrics, rows = _evaluate_modality_celltype_compositions(
            method,
            prediction,
            heldout_rna,
            heldout_rna_labels,
            heldout_atac_labels,
            prepared,
            classes,
            query_log_mass=native_mass,
        )
        summary.update(celltype_metrics)
        summary["effective_coverage"] = summary["atac_coverage_fraction"]
        summary["atac_sinkhorn_divergence"] = summary["atac_sinkhorn_microcluster"]
        summaries.append(summary)
        composition_rows.extend(rows)
        print(
            f"[LOO E8.0] {method}: "
            f"RNA-JSD={summary['rna_celltype_composition_jsd']:.6f} "
            f"ATAC-JSD={summary['atac_celltype_composition_jsd']:.6f} "
            f"coverage={summary['effective_coverage']:.6f} "
            f"ATAC-Sinkhorn={summary['atac_sinkhorn_divergence']:.6f}",
            flush=True,
        )

    scores = pd.DataFrame(summaries).sort_values(
        [
            "rna_celltype_composition_jsd",
            "atac_celltype_composition_jsd",
            "atac_sinkhorn_divergence",
        ],
        ascending=[True, True, True],
        ignore_index=True,
    )
    composition = pd.DataFrame(composition_rows)
    scores.to_csv(args.output_dir / "loo_time1_scores.csv", index=False)
    composition.to_csv(args.output_dir / "loo_time1_celltype_composition.csv", index=False)
    pd.DataFrame([diagnostic]).to_csv(
        args.output_dir / "loo_time1_reference_self_diagnostic.csv",
        index=False,
    )

    manifest = {
        "task": "Gastrulation LOO time1/E8.0",
        "stage": STAGE_NAMES[HELDOUT_STAGE_IDX],
        "physical_time": HELDOUT_TIME,
        "training_physical_times": [0.0, 2.0, 2.5],
        "n_initial": int(x0.shape[0]),
        "n_heldout_paired_reference": int(heldout_rna.shape[0]),
        "rna_dim": int(heldout_rna.shape[1]),
        "atac_dim": int(heldout_atac.shape[1]),
        "protocol": {
            "mapping": (
                "Each predicted RNA particle transfers all normalized mass to its 1NN real "
                "E8.0 RNA cell and therefore to that cell's paired real ATAC cell."
            ),
            "rna_composition": (
                "Base-2 JSD between the E8.0 RNA cell-type composition induced by RNA 1NN "
                "assignment and the observed E8.0 RNA composition."
            ),
            "atac_composition": (
                "Base-2 JSD after transferring the same 1NN weights to paired E8.0 ATAC "
                "cells and comparing against the observed E8.0 ATAC composition. Unknown "
                "is reported, excluded, and remaining types are renormalized in both modalities."
            ),
            "paired_composition_consequence": (
                "The processed RNA and ATAC references contain the same paired cells and "
                "identical cell-type annotations. Under 1NN paired-cell transfer, RNA-JSD "
                "and ATAC-JSD are therefore equal by construction; they are stored separately "
                "to make the modality semantics explicit."
            ),
            "effective_coverage": "(1 / sum_i q_i^2) / n_E8.0 after 1NN transfer.",
            "atac_sinkhorn": (
                "Debiased p=2 weighted Sinkhorn divergence on fixed real-ATAC "
                "MiniBatchKMeans microclusters."
            ),
            "sinkhorn_clusters": int(args.sinkhorn_clusters),
            "sinkhorn_blur": float(args.sinkhorn_blur),
            "mass": (
                "USOT, CytoBridge unbalanced, and TIGON use normalized native particle "
                "mass; balanced methods and MIOFlow use uniform particle mass."
            ),
            "mioflow_time": (
                "Observed LOO stages are factorized to [0,1,2]; E8.0 is evaluated at "
                "native model time 0.5, halfway from E7.5 to E8.5."
            ),
            "trajectorynet": (
                "The existing checkpoint labels [0,2,2.5] were paired by the upstream "
                "reverse-zip implementation with solver endpoints [1.0,1.5,2.0]. E8.0 "
                "was evaluated at solver time 1.25, the physically calibrated midpoint "
                "between E7.5 and E8.5."
            ),
        },
        "modality_label_audit": label_audit,
        "scales": {"rna": rna_scale, "atac": atac_scale},
        "methods": provenance,
    }
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n[LOO E8.0 ranking by RNA and ATAC cell-type JSD]", flush=True)
    print(
        scores[
            [
                "method",
                "rna_celltype_composition_jsd",
                "atac_celltype_composition_jsd",
                "effective_coverage",
                "atac_sinkhorn_divergence",
                "rna_projection_distance_ratio",
            ]
        ].round(6).to_string(index=False),
        flush=True,
    )
    print(f"\n[write] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
