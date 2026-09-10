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
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from torchdiffeq import odeint as torchdiffeq_odeint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

from evaluate_gastrulation_full_cmcc import (
    PHYSICAL_TIMES,
    STAGE_KEYS,
    STAGE_NAMES,
    UNKNOWN,
    _audit_inputs,
    _cytobridge_rollout,
    _load_references,
    _mioflow_rollout,
    _own_sync_rollouts,
    _tigon_rollout,
    _trajectorynet_rollout,
)
from evaluate_gastrulation_flow_methods_normalized import (
    _cytobridge_velocity,
    _load_cytobridge_model,
)


TRAINF_ROOT = Path("external/COATI")
FILMSYNC_ROOT = TRAINF_ROOT / "Gastrulation/FilmSync"
FILMSYNC_SEED0_20K_CHECKPOINTS = (
    ROOT / "results/filmsync_balanced_seed0_20000/checkpoints"
)
UNBALANCED_SYNC_20K_CHECKPOINTS = (
    TRAINF_ROOT
    / "Gastrulation/UnbalancedSync_biological_num/checkpoint_hpc_iter20000"
)


def _js_divergence_base2(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_p = np.where(p > 0, p * np.log2(p / midpoint), 0.0).sum()
        kl_q = np.where(q > 0, q * np.log2(q / midpoint), 0.0).sum()
    return float(0.5 * (kl_p + kl_q))


def _soft_knn_transfer(
    neighbors: NearestNeighbors,
    points: np.ndarray,
    n_reference: int,
    query_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if query_weights is None:
        query_weights = np.full(points.shape[0], 1.0 / points.shape[0], dtype=np.float64)
    else:
        query_weights = np.asarray(query_weights, dtype=np.float64).reshape(-1)
        if query_weights.shape[0] != points.shape[0]:
            raise ValueError(
                f"Expected {points.shape[0]} query weights, got {query_weights.shape[0]}"
            )
        if not np.isfinite(query_weights).all() or np.any(query_weights < 0):
            raise ValueError("Query weights must be finite and nonnegative")
        weight_sum = float(query_weights.sum())
        if weight_sum <= 0:
            raise ValueError("Query weights must have positive total mass")
        query_weights = query_weights / weight_sum
    distances, indices = neighbors.kneighbors(points, return_distance=True)
    bandwidth = np.maximum(distances[:, -1], 1e-12)
    weights = np.exp(-np.square(distances / bandwidth[:, None]))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-300)
    transferred = np.zeros(n_reference, dtype=np.float64)
    np.add.at(
        transferred,
        indices.reshape(-1),
        (weights * query_weights[:, None]).reshape(-1),
    )
    transferred /= transferred.sum()
    return transferred, distances[:, 0], indices, weights


def _normalized_particle_weights(
    n_query: int,
    log_mass: np.ndarray | None,
) -> tuple[np.ndarray, float, float, float]:
    if log_mass is None:
        weights = np.full(n_query, 1.0 / n_query, dtype=np.float64)
        return weights, 1.0, float(n_query), float(1.0 / n_query)

    values = np.asarray(log_mass, dtype=np.float64).reshape(-1)
    if values.shape[0] != n_query:
        raise ValueError(f"Expected {n_query} log masses, got {values.shape[0]}")
    if not np.isfinite(values).all():
        raise ValueError("Particle log masses must be finite")
    max_log_mass = float(values.max())
    relative = np.exp(values - max_log_mass)
    relative_sum = float(relative.sum())
    weights = relative / relative_sum
    log_total_mass = max_log_mass + np.log(relative_sum)
    total_mass = float(np.exp(log_total_mass))
    effective_particles = float(1.0 / np.sum(np.square(weights)))
    return weights, total_mass, effective_particles, float(weights.max())


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape != weights.shape:
        raise ValueError(f"Weighted median shape mismatch: {values.shape} vs {weights.shape}")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(sorted_values[np.searchsorted(cumulative, 0.5, side="left")])


def _composition(
    cell_weights: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
) -> np.ndarray:
    return np.asarray(
        [cell_weights[labels == celltype].sum() for celltype in classes],
        dtype=np.float64,
    )


def _weighted_microcluster_sinkhorn(
    transferred: np.ndarray,
    cluster_ids: np.ndarray,
    centers: np.ndarray,
    sinkhorn: SamplesLoss,
) -> float:
    n_clusters = centers.shape[0]
    q = np.bincount(cluster_ids, weights=transferred, minlength=n_clusters).astype(np.float32)
    p = np.bincount(cluster_ids, minlength=n_clusters).astype(np.float32)
    p /= p.sum()
    q = np.maximum(q, 1e-12)
    p = np.maximum(p, 1e-12)
    q /= q.sum()
    p /= p.sum()
    center_tensor = torch.as_tensor(centers, dtype=torch.float32)
    with torch.no_grad():
        value = sinkhorn(
            torch.as_tensor(q),
            center_tensor,
            torch.as_tensor(p),
            center_tensor,
        )
    return float(value.detach().cpu())


def _microcluster_distribution(
    points: np.ndarray,
    weights: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    points = np.asarray(points, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or weights.shape[0] != points.shape[0]:
        raise ValueError(
            f"Microcluster shape mismatch: points={points.shape}, weights={weights.shape}"
        )
    if not np.isfinite(points).all() or not np.isfinite(weights).all():
        raise ValueError("Microcluster inputs must be finite")
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("Microcluster weights must be nonnegative with positive total mass")
    weights = weights / weights.sum()
    n_clusters = min(int(n_clusters), points.shape[0])
    if n_clusters == points.shape[0]:
        return weights.astype(np.float32), points, 0.0

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=0,
        batch_size=2048,
        n_init=3,
        max_iter=200,
    )
    cluster_ids = kmeans.fit_predict(points, sample_weight=weights)
    cluster_weights = np.bincount(
        cluster_ids,
        weights=weights,
        minlength=n_clusters,
    ).astype(np.float64)
    keep = cluster_weights > 0
    cluster_weights = cluster_weights[keep]
    cluster_weights /= cluster_weights.sum()
    centers = np.asarray(kmeans.cluster_centers_[keep], dtype=np.float32)
    return cluster_weights.astype(np.float32), centers, float(kmeans.inertia_)


def _weighted_support_sinkhorn(
    weights_x: np.ndarray,
    points_x: np.ndarray,
    weights_y: np.ndarray,
    points_y: np.ndarray,
    sinkhorn: SamplesLoss,
) -> float:
    weights_x = np.asarray(weights_x, dtype=np.float32).reshape(-1)
    weights_y = np.asarray(weights_y, dtype=np.float32).reshape(-1)
    points_x = np.asarray(points_x, dtype=np.float32)
    points_y = np.asarray(points_y, dtype=np.float32)
    weights_x /= weights_x.sum()
    weights_y /= weights_y.sum()
    with torch.no_grad():
        value = sinkhorn(
            torch.as_tensor(weights_x),
            torch.as_tensor(points_x),
            torch.as_tensor(weights_y),
            torch.as_tensor(points_y),
        )
    # Debiased Sinkhorn is nonnegative; guard only against float32 roundoff.
    return max(float(value.detach().cpu()), 0.0)


def _prepare_stage_references(
    rna_ref: list[np.ndarray],
    atac_ref: list[np.ndarray],
    k: int,
    sinkhorn_clusters: int,
) -> list[dict]:
    prepared = []
    for stage_idx in range(4):
        rna = np.asarray(rna_ref[stage_idx], dtype=np.float32)
        atac = np.asarray(atac_ref[stage_idx], dtype=np.float32)
        neighbors = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1).fit(rna)
        loo_neighbors = NearestNeighbors(n_neighbors=2, algorithm="auto", n_jobs=-1).fit(rna)
        loo_distances = loo_neighbors.kneighbors(rna, return_distance=True)[0][:, 1]
        rna_uniform = np.full(rna.shape[0], 1.0 / rna.shape[0], dtype=np.float64)
        rna_cluster_weights, rna_cluster_centers, rna_kmeans_inertia = (
            _microcluster_distribution(rna, rna_uniform, sinkhorn_clusters)
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
        prepared.append(
            {
                "neighbors": neighbors,
                "reference_loo_1nn_median": float(np.median(loo_distances)),
                "rna_cluster_weights": rna_cluster_weights,
                "rna_cluster_centers": rna_cluster_centers,
                "rna_kmeans_inertia": rna_kmeans_inertia,
                "cluster_ids": cluster_ids.astype(np.int64, copy=False),
                "cluster_centers": np.asarray(kmeans.cluster_centers_, dtype=np.float32),
                "kmeans_inertia": float(kmeans.inertia_),
            }
        )
        print(
            f"[reference] {STAGE_NAMES[stage_idx]} RNA={rna.shape} ATAC={atac.shape} "
            f"loo1={np.median(loo_distances):.6g} "
            f"rna_clusters={rna_cluster_centers.shape[0]} atac_clusters={n_clusters}",
            flush=True,
        )
    return prepared


def _filmsync_c_y(checkpoint_path: Path) -> float:
    match = re.search(r"_a([0-9.]+)_iter20000\.pth$", checkpoint_path.name)
    if not match:
        raise ValueError(f"Could not parse FilmSync C_y from {checkpoint_path}")
    return float(match.group(1))


def _filmsync_seed0_checkpoints(include_c_y_one: bool = True) -> list[Path]:
    checkpoints = sorted(
        FILMSYNC_SEED0_20K_CHECKPOINTS.glob(
            "ckpt_s0_e0.1_m100.0_d0.1_a*_iter20000.pth"
        ),
        key=_filmsync_c_y,
    )
    if len(checkpoints) != 10:
        raise FileNotFoundError(
            f"Expected ten HPC FilmSync seed-0 20k checkpoints, found {len(checkpoints)} "
            f"under {FILMSYNC_SEED0_20K_CHECKPOINTS}"
        )
    expected = np.arange(0.1, 1.01, 0.1)
    observed = np.asarray([_filmsync_c_y(path) for path in checkpoints])
    if not np.allclose(observed, expected, rtol=0.0, atol=1e-12):
        raise ValueError(f"Unexpected FilmSync C_y sweep: {observed.tolist()}")
    if include_c_y_one:
        return checkpoints
    return [path for path in checkpoints if _filmsync_c_y(path) < 1.0 - 1e-12]


def _filmsync_balanced_rollout(
    x0_norm: np.ndarray,
    device: torch.device,
    checkpoint_path: Path,
) -> list[np.ndarray]:
    """Reconstruct the balanced FilmSync RNA trajectory without applying T."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if str(TRAINF_ROOT) not in sys.path:
        sys.path.insert(0, str(TRAINF_ROOT))
    from src.Neural import MLPVectorField

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if int(checkpoint.get("iteration", -1)) != 20_000:
        raise ValueError(
            f"Expected FilmSync iteration 20000, got {checkpoint.get('iteration')}"
        )
    model = MLPVectorField(
        dim=x0_norm.shape[1],
        hidden_dim=800,
        n_layers=3,
        activation="leaky_relu",
        unbalanced=False,
        alpha_growth=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["func_state_dict"])
    model.eval().requires_grad_(False)

    x0 = torch.as_tensor(x0_norm, dtype=torch.float32, device=device)
    e0 = torch.zeros(x0.shape[0], 1, dtype=x0.dtype, device=device)
    times = torch.as_tensor(PHYSICAL_TIMES, dtype=x0.dtype, device=device)
    with torch.no_grad():
        positions, _ = torchdiffeq_odeint(
            model,
            (x0, e0),
            times,
            method="rk4",
            options={"step_size": 0.1},
        )
    endpoints = [
        positions[i].detach().cpu().numpy().astype(np.float32, copy=False)
        for i in range(len(PHYSICAL_TIMES))
    ]
    if not np.array_equal(endpoints[0], np.asarray(x0_norm, dtype=np.float32)):
        raise RuntimeError("FilmSync rollout did not preserve the common E7.5 initial cells")
    for stage_idx in range(1, len(STAGE_NAMES)):
        print(
            f"[prediction] FilmSync balanced C_y={_filmsync_c_y(checkpoint_path):.1f} "
            f"-> {STAGE_NAMES[stage_idx]} "
            f"{endpoints[stage_idx].shape}",
            flush=True,
        )
    return endpoints


def _cytobridge_unbalanced_rollout_with_mass(
    x0_norm: np.ndarray,
    scale: float,
    device: torch.device,
    batch_size: int,
    steps_per_interval: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Roll out CytoBridge position and cumulative log mass with one shared solver."""
    adata_path = ROOT / "results/cytobridge_gastrulation_rna_20000_unbalanced/adata.h5ad"
    model = _load_cytobridge_model(adata_path, device)
    if "growth" not in model.components or not hasattr(model, "growth_net"):
        raise ValueError("CytoBridge unbalanced checkpoint does not contain a growth network")

    n_query = x0_norm.shape[0]
    endpoints = [np.asarray(x0_norm, dtype=np.float32)]
    log_masses = [np.full(n_query, -np.log(n_query), dtype=np.float32)]
    x_all = endpoints[0]
    log_mass_all = log_masses[0]
    for interval in range(3):
        t0 = float(PHYSICAL_TIMES[interval])
        t1 = float(PHYSICAL_TIMES[interval + 1])
        dt = (t1 - t0) / int(steps_per_interval)
        position_batches = []
        log_mass_batches = []
        for start in range(0, n_query, batch_size):
            stop = min(start + batch_size, n_query)
            x = torch.as_tensor(x_all[start:stop], dtype=torch.float32, device=device)
            lnw = torch.as_tensor(log_mass_all[start:stop], dtype=torch.float32, device=device).unsqueeze(1)
            with torch.no_grad():
                for step in range(int(steps_per_interval)):
                    t_mid = torch.tensor(
                        [t0 + (step + 0.5) * dt],
                        dtype=torch.float32,
                        device=device,
                    )
                    x_raw = x * scale
                    velocity_raw = _cytobridge_velocity(model, t_mid, x_raw)
                    t_column = t_mid.expand(x.shape[0], 1)
                    growth = model.growth_net(torch.cat([x_raw, t_column], dim=1))
                    x = x + dt * (velocity_raw / scale)
                    lnw = lnw + dt * growth
            position_batches.append(x.detach().cpu().numpy())
            log_mass_batches.append(lnw.squeeze(1).detach().cpu().numpy())
        x_all = np.concatenate(position_batches).astype(np.float32, copy=False)
        log_mass_all = np.concatenate(log_mass_batches).astype(np.float32, copy=False)
        endpoints.append(x_all)
        log_masses.append(log_mass_all)
        total_mass = _normalized_particle_weights(n_query, log_mass_all)[1]
        print(
            f"[prediction] CytoBridge unbalanced -> {STAGE_NAMES[interval + 1]} "
            f"{x_all.shape} total_mass={total_mass:.6g}",
            flush=True,
        )
    return endpoints, log_masses


def _own_sync_log_masses(
    predictions: dict[str, list[np.ndarray]],
) -> dict[str, list[np.ndarray]]:
    trajectory_dir = (
        TRAINF_ROOT
        / "Gastrulation/UnbalancedSync_biological_num/trajectory_hpc_iter20000"
    )
    result: dict[str, list[np.ndarray]] = {}
    for c_y in np.arange(0.1, 1.0, 0.1):
        tag = f"{float(c_y):.1f}"
        name = f"UnbalancedSync biological-number C_y={tag}"
        if name not in predictions:
            raise KeyError(f"Missing UnbalancedSync prediction {name}")
        mass_path = trajectory_dir / f"mass_lnw_trajectory_s0_a{tag}_iter20000.pt"
        if not mass_path.exists():
            raise FileNotFoundError(mass_path)
        trajectory = torch.load(mass_path, map_location="cpu", weights_only=False)
        values = np.asarray(trajectory.detach().cpu(), dtype=np.float32)
        expected_shape = (26, predictions[name][0].shape[0], 1)
        if values.shape != expected_shape:
            raise ValueError(
                f"Unexpected UnbalancedSync log-mass shape {values.shape}; expected {expected_shape}"
            )
        result[name] = [values[step, :, 0] for step in (0, 10, 20, 25)]
    return result


def _evaluate_stage(
    method: str,
    stage_idx: int,
    points: np.ndarray,
    rna_reference: np.ndarray,
    labels: np.ndarray,
    prepared: dict,
    classes: list[str],
    sinkhorn: SamplesLoss,
    query_log_mass: np.ndarray | None = None,
    query_weighting: str = "uniform",
) -> tuple[dict, list[dict]]:
    query_weights, total_mass, effective_particles, max_particle_weight = (
        _normalized_particle_weights(points.shape[0], query_log_mass)
    )
    transferred, projection_distances, _, _ = _soft_knn_transfer(
        prepared["neighbors"],
        np.asarray(points, dtype=np.float32),
        rna_reference.shape[0],
        query_weights,
    )
    observed = np.full(rna_reference.shape[0], 1.0 / rna_reference.shape[0], dtype=np.float64)
    transferred_composition = _composition(transferred, labels, classes)
    observed_composition = _composition(observed, labels, classes)
    unknown_idx = classes.index(UNKNOWN)
    known_indices = np.asarray([i for i in range(len(classes)) if i != unknown_idx], dtype=int)
    transferred_known = transferred_composition[known_indices]
    observed_known = observed_composition[known_indices]
    composition_jsd = _js_divergence_base2(transferred_known, observed_known)

    effective_cells = float(1.0 / np.sum(np.square(transferred)))
    coverage_fraction = effective_cells / rna_reference.shape[0]
    reference_loo_median = float(prepared["reference_loo_1nn_median"])
    projection_median = _weighted_median(projection_distances, query_weights)
    projection_ratio = projection_median / reference_loo_median
    uniform_query = np.full(points.shape[0], 1.0 / points.shape[0], dtype=np.float64)
    if np.array_equal(np.asarray(points, dtype=np.float32), rna_reference) and np.allclose(
        query_weights, uniform_query, rtol=0.0, atol=1e-12
    ):
        rna_query_cluster_weights = prepared["rna_cluster_weights"]
        rna_query_cluster_centers = prepared["rna_cluster_centers"]
    else:
        rna_query_cluster_weights, rna_query_cluster_centers, _ = _microcluster_distribution(
            points,
            query_weights,
            prepared["rna_cluster_centers"].shape[0],
        )
    rna_sinkhorn_value = _weighted_support_sinkhorn(
        rna_query_cluster_weights,
        rna_query_cluster_centers,
        prepared["rna_cluster_weights"],
        prepared["rna_cluster_centers"],
        sinkhorn,
    )
    sinkhorn_value = _weighted_microcluster_sinkhorn(
        transferred,
        prepared["cluster_ids"],
        prepared["cluster_centers"],
        sinkhorn,
    )

    summary = {
        "method": method,
        "stage": STAGE_NAMES[stage_idx],
        "stage_key": STAGE_KEYS[stage_idx],
        "physical_time": float(PHYSICAL_TIMES[stage_idx]),
        "n_query": int(points.shape[0]),
        "n_reference": int(rna_reference.shape[0]),
        "query_weighting": query_weighting,
        "query_total_mass": total_mass,
        "query_effective_particle_count": effective_particles,
        "query_max_normalized_particle_mass": max_particle_weight,
        "atac_cjs_score": float(1.0 - composition_jsd),
        "atac_composition_jsd": composition_jsd,
        "rna_projection_1nn_median": projection_median,
        "rna_reference_loo_1nn_median": reference_loo_median,
        "rna_projection_distance_ratio": float(projection_ratio),
        "rna_sinkhorn_microcluster": rna_sinkhorn_value,
        "atac_effective_cell_count": effective_cells,
        "atac_coverage_fraction": float(coverage_fraction),
        "atac_sinkhorn_microcluster": sinkhorn_value,
        "transferred_unknown_mass": float(transferred_composition[unknown_idx]),
        "observed_unknown_fraction": float(observed_composition[unknown_idx]),
        "max_transferred_cell_mass": float(transferred.max()),
    }
    composition_rows = [
        {
            "method": method,
            "stage": STAGE_NAMES[stage_idx],
            "celltype": celltype,
            "transferred_fraction": float(transferred_composition[class_idx]),
            "observed_fraction": float(observed_composition[class_idx]),
            "difference": float(transferred_composition[class_idx] - observed_composition[class_idx]),
            "absolute_difference": float(
                abs(transferred_composition[class_idx] - observed_composition[class_idx])
            ),
        }
        for class_idx, celltype in enumerate(classes)
    ]
    return summary, composition_rows


def _build_predictions(
    x0: np.ndarray,
    rna_scale: float,
    device: torch.device,
    batch_size: int,
    steps_per_interval: int,
    include_c_y_one: bool,
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    predictions = {}
    native_log_masses: dict[str, list[np.ndarray]] = {}
    for checkpoint_path in _filmsync_seed0_checkpoints(include_c_y_one):
        c_y = _filmsync_c_y(checkpoint_path)
        predictions[f"FilmSync balanced C_y={c_y:.1f} 20k"] = (
            _filmsync_balanced_rollout(x0, device, checkpoint_path)
        )
    predictions.update({
        "TrajectoryNet 20k": _trajectorynet_rollout(
            x0, rna_scale, device, batch_size, steps_per_interval
        ),
        "CytoBridge balanced 20k": _cytobridge_rollout(
            "CytoBridge balanced",
            ROOT / "results/cytobridge_gastrulation_rna_20000/adata.h5ad",
            x0,
            rna_scale,
            device,
            batch_size,
            steps_per_interval,
        ),
        "MIOFlow 20k": _mioflow_rollout(
            x0, rna_scale, device, batch_size, steps_per_interval
        ),
    })
    cytobridge_predictions, cytobridge_log_masses = _cytobridge_unbalanced_rollout_with_mass(
        x0, rna_scale, device, batch_size, steps_per_interval
    )
    predictions["CytoBridge unbalanced 20k"] = cytobridge_predictions
    native_log_masses["CytoBridge unbalanced 20k"] = cytobridge_log_masses

    tigon_predictions, tigon_cumulative_log_growth = _tigon_rollout(x0, rna_scale, device)
    predictions["TIGON 20k"] = tigon_predictions
    initial_log_mass = -np.log(x0.shape[0])
    native_log_masses["TIGON 20k"] = [
        np.asarray(values, dtype=np.float32).reshape(-1) + initial_log_mass
        for values in tigon_cumulative_log_growth
    ]
    own_predictions_legacy, _ = _own_sync_rollouts(x0)
    own_predictions = {
        name.replace(" alpha=", " C_y="): values
        for name, values in own_predictions_legacy.items()
    }
    predictions.update(own_predictions)
    native_log_masses.update(_own_sync_log_masses(predictions))
    return predictions, native_log_masses


def _overall_table(stage_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in stage_df.groupby("method", sort=False):
        scored = group[group["stage"].isin(STAGE_NAMES[1:])].set_index("stage").reindex(STAGE_NAMES[1:])
        sanity = group[group["stage"] == STAGE_NAMES[0]].iloc[0]
        rows.append(
            {
                "method": method,
                "overall_atac_cjs_equal_stage_mean": float(scored["atac_cjs_score"].mean()),
                "E7.5_sanity_atac_cjs": float(sanity["atac_cjs_score"]),
                "E8.0_atac_cjs": float(scored.loc["E8.0", "atac_cjs_score"]),
                "E8.5_atac_cjs": float(scored.loc["E8.5", "atac_cjs_score"]),
                "E8.75_atac_cjs": float(scored.loc["E8.75", "atac_cjs_score"]),
                "overall_projection_ratio_equal_stage_mean": float(
                    scored["rna_projection_distance_ratio"].mean()
                ),
                "overall_rna_sinkhorn_equal_stage_mean": float(
                    scored["rna_sinkhorn_microcluster"].mean()
                ),
                "overall_atac_coverage_equal_stage_mean": float(
                    scored["atac_coverage_fraction"].mean()
                ),
                "overall_atac_sinkhorn_equal_stage_mean": float(
                    scored["atac_sinkhorn_microcluster"].mean()
                ),
                "overall_transferred_unknown_mass_equal_stage_mean": float(
                    scored["transferred_unknown_mass"].mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "overall_atac_cjs_equal_stage_mean", ascending=False, ignore_index=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate gastrulation RNA trajectories through a paired soft-kNN transfer to real ATAC cells."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/gastrulation_paired_knn_atac"),
    )
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument(
        "--mass-weighting",
        choices=("uniform", "native"),
        default="uniform",
        help=(
            "uniform gives every query particle weight 1/N; native uses cumulative "
            "particle mass for unbalanced methods and uniform mass for balanced methods"
        ),
    )
    parser.add_argument("--sinkhorn-clusters", type=int, default=512)
    parser.add_argument("--sinkhorn-blur", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--steps-per-interval", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--exclude-c-y-one",
        action="store_true",
        help="Exclude the FilmSync balanced C_y=1.0 checkpoint from this evaluation.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite to replace its files")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    rna_ref, atac_ref, labels_by_stage, rna_scale, atac_scale = _load_references()
    input_audit, input_identity = _audit_inputs(rna_scale)
    if not input_identity["all_matrix_inputs_exact"]:
        raise RuntimeError("Input matrix identity audit failed")
    predictions, native_log_masses = _build_predictions(
        rna_ref[0],
        rna_scale,
        device,
        args.batch_size,
        args.steps_per_interval,
        include_c_y_one=not args.exclude_c_y_one,
    )
    prepared = _prepare_stage_references(
        rna_ref, atac_ref, args.k, args.sinkhorn_clusters
    )
    classes = sorted(set(np.concatenate(labels_by_stage).tolist()) | {UNKNOWN})
    sinkhorn = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=args.sinkhorn_blur,
        debias=True,
        backend="tensorized",
    )

    summaries: list[dict] = []
    composition_rows: list[dict] = []
    reference_diagnostic_rows: list[dict] = []
    for stage_idx in range(4):
        diagnostic, _ = _evaluate_stage(
            "Observed RNA self-transfer diagnostic",
            stage_idx,
            rna_ref[stage_idx],
            rna_ref[stage_idx],
            labels_by_stage[stage_idx],
            prepared[stage_idx],
            classes,
            sinkhorn,
            query_log_mass=None,
            query_weighting="uniform",
        )
        reference_diagnostic_rows.append(diagnostic)
    for method, method_predictions in predictions.items():
        for stage_idx in range(4):
            use_native_mass = args.mass_weighting == "native" and method in native_log_masses
            stage_log_mass = native_log_masses[method][stage_idx] if use_native_mass else None
            summary, stage_composition = _evaluate_stage(
                method,
                stage_idx,
                method_predictions[stage_idx],
                rna_ref[stage_idx],
                labels_by_stage[stage_idx],
                prepared[stage_idx],
                classes,
                sinkhorn,
                query_log_mass=stage_log_mass,
                query_weighting="native_mass" if use_native_mass else "uniform",
            )
            summaries.append(summary)
            composition_rows.extend(stage_composition)
            print(
                f"[paired-kNN] {method} {STAGE_NAMES[stage_idx]} "
                f"CJS={summary['atac_cjs_score']:.6f} "
                f"proj={summary['rna_projection_distance_ratio']:.3f} "
                f"rna_sinkhorn={summary['rna_sinkhorn_microcluster']:.6f} "
                f"coverage={summary['atac_coverage_fraction']:.3f} "
                f"sinkhorn={summary['atac_sinkhorn_microcluster']:.6f}",
                flush=True,
            )

    stage_df = pd.DataFrame(summaries)
    composition_df = pd.DataFrame(composition_rows)
    overall_df = _overall_table(stage_df)
    input_audit.to_csv(args.output_dir / "input_identity_audit.csv", index=False)
    stage_df.to_csv(args.output_dir / "paired_knn_atac_stage_scores.csv", index=False)
    overall_df.to_csv(args.output_dir / "paired_knn_atac_overall_scores.csv", index=False)
    composition_df.to_csv(args.output_dir / "paired_knn_atac_celltype_composition.csv", index=False)
    pd.DataFrame(reference_diagnostic_rows).to_csv(
        args.output_dir / "paired_knn_reference_self_diagnostic.csv", index=False
    )

    manifest = {
        "protocol": {
            "query": (
                "all 9,018 E7.5 RNA cells; balanced methods use uniform particle weights; "
                "when mass_weighting=native, unbalanced methods use normalized cumulative "
                "particle mass"
            ),
            "mapping": (
                "At each observed stage, kNN in the paired real RNA reference. For k=1 each query's "
                "entire normalized particle mass is transferred to its nearest real RNA cell and the "
                "paired real ATAC cell. For k>1, mass is split by adaptive Gaussian distance weights."
            ),
            "k": int(args.k),
            "atac_cjs": (
                "1 - base-2 JSD between known-cell-type composition of transferred ATAC weights "
                "and the observed ATAC composition; Unknown is reported then excluded and renormalized."
            ),
            "projection_ratio": (
                "particle-weighted median predicted-to-reference RNA 1NN distance divided by the "
                "stage's uniform median real-RNA leave-one-out 1NN distance."
            ),
            "rna_sinkhorn": (
                "Debiased p=2 weighted Sinkhorn divergence between method-specific predicted-RNA "
                "MiniBatchKMeans microclusters and fixed real-RNA microclusters in the shared "
                "normalized 50D PCA space. Native normalized particle mass is used when enabled."
            ),
            "coverage": "normalized effective sample size: (1 / sum_i q_i^2) / n_stage.",
            "atac_sinkhorn": (
                "Debiased p=2 weighted Sinkhorn divergence on fixed ATAC MiniBatchKMeans microclusters; "
                "all real-cell transferred and observed masses are aggregated to the same centers."
            ),
            "sinkhorn_clusters": int(args.sinkhorn_clusters),
            "sinkhorn_blur": float(args.sinkhorn_blur),
            "overall": "equal mean across E8.0, E8.5, E8.75; E7.5 is sanity only",
            "mass_weighting": args.mass_weighting,
            "native_mass_methods": sorted(native_log_masses),
            "mass_normalization": (
                "Within each stage, native particle masses are normalized to sum to one before RNA-to-ATAC "
                "transfer. Raw predicted total mass is retained as query_total_mass in stage scores."
            ),
            "rna_scale": rna_scale,
            "atac_scale": atac_scale,
        },
        "methods": list(predictions.keys()),
        "filmsync_balanced_20k": {
            "checkpoints": [
                str(path)
                for path in _filmsync_seed0_checkpoints(not args.exclude_c_y_one)
            ],
            "remote_source": (
                "hpc:/rs1/researchers/z/zcang/jwang284/TraInf/TraInf/"
                "Gastrulation/FilmSync/hpc/checkpoint"
            ),
            "checkpoint_iteration": 20000,
            "seed": 0,
            "C_y": [
                _filmsync_c_y(path)
                for path in _filmsync_seed0_checkpoints(not args.exclude_c_y_one)
            ],
            "model": "balanced 50D neural ODE; hidden_dim=800, n_layers=3, leaky_relu; no growth state",
            "input": str(TRAINF_ROOT / "Gastrulation/data/rna_pca_by_time.npz"),
            "paired_sync_input": str(
                TRAINF_ROOT / "Gastrulation/data/atac_lsi_by_time_14D.npz"
            ),
            "preprocessing": (
                "RNA raw 50D PCA divided by primal global scale; paired ATAC first-14 LSI "
                "divided by secondary global scale; physical times 0,1,2,2.5."
            ),
            "training_loss": (
                "RNA and mapped-ATAC terminal Sinkhorn constraints use adaptive dual weights "
                "initialized/clipped at 50 and 100 (bounds 5..50 and 10..100); RNA and ATAC "
                "energy+density regularizers are mixed according to each checkpoint's C_y; "
                "energy_coefficient=0.1, density_coefficient=0.1; mass/growth losses absent."
            ),
            "evaluation_rollout": (
                "Primary RNA ODE only, deterministic RK4 with step_size=0.1 at physical times; "
                "T is not applied by this paired-kNN evaluation."
            ),
        },
        "training_instances": [
            {
                "method": "TrajectoryNet 20k",
                "checkpoint": str(
                    ROOT / "results/trajectorynet_gastrulation_rna_20000/checkpt-20000.pth"
                ),
                "input": str(ROOT / "data/gastrulation_rna_trajectorynet.npz"),
                "preprocessing": "raw 50D PCA; internal stage labels 0,1,2,3",
            },
            {
                "method": "CytoBridge balanced 20k",
                "checkpoint": str(
                    ROOT
                    / "results/cytobridge_gastrulation_rna_20000/Train/last_model.pth"
                ),
                "input": str(
                    ROOT / "results/cytobridge_gastrulation_rna_20000/adata.h5ad"
                ),
                "preprocessing": "raw 50D X_latent; physical times 0,1,2,2.5",
            },
            {
                "method": "CytoBridge unbalanced 20k",
                "checkpoint": str(
                    ROOT
                    / "results/cytobridge_gastrulation_rna_20000_unbalanced/Train/last_model.pth"
                ),
                "input": str(
                    ROOT
                    / "results/cytobridge_gastrulation_rna_20000_unbalanced/adata.h5ad"
                ),
                "preprocessing": "raw 50D X_latent; physical times 0,1,2,2.5",
            },
            {
                "method": "MIOFlow 20k",
                "checkpoint": str(ROOT / "results/mioflow_gastrulation_rna_20000/model.pt"),
                "input": str(ROOT / "data/gastrulation_rna_cytobridge.h5ad"),
                "preprocessing": (
                    "raw 50D X_latent with checkpoint-verified identity normalization; "
                    "no GAGA; internal stages 0,1,2,3"
                ),
            },
            {
                "method": "TIGON 20k no-AE",
                "checkpoint": str(ROOT / "results/tigon_gastrulation_rna_20000/tigon.pt"),
                "config": str(ROOT / "results/tigon_gastrulation_rna_20000/config.json"),
                "preprocessing": "raw 50D X_latent; no autoencoder; physical times 0,1,2,2.5",
            },
            {
                "method": "FilmSync balanced C_y sweep, seed 0, 20k",
                "checkpoints": [
                    str(path)
                    for path in _filmsync_seed0_checkpoints(not args.exclude_c_y_one)
                ],
                "input": str(TRAINF_ROOT / "Gastrulation/data/rna_pca_by_time.npz"),
                "preprocessing": "raw 50D RNA PCA divided by the primal global scale",
            },
            {
                "method": "UnbalancedSync biological-number C_y=0.1..0.9, seed 0, 20k",
                "checkpoints": [
                    str(path)
                    for path in sorted(
                        UNBALANCED_SYNC_20K_CHECKPOINTS.glob(
                            "ckpt_s0_e0.01_m100.0_d0.01_a*_iter20000.pth"
                        )
                    )
                ],
                "trajectories": str(
                    TRAINF_ROOT
                    / "Gastrulation/UnbalancedSync_biological_num/trajectory_hpc_iter20000"
                ),
                "preprocessing": "raw 50D RNA PCA divided by the same primal global scale",
            },
        ],
        "input_identity": input_identity,
        "stage_reference": [
            {
                "stage": STAGE_NAMES[i],
                "n_rna": int(rna_ref[i].shape[0]),
                "n_atac": int(atac_ref[i].shape[0]),
                "reference_loo_1nn_median": prepared[i]["reference_loo_1nn_median"],
                "rna_sinkhorn_clusters": int(prepared[i]["rna_cluster_centers"].shape[0]),
                "rna_kmeans_inertia": prepared[i]["rna_kmeans_inertia"],
                "sinkhorn_clusters": int(prepared[i]["cluster_centers"].shape[0]),
                "kmeans_inertia": prepared[i]["kmeans_inertia"],
            }
            for i in range(4)
        ],
    }
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\n[overall paired-kNN ATAC metrics]", flush=True)
    print(
        overall_df[
            [
                "method",
                "overall_atac_cjs_equal_stage_mean",
                "overall_projection_ratio_equal_stage_mean",
                "overall_rna_sinkhorn_equal_stage_mean",
                "overall_atac_coverage_equal_stage_mean",
                "overall_atac_sinkhorn_equal_stage_mean",
            ]
        ].round(6).to_string(index=False),
        flush=True,
    )
    print(f"\n[write] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
