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
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import anndata as ad
import numpy as np
import pandas as pd
import torch
from geomloss import SamplesLoss
from torchdiffeq import odeint


ROOT = Path(__file__).resolve().parents[2]
TRA_INF_ROOT = Path("external/COATI")
MOUSEBRAIN_ROOT = TRA_INF_ROOT / "MouseBrain"
ARCHIVE = ROOT / "results" / "palate_trajectory_archive"
DEFAULT_OUTPUT = ROOT / "results" / "palate_loo_same_space_sinkhorn"
BALANCED_SYNC_ROOT = ROOT / "results" / "palate_balanced_sync_loo_checkpoints"
CY_VALUES = tuple(round(i / 10.0, 1) for i in range(1, 10))

SCENARIOS = {
    "loo_time1": {
        "heldout_time": 1.0,
        "heldout_stage": "E13.5",
        "sync_experiment": "FiLMUnbalancedSync_LOO_time1",
        "balanced_sync_dir": BALANCED_SYNC_ROOT / "loo_time1",
        "reference": ROOT / "data" / "palate_rna_loo_time1_reference.npz",
        "trajectorynet_data": ROOT / "data" / "palate_rna_loo_time1_trajectorynet.npz",
        "trajectorynet_model": ROOT
        / "results"
        / "trajectorynet_palate_loo_time1_20000"
        / "checkpt-20000.pth",
        "mioflow_input": ROOT / "data" / "palate_rna_loo_time1_cytobridge.h5ad",
        "mioflow_model": ROOT
        / "results"
        / "mioflow_palate_loo_time1_20000"
        / "model.pt",
        "cytobridge_balanced": ROOT
        / "results"
        / "cytobridge_palate_loo_time1_20000"
        / "adata.h5ad",
        "cytobridge_unbalanced": ROOT
        / "results"
        / "cytobridge_palate_loo_time1_20000_unbalanced"
        / "adata.h5ad",
        "rna_only": MOUSEBRAIN_ROOT
        / "UnbalancedRNAOnly_LOO_time1_all1"
        / "checkpoint"
        / "ckpt_s0_e0.01_m100.0_d0.01_iter20000.pth",
        "rna_only_training_times": [0.0, 1.5, 2.0],
    },
    "loo_time2": {
        "heldout_time": 1.5,
        "heldout_stage": "E14.0",
        "sync_experiment": "FiLMUnbalancedSync_LOO_time2",
        "balanced_sync_dir": BALANCED_SYNC_ROOT / "loo_time2",
        "reference": ROOT / "data" / "palate_rna_loo_time2_reference.npz",
        "trajectorynet_data": ROOT / "data" / "palate_rna_loo_time2_trajectorynet.npz",
        "trajectorynet_model": ROOT
        / "results"
        / "trajectorynet_palate_loo_time2_20000"
        / "checkpt-20000.pth",
        "mioflow_input": ROOT / "data" / "palate_rna_loo_time2_cytobridge.h5ad",
        "mioflow_model": ROOT
        / "results"
        / "mioflow_palate_loo_time2_20000"
        / "model.pt",
        "cytobridge_balanced": ROOT
        / "results"
        / "cytobridge_palate_loo_time2_20000"
        / "adata.h5ad",
        "cytobridge_unbalanced": ROOT
        / "results"
        / "cytobridge_palate_loo_time2_20000_unbalanced"
        / "adata.h5ad",
        "rna_only": MOUSEBRAIN_ROOT
        / "UnbalancedRNAOnly_LOO_time2_all1"
        / "checkpoint"
        / "ckpt_s0_e0.01_m100.0_d0.01_iter20000.pth",
        "rna_only_training_times": [0.0, 1.0, 2.0],
    },
}


def _import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _load_scale(path: Path) -> float:
    state = torch.load(path, map_location="cpu", weights_only=False)
    scale = float(np.asarray(state["scale"]).reshape(-1)[0])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid scale in {path}: {scale}")
    return scale


def _deterministic_subsample(n: int, k: int) -> np.ndarray:
    if k <= 0 or k >= n:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, k).round().astype(np.int64)


def _normalize_weights(log_weight: np.ndarray | None, n: int) -> np.ndarray:
    if log_weight is None:
        return np.full(n, 1.0 / n, dtype=np.float32)
    values = np.asarray(log_weight, dtype=np.float64).reshape(-1)
    values -= np.max(values)
    weights = np.exp(np.clip(values, -80.0, 0.0))
    weights /= weights.sum()
    return weights.astype(np.float32)


def _load_film(device: torch.device):
    source = (
        ARCHIVE
        / "source_snapshot"
        / "TrainMap"
        / "train_FiLM_MLP_lsi15.py"
    )
    checkpoint = ROOT / "data" / "palate_rna_to_atac_film_lsi15.pt"
    sys.path.insert(0, str(TRA_INF_ROOT))
    sys.path.insert(0, str(ARCHIVE / "source_snapshot"))
    module = _import_module(source, "palate_loo_shared_film")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = module.FiLMMLP(**state["config"]).to(device)
    model.load_state_dict(state["state_dict"])
    model.requires_grad_(False)
    model.eval()
    return model, source, checkpoint, state


def _map_to_atac(
    model: torch.nn.Module,
    rna_norm: np.ndarray,
    heldout_time: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, rna_norm.shape[0], batch_size):
            x = torch.as_tensor(
                rna_norm[start : start + batch_size], dtype=torch.float32, device=device
            )
            t = torch.full((x.shape[0],), heldout_time, dtype=x.dtype, device=device)
            outputs.append(model(x, t).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def _load_internal_model(
    checkpoint: Path,
    device: torch.device,
    *,
    hidden_dim: int = 800,
    n_layers: int = 3,
    unbalanced: bool = True,
    alpha_growth: float = 1.0,
):
    neural = _import_module(
        ARCHIVE / "source_snapshot" / "src" / "Neural.py",
        "palate_loo_internal_neural",
    )
    model = neural.MLPVectorField(
        dim=40,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        activation="leaky_relu",
        unbalanced=unbalanced,
        alpha_growth=alpha_growth,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if int(state.get("iteration", -1)) != 20000:
        raise ValueError(f"Expected iteration 20000 in {checkpoint}")
    model.load_state_dict(state["func_state_dict"], strict=True)
    model.eval()
    return model


def _rollout_internal_endpoint(
    checkpoint: Path,
    x0_norm: np.ndarray,
    heldout_time: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model = _load_internal_model(checkpoint, device)
    x0 = torch.as_tensor(x0_norm, dtype=torch.float32, device=device)
    log_weight0 = torch.full(
        (x0.shape[0], 1), -math.log(x0.shape[0]), dtype=x0.dtype, device=device
    )
    zeros = torch.zeros_like(log_weight0)
    grid = torch.linspace(0.0, heldout_time, int(round(heldout_time / 0.025)) + 1, device=device)
    with torch.no_grad():
        x_t, log_weight_t, _, _ = odeint(
            model,
            (x0, log_weight0, zeros, zeros),
            grid,
            method="rk4",
            atol=1e-5,
            rtol=1e-5,
        )
    return (
        x_t[-1].detach().cpu().numpy().astype(np.float32, copy=False),
        log_weight_t[-1].detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _rollout_balanced_internal_endpoint(
    checkpoint: Path,
    x0_norm: np.ndarray,
    heldout_time: float,
    device: torch.device,
) -> np.ndarray:
    model = _load_internal_model(
        checkpoint,
        device,
        hidden_dim=400,
        n_layers=2,
        unbalanced=False,
        alpha_growth=0.0,
    )
    x0 = torch.as_tensor(x0_norm, dtype=torch.float32, device=device)
    energy0 = torch.zeros((x0.shape[0], 1), dtype=x0.dtype, device=device)
    grid = torch.linspace(
        0.0,
        heldout_time,
        int(round(heldout_time / 0.025)) + 1,
        device=device,
    )
    with torch.no_grad():
        x_t, _ = odeint(
            model,
            (x0, energy0),
            grid,
            method="rk4",
            atol=1e-5,
            rtol=1e-5,
        )
    return x_t[-1].detach().cpu().numpy().astype(np.float32, copy=False)


def _balanced_sync_checkpoint(config: dict, cy: float) -> Path:
    return Path(config["balanced_sync_dir"]) / (
        f"ckpt_s0_e0.1_m100.0_d0.1_a{cy:.1f}_iter20000.pth"
    )


def _load_trajectorynet_endpoint(
    checkpoint: Path,
    x0_norm: np.ndarray,
    scale: float,
    source_model_time: float,
    heldout_model_time: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    sys.path.insert(0, str(ROOT / "common"))
    from evaluate_terminal_push import _load_trajectorynet_model, _trajectorynet_diffeq

    model, model_args = _load_trajectorynet_model(checkpoint, 40, device, "rk4", 0.1)
    diffeq = _trajectorynet_diffeq(model)

    def velocity(t_value: torch.Tensor, x_norm: torch.Tensor) -> torch.Tensor:
        return diffeq(t_value, x_norm * scale) / scale

    source_internal = (float(source_model_time) + 1.0) * model_args.time_scale
    target_internal = (float(heldout_model_time) + 1.0) * model_args.time_scale
    # TrajectoryNet learns the reverse-time CNF. This descending integration is
    # the same convention used by the existing full-data palate evaluation.
    ts = torch.linspace(target_internal, source_internal, 21, device=device)
    outputs = []
    for start in range(0, x0_norm.shape[0], batch_size):
        x = torch.as_tensor(
            x0_norm[start : start + batch_size], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            path = odeint(velocity, x, ts, method="rk4", options={"step_size": 0.1})
        outputs.append(path[-1].detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def _load_cytobridge_model(path: Path, device: torch.device):
    sys.path.insert(0, str(ROOT / "external" / "CytoBridge"))
    from CytoBridge.utils import load_model_from_adata

    model = load_model_from_adata(ad.read_h5ad(path)).to(device)
    model.eval()
    return model


def _rollout_mioflow_endpoint(
    path: Path,
    x0_norm: np.ndarray,
    scale: float,
    heldout_time: float,
    training_physical_times: list[float],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    from mioflow.core.models.ode_model import ODEFunc

    state = torch.load(path, map_location=device, weights_only=False)
    if bool(state.get("use_gaga", False)):
        raise ValueError("Palate MIOFlow comparison expects a PCA-space model")

    input_space = str(state.get("input_space", "obsm"))
    if input_space == "moscot-normalized":
        x_input = np.asarray(x0_norm, dtype=np.float32)
    elif input_space == "obsm":
        x_input = np.asarray(x0_norm * scale, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported MIOFlow input space: {input_space}")

    mean_vals = (
        state["mean_vals"].detach().cpu().numpy().astype(np.float32)
    )
    std_vals = (
        state["std_vals"].detach().cpu().numpy().astype(np.float32)
    )
    x_model = (x_input - mean_vals) / std_vals

    model_times = np.asarray(state["time_points"], dtype=float)
    physical_times = np.asarray(training_physical_times, dtype=float)
    if (
        len(model_times) != len(physical_times)
        or np.any(np.diff(model_times) <= 0)
        or np.any(np.diff(physical_times) <= 0)
    ):
        raise ValueError("MIOFlow model and physical training times do not align")
    if not physical_times[0] <= heldout_time <= physical_times[-1]:
        raise ValueError("Held-out time lies outside the MIOFlow training span")
    target_model_time = float(
        np.interp(heldout_time, physical_times, model_times)
    )

    model = ODEFunc(
        input_dim=int(state["input_dim"]),
        hidden_dim=int(state["hidden_dim"]),
        momentum_beta=float(state.get("momentum_beta", 0.0)),
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    ts = torch.tensor(
        [float(model_times[0]), target_model_time],
        dtype=torch.float32,
        device=device,
    )
    endpoints: list[np.ndarray] = []
    for start in range(0, len(x_model), batch_size):
        x = torch.as_tensor(
            x_model[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        if hasattr(model, "reset_momentum"):
            model.reset_momentum()
        with torch.no_grad():
            endpoint = odeint(model, x, ts)[-1]
        endpoints.append(endpoint.detach().cpu().numpy())
    endpoint_model = np.concatenate(endpoints).astype(
        np.float32, copy=False
    )
    endpoint_input = endpoint_model * std_vals + mean_vals
    endpoint_norm = (
        endpoint_input
        if input_space == "moscot-normalized"
        else endpoint_input / scale
    )
    return endpoint_norm.astype(np.float32, copy=False), target_model_time


def _rollout_cytobridge_endpoint(
    path: Path,
    x0_norm: np.ndarray,
    scale: float,
    heldout_time: float,
    device: torch.device,
    batch_size: int,
    steps_per_unit: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    model = _load_cytobridge_model(path, device)
    n_steps = int(round(heldout_time * steps_per_unit))
    dt = heldout_time / n_steps
    endpoints = []
    log_weights = []
    for start in range(0, x0_norm.shape[0], batch_size):
        x = torch.as_tensor(
            x0_norm[start : start + batch_size], dtype=torch.float32, device=device
        )
        log_weight = torch.full(
            (x.shape[0], 1), -math.log(x0_norm.shape[0]), dtype=x.dtype, device=device
        )
        with torch.no_grad():
            for step in range(n_steps):
                t_mid = torch.tensor(
                    [(step + 0.5) * dt], dtype=torch.float32, device=device
                )
                t_column = t_mid.expand(x.shape[0], 1)
                net_input = torch.cat([x * scale, t_column], dim=1)
                velocity_raw = model.velocity_net(net_input)
                x = x + dt * velocity_raw / scale
                if "growth" in model.components:
                    log_weight = log_weight + dt * model.growth_net(net_input)
        endpoints.append(x.detach().cpu().numpy())
        if "growth" in model.components:
            log_weights.append(log_weight.detach().cpu().numpy())
    endpoint = np.concatenate(endpoints).astype(np.float32, copy=False)
    log_weight_np = (
        np.concatenate(log_weights).astype(np.float32, copy=False) if log_weights else None
    )
    return endpoint, log_weight_np


def _sinkhorn(
    loss: SamplesLoss,
    points_x: np.ndarray,
    weights_x: np.ndarray,
    points_y: np.ndarray,
    weights_y: np.ndarray,
    device: torch.device,
) -> float:
    with torch.no_grad():
        value = loss(
            torch.as_tensor(weights_x, dtype=torch.float32, device=device),
            torch.as_tensor(points_x, dtype=torch.float32, device=device),
            torch.as_tensor(weights_y, dtype=torch.float32, device=device),
            torch.as_tensor(points_y, dtype=torch.float32, device=device),
        )
    return max(float(value.detach().cpu()), 0.0)


def _audit_training_inputs(
    scenario: str,
    config: dict,
    rna: ad.AnnData,
    time_values: np.ndarray,
) -> list[dict]:
    rows = []
    heldout = float(config["heldout_time"])
    train_mask = ~np.isclose(time_values, heldout)
    expected = np.asarray(rna.obsm["X_latent"], dtype=np.float32)[train_mask]
    expected_ids = rna.obs_names.to_numpy(str)[train_mask]

    tn = np.load(config["trajectorynet_data"], allow_pickle=True)
    tn_values = np.asarray(tn["pca"], dtype=np.float32)
    rows.append(
        {
            "scenario": scenario,
            "input": "TrajectoryNet training PCA",
            "exact": bool(np.array_equal(tn_values, expected)),
            "shape": str(tn_values.shape),
            "max_abs_diff": float(np.max(np.abs(tn_values - expected))),
        }
    )
    for key, label in [
        ("mioflow_input", "MIOFlow training PCA"),
        ("cytobridge_balanced", "CytoBridge balanced training PCA"),
        ("cytobridge_unbalanced", "CytoBridge unbalanced training PCA"),
    ]:
        saved = ad.read_h5ad(config[key])
        values = np.asarray(saved.obsm["X_latent"], dtype=np.float32)
        ids_match = bool(np.array_equal(saved.obs_names.to_numpy(str), expected_ids))
        rows.append(
            {
                "scenario": scenario,
                "input": label,
                "exact": bool(np.array_equal(values, expected) and ids_match),
                "shape": str(values.shape),
                "max_abs_diff": float(np.max(np.abs(values - expected))),
            }
        )
    return rows


def _save_prediction(
    output: Path,
    *,
    rna_norm: np.ndarray,
    atac_norm: np.ndarray,
    weights: np.ndarray,
    initial_indices: np.ndarray,
    target_indices: np.ndarray,
    heldout_time: float,
    method: str,
    cy: float | None,
    particle_origin: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        rna_norm=rna_norm,
        atac_norm=atac_norm,
        weights=weights,
        initial_indices=initial_indices,
        target_indices=target_indices,
        heldout_time=np.asarray(heldout_time, dtype=np.float32),
        method=np.asarray(method),
        cy=np.asarray(np.nan if cy is None else cy, dtype=np.float32),
        particle_origin=np.asarray(particle_origin),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare palate LOO predictions from one common initial population in shared "
            "normalized RNA PCA40 and ATAC LSI15 spaces."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cytobridge-steps-per-unit", type=int, default=40)
    parser.add_argument("--sinkhorn-blur", type=float, default=1e-4)
    parser.add_argument("--sinkhorn-backend", default="tensorized")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    required = [
        ROOT / "data" / "palate_rna_cytobridge.h5ad",
        ROOT / "data" / "palate_atac_benchmark.h5ad",
        ROOT / "data" / "palate_rna_primal_norm_params.pt",
        ROOT / "data" / "palate_atac_secondary_norm_params_lsi15.pt",
        ROOT / "data" / "palate_rna_to_atac_film_lsi15.pt",
        ARCHIVE / "manifest" / "common_initial_indices.npy",
        ARCHIVE / "manifest" / "common_initial_rna_norm.npy",
        MOUSEBRAIN_ROOT / "data" / "rna_pca_by_time.npz",
        MOUSEBRAIN_ROOT / "data" / "secondary_norm_params_lsi15.pt",
        MOUSEBRAIN_ROOT / "TrainMap" / "T_FiLM_lsi15.pt",
    ]
    for config in SCENARIOS.values():
        required.extend(
            [
                config["reference"],
                config["trajectorynet_data"],
                config["trajectorynet_model"],
                config["mioflow_input"],
                config["mioflow_model"],
                config["cytobridge_balanced"],
                config["cytobridge_unbalanced"],
                config["rna_only"],
            ]
        )
        for cy in CY_VALUES:
            required.append(
                ARCHIVE
                / "trajectories"
                / "sync"
                / config["sync_experiment"]
                / "seed_0"
                / f"cy_{cy:.1f}_iter_20000.npz"
            )
            required.append(_balanced_sync_checkpoint(config, cy))
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    device = torch.device(args.device)
    rna_scale = _load_scale(ROOT / "data" / "palate_rna_primal_norm_params.pt")
    atac_scale = _load_scale(
        ROOT / "data" / "palate_atac_secondary_norm_params_lsi15.pt"
    )
    rna = ad.read_h5ad(ROOT / "data" / "palate_rna_cytobridge.h5ad")
    atac = ad.read_h5ad(ROOT / "data" / "palate_atac_benchmark.h5ad")
    if not np.array_equal(rna.obs_names.to_numpy(str), atac.obs_names.to_numpy(str)):
        raise ValueError("RNA and ATAC cells are not paired in the same order")
    time_values = pd.to_numeric(
        rna.obs["time_point_processed"], errors="raise"
    ).to_numpy(float)
    if not np.array_equal(
        time_values,
        pd.to_numeric(atac.obs["time_point_processed"], errors="raise").to_numpy(float),
    ):
        raise ValueError("RNA and ATAC physical times differ")

    x_rna_raw = np.asarray(rna.obsm["X_latent"], dtype=np.float32)
    x_atac_raw = np.asarray(atac.obsm["X_lsi15"], dtype=np.float32)
    initial_population = np.flatnonzero(np.isclose(time_values, 0.0))
    local_initial_indices = np.load(ARCHIVE / "manifest" / "common_initial_indices.npy")
    if np.any(local_initial_indices >= len(initial_population)):
        raise ValueError("Archive initial indices exceed the E12.5 population")
    initial_indices = initial_population[local_initial_indices]
    x0_norm = (x_rna_raw[initial_indices] / rna_scale).astype(np.float32, copy=False)
    archived_x0 = np.load(ARCHIVE / "manifest" / "common_initial_rna_norm.npy")
    if not np.array_equal(x0_norm, archived_x0):
        raise ValueError("Common initial RNA population differs from the trajectory archive")

    film, film_source, film_checkpoint, film_state = _load_film(device)
    loss = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=args.sinkhorn_blur,
        debias=True,
        backend=args.sinkhorn_backend,
    ).to(device)

    score_rows: list[dict] = []
    prediction_rows: list[dict] = []
    audit_rows = [
        {
            "scenario": "all",
            "input": "RNA and ATAC paired cell order",
            "exact": True,
            "shape": str((rna.n_obs,)),
            "max_abs_diff": 0.0,
        },
        {
            "scenario": "all",
            "input": "common initial RNA equals trajectory archive",
            "exact": True,
            "shape": str(x0_norm.shape),
            "max_abs_diff": 0.0,
        },
    ]
    source_rna = np.load(MOUSEBRAIN_ROOT / "data" / "rna_pca_by_time.npz")
    source_max_diff = 0.0
    source_exact = True
    for time_index, physical_time in enumerate([0.0, 1.0, 1.5, 2.0]):
        expected = x_rna_raw[np.isclose(time_values, physical_time)]
        observed = np.asarray(source_rna[f"time_{time_index}"], dtype=np.float32)
        source_exact = source_exact and bool(np.array_equal(observed, expected))
        source_max_diff = max(source_max_diff, float(np.max(np.abs(observed - expected))))
    audit_rows.extend(
        [
            {
                "scenario": "all",
                "input": "USOT/BalancedSync/RNA-only source RNA arrays equal shared PCA40",
                "exact": source_exact,
                "shape": "four physical-time arrays",
                "max_abs_diff": source_max_diff,
            },
            {
                "scenario": "all",
                "input": "FiLM LSI15 checkpoint equals original TrainMap artifact",
                "exact": _sha256(film_checkpoint)
                == _sha256(MOUSEBRAIN_ROOT / "TrainMap" / "T_FiLM_lsi15.pt"),
                "shape": "checkpoint sha256",
                "max_abs_diff": 0.0,
            },
            {
                "scenario": "all",
                "input": "ATAC LSI15 normalization equals original data artifact",
                "exact": _sha256(
                    ROOT / "data" / "palate_atac_secondary_norm_params_lsi15.pt"
                )
                == _sha256(MOUSEBRAIN_ROOT / "data" / "secondary_norm_params_lsi15.pt"),
                "shape": "checkpoint sha256",
                "max_abs_diff": 0.0,
            },
        ]
    )

    for scenario, config in SCENARIOS.items():
        heldout_time = float(config["heldout_time"])
        heldout_all = np.flatnonzero(np.isclose(time_values, heldout_time))
        target_local = _deterministic_subsample(
            len(heldout_all), min(args.eval_samples, len(heldout_all))
        )
        target_indices = heldout_all[target_local]
        target_rna = (x_rna_raw[target_indices] / rna_scale).astype(np.float32, copy=False)
        target_atac = (x_atac_raw[target_indices] / atac_scale).astype(np.float32, copy=False)
        target_weights = np.full(
            len(target_indices), 1.0 / len(target_indices), dtype=np.float32
        )

        reference = np.load(config["reference"], allow_pickle=True)
        if not np.array_equal(reference["heldout_indices"], heldout_all):
            raise ValueError(f"{scenario} held-out indices differ from the shared data")
        audit_rows.extend(_audit_training_inputs(scenario, config, rna, time_values))
        terminal_all = np.asarray(reference["terminal_indices"], dtype=np.int64)
        terminal_local = _deterministic_subsample(
            len(terminal_all), min(args.eval_samples, len(terminal_all))
        )
        tn_terminal_indices = terminal_all[terminal_local]
        tn_terminal_norm = (
            x_rna_raw[tn_terminal_indices] / rna_scale
        ).astype(np.float32, copy=False)

        predictions: list[dict] = []
        for cy in CY_VALUES:
            trajectory_file = (
                ARCHIVE
                / "trajectories"
                / "sync"
                / config["sync_experiment"]
                / "seed_0"
                / f"cy_{cy:.1f}_iter_20000.npz"
            )
            with np.load(trajectory_file) as trajectory:
                time_idx = int(
                    np.flatnonzero(np.isclose(trajectory["time"], heldout_time))[0]
                )
                prediction_rna = trajectory["rna_norm"][time_idx].astype(
                    np.float32, copy=False
                )
                log_weight = trajectory["log_weight"][time_idx]
                archived_atac = trajectory["atac_norm"][time_idx]
            prediction_atac = _map_to_atac(
                film, prediction_rna, heldout_time, device, args.batch_size
            )
            max_map_diff = float(np.max(np.abs(prediction_atac - archived_atac)))
            if max_map_diff > 1e-6:
                raise ValueError(
                    f"{scenario} cy={cy:.1f} archived ATAC map mismatch: {max_map_diff}"
                )
            predictions.append(
                {
                    "method": "USOT",
                    "slug": f"usot_cy_{cy:.1f}".replace(".", "p"),
                    "cy": cy,
                    "rna": prediction_rna,
                    "atac": prediction_atac,
                    "weights": _normalize_weights(log_weight, len(prediction_rna)),
                    "weight_source": "native learned growth mass",
                    "artifact": trajectory_file,
                }
            )

        for cy in CY_VALUES:
            checkpoint = _balanced_sync_checkpoint(config, cy)
            prediction_rna = _rollout_balanced_internal_endpoint(
                checkpoint, x0_norm, heldout_time, device
            )
            predictions.append(
                {
                    "method": "BalancedSync",
                    "slug": f"balanced_sync_cy_{cy:.1f}".replace(".", "p"),
                    "cy": cy,
                    "rna": prediction_rna,
                    "atac": _map_to_atac(
                        film, prediction_rna, heldout_time, device, args.batch_size
                    ),
                    "weights": _normalize_weights(None, len(prediction_rna)),
                    "weight_source": "uniform (balanced model)",
                    "artifact": checkpoint,
                }
            )

        rna_only_rna, rna_only_log_weight = _rollout_internal_endpoint(
            config["rna_only"], x0_norm, heldout_time, device
        )
        predictions.append(
            {
                "method": "Unbalanced RNA-only",
                "slug": "unbalanced_rna_only",
                "cy": None,
                "rna": rna_only_rna,
                "atac": _map_to_atac(
                    film, rna_only_rna, heldout_time, device, args.batch_size
                ),
                "weights": _normalize_weights(rna_only_log_weight, len(rna_only_rna)),
                "weight_source": "native learned growth mass",
                "artifact": config["rna_only"],
            }
        )

        heldout_model_time = float(reference["trajectorynet_heldout_model_time"])
        terminal_model_time = float(reference["trajectorynet_model_train_times"][-1])
        tn_rna = _load_trajectorynet_endpoint(
            config["trajectorynet_model"],
            tn_terminal_norm,
            rna_scale,
            heldout_model_time,
            terminal_model_time,
            device,
            args.batch_size,
        )
        predictions.append(
            {
                "method": "TrajectoryNet",
                "slug": "trajectorynet",
                "cy": None,
                "rna": tn_rna,
                "atac": _map_to_atac(film, tn_rna, heldout_time, device, args.batch_size),
                "weights": _normalize_weights(None, len(tn_rna)),
                "weight_source": "uniform",
                "artifact": config["trajectorynet_model"],
                "particle_indices": tn_terminal_indices,
                "particle_origin": "observed E14.5 terminal RNA cells",
            }
        )

        mioflow_rna, mioflow_target_time = _rollout_mioflow_endpoint(
            config["mioflow_model"],
            x0_norm,
            rna_scale,
            heldout_time,
            config["rna_only_training_times"],
            device,
            args.batch_size,
        )
        predictions.append(
            {
                "method": "MIOFlow",
                "slug": "mioflow",
                "cy": None,
                "rna": mioflow_rna,
                "atac": _map_to_atac(
                    film,
                    mioflow_rna,
                    heldout_time,
                    device,
                    args.batch_size,
                ),
                "weights": _normalize_weights(None, len(mioflow_rna)),
                "weight_source": "uniform",
                "artifact": config["mioflow_model"],
                "model_target_time": mioflow_target_time,
            }
        )

        for key, method, slug in [
            ("cytobridge_balanced", "CytoBridge balanced", "cytobridge_balanced"),
            ("cytobridge_unbalanced", "CytoBridge unbalanced", "cytobridge_unbalanced"),
        ]:
            cb_rna, cb_log_weight = _rollout_cytobridge_endpoint(
                config[key],
                x0_norm,
                rna_scale,
                heldout_time,
                device,
                args.batch_size,
                args.cytobridge_steps_per_unit,
            )
            predictions.append(
                {
                    "method": method,
                    "slug": slug,
                    "cy": None,
                    "rna": cb_rna,
                    "atac": _map_to_atac(
                        film, cb_rna, heldout_time, device, args.batch_size
                    ),
                    "weights": _normalize_weights(cb_log_weight, len(cb_rna)),
                    "weight_source": (
                        "native learned growth mass"
                        if cb_log_weight is not None
                        else "uniform"
                    ),
                    "artifact": config[key],
                }
            )

        for prediction in predictions:
            output = (
                args.output_dir
                / "predictions"
                / scenario
                / f"{prediction['slug']}.npz"
            )
            _save_prediction(
                output,
                rna_norm=prediction["rna"],
                atac_norm=prediction["atac"],
                weights=prediction["weights"],
                initial_indices=prediction.get("particle_indices", initial_indices),
                target_indices=target_indices,
                heldout_time=heldout_time,
                method=prediction["method"],
                cy=prediction["cy"],
                particle_origin=prediction.get(
                    "particle_origin", "common observed E12.5 RNA cells"
                ),
            )
            prediction_rows.append(
                {
                    "scenario": scenario,
                    "heldout_stage": config["heldout_stage"],
                    "heldout_time": heldout_time,
                    "method": prediction["method"],
                    "cy": prediction["cy"],
                    "weight_source": prediction["weight_source"],
                    "n_initial": len(prediction["rna"]),
                    "particle_origin": prediction.get(
                        "particle_origin", "common observed E12.5 RNA cells"
                    ),
                    "n_target_eval": len(target_indices),
                    "prediction_file": str(output.relative_to(args.output_dir)),
                    "prediction_sha256": _sha256(output),
                    "model_artifact": str(prediction["artifact"]),
                    "model_artifact_sha256": _sha256(Path(prediction["artifact"])),
                }
            )
            for modality, points, target in [
                ("RNA_PCA40", prediction["rna"], target_rna),
                ("ATAC_LSI15", prediction["atac"], target_atac),
            ]:
                native_value = _sinkhorn(
                    loss,
                    points,
                    prediction["weights"],
                    target,
                    target_weights,
                    device,
                )
                uniform_weights = _normalize_weights(None, len(points))
                uniform_value = _sinkhorn(
                    loss,
                    points,
                    uniform_weights,
                    target,
                    target_weights,
                    device,
                )
                score_rows.append(
                    {
                        "scenario": scenario,
                        "heldout_stage": config["heldout_stage"],
                        "heldout_time": heldout_time,
                        "method": prediction["method"],
                        "cy": prediction["cy"],
                        "modality": modality,
                        "sinkhorn_native_mass": native_value,
                        "sinkhorn_equal_particle": uniform_value,
                        "weight_source": prediction["weight_source"],
                        "n_initial": len(points),
                        "particle_origin": prediction.get(
                            "particle_origin", "common observed E12.5 RNA cells"
                        ),
                        "n_target_eval": len(target_indices),
                        "p": 2,
                        "blur": args.sinkhorn_blur,
                        "debias": True,
                        "backend": args.sinkhorn_backend,
                    }
                )
            print(
                f"[{scenario}] {prediction['method']}"
                + (f" cy={prediction['cy']:.1f}" if prediction["cy"] is not None else "")
                + " complete",
                flush=True,
            )

    scores = pd.DataFrame(score_rows)
    predictions = pd.DataFrame(prediction_rows)
    audits = pd.DataFrame(audit_rows)
    if not bool(audits["exact"].all()):
        raise ValueError("Input audit failed")
    scores.to_csv(args.output_dir / "palate_loo_sinkhorn_all.csv", index=False)
    predictions.to_csv(args.output_dir / "prediction_manifest.csv", index=False)
    audits.to_csv(args.output_dir / "input_space_audit.csv", index=False)

    primary_keys = ["scenario", "heldout_stage", "heldout_time", "method", "cy"]
    rna_primary = scores[scores["modality"] == "RNA_PCA40"][
        primary_keys + ["sinkhorn_native_mass"]
    ].rename(columns={"sinkhorn_native_mass": "rna_sinkhorn"})
    atac_primary = scores[scores["modality"] == "ATAC_LSI15"][
        primary_keys + ["sinkhorn_native_mass"]
    ].rename(columns={"sinkhorn_native_mass": "atac_sinkhorn"})
    primary = rna_primary.merge(atac_primary, on=primary_keys, how="outer").sort_values(
        ["scenario", "method", "cy"], na_position="last"
    )
    primary.to_csv(args.output_dir / "palate_loo_sinkhorn_primary.csv", index=False)
    primary[primary["method"] == "USOT"].to_csv(
        args.output_dir / "palate_loo_usot_by_cy.csv", index=False
    )
    primary[primary["method"] == "BalancedSync"].to_csv(
        args.output_dir / "palate_loo_balanced_sync_by_cy.csv", index=False
    )

    manifest = {
        "task": "Palate LOO same-space RNA and ATAC Sinkhorn comparison with each method's native particle origin",
        "common_initial": {
            "n": int(len(initial_indices)),
            "global_indices_sha256": _array_sha256(initial_indices),
            "rna_normalized_sha256": _array_sha256(x0_norm),
            "selection": "deterministic linspace subset of all E12.5 cells; exact archive population",
        },
        "spaces": {
            "RNA_PCA40": {
                "definition": "shared raw RNA PCA40 divided by one global W2 scale",
                "scale": rna_scale,
            },
            "ATAC_LSI15": {
                "definition": (
                    "observed raw ATAC LSI15 divided by one global W2 scale; predictions "
                    "mapped by the same frozen full-data FiLM T_theta used by palate USOT"
                ),
                "scale": atac_scale,
                "film_source": str(film_source),
                "film_checkpoint": str(film_checkpoint),
                "film_checkpoint_sha256": _sha256(film_checkpoint),
                "film_config": film_state["config"],
            },
        },
        "sinkhorn": {
            "primary": "native-mass debiased Sinkhorn divergence",
            "sensitivity": "equal-particle debiased Sinkhorn divergence",
            "p": 2,
            "blur": args.sinkhorn_blur,
            "debias": True,
            "backend": args.sinkhorn_backend,
            "target_sampling": "2048 deterministic linspace cells from the held-out population",
        },
        "methods": [
            "USOT cy=0.1..0.9, seed0, 20k",
            "BalancedSync cy=0.1..0.9, seed0, 20k",
            "Unbalanced RNA-only seed0, 20k",
            "TrajectoryNet 20k",
            "MIOFlow 20k",
            "CytoBridge balanced 20k",
            "CytoBridge unbalanced 20k",
        ],
        "scenarios": {
            name: {
                "heldout_stage": config["heldout_stage"],
                "heldout_time": config["heldout_time"],
                "training_times": config["rna_only_training_times"],
            }
            for name, config in SCENARIOS.items()
        },
        "cytobridge_integrator": {
            "method": "midpoint-time Euler",
            "steps_per_physical_time_unit": args.cytobridge_steps_per_unit,
            "growth": "integrated with the same midpoint evaluations for native mass",
        },
        "balanced_sync_integrator": {
            "model": "balanced MLPVectorField, hidden_dim=400, n_layers=2",
            "method": "RK4 on the saved normalized RNA vector field",
            "step_size": 0.025,
            "particle_mass": "uniform",
            "checkpoints": str(BALANCED_SYNC_ROOT / "<scenario>"),
        },
        "trajectorynet": {
            "solver": "rk4",
            "step_size": 0.1,
            "particle_origin": "2,048 deterministic observed E14.5 terminal RNA cells",
            "direction": "native density direction from terminal model time down to the held-out model time",
        },
        "mioflow": {
            "solver": "torchdiffeq adaptive odeint",
            "particle_mass": "uniform",
            "initial_population": "the same 2048 normalized E12.5 cells used by every method",
            "time_mapping": (
                "piecewise-linear interpolation from physical palate time to the "
                "ordinal MIOFlow times stored in each LOO checkpoint"
            ),
        },
        "files": {
            "primary_scores": "palate_loo_sinkhorn_primary.csv",
            "all_scores": "palate_loo_sinkhorn_all.csv",
            "usot_by_cy": "palate_loo_usot_by_cy.csv",
            "balanced_sync_by_cy": "palate_loo_balanced_sync_by_cy.csv",
            "predictions": "predictions/<scenario>/<method>.npz",
            "prediction_manifest": "prediction_manifest.csv",
            "input_audit": "input_space_audit.csv",
        },
    }
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(primary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
