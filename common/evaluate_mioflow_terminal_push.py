#!/usr/bin/env python
from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[1]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import anndata as ad
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torchdiffeq import odeint


def _load_norm_scale(norm_params: Path) -> float:
    params = torch.load(norm_params, map_location="cpu", weights_only=False)
    if not isinstance(params, dict) or "scale" not in params:
        raise ValueError(f"Expected {norm_params} to contain a dict with key 'scale'")
    scale = params["scale"]
    return float(scale.item() if hasattr(scale, "item") else scale)


def _as_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype="float32"), device=device)


def _batched_slices(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield slice(start, min(start + batch_size, n))


def _sinkhorn_geomloss(
    x: np.ndarray,
    y: np.ndarray,
    blur: float,
    backend: str,
    device: torch.device,
) -> float:
    from geomloss import SamplesLoss

    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=blur, backend=backend).to(device)
    with torch.no_grad():
        return float(loss_fn(_as_tensor(x, device), _as_tensor(y, device)).item())


def _approx_energy(path: np.ndarray, duration: float) -> float:
    if path.shape[0] < 2:
        return 0.0
    dt = duration / float(path.shape[0] - 1)
    velocity = (path[1:] - path[:-1]) / dt
    energy_per_step = 0.5 * np.sum(velocity * velocity, axis=2)
    return float(np.mean(np.sum(energy_per_step, axis=0) * dt))


def _restore_scaler(mean: torch.Tensor, scale: torch.Tensor) -> StandardScaler:
    scaler = StandardScaler()
    mean_np = mean.detach().cpu().numpy().astype(np.float64)
    scale_np = scale.detach().cpu().numpy().astype(np.float64)
    scaler.mean_ = mean_np
    scaler.scale_ = scale_np
    scaler.var_ = scale_np * scale_np
    scaler.n_features_in_ = mean_np.shape[0]
    return scaler


def _load_gaga_model(gaga_path: Path, device: torch.device):
    from mioflow import Autoencoder

    state = torch.load(gaga_path, map_location=device, weights_only=False)
    model = Autoencoder(
        input_dim=int(state["input_dim"]),
        latent_dim=int(state["latent_dim"]),
        hidden_dims=[int(x) for x in state["hidden_dims"]],
    ).to(device)
    model.load_state_dict(state["state_dict"])
    model.input_scaler = _restore_scaler(
        state["input_scaler_mean"],
        state["input_scaler_scale"],
    )
    model.eval()
    return model


def _encode_gaga(
    x_input_space: np.ndarray,
    model,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    scaler = getattr(model, "input_scaler", None)
    x_scaled = scaler.transform(x_input_space) if scaler is not None else x_input_space
    batches = []
    with torch.no_grad():
        for batch_slice in _batched_slices(x_scaled.shape[0], batch_size):
            z = model.encode(_as_tensor(x_scaled[batch_slice], device))
            batches.append(z.detach().cpu().numpy())
    return np.vstack(batches).astype("float32", copy=False)


def _decode_gaga_path(
    z_path: np.ndarray,
    model,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    shape = z_path.shape
    flat = z_path.reshape(-1, shape[-1]).astype("float32", copy=False)
    decoded_batches = []
    with torch.no_grad():
        for batch_slice in _batched_slices(flat.shape[0], batch_size):
            decoded = model.decode(_as_tensor(flat[batch_slice], device))
            decoded_batches.append(decoded.detach().cpu().numpy())
    decoded_scaled = np.vstack(decoded_batches)
    scaler = getattr(model, "input_scaler", None)
    decoded = scaler.inverse_transform(decoded_scaled) if scaler is not None else decoded_scaled
    return decoded.reshape(shape[0], shape[1], -1).astype("float32", copy=False)


def mioflow_terminal_path(
    x0_norm: np.ndarray,
    model_path: Path,
    norm_scale: float,
    source_time: float,
    terminal_time: float,
    device: torch.device,
    batch_size: int,
    steps_per_interval: int,
) -> tuple[np.ndarray, dict]:
    from mioflow.core.models.ode_model import ODEFunc

    state = torch.load(model_path, map_location=device, weights_only=False)
    input_space = str(state.get("input_space", "obsm"))
    use_gaga = bool(state.get("use_gaga", False))

    if input_space == "moscot-normalized":
        x_input_space = x0_norm.astype("float32", copy=False)
    else:
        x_input_space = (x0_norm * norm_scale).astype("float32", copy=False)

    gaga_model = None
    if use_gaga:
        gaga_model = _load_gaga_model(Path(state["gaga_model_path"]), device)
        z0 = _encode_gaga(x_input_space, gaga_model, device, batch_size)
    else:
        z0 = x_input_space

    mean_vals = state["mean_vals"].detach().cpu().numpy().astype("float32")
    std_vals = state["std_vals"].detach().cpu().numpy().astype("float32")
    z0_model = (z0 - mean_vals) / std_vals

    time_points = [float(t) for t in state.get("time_points", [source_time, terminal_time])]
    train_duration = max(time_points) - min(time_points)
    data_duration = terminal_time - source_time
    if train_duration <= 0 or data_duration <= 0:
        raise ValueError("Source/terminal duration must be positive")
    total_steps = max(1, int(round(steps_per_interval * data_duration)))
    ts = torch.linspace(min(time_points), max(time_points), total_steps + 1, device=device, dtype=torch.float32)

    model = ODEFunc(
        input_dim=int(state["input_dim"]),
        hidden_dim=int(state["hidden_dim"]),
        momentum_beta=float(state.get("momentum_beta", 0.0)),
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    paths = []
    for batch_slice in _batched_slices(z0_model.shape[0], batch_size):
        x = _as_tensor(z0_model[batch_slice], device)
        if hasattr(model, "reset_momentum"):
            model.reset_momentum()
        with torch.no_grad():
            path = odeint(model, x, ts)
        paths.append(path.detach().cpu().numpy())
    model_path_norm = np.concatenate(paths, axis=1).astype("float32", copy=False)
    model_path_denorm = model_path_norm * std_vals + mean_vals

    if gaga_model is not None:
        input_path = _decode_gaga_path(model_path_denorm, gaga_model, device, batch_size)
    else:
        input_path = model_path_denorm

    if input_space == "moscot-normalized":
        normalized_path = input_path.astype("float32", copy=False)
    else:
        normalized_path = (input_path / norm_scale).astype("float32", copy=False)

    meta = {
        "input_space": input_space,
        "use_gaga": use_gaga,
        "model_time_points": time_points,
        "total_steps": total_steps,
        "model_input_dim": int(state["input_dim"]),
    }
    return normalized_path, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate full-data MIOFlow terminal push in normalized PCA space.")
    parser.add_argument("--input-h5ad", type=Path, default=Path("data/moscot_rna_cytobridge.h5ad"))
    parser.add_argument("--time-key", default="time_point_processed")
    parser.add_argument("--latent-key", default="X_latent")
    parser.add_argument("--mioflow-model", type=Path, default=Path("results/mioflow_moscot_rna_20000/model.pt"))
    parser.add_argument("--norm-params", type=Path, default=Path("external/COATI/moscot/data/primal_norm_params.pt"))
    parser.add_argument("--output-json", type=Path, default=Path("results/terminal_push_mioflow_moscot_rna_20000_normalized.json"))
    parser.add_argument("--output-npz", type=Path, default=Path("results/terminal_push_mioflow_moscot_rna_20000_normalized_predictions.npz"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--steps-per-interval", type=int, default=20)
    parser.add_argument("--geomloss-blurs", default="1e-4,1e-3")
    parser.add_argument("--geomloss-backend", default="tensorized")
    args = parser.parse_args()

    device = torch.device(args.device)
    scale = _load_norm_scale(args.norm_params)
    adata = ad.read_h5ad(args.input_h5ad)
    times = np.asarray(adata.obs[args.time_key], dtype=float)
    x_raw = np.asarray(adata.obsm[args.latent_key], dtype="float32")
    unique_times = np.array(sorted(np.unique(times)), dtype=float)
    source_time = float(unique_times.min())
    terminal_time = float(unique_times.max())

    x_by_time_norm = [x_raw[np.isclose(times, t)] / scale for t in unique_times]
    x0_norm = x_by_time_norm[0]
    terminal_norm = x_by_time_norm[-1]

    print(f"eval_space: normalized PCA, scale={scale:.12g}")
    print(f"time_points: {unique_times.tolist()}")
    print(f"initial_cells: {x0_norm.shape[0]}, terminal_cells: {terminal_norm.shape[0]}")

    pred_path, model_meta = mioflow_terminal_path(
        x0_norm=x0_norm,
        model_path=args.mioflow_model,
        norm_scale=scale,
        source_time=source_time,
        terminal_time=terminal_time,
        device=device,
        batch_size=args.batch_size,
        steps_per_interval=args.steps_per_interval,
    )
    pred_terminal = pred_path[-1]
    energy = _approx_energy(pred_path, terminal_time - source_time)
    print(f"MIOFlow energy_mean: {energy:.6f}")

    blurs = [float(x) for x in args.geomloss_blurs.split(",") if x.strip()]
    sinkhorn_by_blur = {}
    real_by_blur = {}
    for blur in blurs:
        real_adjacent = [
            _sinkhorn_geomloss(x_by_time_norm[i], x_by_time_norm[i + 1], blur, args.geomloss_backend, device)
            for i in range(len(x_by_time_norm) - 1)
        ]
        real_initial_terminal = _sinkhorn_geomloss(x0_norm, terminal_norm, blur, args.geomloss_backend, device)
        pred_terminal_cost = _sinkhorn_geomloss(pred_terminal, terminal_norm, blur, args.geomloss_backend, device)
        sinkhorn_by_blur[str(blur)] = pred_terminal_cost
        real_by_blur[str(blur)] = {
            "adjacent_sinkhorn_costs": real_adjacent,
            "adjacent_sinkhorn_sum": float(np.sum(real_adjacent)),
            "initial_terminal_sinkhorn_cost": real_initial_terminal,
        }
        print(
            f"blur={blur:.6g} real_adjacent_sum={np.sum(real_adjacent):.6f} "
            f"real_initial_terminal={real_initial_terminal:.6f} "
            f"mioflow_terminal_sinkhorn={pred_terminal_cost:.6f}"
        )

    result = {
        "input_h5ad": str(args.input_h5ad),
        "latent_key": args.latent_key,
        "time_key": args.time_key,
        "eval_space": "normalized_pca",
        "norm_params": str(args.norm_params),
        "scale": scale,
        "time_points": unique_times.tolist(),
        "initial_cells": int(x0_norm.shape[0]),
        "terminal_cells": int(terminal_norm.shape[0]),
        "steps_per_interval": int(args.steps_per_interval),
        "geomloss_backend": args.geomloss_backend,
        "real_data_sanity_by_blur": real_by_blur,
        "models": {
            "mioflow": {
                "path": str(args.mioflow_model),
                "energy_mean": energy,
                "terminal_sinkhorn_cost_by_blur": sinkhorn_by_blur,
                **model_meta,
            }
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        x0_norm=x0_norm.astype("float32", copy=False),
        terminal_norm=terminal_norm.astype("float32", copy=False),
        predicted_terminal_norm=pred_terminal.astype("float32", copy=False),
        trajectory_norm=pred_path.astype("float32", copy=False),
        time_grid=np.linspace(source_time, terminal_time, pred_path.shape[0], dtype="float32"),
    )
    print(f"wrote: {args.output_json}")
    print(f"wrote: {args.output_npz}")


if __name__ == "__main__":
    main()
