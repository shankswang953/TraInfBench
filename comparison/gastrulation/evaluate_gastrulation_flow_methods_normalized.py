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
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import anndata as ad
import numpy as np
import pandas as pd
import torch
from geomloss import SamplesLoss
from torchdiffeq import odeint

ROOT = Path(__file__).resolve().parents[2]
for local_pkg in [ROOT / "external" / "CytoBridge", ROOT / "external" / "MIOFlow"]:
    if local_pkg.exists():
        sys.path.insert(0, str(local_pkg))

from evaluate_mioflow_terminal_push import _decode_gaga_path, _encode_gaga, _load_gaga_model


def _load_norm_scale(norm_params: Path) -> float:
    params = torch.load(norm_params, map_location="cpu", weights_only=False)
    if not isinstance(params, dict) or "scale" not in params:
        raise ValueError(f"Expected {norm_params} to contain a dict with key 'scale'")
    scale = float(np.asarray(params["scale"]).reshape(-1)[0])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid normalization scale: {scale}")
    return scale


def _as_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype="float32"), device=device)


def _batched_slices(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield slice(start, min(start + batch_size, n))


def _deterministic_subsample(n: int, k: int) -> np.ndarray:
    if k <= 0 or k >= n:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, k).round().astype(int)


def _sinkhorn(loss_fn: SamplesLoss, pushed: np.ndarray, target: np.ndarray, device: torch.device) -> float:
    with torch.no_grad():
        return float(loss_fn(_as_tensor(pushed, device), _as_tensor(target, device)).detach().cpu())


def _energy_from_path(path_norm: np.ndarray, duration: float) -> np.ndarray:
    if path_norm.shape[0] < 2:
        return np.zeros(path_norm.shape[1], dtype=np.float32)
    dt = float(duration) / float(path_norm.shape[0] - 1)
    velocity = (path_norm[1:] - path_norm[:-1]) / dt
    per_step = 0.5 * np.sum(velocity * velocity, axis=2) * abs(dt)
    return np.sum(per_step, axis=0).astype("float32", copy=False)


def _load_normalized_space(input_h5ad: Path, time_key: str, latent_key: str, scale: float):
    adata = ad.read_h5ad(input_h5ad)
    if latent_key not in adata.obsm:
        raise KeyError(f"{latent_key!r} not found in {input_h5ad}. Available obsm: {list(adata.obsm.keys())}")
    times = pd.to_numeric(adata.obs[time_key], errors="raise").to_numpy().astype(np.float32)
    unique_times = np.asarray(sorted(np.unique(times)), dtype=np.float32)
    x_raw = np.asarray(adata.obsm[latent_key], dtype="float32")
    x_norm = (x_raw / scale).astype("float32", copy=False)
    x_by_time = [x_norm[np.isclose(times, t)] for t in unique_times]
    counts = [int(x.shape[0]) for x in x_by_time]
    return x_by_time, unique_times, counts


def _real_data_sanity(
    x_by_time: list[np.ndarray],
    unique_times: np.ndarray,
    eval_samples: int,
    loss_fn: SamplesLoss,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    for k in range(len(unique_times) - 1):
        src = x_by_time[k]
        tgt = x_by_time[k + 1]
        src_idx = _deterministic_subsample(src.shape[0], min(eval_samples, src.shape[0]))
        tgt_idx = _deterministic_subsample(tgt.shape[0], min(eval_samples, tgt.shape[0]))
        rows.append(
            {
                "from_time": float(unique_times[k]),
                "to_time": float(unique_times[k + 1]),
                "n_source_eval": int(src_idx.shape[0]),
                "n_target_eval": int(tgt_idx.shape[0]),
                "observed_adjacent_sinkhorn_norm": _sinkhorn(
                    loss_fn,
                    src[src_idx],
                    tgt[tgt_idx],
                    device,
                ),
            }
        )
    return pd.DataFrame(rows)


def _load_cytobridge_model(cytobridge_adata: Path, device: torch.device):
    from CytoBridge.utils import load_model_from_adata

    adata = ad.read_h5ad(cytobridge_adata)
    model = load_model_from_adata(adata).to(device)
    model.eval()
    return model


def _cytobridge_velocity(model, t_value: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
    t = t_value.expand(x_raw.shape[0], 1)
    velocity = model.velocity_net(torch.cat([x_raw, t], dim=1))
    if "score" in model.components:
        with torch.enable_grad():
            _, score_grad = model.compute_score(t_value, x_raw)
        velocity = velocity + score_grad
    return velocity


def _cytobridge_interval_path(
    model,
    x0_norm: np.ndarray,
    t0: float,
    t1: float,
    steps: int,
    scale: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    paths = []
    dt = float(t1 - t0) / int(steps)
    for batch_slice in _batched_slices(x0_norm.shape[0], batch_size):
        x = _as_tensor(x0_norm[batch_slice], device)
        states = [x.detach().cpu().numpy()]
        with torch.no_grad():
            for i in range(int(steps)):
                t_mid = torch.tensor([float(t0) + (i + 0.5) * dt], dtype=torch.float32, device=device)
                v_raw = _cytobridge_velocity(model, t_mid, x * scale)
                x = x + dt * (v_raw / scale)
                states.append(x.detach().cpu().numpy())
        paths.append(np.stack(states, axis=0))
    return np.concatenate(paths, axis=1).astype("float32", copy=False)


@dataclass
class MIOFlowBundle:
    model: torch.nn.Module
    mean_vals: np.ndarray
    std_vals: np.ndarray
    input_space: str
    use_gaga: bool
    gaga_model: object | None
    model_times: np.ndarray


def _load_mioflow_bundle(model_path: Path, device: torch.device) -> MIOFlowBundle:
    from mioflow.core.models.ode_model import ODEFunc

    state = torch.load(model_path, map_location=device, weights_only=False)
    model = ODEFunc(
        input_dim=int(state["input_dim"]),
        hidden_dim=int(state["hidden_dim"]),
        momentum_beta=float(state.get("momentum_beta", 0.0)),
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    use_gaga = bool(state.get("use_gaga", False))
    gaga_model = _load_gaga_model(Path(state["gaga_model_path"]), device) if use_gaga else None
    model_times = np.asarray([float(t) for t in state.get("time_points", [])], dtype=np.float32)
    if model_times.size < 2:
        raise ValueError(f"No usable MIOFlow time_points in {model_path}")
    return MIOFlowBundle(
        model=model,
        mean_vals=state["mean_vals"].detach().cpu().numpy().astype("float32"),
        std_vals=state["std_vals"].detach().cpu().numpy().astype("float32"),
        input_space=str(state.get("input_space", "obsm")),
        use_gaga=use_gaga,
        gaga_model=gaga_model,
        model_times=model_times,
    )


def _mioflow_xnorm_to_model_space(
    x_norm: np.ndarray,
    bundle: MIOFlowBundle,
    norm_scale: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if bundle.input_space == "moscot-normalized":
        x_input = x_norm.astype("float32", copy=False)
    else:
        x_input = (x_norm * norm_scale).astype("float32", copy=False)
    if bundle.use_gaga:
        z_input = _encode_gaga(x_input, bundle.gaga_model, device, batch_size)
    else:
        z_input = x_input
    return ((z_input - bundle.mean_vals) / bundle.std_vals).astype("float32", copy=False)


def _mioflow_model_path_to_xnorm(
    path_model: np.ndarray,
    bundle: MIOFlowBundle,
    norm_scale: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    path_input = path_model * bundle.std_vals + bundle.mean_vals
    if bundle.use_gaga:
        path_input = _decode_gaga_path(path_input, bundle.gaga_model, device, batch_size)
    if bundle.input_space == "moscot-normalized":
        return path_input.astype("float32", copy=False)
    return (path_input / norm_scale).astype("float32", copy=False)


def _mioflow_interval_path(
    bundle: MIOFlowBundle,
    x0_norm: np.ndarray,
    interval_idx: int,
    steps: int,
    norm_scale: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    z0 = _mioflow_xnorm_to_model_space(x0_norm, bundle, norm_scale, device, batch_size)
    tau0 = float(bundle.model_times[interval_idx])
    tau1 = float(bundle.model_times[interval_idx + 1])
    ts = torch.linspace(tau0, tau1, int(steps) + 1, device=device, dtype=torch.float32)
    paths = []
    for batch_slice in _batched_slices(z0.shape[0], batch_size):
        x = _as_tensor(z0[batch_slice], device)
        if hasattr(bundle.model, "reset_momentum"):
            bundle.model.reset_momentum()
        with torch.no_grad():
            path = odeint(bundle.model, x, ts)
        paths.append(path.detach().cpu().numpy())
    path_model = np.concatenate(paths, axis=1).astype("float32", copy=False)
    return _mioflow_model_path_to_xnorm(path_model, bundle, norm_scale, device, batch_size)


def _evaluate_method(
    method: str,
    interval_path_fn,
    x_by_time: list[np.ndarray],
    unique_times: np.ndarray,
    counts: list[int],
    eval_samples: int,
    steps_per_interval: int,
    loss_fn: SamplesLoss,
    device: torch.device,
    output_dir: Path,
    extra_summary: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    adjacent_rows = []
    rollout_rows = []
    rollout_predictions = {"source_indices": _deterministic_subsample(x_by_time[0].shape[0], min(eval_samples, x_by_time[0].shape[0]))}

    for k in range(len(unique_times) - 1):
        src = x_by_time[k]
        tgt = x_by_time[k + 1]
        src_idx = _deterministic_subsample(src.shape[0], min(eval_samples, src.shape[0]))
        tgt_idx = _deterministic_subsample(tgt.shape[0], min(eval_samples, tgt.shape[0]))
        path = interval_path_fn(src[src_idx], k)
        pred = path[-1]
        energy = _energy_from_path(path, float(unique_times[k + 1] - unique_times[k]))
        adjacent_rows.append(
            {
                "method": method,
                "from_time": float(unique_times[k]),
                "to_time": float(unique_times[k + 1]),
                "n_source_eval": int(src_idx.shape[0]),
                "n_target_eval": int(tgt_idx.shape[0]),
                "observed_count_ratio": counts[k + 1] / counts[k],
                "sinkhorn_unweighted_norm": _sinkhorn(loss_fn, pred, tgt[tgt_idx], device),
                "mean_energy_half_norm": float(np.mean(energy)),
                "mean_energy_trainf_norm": float(2.0 * np.mean(energy)),
            }
        )

    x_roll = x_by_time[0][rollout_predictions["source_indices"]]
    cumulative_energy = np.zeros(x_roll.shape[0], dtype=np.float32)
    for k in range(len(unique_times) - 1):
        tgt = x_by_time[k + 1]
        tgt_idx = _deterministic_subsample(tgt.shape[0], min(eval_samples, tgt.shape[0]))
        path = interval_path_fn(x_roll, k)
        x_roll = path[-1]
        cumulative_energy += _energy_from_path(path, float(unique_times[k + 1] - unique_times[k]))
        time_value = float(unique_times[k + 1])
        rollout_predictions[f"{method}_rollout_to_{time_value:g}_norm"] = x_roll.astype("float32", copy=False)
        rollout_rows.append(
            {
                "method": method,
                "source_time": float(unique_times[0]),
                "to_time": time_value,
                "n_source_eval": int(x_roll.shape[0]),
                "n_target_eval": int(tgt_idx.shape[0]),
                "sinkhorn_unweighted_norm": _sinkhorn(loss_fn, x_roll, tgt[tgt_idx], device),
                "mean_cumulative_energy_half_norm": float(np.mean(cumulative_energy)),
                "mean_cumulative_energy_trainf_norm": float(2.0 * np.mean(cumulative_energy)),
            }
        )

    adjacent = pd.DataFrame(adjacent_rows)
    rollout = pd.DataFrame(rollout_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    adjacent_path = output_dir / "normalized_interval_eval.csv"
    rollout_path = output_dir / "normalized_rollout_eval.csv"
    pred_path = output_dir / "normalized_rollout_predictions.npz"
    summary_path = output_dir / "normalized_eval_summary.json"
    adjacent.to_csv(adjacent_path, index=False)
    rollout.to_csv(rollout_path, index=False)
    np.savez_compressed(pred_path, **rollout_predictions)

    summary = {
        "method": method,
        "time_points": [float(t) for t in unique_times],
        "counts_by_time": counts,
        "eval_samples": int(eval_samples),
        "steps_per_interval": int(steps_per_interval),
        "energy_half_convention": "integral 0.5 * ||v_normalized(t)||^2 dt over physical time",
        "energy_trainf_convention": "integral ||v_normalized(t)||^2 dt over physical time",
        "interval_eval_csv": str(adjacent_path),
        "rollout_eval_csv": str(rollout_path),
        "rollout_predictions_npz": str(pred_path),
        "terminal_rollout": rollout.iloc[-1].to_dict() if len(rollout) else {},
        "interval_eval": adjacent.to_dict(orient="records"),
        **extra_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return adjacent, rollout


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CytoBridge and MIOFlow in normalized gastrulation PCA space.")
    parser.add_argument("--input-h5ad", type=Path, default=Path("data/gastrulation_rna_cytobridge.h5ad"))
    parser.add_argument("--time-key", default="time_point_processed")
    parser.add_argument("--latent-key", default="X_latent")
    parser.add_argument("--norm-params", type=Path, default=Path("data/gastrulation_rna_primal_norm_params.pt"))
    parser.add_argument("--cytobridge-adata", type=Path, default=Path("results/cytobridge_gastrulation_rna_20000/adata.h5ad"))
    parser.add_argument("--mioflow-model", type=Path, default=Path("results/mioflow_gastrulation_rna_20000/model.pt"))
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--steps-per-interval", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--geomloss-blur", type=float, default=1e-4)
    parser.add_argument("--geomloss-backend", default="tensorized")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)
    scale = _load_norm_scale(args.norm_params)
    x_by_time, unique_times, counts = _load_normalized_space(args.input_h5ad, args.time_key, args.latent_key, scale)
    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=args.geomloss_blur, backend=args.geomloss_backend).to(device)

    sanity = _real_data_sanity(x_by_time, unique_times, args.eval_samples, loss_fn, device)
    sanity.to_csv("results/gastrulation_normalized_real_data_sanity.csv", index=False)

    cb_model = _load_cytobridge_model(args.cytobridge_adata, device)

    def cb_interval(x0_norm: np.ndarray, interval_idx: int) -> np.ndarray:
        return _cytobridge_interval_path(
            cb_model,
            x0_norm,
            float(unique_times[interval_idx]),
            float(unique_times[interval_idx + 1]),
            args.steps_per_interval,
            scale,
            args.batch_size,
            device,
        )

    cb_adjacent, cb_rollout = _evaluate_method(
        method="cytobridge",
        interval_path_fn=cb_interval,
        x_by_time=x_by_time,
        unique_times=unique_times,
        counts=counts,
        eval_samples=args.eval_samples,
        steps_per_interval=args.steps_per_interval,
        loss_fn=loss_fn,
        device=device,
        output_dir=args.cytobridge_adata.parent,
        extra_summary={
            "input_h5ad": str(args.input_h5ad),
            "latent_key": args.latent_key,
            "time_key": args.time_key,
            "norm_params": str(args.norm_params),
            "scale": scale,
            "geomloss_blur": float(args.geomloss_blur),
            "geomloss_backend": args.geomloss_backend,
            "cytobridge_adata": str(args.cytobridge_adata),
            "real_data_sanity_csv": "results/gastrulation_normalized_real_data_sanity.csv",
        },
    )

    mf_bundle = _load_mioflow_bundle(args.mioflow_model, device)
    if mf_bundle.model_times.shape[0] != unique_times.shape[0]:
        raise ValueError(
            "MIOFlow model time grid and physical time grid have different lengths: "
            f"{mf_bundle.model_times.tolist()} vs {unique_times.tolist()}"
        )

    def mf_interval(x0_norm: np.ndarray, interval_idx: int) -> np.ndarray:
        return _mioflow_interval_path(
            mf_bundle,
            x0_norm,
            interval_idx,
            args.steps_per_interval,
            scale,
            args.batch_size,
            device,
        )

    mf_adjacent, mf_rollout = _evaluate_method(
        method="mioflow",
        interval_path_fn=mf_interval,
        x_by_time=x_by_time,
        unique_times=unique_times,
        counts=counts,
        eval_samples=args.eval_samples,
        steps_per_interval=args.steps_per_interval,
        loss_fn=loss_fn,
        device=device,
        output_dir=args.mioflow_model.parent,
        extra_summary={
            "input_h5ad": str(args.input_h5ad),
            "latent_key": args.latent_key,
            "time_key": args.time_key,
            "norm_params": str(args.norm_params),
            "scale": scale,
            "geomloss_blur": float(args.geomloss_blur),
            "geomloss_backend": args.geomloss_backend,
            "mioflow_model": str(args.mioflow_model),
            "mioflow_model_time_points": [float(t) for t in mf_bundle.model_times],
            "mioflow_input_space": mf_bundle.input_space,
            "mioflow_use_gaga": bool(mf_bundle.use_gaga),
            "real_data_sanity_csv": "results/gastrulation_normalized_real_data_sanity.csv",
            "time_mapping_note": "MIOFlow internal times are mapped index-wise to physical gastrulation times.",
        },
    )

    combined_interval = pd.concat([cb_adjacent, mf_adjacent], ignore_index=True)
    combined_rollout = pd.concat([cb_rollout, mf_rollout], ignore_index=True)
    combined_interval.to_csv("results/gastrulation_flow_methods_normalized_interval_eval.csv", index=False)
    combined_rollout.to_csv("results/gastrulation_flow_methods_normalized_rollout_eval.csv", index=False)

    print(f"scale: {scale:.12g}")
    print(f"time_points: {[float(t) for t in unique_times]}")
    print("\n[real data sanity]")
    print(sanity.to_string(index=False))
    print("\n[adjacent]")
    print(combined_interval.to_string(index=False))
    print("\n[rollout]")
    print(combined_rollout.to_string(index=False))
    print("wrote: results/gastrulation_flow_methods_normalized_interval_eval.csv")
    print("wrote: results/gastrulation_flow_methods_normalized_rollout_eval.csv")


if __name__ == "__main__":
    main()
