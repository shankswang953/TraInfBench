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
import importlib.util
import json
import math
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from geomloss import SamplesLoss
from torchdiffeq import odeint


def _load_tigon_module():
    module_path = Path(__file__).resolve().parents[2] / "model" / "TIGON" / "run_tigon_moscot_ae.py"
    spec = importlib.util.spec_from_file_location("run_tigon_moscot_ae", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load TIGON module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_norm_scale(norm_params: Path) -> float:
    params = torch.load(norm_params, map_location="cpu", weights_only=False)
    if not isinstance(params, dict) or "scale" not in params:
        raise ValueError(f"Expected {norm_params} to contain a dict with key 'scale'")
    scale = float(np.asarray(params["scale"]).reshape(-1)[0])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid normalization scale: {scale}")
    return scale


def _deterministic_subsample(n: int, k: int) -> np.ndarray:
    if k <= 0 or k >= n:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, k).round().astype(int)


def _as_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype="float32"), device=device)


def _uniform_weights(n: int, device: torch.device) -> torch.Tensor:
    return torch.full((n,), 1.0 / n, dtype=torch.float32, device=device)


def _sinkhorn(
    loss_fn: SamplesLoss,
    pushed: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> float:
    with torch.no_grad():
        if weights is None:
            return float(loss_fn(pushed, target).detach().cpu())
        target_weights = _uniform_weights(target.shape[0], target.device)
        return float(loss_fn(weights, pushed, target_weights, target).detach().cpu())


def _load_space(input_h5ad: Path, time_key: str, latent_key: str, scale: float):
    adata = ad.read_h5ad(input_h5ad)
    if latent_key not in adata.obsm:
        raise KeyError(f"{latent_key!r} not found in {input_h5ad}. Available obsm: {list(adata.obsm.keys())}")
    times = pd.to_numeric(adata.obs[time_key], errors="raise").to_numpy().astype(np.float32)
    unique_times = np.asarray(sorted(np.unique(times)), dtype=np.float32)
    x_raw = np.asarray(adata.obsm[latent_key], dtype="float32")
    x_norm = (x_raw / scale).astype("float32", copy=False)
    x_by_time = [x_norm[np.isclose(times, t)] for t in unique_times]
    counts = [int(arr.shape[0]) for arr in x_by_time]
    return x_by_time, unique_times, counts


def _load_model(
    tigon_module,
    model_path: Path,
    config_path: Path,
    device: torch.device,
):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    state = torch.load(model_path, map_location=device, weights_only=False)
    args = state.get("args", {})
    model_dim = int(state.get("model_dim", config.get("model_dim", config.get("latent_dim", 0))))
    hidden_dim = int(config.get("tigon_hidden_dim", args.get("tigon_hidden_dim", 64)))
    hidden_layers = int(config.get("tigon_hidden_layers", args.get("tigon_hidden_layers", 4)))
    if model_dim <= 0:
        raise ValueError("Could not infer TIGON model dimension")
    model = tigon_module.TIGONModel(model_dim, hidden_dim, hidden_layers).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, config, state


def _integrate_normalized(
    model,
    x0_norm: torch.Tensor,
    t0: float,
    t1: float,
    steps: int,
    scale: float,
    model_input_space: str,
    ode_solver: str = "midpoint",
    ode_rtol: float = 1e-3,
    ode_atol: float = 1e-5,
):
    solver = ode_solver.lower()
    if solver in {"dopri5", "rk4", "euler"}:
        dim = int(x0_norm.shape[1])
        direction = 1.0 if t1 >= t0 else -1.0

        class _NormalizedEvaluationDynamics(torch.nn.Module):
            def forward(self, t, augmented):
                x_norm = augmented[:, :dim]
                if model_input_space == "raw-pca":
                    x_model = x_norm * scale
                    v_norm = model.velocity(t, x_model) / scale
                elif model_input_space == "normalized-pca":
                    x_model = x_norm
                    v_norm = model.velocity(t, x_model)
                else:
                    raise ValueError(f"Unexpected model_input_space: {model_input_space}")
                growth = model.growth(t, x_model)
                energy_rate = direction * 0.5 * torch.sum(v_norm * v_norm, dim=1, keepdim=True)
                growth_l2_rate = direction * growth * growth
                return torch.cat([v_norm, growth, energy_rate, growth_l2_rate], dim=1)

        zeros = torch.zeros(x0_norm.shape[0], 3, dtype=x0_norm.dtype, device=x0_norm.device)
        initial = torch.cat([x0_norm, zeros], dim=1)
        times = torch.tensor([float(t0), float(t1)], dtype=x0_norm.dtype, device=x0_norm.device)
        ode_kwargs = {
            "method": solver,
            "rtol": float(ode_rtol),
            "atol": float(ode_atol),
        }
        if solver in {"rk4", "euler"}:
            if steps <= 0:
                raise ValueError(f"steps must be positive for fixed-step {solver}")
            ode_kwargs["options"] = {"step_size": abs(float(t1 - t0)) / int(steps)}
        final = odeint(_NormalizedEvaluationDynamics(), initial, times, **ode_kwargs)[-1]
        return final[:, :dim], final[:, dim], final[:, dim + 1], final[:, dim + 2]

    if ode_solver.lower() not in {"midpoint", "midpoint-euler", "fixed-midpoint"}:
        raise ValueError(f"Unsupported evaluation ODE solver: {ode_solver}")
    dt = float(t1 - t0) / int(steps)
    x_norm = x0_norm
    log_growth = torch.zeros(x_norm.shape[0], 1, dtype=x_norm.dtype, device=x_norm.device)
    energy_half = torch.zeros(x_norm.shape[0], dtype=x_norm.dtype, device=x_norm.device)
    growth_l2 = torch.zeros(x_norm.shape[0], dtype=x_norm.dtype, device=x_norm.device)
    for i in range(int(steps)):
        t = float(t0) + (i + 0.5) * dt
        if model_input_space == "raw-pca":
            x_model = x_norm * scale
            v_norm = model.velocity(t, x_model) / scale
        elif model_input_space == "normalized-pca":
            x_model = x_norm
            v_norm = model.velocity(t, x_model)
        else:
            raise ValueError(f"Unexpected model_input_space: {model_input_space}")
        g = model.growth(t, x_model)
        x_norm = x_norm + dt * v_norm
        log_growth = log_growth + dt * g
        energy_half = energy_half + abs(dt) * 0.5 * torch.sum(v_norm * v_norm, dim=1)
        growth_l2 = growth_l2 + abs(dt) * (g.squeeze(1) ** 2)
    return x_norm, log_growth.squeeze(1), energy_half, growth_l2


def _adjacent_eval(
    model,
    x_by_time: list[np.ndarray],
    unique_times: np.ndarray,
    counts: list[int],
    eval_samples: int,
    ode_steps: int,
    scale: float,
    model_input_space: str,
    ode_solver: str,
    ode_rtol: float,
    ode_atol: float,
    loss_fn: SamplesLoss,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    with torch.no_grad():
        for k in range(len(unique_times) - 1):
            src = x_by_time[k]
            tgt = x_by_time[k + 1]
            n_src = min(eval_samples, src.shape[0])
            n_tgt = min(eval_samples, tgt.shape[0])
            src_idx = _deterministic_subsample(src.shape[0], n_src)
            tgt_idx = _deterministic_subsample(tgt.shape[0], n_tgt)
            x0 = _as_tensor(src[src_idx], device)
            y = _as_tensor(tgt[tgt_idx], device)
            pred, log_growth, energy_half, growth_l2 = _integrate_normalized(
                model,
                x0,
                float(unique_times[k]),
                float(unique_times[k + 1]),
                ode_steps,
                scale,
                model_input_space,
                ode_solver,
                ode_rtol,
                ode_atol,
            )
            weights = torch.softmax(log_growth.clamp(-12, 12), dim=0)
            rows.append(
                {
                    "from_time": float(unique_times[k]),
                    "to_time": float(unique_times[k + 1]),
                    "n_source_eval": int(n_src),
                    "n_target_eval": int(n_tgt),
                    "observed_count_ratio": counts[k + 1] / counts[k],
                    "observed_log_count_ratio": math.log(counts[k + 1] / counts[k]),
                    "mean_integrated_log_growth": float(log_growth.mean().detach().cpu()),
                    "median_integrated_log_growth": float(log_growth.median().detach().cpu()),
                    "mean_mass_factor": float(torch.mean(torch.exp(log_growth.clamp(-12, 12))).detach().cpu()),
                    "sinkhorn_weighted_growth_norm": _sinkhorn(loss_fn, pred, y, weights),
                    "sinkhorn_unweighted_norm": _sinkhorn(loss_fn, pred, y, None),
                    "mean_energy_half_norm": float(energy_half.mean().detach().cpu()),
                    "mean_energy_trainf_norm": float((2.0 * energy_half.mean()).detach().cpu()),
                    "mean_growth_l2": float(growth_l2.mean().detach().cpu()),
                }
            )
    return pd.DataFrame(rows)


def _rollout_eval(
    model,
    x_by_time: list[np.ndarray],
    unique_times: np.ndarray,
    eval_samples: int,
    ode_steps: int,
    scale: float,
    model_input_space: str,
    ode_solver: str,
    ode_rtol: float,
    ode_atol: float,
    loss_fn: SamplesLoss,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    src = x_by_time[0]
    n_src = min(eval_samples, src.shape[0])
    src_idx = _deterministic_subsample(src.shape[0], n_src)
    x = _as_tensor(src[src_idx], device)
    cumulative_log_growth = torch.zeros(x.shape[0], dtype=x.dtype, device=device)
    cumulative_energy_half = torch.zeros(x.shape[0], dtype=x.dtype, device=device)
    cumulative_growth_l2 = torch.zeros(x.shape[0], dtype=x.dtype, device=device)
    rows = []
    predictions: dict[str, np.ndarray] = {"source_indices": src_idx.astype("int64")}

    with torch.no_grad():
        for k in range(len(unique_times) - 1):
            x, log_growth, energy_half, growth_l2 = _integrate_normalized(
                model,
                x,
                float(unique_times[k]),
                float(unique_times[k + 1]),
                ode_steps,
                scale,
                model_input_space,
                ode_solver,
                ode_rtol,
                ode_atol,
            )
            cumulative_log_growth = cumulative_log_growth + log_growth
            cumulative_energy_half = cumulative_energy_half + energy_half
            cumulative_growth_l2 = cumulative_growth_l2 + growth_l2

            target = x_by_time[k + 1]
            n_tgt = min(eval_samples, target.shape[0])
            target_idx = _deterministic_subsample(target.shape[0], n_tgt)
            y = _as_tensor(target[target_idx], device)
            weights = torch.softmax(cumulative_log_growth.clamp(-12, 12), dim=0)
            time_value = float(unique_times[k + 1])
            predictions[f"rollout_to_{time_value:g}_norm"] = x.detach().cpu().numpy().astype("float32", copy=False)
            predictions[f"rollout_to_{time_value:g}_cumulative_log_growth"] = (
                cumulative_log_growth.detach().cpu().numpy().astype("float32", copy=False)
            )
            rows.append(
                {
                    "source_time": float(unique_times[0]),
                    "to_time": time_value,
                    "n_source_eval": int(n_src),
                    "n_target_eval": int(n_tgt),
                    "sinkhorn_weighted_growth_norm": _sinkhorn(loss_fn, x, y, weights),
                    "sinkhorn_unweighted_norm": _sinkhorn(loss_fn, x, y, None),
                    "mean_cumulative_log_growth": float(cumulative_log_growth.mean().detach().cpu()),
                    "median_cumulative_log_growth": float(cumulative_log_growth.median().detach().cpu()),
                    "mean_cumulative_mass_factor": float(
                        torch.mean(torch.exp(cumulative_log_growth.clamp(-12, 12))).detach().cpu()
                    ),
                    "mean_cumulative_energy_half_norm": float(cumulative_energy_half.mean().detach().cpu()),
                    "mean_cumulative_energy_trainf_norm": float((2.0 * cumulative_energy_half.mean()).detach().cpu()),
                    "mean_cumulative_growth_l2": float(cumulative_growth_l2.mean().detach().cpu()),
                }
            )
    return pd.DataFrame(rows), predictions


def _real_data_sanity(
    x_by_time: list[np.ndarray],
    unique_times: np.ndarray,
    eval_samples: int,
    loss_fn: SamplesLoss,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    with torch.no_grad():
        for k in range(len(unique_times) - 1):
            a = x_by_time[k]
            b = x_by_time[k + 1]
            ia = _deterministic_subsample(a.shape[0], min(eval_samples, a.shape[0]))
            ib = _deterministic_subsample(b.shape[0], min(eval_samples, b.shape[0]))
            x = _as_tensor(a[ia], device)
            y = _as_tensor(b[ib], device)
            rows.append(
                {
                    "from_time": float(unique_times[k]),
                    "to_time": float(unique_times[k + 1]),
                    "n_source_eval": int(ia.shape[0]),
                    "n_target_eval": int(ib.shape[0]),
                    "observed_adjacent_sinkhorn_norm": _sinkhorn(loss_fn, x, y, None),
                }
            )
    return pd.DataFrame(rows)


def _infer_model_input_space(config: dict, requested: str) -> str:
    if requested != "auto":
        return requested
    if config.get("embedding_normalization") == "global-scale":
        return "normalized-pca"
    if config.get("model_input_space") == "normalized-pca":
        return "normalized-pca"
    return "raw-pca"


def _infer_ode_settings(config: dict, state: dict, requested_solver: str) -> tuple[str, float, float]:
    training_objective = state.get("training_objective", config.get("training_objective"))
    if requested_solver == "auto":
        solver = str(state.get("ode_solver", config.get("ode_solver", "dopri5")))
        if training_objective != "tigon-density":
            solver = "midpoint"
    else:
        solver = requested_solver
    rtol = float(state.get("ode_rtol", config.get("ode_rtol", 1e-3)) or 1e-3)
    atol = float(state.get("ode_atol", config.get("ode_atol", 1e-5)) or 1e-5)
    return solver.lower(), rtol, atol


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained TIGON in normalized PCA space.")
    parser.add_argument("--input-h5ad", type=Path, default=Path("data/gastrulation_rna_cytobridge.h5ad"))
    parser.add_argument("--time-key", default="time_point_processed")
    parser.add_argument("--latent-key", default="X_latent")
    parser.add_argument("--norm-params", type=Path, default=Path("data/gastrulation_rna_primal_norm_params.pt"))
    parser.add_argument("--tigon-model", type=Path, default=Path("results/tigon_gastrulation_rna_20000_ae/tigon.pt"))
    parser.add_argument("--config", type=Path, default=Path("results/tigon_gastrulation_rna_20000_ae/config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tigon_gastrulation_rna_20000_ae"))
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--ode-steps", type=int, default=None)
    parser.add_argument(
        "--ode-solver",
        choices=("auto", "midpoint", "dopri5", "rk4", "euler"),
        default="auto",
        help="auto reuses the solver saved by tigon-density checkpoints and uses midpoint for legacy checkpoints.",
    )
    parser.add_argument("--ode-rtol", type=float, default=None)
    parser.add_argument("--ode-atol", type=float, default=None)
    parser.add_argument("--geomloss-blur", type=float, default=1e-4)
    parser.add_argument("--geomloss-backend", default="tensorized")
    parser.add_argument(
        "--model-input-space",
        choices=("auto", "raw-pca", "normalized-pca"),
        default="auto",
        help=(
            "How the TIGON model consumes PCA coordinates. auto uses "
            "config['embedding_normalization']=='global-scale' to detect normalized-pca."
        ),
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)
    scale = _load_norm_scale(args.norm_params)
    x_by_time, unique_times, counts = _load_space(args.input_h5ad, args.time_key, args.latent_key, scale)
    tigon_module = _load_tigon_module()
    model, config, state = _load_model(tigon_module, args.tigon_model, args.config, device)
    model_input_space = _infer_model_input_space(config, args.model_input_space)
    ode_steps = int(args.ode_steps if args.ode_steps is not None else config.get("ode_steps", 8))
    ode_solver, ode_rtol, ode_atol = _infer_ode_settings(config, state, args.ode_solver)
    if args.ode_rtol is not None:
        ode_rtol = float(args.ode_rtol)
    if args.ode_atol is not None:
        ode_atol = float(args.ode_atol)
    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=args.geomloss_blur, backend=args.geomloss_backend).to(device)

    adjacent = _adjacent_eval(
        model=model,
        x_by_time=x_by_time,
        unique_times=unique_times,
        counts=counts,
        eval_samples=args.eval_samples,
        ode_steps=ode_steps,
        scale=scale,
        model_input_space=model_input_space,
        ode_solver=ode_solver,
        ode_rtol=ode_rtol,
        ode_atol=ode_atol,
        loss_fn=loss_fn,
        device=device,
    )
    rollout, predictions = _rollout_eval(
        model=model,
        x_by_time=x_by_time,
        unique_times=unique_times,
        eval_samples=args.eval_samples,
        ode_steps=ode_steps,
        scale=scale,
        model_input_space=model_input_space,
        ode_solver=ode_solver,
        ode_rtol=ode_rtol,
        ode_atol=ode_atol,
        loss_fn=loss_fn,
        device=device,
    )
    sanity = _real_data_sanity(x_by_time, unique_times, args.eval_samples, loss_fn, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adjacent_path = args.output_dir / "normalized_interval_eval.csv"
    rollout_path = args.output_dir / "normalized_rollout_eval.csv"
    sanity_path = args.output_dir / "normalized_real_data_sanity.csv"
    summary_path = args.output_dir / "normalized_eval_summary.json"
    predictions_path = args.output_dir / "normalized_rollout_predictions.npz"

    adjacent.to_csv(adjacent_path, index=False)
    rollout.to_csv(rollout_path, index=False)
    sanity.to_csv(sanity_path, index=False)
    np.savez_compressed(predictions_path, **predictions)

    summary = {
        "input_h5ad": str(args.input_h5ad),
        "latent_key": args.latent_key,
        "time_key": args.time_key,
        "time_points": [float(t) for t in unique_times],
        "counts_by_time": counts,
        "norm_params": str(args.norm_params),
        "scale": scale,
        "tigon_model": str(args.tigon_model),
        "config": str(args.config),
        "model_dim": int(state.get("model_dim", config.get("model_dim", 0))),
        "model_input_space": model_input_space,
        "eval_samples": int(args.eval_samples),
        "ode_steps": int(ode_steps),
        "ode_solver": ode_solver,
        "ode_rtol": ode_rtol,
        "ode_atol": ode_atol,
        "geomloss_blur": float(args.geomloss_blur),
        "geomloss_backend": args.geomloss_backend,
        "energy_half_convention": "integral 0.5 * ||v_normalized(t)||^2 dt",
        "energy_trainf_convention": "integral ||v_normalized(t)||^2 dt",
        "adjacent_eval_csv": str(adjacent_path),
        "rollout_eval_csv": str(rollout_path),
        "real_data_sanity_csv": str(sanity_path),
        "rollout_predictions_npz": str(predictions_path),
        "terminal_rollout": rollout.iloc[-1].to_dict() if len(rollout) else {},
        "adjacent_eval": adjacent.to_dict(orient="records"),
        "real_data_sanity": sanity.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"scale: {scale:.12g}")
    print(f"time_points: {[float(t) for t in unique_times]}")
    print(f"ode: {ode_solver} (rtol={ode_rtol:g}, atol={ode_atol:g})")
    print(f"wrote: {adjacent_path}")
    print(f"wrote: {rollout_path}")
    print(f"wrote: {sanity_path}")
    print(f"wrote: {summary_path}")
    print(f"wrote: {predictions_path}")
    print("\n[adjacent]")
    print(adjacent.to_string(index=False))
    print("\n[rollout]")
    print(rollout.to_string(index=False))


if __name__ == "__main__":
    main()
