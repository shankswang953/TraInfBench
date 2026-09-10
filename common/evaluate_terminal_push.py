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
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import ot
import torch
from torchdiffeq import odeint


def _as_float_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype="float32"), device=device)


def _batched_indices(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield slice(start, min(start + batch_size, n))


def _pot_sinkhorn_cost(
    pushed: np.ndarray,
    target: np.ndarray,
    reg: float,
    metric: str,
    num_iter: int,
) -> float:
    a = np.full(pushed.shape[0], 1.0 / pushed.shape[0], dtype=np.float64)
    b = np.full(target.shape[0], 1.0 / target.shape[0], dtype=np.float64)
    cost = ot.dist(
        np.asarray(pushed, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        metric=metric,
    )
    return float(ot.sinkhorn2(a, b, cost, reg=reg, numItermax=num_iter))


def _geomloss_sinkhorn_cost(
    pushed: np.ndarray,
    target: np.ndarray,
    blur: float,
    backend: str,
    device: torch.device,
) -> float:
    from geomloss import SamplesLoss

    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=blur, backend=backend).to(device)
    x = _as_float_tensor(pushed, device)
    y = _as_float_tensor(target, device)
    with torch.no_grad():
        return float(loss_fn(x, y).item())


def _sinkhorn_cost(
    pushed: np.ndarray,
    target: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    reg: float | None,
) -> float:
    if args.sinkhorn_engine == "geomloss":
        return _geomloss_sinkhorn_cost(
            pushed,
            target,
            blur=args.geomloss_blur,
            backend=args.geomloss_backend,
            device=device,
        )
    if reg is None:
        raise ValueError("POT Sinkhorn requires a non-null regularization value")
    return _pot_sinkhorn_cost(
        pushed,
        target,
        reg=reg,
        metric=args.sinkhorn_metric,
        num_iter=args.sinkhorn_num_iter,
    )


def _load_space(input_h5ad: Path, time_key: str, latent_key: str):
    adata = ad.read_h5ad(input_h5ad)
    times = np.asarray(adata.obs[time_key], dtype=float)
    unique_times = np.array(sorted(np.unique(times)), dtype=float)
    if unique_times.size < 2:
        raise ValueError(f"Need at least two time points, got {unique_times.tolist()}")

    x = np.asarray(adata.obsm[latent_key], dtype="float32")
    x_by_time = [x[times == t] for t in unique_times]
    return x_by_time, unique_times


def _load_norm_scale(norm_params: Path | None) -> float:
    if norm_params is None:
        return 1.0
    params = torch.load(norm_params, map_location="cpu", weights_only=False)
    if not isinstance(params, dict) or "scale" not in params:
        raise ValueError(f"Expected {norm_params} to contain a dict with key 'scale'")
    return float(params["scale"])


def _load_trajectorynet_model(
    checkpoint: Path,
    dim: int,
    device: torch.device,
    solver: str,
    step_size: float,
):
    from TrajectoryNet.train_misc import build_model_tabular, set_cnf_options

    args = SimpleNamespace(
        dims="64-64-64",
        layer_type="concatsquash",
        nonlinearity="tanh",
        divergence_fn="brute_force",
        residual=False,
        rademacher=False,
        num_blocks=1,
        time_scale=0.5,
        train_T=True,
        solver=solver,
        atol=1e-5,
        rtol=1e-5,
        step_size=step_size,
        test_solver=solver,
        test_atol=1e-5,
        test_rtol=1e-5,
        batch_norm=False,
        bn_lag=0.0,
    )
    model = build_model_tabular(args, dim, regularization_fns=()).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("state_dict", state))
    set_cnf_options(args, model)
    model.eval()
    return model, args


def _trajectorynet_diffeq(model):
    cnf = model.chain[0]
    odefunc = cnf.odefunc
    if hasattr(odefunc, "odefunc"):
        odefunc = odefunc.odefunc
    return odefunc.diffeq


def _trajectorynet_reverse_schedule(
    source_model_time: float,
    target_model_time: float,
    time_scale: float,
    steps_per_interval: int,
) -> list[tuple[float, float, int]]:
    """Build TrajectoryNet's native piecewise reverse-generation clock.

    Upstream training maps snapshot ``k + 1`` toward snapshot ``k`` over
    ``[(k + 1) * time_scale, (k + 2) * time_scale]``. Biological forward
    generation therefore traverses that interval in reverse, starting at the
    upper endpoint. A fractional target must use the corresponding prefix of
    that reverse interval; it cannot be represented by shifting an equal-length
    interval toward the lower endpoint.
    """

    source = float(source_model_time)
    target = float(target_model_time)
    if target < source:
        raise ValueError(
            f"Target model time {target:g} precedes source {source:g}"
        )
    rounded_source = round(source)
    if not np.isclose(source, rounded_source):
        raise ValueError(
            "TrajectoryNet segmented rollout requires an observed integer-rank "
            f"source, found {source:g}"
        )
    if steps_per_interval < 1:
        raise ValueError("steps_per_interval must be positive")

    schedule: list[tuple[float, float, int]] = []
    current = int(rounded_source)
    remaining = target - source
    while remaining > 1e-12:
        fraction = min(1.0, remaining)
        upper = (current + 2.0) * float(time_scale)
        lower = upper - fraction * float(time_scale)
        segment_steps = max(1, int(np.ceil(steps_per_interval * fraction)))
        schedule.append((upper, lower, segment_steps))
        remaining -= fraction
        current += 1
    return schedule


def push_trajectorynet(
    x0: np.ndarray,
    unique_times: np.ndarray,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    solver: str,
    step_size: float,
    energy_steps_per_interval: int,
    model_space_scale: float,
):
    model, model_args = _load_trajectorynet_model(
        checkpoint=checkpoint,
        dim=x0.shape[1],
        device=device,
        solver=solver,
        step_size=step_size,
    )
    diffeq = _trajectorynet_diffeq(model)

    def normalized_diffeq(t_value: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if model_space_scale == 1.0:
            return diffeq(t_value, y)
        return diffeq(t_value, y * model_space_scale) / model_space_scale

    pushed_batches = []
    energy_total = 0.0

    for batch_slice in _batched_indices(x0.shape[0], batch_size):
        x = _as_float_tensor(x0[batch_slice], device)
        batch_energy = torch.zeros(x.shape[0], device=device)

        with torch.no_grad():
            for idx in range(len(unique_times) - 1):
                source_internal = (unique_times[idx] + 1.0) * model_args.time_scale
                target_internal = (unique_times[idx + 1] + 1.0) * model_args.time_scale
                ts = torch.linspace(
                    target_internal,
                    source_internal,
                    energy_steps_per_interval + 1,
                    device=device,
                    dtype=torch.float32,
                )

                path = odeint(
                    lambda t, y: normalized_diffeq(t, y),
                    x,
                    ts,
                    method=solver,
                    options={"step_size": step_size},
                )
                for t_value, state in zip(ts[:-1], path[:-1]):
                    v_internal = normalized_diffeq(t_value, state)
                    v_physical_sq = (model_args.time_scale ** 2) * torch.sum(
                        v_internal * v_internal,
                        dim=1,
                    )
                    batch_energy += 0.5 * v_physical_sq / energy_steps_per_interval
                x = path[-1]

        pushed_batches.append(x.detach().cpu().numpy())
        energy_total += float(batch_energy.sum().detach().cpu())

    pushed = np.vstack(pushed_batches).astype("float32", copy=False)
    return pushed, energy_total / x0.shape[0]


def _load_cytobridge_model(cytobridge_adata: Path, device: torch.device):
    from CytoBridge.utils import load_model_from_adata

    adata = ad.read_h5ad(cytobridge_adata)
    model = load_model_from_adata(adata).to(device)
    model.eval()
    return model


def _cytobridge_velocity(model, t_value: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    t = t_value.expand(x.shape[0], 1)
    net_input = torch.cat([x, t], dim=1)
    velocity = model.velocity_net(net_input)
    if "score" in model.components:
        with torch.enable_grad():
            _, score_grad = model.compute_score(t_value, x)
        velocity = velocity + score_grad
    return velocity


def push_cytobridge(
    x0: np.ndarray,
    unique_times: np.ndarray,
    cytobridge_adata: Path,
    device: torch.device,
    batch_size: int,
    energy_steps_per_interval: int,
    model_space_scale: float,
):
    model = _load_cytobridge_model(cytobridge_adata, device)

    def normalized_velocity(t_value: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if model_space_scale == 1.0:
            return _cytobridge_velocity(model, t_value, y)
        return _cytobridge_velocity(model, t_value, y * model_space_scale) / model_space_scale

    pushed_batches = []
    energy_total = 0.0

    for batch_slice in _batched_indices(x0.shape[0], batch_size):
        x = _as_float_tensor(x0[batch_slice], device)
        batch_energy = torch.zeros(x.shape[0], device=device)

        with torch.no_grad():
            for idx in range(len(unique_times) - 1):
                ts = torch.linspace(
                    unique_times[idx],
                    unique_times[idx + 1],
                    energy_steps_per_interval + 1,
                    device=device,
                    dtype=torch.float32,
                )
                path = odeint(
                    lambda t, y: normalized_velocity(t.reshape(1), y),
                    x,
                    ts,
                    method="euler",
                    options={"step_size": 1.0 / energy_steps_per_interval},
                )
                for t_value, state in zip(ts[:-1], path[:-1]):
                    velocity = normalized_velocity(t_value.reshape(1), state)
                    batch_energy += 0.5 * torch.sum(velocity * velocity, dim=1) / energy_steps_per_interval
                x = path[-1]

        pushed_batches.append(x.detach().cpu().numpy())
        energy_total += float(batch_energy.sum().detach().cpu())

    pushed = np.vstack(pushed_batches).astype("float32", copy=False)
    return pushed, energy_total / x0.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate terminal push in a shared latent space.")
    parser.add_argument("--input-h5ad", type=Path, default=Path("data/moscot_rna_cytobridge.h5ad"))
    parser.add_argument("--time-key", default="time_point_processed")
    parser.add_argument("--latent-key", default="X_latent")
    parser.add_argument("--trajectorynet-checkpoint", type=Path, default=Path("results/trajectorynet_moscot_rna_100/checkpt.pth"))
    parser.add_argument("--cytobridge-adata", type=Path, default=Path("results/cytobridge_moscot_rna_100/adata.h5ad"))
    parser.add_argument("--output-json", type=Path, default=Path("results/terminal_push_moscot_rna_100.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--energy-steps-per-interval", type=int, default=20)
    parser.add_argument("--trajectorynet-solver", default="rk4")
    parser.add_argument("--trajectorynet-step-size", type=float, default=0.05)
    parser.add_argument("--sinkhorn-engine", choices=("geomloss", "pot"), default="geomloss")
    parser.add_argument("--geomloss-blur", type=float, default=1e-4)
    parser.add_argument("--geomloss-backend", default="tensorized")
    parser.add_argument("--sinkhorn-reg", type=float, default=None)
    parser.add_argument("--sinkhorn-reg-frac", type=float, default=0.05)
    parser.add_argument("--sinkhorn-metric", default="sqeuclidean")
    parser.add_argument("--sinkhorn-num-iter", type=int, default=10000)
    parser.add_argument(
        "--norm-params",
        type=Path,
        default=None,
        help=(
            "Torch file containing {'scale': ...}. If set, evaluate in normalized "
            "coordinates x/scale while using coordinate-transformed raw-space models."
        ),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    x_by_time, unique_times = _load_space(args.input_h5ad, args.time_key, args.latent_key)
    model_space_scale = _load_norm_scale(args.norm_params)
    if model_space_scale != 1.0:
        x_by_time = [x / model_space_scale for x in x_by_time]
    x0 = x_by_time[0]
    x_terminal = x_by_time[-1]

    reg = None
    if args.sinkhorn_engine == "pot":
        baseline_cost = ot.dist(
            x0.astype(np.float64),
            x_terminal.astype(np.float64),
            metric=args.sinkhorn_metric,
        )
        reg = args.sinkhorn_reg
        if reg is None:
            reg = float(args.sinkhorn_reg_frac * np.median(baseline_cost))
        del baseline_cost

    print(f"shared_space: {args.latent_key}, dim={x0.shape[1]}")
    print(f"eval_space: {'normalized' if args.norm_params else 'raw'}, model_space_scale: {model_space_scale:.12g}")
    print(f"time_points: {unique_times.tolist()}")
    print(f"initial_cells: {x0.shape[0]}, terminal_cells: {x_terminal.shape[0]}")
    if args.sinkhorn_engine == "geomloss":
        print(
            "sinkhorn_engine: geomloss, "
            f"blur: {args.geomloss_blur:.6g}, backend: {args.geomloss_backend}"
        )
    else:
        print(f"sinkhorn_engine: pot, metric: {args.sinkhorn_metric}, reg: {reg:.6g}")

    real_adjacent_costs = []
    for idx in range(len(x_by_time) - 1):
        cost = _sinkhorn_cost(x_by_time[idx], x_by_time[idx + 1], args, device, reg)
        real_adjacent_costs.append(cost)
        print(f"Real data adjacent_sinkhorn {idx}->{idx + 1}: {cost:.6f}")
    real_initial_terminal_cost = _sinkhorn_cost(x_by_time[0], x_by_time[-1], args, device, reg)
    print(f"Real data initial_terminal_sinkhorn: {real_initial_terminal_cost:.6f}")
    print(f"Real data adjacent_sinkhorn_sum: {sum(real_adjacent_costs):.6f}")

    tn_push, tn_energy = push_trajectorynet(
        x0=x0,
        unique_times=unique_times,
        checkpoint=args.trajectorynet_checkpoint,
        device=device,
        batch_size=args.batch_size,
        solver=args.trajectorynet_solver,
        step_size=args.trajectorynet_step_size,
        energy_steps_per_interval=args.energy_steps_per_interval,
        model_space_scale=model_space_scale,
    )
    tn_marginal_cost = _sinkhorn_cost(
        tn_push,
        x_terminal,
        args=args,
        device=device,
        reg=reg,
    )
    print(f"TrajectoryNet energy_mean: {tn_energy:.6f}")
    print(f"TrajectoryNet terminal_sinkhorn_cost: {tn_marginal_cost:.6f}")

    cb_push, cb_energy = push_cytobridge(
        x0=x0,
        unique_times=unique_times,
        cytobridge_adata=args.cytobridge_adata,
        device=device,
        batch_size=args.batch_size,
        energy_steps_per_interval=args.energy_steps_per_interval,
        model_space_scale=model_space_scale,
    )
    cb_marginal_cost = _sinkhorn_cost(
        cb_push,
        x_terminal,
        args=args,
        device=device,
        reg=reg,
    )
    print(f"CytoBridge energy_mean: {cb_energy:.6f}")
    print(f"CytoBridge terminal_sinkhorn_cost: {cb_marginal_cost:.6f}")

    result = {
        "input_h5ad": str(args.input_h5ad),
        "latent_key": args.latent_key,
        "time_points": unique_times.tolist(),
        "initial_cells": int(x0.shape[0]),
        "terminal_cells": int(x_terminal.shape[0]),
        "dimension": int(x0.shape[1]),
        "eval_space": "normalized" if args.norm_params else "raw",
        "norm_params": str(args.norm_params) if args.norm_params else None,
        "model_space_scale": model_space_scale,
        "sinkhorn_engine": args.sinkhorn_engine,
        "geomloss_blur": args.geomloss_blur if args.sinkhorn_engine == "geomloss" else None,
        "geomloss_backend": args.geomloss_backend if args.sinkhorn_engine == "geomloss" else None,
        "sinkhorn_metric": args.sinkhorn_metric,
        "sinkhorn_reg": reg,
        "energy_steps_per_interval": args.energy_steps_per_interval,
        "real_data_sanity": {
            "adjacent_sinkhorn_costs": real_adjacent_costs,
            "adjacent_sinkhorn_sum": sum(real_adjacent_costs),
            "initial_terminal_sinkhorn_cost": real_initial_terminal_cost,
        },
        "models": {
            "trajectorynet": {
                "checkpoint": str(args.trajectorynet_checkpoint),
                "energy_mean": tn_energy,
                "terminal_sinkhorn_cost": tn_marginal_cost,
            },
            "cytobridge": {
                "adata": str(args.cytobridge_adata),
                "energy_mean": cb_energy,
                "terminal_sinkhorn_cost": cb_marginal_cost,
            },
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote: {args.output_json}")


if __name__ == "__main__":
    main()
