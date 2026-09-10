"""Shared, auditable TIGON checkpoint rollout and distribution metrics."""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from geomloss import SamplesLoss
from torch import nn
from torchdiffeq import odeint

from tigon_official_ae_common import load_tigon_module


NATIVE_INITIAL_COVARIANCE = 0.02


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ForwardDynamics(nn.Module):
    """Forward TIGON state and accumulated log-growth dynamics."""

    def __init__(self, model: nn.Module, dimension: int):
        super().__init__()
        self.model = model
        self.dimension = int(dimension)

    def forward(
        self,
        time: torch.Tensor,
        augmented: torch.Tensor,
    ) -> torch.Tensor:
        state = augmented[:, : self.dimension]
        return torch.cat(
            [
                self.model.velocity(time, state),
                self.model.growth(time, state),
            ],
            dim=1,
        )


def load_checkpoint_model(
    checkpoint: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object], dict[str, object]]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state.get("training_objective") != "tigon-density":
        raise ValueError(f"{checkpoint} is not a TIGON density checkpoint")
    saved_args = dict(state["args"])
    module = load_tigon_module()
    model = module.TIGONModel(
        int(state["model_dim"]),
        int(saved_args["tigon_hidden_dim"]),
        int(saved_args["tigon_hidden_layers"]),
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval().requires_grad_(False)
    return model, saved_args, state


def deterministic_center_indices(
    number_of_centers: int,
    number_of_particles: int,
    seed: int,
) -> np.ndarray:
    if number_of_centers <= 0 or number_of_particles <= 0:
        raise ValueError("Center and particle counts must be positive")
    rng = np.random.default_rng(seed)
    return rng.choice(
        number_of_centers,
        size=number_of_particles,
        replace=number_of_centers < number_of_particles,
    ).astype(np.int64, copy=False)


def initialize_particles(
    centers: np.ndarray,
    number_of_particles: int,
    *,
    initialization: str,
    seed: int,
    covariance: float = NATIVE_INITIAL_COVARIANCE,
) -> tuple[np.ndarray, np.ndarray]:
    indices = deterministic_center_indices(
        len(centers), number_of_particles, seed
    )
    particles = np.asarray(centers[indices], dtype=np.float32).copy()
    if initialization == "native_cov0.02":
        if not np.isclose(covariance, NATIVE_INITIAL_COVARIANCE):
            raise ValueError(
                "The audited TIGON-native protocol requires covariance 0.02"
            )
        rng = np.random.default_rng(seed + 1)
        particles += (
            math.sqrt(covariance)
            * rng.standard_normal(particles.shape).astype(np.float32)
        )
    elif initialization != "exact_cells":
        raise ValueError(f"Unknown initialization: {initialization}")
    return particles, indices


def rollout(
    *,
    model: nn.Module,
    saved_args: dict[str, object],
    source: np.ndarray,
    evaluation_times: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ordered_times = np.asarray(evaluation_times, dtype=np.float32)
    if len(ordered_times) < 2 or np.any(np.diff(ordered_times) <= 0):
        raise ValueError("evaluation_times must be strictly increasing")

    integration_times = torch.as_tensor(
        ordered_times, dtype=torch.float32, device=device
    )
    solver = str(saved_args["ode_solver"]).lower()
    density_ode_backend = str(
        saved_args.get("density_ode_backend", "torchdiffeq")
    ).lower()
    ode_kwargs: dict[str, object] = {
        "method": solver,
        "rtol": float(saved_args["ode_rtol"]),
        "atol": float(saved_args["ode_atol"]),
    }
    if solver in {"euler", "midpoint", "rk4"}:
        steps = int(saved_args.get("ode_steps", 8))
        if steps <= 0:
            raise ValueError("Saved fixed-step ODE configuration is invalid")
        ode_kwargs["options"] = {
            "step_size": float(np.min(np.diff(ordered_times))) / steps
        }

    dynamics = ForwardDynamics(model, source.shape[1])
    positions: list[list[np.ndarray]] = [[] for _ in ordered_times]
    log_growths: list[list[np.ndarray]] = [[] for _ in ordered_times]
    for start in range(0, len(source), batch_size):
        initial = torch.as_tensor(
            source[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        augmented = torch.cat(
            [
                initial,
                torch.zeros(
                    (len(initial), 1),
                    dtype=initial.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        if density_ode_backend == "torchdiffeqpack":
            if solver != "dopri5":
                raise ValueError(
                    "TorchDiffEqPack checkpoint rollout is audited only for Dopri5"
                )
            upstream_odesolve = load_tigon_module()._load_upstream_odesolve()
            batch_solutions = [augmented]
            # TorchDiffEqPack's public TIGON path requests one end point per
            # solve.  Repeating that exact path for each stage avoids relying
            # on the package's dense-output interpolation for benchmark times.
            for end_time in ordered_times[1:]:
                options = {
                    "method": "Dopri5",
                    "h": None,
                    "t0": float(ordered_times[0]),
                    "t1": float(end_time),
                    "rtol": float(saved_args["ode_rtol"]),
                    "atol": float(saved_args["ode_atol"]),
                    "print_neval": False,
                    "neval_max": 1_000_000,
                    "safety": None,
                }
                with torch.no_grad():
                    batch_solutions.append(
                        upstream_odesolve(
                            dynamics,
                            y0=augmented,
                            options=options,
                        )
                    )
            solution = torch.stack(batch_solutions, dim=0)
        elif density_ode_backend == "torchdiffeq":
            with torch.no_grad():
                solution = odeint(
                    dynamics,
                    augmented,
                    integration_times,
                    **ode_kwargs,
                )
        else:
            raise ValueError(
                f"Unknown saved TIGON ODE backend: {density_ode_backend}"
            )
        for index in range(len(ordered_times)):
            positions[index].append(
                solution[index, :, :-1].detach().cpu().numpy()
            )
            log_growths[index].append(
                solution[index, :, -1].detach().cpu().numpy()
            )
    return (
        [
            np.concatenate(chunks).astype(np.float32, copy=False)
            for chunks in positions
        ],
        [
            np.concatenate(chunks).astype(np.float32, copy=False)
            for chunks in log_growths
        ],
    )


def normalized_weights(log_growth: np.ndarray) -> np.ndarray:
    values = np.asarray(log_growth, dtype=np.float64).reshape(-1)
    relative = np.exp(np.clip(values - values.max(), -700.0, 0.0))
    total = float(relative.sum())
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("Invalid TIGON growth weights")
    return relative / total


def deterministic_target_sample(
    target: np.ndarray,
    number_of_particles: int,
) -> np.ndarray:
    if number_of_particles >= len(target):
        return np.asarray(target, dtype=np.float32)
    indices = np.linspace(
        0, len(target) - 1, number_of_particles
    ).round().astype(np.int64)
    return np.asarray(target[indices], dtype=np.float32)


def distribution_metrics(
    *,
    predicted: np.ndarray,
    log_growth: np.ndarray,
    target: np.ndarray,
    loss_fn: SamplesLoss,
    device: torch.device,
) -> dict[str, float]:
    predicted_tensor = torch.as_tensor(
        predicted, dtype=torch.float32, device=device
    )
    target_tensor = torch.as_tensor(
        target, dtype=torch.float32, device=device
    )
    weights = normalized_weights(log_growth)
    predicted_weights = torch.as_tensor(
        weights, dtype=torch.float32, device=device
    )
    target_weights = torch.full(
        (len(target_tensor),),
        1.0 / len(target_tensor),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        weighted = loss_fn(
            predicted_weights,
            predicted_tensor,
            target_weights,
            target_tensor,
        )
        unweighted = loss_fn(predicted_tensor, target_tensor)
    center = np.sum(weights[:, None] * predicted, axis=0)
    coordinate_variance = np.sum(
        weights[:, None] * np.square(predicted - center), axis=0
    )
    clipped_growth = np.clip(
        np.asarray(log_growth, dtype=np.float64), -50.0, 50.0
    )
    return {
        # GeomLoss uses the 1/2 ||x-y||^2 convention for p=2.
        "weighted_w2_squared": 2.0 * float(weighted.detach().cpu()),
        "unweighted_w2_squared": 2.0 * float(unweighted.detach().cpu()),
        "effective_particle_count": float(1.0 / np.square(weights).sum()),
        "predicted_coordinate_sd_mean": float(
            np.sqrt(np.maximum(coordinate_variance, 0.0)).mean()
        ),
        "mean_mass_factor": float(np.exp(clipped_growth).mean()),
    }
