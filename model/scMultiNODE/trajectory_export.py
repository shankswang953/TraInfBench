"""Batch export without changing scMultiNODE's native Euler integration grid.

The official Euler solver uses the requested times as integration steps. Asking
it for a dense time grid therefore changes predictions at the training times.
This module instead solves only on the actual observed training times, linearly
interpolates those *latent* knot states, and then applies the learned decoders.
This is native explicit-Euler dense interpolation, not a finer-step ODE solve
and not interpolation of decoded observations. In a leave-one-time-out run, the
held-out time is an interpolation query, never an integration/training knot.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


_AUDIT_ATOL = 1e-5


def query_time_grid(all_times: Sequence[float] | np.ndarray, dt: float) -> np.ndarray:
    """Dense output grid including each physical knot once, despite float32 roundoff.

    Build the regular grid from dt, not a float32-rounded endpoint (e.g. 1.7).
    Snap near-identical query coordinates to exact supplied knots. This only
    affects saved query times, never the native Euler integration grid.
    """
    times = np.asarray(all_times, dtype=np.float32)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError("Dataset times must increase strictly from zero")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive")
    end_time = float(times[-1])
    steps = round(end_time / dt)
    tolerance = 4 * np.finfo(np.float32).eps * max(1., end_time)
    if steps < 1 or abs(steps * dt - end_time) > tolerance:
        raise ValueError("dt must divide the dataset terminal time")
    query = (np.arange(steps + 1, dtype=np.float64) * dt).astype(np.float32)
    for knot in times:
        nearest = int(np.argmin(np.abs(query - knot)))
        if abs(float(query[nearest]) - float(knot)) <= tolerance:
            query[nearest] = knot
        else:
            query = np.append(query, knot)
    return np.unique(query)


def _time_tensor(
    values: Sequence[float] | np.ndarray | torch.Tensor,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    times = torch.as_tensor(values, device=reference.device, dtype=reference.dtype)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional time grid")
    if not bool(torch.isfinite(times).all()):
        raise ValueError(f"{name} contains nonfinite times")
    if not bool((times[1:] > times[:-1]).all()):
        raise ValueError(f"{name} must be strictly increasing")
    return times


def _interpolate_native_latent(
    native_latent: torch.Tensor,
    train_times: torch.Tensor,
    query_times: torch.Tensor,
) -> torch.Tensor:
    """Piecewise-linear Euler dense output, with exact native states at knots."""
    upper = torch.searchsorted(train_times, query_times).clamp(1, len(train_times) - 1)
    lower = upper - 1
    fraction = (query_times - train_times[lower]) / (train_times[upper] - train_times[lower])
    interpolated = torch.lerp(
        native_latent[:, lower, :], native_latent[:, upper, :], fraction[None, :, None]
    )
    # Avoid cancellation at right endpoints: queried knots are exactly the
    # solver-produced latent values, including the final observed time.
    return torch.where(
        (query_times == train_times[upper])[None, :, None],
        native_latent[:, upper, :],
        interpolated,
    )


def _finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"Nonfinite {name}; refusing to save trajectories")


def _compare(name: str, actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name} shape mismatch: {actual.shape} != {expected.shape}")
    _finite(name, actual)
    _finite(f"reference {name}", expected)
    error = float(torch.max(torch.abs(actual - expected)).item())
    if error > _AUDIT_ATOL:
        raise AssertionError(
            f"Native training-grid equivalence failed for {name}: {error} > {_AUDIT_ATOL}"
        )
    return error


def export_trajectories(
    model: torch.nn.Module,
    initial_tensor: torch.Tensor,
    initial_indices: Sequence[int] | np.ndarray | torch.Tensor,
    train_times: Sequence[float] | np.ndarray | torch.Tensor,
    query_times: Sequence[float] | np.ndarray | torch.Tensor,
    output_path: str | Path,
    batch_size: int = 512,
) -> dict[str, Any]:
    """Export all supplied RNA initial cells once, in their supplied order.

    ``train_times`` is the *actual observed* grid used to fit this model, such
    as [0, 1, 2, 2.5], [0, 2, 2.5], or [0, 1, 2.5]. ``query_times`` may be a
    dense grid (for example 0..2.5 in increments of .025) inside that interval.
    Increasing query density never changes the native Euler integration grid.

    The NPZ contains time-major ``rna_norm``, ``atac_norm``, ``latent``, and
    uniform ``weights``, plus ``time``, ``training_time``, ``initial_rna_norm``,
    and ``initial_indices``. All supplied cells are retained, with no sampling,
    repeats, truncation, denormalization, or decoder adaptation. The caller
    remains responsible for providing the complete intended initial cohort.

    Every batch is audited against the official model forward at the native
    training times with an absolute tolerance of 1e-5. A compact JSON-compatible
    audit is returned for the caller's manifest. An existing output is never
    overwritten. Model parameters and original module training flags are kept.
    """
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    if output_path.suffix.lower() != ".npz":
        raise ValueError("Trajectory output must have an .npz suffix")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(initial_tensor, torch.Tensor):
        raise TypeError("initial_tensor must be a torch.Tensor")
    if initial_tensor.ndim != 2 or min(initial_tensor.shape) < 1:
        raise ValueError("initial_tensor must be a nonempty cells-by-features tensor")
    if initial_tensor.dtype not in (torch.float32, torch.float64):
        raise ValueError("initial_tensor must use float32 or float64")
    _finite("initial RNA coordinates", initial_tensor)
    if getattr(model, "anchor_mod", None) != "rna":
        raise ValueError("This exporter requires the official RNA-anchor model")
    if getattr(model.diffeq_decoder, "ode_method", None) != "euler":
        raise ValueError("Native Euler interpolation requires ode_method='euler'")
    for parameter in model.parameters():
        if parameter.device != initial_tensor.device or parameter.dtype != initial_tensor.dtype:
            raise ValueError("Model parameters and initial_tensor must share dtype and device")
        _finite("model parameters", parameter)

    indices = (
        initial_indices.detach().cpu().numpy()
        if isinstance(initial_indices, torch.Tensor)
        else np.asarray(initial_indices)
    )
    n_initial = len(initial_tensor)
    if indices.ndim != 1 or len(indices) != n_initial or indices.dtype.kind not in "iu":
        raise ValueError("initial_indices must contain one integer index per initial row")
    if np.any(indices < 0) or len(np.unique(indices)) != n_initial:
        raise ValueError("initial_indices must be nonnegative and unique; no duplicate cells")
    indices = indices.copy()
    train = _time_tensor(train_times, initial_tensor, "train_times")
    query = _time_tensor(query_times, initial_tensor, "query_times")
    if len(train) < 2:
        raise ValueError("At least two observed training times are required")
    if bool(query[0] < train[0]) or bool(query[-1] > train[-1]):
        raise ValueError("query_times must lie within the observed training-time interval")

    arrays: dict[str, np.ndarray] = {}
    errors = dict.fromkeys(("rna_norm", "atac_norm", "latent"), 0.0)
    query_errors = dict.fromkeys(errors, 0.0)
    query_positions = torch.searchsorted(query, train)
    present = (query_positions < len(query)) & (query[query_positions.clamp(max=len(query) - 1)] == train)
    observed_positions = torch.nonzero(present, as_tuple=False).flatten()
    query_positions = query_positions[present]
    original_training_flags = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, n_initial, batch_size):
                stop = min(start + batch_size, n_initial)
                initial_batch = initial_tensor[start:stop]
                initial_latent = model.fusion_layer(model.anchor_enc(initial_batch))
                native_latent = model.diffeq_decoder(initial_latent, train)
                if native_latent.ndim != 3 or native_latent.shape[:2] != (stop - start, len(train)):
                    raise AssertionError("Official solver returned an unexpected latent shape")
                _finite("native latent trajectory", native_latent)
                dense_latent = _interpolate_native_latent(native_latent, train, query)
                decoded = {
                    "rna_norm": model.rna_dec(dense_latent),
                    "atac_norm": model.atac_dec(dense_latent),
                    "latent": dense_latent,
                }

                # Independently invoke the unmodified official public forward.
                ref_rna, ref_atac, passed_initial, ref_latent, ref_atac_latent = model(
                    initial_batch, train, train, batch_size=None
                )
                if not torch.equal(passed_initial, initial_batch):
                    raise AssertionError("Official forward sampled or changed initial rows")
                _compare("RNA/ATAC joint latent", ref_latent, ref_atac_latent)
                references = {"rna_norm": ref_rna, "atac_norm": ref_atac, "latent": ref_latent}
                knot_latent = _interpolate_native_latent(native_latent, train, train)
                knot_values = {
                    "rna_norm": model.rna_dec(knot_latent),
                    "atac_norm": model.atac_dec(knot_latent),
                    "latent": knot_latent,
                }
                for key, tensor in decoded.items():
                    _finite(key, tensor)
                    errors[key] = max(errors[key], _compare(key, knot_values[key], references[key]))
                    if len(observed_positions):
                        query_errors[key] = max(
                            query_errors[key],
                            _compare(
                                f"exported {key}", tensor[:, query_positions, :],
                                references[key][:, observed_positions, :],
                            ),
                        )
                    values = tensor.detach().cpu().numpy().transpose(1, 0, 2)
                    if key not in arrays:
                        arrays[key] = np.empty((len(query), n_initial, values.shape[2]), dtype=values.dtype)
                    arrays[key][:, start:stop, :] = values
    finally:
        for module, was_training in original_training_flags:
            module.training = was_training

    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise FloatingPointError("Nonfinite trajectory arrays")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against a competing writer after the
    # initial existence check. No automatic extension/silent overwrite occurs.
    with output_path.open("xb") as stream:
        np.savez_compressed(
            stream, **arrays, time=query.detach().cpu().numpy(),
            training_time=train.detach().cpu().numpy(),
            initial_rna_norm=initial_tensor.detach().cpu().numpy(),
            initial_indices=indices,
            weights=np.full((len(query), n_initial), 1.0 / n_initial, dtype=np.float64),
        )
    return {
        "export_mode": "native_euler",
        "solver": "official Euler on observed training times",
        "dense_output": "piecewise-linear native Euler latent interpolation, then learned decoders",
        "training_times": train.detach().cpu().tolist(),
        "query_time_count": len(query),
        "query_time_interval": [float(query[0]), float(query[-1])],
        "n_initial": n_initial,
        "n_unique_initial_indices": len(np.unique(indices)),
        "all_supplied_initial_once": True,
        "initial_order_preserved": True,
        "initial_indices_cover_zero_to_n_minus_one": bool(np.array_equal(np.sort(indices), np.arange(n_initial))),
        "inference_batch_size": batch_size,
        "inference_batches": (n_initial + batch_size - 1) // batch_size,
        "native_training_grid_max_abs_error": errors,
        "exported_training_time_max_abs_error": query_errors if len(observed_positions) else None,
        "exported_training_times_checked": train[present].detach().cpu().tolist(),
        "equivalence_absolute_tolerance": _AUDIT_ATOL,
        "finite_parameters_and_outputs": True,
        "shapes": {key: list(values.shape) for key, values in arrays.items()},
        "model_retrained": False,
    }
