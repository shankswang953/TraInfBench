#!/usr/bin/env python
"""Replace palate LOO TrajectoryNet pushes with native base-generated marginals."""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/trainfbench-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torchdiffeq import odeint


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

from evaluate_palate_loo_same_space import (  # noqa: E402
    SCENARIOS,
    _load_film,
    _load_scale,
    _map_to_atac,
    _sha256,
)
from evaluate_terminal_push import (  # noqa: E402
    _load_trajectorynet_model,
    _trajectorynet_diffeq,
    _trajectorynet_reverse_schedule,
)


DEFAULT_INPUT = ROOT / "results/palate_loo_same_space_sinkhorn"
DEFAULT_OUTPUT = (
    ROOT / "results/palate_loo_same_space_sinkhorn_trajectorynet_base"
)
RNA_NORM = ROOT / "data/palate_rna_primal_norm_params.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--steps-per-interval", type=int, default=20)
    parser.add_argument("--particles", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fractional_snapshot_rank(
    training_physical_times: np.ndarray,
    heldout_physical_time: float,
) -> float:
    """Map a physical held-out time to an ordinal TrajectoryNet rank."""

    times = np.asarray(training_physical_times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError(f"Invalid physical training times: {times}")
    target = float(heldout_physical_time)
    exact = np.flatnonzero(np.isclose(times, target))
    if len(exact) == 1:
        return float(exact[0])
    upper_index = int(np.searchsorted(times, target))
    if upper_index == 0 or upper_index == len(times):
        raise ValueError(
            f"Held-out time {target:g} lies outside {times.tolist()}"
        )
    lower_index = upper_index - 1
    fraction = (target - times[lower_index]) / (
        times[upper_index] - times[lower_index]
    )
    return float(lower_index + fraction)


def generate_native_marginal(
    checkpoint: Path,
    *,
    n_particles: int,
    dimension: int,
    target_rank: float,
    seed: int,
    device: torch.device,
    batch_size: int,
    steps_per_interval: int,
) -> tuple[np.ndarray, dict]:
    model, model_args = _load_trajectorynet_model(
        checkpoint,
        dimension,
        device,
        "rk4",
        0.1,
    )
    diffeq = _trajectorynet_diffeq(model)
    schedule = [
        (float(model_args.time_scale), 0.0, steps_per_interval),
        *_trajectorynet_reverse_schedule(
            0.0,
            target_rank,
            model_args.time_scale,
            steps_per_interval,
        ),
    ]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    base = torch.randn(
        (n_particles, dimension),
        generator=generator,
        dtype=torch.float32,
    ).numpy()

    pieces: list[np.ndarray] = []
    for start in range(0, n_particles, batch_size):
        stop = min(start + batch_size, n_particles)
        values = torch.as_tensor(
            base[start:stop],
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            for upper, lower, steps in schedule:
                times = torch.linspace(
                    upper,
                    lower,
                    steps + 1,
                    dtype=torch.float32,
                    device=device,
                )
                values = odeint(
                    diffeq,
                    values,
                    times,
                    method="rk4",
                    options={"step_size": 0.1},
                )[-1]
        pieces.append(values.detach().cpu().numpy())
    prediction = np.concatenate(pieces).astype(np.float32, copy=False)
    return prediction, {
        "base": "standard normal",
        "seed": seed,
        "target_fractional_snapshot_rank": target_rank,
        "time_scale": float(model_args.time_scale),
        "reverse_segments": [
            {"upper": upper, "lower": lower, "steps": steps}
            for upper, lower, steps in schedule
        ],
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "prediction_manifest.csv"
    audit_path = output_dir / "trajectorynet_base_generation_manifest.json"
    if (
        (manifest_path.exists() or audit_path.exists())
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Outputs already exist in {output_dir}; pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest_path = input_dir / "prediction_manifest.csv"
    frame = pd.read_csv(source_manifest_path)
    trajectorynet_rows = frame["method"].eq("TrajectoryNet")
    if int(trajectorynet_rows.sum()) != len(SCENARIOS):
        raise ValueError("Expected one TrajectoryNet row per palate LOO scenario")

    # Unchanged methods continue to reference the frozen original predictions.
    for index, row in frame.loc[~trajectorynet_rows].iterrows():
        frame.at[index, "prediction_file"] = str(
            (input_dir / row["prediction_file"]).resolve()
        )

    device = torch.device(args.device)
    film, _, film_checkpoint, _ = _load_film(device)
    rna_scale = _load_scale(RNA_NORM)
    generation_audit: dict[str, dict] = {}

    for scenario, config in SCENARIOS.items():
        selected = frame[
            trajectorynet_rows & frame["scenario"].eq(scenario)
        ]
        if len(selected) != 1:
            raise ValueError(f"Expected one TrajectoryNet row for {scenario}")
        index = int(selected.index[0])
        original_prediction = (
            input_dir / frame.at[index, "prediction_file"]
        ).resolve()
        with np.load(original_prediction, allow_pickle=True) as original:
            initial_indices = np.asarray(
                original["initial_indices"], dtype=np.int64
            )
            target_indices = np.asarray(
                original["target_indices"], dtype=np.int64
            )
        if args.particles != len(initial_indices):
            raise ValueError(
                f"Requested {args.particles} particles but frozen comparison "
                f"uses {len(initial_indices)}"
            )

        with np.load(config["reference"], allow_pickle=True) as reference:
            physical_train_times = np.asarray(
                reference["trajectorynet_physical_train_times"],
                dtype=float,
            )
        heldout_time = float(config["heldout_time"])
        target_rank = fractional_snapshot_rank(
            physical_train_times,
            heldout_time,
        )
        prediction_raw, clock = generate_native_marginal(
            Path(config["trajectorynet_model"]),
            n_particles=args.particles,
            dimension=40,
            target_rank=target_rank,
            seed=args.seed,
            device=device,
            batch_size=args.batch_size,
            steps_per_interval=args.steps_per_interval,
        )
        prediction_rna = (
            prediction_raw / rna_scale
        ).astype(np.float32, copy=False)
        prediction_atac = _map_to_atac(
            film,
            prediction_rna,
            heldout_time,
            device,
            args.batch_size,
        )
        weights = np.full(
            args.particles,
            1.0 / args.particles,
            dtype=np.float32,
        )
        output = output_dir / "predictions" / scenario / "trajectorynet_base.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            rna_norm=prediction_rna,
            atac_norm=prediction_atac,
            weights=weights,
            initial_indices=initial_indices,
            target_indices=target_indices,
            heldout_time=np.asarray(heldout_time, dtype=np.float32),
            method=np.asarray("TrajectoryNet"),
            cy=np.asarray(np.nan, dtype=np.float32),
            particle_origin=np.asarray("standard_normal_base"),
            base_seed=np.asarray(args.seed, dtype=np.int64),
            target_fractional_snapshot_rank=np.asarray(
                target_rank,
                dtype=np.float32,
            ),
        )
        frame.at[index, "prediction_file"] = str(
            output.relative_to(output_dir)
        )
        frame.at[index, "prediction_sha256"] = _sha256(output)
        frame.at[index, "weight_source"] = "uniform standard-normal base"
        generation_audit[scenario] = {
            "heldout_stage": config["heldout_stage"],
            "heldout_physical_time": heldout_time,
            "training_physical_times": physical_train_times.tolist(),
            "checkpoint": str(Path(config["trajectorynet_model"]).resolve()),
            "prediction_file": str(output),
            "prediction_sha256": _sha256(output),
            "clock": clock,
        }
        print(
            f"[{scenario}] base -> {config['heldout_stage']} "
            f"(fractional rank {target_rank:g})",
            flush=True,
        )

    frame.to_csv(manifest_path, index=False)
    source_evaluation = input_dir / "evaluation_manifest.json"
    if source_evaluation.is_file():
        evaluation = json.loads(source_evaluation.read_text(encoding="utf-8"))
    else:
        evaluation = {}
    evaluation.update(
        {
            "definition": (
                "strict-LOO COATI/shared-full-T comparison with forward-trained "
                "TrajectoryNet sampled natively from N(0,I)"
            ),
            "trajectorynet_protocol": (
                "Forward-time LOO checkpoints; standard-normal base -> E12.5 -> "
                "held-out stage using the exact LOO-specific piecewise clock. "
                "TrajectoryNet particles do not share observed E12.5 identities."
            ),
            "trajectorynet_base_generation_manifest": str(audit_path),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_archive": str(input_dir),
        }
    )
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(evaluation, indent=2) + "\n", encoding="utf-8"
    )
    audit = {
        "task": "Palate LOO TrajectoryNet native base-marginal replacement",
        "source_prediction_manifest": str(source_manifest_path),
        "source_prediction_manifest_sha256": _sha256(source_manifest_path),
        "unchanged_methods": (
            "Referenced byte-for-byte from the original prediction directory"
        ),
        "trajectorynet_particle_origin": "standard normal base",
        "trajectorynet_particle_count": args.particles,
        "trajectorynet_base_seed": args.seed,
        "rna_scale": rna_scale,
        "atac_mapping_checkpoint": str(film_checkpoint),
        "atac_mapping_checkpoint_sha256": _sha256(film_checkpoint),
        "scenarios": generation_audit,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(manifest_path)
    print(audit_path)


if __name__ == "__main__":
    main()
