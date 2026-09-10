#!/usr/bin/env python
"""Audit the publication-facing Gastrulation TIGON v2 protocol."""

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
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_tigon_frozen_official_ae_embedding import TASKS, load_frozen_task
from tigon_official_ae_common import (
    OFFICIAL_SOURCE_COMMIT,
    TIGON_UPSTREAM_PUBLIC_EXACT_PROTOCOL,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "results" / "tigon_gastrulation_upstream_public_exact_v2_audit"
)
EQUIVALENCE = (
    ROOT
    / "results"
    / "tigon_upstream_public_equivalence"
    / "equivalence_report.json"
)
UPSTREAM_EMT = (
    ROOT / "results" / "tigon_upstream_emt_reproduction_audit" / "manifest.json"
)
LOCAL_EMT = (
    ROOT
    / "results"
    / "tigon_reimplementation_emt_upstream_public_exact_pack_smoke_5_seed1"
    / "tigon_training_history.csv"
)
SMOKE = (
    ROOT
    / "results"
    / "tigon_gastrulation_full_upstream_public_exact_v2_rngexact_smoke_ae10_n128_1000_seed1"
)
CACHE = (
    ROOT
    / "results"
    / "tigon_official_ae_embeddings"
    / "gastrulation_ae10_upstream_public_exact_v2"
)
UPSTREAM_DIR = ROOT / "external" / "TIGON_upstream_1ed92cf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-result-dir", type=Path, default=SMOKE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def checkpoint_arg(state: dict, name: str) -> object:
    saved = state.get("args", {})
    if isinstance(saved, dict):
        return saved.get(name)
    return getattr(saved, name, None)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    equivalence = json.loads(EQUIVALENCE.read_text(encoding="utf-8"))
    if equivalence.get("status") != "passed":
        raise ValueError("The public-source primitive equivalence check did not pass")
    if equivalence.get("upstream_commit") != OFFICIAL_SOURCE_COMMIT:
        raise ValueError("Equivalence report uses the wrong public TIGON commit")
    upstream_head = subprocess.check_output(
        ["git", "-C", str(UPSTREAM_DIR), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if upstream_head != OFFICIAL_SOURCE_COMMIT:
        raise ValueError(f"Frozen upstream HEAD is {upstream_head}")
    tracked_diff = subprocess.run(
        ["git", "-C", str(UPSTREAM_DIR), "diff", "--quiet"],
        check=False,
    ).returncode
    if tracked_diff != 0:
        raise ValueError("Frozen public TIGON tracked source has local changes")

    upstream_emt = json.loads(UPSTREAM_EMT.read_text(encoding="utf-8"))
    public_losses = np.asarray(
        upstream_emt["checkpoint_metadata"]["fresh tutorial, 5 steps"]["loss_head"],
        dtype=np.float64,
    )
    local_losses = pd.read_csv(LOCAL_EMT)["loss"].to_numpy(np.float64)
    if len(public_losses) != 5 or len(local_losses) != 5:
        raise ValueError("Expected five-step public and local EMT traces")
    emt_absolute = np.abs(public_losses - local_losses)
    emt_relative = emt_absolute / np.maximum(np.abs(public_losses), 1e-12)
    if float(emt_absolute.max()) > 0.2:
        raise ValueError("Local EMT trace is not numerically close to public TIGON")

    cache_config = json.loads((CACHE / "config.json").read_text(encoding="utf-8"))
    ready = json.loads(
        (CACHE / "embedding_ready.json").read_text(encoding="utf-8")
    )
    if ready.get("status") != "ready" or ready.get("latent_dimension") != 10:
        raise ValueError("Frozen Gastrulation AE10 cache is not ready")
    if cache_config.get("ae_architecture") != "3000-300-10-300-3000":
        raise ValueError("Unexpected TIGON-style AE architecture")
    if cache_config.get("input_preprocessing") != (
        "3000-HVG log-expression without per-gene standardization"
    ):
        raise ValueError("Unexpected Gastrulation AE preprocessing")

    frozen_hashes: set[str] = set()
    task_audit: dict[str, object] = {}
    for task in ("full", "loo_time1", "loo_time2"):
        defaults = TASKS[("gastrulation", task)]
        latent, times, unique_times, audit = load_frozen_task(
            "gastrulation",
            task,
            Path(defaults["h5ad"]).resolve(),
            CACHE.resolve(),
            latent_source="scaled_minus2_2",
            latent_standardization="none",
        )
        frozen_hashes.add(str(audit["frozen_latent_sha256"]))
        task_audit[task] = {
            "cells": int(len(latent)),
            "training_times": [float(value) for value in unique_times],
            "counts": [
                int(np.sum(np.isclose(times, value))) for value in unique_times
            ],
            "held_out_time": audit["held_out_time"],
        }
    if len(frozen_hashes) != 1:
        raise ValueError("Full and LOO do not share exactly one frozen AE")

    smoke_dir = args.smoke_result_dir.resolve()
    smoke_config = json.loads(
        (smoke_dir / "config.json").read_text(encoding="utf-8")
    )
    expected_config = {
        "safe_reproduction_protocol": TIGON_UPSTREAM_PUBLIC_EXACT_PROTOCOL,
        "action_state_mode": "upstream_public_exact",
        "ode_solver": "dopri5",
        "ode_backend": "torchdiffeqpack",
        "divergence_estimator": "exact",
        "density_stabilization": "official",
        "official_density_sigma_schedule": True,
        "density_sigma_min": None,
        "density_sigma_hard_clip": None,
    }
    config_mismatches = {
        name: {"expected": wanted, "actual": smoke_config.get(name)}
        for name, wanted in expected_config.items()
        if smoke_config.get(name) != wanted
    }
    if config_mismatches:
        raise ValueError(f"Smoke config mismatch: {config_mismatches}")
    implementation_path = Path(
        smoke_config["benchmark_implementation_file"]
    ).resolve()
    current_implementation_hash = file_sha256(implementation_path)
    if current_implementation_hash != smoke_config.get(
        "benchmark_implementation_sha256"
    ):
        raise ValueError(
            "Benchmark TIGON implementation changed after smoke launch"
        )

    checkpoint = torch.load(
        smoke_dir / "tigon_checkpoint_latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_expected = {
        "safe_reproduction_protocol": TIGON_UPSTREAM_PUBLIC_EXACT_PROTOCOL,
        "action_state_mode": "upstream_public_exact",
        "density_ode_backend": "torchdiffeqpack",
        "density_stabilization": "official",
        "divergence_estimator": "exact",
        "ode_solver": "dopri5",
        "action_ode_solver": "midpoint",
        "official_density_sigma_schedule": True,
    }
    checkpoint_mismatches = {
        name: {"expected": wanted, "actual": checkpoint_arg(checkpoint, name)}
        for name, wanted in checkpoint_expected.items()
        if checkpoint_arg(checkpoint, name) != wanted
    }
    if checkpoint_mismatches:
        raise ValueError(f"Smoke checkpoint mismatch: {checkpoint_mismatches}")
    missing_rng_states = [
        name
        for name in (
            "torch_rng_state",
            "python_random_state",
            "numpy_rng_state",
        )
        if name not in checkpoint
    ]
    if missing_rng_states:
        raise ValueError(
            f"Exact v2 smoke checkpoint lacks RNG states: {missing_rng_states}"
        )

    report: dict[str, object] = {
        "status": "passed",
        "protocol": TIGON_UPSTREAM_PUBLIC_EXACT_PROTOCOL,
        "upstream_commit": OFFICIAL_SOURCE_COMMIT,
        "upstream_tracked_source_modified": False,
        "public_equivalence": equivalence,
        "emt_five_step_reproduction": {
            "public_losses": public_losses.tolist(),
            "local_losses": local_losses.tolist(),
            "max_absolute_difference": float(emt_absolute.max()),
            "max_relative_difference": float(emt_relative.max()),
        },
        "frozen_ae": {
            "architecture": cache_config["ae_architecture"],
            "input_preprocessing": cache_config["input_preprocessing"],
            "latent_axis_scaling": cache_config["latent_axis_scaling"],
            "latent_sha256": next(iter(frozen_hashes)),
            "transductive_for_loo": True,
        },
        "tasks": task_audit,
        "smoke_checkpoint": {
            "path": str(smoke_dir / "tigon_checkpoint_latest.pt"),
            "iteration": int(checkpoint["iter"]),
            "density_sigma_next": float(checkpoint["density_sigma"]),
            "implementation_sha256": current_implementation_hash,
        },
    }

    summary_path = smoke_dir / "distribution_diagnosis_forward" / "checkpoint_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        trained = summary[summary["iteration"] > 0]
        best = trained.loc[
            trained["mean_weighted_w2_squared_common_pca50"].idxmin()
        ]
        final = trained.loc[trained["iteration"].idxmax()]
        report["forward_distribution_diagnosis"] = {
            "status": (
                "improved"
                if float(best["mean_common_improvement_vs_initial_percent"]) > 0
                else "no_improvement_in_this_smoke"
            ),
            "best_iteration": int(best["iteration"]),
            "best_mean_common_pca50_w2_squared": float(
                best["mean_weighted_w2_squared_common_pca50"]
            ),
            "best_mean_improvement_percent": float(
                best["mean_common_improvement_vs_initial_percent"]
            ),
            "best_terminal_improvement_percent": float(
                best["terminal_common_improvement_vs_initial_percent"]
            ),
            "final_iteration": int(final["iteration"]),
            "final_mean_improvement_percent": float(
                final["mean_common_improvement_vs_initial_percent"]
            ),
        }
    else:
        report["forward_distribution_diagnosis"] = {
            "status": "not_evaluated"
        }

    selection_path = smoke_dir / "observed_checkpoint_selection.json"
    common_evaluation_path = (
        smoke_dir / "selected_checkpoint_common_pca50_evaluation.json"
    )
    if selection_path.exists() and common_evaluation_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        common_evaluation = json.loads(
            common_evaluation_path.read_text(encoding="utf-8")
        )
        if selection.get("held_out_distribution_accessed_for_selection") is not False:
            raise ValueError("Selected checkpoint was not certified observed-only")
        if selection.get("frozen_latent_sha256") != next(iter(frozen_hashes)):
            raise ValueError("Selected checkpoint uses the wrong frozen AE")
        if selection.get("required_checkpoint_protocol") != (
            TIGON_UPSTREAM_PUBLIC_EXACT_PROTOCOL
        ):
            raise ValueError("Selected checkpoint uses the wrong protocol")
        if Path(common_evaluation["ae_checkpoint"]).resolve().parent != CACHE:
            raise ValueError("Common-space evaluation decoded with the wrong AE")
        report["postprocessing_chain"] = {
            "status": "passed",
            "selected_iteration": int(selection["selected_iteration"]),
            "selection_scope": selection["selection_scope"],
            "held_out_distribution_accessed_for_selection": False,
            "common_evaluation_space": common_evaluation["evaluation_space"],
            "bridge_sha256": common_evaluation["bridge_sha256"],
        }
    else:
        report["postprocessing_chain"] = {"status": "not_evaluated"}

    report_path = output_dir / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
