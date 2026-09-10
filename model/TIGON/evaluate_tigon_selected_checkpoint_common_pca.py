#!/usr/bin/env python
"""Decode selected TIGON AE10 rollouts into shared normalized PCA50."""

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
from pathlib import Path

import anndata as ad
from geomloss import SamplesLoss
import joblib
import numpy as np
import pandas as pd
import torch

from tigon_checkpoint_evaluation_common import (
    deterministic_target_sample,
    distribution_metrics,
    file_sha256,
)
from tigon_official_ae_common import OfficialAutoEncoder


ROOT = Path(__file__).resolve().parents[2]


def load_norm_scale(path: Path) -> float:
    state = torch.load(path, map_location="cpu", weights_only=False)
    scale = float(np.asarray(state["scale"]).reshape(-1)[0])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid normalization scale in {path}")
    return scale


def transform_expression_to_pca(
    bridge: dict[str, object],
    expression: np.ndarray,
) -> np.ndarray:
    standardized = bridge["standard_scaler"].transform(expression)
    intermediate = bridge["pca"].transform(standardized)
    design = np.column_stack(
        [
            intermediate,
            np.ones(intermediate.shape[0], dtype=intermediate.dtype),
        ]
    )
    return (design @ bridge["affine"]).astype(np.float32, copy=False)


def decode_scaled_latent_to_pca(
    *,
    latent_scaled: np.ndarray,
    ae: OfficialAutoEncoder,
    latent_min: torch.Tensor,
    latent_range: torch.Tensor,
    bridge: dict[str, object],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(latent_scaled), batch_size):
        scaled = torch.as_tensor(
            latent_scaled[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        latent_raw = (scaled + 2.0) * latent_range / 4.0 + latent_min
        with torch.no_grad():
            expression = ae.decode(latent_raw).detach().cpu().numpy()
        chunks.append(transform_expression_to_pca(bridge, expression))
    return np.concatenate(chunks).astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=(
            ROOT
            / "results"
            / "tigon_official_ae_embeddings"
            / "gastrulation_ae10"
        ),
    )
    parser.add_argument(
        "--shared-h5ad",
        type=Path,
        default=ROOT / "data" / "gastrulation_rna_cytobridge.h5ad",
    )
    parser.add_argument(
        "--norm-params",
        type=Path,
        default=ROOT / "data" / "gastrulation_rna_primal_norm_params.pt",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sinkhorn-blur", type=float, default=1e-4)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    ae_metrics_path = result_dir / "selected_checkpoint_ae10_metrics.csv"
    ae_predictions_path = (
        result_dir / "selected_checkpoint_ae10_predictions.npz"
    )
    selection_path = result_dir / "observed_checkpoint_selection.json"
    for required in (
        ae_metrics_path,
        ae_predictions_path,
        selection_path,
        cache_dir / "ae.pt",
        cache_dir / "ae_latent_scaling.npz",
        cache_dir / "expression_to_shared_pca50.joblib",
        cache_dir / "common_pca_bridge_ready.json",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    output_metrics = result_dir / "selected_checkpoint_common_pca50_metrics.csv"
    output_predictions = (
        result_dir / "selected_checkpoint_common_pca50_predictions.npz"
    )
    output_manifest = (
        result_dir / "selected_checkpoint_common_pca50_evaluation.json"
    )
    existing = [
        path
        for path in (output_metrics, output_predictions, output_manifest)
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Common-space evaluation already exists: "
            + ", ".join(str(path) for path in existing)
        )

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    held_out_accessed = selection.get(
        "held_out_distribution_accessed_for_selection",
        selection.get("held_out_data_accessed"),
    )
    if held_out_accessed is not False:
        raise ValueError("Checkpoint selection is not certified observed-only")
    ae_metrics = pd.read_csv(ae_metrics_path)
    ae_predictions = np.load(ae_predictions_path)
    bridge_path = cache_dir / "expression_to_shared_pca50.joblib"
    bridge_ready = json.loads(
        (cache_dir / "common_pca_bridge_ready.json").read_text(
            encoding="utf-8"
        )
    )
    if bridge_ready["bridge_sha256"] != file_sha256(bridge_path):
        raise ValueError("Common-PCA bridge checksum does not match")
    bridge = joblib.load(bridge_path)

    device = torch.device(args.device)
    ae_state = torch.load(
        cache_dir / "ae.pt", map_location=device, weights_only=False
    )
    ae = OfficialAutoEncoder(
        in_dim=int(ae_state["in_dim"]),
        n_layers=int(ae_state["n_layers"]),
        n_hidden=int(ae_state["n_hidden"]),
        n_latent=int(ae_state["n_latent"]),
        dropout=float(ae_state["dropout"]),
        norm=bool(ae_state["batch_norm"]),
        seed=int(ae_state["seed"]),
    ).to(device)
    ae.load_state_dict(ae_state["model_state_dict"])
    ae.eval().requires_grad_(False)
    scaling = np.load(cache_dir / "ae_latent_scaling.npz")
    latent_min = torch.as_tensor(
        scaling["latent_min"], dtype=torch.float32, device=device
    )
    latent_range = torch.as_tensor(
        np.maximum(scaling["latent_max"] - scaling["latent_min"], 1e-8),
        dtype=torch.float32,
        device=device,
    )

    shared = ad.read_h5ad(args.shared_h5ad, backed="r")
    try:
        times = pd.to_numeric(
            shared.obs["time_point_processed"], errors="raise"
        ).to_numpy(np.float32)
        shared_pca = np.asarray(shared.obsm["X_latent"], dtype=np.float32)
    finally:
        shared.file.close()
    scale = load_norm_scale(args.norm_params)
    shared_pca_normalized = shared_pca / scale
    eval_samples = int(
        json.loads(
            (
                result_dir / "selected_checkpoint_ae10_evaluation.json"
            ).read_text(encoding="utf-8")
        )["eval_samples"]
    )
    target_by_time = {
        float(time_value): deterministic_target_sample(
            shared_pca_normalized[np.isclose(times, time_value)],
            eval_samples,
        )
        for time_value in np.unique(times)[1:]
    }
    loss_fn = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=args.sinkhorn_blur,
        debias=True,
        backend="tensorized",
    )

    rows: list[dict[str, object]] = []
    output_arrays: dict[str, np.ndarray] = {}
    for row in ae_metrics.itertuples(index=False):
        initialization = str(row.initialization)
        time_value = float(row.stage_time)
        time_tag = f"{time_value:g}".replace(".", "p")
        latent_key = f"{initialization}_to_{time_tag}"
        growth_key = f"{initialization}_log_growth_to_{time_tag}"
        predicted_pca = decode_scaled_latent_to_pca(
            latent_scaled=ae_predictions[latent_key],
            ae=ae,
            latent_min=latent_min,
            latent_range=latent_range,
            bridge=bridge,
            device=device,
            batch_size=args.batch_size,
        )
        predicted_pca_normalized = predicted_pca / scale
        metrics = distribution_metrics(
            predicted=predicted_pca_normalized,
            log_growth=ae_predictions[growth_key],
            target=target_by_time[time_value],
            loss_fn=loss_fn,
            device=device,
        )
        rows.append(
            {
                "dataset": "gastrulation",
                "task": selection["task"],
                "selected_iteration": selection["selected_iteration"],
                "initialization": initialization,
                "initialization_covariance": row.initialization_covariance,
                "stage_time": time_value,
                "stage_role": row.stage_role,
                "normalization_scale": scale,
                **metrics,
            }
        )
        output_arrays[
            f"{initialization}_to_{time_tag}_normalized_pca50"
        ] = predicted_pca_normalized.astype(np.float32, copy=False)
        output_arrays[
            f"{initialization}_log_growth_to_{time_tag}"
        ] = ae_predictions[growth_key]

    pd.DataFrame(rows).to_csv(output_metrics, index=False)
    np.savez_compressed(output_predictions, **output_arrays)
    manifest = {
        "dataset": "gastrulation",
        "task": selection["task"],
        "selected_iteration": selection["selected_iteration"],
        "checkpoint_selection_scope": selection["selection_scope"],
        "checkpoint_selection_held_out_distribution_accessed": held_out_accessed,
        "frozen_representation_is_transductive": selection.get(
            "frozen_representation_is_transductive"
        ),
        "evaluation_space": "shared normalized PCA50",
        "shared_h5ad": str(args.shared_h5ad.resolve()),
        "shared_pca_key": "X_latent",
        "normalization_params": str(args.norm_params.resolve()),
        "normalization_scale": scale,
        "sinkhorn_metric": (
            "growth-weighted and unweighted 2x debiased p=2 "
            "Sinkhorn divergence"
        ),
        "sinkhorn_blur": float(args.sinkhorn_blur),
        "decode_path": (
            "AE10 scaled latent -> inverse latent scaling -> frozen official "
            "AE expression decoder -> frozen expression-to-shared-PCA50 "
            "bridge -> benchmark normalization"
        ),
        "ae_checkpoint": str((cache_dir / "ae.pt").resolve()),
        "bridge": str(bridge_path),
        "bridge_sha256": file_sha256(bridge_path),
        "bridge_training_r2": bridge_ready["training_r2"],
        "bridge_training_relative_rmse": bridge_ready[
            "training_relative_rmse"
        ],
        "metrics_csv": str(output_metrics),
        "predictions_npz": str(output_predictions),
    }
    output_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] wrote {output_metrics}", flush=True)
    print(f"[done] wrote {output_predictions}", flush=True)


if __name__ == "__main__":
    main()
