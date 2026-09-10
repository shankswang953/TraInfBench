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
import json
import shutil
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from mioflow import Autoencoder, MIOFlow, dataloader_from_pc, train_gaga_two_phase
from mioflow.core.datasets import TimeSeriesDataset
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.dtype.str.encode("utf-8"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _sorted_time_values(adata, time_key: str) -> list[str]:
    values = pd.Series(adata.obs[time_key].to_numpy())
    try:
        return [str(x) for x in sorted(values.unique())]
    except TypeError:
        return [str(x) for x in sorted(values.astype(str).unique())]


def _replace_mioflow_rank_times_with_numeric_times(
    mf: MIOFlow,
    adata: ad.AnnData,
    time_key: str,
) -> None:
    """Keep MIOFlow's encoded observations but restore numeric snapshot times."""
    numeric_times = pd.to_numeric(
        pd.Series(adata.obs[time_key].to_numpy()), errors="raise"
    ).to_numpy(dtype=float)
    unique_times = np.asarray(sorted(np.unique(numeric_times)), dtype=float)
    if unique_times.size != len(mf.dataset.time_series_data):
        raise ValueError(
            "Numeric time count does not match MIOFlow's encoded groups: "
            f"{unique_times.size} versus {len(mf.dataset.time_series_data)}"
        )
    mf.dataset = TimeSeriesDataset(
        [
            (values, float(time_value))
            for (values, _rank_time), time_value in zip(
                mf.dataset.time_series_data, unique_times
            )
        ]
    )
    mf._time_labels = numeric_times


def _load_norm_scale(norm_params: Path) -> float:
    params = torch.load(norm_params, map_location="cpu")
    if not isinstance(params, dict) or "scale" not in params:
        raise ValueError(f"Expected {norm_params} to contain a dict with key 'scale'")
    scale = params["scale"]
    if hasattr(scale, "item"):
        return float(scale.item())
    return float(scale)


def _parse_hidden_dims(value: str) -> list[int]:
    dims = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not dims:
        raise ValueError("--gaga-hidden-dims must contain at least one integer")
    return dims


def _load_embedding_normalization_checkpoint(
    checkpoint_path: Path,
    expected_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load frozen GAGA-latent z-score parameters from a Full MIOFlow run."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"mean_vals", "std_vals", "embedding_normalization", "input_dim"}
    missing = required.difference(state)
    if missing:
        raise ValueError(
            f"Normalization checkpoint {checkpoint_path} is missing keys: "
            f"{sorted(missing)}"
        )
    mode = str(state["embedding_normalization"]).replace("-", "_")
    if mode != "zscore":
        raise ValueError(
            f"Normalization checkpoint {checkpoint_path} used {mode!r}, not 'zscore'"
        )
    if int(state["input_dim"]) != expected_dim:
        raise ValueError(
            f"Normalization checkpoint input_dim={state['input_dim']} does not "
            f"match expected GAGA dimension {expected_dim}"
        )
    mean = np.asarray(state["mean_vals"], dtype=np.float64).reshape(-1)
    std = np.asarray(state["std_vals"], dtype=np.float64).reshape(-1)
    if mean.shape != (expected_dim,) or std.shape != (expected_dim,):
        raise ValueError(
            f"Frozen normalization shapes are mean={mean.shape}, std={std.shape}; "
            f"expected {(expected_dim,)}"
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("Frozen normalization parameters contain non-finite values")
    if np.any(std <= 0):
        raise ValueError("Frozen normalization std must be strictly positive")
    return mean, std


def _stratified_sample_indices(labels: np.ndarray, max_cells: int, seed: int) -> np.ndarray:
    n_obs = labels.shape[0]
    if max_cells <= 0 or max_cells >= n_obs:
        return np.arange(n_obs)

    rng = np.random.default_rng(seed)
    chosen = []
    groups, counts = np.unique(labels, return_counts=True)
    remaining = max_cells
    for group, count in zip(groups, counts):
        group_idx = np.flatnonzero(labels == group)
        n_group = max(1, int(round(max_cells * count / n_obs)))
        n_group = min(n_group, group_idx.shape[0], remaining)
        if n_group > 0:
            chosen.append(rng.choice(group_idx, size=n_group, replace=False))
            remaining -= n_group

    if remaining > 0:
        already = np.concatenate(chosen) if chosen else np.array([], dtype=int)
        mask = np.ones(n_obs, dtype=bool)
        mask[already] = False
        pool = np.flatnonzero(mask)
        extra = rng.choice(pool, size=min(remaining, pool.shape[0]), replace=False)
        chosen.append(extra)

    indices = np.concatenate(chosen)
    rng.shuffle(indices)
    return indices[:max_cells]


def _fit_gaga(
    adata: ad.AnnData,
    x: np.ndarray,
    args: argparse.Namespace,
    device: str,
) -> tuple[Autoencoder, dict, np.ndarray]:
    import phate

    hidden_dims = _parse_hidden_dims(args.gaga_hidden_dims)
    labels = adata.obs[args.time_key].to_numpy()
    fit_idx = _stratified_sample_indices(labels, args.gaga_max_cells, args.seed)

    scaler = StandardScaler().fit(x)
    x_fit_scaled = scaler.transform(x[fit_idx]).astype("float32", copy=False)

    if args.gaga_distance_key:
        if args.gaga_distance_key not in adata.obsm:
            raise KeyError(
                f"{args.gaga_distance_key!r} not found in adata.obsm. "
                f"Available keys: {list(adata.obsm.keys())}"
            )
        distance_basis = np.asarray(adata.obsm[args.gaga_distance_key][fit_idx], dtype="float32")
        distance_basis = StandardScaler().fit_transform(distance_basis)
        distance_source = args.gaga_distance_key
    else:
        n_landmark = min(args.gaga_phate_landmarks, x_fit_scaled.shape[0])
        n_pca = min(x_fit_scaled.shape[1], 100)
        phate_op = phate.PHATE(
            n_components=args.gaga_distance_components,
            knn=args.gaga_phate_knn,
            n_landmark=n_landmark,
            n_pca=n_pca,
            n_jobs=1,
            random_state=args.seed,
            verbose=1,
        )
        distance_basis = phate_op.fit_transform(x_fit_scaled).astype("float32", copy=False)
        distance_basis = StandardScaler().fit_transform(distance_basis)
        distance_source = "computed_phate"

    distances = pairwise_distances(distance_basis, metric="euclidean").astype("float32", copy=False)
    loader = dataloader_from_pc(
        x_fit_scaled,
        distances,
        batch_size=args.gaga_batch_size,
        shuffle=True,
    )
    model = Autoencoder(
        input_dim=x.shape[1],
        latent_dim=args.gaga_latent_dim,
        hidden_dims=hidden_dims,
    )

    print(
        "GAGA: "
        f"cells={fit_idx.shape[0]}, input_dim={x.shape[1]}, "
        f"latent_dim={args.gaga_latent_dim}, hidden_dims={hidden_dims}, "
        f"distance_source={distance_source}"
    )
    history = train_gaga_two_phase(
        model,
        loader,
        encoder_epochs=args.gaga_encoder_epochs,
        decoder_epochs=args.gaga_decoder_epochs,
        learning_rate=args.gaga_learning_rate,
        device=device,
        dist_weight_phase1=args.gaga_dist_weight,
        recon_weight_phase2=args.gaga_recon_weight,
    )
    model.input_scaler = scaler
    model.gaga_distance_source = distance_source
    return model, history, fit_idx


def _load_gaga_checkpoint(
    checkpoint_path: Path,
    x: np.ndarray,
    args: argparse.Namespace,
    device: str,
) -> tuple[Autoencoder, np.ndarray]:
    """Load a frozen, externally trained official GAGA model."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {
        "state_dict",
        "input_dim",
        "latent_dim",
        "hidden_dims",
        "input_scaler_mean",
        "input_scaler_scale",
        "fit_indices",
    }
    missing = required.difference(state)
    if missing:
        raise ValueError(
            f"GAGA checkpoint {checkpoint_path} is missing keys: {sorted(missing)}"
        )
    input_dim = int(state["input_dim"])
    latent_dim = int(state["latent_dim"])
    hidden_dims = [int(value) for value in state["hidden_dims"]]
    if input_dim != x.shape[1]:
        raise ValueError(
            f"GAGA checkpoint input_dim={input_dim} does not match data dim={x.shape[1]}"
        )
    if latent_dim != args.gaga_latent_dim:
        raise ValueError(
            f"GAGA checkpoint latent_dim={latent_dim} does not match "
            f"--gaga-latent-dim={args.gaga_latent_dim}"
        )
    if hidden_dims != _parse_hidden_dims(args.gaga_hidden_dims):
        raise ValueError(
            f"GAGA checkpoint hidden_dims={hidden_dims} do not match "
            f"--gaga-hidden-dims={args.gaga_hidden_dims}"
        )

    model = Autoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
    )
    model.load_state_dict(state["state_dict"])
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(state["input_scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(state["input_scaler_scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = input_dim
    scaler.n_samples_seen_ = int(
        state.get("input_scaler_n_samples_seen", len(state["fit_indices"]))
    )
    model.input_scaler = scaler
    model.gaga_distance_source = "loaded_checkpoint"
    model.to(device).eval().requires_grad_(False)
    fit_idx = np.asarray(state["fit_indices"], dtype=np.int64)
    if np.any(fit_idx < 0):
        raise ValueError("GAGA checkpoint contains negative source fit indices")
    print(
        "GAGA: loaded frozen checkpoint "
        f"{checkpoint_path} | input_dim={input_dim}, latent_dim={latent_dim}, "
        f"hidden_dims={hidden_dims}, source_fit_cells={len(fit_idx)}"
    )
    return model, fit_idx


def _decode_gaga_trajectories(
    trajectories: np.ndarray,
    model: Autoencoder,
    device: str,
    batch_size: int = 8192,
) -> np.ndarray:
    shape = trajectories.shape
    flat = trajectories.reshape(-1, shape[-1]).astype("float32", copy=False)
    decoded_batches = []

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for start in range(0, flat.shape[0], batch_size):
            batch = torch.as_tensor(flat[start : start + batch_size], device=device)
            decoded_batches.append(model.decode(batch).cpu().numpy())

    decoded_scaled = np.concatenate(decoded_batches, axis=0)
    scaler = getattr(model, "input_scaler", None)
    decoded = scaler.inverse_transform(decoded_scaled) if scaler is not None else decoded_scaled
    return decoded.reshape(shape[0], shape[1], -1)


def _flatten_gaga_history(history: dict) -> pd.DataFrame:
    rows = []
    for phase, phase_history in history.items():
        n_epochs = max((len(values) for values in phase_history.values()), default=0)
        for epoch in range(n_epochs):
            row = {"phase": phase, "epoch": epoch + 1}
            for key, values in phase_history.items():
                row[key] = values[epoch] if epoch < len(values) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MIOFlow on the shared RNA-only TraInfBench space.")
    parser.add_argument("--input-h5ad", type=Path, default=Path("data/moscot_rna_cytobridge.h5ad"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/mioflow_moscot_rna_20000"))
    parser.add_argument("--latent-key", default="X_latent")
    parser.add_argument(
        "--require-shared-pca",
        action="store_true",
        help=(
            "Require adata.obsm[latent_key] to equal adata.obsm['X_pca'] "
            "within 1e-6 and record an input checksum."
        ),
    )
    parser.add_argument(
        "--input-space",
        choices=("obsm", "moscot-normalized"),
        default="moscot-normalized",
        help=(
            "Space passed into MIOFlow. obsm uses --latent-key as stored; "
            "moscot-normalized uses adata.obsm[latent_key] / scale from --norm-params."
        ),
    )
    parser.add_argument(
        "--model-input-key",
        default="X_mioflow_input",
        help="Temporary adata.obsm key used when --input-space creates a transformed input.",
    )
    parser.add_argument("--time-key", default="time_point_processed")
    parser.add_argument(
        "--time-axis",
        choices=("rank", "numeric"),
        default="rank",
        help=(
            "rank keeps MIOFlow's native consecutive snapshot encoding; numeric "
            "restores the numeric values from --time-key after encoding."
        ),
    )
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--sample-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lambda-ot", type=float, default=1.0)
    parser.add_argument("--use-density-loss", action="store_true")
    parser.add_argument("--lambda-density", type=float, default=0.1)
    parser.add_argument("--lambda-energy", type=float, default=0.01)
    parser.add_argument("--energy-time-steps", type=int, default=10)
    parser.add_argument("--n-trajectories", type=int, default=1000)
    parser.add_argument("--n-bins", type=int, default=101)
    parser.add_argument("--loss-log-every", type=int, default=10)
    parser.add_argument("--use-gaga", dest="use_gaga", action="store_true", default=True)
    parser.add_argument("--no-gaga", dest="use_gaga", action="store_false")
    parser.add_argument("--gaga-latent-dim", type=int, default=10)
    parser.add_argument("--gaga-hidden-dims", default="128,64")
    parser.add_argument("--gaga-max-cells", type=int, default=5000)
    parser.add_argument("--gaga-batch-size", type=int, default=1024)
    parser.add_argument("--gaga-encoder-epochs", type=int, default=100)
    parser.add_argument("--gaga-decoder-epochs", type=int, default=100)
    parser.add_argument("--gaga-learning-rate", type=float, default=1e-3)
    parser.add_argument("--gaga-dist-weight", type=float, default=1.0)
    parser.add_argument("--gaga-recon-weight", type=float, default=1.0)
    parser.add_argument("--gaga-distance-key", default=None)
    parser.add_argument(
        "--gaga-checkpoint",
        type=Path,
        default=None,
        help=(
            "Load a frozen externally trained GAGA checkpoint instead of fitting "
            "GAGA again. The checkpoint must contain the model, input scaler, and "
            "fit-cell provenance saved by this runner."
        ),
    )
    parser.add_argument("--gaga-distance-components", type=int, default=10)
    parser.add_argument("--gaga-phate-knn", type=int, default=5)
    parser.add_argument("--gaga-phate-landmarks", type=int, default=2000)
    parser.add_argument(
        "--embedding-normalization",
        choices=("identity", "global-scale", "zscore"),
        default="identity",
        help=(
            "Normalization inside MIOFlow. identity leaves the current model "
            "embedding unchanged; global-scale is for no-GAGA PCA-space training; "
            "zscore is the original MIOFlow behavior."
        ),
    )
    parser.add_argument(
        "--embedding-normalization-checkpoint",
        type=Path,
        default=None,
        help=(
            "For zscore normalization, load and freeze mean_vals/std_vals from "
            "a completed Full MIOFlow model checkpoint. This keeps Full and LOO "
            "in exactly the same normalized GAGA coordinate system."
        ),
    )
    parser.add_argument(
        "--norm-params",
        type=Path,
        default=Path("external/COATI/moscot/data/primal_norm_params.pt"),
        help="Torch file with {'scale': ...}, used only by --embedding-normalization global-scale.",
    )
    parser.add_argument(
        "--normalization-scale",
        type=float,
        default=None,
        help="Explicit scale override for --embedding-normalization global-scale.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--scheduler-type", choices=("none", "step", "exponential", "cosine"), default="none")
    parser.add_argument("--scheduler-step-size", type=int, default=30)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--scheduler-t-max", type=int, default=None)
    parser.add_argument("--scheduler-min-lr", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    adata = ad.read_h5ad(args.input_h5ad)
    if args.latent_key not in adata.obsm:
        raise KeyError(f"{args.latent_key!r} not found in adata.obsm. Available keys: {list(adata.obsm.keys())}")
    if args.time_key not in adata.obs:
        raise KeyError(f"{args.time_key!r} not found in adata.obs. Available columns: {list(adata.obs.columns)}")

    x_source = np.asarray(adata.obsm[args.latent_key], dtype="float32")
    if x_source.ndim != 2:
        raise ValueError(f"adata.obsm[{args.latent_key!r}] must be a 2D matrix, got shape {x_source.shape}")
    source_pca_sha256 = _array_sha256(x_source)
    x_latent_x_pca_max_abs_difference = None
    if "X_pca" in adata.obsm:
        x_pca = np.asarray(adata.obsm["X_pca"], dtype="float32")
        if x_pca.shape != x_source.shape:
            raise ValueError(
                f"adata.obsm['X_pca'] shape {x_pca.shape} does not match "
                f"adata.obsm[{args.latent_key!r}] shape {x_source.shape}"
            )
        x_latent_x_pca_max_abs_difference = float(
            np.max(np.abs(x_source - x_pca))
        )
    if args.require_shared_pca:
        if x_latent_x_pca_max_abs_difference is None:
            raise KeyError(
                "--require-shared-pca requested but adata.obsm['X_pca'] is absent"
            )
        if x_latent_x_pca_max_abs_difference > 1e-6:
            raise ValueError(
                f"adata.obsm[{args.latent_key!r}] is not the exact shared PCA; "
                f"max_abs_difference={x_latent_x_pca_max_abs_difference:.6g}"
            )
    if args.use_gaga and args.embedding_normalization == "global-scale":
        raise ValueError(
            "--embedding-normalization global-scale applies to PCA-space MIOFlow. "
            "With --use-gaga, use identity or zscore and decode trajectories back to PCA."
        )
    if args.embedding_normalization_checkpoint is not None:
        if not args.use_gaga:
            raise ValueError(
                "--embedding-normalization-checkpoint currently requires --use-gaga"
            )
        if args.embedding_normalization != "zscore":
            raise ValueError(
                "--embedding-normalization-checkpoint requires "
                "--embedding-normalization zscore"
            )
        if not args.embedding_normalization_checkpoint.is_file():
            raise FileNotFoundError(args.embedding_normalization_checkpoint)

    input_space_scale = 1.0
    model_input_key = args.latent_key
    x = x_source
    if args.input_space == "moscot-normalized":
        input_space_scale = _load_norm_scale(args.norm_params)
        if input_space_scale <= 0:
            raise ValueError(f"Input space scale must be positive, got {input_space_scale}")
        x = (x_source / input_space_scale).astype("float32", copy=False)
        model_input_key = args.model_input_key
        adata.obsm[model_input_key] = x

    normalization_scale = None
    if args.embedding_normalization == "global-scale":
        normalization_scale = (
            float(args.normalization_scale)
            if args.normalization_scale is not None
            else _load_norm_scale(args.norm_params)
        )

    frozen_embedding_mean = None
    frozen_embedding_std = None
    if args.embedding_normalization_checkpoint is not None:
        frozen_embedding_mean, frozen_embedding_std = (
            _load_embedding_normalization_checkpoint(
                args.embedding_normalization_checkpoint,
                args.gaga_latent_dim,
            )
        )

    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    time_values = _sorted_time_values(adata, args.time_key)
    print(f"input: {args.input_h5ad}")
    print(f"output: {args.output_dir}")
    print(f"cells: {adata.n_obs}, input_dim: {x.shape[1]}, time_key: {args.time_key}")
    print(f"time_values: {time_values}")
    print(
        "input_space: "
        f"{args.input_space}, source_key: {args.latent_key}, "
        f"model_input_key: {model_input_key}, scale: {input_space_scale:.12g}"
    )
    print(
        "shared_pca_audit: "
        f"required={args.require_shared_pca}, "
        f"max_abs_difference={x_latent_x_pca_max_abs_difference}, "
        f"sha256={source_pca_sha256[:12]}"
    )
    print(f"epochs: {args.epochs}, sample_size: {args.sample_size}, device: {'cuda' if use_cuda else 'cpu'}")
    print(f"use_gaga: {args.use_gaga}")
    print(
        "loss_weights: "
        f"lambda_ot={args.lambda_ot}, "
        f"use_density_loss={args.use_density_loss}, lambda_density={args.lambda_density}, "
        f"lambda_energy={args.lambda_energy}"
    )
    print(
        "scheduler: "
        f"{args.scheduler_type}, step_size={args.scheduler_step_size}, gamma={args.scheduler_gamma}, "
        f"t_max={args.scheduler_t_max}, min_lr={args.scheduler_min_lr}"
    )
    print(
        "embedding_normalization: "
        f"{args.embedding_normalization}"
        f"{f', scale: {normalization_scale:.12g}' if normalization_scale is not None else ''}"
    )
    if args.embedding_normalization_checkpoint is not None:
        print(
            "embedding_normalization_checkpoint: "
            f"{args.embedding_normalization_checkpoint}"
        )

    start = time.time()
    device = "cuda" if use_cuda else "cpu"
    gaga_model = None
    gaga_history = None
    gaga_fit_idx = None
    if args.use_gaga:
        if args.gaga_checkpoint is not None:
            gaga_model, gaga_fit_idx = _load_gaga_checkpoint(
                args.gaga_checkpoint,
                x,
                args,
                device,
            )
        else:
            gaga_model, gaga_history, gaga_fit_idx = _fit_gaga(
                adata, x, args, device
            )
            _flatten_gaga_history(gaga_history).to_csv(
                args.output_dir / "gaga_losses.csv", index=False
            )
        torch.save(
            {
                "state_dict": gaga_model.state_dict(),
                "input_dim": int(x.shape[1]),
                "latent_dim": int(args.gaga_latent_dim),
                "hidden_dims": _parse_hidden_dims(args.gaga_hidden_dims),
                "input_scaler_mean": torch.as_tensor(gaga_model.input_scaler.mean_, dtype=torch.float32),
                "input_scaler_scale": torch.as_tensor(gaga_model.input_scaler.scale_, dtype=torch.float32),
                "input_scaler_n_samples_seen": int(
                    np.asarray(gaga_model.input_scaler.n_samples_seen_).max()
                ),
                "fit_indices": torch.as_tensor(gaga_fit_idx, dtype=torch.long),
                "fit_indices_coordinate_system": (
                    "source_checkpoint_input"
                    if args.gaga_checkpoint is not None
                    else "current_input"
                ),
                "source_checkpoint": (
                    str(args.gaga_checkpoint.resolve())
                    if args.gaga_checkpoint is not None
                    else None
                ),
                "config": {k: _jsonable(v) for k, v in vars(args).items()},
            },
            args.output_dir / "gaga_model.pt",
        )

    mf = MIOFlow(
        adata,
        gaga_model=gaga_model,
        gaga_input_key=model_input_key,
        obs_time_key=args.time_key,
        hidden_dim=args.hidden_dim,
        use_cuda=use_cuda,
        n_epochs=args.epochs,
        lambda_ot=args.lambda_ot,
        use_density_loss=args.use_density_loss,
        lambda_density=args.lambda_density,
        lambda_energy=args.lambda_energy,
        energy_time_steps=args.energy_time_steps,
        learning_rate=args.learning_rate,
        sample_size=args.sample_size,
        exp_dir=str(args.output_dir),
        n_trajectories=args.n_trajectories,
        n_bins=args.n_bins,
        loss_log_every=args.loss_log_every,
        embedding_normalization=args.embedding_normalization,
        embedding_scale=normalization_scale,
        embedding_mean=frozen_embedding_mean,
        embedding_std=frozen_embedding_std,
        scheduler_type=None if args.scheduler_type == "none" else args.scheduler_type,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        scheduler_t_max=args.scheduler_t_max,
        scheduler_min_lr=args.scheduler_min_lr,
    )
    if args.time_axis == "numeric":
        _replace_mioflow_rank_times_with_numeric_times(mf, adata, args.time_key)
    if gaga_model is not None:
        np.savez_compressed(
            args.output_dir / "gaga10_embedding.npz",
            embedding_raw=np.asarray(mf.embedding, dtype=np.float32),
            embedding_model_input=np.asarray(
                mf.normalized_embedding, dtype=np.float32
            ),
            time_labels=np.asarray(adata.obs[args.time_key]),
            cell_ids=np.asarray(adata.obs_names.astype(str)),
            embedding_normalization=np.asarray(args.embedding_normalization),
            embedding_normalization_checkpoint=np.asarray(
                str(args.embedding_normalization_checkpoint.resolve())
                if args.embedding_normalization_checkpoint is not None
                else ""
            ),
            source_h5ad=np.asarray(str(args.input_h5ad.resolve())),
            gaga_checkpoint=np.asarray(
                str((args.output_dir / "gaga_model.pt").resolve())
            ),
        )
    print(f"mioflow_time_axis: {args.time_axis}, model_time_points: {mf.dataset.times}")
    mf.fit()
    elapsed = time.time() - start
    ode_input_dim = int(mf.dataset.time_series_data[0][0].shape[1])
    trajectories_model_space = mf.trajectories.astype("float32", copy=False)
    trajectories_input_space = (
        _decode_gaga_trajectories(mf.trajectories, gaga_model, device)
        if gaga_model is not None
        else mf.trajectories
    ).astype("float32", copy=False)
    trajectories_raw_pca = (
        trajectories_input_space * input_space_scale
        if args.input_space == "moscot-normalized"
        else trajectories_input_space
    ).astype("float32", copy=False)

    losses = pd.DataFrame(mf.losses)
    losses.to_csv(args.output_dir / "losses.csv", index=False)
    np.savez_compressed(
        args.output_dir / "trajectories.npz",
        trajectories=trajectories_input_space,
        trajectories_input_space=trajectories_input_space,
        trajectories_pca=trajectories_raw_pca,
        trajectories_raw_pca=trajectories_raw_pca,
        trajectories_model_space=trajectories_model_space,
        time_grid=np.linspace(min(mf.dataset.times), max(mf.dataset.times), args.n_bins, dtype="float32"),
        mean_vals=mf.mean_vals.astype("float32", copy=False),
        std_vals=mf.std_vals.astype("float32", copy=False),
        embedding_normalization=np.asarray(args.embedding_normalization),
        embedding_normalization_checkpoint=np.asarray(
            str(args.embedding_normalization_checkpoint.resolve())
            if args.embedding_normalization_checkpoint is not None
            else ""
        ),
        normalization_scale=np.asarray(normalization_scale if normalization_scale is not None else 1.0, dtype="float32"),
        input_space=np.asarray(args.input_space),
        input_space_scale=np.asarray(input_space_scale, dtype="float32"),
        source_latent_key=np.asarray(args.latent_key),
        model_input_key=np.asarray(model_input_key),
        use_gaga=np.asarray(args.use_gaga),
        gaga_latent_dim=np.asarray(args.gaga_latent_dim if args.use_gaga else x.shape[1], dtype="int32"),
    )
    torch.save(
        {
            "model_state_dict": mf.ode_model.state_dict(),
            "input_dim": ode_input_dim,
            "source_pca_dim": int(x_source.shape[1]),
            "source_pca_sha256": source_pca_sha256,
            "x_latent_x_pca_max_abs_difference": (
                x_latent_x_pca_max_abs_difference
            ),
            "input_space_dim": int(x.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "momentum_beta": float(mf.momentum_beta),
            "mean_vals": torch.as_tensor(mf.mean_vals, dtype=torch.float32),
            "std_vals": torch.as_tensor(mf.std_vals, dtype=torch.float32),
            "time_points": list(mf.dataset.times),
            "latent_key": args.latent_key,
            "model_input_key": model_input_key,
            "input_space": args.input_space,
            "input_space_scale": input_space_scale,
            "time_key": args.time_key,
            "use_gaga": bool(args.use_gaga),
            "gaga_model_path": str(args.output_dir / "gaga_model.pt") if args.use_gaga else None,
            "embedding_normalization": args.embedding_normalization,
            "embedding_normalization_checkpoint": (
                str(args.embedding_normalization_checkpoint.resolve())
                if args.embedding_normalization_checkpoint is not None
                else None
            ),
            "normalization_scale": normalization_scale,
            "config": {k: _jsonable(v) for k, v in vars(args).items()},
            "losses": mf.losses,
        },
        args.output_dir / "model.pt",
    )
    metadata = {
        "input_h5ad": str(args.input_h5ad),
        "output_dir": str(args.output_dir),
        "n_obs": int(adata.n_obs),
        "latent_key": args.latent_key,
        "model_input_key": model_input_key,
        "source_pca_dim": int(x_source.shape[1]),
        "source_pca_sha256": source_pca_sha256,
        "x_latent_x_pca_max_abs_difference": (
            x_latent_x_pca_max_abs_difference
        ),
        "require_shared_pca": bool(args.require_shared_pca),
        "input_space_dim": int(x.shape[1]),
        "input_space": args.input_space,
        "input_space_scale": input_space_scale,
        "model_space_dim": ode_input_dim,
        "time_key": args.time_key,
        "time_values": time_values,
        "mioflow_time_points": list(mf.dataset.times),
        "elapsed_seconds": elapsed,
        "device": "cuda" if use_cuda else "cpu",
        "checkpoint": str(args.output_dir / "model.pt"),
        "losses_live_csv": str(args.output_dir / "losses_live.csv"),
        "loss_log_every": int(args.loss_log_every),
        "embedding_normalization": args.embedding_normalization,
        "embedding_normalization_checkpoint": (
            str(args.embedding_normalization_checkpoint.resolve())
            if args.embedding_normalization_checkpoint is not None
            else None
        ),
        "normalization_scale": normalization_scale,
        "norm_params": str(args.norm_params) if args.embedding_normalization == "global-scale" else None,
        "use_gaga": bool(args.use_gaga),
        "gaga_model": str(args.output_dir / "gaga_model.pt") if args.use_gaga else None,
        "gaga_checkpoint_source": (
            str(args.gaga_checkpoint.resolve())
            if args.gaga_checkpoint is not None
            else None
        ),
        "gaga_embedding": (
            str(args.output_dir / "gaga10_embedding.npz")
            if args.use_gaga
            else None
        ),
        "gaga_losses_csv": (
            str(args.output_dir / "gaga_losses.csv")
            if args.use_gaga and gaga_history is not None
            else None
        ),
        "gaga_latent_dim": int(args.gaga_latent_dim) if args.use_gaga else None,
        "gaga_max_cells": int(args.gaga_max_cells) if args.use_gaga else None,
        "training_space": (
            f"MIOFlow GAGA latent space learned from {args.input_space}; saved trajectories are decoded to that input space"
            if args.use_gaga
            else {
                "identity": args.input_space,
                "global-scale": "model input divided by the moscot global scale",
                "zscore": f"feature-wise z-scored {args.input_space}",
            }[args.embedding_normalization]
        ),
        "saved_trajectory_space": args.input_space,
        "raw_pca_trajectory_key": "trajectories_pca",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"elapsed_seconds: {elapsed:.2f}")
    if args.use_gaga:
        print(f"wrote: {args.output_dir / 'gaga_model.pt'}")
        print(f"wrote: {args.output_dir / 'gaga10_embedding.npz'}")
        if gaga_history is not None:
            print(f"wrote: {args.output_dir / 'gaga_losses.csv'}")
    print(f"wrote: {args.output_dir / 'model.pt'}")
    print(f"wrote: {args.output_dir / 'losses.csv'}")
    print(f"wrote: {args.output_dir / 'trajectories.npz'}")


if __name__ == "__main__":
    main()
