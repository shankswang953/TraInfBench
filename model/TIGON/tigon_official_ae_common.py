"""Shared TIGON official-style autoencoder utilities.

The architecture and optimization settings mirror the public TIGON
``AE/{models,trainer}.py`` implementation used by the EMT example.  Dataset
preparation and trajectory training are intentionally kept in separate entry
points so every Full/LOO experiment can reuse one frozen full-data embedding.
"""

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
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


OFFICIAL_SOURCE_COMMIT = "1ed92cfcc250415fc01b4d344a308b0680cc9635"
OFFICIAL_SOURCE_URL = "https://github.com/yutongo/TIGON"
TIGON_SAFE_REPRODUCTION_PROTOCOL = "gastrulation_safe_reproduction_v1"
TIGON_UPSTREAM_PUBLIC_EXACT_PROTOCOL = (
    "gastrulation_tigon_upstream_public_exact_v2"
)
TIGON_DENSITY_SIGMA_CLIP = 0.0625
TIGON_AE10_PUBLICATION_SIGMA_MIN = 0.125


def load_tigon_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "model" / "TIGON" / "run_tigon_moscot_ae.py"
    )
    spec = importlib.util.spec_from_file_location(
        "trainfbench_tigon_density_frozen_ae", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load TIGON module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class OfficialMLP(nn.Module):
    """MLP layout used by the public TIGON autoencoder."""

    def __init__(
        self,
        layers_list: list[int],
        dropout: float,
        norm: bool,
        activation: nn.Module | None,
        last_act: bool = False,
    ):
        super().__init__()
        if len(layers_list) < 2:
            raise ValueError("Need at least input and output dimensions")
        layers: list[nn.Module] = []
        for i in range(len(layers_list) - 2):
            layers.append(nn.Linear(layers_list[i], layers_list[i + 1]))
            if norm:
                layers.append(nn.BatchNorm1d(layers_list[i + 1]))
            if activation is not None:
                layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(layers_list[-2], layers_list[-1]))
        if norm:
            layers.append(nn.BatchNorm1d(layers_list[-1]))
        if last_act and activation is not None:
            layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class OfficialAutoEncoder(nn.Module):
    """TIGON EMT-style AE with one 300-unit hidden layer."""

    def __init__(
        self,
        in_dim: int,
        n_layers: int = 1,
        n_hidden: int = 300,
        n_latent: int = 10,
        dropout: float = 0.2,
        norm: bool = True,
        seed: int = 4232,
    ):
        super().__init__()
        torch.manual_seed(seed)
        activation = nn.ReLU()
        encoder_layers = [in_dim] + [n_hidden] * n_layers
        decoder_layers = [n_latent] + [n_hidden] * n_layers
        self.encoder = OfficialMLP(
            encoder_layers, dropout, norm, activation, last_act=True
        )
        self.encoder_to_latent = OfficialMLP(
            [encoder_layers[-1], n_latent],
            dropout,
            norm,
            activation,
        )
        self.decoder = OfficialMLP(
            decoder_layers, dropout, norm, activation, last_act=True
        )
        self.decoder_to_output = OfficialMLP(
            [decoder_layers[-1], in_dim],
            dropout,
            norm,
            activation=None,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_to_latent(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_to_output(self.decoder(z))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction = self.decode(self.encode(x))
        return torch.sum((reconstruction - x) ** 2) / x.shape[1]


def train_official_ae(
    expression: np.ndarray,
    *,
    device: torch.device,
    outdir: Path,
    seed: int = 4232,
    max_epochs: int = 500,
    n_latent: int = 10,
) -> tuple[OfficialAutoEncoder, np.ndarray, np.ndarray, np.ndarray, float]:
    """Train the official-style AE and persist the frozen full-data embedding."""
    if n_latent <= 0:
        raise ValueError("n_latent must be positive")
    model = OfficialAutoEncoder(
        in_dim=expression.shape[1],
        n_layers=1,
        n_hidden=300,
        n_latent=n_latent,
        dropout=0.2,
        norm=True,
        seed=seed,
    ).to(device)
    train_x, validation_x = train_test_split(
        expression,
        test_size=0.1,
        random_state=seed,
    )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x)),
        batch_size=128,
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation_x)),
        batch_size=128,
        shuffle=False,
    )
    # Public ``Trainer.__init__`` resets Torch after constructing the AE and
    # data loaders.  This makes the first shuffled epoch use the declared AE
    # seed rather than the RNG state left after parameter initialization.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    patience_epochs = 0
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu())
        train_loss /= len(train_loader.dataset)

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for (batch,) in validation_loader:
                validation_loss += float(model(batch.to(device)).detach().cpu())
        validation_loss /= len(validation_loader.dataset)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        pd.DataFrame(history).to_csv(
            outdir / "ae_training_history.csv", index=False
        )
        print(
            f"[official-ae] epoch={epoch} train={train_loss:.6g} "
            f"validation={validation_loss:.6g}",
            flush=True,
        )
        if best_validation - validation_loss >= 1e-2:
            best_validation = validation_loss
            patience_epochs = 0
        else:
            patience_epochs += 1
        if patience_epochs >= 30:
            print(
                "[official-ae] early stopping after 30 epochs without "
                "a validation decrease of at least 0.01",
                flush=True,
            )
            break

    model.eval()
    latent_batches: list[np.ndarray] = []
    reconstruction_sse = 0.0
    with torch.no_grad():
        for start in range(0, expression.shape[0], 512):
            batch = torch.from_numpy(expression[start : start + 512]).to(device)
            latent = model.encode(batch)
            reconstruction = model.decode(latent)
            latent_batches.append(latent.detach().cpu().numpy())
            reconstruction_sse += float(
                torch.sum((reconstruction - batch) ** 2).detach().cpu()
            )
    latent_raw = np.concatenate(latent_batches, axis=0).astype(
        np.float32, copy=False
    )
    latent_min = latent_raw.min(axis=0)
    latent_max = latent_raw.max(axis=0)
    latent_range = np.maximum(latent_max - latent_min, 1e-8)
    latent_scaled = (
        4.0 * (latent_raw - latent_min[None, :]) / latent_range[None, :] - 2.0
    ).astype(np.float32, copy=False)
    reconstruction_mse = reconstruction_sse / expression.size

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_dim": int(expression.shape[1]),
            "n_layers": 1,
            "n_hidden": 300,
            "n_latent": int(n_latent),
            "activation": "relu",
            "dropout": 0.2,
            "batch_norm": True,
            "seed": int(seed),
            "batch_size": 128,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "test_size": 0.1,
            "max_epochs": int(max_epochs),
            "early_stopping_tolerance": 1e-2,
            "early_stopping_patience": 30,
            "reconstruction_mse": reconstruction_mse,
        },
        outdir / "ae.pt",
    )
    np.save(outdir / "ae_latent_raw.npy", latent_raw)
    np.save(outdir / "ae_latent_scaled_minus2_2.npy", latent_scaled)
    np.savez_compressed(
        outdir / "ae_latent_scaling.npz",
        latent_min=latent_min,
        latent_max=latent_max,
        output_min=np.asarray(-2.0, dtype=np.float32),
        output_max=np.asarray(2.0, dtype=np.float32),
    )
    print(
        f"[official-ae] frozen latent={latent_scaled.shape}, "
        f"reconstruction_mse={reconstruction_mse:.6g}",
        flush=True,
    )
    return model, latent_scaled, latent_min, latent_max, reconstruction_mse


def build_tigon_density_args(
    *,
    outdir: Path,
    tigon_iters: int,
    num_samples: int,
    checkpoint_every: int,
    action_steps: int,
    seed: int,
    resume: bool,
    fixed_density_sigma: float | None = None,
    density_sigma_initial: float | None = None,
    density_sigma_min: float | None = None,
    density_sigma_hard_clip: float | None = TIGON_DENSITY_SIGMA_CLIP,
    official_density_sigma_schedule: bool = False,
    density_sigma_official_stop_threshold: float = 0.02,
    action_state_mode: str = "upstream_public_exact",
    density_ode_backend: str = "torchdiffeqpack",
) -> argparse.Namespace:
    """Return the audited official-style TIGON density configuration."""
    if fixed_density_sigma is not None and fixed_density_sigma <= 0:
        raise ValueError("fixed_density_sigma must be positive")
    if density_sigma_initial is not None and density_sigma_initial <= 0:
        raise ValueError("density_sigma_initial must be positive")
    if density_sigma_min is not None and density_sigma_min <= 0:
        raise ValueError("density_sigma_min must be positive")
    if not official_density_sigma_schedule and (
        density_sigma_hard_clip is None or density_sigma_hard_clip <= 0
    ):
        raise ValueError("density_sigma_hard_clip must be positive")
    if official_density_sigma_schedule:
        if fixed_density_sigma is not None or density_sigma_min is not None:
            raise ValueError(
                "The upstream sigma schedule cannot be combined with a fixed "
                "sigma or a density-sigma minimum"
            )
        if density_sigma_initial is not None and not np.isclose(
            density_sigma_initial, 1.0
        ):
            raise ValueError("The upstream sigma schedule must start at sigma=1")
        if density_sigma_official_stop_threshold <= 0:
            raise ValueError(
                "density_sigma_official_stop_threshold must be positive"
            )
    if fixed_density_sigma is not None and density_sigma_min is not None:
        raise ValueError(
            "fixed_density_sigma and density_sigma_min are mutually exclusive"
        )
    if fixed_density_sigma is not None and density_sigma_initial is not None:
        raise ValueError(
            "fixed_density_sigma and density_sigma_initial are mutually exclusive"
        )
    if action_state_mode not in {
        "upstream_public_exact",
        "paper_fixed_sample",
        "official_stationary",
    }:
        raise ValueError(f"Unknown TIGON action-state mode: {action_state_mode}")
    if density_ode_backend not in {"torchdiffeqpack", "torchdiffeq"}:
        raise ValueError(
            f"Unknown TIGON density ODE backend: {density_ode_backend}"
        )
    density_sigma_initial = (
        (
            1.0
            if density_sigma_initial is None
            else float(density_sigma_initial)
        )
        if fixed_density_sigma is None
        else float(fixed_density_sigma)
    )
    if official_density_sigma_schedule:
        effective_density_sigma_min = None
        density_sigma_initial = 1.0
    else:
        effective_density_sigma_min = (
            max(
                density_sigma_hard_clip,
                (
                    density_sigma_hard_clip
                    if density_sigma_min is None
                    else float(density_sigma_min)
                ),
            )
            if fixed_density_sigma is None
            else max(float(fixed_density_sigma), density_sigma_hard_clip)
        )
        density_sigma_initial = max(
            density_sigma_initial, effective_density_sigma_min
        )
    density_sigma_anneal_every = 100 if fixed_density_sigma is None else 0
    return argparse.Namespace(
        tigon_hidden_dim=16,
        tigon_hidden_layers=4,
        tigon_lr=3e-3,
        tigon_iters=int(tigon_iters),
        num_samples=int(num_samples),
        density_sample_sigma=0.02,
        density_sigma_initial=density_sigma_initial,
        density_sigma_min=effective_density_sigma_min,
        density_sigma_hard_clip=(
            None
            if official_density_sigma_schedule
            else float(density_sigma_hard_clip)
        ),
        official_density_sigma_schedule=bool(
            official_density_sigma_schedule
        ),
        density_sigma_official_stop_threshold=(
            float(density_sigma_official_stop_threshold)
            if official_density_sigma_schedule
            else None
        ),
        density_sigma_anneal_every=density_sigma_anneal_every,
        density_sigma_anneal_factor=0.5,
        density_sigma_anneal_threshold=3e-4,
        density_sigma_anneal_stop_before_end=400,
        density_kde_chunk_size=1024,
        density_stabilization="official",
        density_weight=1e4,
        divergence_estimator="exact",
        divergence_fd_epsilon=1e-2,
        ode_solver="dopri5",
        density_ode_backend=density_ode_backend,
        ode_rtol=1e-3,
        ode_atol=1e-5,
        ode_steps=8,
        action_ode_solver="midpoint",
        action_ode_steps=int(action_steps),
        action_state_mode=action_state_mode,
        safe_reproduction_protocol=(
            TIGON_UPSTREAM_PUBLIC_EXACT_PROTOCOL
            if action_state_mode == "upstream_public_exact"
            else TIGON_SAFE_REPRODUCTION_PROTOCOL
        ),
        growth_action_weight=1.0,
        max_grad_norm=0.0,
        checkpoint_every=int(checkpoint_every),
        resume_tigon=bool(resume),
        tigon_init_checkpoint=None,
        skip_tigon_train=False,
        eval_samples=2048,
        sinkhorn_blur=0.05,
        seed=int(seed),
        outdir=str(outdir),
    )
