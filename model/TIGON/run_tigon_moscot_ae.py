#!/usr/bin/env python3
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as st
import torch
import torch.nn as nn
from geomloss import SamplesLoss
from torchdiffeq import odeint
from torch.utils.data import DataLoader, TensorDataset


TIGON_DENSITY_SIGMA_HARD_CLIP = 0.0625
_TORCHDIFFEQPACK_ODESOLVE = None


def _load_upstream_odesolve():
    """Load the exact ODE backend vendored with the frozen TIGON commit."""
    global _TORCHDIFFEQPACK_ODESOLVE
    if _TORCHDIFFEQPACK_ODESOLVE is None:
        dependency_root = (
            Path(__file__).resolve().parents[2]
            / "external" / "TIGON_upstream_1ed92cf"
            / "_deps"
        )
        if not dependency_root.exists():
            raise FileNotFoundError(
                "The frozen TIGON TorchDiffEqPack dependency is missing: "
                f"{dependency_root}"
            )
        dependency_text = str(dependency_root)
        if dependency_text not in sys.path:
            sys.path.insert(0, dependency_text)
        from TorchDiffEqPack import odesolve as upstream_odesolve

        _TORCHDIFFEQPACK_ODESOLVE = upstream_odesolve
    return _TORCHDIFFEQPACK_ODESOLVE


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a TIGON-style AE + growth/velocity model on RNA data."
    )
    p.add_argument(
        "--rna-h5ad",
        default="external/COATI/moscot/data/rna_adata.h5ad",
    )
    p.add_argument(
        "--outdir",
        default="external/COATI/moscot/TIGON/results/moscot_ae_tigon_smoke",
    )
    p.add_argument("--layer", default="sct_logcounts")
    p.add_argument(
        "--embedding-key",
        default=None,
        help=(
            "Optional adata.obsm key. When set, TIGON is trained on this "
            "embedding instead of selecting genes from X/layer."
        ),
    )
    p.add_argument(
        "--embedding-normalization",
        choices=("identity", "global-scale", "per-axis-minmax-minus2-2"),
        default="identity",
        help=(
            "Normalization applied to --embedding-key before TIGON training. "
            "global-scale divides the embedding by --normalization-scale or "
            "the {'scale': ...} value in --norm-params; "
            "per-axis-minmax-minus2-2 applies TIGON's published full-data "
            "coordinate-wise scaling to [-2, 2]."
        ),
    )
    p.add_argument(
        "--embedding-first-dims",
        type=int,
        default=0,
        help=(
            "When positive, keep only the first N coordinates from "
            "--embedding-key before normalization and training."
        ),
    )
    p.add_argument(
        "--norm-params",
        default="data/gastrulation_rna_primal_norm_params.pt",
        help="Torch file containing {'scale': ...}, used by --embedding-normalization global-scale.",
    )
    p.add_argument(
        "--normalization-scale",
        type=float,
        default=None,
        help="Explicit scale override for --embedding-normalization global-scale.",
    )
    p.add_argument("--time-key", default="stage_num")
    p.add_argument("--stage-key", default="stage")
    p.add_argument("--celltype-key", default="cell_type_refined")
    p.add_argument("--hvg-key", default="highly_variable")
    p.add_argument("--hvg-top", type=int, default=500)
    p.add_argument("--latent-dim", type=int, default=6)
    p.add_argument("--ae-hidden-dim", type=int, default=300)
    p.add_argument("--ae-epochs", type=int, default=5)
    p.add_argument("--ae-batch-size", type=int, default=1024)
    p.add_argument("--ae-lr", type=float, default=1e-3)
    p.add_argument("--use-ae", dest="use_ae", action="store_true", default=True)
    p.add_argument("--no-ae", dest="use_ae", action="store_false")
    p.add_argument("--tigon-iters", type=int, default=100)
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--tigon-hidden-dim", type=int, default=64)
    p.add_argument("--tigon-hidden-layers", type=int, default=4)
    p.add_argument("--tigon-lr", type=float, default=3e-3)
    p.add_argument(
        "--training-objective",
        choices=("sinkhorn-adjacent", "tigon-density"),
        default="sinkhorn-adjacent",
        help=(
            "sinkhorn-adjacent preserves the original TraInfBench approximation; "
            "tigon-density uses TIGON's short- plus long-term density reconstruction objective."
        ),
    )
    p.add_argument("--ode-solver", default="dopri5")
    p.add_argument(
        "--density-ode-backend",
        choices=("torchdiffeq", "torchdiffeqpack"),
        default="torchdiffeq",
        help=(
            "ODE backend for density reconstruction. torchdiffeqpack uses "
            "the Dopri5 implementation bundled with the audited TIGON source."
        ),
    )
    p.add_argument("--ode-rtol", type=float, default=1e-3)
    p.add_argument("--ode-atol", type=float, default=1e-5)
    p.add_argument(
        "--action-ode-solver",
        choices=("same", "euler", "midpoint", "rk4", "dopri5"),
        default="same",
        help=(
            "Solver for the WFR action integral. 'same' reuses --ode-solver; "
            "midpoint matches the action integration in the upstream TIGON implementation."
        ),
    )
    p.add_argument(
        "--action-ode-steps",
        type=int,
        default=0,
        help=(
            "Fixed steps across the full initial-to-terminal action interval; "
            "<=0 reuses --ode-steps."
        ),
    )
    p.add_argument(
        "--action-state-mode",
        choices=("trajectory", "upstream_public_exact", "paper_fixed_sample"),
        default="trajectory",
        help=(
            "State convention for the WFR action. upstream_public_exact "
            "reproduces the public TIGON trans_loss implementation."
        ),
    )
    p.add_argument(
        "--divergence-estimator",
        choices=("exact", "hutchinson", "finite-difference"),
        default="exact",
        help=(
            "exact matches TIGON but needs higher-order autodiff; hutchinson uses an autodiff trace "
            "estimate; finite-difference uses ordinary first-order parameter gradients."
        ),
    )
    p.add_argument("--divergence-fd-epsilon", type=float, default=1e-2)
    p.add_argument("--density-weight", type=float, default=1e4)
    p.add_argument("--density-sigma-initial", type=float, default=1.0)
    p.add_argument(
        "--official-density-sigma-schedule",
        action="store_true",
        help=(
            "Use upstream TIGON's exact bandwidth rule: start at sigma=1, "
            "halve while the current sigma is >0.02 when the reconstruction "
            "criterion passes, and do not clamp the updated value to a floor."
        ),
    )
    p.add_argument(
        "--density-sigma-min",
        type=float,
        default=TIGON_DENSITY_SIGMA_HARD_CLIP,
        help=(
            "Requested lower bound for density-bandwidth annealing. A shared "
            f"stability clip enforces sigma >= {TIGON_DENSITY_SIGMA_HARD_CLIP:g} "
            "for every TIGON density run."
        ),
    )
    p.add_argument(
        "--density-sigma-anneal-every",
        type=int,
        default=100,
        help="Check density-bandwidth annealing every N iterations; <=0 disables annealing.",
    )
    p.add_argument(
        "--density-sigma-anneal-factor",
        type=float,
        default=0.5,
        help="Multiplicative density-bandwidth annealing factor in (0, 1).",
    )
    p.add_argument(
        "--density-sigma-anneal-threshold",
        type=float,
        default=3e-4,
        help="Anneal when mean long-term raw density MSE is at or below this threshold.",
    )
    p.add_argument(
        "--density-sigma-anneal-stop-before-end",
        type=int,
        default=400,
        help="Stop density-bandwidth annealing this many iterations before training ends.",
    )
    p.add_argument("--density-sample-sigma", type=float, default=0.02)
    p.add_argument("--density-kde-chunk-size", type=int, default=1024)
    p.add_argument(
        "--density-stabilization",
        choices=("relative", "official", "log"),
        default="relative",
        help=(
            "relative evaluates TIGON's density MSE in a shared batch-relative scale to avoid "
            "40D underflow; official uses raw densities; log compares log densities."
        ),
    )
    p.add_argument("--growth-action-weight", type=float, default=1.0)
    p.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.0,
        help="Clip the TIGON density-objective gradient norm when positive; 0 disables clipping.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help="Save TIGON training checkpoints every N iterations; <=0 disables periodic checkpoints.",
    )
    p.add_argument(
        "--resume-tigon",
        action="store_true",
        help="Resume TIGON training from outdir/tigon_checkpoint_latest.pt.",
    )
    p.add_argument(
        "--tigon-init-checkpoint",
        default=None,
        help=(
            "Initialize the TIGON vector field from a checkpoint, but start a fresh "
            "optimizer, schedule, history, and iteration count. This is useful for "
            "staged bandwidth refinement in a new output directory."
        ),
    )
    p.add_argument("--ode-steps", type=int, default=8)
    p.add_argument("--sinkhorn-blur", type=float, default=0.05)
    p.add_argument("--lambda-energy", type=float, default=0.01)
    p.add_argument("--lambda-mass", type=float, default=10.0)
    p.add_argument("--lambda-growth-l2", type=float, default=0.01)
    p.add_argument("--eval-samples", type=int, default=1024)
    p.add_argument("--saliency-cells-per-time", type=int, default=128)
    p.add_argument("--gene-chunk-size", type=int, default=512)
    p.add_argument("--top-n-genes", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-ae-train", action="store_true")
    p.add_argument("--skip-tigon-train", action="store_true")
    return p.parse_args()


def resolve_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if name == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested but not available.")
    return torch.device(name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_dense_float32(x):
    if sp.issparse(x):
        return x.toarray().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def load_norm_scale(args):
    if args.normalization_scale is not None:
        scale = float(args.normalization_scale)
    else:
        params = torch.load(args.norm_params, map_location="cpu", weights_only=False)
        if not isinstance(params, dict) or "scale" not in params:
            raise ValueError(f"Expected {args.norm_params} to contain a dict with key 'scale'")
        scale = float(np.asarray(params["scale"]).reshape(-1)[0])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid normalization scale: {scale}")
    return scale


def select_genes(adata, args):
    var = adata.var
    if args.hvg_key in var:
        mask = np.asarray(var[args.hvg_key]).astype(bool)
    elif "sct.variable" in var:
        mask = np.asarray(var["sct.variable"]).astype(bool)
    else:
        mask = np.ones(adata.n_vars, dtype=bool)

    idx = np.flatnonzero(mask)
    if args.hvg_top and len(idx) > args.hvg_top:
        score_key = None
        for key in ["dispersions_norm", "sct.residual_variance", "dispersions", "means"]:
            if key in var:
                score_key = key
                break
        if score_key is not None:
            score = np.asarray(var[score_key])[idx].astype(float)
            order = np.argsort(np.nan_to_num(score, nan=-np.inf))[::-1]
            idx = idx[order[: args.hvg_top]]
        else:
            idx = idx[: args.hvg_top]
    idx = np.asarray(idx, dtype=int)
    genes = np.asarray(adata.var_names)[idx]
    return idx, genes


def load_dataset(args):
    adata = ad.read_h5ad(args.rna_h5ad, backed="r")
    resolved_scale = None
    if args.embedding_key:
        key = args.embedding_key
        if key not in adata.obsm and key.startswith("X_") and key[2:] in adata.obsm:
            key = key[2:]
        if key not in adata.obsm:
            raise KeyError(
                f"{args.embedding_key!r} not found in adata.obsm. "
                f"Available keys: {list(adata.obsm.keys())}"
            )
        x = np.asarray(adata.obsm[key], dtype=np.float32)
        if args.embedding_first_dims:
            if args.embedding_first_dims < 1 or args.embedding_first_dims > x.shape[1]:
                raise ValueError(
                    "--embedding-first-dims must be between 1 and the embedding "
                    f"dimension ({x.shape[1]}), got {args.embedding_first_dims}"
                )
            x = np.ascontiguousarray(x[:, : args.embedding_first_dims])
        gene_idx = np.arange(x.shape[1], dtype=int)
        genes = np.asarray([f"{key}_{i}" for i in range(x.shape[1])])
        if args.embedding_normalization == "global-scale":
            resolved_scale = load_norm_scale(args)
            x = (x / resolved_scale).astype(np.float32, copy=False)
        resolved_axis_min = None
        resolved_axis_max = None
        if args.embedding_normalization == "per-axis-minmax-minus2-2":
            resolved_axis_min = np.min(x, axis=0).astype(np.float32)
            resolved_axis_max = np.max(x, axis=0).astype(np.float32)
            axis_range = resolved_axis_max - resolved_axis_min
            bad = np.flatnonzero(~np.isfinite(axis_range) | (axis_range <= 0))
            if bad.size:
                raise ValueError(
                    "Cannot min-max scale constant/non-finite embedding axes: "
                    + ", ".join(map(str, bad.tolist()))
                )
            x = (
                4.0 * (x - resolved_axis_min[None, :]) / axis_range[None, :] - 2.0
            ).astype(np.float32, copy=False)
    else:
        if args.embedding_normalization != "identity":
            raise ValueError("--embedding-normalization is only supported with --embedding-key.")
        resolved_axis_min = None
        resolved_axis_max = None
        gene_idx, genes = select_genes(adata, args)
        layer = adata.layers[args.layer] if args.layer in adata.layers else adata.X
        x = to_dense_float32(layer[:, gene_idx])
    obs = adata.obs.copy()
    umap = np.asarray(adata.obsm["X_umap"], dtype=np.float32) if "X_umap" in adata.obsm else None
    adata.file.close()

    times = pd.to_numeric(obs[args.time_key], errors="raise").to_numpy().astype(np.float32)
    unique_times = np.array(sorted(np.unique(times)), dtype=np.float32)
    if len(unique_times) < 2:
        raise ValueError("Need at least two time points.")
    return {
        "x": x,
        "obs": obs,
        "genes": genes,
        "gene_idx": gene_idx,
        "times": times,
        "unique_times": unique_times,
        "umap": umap,
        "embedding_normalization_scale": resolved_scale,
        "embedding_normalization_axis_min": resolved_axis_min,
        "embedding_normalization_axis_max": resolved_axis_max,
    }


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=300, latent_dim=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_layers, activation=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), activation()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TIGONModel(nn.Module):
    def __init__(self, dim, hidden_dim=64, hidden_layers=4):
        super().__init__()
        self.dim = dim
        self.velocity_net = MLP(dim + 1, dim, hidden_dim, hidden_layers)
        self.growth_net = MLP(dim + 1, 1, hidden_dim, 3)

    def _tx(self, t, x):
        if not torch.is_tensor(t):
            t = torch.tensor(float(t), dtype=x.dtype, device=x.device)
        if t.ndim == 0:
            t = t.expand(x.shape[0], 1)
        elif t.ndim == 1:
            t = t.reshape(-1, 1).to(device=x.device, dtype=x.dtype)
        return torch.cat([t.to(device=x.device, dtype=x.dtype), x], dim=1)

    def velocity(self, t, x):
        return self.velocity_net(self._tx(t, x))

    def growth(self, t, x):
        return self.growth_net(self._tx(t, x))


def train_ae(x, args, device, outdir):
    ckpt_path = outdir / "ae.pt"
    model = AutoEncoder(x.shape[1], args.ae_hidden_dim, args.latent_dim).to(device)
    if args.skip_ae_train and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
    else:
        ds = TensorDataset(torch.from_numpy(x))
        loader = DataLoader(ds, batch_size=args.ae_batch_size, shuffle=True, drop_last=False)
        opt = torch.optim.Adam(model.parameters(), lr=args.ae_lr)
        history = []
        for epoch in range(1, args.ae_epochs + 1):
            model.train()
            total = 0.0
            n = 0
            for (xb,) in loader:
                xb = xb.to(device)
                recon, _ = model(xb)
                loss = torch.mean((recon - xb) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu()) * xb.shape[0]
                n += xb.shape[0]
            row = {"epoch": epoch, "reconstruction_mse": total / max(n, 1)}
            history.append(row)
            print(f"[ae] epoch={epoch} mse={row['reconstruction_mse']:.6f}", flush=True)
        pd.DataFrame(history).to_csv(outdir / "ae_training_history.csv", index=False)
        torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, ckpt_path)

    model.eval()
    zs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], args.ae_batch_size):
            xb = torch.from_numpy(x[start : start + args.ae_batch_size]).to(device)
            zs.append(model.encode(xb).detach().cpu().numpy())
    z = np.concatenate(zs, axis=0).astype(np.float32)
    np.save(outdir / "ae_latent.npy", z)
    return model, z


def sample_rows(z_by_time, counts, n, device):
    xs = []
    idxs = []
    for arr in z_by_time:
        idx = np.random.randint(0, arr.shape[0], size=n)
        xs.append(torch.as_tensor(arr[idx], dtype=torch.float32, device=device))
        idxs.append(idx)
    return xs, idxs


def integrate(model, x0, t0, t1, steps):
    dt = float(t1 - t0) / int(steps)
    x = x0
    log_growth = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
    energy = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    growth_l2 = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    for i in range(int(steps)):
        t = float(t0) + (i + 0.5) * dt
        v = model.velocity(t, x)
        g = model.growth(t, x)
        x = x + dt * v
        log_growth = log_growth + dt * g
        energy = energy + abs(dt) * 0.5 * torch.sum(v * v, dim=1)
        growth_l2 = growth_l2 + abs(dt) * (g.squeeze(1) ** 2)
    return x, log_growth.squeeze(1), energy, growth_l2


def save_tigon_checkpoint(
    path,
    model,
    opt,
    sched,
    args,
    model_dim,
    unique_times,
    counts,
    count_ratios,
    history,
    itr,
):
    state = {
        "iter": int(itr),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "args": vars(args),
        "model_dim": int(model_dim),
        "unique_times": np.asarray(unique_times, dtype=float).tolist(),
        "counts": [int(c) for c in counts],
        "count_ratios": [float(r) for r in count_ratios],
        "history": list(history),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)


def train_tigon(z, times, unique_times, args, device, outdir):
    ckpt_path = outdir / "tigon.pt"
    latest_ckpt_path = outdir / "tigon_checkpoint_latest.pt"
    model_dim = int(z.shape[1])
    model = TIGONModel(model_dim, args.tigon_hidden_dim, args.tigon_hidden_layers).to(device)
    if args.skip_tigon_train and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        return model

    z_by_time = [z[times == t] for t in unique_times]
    counts = [arr.shape[0] for arr in z_by_time]
    count_ratios = [counts[i + 1] / counts[i] for i in range(len(counts) - 1)]
    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=args.sinkhorn_blur, backend="tensorized")
    opt = torch.optim.Adam(model.parameters(), lr=args.tigon_lr, weight_decay=0.01)
    # Match the upstream schedule without forcing negative milestones to step 1.
    # The latter made smoke runs with <= 400 iterations decay the learning rate
    # immediately, even though upstream TIGON would not decay during such runs.
    lr_milestones = [
        milestone
        for milestone in (args.tigon_iters - 400, args.tigon_iters - 200)
        if milestone > 0
    ]
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt,
        milestones=lr_milestones,
        gamma=0.5,
    )
    history = []
    start_iter = 0
    if args.resume_tigon:
        if not latest_ckpt_path.exists():
            raise FileNotFoundError(
                f"Cannot resume TIGON because {latest_ckpt_path} does not exist."
            )
        state = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        if int(state.get("model_dim", -1)) != model_dim:
            raise ValueError(
                f"Checkpoint model_dim={state.get('model_dim')} does not match data dim={model_dim}."
            )
        model.load_state_dict(state["model_state_dict"])
        if "optimizer_state_dict" in state:
            opt.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            sched.load_state_dict(state["scheduler_state_dict"])
        history = list(state.get("history", []))
        start_iter = int(state.get("iter", 0))
        print(f"[tigon] resumed from {latest_ckpt_path} at iter={start_iter}", flush=True)

    if start_iter >= args.tigon_iters:
        print(
            f"[tigon] checkpoint already reached iter={start_iter}; requested {args.tigon_iters}",
            flush=True,
        )
    for itr in range(start_iter + 1, args.tigon_iters + 1):
        opt.zero_grad()
        total = torch.zeros((), dtype=torch.float32, device=device)
        match_total = torch.zeros_like(total)
        energy_total = torch.zeros_like(total)
        mass_total = torch.zeros_like(total)
        growth_l2_total = torch.zeros_like(total)
        for k in range(len(unique_times) - 1):
            src = z_by_time[k]
            tgt = z_by_time[k + 1]
            src_idx = np.random.randint(0, src.shape[0], size=args.num_samples)
            tgt_idx = np.random.randint(0, tgt.shape[0], size=args.num_samples)
            x0 = torch.as_tensor(src[src_idx], dtype=torch.float32, device=device)
            y = torch.as_tensor(tgt[tgt_idx], dtype=torch.float32, device=device)
            pred, log_growth, energy, growth_l2 = integrate(
                model, x0, float(unique_times[k]), float(unique_times[k + 1]), args.ode_steps
            )
            w = torch.softmax(log_growth.clamp(-12, 12), dim=0)
            b = torch.full((y.shape[0],), 1.0 / y.shape[0], dtype=y.dtype, device=device)
            match = loss_fn(w, pred, b, y)
            mass_factor = torch.mean(torch.exp(log_growth.clamp(-12, 12)))
            mass_loss = (torch.log(mass_factor + 1e-8) - math.log(count_ratios[k])) ** 2
            interval_loss = (
                match
                + args.lambda_energy * torch.mean(energy)
                + args.lambda_mass * mass_loss
                + args.lambda_growth_l2 * torch.mean(growth_l2)
            )
            total = total + interval_loss
            match_total = match_total + match.detach()
            energy_total = energy_total + torch.mean(energy).detach()
            mass_total = mass_total + mass_loss.detach()
            growth_l2_total = growth_l2_total + torch.mean(growth_l2).detach()
        total.backward()
        opt.step()
        sched.step()
        row = {
            "iter": itr,
            "loss": float(total.detach().cpu()),
            "sinkhorn": float(match_total.detach().cpu()),
            "energy": float(energy_total.detach().cpu()),
            "mass_loss": float(mass_total.detach().cpu()),
            "growth_l2": float(growth_l2_total.detach().cpu()),
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)
        if itr == 1 or itr % max(1, min(50, args.tigon_iters // 10)) == 0:
            print(
                "[tigon] iter={iter} loss={loss:.6f} sinkhorn={sinkhorn:.6f} "
                "energy={energy:.6f} mass={mass_loss:.6f}".format(**row),
                flush=True,
            )
        if args.checkpoint_every > 0 and (
            itr % args.checkpoint_every == 0 or itr == args.tigon_iters
        ):
            save_tigon_checkpoint(
                latest_ckpt_path,
                model,
                opt,
                sched,
                args,
                model_dim,
                unique_times,
                counts,
                count_ratios,
                history,
                itr,
            )
            iter_ckpt_path = outdir / f"tigon_checkpoint_iter{itr:06d}.pt"
            save_tigon_checkpoint(
                iter_ckpt_path,
                model,
                opt,
                sched,
                args,
                model_dim,
                unique_times,
                counts,
                count_ratios,
                history,
                itr,
            )
            print(f"[tigon] checkpoint saved at iter={itr}", flush=True)
    pd.DataFrame(history).to_csv(outdir / "tigon_training_history.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "model_dim": model_dim,
            "unique_times": unique_times.tolist(),
            "counts": counts,
            "count_ratios": count_ratios,
        },
        ckpt_path,
    )
    return model


def _xavier_initialize(module):
    """Match public TIGON ``initialize_weights`` exactly.

    The upstream helper replaces matrix-valued weights with Xavier-uniform
    draws but deliberately leaves the biases produced by ``nn.Linear``'s
    constructor untouched.  Zeroing those biases changes the seeded public
    EMT training trace, even though the network layout is otherwise the same.
    """
    if hasattr(module, "weight") and module.weight.dim() > 1:
        nn.init.xavier_uniform_(module.weight.data)


def _sample_gaussian_mixture(centers, n, covariance, device):
    """Reproduce public TIGON ``Sampling`` center/noise semantics.

    Upstream TIGON selects mixture centers with Python's ``random`` module
    (without replacement when possible) and draws Gaussian noise with a
    Torch ``MultivariateNormal``.  Keeping both RNG streams aligned makes a
    seeded run directly auditable against the public implementation.
    """
    if covariance <= 0:
        raise ValueError(f"Gaussian sampling covariance must be positive, got {covariance}")
    population = range(int(centers.shape[0]))
    if centers.shape[0] < n:
        selected = random.choices(population, k=int(n))
    else:
        selected = random.sample(population, k=int(n))
    indices = torch.as_tensor(selected, dtype=torch.long, device=device)
    dimension = int(centers.shape[1])
    gaussian = torch.distributions.MultivariateNormal(
        torch.zeros(dimension, dtype=centers.dtype, device=device),
        covariance
        * torch.eye(dimension, dtype=centers.dtype, device=device),
    )
    noise = gaussian.rsample(torch.Size([int(n)]))
    return centers[indices] + noise


def _kde_log_density(points, centers, covariance, chunk_size):
    """Log density of the equally weighted isotropic GMM used by TIGON."""
    if covariance <= 0:
        raise ValueError(f"KDE covariance must be positive, got {covariance}")
    if chunk_size <= 0:
        raise ValueError(f"KDE chunk size must be positive, got {chunk_size}")
    dimension = points.shape[1]
    normalizer = -0.5 * dimension * math.log(2.0 * math.pi * covariance)
    log_sum = None
    for start in range(0, centers.shape[0], chunk_size):
        center_chunk = centers[start : start + chunk_size]
        squared_distance = torch.sum(
            (points[:, None, :] - center_chunk[None, :, :]) ** 2,
            dim=2,
        )
        chunk_log_sum = torch.logsumexp(
            normalizer - 0.5 * squared_distance / covariance,
            dim=1,
        )
        log_sum = chunk_log_sum if log_sum is None else torch.logaddexp(log_sum, chunk_log_sum)
    return log_sum - math.log(centers.shape[0])


def _density_reconstruction_loss(predicted_log_density, target_log_density, mode):
    if mode == "log":
        return torch.mean((predicted_log_density - target_log_density) ** 2)
    if mode == "official":
        # Upstream TIGON evaluates MSE directly on raw densities. Numerical
        # failures are caught by the finite-loss/gradient guards in the
        # training loop instead of silently rescaling or clipping this loss.
        predicted = torch.exp(predicted_log_density)
        target = torch.exp(target_log_density)
        return torch.mean((predicted - target) ** 2)
    if mode == "relative":
        # In raw 40D PCA space, normalized Gaussian densities underflow even
        # though their ratios are informative. A shared offset only rescales
        # the density-MSE term; it does not alter coordinates or trajectories.
        offset = target_log_density.detach().max()
        predicted = torch.exp(
            torch.clamp(predicted_log_density - offset, min=-60.0, max=20.0)
        )
        target = torch.exp(
            torch.clamp(target_log_density - offset, min=-60.0, max=20.0)
        )
        return torch.mean((predicted - target) ** 2)
    raise ValueError(f"Unknown density stabilization mode: {mode}")


class _TIGONDensityDynamics(nn.Module):
    """TIGON continuity-equation ODE: dx/dt=v, dlog(rho)/dt=g-div(v)."""

    def __init__(self, model, divergence_estimator, finite_difference_epsilon=1e-2):
        super().__init__()
        self.model = model
        self.divergence_estimator = divergence_estimator
        self.finite_difference_epsilon = float(finite_difference_epsilon)
        self._noise = None

    def _divergence(self, t, velocity, state):
        if self.divergence_estimator == "exact":
            divergence = torch.zeros(state.shape[0], dtype=state.dtype, device=state.device)
            for dimension in range(state.shape[1]):
                gradient = torch.autograd.grad(
                    velocity[:, dimension].sum(),
                    state,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                divergence = divergence + gradient[:, dimension]
            return divergence
        if self.divergence_estimator == "hutchinson":
            if self._noise is None or self._noise.shape != state.shape:
                self._noise = (
                    torch.randint(0, 2, state.shape, device=state.device, dtype=torch.int64)
                    .to(state.dtype)
                    .mul_(2.0)
                    .sub_(1.0)
                )
            vector_jacobian = torch.autograd.grad(
                torch.sum(velocity * self._noise),
                state,
                create_graph=True,
                retain_graph=True,
            )[0]
            return torch.sum(vector_jacobian * self._noise, dim=1)
        if self.divergence_estimator == "finite-difference":
            if self.finite_difference_epsilon <= 0:
                raise ValueError("--divergence-fd-epsilon must be positive")
            if self._noise is None or self._noise.shape != state.shape:
                self._noise = (
                    torch.randint(0, 2, state.shape, device=state.device, dtype=torch.int64)
                    .to(state.dtype)
                    .mul_(2.0)
                    .sub_(1.0)
                )
            delta = self.finite_difference_epsilon
            velocity_plus = self.model.velocity(t, state + delta * self._noise)
            velocity_minus = self.model.velocity(t, state - delta * self._noise)
            directional_derivative = (velocity_plus - velocity_minus) / (2.0 * delta)
            return torch.sum(directional_derivative * self._noise, dim=1)
        raise ValueError(f"Unknown divergence estimator: {self.divergence_estimator}")

    def forward(self, t, states):
        state, integrated_growth, log_density_change = states
        del integrated_growth, log_density_change
        with torch.enable_grad():
            state.requires_grad_(True)
            velocity = self.model.velocity(t, state)
            growth = self.model.growth(t, state)
            divergence = self._divergence(t, velocity, state).unsqueeze(1)
        return velocity, growth, growth - divergence


class _TIGONActionDynamics(nn.Module):
    """WFR action accumulated along a moving trajectory.

    This is retained for compatibility with older local experiments.  The
    audited official-style runners use ``_TIGONOfficialActionDynamics`` below,
    because upstream TIGON holds the sampled spatial coordinates fixed while
    integrating its action regularizer over time.
    """

    def __init__(self, model, growth_action_weight):
        super().__init__()
        self.model = model
        self.growth_action_weight = growth_action_weight

    def forward(self, t, states):
        state, integrated_growth, action = states
        del action
        velocity = self.model.velocity(t, state)
        growth = self.model.growth(t, state)
        action_rate = (
            torch.sum(velocity * velocity, dim=1, keepdim=True)
            + self.growth_action_weight * growth * growth
        ) * torch.exp(torch.clamp(integrated_growth, min=-20.0, max=20.0))
        return velocity, growth, action_rate


class _TIGONFixedStateGrowthDynamics(nn.Module):
    """Integrate growth in time while keeping the sampled state fixed."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, states):
        state, growth_placeholder, integrated_growth = states
        del integrated_growth
        return (
            torch.zeros_like(state),
            torch.zeros_like(growth_placeholder),
            self.model.growth(t, state),
        )


class _TIGONPaperFixedSampleActionDynamics(nn.Module):
    """Fixed-sample interpretation of the TIGON action formula.

    This evaluates ``exp(integral_0^t g(s, x) ds)`` at each sampled coordinate.
    It is retained as a paper-formula sensitivity mode, but it is not a literal
    reproduction of the public ``trans_loss`` implementation, whose nested
    growth integral is initialized at the zero spatial state.
    """

    def __init__(
        self,
        model,
        growth_action_weight,
        start_time,
        midpoint_step_size,
    ):
        super().__init__()
        self.model = model
        self.growth_action_weight = float(growth_action_weight)
        self.start_time = float(start_time)
        self.midpoint_step_size = float(midpoint_step_size)
        if self.midpoint_step_size <= 0:
            raise ValueError("Official action midpoint step size must be positive")
        self.growth_dynamics = _TIGONFixedStateGrowthDynamics(model)

    def forward(self, t, states):
        state, growth_placeholder, action = states
        del action
        velocity = self.model.velocity(t, state)
        growth = self.model.growth(t, state)
        end_time = float(t.detach().cpu()) if torch.is_tensor(t) else float(t)
        if math.isclose(end_time, self.start_time, rel_tol=0.0, abs_tol=1e-12):
            integrated_growth = torch.zeros_like(growth)
        else:
            integration_times = torch.tensor(
                [self.start_time, end_time],
                dtype=state.dtype,
                device=state.device,
            )
            initial_growth = torch.zeros_like(growth_placeholder)
            integrated_growth = odeint(
                self.growth_dynamics,
                (state, initial_growth, torch.zeros_like(growth)),
                integration_times,
                method="midpoint",
                rtol=1e-5,
                atol=1e-5,
                options={"step_size": self.midpoint_step_size},
            )[2][-1]
        action_rate = (
            torch.sum(velocity * velocity, dim=1, keepdim=True)
            + self.growth_action_weight * growth * growth
        ) * torch.exp(integrated_growth)
        return (
            torch.zeros_like(state),
            torch.zeros_like(growth_placeholder),
            action_rate,
        )


class _TIGONUpstreamPublicActionDynamics(nn.Module):
    """Literal vectorized reproduction of upstream TIGON ``trans_loss``.

    The public implementation evaluates ``v(t, x)`` and ``g(t, x)`` at fixed
    sampled coordinates, but initializes its nested ``ggrowth`` ODE with a
    zero spatial tensor.  Consequently the exponential action weight is
    ``exp(integral_0^t g(s, 0) ds)``.  Keeping this seemingly unusual detail is
    necessary for a reviewer-auditable public-code reproduction.
    """

    def __init__(
        self,
        model,
        growth_action_weight,
        start_time,
        midpoint_step_size,
    ):
        super().__init__()
        self.model = model
        self.growth_action_weight = float(growth_action_weight)
        self.start_time = float(start_time)
        self.midpoint_step_size = float(midpoint_step_size)
        if self.midpoint_step_size <= 0:
            raise ValueError("Upstream action midpoint step size must be positive")
        self.growth_dynamics = _TIGONFixedStateGrowthDynamics(model)

    def forward(self, t, states):
        state, growth_placeholder, action = states
        del action
        velocity = self.model.velocity(t, state)
        growth = self.model.growth(t, state)
        end_time = float(t.detach().cpu()) if torch.is_tensor(t) else float(t)
        if math.isclose(end_time, self.start_time, rel_tol=0.0, abs_tol=1e-12):
            integrated_growth = torch.zeros_like(growth)
        else:
            integration_times = torch.tensor(
                [self.start_time, end_time],
                dtype=state.dtype,
                device=state.device,
            )
            zero_state = torch.zeros_like(state)
            zero_growth = torch.zeros_like(growth_placeholder)
            integrated_growth = odeint(
                self.growth_dynamics,
                (zero_state, zero_growth, torch.zeros_like(growth)),
                integration_times,
                method="midpoint",
                rtol=1e-5,
                atol=1e-5,
                options={"step_size": self.midpoint_step_size},
            )[2][-1]
        action_rate = (
            torch.sum(velocity * velocity, dim=1, keepdim=True)
            + self.growth_action_weight * growth * growth
        ) * torch.exp(integrated_growth)
        return (
            torch.zeros_like(state),
            torch.zeros_like(growth_placeholder),
            action_rate,
        )


def _ode_solve(
    dynamics,
    initial_states,
    start_time,
    end_time,
    args,
    device,
    *,
    solver=None,
    fixed_steps=None,
):
    times = torch.tensor(
        [float(start_time), float(end_time)],
        dtype=torch.float32,
        device=device,
    )
    method = (solver or args.ode_solver).lower()
    density_backend = getattr(args, "density_ode_backend", "torchdiffeq")
    if solver is None and density_backend == "torchdiffeqpack":
        if method != "dopri5":
            raise ValueError(
                "The upstream TorchDiffEqPack path is audited only for Dopri5"
            )
        upstream_odesolve = _load_upstream_odesolve()
        options = {
            "method": "Dopri5",
            "h": None,
            "t0": float(start_time),
            "t1": float(end_time),
            "rtol": args.ode_rtol,
            "atol": args.ode_atol,
            "print_neval": False,
            "neval_max": 1_000_000,
            "safety": None,
        }
        final_states = upstream_odesolve(
            dynamics,
            y0=initial_states,
            options=options,
        )
        if torch.is_tensor(final_states):
            final_states = (final_states,)
        return tuple(
            torch.stack((initial, final), dim=0)
            for initial, final in zip(initial_states, final_states)
        )
    kwargs = {
        "method": method,
        "rtol": args.ode_rtol,
        "atol": args.ode_atol,
    }
    if method in {"euler", "midpoint", "rk4"}:
        steps = int(args.ode_steps if fixed_steps is None else fixed_steps)
        if steps <= 0:
            raise ValueError(f"--ode-steps must be positive for fixed solver {method}")
        duration = abs(float(end_time) - float(start_time))
        kwargs["options"] = {"step_size": duration / steps}
    return odeint(dynamics, initial_states, times, **kwargs)


def _save_tigon_density_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    args,
    model_dim,
    unique_times,
    counts,
    history,
    iteration,
    sigma,
):
    state = {
        "iter": int(iteration),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
        "model_dim": int(model_dim),
        "unique_times": np.asarray(unique_times, dtype=float).tolist(),
        "counts": [int(count) for count in counts],
        "history": list(history),
        "density_sigma": float(sigma),
        "training_objective": "tigon-density",
        "torch_rng_state": torch.get_rng_state(),
        "python_random_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train_tigon_density(z, times, unique_times, args, device, outdir):
    """TIGON short/long density reconstruction with configurable ODE integration."""
    ckpt_path = outdir / "tigon.pt"
    latest_ckpt_path = outdir / "tigon_checkpoint_latest.pt"
    model_dim = int(z.shape[1])
    model = TIGONModel(
        model_dim,
        args.tigon_hidden_dim,
        args.tigon_hidden_layers,
    ).to(device)
    model.apply(_xavier_initialize)
    if args.skip_tigon_train and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        return model

    z_by_time = [
        torch.as_tensor(z[times == time_value], dtype=torch.float32, device=device)
        for time_value in unique_times
    ]
    counts = [values.shape[0] for values in z_by_time]
    count_ratios = [count / counts[0] for count in counts]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.tigon_lr, weight_decay=0.01)
    # Keep the upstream end-of-training schedule, while leaving short diagnostic
    # runs at their requested learning rate instead of decaying at iteration 1.
    lr_milestones = [
        milestone
        for milestone in (args.tigon_iters - 400, args.tigon_iters - 200)
        if milestone > 0
    ]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=lr_milestones,
        gamma=0.5,
    )
    history = []
    start_iteration = 0
    requested_sigma_initial = float(args.density_sigma_initial)
    official_sigma_schedule = bool(
        getattr(args, "official_density_sigma_schedule", False)
    )
    official_sigma_stop_threshold = float(
        getattr(args, "density_sigma_official_stop_threshold", 0.02)
    )
    if official_sigma_schedule:
        # The upstream public schedule deliberately has no covariance floor:
        # test the current sigma against 0.02, then halve it without clamping.
        # The protected runner therefore passes None for both minimum fields.
        requested_sigma_min = None
        density_sigma_hard_clip = None
        effective_sigma_min = None
        if not math.isclose(
            requested_sigma_initial, 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "Upstream TIGON sigma scheduling requires initial sigma=1"
            )
        if official_sigma_stop_threshold <= 0:
            raise ValueError(
                "density_sigma_official_stop_threshold must be positive"
            )
        sigma = requested_sigma_initial
    else:
        requested_sigma_min = float(args.density_sigma_min)
        density_sigma_hard_clip = float(
            getattr(
                args,
                "density_sigma_hard_clip",
                TIGON_DENSITY_SIGMA_HARD_CLIP,
            )
        )
        if density_sigma_hard_clip <= 0:
            raise ValueError("density_sigma_hard_clip must be positive")
        effective_sigma_min = max(requested_sigma_min, density_sigma_hard_clip)
        if requested_sigma_min < effective_sigma_min:
            print(
                "[tigon-density] clipped requested density sigma minimum "
                f"{requested_sigma_min:g} -> {effective_sigma_min:g}",
                flush=True,
            )
        sigma = max(requested_sigma_initial, effective_sigma_min)
        if requested_sigma_initial < sigma:
            print(
                "[tigon-density] clipped requested initial density sigma "
                f"{requested_sigma_initial:g} -> {sigma:g}",
                flush=True,
            )
    # Mutate the runtime namespace so checkpoints and downstream audit outputs
    # record the effective policy rather than a lower, ignored request.
    args.density_sigma_initial = sigma
    args.density_sigma_min = effective_sigma_min
    args.density_sigma_hard_clip = density_sigma_hard_clip
    args.official_density_sigma_schedule = official_sigma_schedule
    args.density_sigma_official_stop_threshold = (
        official_sigma_stop_threshold if official_sigma_schedule else None
    )
    if args.density_sigma_anneal_every > 0 and not (
        0.0 < args.density_sigma_anneal_factor < 1.0
    ):
        raise ValueError("--density-sigma-anneal-factor must be in (0, 1)")
    if args.density_sigma_anneal_threshold < 0:
        raise ValueError("--density-sigma-anneal-threshold must be non-negative")
    if args.density_sigma_anneal_stop_before_end < 0:
        raise ValueError("--density-sigma-anneal-stop-before-end must be non-negative")
    if args.max_grad_norm < 0:
        raise ValueError("--max-grad-norm must be non-negative")

    if args.resume_tigon and args.tigon_init_checkpoint:
        raise ValueError(
            "--resume-tigon and --tigon-init-checkpoint are mutually exclusive"
        )
    if args.tigon_init_checkpoint:
        init_checkpoint_path = Path(args.tigon_init_checkpoint)
        if not init_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"TIGON initialization checkpoint not found: {init_checkpoint_path}"
            )
        state = torch.load(
            init_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if state.get("training_objective") != "tigon-density":
            raise ValueError(
                "Initialization checkpoint is not a TIGON density-objective checkpoint"
            )
        saved_model_dim = int(state.get("model_dim", model_dim))
        if saved_model_dim != model_dim:
            raise ValueError(
                "Initialization checkpoint model dimension does not match current data: "
                f"{saved_model_dim} != {model_dim}"
            )
        model.load_state_dict(state["model_state_dict"])
        print(
            "[tigon-density] initialized model from "
            f"{init_checkpoint_path} at saved_iter={int(state.get('iter', -1))}; "
            f"fresh optimizer, lr={args.tigon_lr:g}, sigma={sigma:g}",
            flush=True,
        )

    if args.resume_tigon:
        if not latest_ckpt_path.exists():
            raise FileNotFoundError(
                f"Cannot resume TIGON because {latest_ckpt_path} does not exist."
            )
        state = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        if state.get("training_objective") != "tigon-density":
            raise ValueError("Checkpoint is not a TIGON density-objective checkpoint")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        history = list(state.get("history", []))
        start_iteration = int(state.get("iter", 0))
        saved_sigma = float(state.get("density_sigma", sigma))
        sigma = (
            saved_sigma
            if official_sigma_schedule
            else max(saved_sigma, effective_sigma_min)
        )
        if saved_sigma < sigma:
            print(
                "[tigon-density] clipped resumed checkpoint density sigma "
                f"{saved_sigma:g} -> {sigma:g}",
                flush=True,
            )
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"])
        if "python_random_state" in state:
            random.setstate(state["python_random_state"])
        if "numpy_rng_state" in state:
            np.random.set_state(state["numpy_rng_state"])
        print(
            f"[tigon-density] resumed at iter={start_iteration}, sigma={sigma:.6g}",
            flush=True,
        )

    for iteration in range(start_iteration + 1, args.tigon_iters + 1):
        # Keep the covariance used to compute this iteration separate from the
        # covariance produced by the end-of-iteration annealing update.  Older
        # logs recorded only the updated value, which made the first loss after
        # every annealing boundary look as though it had used the new sigma.
        sigma_used = float(sigma)
        optimizer.zero_grad()
        long_term_total = torch.zeros((), dtype=torch.float32, device=device)
        short_term_total = torch.zeros((), dtype=torch.float32, device=device)

        for interval in range(len(unique_times) - 1):
            target = _sample_gaussian_mixture(
                z_by_time[interval + 1],
                args.num_samples,
                args.density_sample_sigma,
                device,
            )
            target.requires_grad_(True)
            target_log_density = _kde_log_density(
                target,
                z_by_time[interval + 1],
                sigma,
                args.density_kde_chunk_size,
            ) + math.log(count_ratios[interval + 1])
            zeros = torch.zeros(
                (target.shape[0], 1), dtype=target.dtype, device=device
            )

            long_dynamics = _TIGONDensityDynamics(
                model,
                args.divergence_estimator,
                args.divergence_fd_epsilon,
            )
            long_solution = _ode_solve(
                long_dynamics,
                (target, zeros, zeros),
                unique_times[interval + 1],
                unique_times[0],
                args,
                device,
            )
            long_start = long_solution[0][-1]
            long_log_change = long_solution[2][-1].squeeze(1)
            long_base_log_density = _kde_log_density(
                long_start,
                z_by_time[0],
                sigma,
                args.density_kde_chunk_size,
            )
            if args.density_stabilization == "official":
                # Match upstream TIGON's aa[aa < 1e-16] = 1e-16 guard before
                # taking log density and applying the continuity correction.
                long_base_log_density = torch.clamp(
                    long_base_log_density,
                    min=math.log(1e-16),
                )
            long_predicted_log_density = long_base_log_density - long_log_change
            long_term_total = long_term_total + _density_reconstruction_loss(
                long_predicted_log_density,
                target_log_density,
                args.density_stabilization,
            )

            short_dynamics = _TIGONDensityDynamics(
                model,
                args.divergence_estimator,
                args.divergence_fd_epsilon,
            )
            short_solution = _ode_solve(
                short_dynamics,
                (target, zeros, zeros),
                unique_times[interval + 1],
                unique_times[interval],
                args,
                device,
            )
            short_start = short_solution[0][-1]
            short_log_change = short_solution[2][-1].squeeze(1)
            short_base_log_density = _kde_log_density(
                short_start,
                z_by_time[interval],
                sigma,
                args.density_kde_chunk_size,
            ) + math.log(count_ratios[interval])
            if args.density_stabilization == "official":
                short_base_log_density = torch.clamp(
                    short_base_log_density,
                    min=math.log(1e-16),
                )
            short_predicted_log_density = short_base_log_density - short_log_change
            short_term_total = short_term_total + _density_reconstruction_loss(
                short_predicted_log_density,
                target_log_density,
                args.density_stabilization,
            )

        action_initial = _sample_gaussian_mixture(
            z_by_time[0],
            args.num_samples,
            args.density_sample_sigma,
            device,
        )
        action_zeros = torch.zeros(
            (action_initial.shape[0], 1),
            dtype=action_initial.dtype,
            device=device,
        )
        action_solver = (
            args.ode_solver
            if args.action_ode_solver == "same"
            else args.action_ode_solver
        )
        action_steps = (
            args.ode_steps
            if args.action_ode_steps <= 0
            else args.action_ode_steps
        )
        action_duration = abs(float(unique_times[-1] - unique_times[0]))
        action_step_size = action_duration / action_steps
        action_state_mode = getattr(args, "action_state_mode", "trajectory")
        if action_state_mode in {
            "upstream_public_exact",
            "paper_fixed_sample",
            "official_stationary",
        }:
            if action_solver.lower() != "midpoint":
                raise ValueError(
                    "TIGON public/fixed-sample action requires midpoint integration"
                )
            action_class = (
                _TIGONUpstreamPublicActionDynamics
                if action_state_mode == "upstream_public_exact"
                else _TIGONPaperFixedSampleActionDynamics
            )
            action_dynamics = action_class(
                model,
                args.growth_action_weight,
                start_time=float(unique_times[0]),
                midpoint_step_size=action_step_size,
            )
        elif action_state_mode == "trajectory":
            action_dynamics = _TIGONActionDynamics(
                model, args.growth_action_weight
            )
        else:
            raise ValueError(
                f"Unknown TIGON action state mode: {action_state_mode}"
            )
        action_solution = _ode_solve(
            action_dynamics,
            (action_initial, action_zeros, action_zeros),
            unique_times[0],
            unique_times[-1],
            args,
            device,
            solver=action_solver,
            fixed_steps=action_steps,
        )
        duration = float(unique_times[-1] - unique_times[0])
        transport_cost = duration * action_solution[2][-1].mean()
        reconstruction = long_term_total + short_term_total
        loss = transport_cost + args.density_weight * reconstruction
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite TIGON loss at iteration {iteration}, sigma={sigma:.6g}"
            )
        loss.backward()
        gradient_norm = float("nan")
        if args.max_grad_norm > 0:
            gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.max_grad_norm,
                error_if_nonfinite=True,
            )
            gradient_norm = float(gradient_norm_tensor.detach().cpu())
        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise FloatingPointError(
                f"Non-finite TIGON parameter at iteration {iteration}, sigma={sigma:.6g}"
            )
        scheduler.step()

        sigma_annealed = False
        if (
            iteration > 1
            and args.density_sigma_anneal_every > 0
            and iteration % args.density_sigma_anneal_every == 0
            and iteration <= args.tigon_iters - args.density_sigma_anneal_stop_before_end
            and sigma
            > (
                official_sigma_stop_threshold
                if official_sigma_schedule
                else effective_sigma_min
            )
            and float(long_term_total.detach().cpu()) / (len(unique_times) - 1)
            <= args.density_sigma_anneal_threshold
        ):
            sigma = sigma * args.density_sigma_anneal_factor
            if not official_sigma_schedule:
                sigma = max(effective_sigma_min, sigma)
            sigma_annealed = True

        row = {
            "iter": iteration,
            "loss": float(loss.detach().cpu()),
            "transport_cost": float(transport_cost.detach().cpu()),
            "long_term_reconstruction": float(long_term_total.detach().cpu()),
            "short_term_reconstruction": float(short_term_total.detach().cpu()),
            "density_reconstruction": float(reconstruction.detach().cpu()),
            "density_sigma_used": sigma_used,
            "density_sigma_next": float(sigma),
            # Backward-compatible alias: this is the covariance saved in a
            # checkpoint and used by the following iteration.
            "density_sigma": float(sigma),
            "density_sigma_annealed": bool(sigma_annealed),
            "gradient_norm_pre_clip": gradient_norm,
            "lr": float(optimizer.param_groups[0]["lr"]),
            # Compatibility columns used by the existing plot helper.
            "sinkhorn": 0.0,
            "energy": float(transport_cost.detach().cpu()),
            "mass_loss": float(long_term_total.detach().cpu()),
            "growth_l2": float(short_term_total.detach().cpu()),
        }
        history.append(row)
        if iteration == 1 or iteration % max(1, min(50, args.tigon_iters // 10)) == 0:
            print(
                "[tigon-density] iter={iter} loss={loss:.6g} transport={transport_cost:.6g} "
                "long={long_term_reconstruction:.6g} short={short_term_reconstruction:.6g} "
                "sigma_used={density_sigma_used:.6g} "
                "sigma_next={density_sigma_next:.6g}".format(**row),
                flush=True,
            )

        # Besides the regular cadence, preserve the parameter state at every
        # bandwidth transition.  The loss at an annealing iteration was
        # computed with ``sigma_used`` while the following iteration uses the
        # smaller saved sigma; retaining this boundary prevents checkpoint
        # selection from silently skipping the pre-transition model.
        if args.checkpoint_every > 0 and (
            iteration % args.checkpoint_every == 0
            or iteration == args.tigon_iters
            or sigma_annealed
        ):
            _save_tigon_density_checkpoint(
                latest_ckpt_path,
                model,
                optimizer,
                scheduler,
                args,
                model_dim,
                unique_times,
                counts,
                history,
                iteration,
                sigma,
            )
            iteration_path = outdir / f"tigon_checkpoint_iter{iteration:06d}.pt"
            _save_tigon_density_checkpoint(
                iteration_path,
                model,
                optimizer,
                scheduler,
                args,
                model_dim,
                unique_times,
                counts,
                history,
                iteration,
                sigma,
            )

    pd.DataFrame(history).to_csv(outdir / "tigon_training_history.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "model_dim": model_dim,
            "unique_times": unique_times.tolist(),
            "counts": counts,
            "count_ratios": [counts[i + 1] / counts[i] for i in range(len(counts) - 1)],
            "training_objective": "tigon-density",
            "ode_solver": args.ode_solver,
            "ode_rtol": args.ode_rtol,
            "ode_atol": args.ode_atol,
            "divergence_estimator": args.divergence_estimator,
            "divergence_fd_epsilon": args.divergence_fd_epsilon,
        },
        ckpt_path,
    )
    return model


def eval_growth(model, z, obs, times, unique_times, args, device, outdir):
    growth = np.zeros(z.shape[0], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for t in unique_times:
            idx = np.flatnonzero(times == t)
            vals = []
            for start in range(0, len(idx), 2048):
                batch_idx = idx[start : start + 2048]
                xb = torch.as_tensor(z[batch_idx], dtype=torch.float32, device=device)
                g = model.growth(float(t), xb).squeeze(1)
                vals.append(g.detach().cpu().numpy())
            growth[idx] = np.concatenate(vals)

    df = pd.DataFrame(
        {
            "cell": obs.index.to_numpy(),
            "time": times,
            "growth_log_rate": growth,
            "growth_factor_per_unit_time": np.exp(np.clip(growth, -20, 20)),
        }
    )
    for key in [args.stage_key, args.celltype_key, "phase", "proliferation"]:
        if key in obs:
            df[key] = obs[key].to_numpy()
    df.to_csv(outdir / "cell_growth_scores.csv", index=False)

    summary_cols = ["time"]
    if args.stage_key in df:
        summary_cols.append(args.stage_key)
    summary = (
        df.groupby(summary_cols, dropna=False)["growth_log_rate"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    summary.to_csv(outdir / "growth_summary_by_time.csv", index=False)
    if args.celltype_key in df and args.stage_key in df:
        ct = (
            df.groupby([args.stage_key, args.celltype_key], dropna=False)["growth_log_rate"]
            .agg(["count", "mean", "std", "median"])
            .reset_index()
            .sort_values(["mean"], ascending=False)
        )
        ct.to_csv(outdir / "growth_summary_by_celltype.csv", index=False)
    return growth, df


def eval_intervals(model, z, times, unique_times, args, device, outdir):
    z_by_time = [z[times == t] for t in unique_times]
    counts = [arr.shape[0] for arr in z_by_time]
    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=args.sinkhorn_blur, backend="tensorized")
    rows = []
    with torch.no_grad():
        for k in range(len(unique_times) - 1):
            src = z_by_time[k]
            tgt = z_by_time[k + 1]
            n = min(args.eval_samples, src.shape[0], tgt.shape[0])
            src_idx = np.linspace(0, src.shape[0] - 1, n).round().astype(int)
            tgt_idx = np.linspace(0, tgt.shape[0] - 1, n).round().astype(int)
            x0 = torch.as_tensor(src[src_idx], dtype=torch.float32, device=device)
            y = torch.as_tensor(tgt[tgt_idx], dtype=torch.float32, device=device)
            pred, log_growth, energy, growth_l2 = integrate(
                model, x0, float(unique_times[k]), float(unique_times[k + 1]), args.ode_steps
            )
            w = torch.softmax(log_growth.clamp(-12, 12), dim=0)
            b = torch.full((n,), 1.0 / n, dtype=y.dtype, device=device)
            sink = loss_fn(w, pred, b, y)
            mass_factor = torch.mean(torch.exp(log_growth.clamp(-12, 12)))
            rows.append(
                {
                    "from_time": float(unique_times[k]),
                    "to_time": float(unique_times[k + 1]),
                    "n_eval": int(n),
                    "observed_count_ratio": counts[k + 1] / counts[k],
                    "observed_log_count_ratio": math.log(counts[k + 1] / counts[k]),
                    "mean_integrated_log_growth": float(log_growth.detach().mean().cpu()),
                    "median_integrated_log_growth": float(log_growth.detach().median().cpu()),
                    "mean_mass_factor": float(mass_factor.detach().cpu()),
                    "sinkhorn_model_space": float(sink.detach().cpu()),
                    "mean_energy_model_space": float(energy.detach().mean().cpu()),
                    "sinkhorn_ae_latent": float(sink.detach().cpu()),
                    "mean_energy_ae_latent": float(energy.detach().mean().cpu()),
                    "mean_growth_l2": float(growth_l2.detach().mean().cpu()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "interval_eval.csv", index=False)
    return out


def zscore(v):
    v = np.asarray(v, dtype=np.float64)
    sd = np.nanstd(v)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(v)
    return (v - np.nanmean(v)) / sd


def make_covariates(obs, args):
    cov = [pd.Series(1.0, index=obs.index, name="intercept")]
    cov.append(pd.get_dummies(obs[args.time_key].astype(str), prefix="time", drop_first=True))
    if args.celltype_key in obs:
        cov.append(pd.get_dummies(obs[args.celltype_key].astype(str), prefix="cell", drop_first=True))
    for key in ["log_counts", "n_genes", "S_score", "G2M_score"]:
        if key in obs:
            cov.append(pd.Series(zscore(obs[key].to_numpy()), index=obs.index, name=key))
    for key in ["proliferation", "phase"]:
        if key in obs:
            cov.append(pd.get_dummies(obs[key].astype(str), prefix=key, drop_first=True))
    design = pd.concat(cov, axis=1).astype(np.float64)
    return design


def residualize_with_q(x, q):
    return x - q @ (q.T @ x)


def bh_fdr(p):
    p = np.asarray(p, dtype=np.float64)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[order] = np.minimum(adj, 1.0)
    return out


def partial_corr_growth_genes(x, genes, growth, obs, args, outdir):
    design = make_covariates(obs, args)
    q, _ = np.linalg.qr(design.to_numpy(), mode="reduced")
    growth_resid = residualize_with_q(growth.reshape(-1, 1), q).ravel()
    growth_resid = growth_resid - growth_resid.mean()
    growth_norm = np.sqrt(np.sum(growth_resid**2))
    n_cells, n_genes = x.shape
    corrs = np.full(n_genes, np.nan)
    pvals = np.full(n_genes, np.nan)
    detected = np.mean(x > 0, axis=0)
    df = max(n_cells - design.shape[1] - 2, 1)

    high = growth_resid >= np.quantile(growth_resid, 0.9)
    low = growth_resid <= np.quantile(growth_resid, 0.1)
    delta = np.full(n_genes, np.nan)
    for start in range(0, n_genes, args.gene_chunk_size):
        end = min(start + args.gene_chunk_size, n_genes)
        xb = x[:, start:end].astype(np.float64, copy=False)
        xr = residualize_with_q(xb, q)
        xr = xr - xr.mean(axis=0, keepdims=True)
        denom = growth_norm * np.sqrt(np.sum(xr**2, axis=0))
        valid = denom > 1e-12
        local = np.full(end - start, np.nan)
        local[valid] = growth_resid @ xr[:, valid] / denom[valid]
        corrs[start:end] = local
        tstat = local[valid] * np.sqrt(df / np.maximum(1.0 - local[valid] ** 2, 1e-12))
        p = np.full(end - start, np.nan)
        p[valid] = 2 * st.t.sf(np.abs(tstat), df)
        pvals[start:end] = p
        delta[start:end] = xb[high].mean(axis=0) - xb[low].mean(axis=0)

    res = pd.DataFrame(
        {
            "gene": genes,
            "partial_corr_growth": corrs,
            "p_value": pvals,
            "fdr": bh_fdr(pvals),
            "detected_frac": detected,
            "high_minus_low_growth_expr": delta,
        }
    ).sort_values("partial_corr_growth", ascending=False, na_position="last")
    res.to_csv(outdir / "growth_related_genes_partial_corr.csv", index=False)
    return res


def growth_saliency(ae, tigon, x, genes, times, unique_times, args, device, outdir):
    """TIGON-style growth gene contribution.

    For latent state z and inverse transform h(z)=decoder(z), TIGON estimates
    d growth / d gene_j by J_encoder(h(z))^T @ grad_z growth(z, t).
    This keeps the derivative on the learned AE manifold instead of taking a
    direct derivative at the original noisy expression vector.

    When AE is disabled, x is already the TIGON model space. In that case this
    reports the direct gradient of growth with respect to the input features,
    e.g. PCA dimensions for an X_latent run.
    """
    rows = []
    if ae is not None:
        ae.eval()
    tigon.eval()
    for t in unique_times:
        idx_all = np.flatnonzero(times == t)
        n = min(args.saliency_cells_per_time, len(idx_all))
        idx = np.linspace(0, len(idx_all) - 1, n).round().astype(int)
        idx = idx_all[idx]
        signed_sum = np.zeros(len(genes), dtype=np.float64)
        abs_sum = np.zeros(len(genes), dtype=np.float64)
        seen = 0
        for start in range(0, n, 64):
            batch_idx = idx[start : start + 64]
            xb = torch.as_tensor(x[batch_idx], dtype=torch.float32, device=device)
            if ae is None:
                x_for_growth = xb.detach().requires_grad_(True)
                growth = tigon.growth(float(t), x_for_growth).sum()
                contrib = torch.autograd.grad(
                    growth, x_for_growth, retain_graph=False, create_graph=False
                )[0]
            else:
                with torch.no_grad():
                    z = ae.encode(xb)

                z_for_growth = z.detach().requires_grad_(True)
                growth = tigon.growth(float(t), z_for_growth).sum()
                grad_growth_z = torch.autograd.grad(
                    growth, z_for_growth, retain_graph=False, create_graph=False
                )[0].detach()

                # Inverse transform h(z) first, then evaluate encoder Jacobian at h(z).
                x_hat = ae.decode(z.detach()).detach().requires_grad_(True)
                z_hat = ae.encode(x_hat)
                weighted_latent = torch.sum(z_hat * grad_growth_z)
                contrib = torch.autograd.grad(
                    weighted_latent, x_hat, retain_graph=False, create_graph=False
                )[0]
            contrib_np = contrib.detach().cpu().numpy()
            signed_sum += contrib_np.sum(axis=0)
            abs_sum += np.abs(contrib_np).sum(axis=0)
            seen += contrib_np.shape[0]
        signed_mean = signed_sum / max(seen, 1)
        abs_mean = abs_sum / max(seen, 1)
        for j, gene in enumerate(genes):
            rows.append(
                {
                    "time": float(t),
                    "gene": gene,
                    "signed_growth_feature_contribution": signed_mean[j],
                    "abs_growth_feature_contribution": abs_mean[j],
                    "signed_tigon_inverse_dgrowth_dexpr": signed_mean[j],
                    "abs_tigon_inverse_dgrowth_dexpr": abs_mean[j],
                    "n_cells": int(seen),
                }
            )
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("gene", as_index=False)
        .agg(
            mean_signed_growth_feature_contribution=(
                "signed_growth_feature_contribution",
                "mean",
            ),
            mean_abs_growth_feature_contribution=(
                "abs_growth_feature_contribution",
                "mean",
            ),
            mean_signed_tigon_inverse_dgrowth_dexpr=(
                "signed_tigon_inverse_dgrowth_dexpr",
                "mean",
            ),
            mean_abs_tigon_inverse_dgrowth_dexpr=(
                "abs_tigon_inverse_dgrowth_dexpr",
                "mean",
            ),
        )
        .sort_values("mean_abs_growth_feature_contribution", ascending=False)
    )
    df.to_csv(outdir / "growth_gene_saliency_by_time.csv", index=False)
    summary.to_csv(outdir / "growth_gene_saliency.csv", index=False)
    df.to_csv(outdir / "growth_gene_tigon_inverse_saliency_by_time.csv", index=False)
    summary.to_csv(outdir / "growth_gene_tigon_inverse_saliency.csv", index=False)
    return summary


def plot_training(outdir):
    hist_path = outdir / "tigon_training_history.csv"
    if not hist_path.exists():
        return
    hist = pd.read_csv(hist_path)
    density_columns = [
        "loss",
        "transport_cost",
        "long_term_reconstruction",
        "short_term_reconstruction",
    ]
    columns = (
        density_columns
        if all(column in hist.columns for column in density_columns)
        else ["loss", "sinkhorn", "energy", "mass_loss"]
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, col in zip(axes.ravel(), columns):
        ax.plot(hist["iter"], hist[col], linewidth=1.8)
        ax.set_title(col)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.25)
    fig.savefig(outdir / "tigon_training_curves.png", dpi=220)
    plt.close(fig)


def plot_growth_umap(umap, growth, obs, args, outdir):
    if umap is None:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    lo, hi = np.quantile(growth, [0.01, 0.99])
    sc = ax.scatter(
        umap[:, 0],
        umap[:, 1],
        c=np.clip(growth, lo, hi),
        s=4,
        cmap="coolwarm",
        linewidths=0,
        alpha=0.85,
    )
    ax.set_xlabel("UMAP1", fontsize=13)
    ax.set_ylabel("UMAP2", fontsize=13)
    ax.set_title("TIGON growth log-rate", fontsize=15)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("growth log-rate per unit time")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(outdir / "growth_umap.png", dpi=240)
    plt.close(fig)


def plot_gene_tables(partial, saliency, args, outdir):
    pos = partial.dropna(subset=["partial_corr_growth"]).head(args.top_n_genes).iloc[::-1]
    neg = partial.dropna(subset=["partial_corr_growth"]).tail(args.top_n_genes).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)
    axes[0].barh(pos["gene"], pos["partial_corr_growth"], color="#c4473a")
    axes[0].set_title("Positive growth partial corr")
    axes[1].barh(neg["gene"], neg["partial_corr_growth"], color="#2f6f9f")
    axes[1].set_title("Negative growth partial corr")
    for ax in axes:
        ax.axvline(0, color="black", linewidth=0.8)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_xlabel("partial correlation")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(outdir / "growth_related_genes_partial_corr.png", dpi=240)
    plt.close(fig)

    saliency_col = (
        "mean_abs_tigon_inverse_dgrowth_dexpr"
        if args.use_ae
        else "mean_abs_growth_feature_contribution"
    )
    top = saliency.head(args.top_n_genes).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 8.5), constrained_layout=True)
    ax.barh(top["gene"], top[saliency_col], color="#5c7c3b")
    ax.set_title("TIGON growth saliency")
    ax.set_xlabel(
        "mean |J_encoder(decoder(z))^T grad_z growth|"
        if args.use_ae
        else "mean |d growth / d input feature|"
    )
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(outdir / "growth_gene_saliency.png", dpi=240)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    outdir = Path(args.outdir)
    if outdir.exists() and any(outdir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{outdir} exists and is not empty. Use --overwrite.")
    outdir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"[setup] device={device}", flush=True)
    data = load_dataset(args)
    np.save(outdir / "selected_gene_indices.npy", data["gene_idx"])
    pd.Series(data["genes"]).to_csv(outdir / "selected_genes.csv", index=False, header=["gene"])
    print(
        f"[data] cells={data['x'].shape[0]} features={data['x'].shape[1]} "
        f"times={data['unique_times'].tolist()}",
        flush=True,
    )
    args.resolved_embedding_normalization_scale = data["embedding_normalization_scale"]
    args.resolved_embedding_normalization_axis_min = (
        None
        if data["embedding_normalization_axis_min"] is None
        else data["embedding_normalization_axis_min"].tolist()
    )
    args.resolved_embedding_normalization_axis_max = (
        None
        if data["embedding_normalization_axis_max"] is None
        else data["embedding_normalization_axis_max"].tolist()
    )
    if args.embedding_key:
        scale_msg = (
            ""
            if data["embedding_normalization_scale"] is None
            else f", scale={data['embedding_normalization_scale']:.12g}"
        )
        print(
            f"[data] embedding_normalization={args.embedding_normalization}{scale_msg}",
            flush=True,
        )
        if args.embedding_normalization == "per-axis-minmax-minus2-2":
            print(
                "[data] embedding axes scaled jointly across all time points "
                "to [-2, 2]",
                flush=True,
            )

    t0 = time.time()
    ae = None
    if args.use_ae:
        ae, z = train_ae(data["x"], args, device, outdir)
        print(f"[ae] latent_shape={z.shape}", flush=True)
    else:
        z = data["x"].astype(np.float32, copy=False)
        np.save(outdir / "model_space.npy", z)
        print(f"[ae] disabled; model_space_shape={z.shape}", flush=True)
    args.model_dim = int(z.shape[1])
    with open(outdir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    if args.training_objective == "tigon-density":
        action_solver = (
            args.ode_solver
            if args.action_ode_solver == "same"
            else args.action_ode_solver
        )
        action_steps = (
            args.ode_steps
            if args.action_ode_steps <= 0
            else args.action_ode_steps
        )
        print(
            "[tigon-density] short+long reconstruction; "
            f"solver={args.ode_solver}, rtol={args.ode_rtol:g}, atol={args.ode_atol:g}, "
            f"divergence={args.divergence_estimator}, stabilization={args.density_stabilization}, "
            f"action_solver={action_solver}, action_steps={action_steps}",
            flush=True,
        )
        tigon = train_tigon_density(
            z,
            data["times"],
            data["unique_times"],
            args,
            device,
            outdir,
        )
    else:
        tigon = train_tigon(z, data["times"], data["unique_times"], args, device, outdir)
    growth, growth_df = eval_growth(
        tigon, z, data["obs"], data["times"], data["unique_times"], args, device, outdir
    )
    interval_df = eval_intervals(tigon, z, data["times"], data["unique_times"], args, device, outdir)
    partial = partial_corr_growth_genes(
        data["x"], data["genes"], growth, data["obs"], args, outdir
    )
    saliency = growth_saliency(
        ae, tigon, data["x"], data["genes"], data["times"], data["unique_times"], args, device, outdir
    )
    plot_training(outdir)
    plot_growth_umap(data["umap"], growth, data["obs"], args, outdir)
    plot_gene_tables(partial, saliency, args, outdir)

    scale_summary = {
        "growth_log_rate_mean": float(np.mean(growth)),
        "growth_log_rate_std": float(np.std(growth)),
        "growth_log_rate_quantiles": {
            str(q): float(np.quantile(growth, q)) for q in [0.01, 0.05, 0.5, 0.95, 0.99]
        },
        "use_ae": bool(args.use_ae),
        "model_dim": int(z.shape[1]),
        "embedding_normalization": args.embedding_normalization,
        "embedding_normalization_scale": data["embedding_normalization_scale"],
        "embedding_normalization_axis_min": args.resolved_embedding_normalization_axis_min,
        "embedding_normalization_axis_max": args.resolved_embedding_normalization_axis_max,
        "training_objective": args.training_objective,
        "ode_solver": args.ode_solver if args.training_objective == "tigon-density" else "fixed-step midpoint Euler",
        "ode_rtol": args.ode_rtol if args.training_objective == "tigon-density" else None,
        "ode_atol": args.ode_atol if args.training_objective == "tigon-density" else None,
        "divergence_estimator": args.divergence_estimator if args.training_objective == "tigon-density" else None,
        "divergence_fd_epsilon": (
            args.divergence_fd_epsilon
            if args.training_objective == "tigon-density"
            and args.divergence_estimator == "finite-difference"
            else None
        ),
        "interval_eval": interval_df.to_dict(orient="records"),
        "elapsed_seconds": time.time() - t0,
        "note": "growth is a log-rate per unit time; exp(integral growth dt) is the local mass multiplier.",
    }
    with open(outdir / "growth_scale_summary.json", "w") as f:
        json.dump(scale_summary, f, indent=2)

    print("[done] outdir", outdir, flush=True)
    print("[done] growth_scale_summary.json", flush=True)
    print("[top positive partial]", ", ".join(partial.head(10)["gene"].tolist()), flush=True)
    print(
        "[top saliency]",
        ", ".join(saliency.head(10)["gene"].tolist()),
        flush=True,
    )


if __name__ == "__main__":
    main()
