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
import shutil
import time
from pathlib import Path

import anndata as ad
from CytoBridge.tl.fit import fit
from CytoBridge.utils.utils import set_seed


def build_unbalanced_config(args: argparse.Namespace) -> dict:
    plan = []
    if args.pretrain_epochs > 0:
        pretrain_stage = {
            "name": "Pretrain",
            "mode": "neural_ode",
            "method": "ODEFunc",
            "epochs": args.pretrain_epochs,
            "global_mass": False,
            "OT_loss": "sinkhorn_detach",
            "lambda_ot": args.lambda_ot,
            "lambda_mass": args.lambda_mass,
            "lambda_energy": 0.0,
            "save_strategy": "last",
        }
        if args.pretrain_lambda_density > 0:
            pretrain_stage.update(
                {
                    "use_density_loss": True,
                    "lambda_density": args.pretrain_lambda_density,
                    "density_top_k": args.density_top_k,
                    "density_hinge_value": args.density_hinge_value,
                }
            )
        plan.append(pretrain_stage)

    train_stage = {
        "name": "Train",
        "mode": "neural_ode",
        "method": "ODEFunc",
        "epochs": args.epochs,
        "global_mass": args.global_mass,
        "OT_loss": "sinkhorn",
        "lambda_ot": args.lambda_ot,
        "lambda_mass": args.lambda_mass,
        "lambda_energy": args.lambda_energy,
        "save_strategy": "last",
    }
    if args.lambda_density > 0:
        train_stage.update(
            {
                "use_density_loss": True,
                "lambda_density": args.lambda_density,
                "density_top_k": args.density_top_k,
                "density_hinge_value": args.density_hinge_value,
            }
        )
    plan.append(train_stage)

    return {
        "ckpt_dir": str(args.output_dir),
        "model": {
            "components": ["velocity", "growth"],
            "velocity_net": {
                "hidden_dim": args.hidden_dim,
                "n_layers": args.n_layers,
                "residual": False,
                "activation": "leaky_relu",
            },
            "growth_net": {
                "hidden_dim": args.hidden_dim,
                "n_layers": args.n_layers,
                "residual": False,
                "activation": "leaky_relu",
            },
        },
        "training": {
            "defaults": {
                "lr": args.lr,
                "lambda_ot": args.lambda_ot,
                "lambda_mass": args.lambda_mass,
                "lambda_energy": args.lambda_energy,
                "lambda_density": args.lambda_density,
                "batch_size": args.batch_size,
            },
            "plan": plan,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train unbalanced CytoBridge (velocity + growth) on shared RNA latent space."
    )
    parser.add_argument("--input-h5ad", type=Path, default=Path("data/gastrulation_rna_cytobridge.h5ad"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/cytobridge_gastrulation_rna_20000_unbalanced"))
    parser.add_argument("--epochs", type=int, default=20000, help="Epochs for the main unbalanced Train stage.")
    parser.add_argument("--pretrain-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=400)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-ot", type=float, default=10.0)
    parser.add_argument("--lambda-mass", type=float, default=10.0)
    parser.add_argument("--lambda-energy", type=float, default=0.01)
    parser.add_argument("--lambda-density", type=float, default=0.0)
    parser.add_argument(
        "--pretrain-lambda-density",
        type=float,
        default=0.0,
        help="Density coefficient during growth pretraining; energy stays zero.",
    )
    parser.add_argument("--density-top-k", type=int, default=5)
    parser.add_argument("--density-hinge-value", type=float, default=0.01)
    parser.add_argument("--global-mass", dest="global_mass", action="store_true", default=True)
    parser.add_argument("--no-global-mass", dest="global_mass", action="store_false")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if min(
        args.lambda_ot,
        args.lambda_mass,
        args.lambda_energy,
        args.lambda_density,
        args.pretrain_lambda_density,
        args.density_hinge_value,
    ) < 0:
        raise ValueError("Loss coefficients and density hinge must be non-negative")
    if args.density_top_k < 1:
        raise ValueError("--density-top-k must be positive")

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_dir)

    set_seed(args.seed)
    adata = ad.read_h5ad(args.input_h5ad)
    if "X_latent" not in adata.obsm:
        raise KeyError(f"X_latent not found in {args.input_h5ad}.obsm")
    if "time_point_processed" not in adata.obs:
        raise KeyError(f"time_point_processed not found in {args.input_h5ad}.obs")

    config = build_unbalanced_config(args)
    print(f"input: {args.input_h5ad}")
    print(f"output: {args.output_dir}")
    print(
        "epochs: "
        f"pretrain={args.pretrain_epochs}, train={args.epochs}, "
        f"batch_size={args.batch_size}, device={args.device}"
    )
    print(f"components: {config['model']['components']}")
    print(
        "loss coefficients: "
        f"ot={args.lambda_ot}, mass={args.lambda_mass}, "
        f"pretrain_density={args.pretrain_lambda_density}, "
        f"density={args.lambda_density}, energy={args.lambda_energy}"
    )
    print(f"shape: {adata.shape}, X_latent: {adata.obsm['X_latent'].shape}")
    print(f"time grid: {sorted(adata.obs['time_point_processed'].unique())}")

    start = time.time()
    adata = fit(adata, config=config, batch_size=args.batch_size, device=args.device)
    if "growth_rate" not in adata.obsm:
        raise RuntimeError("Unbalanced CytoBridge finished without adata.obsm['growth_rate'].")
    print(f"growth_rate: {adata.obsm['growth_rate'].shape}")
    print(f"elapsed_seconds: {time.time() - start:.2f}")


if __name__ == "__main__":
    main()
