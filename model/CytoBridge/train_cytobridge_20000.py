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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CytoBridge on shared moscot RNA space.")
    parser.add_argument("--input-h5ad", type=Path, default=Path("data/moscot_rna_cytobridge.h5ad"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/cytobridge_moscot_rna_50000"))
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lambda-ot", type=float, default=10.0)
    parser.add_argument("--lambda-mass", type=float, default=10.0)
    parser.add_argument("--lambda-energy", type=float, default=0.01)
    parser.add_argument("--lambda-density", type=float, default=0.0)
    parser.add_argument("--density-top-k", type=int, default=5)
    parser.add_argument("--density-hinge-value", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if min(
        args.lambda_ot,
        args.lambda_mass,
        args.lambda_energy,
        args.lambda_density,
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
    train_stage = {
        "name": "Train",
        "mode": "neural_ode",
        "method": "ODEFunc",
        "epochs": args.epochs,
        "global_mass": True,
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

    config = {
        "ckpt_dir": str(args.output_dir),
        "model": {
            "components": ["velocity"],
            "velocity_net": {
                "hidden_dim": 400,
                "n_layers": 2,
                "residual": False,
                "activation": "leaky_relu",
            },
        },
        "training": {
            "defaults": {
                "lr": 0.0001,
                "lambda_ot": args.lambda_ot,
                "lambda_mass": args.lambda_mass,
                "lambda_energy": args.lambda_energy,
                "lambda_density": args.lambda_density,
                "batch_size": args.batch_size,
            },
            "plan": [train_stage],
        },
    }
    print(f"input: {args.input_h5ad}")
    print(f"output: {args.output_dir}")
    print(f"epochs: {args.epochs}, batch_size: {args.batch_size}, device: {args.device}")
    print(
        "loss coefficients: "
        f"ot={args.lambda_ot}, mass={args.lambda_mass}, "
        f"density={args.lambda_density}, energy={args.lambda_energy}"
    )
    print(f"shape: {adata.shape}, X_latent: {adata.obsm['X_latent'].shape}")
    print(
        "time_values: "
        f"{[float(value) for value in sorted(adata.obs['time_point_processed'].unique())]}"
    )
    start = time.time()
    fit(adata, config=config, batch_size=args.batch_size, device=args.device)
    print(f"elapsed_seconds: {time.time() - start:.2f}")


if __name__ == "__main__":
    main()
