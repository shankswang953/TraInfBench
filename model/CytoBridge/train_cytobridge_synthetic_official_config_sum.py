#!/usr/bin/env python
"""Train synthetic CytoBridge with official OT plans and weighted-sum energy."""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
import inspect
import shutil
import time
from pathlib import Path

import anndata as ad

from CytoBridge.tl.fit import fit
from CytoBridge.tl.trainer import TrainingPipeline
from CytoBridge.utils.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("balanced", "unbalanced"), required=True)
    parser.add_argument(
        "--input-h5ad",
        type=Path,
        default=Path("data/synthetic_rna10_cytobridge.h5ad"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--pretrain-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def official_config(args: argparse.Namespace) -> dict:
    velocity = {
        "hidden_dim": 400,
        "n_layers": 2,
        "residual": False,
    }
    if args.mode == "balanced":
        model = {"components": ["velocity"], "velocity_net": velocity}
        defaults = {
            "lr": 1e-4,
            "lambda_ot": 1.0,
            "lambda_mass": 0.01,
            "lambda_energy": 0.01,
        }
        plan = [
            {
                "name": "Train",
                "mode": "neural_ode",
                "method": "ODEFunc",
                "epochs": args.epochs,
                "global_mass": True,
                "OT_loss": "sinkhorn",
                "lambda_ot": 10.0,
                "lambda_mass": 10.0,
                "lambda_energy": 0.01,
            }
        ]
    else:
        model = {
            "components": ["velocity", "growth"],
            "velocity_net": velocity,
            "growth_net": velocity.copy(),
        }
        defaults = {
            "lr": 1e-4,
            "lambda_ot": 1.0,
            "lambda_mass": 0.01,
            "lambda_energy": 0.01,
        }
        plan = [
            {
                "name": "Pretrain",
                "mode": "neural_ode",
                "method": "ODEFunc",
                "epochs": args.pretrain_epochs,
                "global_mass": False,
                "OT_loss": "sinkhorn_detach",
                "lambda_energy": 0.0,
            },
            {
                "name": "Train",
                "mode": "neural_ode",
                "method": "ODEFunc",
                "epochs": args.epochs,
                "global_mass": True,
                "OT_loss": "sinkhorn",
                "lambda_ot": 10.0,
                "lambda_mass": 10.0,
                "lambda_energy": 0.01,
            },
        ]
    return {
        "ckpt_dir": str(args.output_dir),
        "model": model,
        "training": {"defaults": defaults, "plan": plan},
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.pretrain_epochs < 0 or args.batch_size < 1:
        raise ValueError("Epoch and batch-size arguments must be positive")
    trainer_source = inspect.getsource(TrainingPipeline.train_neural_ode_epoch)
    if "loss_energy = e1.sum()" not in trainer_source:
        raise RuntimeError(
            "This experiment requires the weighted-sum energy implementation."
        )
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} exists; pass --overwrite to replace it"
            )
        shutil.rmtree(args.output_dir)

    adata = ad.read_h5ad(args.input_h5ad)
    config = official_config(args)
    set_seed(args.seed)
    print(f"mode: {args.mode}")
    print(f"input: {args.input_h5ad}")
    print(f"output: {args.output_dir}")
    print(f"epochs: {args.epochs}; pretrain_epochs: {args.pretrain_epochs}")
    print(f"batch_size: {args.batch_size}; seed: {args.seed}; device: {args.device}")
    print("energy aggregation: sum over already weighted particles")
    print(f"training plan: {config['training']['plan']}")
    start = time.time()
    fitted = fit(
        adata,
        config=config,
        batch_size=args.batch_size,
        device=args.device,
    )
    if args.mode == "unbalanced" and "growth_rate" not in fitted.obsm:
        raise RuntimeError("Unbalanced fit did not produce growth_rate")
    print(f"elapsed_seconds: {time.time() - start:.2f}")


if __name__ == "__main__":
    main()
