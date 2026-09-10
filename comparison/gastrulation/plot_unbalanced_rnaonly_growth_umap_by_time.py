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
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

ROOT_TRAINF = Path("external/COATI")
if ROOT_TRAINF.exists():
    sys.path.insert(0, str(ROOT_TRAINF))

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.Neural import MLPVectorField


def parse_limits(value: str) -> tuple[float, float]:
    parts = [float(x.strip()) for x in value.split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError("Clip limits must be two comma-separated numbers")
    lo, hi = parts
    if lo >= hi:
        raise ValueError(f"Invalid limits: {lo}, {hi}")
    return lo, hi


def load_scale(norm_params: Path) -> float:
    params = torch.load(norm_params, map_location="cpu", weights_only=False)
    scale = float(np.asarray(params["scale"]).reshape(-1)[0])
    if scale <= 0:
        raise ValueError(f"Invalid scale: {scale}")
    return scale


def load_model(checkpoint: Path, device: torch.device) -> MLPVectorField:
    model = MLPVectorField(
        dim=50,
        hidden_dim=800,
        n_layers=3,
        activation="leaky_relu",
        unbalanced=True,
        alpha_growth=1.0,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["func_state_dict"])
    model.eval()
    return model


def compute_growth(
    model: MLPVectorField,
    x_norm: np.ndarray,
    times: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    out = np.zeros(x_norm.shape[0], dtype=np.float32)
    with torch.no_grad():
        for t in sorted(np.unique(times)):
            idx = np.flatnonzero(np.isclose(times, t))
            vals = []
            for start in range(0, idx.shape[0], batch_size):
                batch_idx = idx[start : start + batch_size]
                xb = torch.as_tensor(x_norm[batch_idx], dtype=torch.float32, device=device)
                tb = torch.full((xb.shape[0], 1), float(t), dtype=torch.float32, device=device)
                inp = torch.cat([xb, tb], dim=1)
                vals.append(model.growth_net(inp).squeeze(1).detach().cpu().numpy())
            out[idx] = np.concatenate(vals).astype(np.float32, copy=False)
    return out


def summary_by_stage(df: pd.DataFrame, stage_key: str) -> pd.DataFrame:
    rows = []
    for stage, part in df.groupby(stage_key, sort=True):
        vals = part["growth_log_rate"].to_numpy(dtype=float)
        rows.append(
            {
                stage_key: stage,
                "n_cells": int(vals.size),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "q01": float(np.quantile(vals, 0.01)),
                "q05": float(np.quantile(vals, 0.05)),
                "median": float(np.quantile(vals, 0.5)),
                "q95": float(np.quantile(vals, 0.95)),
                "q99": float(np.quantile(vals, 0.99)),
                "max": float(np.max(vals)),
                "frac_gt_0": float(np.mean(vals > 0)),
                "frac_lt_0": float(np.mean(vals < 0)),
            }
        )
    return pd.DataFrame(rows)


def sorted_stages(values) -> list[str]:
    stages = list(pd.unique(values))
    try:
        return sorted(stages, key=lambda x: float(str(x).replace("E", "")))
    except ValueError:
        return sorted(stages)


def plot_umap_by_time(
    df: pd.DataFrame,
    stage_key: str,
    clip: tuple[float, float],
    title: str,
    output_png: Path,
    output_pdf: Path,
    point_size: float,
    alpha: float,
) -> None:
    stages = sorted_stages(df[stage_key])
    fig, axes = plt.subplots(
        1,
        len(stages),
        figsize=(4.25 * len(stages) + 0.65, 4.05),
        dpi=240,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(stages) == 1:
        axes = [axes]

    vmin, vmax = clip
    scatter = None
    for ax, stage in zip(axes, stages):
        part = df[df[stage_key] == stage]
        scatter = ax.scatter(
            part["umap1"],
            part["umap2"],
            c=np.clip(part["growth_log_rate"], vmin, vmax),
            s=point_size,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            alpha=alpha,
            rasterized=True,
        )
        ax.set_title(str(stage), fontsize=12, pad=8)
        ax.set_xlabel("UMAP 1", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("UMAP 2", fontsize=10)
    fig.suptitle(title, fontsize=14)
    if scatter is not None:
        cb = fig.colorbar(scatter, ax=axes, shrink=0.88, pad=0.012)
        cb.set_label("Growth log-rate", fontsize=10)
        cb.ax.tick_params(labelsize=8)
    fig.savefig(output_png, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_hist_by_time(df: pd.DataFrame, stage_key: str, output_png: Path, output_pdf: Path) -> None:
    stages = sorted_stages(df[stage_key])
    fig, ax = plt.subplots(figsize=(7.6, 4.5), dpi=220, constrained_layout=True)
    bins = np.linspace(
        np.quantile(df["growth_log_rate"], 0.005),
        np.quantile(df["growth_log_rate"], 0.995),
        75,
    )
    for stage in stages:
        vals = df.loc[df[stage_key] == stage, "growth_log_rate"].to_numpy(dtype=float)
        ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.8, label=str(stage))
    ax.axvline(0, color="#111111", linewidth=0.9, alpha=0.7)
    ax.set_xlabel("Growth log-rate", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Unbalanced RNA-only growth distribution by time", fontsize=13)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(output_png, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-cell growth from UnbalancedRNAOnly on gastrulation UMAP.")
    parser.add_argument("--input-h5ad", type=Path, default=Path("data/gastrulation_rna_cytobridge.h5ad"))
    parser.add_argument("--latent-key", default="X_latent")
    parser.add_argument("--time-key", default="time_point_processed")
    parser.add_argument("--stage-key", default="stage")
    parser.add_argument("--norm-params", type=Path, default=Path("data/gastrulation_rna_primal_norm_params.pt"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "external/COATI/Gastrulation/UnbalancedRNAOnly/"
            "checkpoint/ckpt_s0_e0.1_m100.0_d0.1_iter20000.pth"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/unbalanced_rnaonly_gastrulation_iter20000"))
    parser.add_argument("--clip", default="-2,2")
    parser.add_argument("--wide-clip", default="-15,3")
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)
    adata = ad.read_h5ad(args.input_h5ad)
    scale = load_scale(args.norm_params)
    x_raw = np.asarray(adata.obsm[args.latent_key], dtype="float32")
    x_norm = (x_raw / scale).astype("float32", copy=False)
    times = pd.to_numeric(adata.obs[args.time_key], errors="raise").to_numpy().astype(np.float32)
    umap = np.asarray(adata.obsm["X_umap"], dtype="float32")

    model = load_model(args.checkpoint, device)
    growth = compute_growth(model, x_norm, times, args.batch_size, device)
    df = pd.DataFrame(
        {
            "cell": adata.obs_names.to_numpy(dtype=str),
            "time": times,
            "stage": adata.obs[args.stage_key].astype(str).to_numpy(),
            "umap1": umap[:, 0],
            "umap2": umap[:, 1],
            "growth_log_rate": growth,
            "growth_factor_per_unit_time": np.exp(np.clip(growth, -20, 20)),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_cell_path = args.output_dir / "unbalanced_rnaonly_growth_umap_per_cell.csv"
    summary_path = args.output_dir / "unbalanced_rnaonly_growth_by_time_summary.csv"
    df.to_csv(per_cell_path, index=False)
    summary_by_stage(df, "stage").to_csv(summary_path, index=False)

    clip = parse_limits(args.clip)
    wide_clip = parse_limits(args.wide_clip)
    plot_umap_by_time(
        df,
        "stage",
        clip,
        "Unbalanced RNA-only growth log-rate by time",
        args.output_dir / "unbalanced_rnaonly_growth_umap_by_time_clipped_pm2.png",
        args.output_dir / "unbalanced_rnaonly_growth_umap_by_time_clipped_pm2.pdf",
        args.point_size,
        args.alpha,
    )
    plot_umap_by_time(
        df,
        "stage",
        wide_clip,
        "Unbalanced RNA-only growth log-rate by time",
        args.output_dir / "unbalanced_rnaonly_growth_umap_by_time_wide.png",
        args.output_dir / "unbalanced_rnaonly_growth_umap_by_time_wide.pdf",
        args.point_size,
        args.alpha,
    )
    plot_hist_by_time(
        df,
        "stage",
        args.output_dir / "unbalanced_rnaonly_growth_log_rate_hist_by_time.png",
        args.output_dir / "unbalanced_rnaonly_growth_log_rate_hist_by_time.pdf",
    )

    print(f"cells: {df.shape[0]}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"scale: {scale:.12g}")
    print(f"growth range: {float(growth.min()):.6g} .. {float(growth.max()):.6g}")
    print(f"wrote: {per_cell_path}")
    print(f"wrote: {summary_path}")
    print(f"wrote: {args.output_dir / 'unbalanced_rnaonly_growth_umap_by_time_clipped_pm2.png'}")
    print(f"wrote: {args.output_dir / 'unbalanced_rnaonly_growth_umap_by_time_wide.png'}")
    print(f"wrote: {args.output_dir / 'unbalanced_rnaonly_growth_log_rate_hist_by_time.png'}")


if __name__ == "__main__":
    main()
