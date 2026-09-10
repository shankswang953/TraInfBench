#!/usr/bin/env python
"""Evaluate full and strict-LOTO pancreas RNA-to-ATAC maps by cell type."""

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
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import torch

from trainfbench_plot_style import NATURE_CUD, apply_nature_rc


ROOT = Path(__file__).resolve().parents[2]
MOSCOT_DATA = Path("external/COATI/moscot/data")
TRAIN_T = MOSCOT_DATA / "TrainT"
DEFAULT_OUTPUT = ROOT / "results/pancreas_t_celltype_auroc"
sys.path.insert(0, str(MOSCOT_DATA.parents[1]))

MODEL_ORDER = ("ATAC atlas reference", "Full T", "Strict LOTO T")
DISPLAY = {
    "Ngn3 high cycling": r"Ngn3$^{high}$ cycling",
    "Ngn3 high": r"Ngn3$^{high}$",
    "Ngn3 low": r"Ngn3$^{low}$",
    "Eps. progenitors": "Eps. prog.",
    "Fev+": r"Fev$^{+}$",
    "Fev+ Alpha": r"Fev$^{+}$ alpha",
    "Fev+ Beta": r"Fev$^{+}$ beta",
    "Fev+ Delta": r"Fev$^{+}$ delta",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--minimum-cells", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model_file = TRAIN_T / "train_FiLM_MLP.py"
    spec = importlib.util.spec_from_file_location(
        f"pancreas_film_{checkpoint.stem}", model_file
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {model_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = module.FiLMMLP(**state["config"]).to(device)
    model.load_state_dict(state["state_dict"])
    return model.eval().requires_grad_(False)


@torch.no_grad()
def apply_model(
    model: torch.nn.Module,
    points: np.ndarray,
    physical_time: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(points), batch_size):
        batch = torch.as_tensor(
            points[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        outputs.append(model(batch, physical_time).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def paired_excluded_knn_probabilities(
    query: np.ndarray,
    reference: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> tuple[list[str], np.ndarray]:
    if len(query) != len(reference) or len(query) != len(labels):
        raise ValueError("RNA, ATAC and labels must be paired and row-aligned")
    distances, indices = NearestNeighbors(
        n_neighbors=min(k + 1, len(reference)), n_jobs=-1
    ).fit(reference).kneighbors(query, return_distance=True)
    classes = sorted(set(labels.tolist()))
    class_index = {name: index for index, name in enumerate(classes)}
    probabilities = np.zeros((len(query), len(classes)), dtype=np.float64)
    for row in range(len(query)):
        keep = indices[row] != row
        local_indices = indices[row][keep][:k]
        local_distances = distances[row][keep][:k]
        if len(local_indices) != k:
            raise RuntimeError(f"Could not form paired-cell-excluded {k}-NN")
        bandwidth = max(float(local_distances[-1]), 1e-12)
        weights = np.exp(-np.square(local_distances / bandwidth))
        weights /= weights.sum()
        for label, weight in zip(labels[local_indices], weights):
            probabilities[row, class_index[str(label)]] += float(weight)
    return classes, probabilities


def load_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rna_scale = float(
        torch.load(
            MOSCOT_DATA / "primal_norm_params.pt",
            map_location="cpu",
            weights_only=False,
        )["scale"]
    )
    atac_scale = float(
        torch.load(
            MOSCOT_DATA / "secondary_norm_params_poissonvi.pt",
            map_location="cpu",
            weights_only=False,
        )["scale"]
    )
    with np.load(MOSCOT_DATA / "rna_time_data.npz") as source:
        rna = np.asarray(source["time_1"], dtype=np.float32) / rna_scale
    with np.load(MOSCOT_DATA / "atac_poissonvi_time_data.npz") as source:
        atac = np.asarray(source["time_1"], dtype=np.float32) / atac_scale
    with np.load(MOSCOT_DATA / "celltype_sub_by_stage.npz", allow_pickle=True) as source:
        labels = np.asarray(source["time_1"], dtype=str)
    if not (len(rna) == len(atac) == len(labels)):
        raise ValueError("E15.5 paired arrays are not row-aligned")
    return rna, atac, labels


def calculate(
    k: int,
    minimum_cells: int,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device("cpu")
    rna, atac, labels = load_inputs()
    full_model = load_model(TRAIN_T / "T_FiLM_poissonvi.pt", device)
    loto_model = load_model(
        TRAIN_T / "T_FiLM_poissonvi_LOTO_holdout_1.pt", device
    )
    queries = {
        "ATAC atlas reference": atac,
        "Full T": apply_model(full_model, rna, 1.0, device, batch_size),
        "Strict LOTO T": apply_model(loto_model, rna, 1.0, device, batch_size),
    }
    probability_sets = {
        name: paired_excluded_knn_probabilities(query, atac, labels, k)
        for name, query in queries.items()
    }
    rows: list[dict[str, object]] = []
    for celltype in sorted(set(labels.tolist())):
        positive = labels == celltype
        n_cells = int(positive.sum())
        if n_cells < minimum_cells or n_cells == len(labels):
            continue
        for model_name, (classes, probabilities) in probability_sets.items():
            class_index = classes.index(celltype)
            rows.append(
                {
                    "stage": "E15.5",
                    "celltype": celltype,
                    "n_cells": n_cells,
                    "model": model_name,
                    "one_vs_rest_auroc": float(
                        roc_auc_score(
                            positive.astype(np.int8), probabilities[:, class_index]
                        )
                    ),
                }
            )
    detail = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for model_name, group in detail.groupby("model", sort=False):
        correlation = spearmanr(
            np.log(group["n_cells"].to_numpy(float)),
            group["one_vs_rest_auroc"].to_numpy(float),
        )
        summary_rows.append(
            {
                "stage": "E15.5",
                "model": model_name,
                "n_celltypes": int(len(group)),
                "macro_mean_auroc": float(group["one_vs_rest_auroc"].mean()),
                "macro_median_auroc": float(group["one_vs_rest_auroc"].median()),
                "minimum_celltype_auroc": float(group["one_vs_rest_auroc"].min()),
                "spearman_auroc_vs_log_cell_number": float(correlation.statistic),
                "spearman_pvalue": float(correlation.pvalue),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def reference_adjusted_summary(detail: pd.DataFrame) -> pd.DataFrame:
    wide = detail.pivot(
        index=["celltype", "n_cells"],
        columns="model",
        values="one_vs_rest_auroc",
    ).reset_index()
    rows: list[dict[str, object]] = []
    for model_name in ("Full T", "Strict LOTO T"):
        delta = wide[model_name] - wide["ATAC atlas reference"]
        correlation = spearmanr(np.log(wide["n_cells"].to_numpy(float)), delta)
        rows.append(
            {
                "model": model_name,
                "n_celltypes": int(len(wide)),
                "mean_auroc_minus_atac_reference": float(delta.mean()),
                "median_auroc_minus_atac_reference": float(delta.median()),
                "minimum_auroc_minus_atac_reference": float(delta.min()),
                "maximum_auroc_minus_atac_reference": float(delta.max()),
                "spearman_delta_vs_log_cell_number": float(correlation.statistic),
                "spearman_pvalue": float(correlation.pvalue),
            }
        )
    return pd.DataFrame(rows)


def plot(detail: pd.DataFrame, output_dir: Path) -> None:
    strict_order = (
        detail.loc[detail["model"].eq("Strict LOTO T")]
        .sort_values("one_vs_rest_auroc")
        ["celltype"]
        .tolist()
    )
    y_lookup = {celltype: index for index, celltype in enumerate(strict_order)}
    pivot = detail.pivot(
        index="celltype", columns="model", values="one_vs_rest_auroc"
    ).reindex(strict_order)

    apply_nature_rc(font_size=10.0)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "font.weight": "normal",
            "axes.titlesize": 10.0,
            "axes.titleweight": "normal",
            "axes.labelsize": 10.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.0,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(4.05, 4.45), facecolor="white")
    for celltype, row in pivot.iterrows():
        values = row.reindex(MODEL_ORDER).dropna().to_numpy(float)
        ax.plot(
            [values.min(), values.max()],
            [y_lookup[celltype], y_lookup[celltype]],
            color="#BDBDBD",
            linewidth=0.7,
            zorder=1,
        )
    styles = {
        "ATAC atlas reference": {
            "marker": "x",
            "s": 25,
            "color": NATURE_CUD["black"],
            "linewidths": 0.9,
        },
        "Full T": {
            "marker": "o",
            "s": 31,
            "facecolors": "white",
            "edgecolors": NATURE_CUD["sky_blue"],
            "linewidths": 1.0,
        },
        "Strict LOTO T": {
            "marker": "D",
            "s": 27,
            "facecolors": NATURE_CUD["vermillion"],
            "edgecolors": NATURE_CUD["vermillion"],
            "linewidths": 0.8,
        },
    }
    for model_name in MODEL_ORDER:
        local = detail.loc[detail["model"].eq(model_name)]
        ax.scatter(
            local["one_vs_rest_auroc"],
            [y_lookup[celltype] for celltype in local["celltype"]],
            label=model_name,
            zorder=3,
            **styles[model_name],
        )
    macro = detail.groupby("model")["one_vs_rest_auroc"].mean()
    minimum = float(detail["one_vs_rest_auroc"].min())
    lower = max(0.50, np.floor((minimum - 0.02) * 20.0) / 20.0)
    ax.set_xlim(lower, 1.002)
    ax.set_xticks(np.linspace(lower, 1.0, 4))
    ax.set_yticks(
        np.arange(len(strict_order)),
        [DISPLAY.get(celltype, celltype) for celltype in strict_order],
    )
    ax.set_ylim(len(strict_order) - 0.5, -0.5)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("One-vs-rest AUROC")
    ax.set_title(
        f"E15.5 LOTO T macro AUROC = {macro['Strict LOTO T']:.3f}",
        pad=3,
        fontweight="normal",
    )
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        ["ATAC reference", "Full T", "LOTO T"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.055),
        ncol=3,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.7,
        borderaxespad=0.0,
    )
    fig.suptitle(
        "Cross-fitted RNA-to-ATAC mapping by cell type",
        fontsize=10.0,
        fontweight="normal",
        y=0.995,
    )
    fig.subplots_adjust(left=0.39, right=0.985, top=0.84, bottom=0.12)
    stem = output_dir / "pancreas_t_celltype_auroc_5nn_dotplot"
    options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.02}
    fig.savefig(stem.with_suffix(".png"), dpi=600, **options)
    fig.savefig(stem.with_suffix(".pdf"), **options)
    fig.savefig(stem.with_suffix(".svg"), **options)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} is not empty; pass --overwrite to replace its files"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail, summary = calculate(args.knn_k, args.minimum_cells, args.batch_size)
    adjusted = reference_adjusted_summary(detail)
    detail.to_csv(args.output_dir / "t_celltype_auroc.csv", index=False)
    summary.to_csv(args.output_dir / "t_celltype_auroc_summary.csv", index=False)
    adjusted.to_csv(
        args.output_dir / "t_celltype_reference_adjusted_summary.csv", index=False
    )
    plot(detail, args.output_dir)
    manifest = {
        "task": "Pancreas E15.5 per-cell-type T-to-ATAC one-vs-rest AUROC",
        "full_checkpoint": str(TRAIN_T / "T_FiLM_poissonvi.pt"),
        "strict_loto_checkpoint": str(
            TRAIN_T / "T_FiLM_poissonvi_LOTO_holdout_1.pt"
        ),
        "heldout_stage": "E15.5 / time_1",
        "reference": "Observed paired PoissonVI ATAC 22D atlas",
        "paired_reference_exclusion": True,
        "knn": args.knn_k,
        "minimum_cells_per_celltype": args.minimum_cells,
        "primary_summary": "Macro mean across retained cell types",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(adjusted.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
