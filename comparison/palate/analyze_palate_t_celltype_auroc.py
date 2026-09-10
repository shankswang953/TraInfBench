#!/usr/bin/env python
"""Evaluate full and strict-LOTO palate RNA-to-ATAC maps by cell type."""

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

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import torch


ROOT = Path(__file__).resolve().parents[2]
PALATE_ROOT = Path("external/COATI/MouseBrain")
TRAIN_T = PALATE_ROOT / "TrainMap"
DEFAULT_OUTPUT = ROOT / "results/palate_t_celltype_auroc"
RNA_H5AD = ROOT / "data/palate_rna_cytobridge.h5ad"
ATAC_H5AD = ROOT / "data/palate_atac_benchmark.h5ad"
RNA_NORM = ROOT / "data/palate_rna_primal_norm_params.pt"
ATAC_NORM = ROOT / "data/palate_atac_secondary_norm_params_lsi15.pt"
if str(PALATE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PALATE_ROOT.parent))

STAGES = (("E13.5", 1, 1.0), ("E14.0", 2, 1.5))
MODEL_ORDER = ("ATAC atlas reference", "Full T", "Strict LOTO T")
DISPLAY = {
    "CNC-derived progenitors": "CNC progenitor",
    "anterior palatal mesenchymal": "Anterior",
    "dental mesenchymal": "Dental",
    "intermidiate cells": "Intermediate",
    "osteogenic": "Osteogenic",
    "perimysial": "Perimysial",
    "posterior palatal mesenchymal": "Posterior",
}
EXCLUDED_CELLTYPES = {"intermidiate cells"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--min-cells", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_scale(path: Path) -> float:
    state = torch.load(path, map_location="cpu", weights_only=False)
    value = float(np.asarray(state["scale"]).reshape(-1)[0])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid normalization scale in {path}: {value}")
    return value


def _load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    source = TRAIN_T / "train_FiLM_MLP_lsi15.py"
    spec = importlib.util.spec_from_file_location(
        f"palate_film_{checkpoint.stem}", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import FiLM model from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = module.FiLMMLP(**state["config"]).to(device)
    model.load_state_dict(state["state_dict"])
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def _apply_model(
    model: torch.nn.Module,
    points: np.ndarray,
    physical_time: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(points), batch_size):
        batch = torch.as_tensor(
            points[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        time = torch.full(
            (len(batch),), physical_time, dtype=batch.dtype, device=device
        )
        outputs.append(model(batch, time).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def _paired_excluded_knn_probabilities(
    query: np.ndarray,
    reference: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> tuple[list[str], np.ndarray]:
    if len(query) != len(reference) or len(query) != len(labels):
        raise ValueError("Query, reference, and labels must be paired and row-aligned")
    distances, indices = (
        NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
        .fit(reference)
        .kneighbors(query, return_distance=True)
    )
    classes = sorted(set(labels.tolist()))
    class_index = {label: index for index, label in enumerate(classes)}
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


def _compute(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        RNA_H5AD,
        ATAC_H5AD,
        RNA_NORM,
        ATAC_NORM,
        TRAIN_T / "train_FiLM_MLP_lsi15.py",
        TRAIN_T / "T_FiLM_lsi15.pt",
        TRAIN_T / "T_FiLM_lsi15_LOTO_holdout_1.pt",
        TRAIN_T / "T_FiLM_lsi15_LOTO_holdout_2.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))

    rna = ad.read_h5ad(RNA_H5AD)
    atac = ad.read_h5ad(ATAC_H5AD)
    if not np.array_equal(rna.obs_names.to_numpy(str), atac.obs_names.to_numpy(str)):
        raise ValueError("RNA and ATAC cells are not paired in the same order")
    rna_time = pd.to_numeric(
        rna.obs["time_point_processed"], errors="raise"
    ).to_numpy(float)
    atac_time = pd.to_numeric(
        atac.obs["time_point_processed"], errors="raise"
    ).to_numpy(float)
    if not np.array_equal(rna_time, atac_time):
        raise ValueError("RNA and ATAC stage annotations differ")
    labels_all = rna.obs["celltype"].astype(str).to_numpy()
    if not np.array_equal(labels_all, atac.obs["celltype"].astype(str).to_numpy()):
        raise ValueError("RNA and ATAC cell-type annotations differ")

    rna_norm = (
        np.asarray(rna.obsm["X_latent"], dtype=np.float32) / _load_scale(RNA_NORM)
    ).astype(np.float32, copy=False)
    atac_norm = (
        np.asarray(atac.obsm["X_lsi15"], dtype=np.float32) / _load_scale(ATAC_NORM)
    ).astype(np.float32, copy=False)

    device = torch.device(args.device)
    full_model = _load_model(TRAIN_T / "T_FiLM_lsi15.pt", device)
    rows: list[dict[str, object]] = []
    for stage, heldout_index, physical_time in STAGES:
        keep = np.isclose(rna_time, physical_time)
        stage_rna = rna_norm[keep]
        stage_atac = atac_norm[keep]
        labels = labels_all[keep]
        strict_model = _load_model(
            TRAIN_T / f"T_FiLM_lsi15_LOTO_holdout_{heldout_index}.pt", device
        )
        query_sets = {
            "ATAC atlas reference": stage_atac,
            "Full T": _apply_model(
                full_model,
                stage_rna,
                physical_time,
                device,
                args.batch_size,
            ),
            "Strict LOTO T": _apply_model(
                strict_model,
                stage_rna,
                physical_time,
                device,
                args.batch_size,
            ),
        }
        probabilities = {
            model: _paired_excluded_knn_probabilities(
                query, stage_atac, labels, args.knn_k
            )
            for model, query in query_sets.items()
        }
        for celltype in sorted(set(labels.tolist())):
            if celltype in EXCLUDED_CELLTYPES:
                continue
            positive = labels == celltype
            n_cells = int(positive.sum())
            if n_cells < args.min_cells or n_cells == len(labels):
                continue
            for model, (classes, probability) in probabilities.items():
                score = probability[:, classes.index(celltype)]
                rows.append(
                    {
                        "stage": stage,
                        "heldout_time_index": heldout_index,
                        "celltype": celltype,
                        "display_celltype": DISPLAY.get(celltype, celltype),
                        "n_cells": n_cells,
                        "model": model,
                        "one_vs_rest_auroc": float(
                            roc_auc_score(positive.astype(np.int8), score)
                        ),
                    }
                )

    detail = pd.DataFrame(rows)
    summary_rows = []
    for (stage, model), group in detail.groupby(["stage", "model"], sort=False):
        correlation = spearmanr(
            np.log(group["n_cells"].to_numpy(float)),
            group["one_vs_rest_auroc"].to_numpy(float),
        )
        summary_rows.append(
            {
                "stage": stage,
                "model": model,
                "n_celltypes": len(group),
                "macro_mean_auroc": float(group["one_vs_rest_auroc"].mean()),
                "macro_median_auroc": float(group["one_vs_rest_auroc"].median()),
                "minimum_celltype_auroc": float(group["one_vs_rest_auroc"].min()),
                "spearman_auroc_vs_log_cell_number": float(correlation.statistic),
                "spearman_pvalue": float(correlation.pvalue),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def _plot(detail: pd.DataFrame, output_dir: Path) -> None:
    strict_order = (
        detail[detail["model"].eq("Strict LOTO T")]
        .groupby("celltype", observed=True)["one_vs_rest_auroc"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    y_lookup = {celltype: index for index, celltype in enumerate(strict_order)}

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "font.weight": "normal",
            "axes.titlesize": 12,
            "axes.titleweight": "normal",
            "axes.labelsize": 12,
            "axes.labelweight": "normal",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.235, right=0.985, top=0.80, bottom=0.14, wspace=0.14)
    styles = {
        "ATAC atlas reference": {
            "marker": "x",
            "s": 28,
            "color": "#000000",
            "linewidths": 1.0,
        },
        "Full T": {
            "marker": "o",
            "s": 36,
            "facecolors": "white",
            "edgecolors": "#56B4E9",
            "linewidths": 1.1,
        },
        "Strict LOTO T": {
            "marker": "D",
            "s": 30,
            "facecolors": "#D55E00",
            "edgecolors": "#D55E00",
            "linewidths": 0.8,
        },
    }
    for ax, (stage, _, _) in zip(axes, STAGES):
        local = detail[detail["stage"].eq(stage)]
        pivot = local.pivot(
            index="celltype", columns="model", values="one_vs_rest_auroc"
        )
        for celltype, row in pivot.iterrows():
            values = row.reindex(MODEL_ORDER).dropna().to_numpy(float)
            if len(values) >= 2:
                y = y_lookup[celltype]
                ax.plot(
                    [values.min(), values.max()],
                    [y, y],
                    color="#BDBDBD",
                    linewidth=0.7,
                    zorder=1,
                )
        for model in MODEL_ORDER:
            model_data = local[local["model"].eq(model)]
            ax.scatter(
                model_data["one_vs_rest_auroc"],
                [y_lookup[value] for value in model_data["celltype"]],
                label=model,
                zorder=3,
                **styles[model],
            )
        macro = float(
            local[local["model"].eq("Strict LOTO T")][
                "one_vs_rest_auroc"
            ].mean()
        )
        ax.set_title(
            f"{stage}\nLOTO T macro AUROC = {macro:.3f}",
            pad=5,
            fontweight="normal",
        )
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.6)
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    minimum = float(detail["one_vs_rest_auroc"].min())
    axes[0].set_xlim(max(0.5, minimum - 0.04), 1.002)
    axes[0].set_xticks([0.88, 0.92, 0.96, 1.00])
    axes[0].set_yticks(
        np.arange(len(strict_order)),
        [DISPLAY.get(celltype, celltype) for celltype in strict_order],
    )
    axes[0].set_ylim(len(strict_order) - 0.5, -0.5)
    axes[1].tick_params(labelleft=False)
    fig.suptitle(
        "Cross-fitted RNA-to-ATAC mapping by cell type",
        x=0.61,
        y=0.985,
        fontsize=12,
        fontweight="normal",
    )
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        ["ATAC reference", "Full T", "LOTO T"],
        loc="upper center",
        bbox_to_anchor=(0.61, 0.925),
        ncol=3,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.1,
    )
    fig.supxlabel("One-vs-rest AUROC", x=0.61, y=0.035)
    stem = output_dir / "t_celltype_auroc_5nn_dotplot"
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "t_celltype_auroc.csv"
    summary_path = output_dir / "t_celltype_auroc_summary.csv"
    figure_path = output_dir / "t_celltype_auroc_5nn_dotplot.png"
    if not args.overwrite and any(
        path.exists() for path in (detail_path, summary_path, figure_path)
    ):
        raise FileExistsError(f"Refusing to overwrite {output_dir}; pass --overwrite")

    detail, summary = _compute(args)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    _plot(detail, output_dir)
    manifest = {
        "task": "Palate per-cell-type RNA-to-ATAC T one-vs-rest AUROC",
        "stages": [stage for stage, _, _ in STAGES],
        "models": {
            "Full T": str(TRAIN_T / "T_FiLM_lsi15.pt"),
            "Strict LOTO T E13.5": str(
                TRAIN_T / "T_FiLM_lsi15_LOTO_holdout_1.pt"
            ),
            "Strict LOTO T E14.0": str(
                TRAIN_T / "T_FiLM_lsi15_LOTO_holdout_2.pt"
            ),
        },
        "space": "globally normalized ATAC LSI15",
        "reference": "same-stage paired observed ATAC atlas",
        "paired_reference_exclusion": True,
        "knn": args.knn_k,
        "knn_weight": "adaptive exp(-(distance / distance_to_kth)^2)",
        "minimum_cells_per_celltype": args.min_cells,
        "excluded_celltypes": sorted(EXCLUDED_CELLTYPES),
        "primary_summary": "macro mean AUROC across retained cell types",
        "outputs": {
            "detail": str(detail_path),
            "summary": str(summary_path),
            "png": str(figure_path),
            "pdf": str(figure_path.with_suffix(".pdf")),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
