#!/usr/bin/env python3
"""Combine four synchronization-ablation experiments into one figure.

The figure follows the established Pancreas/Palate/Gastrulation small-multiple
style.  Pancreas and Human cerebral share the first two-row block, followed by
Palate and Gastrulation.  A single legend is shared by all panels.
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
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from trainfbench_plot_style import NATURE_CUD, apply_nature_rc, method_style


ROOT = Path(__file__).resolve().parents[2]
HUMAN_RESULT_DIR = Path(
    "external/COATI/humanCerebral/ResultCompare/"
    "synchronization_ablation_7time_iter30000"
)
DEFAULT_OUTPUT_DIR = ROOT / "results/combined_synchronization_ablation"
METHOD_ORDER = (
    "COATI unbal",
    "UOT baseline RNA",
    "UOT baseline ATAC",
    "COATI bal",
    "OT baseline RNA",
    "OT baseline ATAC",
)
DISPLAY_LABELS = {
    "UOT baseline RNA": "UOT(RNA)",
    "UOT baseline ATAC": "UOT(ATAC)",
    "OT baseline RNA": "OT(RNA)",
    "OT baseline ATAC": "OT(ATAC)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pancreas-unbalanced",
        type=Path,
        default=ROOT
        / "results/pancreas_sync_ablation_full20k_inverse_readout/"
        "sinkhorn_ablation_unbalanced.csv",
    )
    parser.add_argument(
        "--pancreas-balanced",
        type=Path,
        default=ROOT
        / "results/pancreas_sync_ablation_full20k_inverse_readout/"
        "sinkhorn_ablation_balanced.csv",
    )
    parser.add_argument(
        "--palate-unbalanced",
        type=Path,
        default=ROOT
        / "results/palate_sinkhorn_ablation/sinkhorn_ablation_unbalanced.csv",
    )
    parser.add_argument(
        "--palate-balanced",
        type=Path,
        default=ROOT
        / "results/palate_sinkhorn_ablation/sinkhorn_ablation_balanced.csv",
    )
    parser.add_argument(
        "--gastrulation-unbalanced",
        type=Path,
        default=ROOT
        / "results/gastrulation_sinkhorn_ablation/sinkhorn_ablation_unbalanced.csv",
    )
    parser.add_argument(
        "--gastrulation-balanced",
        type=Path,
        default=ROOT
        / "results/gastrulation_sinkhorn_ablation/sinkhorn_ablation_balanced.csv",
    )
    parser.add_argument(
        "--human-existing",
        type=Path,
        default=HUMAN_RESULT_DIR
        / "scatter_points_sum_all_times_coati_unbal_all_Cy_iter40000_uot_iter40000.csv",
    )
    parser.add_argument(
        "--human-balanced-sweep",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "human_cerebral_balanced_cy_sweep_all_time.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_standard_pair(
    unbalanced_path: Path,
    balanced_path: Path,
    dataset: str,
) -> pd.DataFrame:
    parts = []
    for path in (unbalanced_path, balanced_path):
        if not path.is_file():
            raise FileNotFoundError(path)
        part = pd.read_csv(path)
        required = {"stage", "method", "rna_sinkhorn", "atac_sinkhorn"}
        missing = required.difference(part.columns)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        if "cy" not in part.columns:
            if "alpha" not in part.columns:
                raise ValueError(f"{path} has neither cy nor alpha")
            part = part.rename(columns={"alpha": "cy"})
        part = part.copy()
        part["dataset"] = dataset
        part["source_csv"] = str(path.resolve())
        parts.append(part)
    return pd.concat(parts, ignore_index=True, sort=False)


def _read_human(existing_path: Path, balanced_path: Path) -> pd.DataFrame:
    if not existing_path.is_file():
        raise FileNotFoundError(existing_path)
    if not balanced_path.is_file():
        raise FileNotFoundError(
            f"Missing balanced sweep {balanced_path}; run "
            "comparison/human_cerebral/evaluate_human_cerebral_balanced_cy_sweep_all_time.py"
        )
    existing = pd.read_csv(existing_path)
    balanced = pd.read_csv(balanced_path)
    required_existing = {"method", "D_RNA", "D_ATAC"}
    required_balanced = {"method", "cy", "D_RNA", "D_ATAC"}
    if required_existing.difference(existing.columns):
        raise ValueError(f"Unexpected Human cerebral table: {existing_path}")
    if required_balanced.difference(balanced.columns):
        raise ValueError(f"Unexpected Human balanced table: {balanced_path}")

    rows: list[dict[str, object]] = []
    for _, item in existing.iterrows():
        name = str(item["method"])
        cy = np.nan
        if name.startswith("COATI unbal C_y="):
            cy = float(name.rsplit("=", 1)[1])
            method = "COATI unbal"
        elif name == "UOT(RNA)":
            method = "UOT baseline RNA"
        elif name == "UOT(ATAC)":
            method = "UOT baseline ATAC"
        elif name == "OT(RNA)":
            method = "OT baseline RNA"
        elif name == "OT(ATAC)":
            method = "OT baseline ATAC"
        elif name == "COATI bal":
            # Replaced below by the complete independently evaluated C_y sweep.
            continue
        else:
            raise ValueError(f"Unexpected Human cerebral method: {name}")
        rows.append(
            {
                "dataset": "Human cerebral",
                "stage": "All times",
                "method": method,
                "cy": cy,
                "rna_sinkhorn": float(item["D_RNA"]),
                "atac_sinkhorn": float(item["D_ATAC"]),
                "source_csv": str(existing_path.resolve()),
            }
        )
    for _, item in balanced.iterrows():
        rows.append(
            {
                "dataset": "Human cerebral",
                "stage": "All times",
                "method": "COATI bal",
                "cy": float(item["cy"]),
                "rna_sinkhorn": float(item["D_RNA"]),
                "atac_sinkhorn": float(item["D_ATAC"]),
                "source_csv": str(balanced_path.resolve()),
            }
        )
    return pd.DataFrame(rows)


def load_all(args: argparse.Namespace) -> pd.DataFrame:
    table = pd.concat(
        [
            _read_standard_pair(
                args.pancreas_unbalanced, args.pancreas_balanced, "Pancreas"
            ),
            _read_human(args.human_existing, args.human_balanced_sweep),
            _read_standard_pair(
                args.palate_unbalanced, args.palate_balanced, "Palate"
            ),
            _read_standard_pair(
                args.gastrulation_unbalanced,
                args.gastrulation_balanced,
                "Gastrulation",
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    table["rna_sinkhorn"] = pd.to_numeric(table["rna_sinkhorn"], errors="raise")
    table["atac_sinkhorn"] = pd.to_numeric(table["atac_sinkhorn"], errors="raise")
    if not np.isfinite(table[["rna_sinkhorn", "atac_sinkhorn"]].to_numpy()).all():
        raise ValueError("Non-finite synchronization errors")
    validate(table)
    return table


def validate(table: pd.DataFrame) -> None:
    expected_stages = {
        "Pancreas": ("E15.5", "E16.5"),
        "Human cerebral": ("All times",),
        "Palate": ("E13.5", "E14.0", "E14.5"),
        "Gastrulation": ("E8.0", "E8.5", "E8.75"),
    }
    for dataset, stages in expected_stages.items():
        subset = table[table["dataset"] == dataset]
        if set(subset["stage"]) != set(stages):
            raise ValueError(f"{dataset}: unexpected stages {sorted(subset['stage'].unique())}")
        for stage in stages:
            panel = subset[subset["stage"] == stage]
            for method in METHOD_ORDER:
                count = int((panel["method"] == method).sum())
                expected = 1
                if method == "COATI bal":
                    expected = 9
                elif method == "COATI unbal":
                    expected = 8 if dataset == "Human cerebral" else 9
                if count != expected:
                    raise ValueError(
                        f"{dataset} {stage} {method}: expected {expected}, found {count}"
                    )


def _style(method: str) -> dict[str, object]:
    if method == "COATI unbal":
        style = method_style("COATI unbalanced")
        return {"marker": style.marker, "face": style.color, "edge": "white"}
    if method == "COATI bal":
        style = method_style("COATI balanced")
        return {"marker": style.marker, "face": style.color, "edge": "white"}
    if method in {"UOT baseline RNA", "OT baseline RNA"}:
        color = NATURE_CUD["bluish_green"]
        return {
            "marker": "^",
            "face": color if method.startswith("UOT") else "none",
            "edge": color,
        }
    if method in {"UOT baseline ATAC", "OT baseline ATAC"}:
        color = NATURE_CUD["reddish_purple"]
        return {
            "marker": "s",
            "face": color if method.startswith("UOT") else "none",
            "edge": color,
        }
    raise KeyError(method)


def _plot_panel(ax: plt.Axes, panel: pd.DataFrame, family: str) -> None:
    if family == "unbalanced":
        coupled = "COATI unbal"
        baselines = ("UOT baseline RNA", "UOT baseline ATAC")
    else:
        coupled = "COATI bal"
        baselines = ("OT baseline RNA", "OT baseline ATAC")

    sweep = panel[panel["method"] == coupled].sort_values("cy")
    style = _style(coupled)
    ax.scatter(
        sweep["rna_sinkhorn"],
        sweep["atac_sinkhorn"],
        s=20,
        marker=style["marker"],
        facecolor=style["face"],
        edgecolor=style["edge"],
        linewidth=0.55,
        zorder=4,
    )
    for method in baselines:
        point = panel[panel["method"] == method].iloc[0]
        style = _style(method)
        ax.scatter(
            float(point["rna_sinkhorn"]),
            float(point["atac_sinkhorn"]),
            s=43,
            marker=style["marker"],
            facecolor=style["face"],
            edgecolor=style["edge"],
            linewidth=1.0,
            zorder=6,
        )

    selected_methods = (coupled, *baselines)
    visible = panel[panel["method"].isin(selected_methods)]
    x_values = visible["rna_sinkhorn"].to_numpy(float)
    y_values = visible["atac_sinkhorn"].to_numpy(float)
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    x_pad = max(0.10 * (x_max - x_min), 5e-4)
    y_pad = max(0.12 * (y_max - y_min), 7e-4)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel(r"$D_{\mathrm{RNA}}$", labelpad=1.5)
    ax.set_ylabel(r"$D_{\mathrm{ATAC}}$", labelpad=1.5)
    ax.grid(color="#D9D9D9", linewidth=0.5, alpha=0.7)
    ax.tick_params(direction="out", pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)


def _legend_handles() -> list[Line2D]:
    handles = []
    for method in METHOD_ORDER:
        style = _style(method)
        handles.append(
            Line2D(
                [0],
                [0],
                color="none",
                marker=style["marker"],
                markerfacecolor=style["face"],
                markeredgecolor=style["edge"],
                markeredgewidth=1.0,
                markersize=5.4,
                label=DISPLAY_LABELS.get(method, method),
            )
        )
    return handles


def _experiment_title(
    fig: plt.Figure,
    axes: list[plt.Axes],
    title: str,
    *,
    offset: float = 0.043,
) -> None:
    left = min(ax.get_position().x0 for ax in axes)
    right = max(ax.get_position().x1 for ax in axes)
    top = max(ax.get_position().y1 for ax in axes)
    fig.text(
        0.5 * (left + right),
        top + offset,
        title,
        ha="center",
        va="bottom",
        fontsize=10.0,
        fontweight="normal",
    )


def _row_label(fig: plt.Figure, ax: plt.Axes, text: str) -> None:
    box = ax.get_position()
    fig.text(
        box.x0 - 0.108,
        0.5 * (box.y0 + box.y1),
        text,
        rotation=90,
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="normal",
    )


def plot(table: pd.DataFrame, output_dir: Path) -> list[Path]:
    apply_nature_rc(font_size=10.0)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "font.weight": "normal",
            "axes.titlesize": 10.0,
            "axes.titleweight": "normal",
            "axes.labelsize": 10.0,
            "axes.labelweight": "normal",
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.0,
            "figure.titleweight": "normal",
        }
    )
    fig = plt.figure(figsize=(7.2, 10.5))
    grid = fig.add_gridspec(
        8,
        3,
        height_ratios=(1.0, 1.0, 0.34, 1.0, 1.0, 0.34, 1.0, 1.0),
    )
    axes = np.empty((6, 3), dtype=object)
    for logical_row, grid_row in enumerate((0, 1, 3, 4, 6, 7)):
        for column in range(3):
            axes[logical_row, column] = fig.add_subplot(grid[grid_row, column])

    blocks = (
        (0, "Gastrulation", ("E8.0", "E8.5", "E8.75")),
        (2, "Palate", ("E13.5", "E14.0", "E14.5")),
    )
    for start_row, dataset, stages in blocks:
        for column, stage in enumerate(stages):
            panel = table[
                (table["dataset"] == dataset) & (table["stage"] == stage)
            ]
            _plot_panel(axes[start_row, column], panel, "unbalanced")
            _plot_panel(axes[start_row + 1, column], panel, "balanced")
            axes[start_row, column].set_title(stage, pad=2.5, fontweight="normal")

    # Final block: Pancreas in columns 1--2, Human cerebral in column 3.
    bottom_specs = (
        ("Pancreas", "E15.5"),
        ("Pancreas", "E16.5"),
        ("Human cerebral", "All times"),
    )
    for column, (dataset, stage) in enumerate(bottom_specs):
        panel = table[(table["dataset"] == dataset) & (table["stage"] == stage)]
        _plot_panel(axes[4, column], panel, "unbalanced")
        _plot_panel(axes[5, column], panel, "balanced")
        axes[4, column].set_title(stage, pad=2.5, fontweight="normal")

    fig.subplots_adjust(
        left=0.14,
        right=0.99,
        top=0.945,
        bottom=0.095,
        hspace=0.56,
        wspace=0.48,
    )
    _experiment_title(fig, list(axes[0, :]), "Gastrulation", offset=0.034)
    _experiment_title(fig, list(axes[2, :]), "Palate", offset=0.034)
    _experiment_title(fig, [axes[4, 0], axes[4, 1]], "Pancreas", offset=0.034)
    _experiment_title(fig, [axes[4, 2]], "Human cerebral", offset=0.034)

    for start_row in (0, 2, 4):
        _row_label(fig, axes[start_row, 0], "COATI unbal")
        _row_label(fig, axes[start_row + 1, 0], "COATI bal")

    fig.legend(
        handles=_legend_handles(),
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.012),
        ncol=3,
        columnspacing=1.05,
        handletextpad=0.38,
    )

    stem = output_dir / "combined_synchronization_ablation"
    outputs = [stem.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg")]
    fig.savefig(outputs[0], dpi=600, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], bbox_inches="tight")
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        args.output_dir / f"combined_synchronization_ablation.{suffix}"
        for suffix in ("png", "pdf", "svg")
    ]
    combined_csv = args.output_dir / "combined_synchronization_ablation_points.csv"
    manifest_path = args.output_dir / "combined_synchronization_ablation_manifest.json"
    if not args.overwrite and any(
        path.exists() for path in (*outputs, combined_csv, manifest_path)
    ):
        raise FileExistsError(
            f"Outputs exist in {args.output_dir}; pass --overwrite to replace them"
        )

    table = load_all(args)
    table.to_csv(combined_csv, index=False)
    rendered = plot(table, args.output_dir)
    manifest = {
        "layout": {
            "first_block": ["Gastrulation"],
            "second_block": ["Palate"],
            "third_block": ["Pancreas", "Human cerebral"],
            "rows_per_block": ["COATI unbal", "COATI bal"],
            "panel_labels": False,
            "shared_legend": True,
            "typography": "all text is 10 pt Arial",
        },
        "point_counts": {
            f"{dataset} | {stage} | {method}": int(len(group))
            for (dataset, stage, method), group in table.groupby(
                ["dataset", "stage", "method"], sort=True
            )
        },
        "human_cerebral": {
            "unbalanced_cy": "0.1--0.8, 40k checkpoints",
            "balanced_cy": "0.1--0.9, 30k checkpoints",
            "quantity": "sum of per-age synchronization errors",
        },
        "pancreas_atac_only_rna_readout": "frozen full ATAC-to-RNA FiLM",
        "sources": sorted(set(table["source_csv"].astype(str))),
        "outputs": [str(path.resolve()) for path in rendered]
        + [str(combined_csv.resolve())],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Validated and plotted {len(table)} points.")
    for path in (*rendered, combined_csv, manifest_path):
        print(path)


if __name__ == "__main__":
    main()
