#!/usr/bin/env python
"""Plot the strict-LOO synchronization ablation as two compact matrices."""

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
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from trainfbench_plot_style import NATURE_CUD, apply_nature_rc


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "results/gastrulation_strict_loo_six_metrics_20k_cy0p3"
    / "strict_loo_e80_e85_six_metric_scores.csv"
)
DEFAULT_OUTPUT = ROOT / "results/gastrulation_strict_loo_sync_ablation_cy0p3"


@dataclass(frozen=True)
class Metric:
    column: str
    label: str
    higher_is_better: bool

    @property
    def is_w2(self) -> bool:
        return self.column.endswith("sinkhorn_divergence")


METRICS = (
    Metric("rna_effective_coverage", "RNA coverage ↑", True),
    Metric("atac_effective_coverage", "ATAC coverage ↑", True),
    Metric("rna_sinkhorn_divergence", r"RNA $W_2$ ↓", False),
    Metric("atac_sinkhorn_divergence", r"ATAC $W_2$ ↓", False),
    Metric("rna_celltype_composition_jsd", "RNA comp. JSD ↓", False),
    Metric("atac_celltype_composition_jsd", "ATAC comp. JSD ↓", False),
)

STAGES = ("E8.0", "E8.5")
METHODS = (
    ("OT(RNA)", "OT", "Balanced", False),
    ("COATI balanced", "COATI bal.", "Balanced", True),
    ("UOT(RNA)", "UOT", "Unbalanced", False),
    ("COATI unbalanced", "COATI unbal.", "Unbalanced", True),
)

PAIR_BASELINE = {
    "COATI balanced": "OT(RNA)",
    "COATI unbalanced": "UOT(RNA)",
}

COLORS = {
    "balanced": NATURE_CUD["blue"],
    "unbalanced": NATURE_CUD["vermillion"],
    "balanced_fill": "#E4F1F8",
    "unbalanced_fill": "#FAE9E2",
    "baseline_fill": "#F7F7F7",
    "grid": "#D9D9D9",
    "negative": "#6B6B6B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def display_value(metric: Metric, value: float) -> float:
    if metric.is_w2:
        return float(np.sqrt(2.0 * max(value, 0.0)))
    return float(value)


def get_value(
    frame: pd.DataFrame, *, stage: str, method: str, metric: Metric
) -> float:
    row = frame.loc[frame["stage"].eq(stage) & frame["method"].eq(method)]
    if len(row) != 1:
        raise ValueError(
            f"Expected one row for stage={stage}, method={method}; found {len(row)}"
        )
    return display_value(metric, float(row.iloc[0][metric.column]))


def favorable_delta(metric: Metric, baseline: float, coati: float) -> float:
    delta = coati - baseline if metric.higher_is_better else baseline - coati
    return 0.0 if abs(delta) < 5e-7 else float(delta)


def draw_matrix(
    ax: plt.Axes,
    frame: pd.DataFrame,
    stage: str,
    *,
    show_metric_labels: bool,
) -> list[dict[str, object]]:
    n_rows = len(METRICS)
    n_cols = len(METHODS)
    rows: list[dict[str, object]] = []

    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, -1.42)
    ax.axis("off")

    ax.text(
        n_cols / 2,
        -1.22,
        f"Held-out {stage}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="normal",
    )
    ax.text(1.0, -0.86, "Balanced", ha="center", va="center", fontsize=10)
    ax.text(3.0, -0.86, "Unbalanced", ha="center", va="center", fontsize=10)
    ax.plot([2, 2], [-0.98, n_rows], color="#A8A8A8", linewidth=0.9)

    for column, (_, display, regime, is_coati) in enumerate(METHODS):
        color = COLORS["balanced"] if regime == "Balanced" else COLORS["unbalanced"]
        ax.text(
            column + 0.5,
            -0.36,
            display,
            ha="center",
            va="center",
            fontsize=10,
            color=color if is_coati else "black",
        )

    for row_index, metric in enumerate(METRICS):
        if show_metric_labels:
            ax.text(
                -0.12,
                row_index + 0.5,
                metric.label,
                ha="right",
                va="center",
                fontsize=10,
                clip_on=False,
            )
        for column, (method, display, regime, is_coati) in enumerate(METHODS):
            fill = COLORS["baseline_fill"]
            if is_coati:
                fill = (
                    COLORS["balanced_fill"]
                    if regime == "Balanced"
                    else COLORS["unbalanced_fill"]
                )
            ax.add_patch(
                Rectangle(
                    (column, row_index),
                    1,
                    1,
                    facecolor=fill,
                    edgecolor=COLORS["grid"],
                    linewidth=0.65,
                )
            )
            value = get_value(frame, stage=stage, method=method, metric=metric)
            if not is_coati:
                ax.text(
                    column + 0.5,
                    row_index + 0.5,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                )
                delta = np.nan
                baseline = np.nan
            else:
                baseline_method = PAIR_BASELINE[method]
                baseline = get_value(
                    frame,
                    stage=stage,
                    method=baseline_method,
                    metric=metric,
                )
                delta = favorable_delta(metric, baseline, value)
                ax.text(
                    column + 0.5,
                    row_index + 0.5,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                )

            rows.append(
                {
                    "stage": stage,
                    "metric": metric.column,
                    "display_metric": metric.label,
                    "method": method,
                    "display_method": display,
                    "value": value,
                    "paired_baseline_value": baseline,
                    "favorable_delta": delta,
                }
            )

    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "gastrulation_strict_loo_sync_ablation_matrix_cy0p3"
    outputs = [stem.with_suffix(ext) for ext in (".png", ".pdf", ".svg", ".csv")]
    if not args.overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite outputs; pass --overwrite:\n"
                + "\n".join(str(path) for path in existing)
            )

    frame = pd.read_csv(args.input)
    apply_nature_rc(font_size=10.0)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "font.weight": "normal",
            "axes.titlesize": 12.0,
            "axes.titleweight": "normal",
            "axes.labelsize": 12.0,
            "axes.labelweight": "normal",
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.0,
            "figure.titleweight": "normal",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.65, 3.40), facecolor="white")
    rows: list[dict[str, object]] = []
    for index, (ax, stage) in enumerate(zip(axes, STAGES)):
        rows.extend(
            draw_matrix(
                ax,
                frame,
                stage,
                show_metric_labels=index == 0,
            )
        )

    fig.suptitle("Cross-modal synchronization ablation", fontsize=12, y=0.985)
    fig.subplots_adjust(
        left=0.255,
        right=0.995,
        top=0.91,
        bottom=0.035,
        wspace=0.075,
    )

    options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.025}
    fig.savefig(stem.with_suffix(".png"), dpi=600, **options)
    fig.savefig(stem.with_suffix(".pdf"), **options)
    fig.savefig(stem.with_suffix(".svg"), **options)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(stem.with_suffix(".csv"), index=False)
    print(f"Saved {stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
