#!/usr/bin/env python
"""Plot compact Gastrulation LOO metrics with base TrajectoryNet and RNA-only."""

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
import numpy as np
import pandas as pd

from trainfbench_plot_style import apply_nature_rc, method_style


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = (
    ROOT
    / "results"
    / "gastrulation_loo_shared_t_trajectorynet_base_rna_only_floor"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR

COVERAGE_METHODS = (
    ("BSOT C_y=0.3", "COATI bal", "COATI balanced"),
    ("USOT C_y=0.3", "COATI unbal", "COATI unbalanced"),
    ("BSOT RNA-only (no Sync)", "RNA-only bal", "COATI balanced"),
    ("USOT RNA-only (no Sync)", "RNA-only unbal", "COATI unbalanced"),
    ("CytoBridge balanced 20k", "CytoBridge bal", "CytoBridge balanced"),
    ("CytoBridge unbalanced 20k", "CytoBridge unbal", "CytoBridge unbalanced"),
    ("MIOFlow 20k", "MIOFlow", "MIOFlow"),
    ("TIGON 20k", "TIGON", "TIGON"),
    ("TrajectoryNet 20k", "TrajectoryNet", "TrajectoryNet"),
)
W2_METHODS = tuple(
    record for record in COVERAGE_METHODS if record[1] != "MIOFlow"
)
FLOOR_COLOR = "#B3B3B3"
FLOOR_EDGE = "#666666"
RNA_ONLY_METHODS = {
    "BSOT RNA-only (no Sync)",
    "USOT RNA-only (no Sync)",
}


@dataclass(frozen=True)
class Metric:
    column: str
    slug: str
    title: str
    decimals: int = 3

    @property
    def is_w2(self) -> bool:
        return self.column.endswith("sinkhorn_divergence")

    @property
    def modality(self) -> str:
        return self.column.split("_", maxsplit=1)[0].upper()


METRICS = (
    Metric("rna_effective_coverage", "rna_coverage", "RNA cov. ↑"),
    Metric("atac_effective_coverage", "atac_coverage", "ATAC cov. ↑"),
    Metric("rna_sinkhorn_divergence", "rna_w2", r"RNA $W_2$ ↓"),
    Metric("atac_sinkhorn_divergence", "atac_w2", r"ATAC $W_2$ ↓"),
    Metric("rna_celltype_composition_jsd", "rna_jsd", "RNA comp.\nJSD ↓"),
    Metric("atac_celltype_composition_jsd", "atac_jsd", "ATAC comp.\nJSD ↓"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def display_value(metric: Metric, value: float) -> float:
    if metric.is_w2:
        return float(np.sqrt(2.0 * max(value, 0.0)))
    return float(value)


def method_row(frame: pd.DataFrame, source_method: str) -> pd.Series:
    rows = frame[frame["method"].eq(source_method)]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one row for {source_method}; found {len(rows)}"
        )
    return rows.iloc[0]


def floor_value(
    floor_frame: pd.DataFrame,
    *,
    stage: str,
    modality: str,
) -> float:
    rows = floor_frame[
        floor_frame["stage"].eq(stage)
        & floor_frame["modality"].eq(modality)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one sampling floor for {stage} {modality}; "
            f"found {len(rows)}"
        )
    return float(rows.iloc[0]["mean_w2"])


def plot_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    floor_frame: pd.DataFrame,
    metric: Metric,
    *,
    stage: str,
    show_ylabels: bool,
    exclude_displays: frozenset[str] = frozenset(),
) -> list[dict[str, object]]:
    methods = W2_METHODS if metric.is_w2 else COVERAGE_METHODS
    methods = tuple(record for record in methods if record[1] not in exclude_displays)
    available = set(frame["method"].astype(str))
    methods = tuple(record for record in methods if record[0] in available)
    records: list[dict[str, object]] = []
    for source, display, style_name in methods:
        row = method_row(frame, source)
        style = method_style(style_name)
        records.append(
            {
                "source": source,
                "display": display,
                "row": row,
                "value": display_value(metric, float(row[metric.column])),
                "color": style.color,
                "edge": style.markeredgecolor or style.color,
                "is_rna_only": source in RNA_ONLY_METHODS,
                "definition": (
                    "W2=sqrt(2*debiased_sinkhorn_divergence)"
                    if metric.is_w2
                    else "identity"
                ),
            }
        )
    if metric.is_w2:
        records.append(
            {
                "source": "Reference self-sampling",
                "display": "Sampling floor",
                "row": None,
                "value": floor_value(
                    floor_frame,
                    stage=stage,
                    modality=metric.modality,
                ),
                "color": FLOOR_COLOR,
                "edge": FLOOR_EDGE,
                "is_rna_only": False,
                "definition": "mean W2 of repeated disjoint real-data half-splits",
            }
        )

    values = np.asarray([float(record["value"]) for record in records])
    positions = np.arange(len(records))
    maximum = max(float(values.max()), np.finfo(float).eps)
    relative_values = 0.66 * values / maximum
    for position, width, record in zip(positions, relative_values, records):
        is_rna_only = bool(record["is_rna_only"])
        ax.barh(
            position,
            width,
            height=0.54,
            color=("white" if is_rna_only else str(record["color"])),
            edgecolor=str(record["edge"]),
            linewidth=(0.9 if is_rna_only else 0.65),
            hatch=("///" if is_rna_only else None),
        )

    value_column_x = 1.16
    ax.set_xlim(0.0, 1.19)
    ax.set_ylim(len(records) - 0.35, -0.65)
    ax.set_yticks(
        positions,
        [str(record["display"]) for record in records]
        if show_ylabels
        else [],
    )
    ax.set_title(
        metric.title,
        fontsize=12,
        fontweight="normal",
        pad=3,
        x=0.44 if not show_ylabels else 0.50,
    )
    ax.set_xticks([])
    ax.tick_params(
        axis="x",
        labelbottom=False,
        width=0.0,
        length=0.0,
        pad=0,
    )
    ax.tick_params(axis="y", labelsize=10, length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for position, value in zip(positions, values):
        ax.text(
            value_column_x,
            position,
            f"{value:.{metric.decimals}f}",
            ha="right",
            va="center",
            color="black",
            fontsize=10,
            fontweight="normal",
            clip_on=False,
        )

    return [
        {
            "stage": stage,
            "metric": metric.slug,
            "display_method": str(record["display"]),
            "source_method": str(record["source"]),
            "source_value": (
                float(record["row"][metric.column])
                if record["row"] is not None
                else np.nan
            ),
            "display_value": float(record["value"]),
            "display_definition": str(record["definition"]),
            "bar_definition": (
                "66% of display_value divided by the maximum display value "
                "within the panel"
            ),
        }
        for record in records
    ]


def output_paths(output_dir: Path, stem: str) -> tuple[Path, ...]:
    return tuple(
        (output_dir / stem).with_suffix(suffix)
        for suffix in (".png", ".pdf", ".svg", ".csv")
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    floor_frame = pd.read_csv(args.input_dir / "sampling_floor_summary.csv")
    apply_nature_rc(font_size=10.0)
    mpl.rcParams.update(
        {
            "axes.titlesize": 12.0,
            "axes.labelsize": 12.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.0,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "figure.titleweight": "normal",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
        }
    )

    for scenario, stage, stage_tag in (
        ("loo_time1", "E8.0", "e80"),
        ("loo_time2", "E8.5", "e85"),
    ):
        input_path = (
            args.input_dir
            / f"{scenario}_shared_t_scores_trajectorynet_base_rna_only.csv"
        )
        frame = pd.read_csv(input_path)
        stem = (
            f"gastrulation_loo_{stage_tag}_six_metrics_"
            "cy0p3_trajectorynet_base_rna_only_with_floor"
        )
        outputs = output_paths(args.output_dir, stem)
        existing = [path for path in outputs if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                "Refusing to overwrite existing outputs; pass --overwrite:\n"
                + "\n".join(str(path) for path in existing)
            )

        fig, axes = plt.subplots(
            3,
            2,
            figsize=(3.75, 4.85),
            facecolor="white",
        )
        rows: list[dict[str, object]] = []
        for index, (ax, metric) in enumerate(zip(axes.ravel(), METRICS)):
            rows.extend(
                plot_panel(
                    ax,
                    frame,
                    floor_frame,
                    metric,
                    stage=stage,
                    show_ylabels=index % 2 == 0,
                )
            )
        fig.suptitle(
            f"Held-out {stage}",
            fontsize=12,
            fontweight="normal",
            y=0.995,
        )
        fig.subplots_adjust(
            left=0.32,
            right=0.995,
            top=0.91,
            bottom=0.025,
            hspace=0.48,
            wspace=0.16,
        )
        png_path, pdf_path, svg_path, csv_path = outputs
        save_options = {
            "facecolor": "white",
            "bbox_inches": "tight",
            "pad_inches": 0.02,
        }
        fig.savefig(png_path, dpi=600, **save_options)
        fig.savefig(pdf_path, **save_options)
        fig.savefig(svg_path, **save_options)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        plt.close(fig)
        for path in outputs:
            print(path)


if __name__ == "__main__":
    main()
