#!/usr/bin/env python
"""Plot standalone endocrine stress and mass-decomposition figures."""

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
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results/pancreas_external_stress_support_cy05_20k"
DEFAULT_MASS = (
    ROOT
    / "results/pancreas_full_unbalanced_delta_epsilon_mass_change_decomposition_15nn_20k_cy05"
    / "celltype_mass_change_decomposition.csv"
)
CELL_TYPES = ("Alpha", "Beta", "Delta")
METHODS = ("COATI", "CytoBridge")
MODULES = ("Apoptosis", "p53 pathway", "UPR", "ROS pathway")
COLORS = {
    "retained": "#0072B2",
    "incoming": "#E69F00",
    "outgoing": "#CC79A7",
    "observed": "#B2182B",
    "predicted": "#222222",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--mass", type=Path, default=DEFAULT_MASS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "axes.titleweight": "normal",
            "font.weight": "normal",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def stress_changes(observed: pd.DataFrame) -> np.ndarray:
    pivot = observed.pivot_table(
        index="celltype", columns=["module_short", "stage"], values="score_mean"
    )
    return np.asarray(
        [
            [
                pivot.loc[celltype, (module, "E16.5")]
                - pivot.loc[celltype, (module, "E15.5")]
                for module in MODULES
            ]
            for celltype in CELL_TYPES
        ],
        dtype=float,
    )


def plot_stress(ax: plt.Axes, observed: pd.DataFrame) -> None:
    values = stress_changes(observed)
    limit = 0.08
    image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(MODULES)), ["Apoptosis", "p53", "UPR", "ROS"])
    ax.set_yticks(range(len(CELL_TYPES)), CELL_TYPES)
    ax.tick_params(length=0)
    ax.set_title(
        "Observed stress-program change\nE15.5→E16.5",
        loc="left",
        weight="bold",
        pad=8,
    )

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            color = "white" if abs(value) > 0.055 else "#222222"
            ax.text(
                column,
                row,
                f"{value:+.3f}".replace("-", "−"),
                ha="center",
                va="center",
                color=color,
                fontsize=8.5,
            )

    for edge in np.arange(-0.5, len(MODULES), 1.0):
        ax.axvline(edge, color="white", linewidth=1.0)
    for edge in np.arange(-0.5, len(CELL_TYPES), 1.0):
        ax.axhline(edge, color="white", linewidth=1.0)
    colorbar = ax.figure.colorbar(
        image, ax=ax, orientation="horizontal", fraction=0.08, pad=0.16, aspect=22
    )
    colorbar.set_label("Standardized score change", fontsize=9)
    colorbar.set_ticks([-0.08, 0.0, 0.08])
    colorbar.ax.tick_params(labelsize=8, length=2)


def signed_stack(
    ax: plt.Axes,
    y: float,
    components: list[tuple[float, str]],
    height: float = 0.52,
) -> None:
    positive = 0.0
    negative = 0.0
    for value, name in components:
        color = COLORS[name]
        if value >= 0:
            ax.barh(
                y, value, left=positive, height=height, color=color,
                edgecolor="white", linewidth=0.5, zorder=2,
            )
            positive += value
        else:
            ax.barh(
                y, value, left=negative, height=height, color=color,
                edgecolor="white", linewidth=0.5, zorder=2,
            )
            negative += value


def interval_summary(mass: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Convert the long decomposition table into one plotting row per cell type."""
    selected = mass[
        mass["weighting"].eq("composition_relative")
        & mass["interval"].eq(interval)
        & mass["celltype"].isin(CELL_TYPES)
        & mass["method"].isin(("COATI unbalanced", "CytoBridge unbalanced"))
    ].copy()
    if len(selected) != len(CELL_TYPES) * len(METHODS):
        raise ValueError(
            f"Expected {len(CELL_TYPES) * len(METHODS)} rows for {interval}; "
            f"found {len(selected)}"
        )

    records: list[dict[str, float | str]] = []
    for celltype in CELL_TYPES:
        record: dict[str, float | str] = {"celltype": celltype}
        cell_rows = selected[selected["celltype"].eq(celltype)]
        record["observed_composition_delta_pp"] = (
            100.0 * float(cell_rows["observed_delta_fraction"].iloc[0])
        )
        for method in METHODS:
            method_name = f"{method} unbalanced"
            row = cell_rows[cell_rows["method"].eq(method_name)].iloc[0]
            prefix = method.lower()
            record[f"{prefix}_retained_reweight_pp"] = 100.0 * float(
                row["retained_reweight"]
            )
            record[f"{prefix}_incoming_pp"] = 100.0 * float(
                row["incoming_final_mass"]
            )
            record[f"{prefix}_outgoing_pp"] = 100.0 * float(
                row["outgoing_initial_mass"]
            )
            record[f"{prefix}_net_delta_pp"] = 100.0 * float(row["delta_mass"])
        records.append(record)
    return pd.DataFrame.from_records(records)


def mass_legend_handles() -> list[Patch | Line2D]:
    return [
        Patch(facecolor=COLORS["retained"], label="Reweight."),
        Patch(facecolor=COLORS["incoming"], label="Incoming"),
        Patch(facecolor=COLORS["outgoing"], label="Outgoing"),
        Line2D(
            [], [], marker="o", linestyle="", markerfacecolor=COLORS["predicted"],
            markeredgecolor="white", markersize=5.5, label="Pred.",
        ),
        Line2D(
            [], [], marker="D", linestyle="", markerfacecolor="white",
            markeredgecolor=COLORS["observed"], markeredgewidth=1.2,
            markersize=5.5, label="Obs.",
        ),
    ]


def plot_mass(
    ax: plt.Axes,
    summary: pd.DataFrame,
    interval: str,
    *,
    show_legend: bool = True,
    show_xlabel: bool = True,
    legend_position: str = "bottom",
) -> None:
    positions: list[float] = []
    labels: list[str] = []
    current = 5.5
    for celltype in CELL_TYPES:
        for method in METHODS:
            row = summary[summary["celltype"].eq(celltype)].iloc[0]
            prefix = method.lower()
            retained = float(row[f"{prefix}_retained_reweight_pp"])
            incoming = float(row[f"{prefix}_incoming_pp"])
            outgoing = -float(row[f"{prefix}_outgoing_pp"])
            predicted = float(row[f"{prefix}_net_delta_pp"])
            observed = float(row["observed_composition_delta_pp"])
            signed_stack(
                ax,
                current,
                [(retained, "retained"), (incoming, "incoming"), (outgoing, "outgoing")],
            )
            ax.scatter(
                predicted, current, s=28, marker="o", color=COLORS["predicted"],
                edgecolor="white", linewidth=0.5, zorder=4,
            )
            ax.scatter(
                observed, current, s=38, marker="D", facecolor="white",
                edgecolor=COLORS["observed"], linewidth=1.4, zorder=5,
            )
            positions.append(current)
            labels.append(f"{celltype} · {method}")
            current -= 0.75
        current -= 0.30

    # Fit the horizontal range to the actual stacks and markers.  The earlier
    # fixed -1.25..6.65 range left substantial unused space, especially in the
    # E14.5-to-E15.5 panel.
    horizontal_values: list[float] = []
    for _, row in summary.iterrows():
        observed = float(row["observed_composition_delta_pp"])
        horizontal_values.append(observed)
        for method in METHODS:
            prefix = method.lower()
            retained = float(row[f"{prefix}_retained_reweight_pp"])
            incoming = float(row[f"{prefix}_incoming_pp"])
            outgoing = float(row[f"{prefix}_outgoing_pp"])
            predicted = float(row[f"{prefix}_net_delta_pp"])
            horizontal_values.extend(
                [
                    min(retained, 0.0) - outgoing,
                    max(retained, 0.0) + incoming,
                    predicted,
                ]
            )
    lower = min(-0.35, min(horizontal_values) - 0.25)
    # Six percentage points is sufficient for both intervals.  Keep the
    # data-driven compact limit for the early interval, but never extend the
    # axis beyond 6 in the late interval.
    upper = min(6.0, max(1.0, max(horizontal_values) + 0.28))

    ax.axvline(0.0, color="#555555", linewidth=0.8, zorder=1)
    ax.set_yticks(positions, labels)
    ax.set_xlim(lower, upper)
    ax.set_ylim(min(positions) - 0.55, max(positions) + 0.55)
    tick_start = int(np.floor(lower))
    tick_stop = int(np.ceil(upper))
    ax.set_xticks(np.arange(tick_start, tick_stop + 1, 1))
    if show_xlabel:
        ax.set_xlabel("Contribution (percentage)")
        # Centre the label under the plotting area.  The right-legend layout
        # needs a small visual correction because its saved bounding box also
        # includes the legend.
        label_x = 0.40 if legend_position == "right" else 0.50
        ax.xaxis.set_label_coords(label_x, -0.15)
    ax.grid(axis="x", color="#E1E1E1", linewidth=0.6, zorder=0)
    ax.set_title(
        interval,
        loc="left", weight="normal", pad=5,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if show_legend:
        if legend_position == "right":
            ax.legend(
                handles=mass_legend_handles(), loc="center left",
                bbox_to_anchor=(1.005, 0.50), ncol=1, frameon=False,
                handlelength=0.75, handletextpad=0.35,
                labelspacing=0.45, borderaxespad=0.0,
            )
        else:
            ax.legend(
                handles=mass_legend_handles(), loc="upper center",
                bbox_to_anchor=(0.50, -0.34), ncol=3, frameon=False,
                handlelength=0.75, columnspacing=0.40, handletextpad=0.30,
                borderaxespad=0.0,
            )


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    outputs = [stem.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg")]
    for path in outputs:
        fig.savefig(
            path,
            dpi=360 if path.suffix == ".png" else None,
            bbox_inches="tight",
            pad_inches=0.01,
            facecolor="white",
        )
    return outputs


def main() -> None:
    args = parse_args()
    observed_path = args.results_dir / "observed_external_stress_scores.csv"
    observed = pd.read_csv(observed_path)
    mass = pd.read_csv(args.mass)

    stems = (
        args.output_dir / "pancreas_observed_stress_change_e155_e165",
        args.output_dir / "pancreas_mass_decomposition_e145_e155",
        args.output_dir / "pancreas_mass_decomposition_e155_e165",
        args.output_dir / "pancreas_mass_decomposition_both_intervals_stacked",
    )
    targets = [
        stem.with_suffix(suffix)
        for stem in stems
        for suffix in (".png", ".pdf", ".svg")
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configure_plotting()
    fig, ax = plt.subplots(figsize=(3.75, 3.20))
    plot_stress(ax, observed)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.82, bottom=0.28)
    stress_outputs = save_figure(fig, stems[0])
    plt.close(fig)

    outputs = list(stress_outputs)
    for interval, stem in zip(
        ("E14.5→E15.5", "E15.5→E16.5"), stems[1:3], strict=True
    ):
        summary = interval_summary(mass, interval)
        # Keep the mass panels comfortably below half an A4 page width while
        # preserving the fixed 10 pt typography used throughout the figures.
        fig, ax = plt.subplots(figsize=(3.88, 2.85))
        plot_mass(ax, summary, interval, legend_position="bottom")
        fig.subplots_adjust(left=0.315, right=0.985, top=0.91, bottom=0.36)
        outputs.extend(save_figure(fig, stem))
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(3.05, 4.75))
    for axis, interval in zip(
        axes, ("E14.5→E15.5", "E15.5→E16.5"), strict=True
    ):
        plot_mass(
            axis,
            interval_summary(mass, interval),
            interval,
            show_legend=False,
            show_xlabel=False,
        )
    axes[-1].set_xlabel("Contribution (percentage)")
    fig.legend(
        handles=mass_legend_handles(), loc="lower center",
        bbox_to_anchor=(0.61, 0.012), ncol=3, frameon=False,
        handlelength=0.75, columnspacing=0.40, handletextpad=0.30,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(
        left=0.40, right=0.985, top=0.965, bottom=0.19, hspace=0.30
    )
    outputs.extend(save_figure(fig, stems[3]))
    plt.close(fig)

    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
