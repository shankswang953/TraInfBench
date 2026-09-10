#!/usr/bin/env python
"""Render landscape versions of two gastrulation follow-up figures.

The script only reformats existing CSV summaries; it does not rerun any model.
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
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

from trainfbench_plot_style import (  # noqa: E402
    GASTRULATION_FATE_STYLES,
    apply_nature_rc,
)


DEFAULT_COMPOSITION = Path(
    "external/COATI/Gastrulation/"
    "ResultCompare/ThreeCompare/trajectorynet_native_base_conditioned_composition/"
    "caudal_epiblast_seven_methods_trajectorynet_native_base.csv"
)
DEFAULT_DECOMPOSITION_DIR = (
    ROOT
    / "results/gastrulation_full_nmp_mass_decomposition_three_unbalanced_k1"
)
DEFAULT_OUTPUT_DIR = ROOT / "results/jsm2026_gastrulation_followups_landscape"

STAGE_STEPS = (10, 20, 25)
FATE_GROUPS = (
    "Caudal epiblast",
    "NMP",
    "Posterior neural-spinal",
    "Paraxial-somitic",
    "Other",
)
METHODS = (
    "COATI balanced",
    "COATI unbalanced",
    "CytoBridge balanced",
    "CytoBridge unbalanced",
    "TrajectoryNet",
    "TIGON",
    "MIOFlow",
)
METHOD_TITLES = {
    "COATI balanced": "COATI bal.",
    "COATI unbalanced": "COATI unbal.",
    "CytoBridge balanced": "CytoBridge bal.",
    "CytoBridge unbalanced": "CytoBridge unbal.",
    "TrajectoryNet": "TrajectoryNet",
    "TIGON": "TIGON",
    "MIOFlow": "MIOFlow",
}
FATE_LABELS = {
    "Caudal epiblast": "Caudal epi.",
    "NMP": "NMP",
    "Posterior neural-spinal": "Neural–spinal",
    "Paraxial-somitic": "Paraxial–somitic",
    "Other": "Other",
}

INTERVALS = (("E7.5", "E8.0"), ("E8.0", "E8.5"), ("E8.5", "E8.75"))
DECOMP_METHODS = ("COATI unbalanced", "CytoBridge unbalanced", "TIGON")
DECOMP_LABELS = {
    "COATI unbalanced": "COATI unbal.",
    "CytoBridge unbalanced": "CytoBridge unbal.",
    "TIGON": "TIGON",
}
COMPONENTS = (
    ("retained_nmp_weight_change", "Retained reweighting", "#0072B2"),
    ("incoming_final_mass", "Incoming", "#E69F00"),
    ("outgoing_plot", "Outgoing", "#CC79A7"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-csv", type=Path, default=DEFAULT_COMPOSITION)
    parser.add_argument(
        "--decomposition-dir", type=Path, default=DEFAULT_DECOMPOSITION_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_figure(fig: plt.Figure, prefix: Path) -> None:
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        prefix.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(fig)


def plot_composition(frame: pd.DataFrame, prefix: Path) -> None:
    apply_nature_rc(font_size=10)
    fig, axes_array = plt.subplots(
        2,
        4,
        figsize=(11.6, 4.65),
        sharex=True,
        sharey=True,
    )
    axes = axes_array.ravel()
    for panel, (ax, method) in enumerate(zip(axes[:7], METHODS)):
        local = frame[frame["method"] == method]
        for group in FATE_GROUPS:
            curve = local[local["fate_group"] == group].sort_values("step")
            style = GASTRULATION_FATE_STYLES[group]
            ax.plot(
                curve["step"],
                curve["percent"],
                color=style.color,
                linewidth=1.45,
                marker=style.marker,
                markersize=3.0,
                markevery=2,
            )
        ax.set_title(METHOD_TITLES[method], fontsize=11, pad=2)
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 100)
        ax.set_xticks((0, *STAGE_STEPS))
        ax.set_xticklabels(("E7.5", "E8.0", "E8.5", "E8.75"))
        ax.set_yticks((0, 50, 100))
        for stage_step in STAGE_STEPS:
            ax.axvline(
                stage_step,
                color="#B8B8B8",
                linewidth=0.7,
                linestyle=":",
                zorder=0,
            )
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(
            labelleft=(panel % 4 == 0),
            labelbottom=(panel >= 4),
            labelsize=9,
            pad=1,
        )

    legend_ax = axes[7]
    legend_ax.axis("off")
    handles = []
    labels = []
    for group in FATE_GROUPS:
        style = GASTRULATION_FATE_STYLES[group]
        handles.append(
            Line2D(
                [],
                [],
                color=style.color,
                linewidth=1.45,
                marker=style.marker,
                markersize=4,
            )
        )
        labels.append(FATE_LABELS[group])
    legend_ax.legend(
        handles,
        labels,
        loc="center left",
        frameon=False,
        handlelength=1.7,
        labelspacing=0.35,
        fontsize=10,
    )
    fig.supylabel("Composition (%)", x=0.008, fontsize=12)
    fig.subplots_adjust(
        left=0.065,
        right=0.995,
        bottom=0.11,
        top=0.97,
        wspace=0.16,
        hspace=0.24,
    )
    save_figure(fig, prefix)


def plot_decomposition(
    summary: pd.DataFrame,
    observed: pd.DataFrame,
    prefix: Path,
) -> None:
    apply_nature_rc(font_size=10)
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.65), sharey=True)
    y = np.arange(len(DECOMP_METHODS))
    observed_lookup = observed.set_index("stage")["observed_nmp_share"]

    for panel, (ax, (stage0, stage1)) in enumerate(zip(axes, INTERVALS)):
        interval = f"{stage0}→{stage1}"
        local = (
            summary[summary["interval"] == interval]
            .set_index("method")
            .loc[list(DECOMP_METHODS)]
        )
        positive = np.zeros(len(DECOMP_METHODS))
        negative = np.zeros(len(DECOMP_METHODS))
        plot_values = {
            "retained_nmp_weight_change": (
                local["retained_nmp_weight_change_mean"].to_numpy(float) * 100.0
            ),
            "incoming_final_mass": (
                local["incoming_final_mass_mean"].to_numpy(float) * 100.0
            ),
            "outgoing_plot": (
                -local["outgoing_initial_mass_mean"].to_numpy(float) * 100.0
            ),
        }
        for column, _label, color in COMPONENTS:
            values = plot_values[column]
            left = np.where(values >= 0, positive, negative)
            ax.barh(y, values, left=left, height=0.28, color=color, zorder=2)
            positive += np.where(values >= 0, values, 0.0)
            negative += np.where(values < 0, values, 0.0)

        net = local["delta_nmp_share_mean"].to_numpy(float) * 100.0
        ax.scatter(net, y, marker="D", s=42, color="#222222", zorder=4)
        for index, value in enumerate(net):
            if panel == 0:
                offset = (5, 0, "left") if value > 0.2 else (0, -11, "center")
            elif panel == 1:
                offset = (5, 0, "left") if index < 2 else (0, -11, "center")
            else:
                offset = (5, 0, "left") if value < 0 else (0, -11, "center")
            x_offset, y_offset, horizontal_alignment = offset
            ax.annotate(
                f"{value:+.2f}",
                (value, index),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha=horizontal_alignment,
                va="center",
                fontsize=9.5,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 0.2,
                },
                zorder=5,
            )

        observed_delta = (
            float(observed_lookup.loc[stage1])
            - float(observed_lookup.loc[stage0])
        ) * 100.0
        ax.axvline(
            observed_delta,
            color="#222222",
            linestyle="--",
            linewidth=1.15,
            zorder=1,
        )
        ax.axvline(0.0, color="#555555", linewidth=0.8, zorder=1)
        ax.set_title(
            f"{interval}\nObserved: {observed_delta:+.2f}",
            fontsize=11,
            pad=5,
        )
        ax.set_yticks(y)
        if panel == 0:
            ax.set_yticklabels(
                [DECOMP_LABELS[method] for method in DECOMP_METHODS],
                fontsize=10,
            )
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.invert_yaxis()
        ax.set_ylim(len(DECOMP_METHODS) - 0.40, -0.72)
        ax.grid(axis="x", color="#E1E1E1", linewidth=0.7, zorder=0)
        x_min, x_max = ax.get_xlim()
        span = x_max - x_min
        ax.set_xlim(x_min - 0.04 * span, x_max + 0.14 * span)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=9)

    handles = [Patch(facecolor=color, label=label) for _, label, color in COMPONENTS]
    handles.extend(
        [
            Line2D(
                [],
                [],
                marker="D",
                color="none",
                markerfacecolor="#222222",
                markersize=6,
                label="Predicted net change",
            ),
            Line2D(
                [],
                [],
                color="#222222",
                linestyle="--",
                linewidth=1.15,
                label="Observed",
            ),
        ]
    )
    fig.supxlabel(
        "Δ NMP proportion (percentage points)",
        x=0.56,
        y=0.16,
        fontsize=11,
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.015),
        ncol=5,
        frameon=False,
        fontsize=9.5,
        handlelength=1.2,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    fig.subplots_adjust(
        left=0.16,
        right=0.995,
        top=0.88,
        bottom=0.31,
        wspace=0.25,
    )
    save_figure(fig, prefix)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        args.output_dir / "caudal_epiblast_seven_methods_landscape.pdf",
        args.output_dir / "caudal_epiblast_seven_methods_landscape.png",
        args.output_dir / "nmp_mass_change_decomposition_landscape.pdf",
        args.output_dir / "nmp_mass_change_decomposition_landscape.png",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to overwrite: {names}; pass --overwrite")

    composition = pd.read_csv(args.composition_csv)
    composition_totals = composition.groupby(["method", "step"])["percent"].sum()
    if not np.allclose(composition_totals.to_numpy(), 100.0, atol=1e-6):
        raise RuntimeError("Composition percentages do not sum to 100")
    plot_composition(
        composition,
        args.output_dir / "caudal_epiblast_seven_methods_landscape",
    )

    summary_path = args.decomposition_dir / "nmp_full_mass_decomposition_summary.csv"
    observed_path = args.decomposition_dir / "observed_nmp_mass_share.csv"
    summary = pd.read_csv(summary_path)
    observed = pd.read_csv(observed_path)
    plot_decomposition(
        summary,
        observed,
        args.output_dir / "nmp_mass_change_decomposition_landscape",
    )

    manifest = {
        "analysis": "JSM 2026 landscape re-layouts of existing gastrulation results",
        "composition_source": str(args.composition_csv),
        "decomposition_summary_source": str(summary_path),
        "observed_source": str(observed_path),
        "numerical_changes": False,
    }
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
