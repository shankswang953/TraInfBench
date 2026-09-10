from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "results/pancreas_full_unbalanced_delta_epsilon_mass_change_decomposition_15nn_20k_cy05/"
    "global_total_mass.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/pancreas_full_unbalanced_delta_epsilon_mass_change_decomposition_15nn_20k_cy05"
)

STAGES = ("E14.5", "E15.5", "E16.5")
METHODS = (
    ("COATI unbalanced", "COATI unbal.", "#D95F02", "D"),
    ("CytoBridge unbalanced", "CytoBridge unbal.", "#CC79A7", "s"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot total native mass relative to the initial E14.5 mass."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "font.weight": "normal",
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "axes.labelsize": 10,
            "axes.labelweight": "normal",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)
    frame = frame[
        (frame["design"] == "full")
        & (frame["weighting"] == "native_total_mass")
    ].copy()

    apply_style()
    fig, ax = plt.subplots(figsize=(3.75, 2.55))
    x = np.arange(len(STAGES), dtype=float)

    for method, label, color, marker in METHODS:
        values = (
            frame[frame["method"] == method]
            .set_index("stage")
            .loc[list(STAGES), "model_total_mass"]
            .to_numpy(float)
        )
        values /= values[0]
        ax.plot(
            x,
            values,
            color=color,
            marker=marker,
            linewidth=1.6,
            markersize=5.2,
            label=label,
            zorder=3,
        )

    observed = (
        frame.drop_duplicates("stage")
        .set_index("stage")
        .loc[list(STAGES), "observed_sample_total_mass"]
        .to_numpy(float)
    )
    observed /= observed[0]
    ax.plot(
        x,
        observed,
        color="#666666",
        marker="x",
        linestyle="--",
        linewidth=1.2,
        markersize=5.5,
        label="Observed cell number",
        zorder=2,
    )

    ax.axhline(1.0, color="#A0A0A0", linewidth=0.7, zorder=1)
    ax.set_xticks(x, STAGES)
    ax.set_ylim(0.30, 1.20)
    ax.set_yticks((0.4, 0.6, 0.8, 1.0, 1.2))
    ax.set_ylabel("Mass relative to E14.5")
    ax.set_title("Total mass")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=2,
        columnspacing=0.9,
        handlelength=1.5,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.20, right=0.99, top=0.90, bottom=0.35)

    stem = args.output_dir / "pancreas_full_unbalanced_total_mass_relative_to_e145_compact"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            stem.with_suffix(f".{suffix}"),
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
