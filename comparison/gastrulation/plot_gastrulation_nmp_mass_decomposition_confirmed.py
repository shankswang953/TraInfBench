from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/gastrulation_nmp_mass_decomposition_interval_restart_pilot"
INTERVAL = "E8.5→E8.75"
K = 20

METHODS = [
    "USOT Sync mean",
    "USOT RNA-only",
    "CytoBridge unbalanced",
    "TIGON",
]
LABELS = {
    "USOT Sync mean": "USOT Sync",
    "USOT RNA-only": "USOT RNA-only",
    "CytoBridge unbalanced": "CytoBridge unbal.",
    "TIGON": "TIGON",
}
COMPONENTS = [
    ("retained_NMP_reweighting", "Retained", "#0072B2"),
    ("incoming_to_NMP", "Incoming", "#E69F00"),
    ("outgoing_from_NMP", "Outgoing", "#CC79A7"),
]


def main() -> None:
    frame = pd.read_csv(RESULTS / "interval_restart_nmp_mass_decomposition_summary.csv")
    frame = frame[(frame["k"] == K) & (frame["interval"] == INTERVAL)]
    frame = frame.set_index("method").loc[METHODS]

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(5.1, 3.25))
    y = np.arange(len(METHODS))
    positive_base = np.zeros(len(METHODS), dtype=float)
    negative_base = np.zeros(len(METHODS), dtype=float)

    for column, label, color in COMPONENTS:
        values = frame[column].to_numpy(float) * 100.0
        left = np.where(values >= 0, positive_base, negative_base)
        ax.barh(
            y,
            values,
            left=left,
            height=0.56,
            color=color,
            label=label,
            zorder=2,
        )
        positive_base += np.where(values >= 0, values, 0.0)
        negative_base += np.where(values < 0, values, 0.0)

    net = frame["delta_NMP_share"].to_numpy(float) * 100.0
    ax.scatter(net, y, marker="D", s=45, color="#222222", zorder=4)
    for value, index in zip(net, y):
        ax.annotate(
            f"{value:+.2f}",
            (value, index),
            xytext=(-6, -11),
            textcoords="offset points",
            ha="right",
            va="center",
        )

    observed = (420 / 7219 - 887 / 11470) * 100.0
    ax.axvline(observed, color="#222222", linestyle="--", linewidth=1.2, zorder=1)
    ax.axvline(0.0, color="#555555", linewidth=0.8, zorder=1)
    ax.set_yticks(y, [LABELS[method] for method in METHODS])
    ax.invert_yaxis()
    ax.set_xlim(-2.80, 1.30)
    ax.set_xlabel("Change in NMP relative mass (percentage points)")
    ax.set_title("NMP mass-change decomposition", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E1E1E1", linewidth=0.7, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [Patch(facecolor=color, label=label) for _, label, color in COMPONENTS]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor="#222222",
                markersize=6,
                label="Net",
            ),
            Line2D(
                [0],
                [0],
                color="#222222",
                linestyle="--",
                linewidth=1.2,
                label=f"Observed ({observed:+.2f})",
            ),
        ]
    )
    ax.legend(
        handles=handles,
        frameon=False,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(-0.02, -0.24),
        handlelength=1.2,
        columnspacing=0.9,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.31, right=0.99, top=0.88, bottom=0.34)

    for suffix in ("png", "pdf"):
        fig.savefig(
            RESULTS / f"nmp_mass_change_decomposition_confirmed.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
