#!/usr/bin/env python
"""Plot K=3 palate ATAC-substate F1 and rare-state recall for six methods."""

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

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trainfbench_plot_style import apply_nature_rc, method_style


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results/palate_loo_atac_regulatory_substates/macro_summary.csv"
DEFAULT_OUTPUT = ROOT / "results/palate_loo_atac_regulatory_substates"

METHODS = (
    "COATI-B",
    "COATI-U",
    "CytoBridge-B",
    "CytoBridge-U",
    "MIOFlow",
    "TrajectoryNet",
)
DISPLAY = {
    "COATI-B": "COATI bal.",
    "COATI-U": "COATI unbal.",
    "CytoBridge-B": "CytoBridge bal.",
    "CytoBridge-U": "CytoBridge unbal.",
    "MIOFlow": "MIOFlow",
    "TrajectoryNet": "TrajectoryNet",
}
STYLE_NAME = {
    "COATI-B": "COATI balanced",
    "COATI-U": "COATI unbalanced",
    "CytoBridge-B": "CytoBridge balanced",
    "CytoBridge-U": "CytoBridge unbalanced",
    "MIOFlow": "MIOFlow",
    "TrajectoryNet": "TrajectoryNet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_plot() -> None:
    apply_nature_rc(font_size=10)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "palate_atac_substate_f1_rare_k3_no_rna_only"
    targets = [stem.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg")]
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError("Outputs exist; pass --overwrite")

    frame = pd.read_csv(args.input)
    frame = frame[frame["k"].eq(args.k) & frame["method"].isin(METHODS)].copy()
    expected = {(stage, method) for stage in ("E13.5", "E14.0") for method in METHODS}
    observed = set(zip(frame["heldout_stage"], frame["method"]))
    if observed != expected:
        raise RuntimeError(f"Missing or extra stage/method rows: {sorted(expected ^ observed)}")

    configure_plot()
    figure, axes = plt.subplots(2, 2, figsize=(3.15, 4.15), sharex=True, sharey=True)
    panels = (
        ("E13.5", "macro_f1", "E13.5\nSubstate F1 ↑"),
        ("E14.0", "macro_f1", "E14.0\nSubstate F1 ↑"),
        ("E13.5", "macro_rarest_substate_recall", "E13.5\nRare recall ↑"),
        ("E14.0", "macro_rarest_substate_recall", "E14.0\nRare recall ↑"),
    )
    for panel_index, (axis, (stage, metric, title)) in enumerate(zip(axes.ravel(), panels)):
        local = frame[frame["heldout_stage"].eq(stage)].set_index("method")
        values = np.asarray([float(local.loc[method, metric]) for method in METHODS])
        for index, (method, value) in enumerate(zip(METHODS, values)):
            axis.barh(
                index,
                value,
                height=0.58,
                color=method_style(STYLE_NAME[method]).color,
                edgecolor="none",
            )
            axis.text(
                1.04,
                index,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=8.3,
                color="#222222",
            )
        axis.set_title(title, pad=3, fontsize=10)
        # Reserve a compact label gutter after x=1 so the value labels remain
        # inside each facet while the colored bars themselves stay short.
        axis.set_xlim(0, 1.35)
        axis.set_xticks([0.0, 1.0], ["0.0", "1.0"])
        axis.set_ylim(len(METHODS) - 0.5, -0.5)
        axis.grid(axis="x", color="#DDDDDD", lw=0.5)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0, pad=2)
        if panel_index % 2 == 0:
            axis.set_yticks(np.arange(len(METHODS)), [DISPLAY[method] for method in METHODS])
        else:
            axis.set_yticks(np.arange(len(METHODS)))
            axis.tick_params(labelleft=False)

    figure.subplots_adjust(
        left=0.47,
        right=0.985,
        top=0.965,
        bottom=0.13,
        wspace=1.00,
        hspace=0.48,
    )
    figure.savefig(targets[0], dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.025)
    figure.savefig(targets[1], facecolor="white", bbox_inches="tight", pad_inches=0.025)
    figure.savefig(targets[2], facecolor="white", bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)
    for target in targets:
        print(target)


if __name__ == "__main__":
    main()
