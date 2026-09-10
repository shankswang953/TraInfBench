#!/usr/bin/env python3
"""Plot all-method time-resolved original-cell RNA distribution recovery."""

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

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from trainfbench_plot_style import apply_nature_rc, method_style


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "results/human_cerebral_full_biological_interpretability"
    / "10_raw_cell_fate_recovery/numeric_tables_by_time"
    / "appendix_all_methods_rna_w2_by_time.csv"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.parent
TIMES = ("D7", "D9", "D11", "D12", "D18", "D21")
ORDER = (
    "COATI bal. (Sync)",
    "COATI unbal. (Sync)",
    "CytoBridge bal.",
    "CytoBridge unbal.",
    "MIOFlow",
    "TrajectoryNet",
    "RNA-only bal.",
    "RNA-only unbal.",
)
STYLE_NAMES = {
    "COATI bal. (Sync)": "COATI balanced",
    "COATI unbal. (Sync)": "COATI unbalanced",
    "CytoBridge bal.": "CytoBridge balanced",
    "CytoBridge unbal.": "CytoBridge unbalanced",
    "MIOFlow": "MIOFlow",
    "TrajectoryNet": "TrajectoryNet",
    "RNA-only bal.": "Balanced RNA-only",
    "RNA-only unbal.": "Unbalanced RNA-only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.input).set_index("Method").reindex(ORDER)
    if table.loc[:, list(TIMES)].isna().any().any():
        raise ValueError("Missing method/time values in input table")

    apply_nature_rc(font_size=8.0)
    fig, ax = plt.subplots(figsize=(7.15, 3.5))
    centers = np.arange(len(TIMES), dtype=float)
    offsets = np.linspace(-0.22, 0.22, len(ORDER))
    handles: list[Line2D] = []
    for method_index, method in enumerate(ORDER):
        style = method_style(STYLE_NAMES[method])
        is_coati = method.startswith("COATI")
        face = style.color if style.markerfacecolor is None else style.markerfacecolor
        edge = style.color if style.markeredgecolor is None else style.markeredgecolor
        ax.scatter(
            centers + offsets[method_index],
            table.loc[method, list(TIMES)].to_numpy(dtype=float),
            s=49 if is_coati else 40,
            marker=style.marker,
            facecolors=face,
            edgecolors=edge,
            linewidths=0.9 if is_coati else 0.75,
            zorder=4 if is_coati else 3,
        )
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=style.marker,
                markersize=5.5,
                markerfacecolor=face,
                markeredgecolor=edge,
                markeredgewidth=0.8,
                label=method,
            )
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(r"RNA sliced $W_2$ (normalized PCA30) $\downarrow$")
    ax.set_xlim(-0.48, len(TIMES) - 0.52)
    ax.set_xticks(centers, TIMES)
    values = table.loc[:, list(TIMES)].to_numpy(dtype=float)
    lo, hi = float(values.min()), float(values.max())
    pad = 0.08 * (hi - lo)
    ax.set_ylim(max(0.0, lo - pad), hi + pad)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#666666")
        spine.set_linewidth(0.75)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=7.2,
        handletextpad=0.45,
        columnspacing=1.15,
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.79, bottom=0.17)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "appendix_all_methods_rna_w2_by_time_scatter"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
