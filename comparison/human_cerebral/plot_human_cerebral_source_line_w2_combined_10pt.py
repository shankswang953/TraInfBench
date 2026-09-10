#!/usr/bin/env python3
"""Combine the source-line RNA and ATAC W2 panels with one shared legend."""

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
import numpy as np

from plot_human_cerebral_publication_redraw_10pt import (
    METHODS,
    SOURCE_INPUT,
    SOURCE_LABELS,
    SOURCE_ORDER,
    finish_axis,
    legend_handle,
    prepare_source,
    setup_style,
    visual,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/human_cerebral_publication_redraw_10pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / "source_line_rna_atac_w2_combined.pdf"
    if pdf.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {pdf}; pass --overwrite")

    setup_style()
    data = prepare_source(SOURCE_INPUT)
    fig, axes = plt.subplots(2, 1, figsize=(3.65, 5.10), sharex=True)
    centers = np.arange(4, dtype=float)
    offsets = np.linspace(-0.22, 0.22, len(METHODS))
    handles = []

    for axis, modality in zip(axes, ("RNA", "ATAC")):
        local = data.loc[data.modality.eq(modality)]
        for index, (old, display, style_name) in enumerate(METHODS):
            values = (
                local.loc[local.method.eq(old)]
                .set_index("source_line")
                .reindex(SOURCE_ORDER)
                .sum_w2.to_numpy(float)
            )
            marker, face, edge = visual(style_name)
            axis.scatter(
                centers + offsets[index],
                values,
                s=24,
                marker=marker,
                facecolors=face,
                edgecolors=edge,
                linewidths=0.55,
                zorder=3,
            )
            if modality == "RNA":
                handle = legend_handle(display, style_name)
                handle.set_markersize(4.3)
                handle.set_markeredgewidth(0.55)
                handles.append(handle)
        lo, hi = float(local.sum_w2.min()), float(local.sum_w2.max())
        pad = max(0.035, 0.07 * (hi - lo))
        axis.set_xlim(-0.5, 3.5)
        axis.set_ylim(max(0, lo - pad), hi + pad)
        axis.set_ylabel(r"Summed $W_2$ $\downarrow$")
        axis.set_title(modality, pad=3, fontweight="normal")
        finish_axis(axis)

    axes[-1].set_xticks(centers, SOURCE_LABELS)
    axes[-1].set_xlabel("Source cell line", labelpad=2)
    axes[0].tick_params(labelbottom=False)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=2,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.9,
        labelspacing=0.25,
    )
    fig.subplots_adjust(left=0.205, right=0.985, top=0.965, bottom=0.325, hspace=0.18)

    outputs = []
    for suffix in ("pdf", "png", "svg"):
        path = output / f"source_line_rna_atac_w2_combined.{suffix}"
        fig.savefig(
            path,
            dpi=600 if suffix == "png" else None,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.01,
        )
        outputs.append(str(path.resolve()))
    plt.close(fig)
    manifest = {
        "description": "Vertically stacked source-line RNA and ATAC W2 panels",
        "font": "Arial 10 pt",
        "shared_legend": True,
        "ylabel": "Summed W2",
        "input": str(SOURCE_INPUT.resolve()),
        "outputs": outputs,
    }
    (output / "source_line_rna_atac_w2_combined_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
