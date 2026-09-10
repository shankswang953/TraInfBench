#!/usr/bin/env python3
"""Plot source-line W2 sums as separate RNA and ATAC grouped dot plots."""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from trainfbench_plot_style import apply_nature_rc, method_style


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/human_cerebral_full_biological_interpretability/09_source_line_group_w2"
DEFAULT_INPUT = RESULTS / "source_line_time_scores.csv"
DEFAULT_OUTPUT = RESULTS / "figures"

SOURCES = ("409b2", "h9", "hoik1", "wibj2")
SOURCE_LABELS = {
    "409b2": "409B2\niPSC",
    "h9": "H9\nESC",
    "hoik1": "HOIK1\niPSC",
    "wibj2": "WIBJ2\niPSC",
}
METHODS = (
    ("COATI bal.", "COATI balanced"),
    ("COATI unbal.", "COATI unbalanced"),
    ("CytoBridge bal.", "CytoBridge balanced"),
    ("CytoBridge unbal.", "CytoBridge unbalanced"),
    ("MIOFlow", "MIOFlow"),
    ("TrajectoryNet", "TrajectoryNet"),
    ("OT(RNA)", "Balanced RNA-only"),
    ("UOT(RNA)", "Unbalanced RNA-only"),
)
TIMES = ("D7", "D9", "D11", "D12", "D18", "D21")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[
        frame["query_weighting"].eq("uniform")
        & frame["method"].isin([method for method, _ in METHODS])
        & frame["source_line"].isin(SOURCES)
        & frame["time"].isin(TIMES)
        & frame["modality"].isin(("RNA", "ATAC"))
    ].copy()
    duplicate = frame.duplicated(["method", "source_line", "time", "modality"])
    if duplicate.any():
        raise ValueError("Duplicate method/source/time/modality rows")
    expected = len(METHODS) * len(SOURCES) * len(TIMES) * 2
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} score rows; found {len(frame)}")
    summary = (
        frame.groupby(["method_order", "method", "source_line", "modality"], as_index=False)
        .agg(sum_w2=("w2_score", "sum"), mean_w2=("w2_score", "mean"), n_times=("time", "nunique"))
    )
    if not summary["n_times"].eq(len(TIMES)).all():
        raise ValueError("Every point must contain all six later times")
    method_order = {method: index for index, (method, _) in enumerate(METHODS)}
    source_order = {source: index for index, source in enumerate(SOURCES)}
    summary["method_order"] = summary["method"].map(method_order)
    summary["source_order"] = summary["source_line"].map(source_order)
    return summary.sort_values(["modality", "source_order", "method_order"]).reset_index(drop=True)


def draw_modality(data: pd.DataFrame, modality: str, output_dir: Path) -> list[Path]:
    apply_nature_rc(font_size=8.0)
    mpl.rcParams.update(
        {
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "legend.handletextpad": 0.45,
            "legend.columnspacing": 1.1,
        }
    )
    local = data.loc[data["modality"].eq(modality)].copy()
    figure, axis = plt.subplots(figsize=(4.8, 3.15))
    centers = np.arange(len(SOURCES), dtype=float)
    offsets = np.linspace(-0.245, 0.245, len(METHODS))
    handles: list[Line2D] = []
    for method_index, (method, style_name) in enumerate(METHODS):
        style = method_style(style_name)
        values = (
            local.loc[local["method"].eq(method)]
            .set_index("source_line")
            .reindex(SOURCES)["sum_w2"]
            .to_numpy(float)
        )
        if not np.isfinite(values).all():
            raise ValueError(f"Missing {modality} values for {method}")
        face = style.color if style.markerfacecolor is None else style.markerfacecolor
        edge = style.color if style.markeredgecolor is None else style.markeredgecolor
        is_coati = method.startswith("COATI")
        axis.scatter(
            centers + offsets[method_index],
            values,
            s=46 if is_coati else 37,
            marker=style.marker,
            facecolors=face,
            edgecolors=edge,
            linewidths=0.9 if is_coati else 0.7,
            zorder=4 if is_coati else 3,
        )
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=style.marker,
                markersize=5.2,
                markerfacecolor=face,
                markeredgecolor=edge,
                markeredgewidth=0.8,
                label=method,
            )
        )

    lo, hi = float(local["sum_w2"].min()), float(local["sum_w2"].max())
    pad = max(0.06, 0.10 * (hi - lo))
    axis.set_xlim(-0.55, len(SOURCES) - 0.45)
    axis.set_ylim(max(0.0, lo - pad), hi + pad)
    axis.set_xticks(centers, [SOURCE_LABELS[source] for source in SOURCES])
    axis.set_xlabel("Source cell line", labelpad=5)
    axis.set_ylabel(r"Summed $W_2$ (six times) $\downarrow$")
    axis.set_title(modality, pad=7, fontsize=10.0)
    axis.yaxis.grid(True, color="#E5E5E5", linewidth=0.55, zorder=0)
    axis.xaxis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#666666")
        spine.set_linewidth(0.75)
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=False,
        fontsize=7.0,
    )
    figure.subplots_adjust(left=0.16, right=0.985, top=0.90, bottom=0.30)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"source_line_{modality.lower()}_w2_sum_grouped_dots"
    outputs = [stem.with_suffix(".pdf"), stem.with_suffix(".png"), stem.with_suffix(".svg")]
    figure.savefig(outputs[0], bbox_inches="tight", pad_inches=0.025)
    figure.savefig(outputs[1], dpi=600, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(outputs[2], bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)
    return outputs


def main() -> None:
    args = parse_args()
    outputs = {
        modality: [
            args.output_dir / f"source_line_{modality.lower()}_w2_sum_grouped_dots.{suffix}"
            for suffix in ("pdf", "png", "svg")
        ]
        for modality in ("RNA", "ATAC")
    }
    data_path = args.output_dir / "source_line_w2_sum_grouped_dots_data.csv"
    manifest_path = args.output_dir / "source_line_w2_sum_grouped_dots_manifest.json"
    expected_outputs = [path for paths in outputs.values() for path in paths] + [data_path, manifest_path]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs; pass --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )
    data = prepare(args.input)
    written: list[Path] = []
    for modality in ("RNA", "ATAC"):
        written.extend(draw_modality(data, modality, args.output_dir))
    data.to_csv(data_path, index=False)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "outputs": [str(path.resolve()) for path in written],
        "weighting": "uniform particles and uniform observed metacells",
        "aggregation": "sum across D7, D9, D11, D12, D18 and D21, separately for each source line",
        "w2_definition": "sqrt(2 * debiased p=2 Sinkhorn divergence)",
        "coati_cy": 0.5,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n".join(str(path) for path in expected_outputs), flush=True)


if __name__ == "__main__":
    main()
