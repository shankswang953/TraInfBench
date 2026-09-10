#!/usr/bin/env python3
"""Render the main-text GLI3 eight-peak source-line result as a numeric table.

The input is the all-model, source-line score table produced by
``analyze_human_cerebral_gli3_all13_peak_gene_comparison.py``.  The figure uses
the fair common-290 cohort with uniform particle weights.  Within each source
line and method, peaks are averaged within target gene and HES4, HES5 and
CREB5 are then weighted equally.  This avoids allowing the four HES5 peaks to
dominate the eight-peak summary.
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
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from trainfbench_plot_style import apply_nature_rc, method_style


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "results/human_cerebral_gli3_all13_peak_gene_comparison_v2"
    / "tables/strict_13pair_source_line_scores.csv"
)
DEFAULT_OUTPUT = ROOT / "results/human_cerebral_gli3_8peak_source_line_main"

PAIR_IDS = (
    "PG00309",  # HES4
    "PG00326", "PG00327", "PG00329", "PG00332",  # HES5
    "PG00095", "PG00102", "PG00127",  # CREB5
)
GENES = ("HES4", "HES5", "CREB5")
SOURCES = ("409b2", "H9", "Hoik1", "Wibj2")
DISPLAY_COLUMNS = SOURCES + ("Mean",)
SOURCE_LABELS = {
    "409b2": "409B2\niPSC",
    "H9": "H9\nESC",
    "Hoik1": "HOIK1\niPSC",
    "Wibj2": "WIBJ2\niPSC",
    "Mean": "Mean",
}

# Publication order follows the existing palate external-regulatory table.
METHODS = (
    (
        "coati_sync_balanced_cy0.5_s0_i30000",
        "COATI bal.",
        "COATI balanced",
        "balanced",
    ),
    (
        "coati_sync_unbalanced_cy0.5_s0_i40000",
        "COATI unbal.",
        "COATI unbalanced",
        "unbalanced",
    ),
    (
        "cytobridge_balanced_s42_i30000",
        "CytoBridge bal.",
        "CytoBridge balanced",
        "balanced",
    ),
    (
        "cytobridge_unbalanced_biological_prior_s42_i30000",
        "CytoBridge unbal.",
        "CytoBridge unbalanced",
        "unbalanced",
    ),
    ("mioflow_gaga10_balanced_s42_i30000", "MIOFlow", "MIOFlow", "balanced"),
    (
        "trajectorynet_forward_balanced_s0_i30000",
        "TrajectoryNet",
        "TrajectoryNet",
        "balanced",
    ),
    (
        "coati_rna_only_balanced_s0_i30000",
        "OT(RNA)",
        "Balanced RNA-only",
        "balanced",
    ),
    (
        "coati_rna_only_unbalanced_biological_prior_s0_i30000",
        "UOT(RNA)",
        "Unbalanced RNA-only",
        "unbalanced",
    ),
)

WINDOWS = {
    "D9-D18": ("D9_D18_zero_lag_derivative_cosine", "D9–D18"),
    "D9-D21": ("D9_D21_zero_lag_derivative_cosine", "D9–D21"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window", choices=tuple(WINDOWS), default="D9-D21")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_values(frame: pd.DataFrame, score_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "model_id", "scheme", "source_line", "pair_id", "gene", score_column,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")

    model_ids = [row[0] for row in METHODS]
    selected = frame.loc[
        frame["scheme"].eq("common_290_uniform")
        & frame["model_id"].isin(model_ids)
        & frame["source_line"].isin(SOURCES)
        & frame["pair_id"].isin(PAIR_IDS)
    ].copy()
    expected_rows = len(METHODS) * len(SOURCES) * len(PAIR_IDS)
    if len(selected) != expected_rows:
        raise ValueError(f"Expected {expected_rows} source-line pair rows; found {len(selected)}")
    if set(selected["pair_id"].astype(str)) != set(PAIR_IDS):
        raise ValueError("The frozen eight-peak set is incomplete or changed")
    if set(selected["gene"].astype(str)) != set(GENES):
        raise ValueError("Unexpected target genes in the frozen eight-peak set")
    duplicate = selected.duplicated(["model_id", "source_line", "pair_id"])
    if duplicate.any():
        raise ValueError("Duplicate model/source/pair rows in input")

    gene = (
        selected.groupby(["model_id", "source_line", "gene"], as_index=False)[score_column]
        .mean()
        .rename(columns={score_column: "within_gene_peak_mean"})
    )
    summary = (
        gene.groupby(["model_id", "source_line"], as_index=False)["within_gene_peak_mean"]
        .mean()
        .rename(columns={"within_gene_peak_mean": "three_gene_equal_coupling"})
    )
    metadata = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "display_method": display,
                "style_name": style_name,
                "balance_tier": tier,
                "method_order": index,
            }
            for index, (model_id, display, style_name, tier) in enumerate(METHODS)
        ]
    )
    summary = summary.merge(metadata, on="model_id", validate="many_to_one")
    summary["source_order"] = summary["source_line"].map(
        {source: index for index, source in enumerate(SOURCES)}
    )
    summary["n_peaks"] = len(PAIR_IDS)
    summary["n_genes"] = len(GENES)
    summary["cohort"] = "common_290_uniform"
    line_means = (
        summary.groupby(
            [
                "model_id",
                "display_method",
                "style_name",
                "balance_tier",
                "method_order",
                "n_peaks",
                "n_genes",
                "cohort",
            ],
            as_index=False,
        )["three_gene_equal_coupling"]
        .mean()
        .assign(source_line="Mean", source_order=len(SOURCES))
    )
    summary = pd.concat([summary, line_means], ignore_index=True)
    return (
        summary.sort_values(["method_order", "source_order"]).reset_index(drop=True),
        gene.sort_values(["model_id", "source_line", "gene"]).reset_index(drop=True),
    )


def draw_table(summary: pd.DataFrame, output_dir: Path, window_label: str, slug: str) -> list[Path]:
    apply_nature_rc(font_size=8.5)
    mpl.rcParams.update(
        {
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    matrix = (
        summary.pivot(index="model_id", columns="source_line", values="three_gene_equal_coupling")
        .reindex(index=[row[0] for row in METHODS], columns=DISPLAY_COLUMNS)
    )
    if matrix.isna().any().any():
        raise ValueError("Numeric table contains missing model/source values")

    figure = plt.figure(figsize=(6.25, 2.65), facecolor="white")
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.5,
        0.975,
        "GLI3-KO-supported peak–RNA trajectory coupling",
        ha="center",
        va="top",
        fontsize=11.0,
        color="#202124",
    )
    left = 0.012
    label_right = 0.325
    right = 0.988
    table_bottom = 0.035
    table_top = 0.865
    header_rule = 0.645
    column_width = (right - label_right) / len(DISPLAY_COLUMNS)
    centers = np.asarray(
        [
            label_right + (index + 0.5) * column_width
            for index in range(len(DISPLAY_COLUMNS))
        ]
    )

    axis.text(
        (label_right + label_right + len(SOURCES) * column_width) / 2,
        0.815,
        "Source cell line",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#202124",
    )
    for center, source in zip(centers, DISPLAY_COLUMNS):
        axis.text(
            center,
            0.715,
            SOURCE_LABELS[source],
            ha="center",
            va="center",
            fontsize=8.5,
            color="#202124",
            linespacing=1.05,
        )
    axis.text(left + 0.032, 0.715, "Method", ha="left", va="center", fontsize=8.5)
    axis.plot([left, right], [header_rule, header_rule], color="#777777", linewidth=0.7)

    row_centers = np.linspace(0.600, 0.080, len(METHODS))
    row_height = 0.066
    row_lookup = summary.set_index(["model_id", "source_line"])

    for row_index, (center_y, method) in enumerate(zip(row_centers, METHODS)):
        model_id, display, style_name, tier = method
        if row_index % 2 == 0:
            axis.add_patch(
                Rectangle(
                    (left, center_y - row_height / 2),
                    right - left,
                    row_height,
                    facecolor="#F5F5F5",
                    edgecolor="none",
                    zorder=0,
                )
            )
        style = method_style(style_name)
        face = style.color if style.markerfacecolor is None else style.markerfacecolor
        edge = style.color if style.markeredgecolor is None else style.markeredgecolor
        swatch_has_border = display != "TrajectoryNet"
        axis.add_patch(
            Rectangle(
                (left + 0.004, center_y - 0.014),
                0.026,
                0.028,
                facecolor=face,
                edgecolor=edge if swatch_has_border else "none",
                linewidth=0.65 if swatch_has_border else 0.0,
                zorder=3,
            )
        )
        axis.text(
            left + 0.052,
            center_y,
            display,
            ha="left",
            va="center",
            fontsize=8.5,
            color="#202124",
        )
        for center_x, source in zip(centers, DISPLAY_COLUMNS):
            value = float(row_lookup.loc[(model_id, source), "three_gene_equal_coupling"])
            axis.text(
                center_x,
                center_y,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="#202124",
                fontweight="normal",
            )

    axis.plot(
        [label_right, label_right],
        [table_bottom, table_top],
        color="#B3B3B3",
        linewidth=0.6,
    )
    for boundary in range(1, len(DISPLAY_COLUMNS)):
        x = label_right + boundary * column_width
        linewidth = 0.75 if boundary == len(SOURCES) else 0.45
        color = "#B3B3B3" if boundary == len(SOURCES) else "#E0E0E0"
        upper = table_top if boundary == len(SOURCES) else header_rule
        axis.plot([x, x], [table_bottom, upper], color=color, linewidth=linewidth)
    axis.add_patch(
        Rectangle(
            (left, table_bottom),
            right - left,
            table_top - table_bottom,
            facecolor="none",
            edgecolor="#777777",
            linewidth=0.85,
            zorder=6,
            clip_on=False,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"gli3_8peak_source_line_numeric_table_{slug}"
    outputs: list[Path] = []
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        kwargs: dict[str, object] = {
            "facecolor": "white",
            "bbox_inches": "tight",
            "pad_inches": 0.01,
        }
        if suffix == "png":
            kwargs["dpi"] = 600
        figure.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(figure)
    return outputs


def main() -> None:
    args = parse_args()
    score_column, window_label = WINDOWS[args.window]
    slug = args.window.lower().replace("-", "_")
    output_dir = args.output_dir.resolve()
    expected = [
        output_dir / f"gli3_8peak_source_line_numeric_table_{slug}.{suffix}"
        for suffix in ("png", "pdf", "svg")
    ] + [
        output_dir / f"gli3_8peak_source_line_numeric_values_{slug}.csv",
        output_dir / f"gli3_8peak_source_line_gene_values_{slug}.csv",
        output_dir / f"analysis_manifest_{slug}.json",
    ]
    existing = [path for path in expected if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite outputs; pass --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )

    frame = pd.read_csv(args.input)
    summary, gene = aggregate_values(frame, score_column)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        output_dir / f"gli3_8peak_source_line_numeric_values_{slug}.csv",
        index=False,
    )
    gene.to_csv(
        output_dir / f"gli3_8peak_source_line_gene_values_{slug}.csv",
        index=False,
    )
    outputs = draw_table(summary, output_dir, window_label, slug)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "score_window": window_label,
        "trajectory_support": "D4 to D21; source strata follow D4 origin",
        "score": "zero-lag cosine of SG(7,2)-smoothed RNA and ATAC derivatives",
        "pair_ids": list(PAIR_IDS),
        "genes": list(GENES),
        "aggregation": "mean across peaks within gene, then equal mean across three genes",
        "cohort": "common 290 all-method endpoint-intersection D4 particles",
        "particle_weighting": "uniform",
        "source_lines": list(SOURCES),
        "mean_column": "equal arithmetic mean across the four biological source lines",
        "source_line_semantics": "biological stem-cell source lines, not technical batches",
        "models": [row[0] for row in METHODS],
        "claim_limits": [
            "targeted eight-peak, three-gene GLI3 case study; not genome-wide recovery",
            "baseline ATAC is a fixed-T post-hoc readout; only COATI has native synchronized ATAC",
            "single seed; COATI unbalanced uses 40k iterations and the other models use 30k",
        ],
        "outputs": [str(path.resolve()) for path in outputs],
    }
    (output_dir / f"analysis_manifest_{slug}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
