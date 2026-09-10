#!/usr/bin/env python
"""Plot rostral regulatory-state ablation with strict-LOO E8.5 blood routing."""

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
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

from analyze_gastrulation_loo_rostral_gross_cross_lineage import (  # noqa: E402
    BLOOD_ERYTHROID,
)
from analyze_gastrulation_strict_loo2_all_source_offlineage import (  # noqa: E402
    load_records,
)
from evaluate_gastrulation_full_cmcc import _load_references  # noqa: E402
from plot_gastrulation_rostral_tss_ablation import (  # noqa: E402
    AMBIGUOUS_COLOR,
    DEFAULT_TSS_DIR,
    HEP_COLOR,
    NEURAL_PROGRAM_COLOR,
    clean_axis,
    configure_style,
    neural_open_indices,
    select_tss_inputs,
)
from trainfbench_plot_style import method_style  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    ROOT / "results/gastrulation_rostral_tss_heldout_e85"
)
METHOD_ORDER = (
    "OT(RNA)",
    "MIOFlow (GAGA10D)",
    "CytoBridge bal.",
    "COATI bal.",
)
METHOD_LABELS = {
    "OT(RNA)": "OT(RNA)",
    "MIOFlow (GAGA10D)": "MIOFlow",
    "CytoBridge bal.": "CytoBridge bal.",
    "COATI bal.": "COATI bal.",
}
METHOD_STYLE_KEYS = {
    "OT(RNA)": "OT baseline RNA",
    "MIOFlow (GAGA10D)": "MIOFlow",
    "CytoBridge bal.": "CytoBridge balanced",
    "COATI bal.": "COATI balanced",
}
K = 5
P_BLOOD_THRESHOLD = 0.5
HELDOUT_STAGE_INDEX = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tss-dir", type=Path, default=DEFAULT_TSS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite outputs; pass --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )


def compute_e85_blood_routing(
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    cohort = neural_open_indices(scores)
    records = {str(record["method"]): record for record in load_records()}

    rna_references, _, labels_by_stage, _, _ = _load_references()
    reference = np.asarray(
        rna_references[HELDOUT_STAGE_INDEX], dtype=np.float32
    )
    target_labels = np.asarray(
        labels_by_stage[HELDOUT_STAGE_INDEX], dtype=str
    )
    neighbours = NearestNeighbors(
        n_neighbors=K,
        algorithm="brute",
        metric="euclidean",
        n_jobs=-1,
    ).fit(reference)

    summary_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    provenance: dict[str, str] = {}
    for method in METHOD_ORDER:
        record = records[method]
        prediction = np.asarray(record["rna"], dtype=np.float32)[cohort]
        neighbor_index = neighbours.kneighbors(
            prediction, return_distance=False
        )
        neighbor_labels = target_labels[neighbor_index]
        p_blood = np.mean(
            np.isin(neighbor_labels, sorted(BLOOD_ERYTHROID)), axis=1
        )
        assigned = p_blood >= P_BLOOD_THRESHOLD
        count = int(np.sum(assigned))
        provenance[method] = str(record["provenance"])
        summary_rows.append(
            {
                "method": METHOD_LABELS[method],
                "source_method_name": method,
                "heldout_stage": "E8.5",
                "n_neural_open": int(len(cohort)),
                "n_assigned_blood": count,
                "fraction_assigned_blood": float(np.mean(assigned)),
                "percent_assigned_blood": float(100.0 * np.mean(assigned)),
                "mean_soft_5nn_blood_probability": float(np.mean(p_blood)),
                "k": K,
                "p_blood_threshold": P_BLOOD_THRESHOLD,
                "weighting": "uniform",
            }
        )
        for source_index, probability, is_assigned in zip(
            cohort, p_blood, assigned
        ):
            cell_rows.append(
                {
                    "method": METHOD_LABELS[method],
                    "source_index": int(source_index),
                    "p_blood_5nn": float(probability),
                    "assigned_to_blood": bool(is_assigned),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(cell_rows), provenance


def plot_figure(
    scores: pd.DataFrame,
    category_row: pd.Series,
    summary: pd.DataFrame,
) -> plt.Figure:
    configure_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(8.25, 2.55),
        gridspec_kw={
            "width_ratios": (1.00, 0.93, 1.52),
            "left": 0.105,
            "right": 0.995,
            "bottom": 0.27,
            "top": 0.86,
            "wspace": 0.39,
        },
    )

    # Preserve the two original regulatory-state panels exactly.
    ax = axes[0]
    violin_data = (
        scores["neural_matched_bg_open0"].to_numpy(dtype=float),
        scores["blood_endothelial_matched_bg_open0"].to_numpy(dtype=float),
    )
    violin_colors = (NEURAL_PROGRAM_COLOR, HEP_COLOR)
    parts = ax.violinplot(
        violin_data,
        positions=(0, 1),
        widths=0.72,
        showmeans=False,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], violin_colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_linewidth(0.6)
        body.set_alpha(0.25)
    rng = np.random.default_rng(0)
    for position, values, color in zip((0, 1), violin_data, violin_colors):
        sampled = rng.choice(values, size=min(250, len(values)), replace=False)
        ax.scatter(
            position + rng.normal(0.0, 0.045, size=len(sampled)),
            sampled,
            s=4.0,
            color=color,
            alpha=0.34,
            linewidths=0.0,
        )
        mean = float(np.mean(values))
        ax.scatter(position, mean, marker="D", s=24.0, color="#000000", zorder=4)
        ax.annotate(
            f"{mean:.2f}",
            (position, mean),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title("Starting TSS accessibility", pad=5)
    ax.set_ylabel("Above-background\nTSS openness")
    ax.set_xticks((0, 1), ("Neural", "Blood/endo"))
    ax.set_ylim(-0.02, 1.50)
    ax.set_yticks((0.0, 0.5, 1.0, 1.5))
    clean_axis(ax)

    ax = axes[1]
    class_labels = ("Neural open", "Blood/endo open", "Ambiguous")
    class_values = 100.0 * np.asarray(
        (
            category_row["frac_neural_open"],
            category_row["frac_blood_open"],
            category_row["frac_ambiguous"],
        ),
        dtype=float,
    )
    class_counts = np.asarray(
        (
            category_row["n_neural_open"],
            category_row["n_blood_open"],
            category_row["n_ambiguous"],
        ),
        dtype=int,
    )
    class_x = np.arange(3)
    bars = ax.bar(
        class_x,
        class_values,
        width=0.64,
        color=(NEURAL_PROGRAM_COLOR, HEP_COLOR, AMBIGUOUS_COLOR),
        linewidth=0.0,
    )
    for bar, value, count in zip(bars, class_values, class_counts):
        ax.annotate(
            f"{value:.1f}%\n(n={count})",
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            linespacing=0.95,
        )
    ax.set_title("Starting regulatory state", pad=5)
    ax.set_ylabel("% of rostral starts")
    ax.set_xticks(class_x, class_labels, rotation=16, ha="right")
    ax.set_xlim(-0.70, 2.50)
    ax.set_ylim(0, 92)
    ax.set_yticks((0, 40, 80))
    clean_axis(ax)

    ax = axes[2]
    values = summary["percent_assigned_blood"].to_numpy(dtype=float)
    counts = summary["n_assigned_blood"].to_numpy(dtype=int)
    x = np.arange(len(summary), dtype=float)
    bars = []
    for index, (method, value) in enumerate(
        zip(METHOD_ORDER, values)
    ):
        style = method_style(METHOD_STYLE_KEYS[method])
        bar = ax.bar(
            index,
            value,
            width=0.62,
            color=style.color,
            linewidth=0.0,
        )[0]
        bars.append(bar)
    annotation_offsets = (-0.07, 0.07, -0.04, 0.04)
    for bar, value, count, x_offset in zip(
        bars, values, counts, annotation_offsets
    ):
        ax.annotate(
            f"{value:.1f}%\n(n={count})",
            (bar.get_x() + bar.get_width() / 2.0 + x_offset, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            linespacing=0.95,
        )
    ax.set_title("Assigned to blood/endothelium\nHeld-out E8.5", pad=5)
    ax.set_ylabel("% of neural-open starts")
    ax.set_xticks(x, summary["method"], rotation=19, ha="right")
    ax.set_xlim(-0.62, len(summary) - 0.38)
    ax.set_ylim(0, max(21.5, float(np.max(values)) + 4.0))
    ax.set_yticks((0, 10, 20))
    clean_axis(ax)
    return fig


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / "rostral_tss_neural_open_e85_blood_methods.pdf"
    png_path = args.output_dir / "rostral_tss_neural_open_e85_blood_methods.png"
    summary_path = args.output_dir / "neural_open_e85_blood_summary.csv"
    cell_path = args.output_dir / "neural_open_e85_blood_per_cell.csv.gz"
    manifest_path = args.output_dir / "neural_open_e85_blood_manifest.json"
    ensure_outputs(
        [pdf_path, png_path, summary_path, cell_path, manifest_path],
        args.overwrite,
    )

    scores, category_row, _ = select_tss_inputs(args.tss_dir)
    summary, per_cell, provenance = compute_e85_blood_routing(scores)
    figure = plot_figure(scores, category_row, summary)
    figure.savefig(pdf_path, bbox_inches=None)
    figure.savefig(png_path, dpi=600, bbox_inches=None)
    plt.close(figure)
    summary.to_csv(summary_path, index=False)
    per_cell.to_csv(cell_path, index=False)

    manifest = {
        "analysis": "strict-LOO E8.5 blood routing of neural-open E7.5 rostral starts",
        "source_cohort": {
            "definition": (
                "neural matched-background openness >= 0 and neural-minus-"
                "blood openness >= 0.1"
            ),
            "n": int(category_row["n_neural_open"]),
        },
        "readout": {
            "reference": "real held-out E8.5 cells in shared normalized 50D RNA PCA",
            "k": K,
            "assigned_to_blood_if": "at least 3 of 5 nearest neighbors are blood/endothelial",
            "p_blood_threshold": P_BLOOD_THRESHOLD,
            "blood_celltypes": sorted(BLOOD_ERYTHROID),
        },
        "method_order": list(METHOD_ORDER),
        "prediction_provenance": provenance,
        "mioflow": {
            "version": "strict-LOO2 GAGA10D 20k",
            "checkpoint": str(
                ROOT
                / "results/mioflow_gastrulation_loo_time2_pca_gaga10_n1024_20000/model.pt"
            ),
            "workflow": "shared PCA -> GAGA10 -> MIOFlow -> decoder -> shared PCA",
            "heldout_model_time": "5/3 on the rank clock after removing E8.5",
        },
        "figure_size_inches": [8.25, 2.55],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(pdf_path)


if __name__ == "__main__":
    main()
