#!/usr/bin/env python
"""Plot palate LOO cell-type composition predictions across methods."""

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
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from trainfbench_plot_style import (
    PALATE_CELLTYPE_COLORS,
    PALATE_CELLTYPE_LABELS,
    PALATE_CELLTYPE_ORDER,
)


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results/palate_loo_celltype_composition"
DEFAULT_OUTPUT = DEFAULT_INPUT / "figures"
METHOD_ORDER = [
    "Observed",
    "USOT best",
    "USOT worst",
    "BalancedSync best",
    "BalancedSync worst",
    "TrajectoryNet",
    "MIOFlow",
    "CytoBridge balanced",
    "CytoBridge unbalanced",
]
METHOD_LABEL = {
    "Observed": "Observed",
    "USOT best": "COATI-U best",
    "USOT worst": "COATI-U worst",
    "BalancedSync best": "COATI-B best",
    "BalancedSync worst": "COATI-B worst",
    "TrajectoryNet": "TrajectoryNet (T)",
    "MIOFlow": "MIOFlow",
    "CytoBridge balanced": "CytoBridge-B",
    "CytoBridge unbalanced": "CytoBridge-U",
}
CELLTYPE_ORDER = list(PALATE_CELLTYPE_ORDER)
CELLTYPE_LABEL = dict(PALATE_CELLTYPE_LABELS)
CELLTYPE_COLOR = dict(PALATE_CELLTYPE_COLORS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_joint_extremes(scores: pd.DataFrame) -> pd.DataFrame:
    native = scores[scores["weighting"].eq("native")].copy()
    selections: list[dict[str, object]] = []
    for scenario in sorted(native["scenario"].unique()):
        for method in ("USOT", "BalancedSync"):
            local = native[
                native["scenario"].eq(scenario)
                & native["method"].eq(method)
            ]
            joint = local.groupby("cy", observed=True)[
                "composition_jsd"
            ].mean()
            for selection_kind, selected_cy in (
                ("best", float(joint.idxmin())),
                ("worst", float(joint.idxmax())),
            ):
                selections.append(
                    {
                        "scenario": scenario,
                        "method": method,
                        "selection_kind": selection_kind,
                        "display_method": (
                            f"{'COATI-U' if method == 'USOT' else 'COATI-B'} "
                            f"{selection_kind}"
                        ),
                        "selected_cy": selected_cy,
                        "joint_mean_RNA_ATAC_jsd": float(
                            joint.loc[selected_cy]
                        ),
                        "selection_rule": (
                            f"{selection_kind} mean native-mass JSD "
                            "across RNA and ATAC"
                        ),
                    }
                )
    return pd.DataFrame(selections)


def build_display(
    scores: pd.DataFrame,
    by_celltype: pd.DataFrame,
    observed: pd.DataFrame,
    selections: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = scores[scores["weighting"].eq("native")].copy()
    by_celltype = by_celltype[
        by_celltype["weighting"].eq("native")
    ].copy()
    score_rows: list[pd.DataFrame] = []
    comp_rows: list[pd.DataFrame] = []
    for scenario in sorted(scores["scenario"].unique()):
        stage = str(
            scores.loc[
                scores["scenario"].eq(scenario), "heldout_stage"
            ].iloc[0]
        )
        reference = observed[observed["scenario"].eq(scenario)].copy()
        for modality in ("RNA", "ATAC"):
            obs_score = pd.DataFrame(
                [
                    {
                        "scenario": scenario,
                        "heldout_stage": stage,
                        "method": "Observed",
                        "cy": np.nan,
                        "modality": modality,
                        "composition_jsd": 0.0,
                    }
                ]
            )
            score_rows.append(obs_score)
            obs_comp = reference[
                ["scenario", "heldout_stage", "celltype"]
            ].copy()
            obs_comp["method"] = "Observed"
            obs_comp["cy"] = np.nan
            obs_comp["modality"] = modality
            obs_comp["predicted_fraction"] = reference[
                "observed_fraction"
            ].to_numpy(float)
            comp_rows.append(obs_comp)

            for display_method in METHOD_ORDER[1:]:
                if display_method.startswith("USOT "):
                    method = "USOT"
                    selection_kind = display_method.removeprefix("USOT ")
                elif display_method.startswith("BalancedSync "):
                    method = "BalancedSync"
                    selection_kind = display_method.removeprefix(
                        "BalancedSync "
                    )
                else:
                    method = display_method
                    selection_kind = None
                local_score = scores[
                    scores["scenario"].eq(scenario)
                    & scores["modality"].eq(modality)
                    & scores["method"].eq(method)
                ]
                local_comp = by_celltype[
                    by_celltype["scenario"].eq(scenario)
                    & by_celltype["modality"].eq(modality)
                    & by_celltype["method"].eq(method)
                ]
                if selection_kind is not None:
                    selected_cy = float(
                        selections.loc[
                            selections["scenario"].eq(scenario)
                            & selections["selection_kind"].eq(
                                selection_kind
                            )
                            & selections["method"].eq(method),
                            "selected_cy",
                        ].iloc[0]
                    )
                    local_score = local_score[
                        np.isclose(local_score["cy"], selected_cy)
                    ]
                    local_comp = local_comp[
                        np.isclose(local_comp["cy"], selected_cy)
                    ]
                if len(local_score) != 1:
                    raise ValueError(
                        f"Expected one score: {scenario} {modality} "
                        f"{method}; got {len(local_score)}"
                    )
                local_score = local_score.copy()
                local_comp = local_comp.copy()
                local_score["source_method"] = method
                local_comp["source_method"] = method
                local_score["method"] = display_method
                local_comp["method"] = display_method
                score_rows.append(local_score)
                comp_rows.append(local_comp)
    display_scores = pd.concat(
        score_rows, ignore_index=True, sort=False
    )
    display_comp = pd.concat(
        comp_rows, ignore_index=True, sort=False
    )
    for frame in (display_scores, display_comp):
        frame["display_label"] = frame["method"].map(METHOD_LABEL)
    return display_scores, display_comp


def panel(
    ax: plt.Axes,
    scenario: str,
    modality: str,
    scores: pd.DataFrame,
    composition: pd.DataFrame,
    show_ylabels: bool,
) -> None:
    local_scores = scores[
        scores["scenario"].eq(scenario)
        & scores["modality"].eq(modality)
    ].set_index("method")
    local_comp = composition[
        composition["scenario"].eq(scenario)
        & composition["modality"].eq(modality)
    ]
    y = np.arange(len(METHOD_ORDER))
    left = np.zeros(len(METHOD_ORDER), dtype=float)
    for celltype in CELLTYPE_ORDER:
        values = []
        for method in METHOD_ORDER:
            value = local_comp[
                local_comp["method"].eq(method)
                & local_comp["celltype"].eq(celltype)
            ]["predicted_fraction"]
            values.append(float(value.iloc[0]) if len(value) else 0.0)
        values_array = np.asarray(values)
        ax.barh(
            y,
            values_array,
            left=left,
            color=CELLTYPE_COLOR[celltype],
            height=0.86,
            edgecolor="white",
            linewidth=0.35,
        )
        left += values_array

    labels = []
    for method in METHOD_ORDER:
        labels.append(METHOD_LABEL[method])
    ax.set_yticks(y)
    ax.set_yticklabels(labels if show_ylabels else [])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.55)
    fraction_ticks = np.array([0.0, 0.5, 1.0])
    ax.set_xticks(fraction_ticks)
    ax.set_xticklabels([f"{x:.0%}" for x in fraction_ticks])
    ax.grid(axis="x", color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for row, method in enumerate(METHOD_ORDER):
        if method == "Observed":
            continue
        jsd = float(local_scores.loc[method, "composition_jsd"])
        ax.text(
            1.015,
            row,
            f"{jsd:.5f}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#1A1A1A",
        )
    ax.text(
        1.015,
        -0.83,
        "JSD ↓",
        fontsize=10,
        fontweight="bold",
        ha="left",
    )
    ax.set_title(
        modality,
        fontsize=10,
        fontweight="bold",
        pad=4,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=10, pad=1)
    ax.tick_params(axis="x", labelsize=10)


def make_composition_figure(
    scores: pd.DataFrame,
    composition: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(105.0 / 25.4, 4.7),
        sharex=True,
        constrained_layout=False,
    )
    scenarios = [
        ("loo_time1", "Held-out E13.5"),
        ("loo_time2", "Held-out E14.0"),
    ]
    for row, (scenario, stage_label) in enumerate(scenarios):
        for col, modality in enumerate(("RNA", "ATAC")):
            panel(
                axes[row, col],
                scenario,
                modality,
                scores,
                composition,
                show_ylabels=(col == 0),
            )
        axes[row, 0].text(
            -0.40,
            1.025,
            stage_label,
            transform=axes[row, 0].transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
        )

    handles = [
        Patch(
            facecolor=CELLTYPE_COLOR[celltype],
            edgecolor="none",
            label=CELLTYPE_LABEL[celltype],
        )
        for celltype in CELLTYPE_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.52, 0.985),
        frameon=False,
        fontsize=10,
        columnspacing=0.8,
        handletextpad=0.4,
        labelspacing=0.25,
        borderaxespad=0.0,
    )
    fig.text(
        0.54,
        0.025,
        "Predicted cell-type fraction",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(
        left=0.30,
        right=0.995,
        top=0.83,
        bottom=0.105,
        hspace=0.18,
        wspace=0.10,
    )
    fig.savefig(output, dpi=300, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def make_cy_figure(
    all_scores: pd.DataFrame,
    output: Path,
) -> None:
    native = all_scores[
        all_scores["weighting"].eq("native")
        & all_scores["method"].isin(["USOT", "BalancedSync"])
    ].copy()
    fig, axes = plt.subplots(
        1, 2, figsize=(6.2, 2.8), sharey=False, constrained_layout=True
    )
    colors = {"USOT": "#E66101", "BalancedSync": "#377EB8"}
    for ax, (scenario, title) in zip(
        axes,
        [("loo_time1", "Held-out E13.5"), ("loo_time2", "Held-out E14.0")],
    ):
        local = native[native["scenario"].eq(scenario)]
        for method in ("USOT", "BalancedSync"):
            values = (
                local[local["method"].eq(method)]
                .groupby("cy", observed=True)["composition_jsd"]
                .mean()
                .sort_index()
            )
            ax.plot(
                values.index,
                values.values,
                marker="o",
                linewidth=2,
                markersize=5,
                color=colors[method],
                label=(
                    "COATI-U"
                    if method == "USOT"
                    else "COATI-B"
                ),
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("$C_y$")
        ax.set_ylabel("Mean RNA–ATAC composition JSD ↓")
        ax.grid(color="#dddddd", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.savefig(output, dpi=300, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        args.output_dir / "palate_loo_celltype_composition.png",
        args.output_dir / "palate_loo_celltype_composition.pdf",
        args.output_dir / "palate_loo_celltype_composition.svg",
        args.output_dir / "palate_loo_celltype_composition_by_cy.png",
        args.output_dir / "palate_loo_celltype_composition_by_cy.pdf",
        args.output_dir / "palate_loo_celltype_composition_display_scores.csv",
        args.output_dir / "palate_loo_celltype_composition_display_values.csv",
        args.output_dir / "palate_loo_celltype_composition_selected_cy.csv",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite: " + ", ".join(map(str, existing))
        )
    scores = pd.read_csv(
        args.input_dir / "palate_loo_celltype_composition_scores.csv"
    )
    by_celltype = pd.read_csv(
        args.input_dir
        / "palate_loo_celltype_composition_by_celltype.csv"
    )
    observed = pd.read_csv(
        args.input_dir / "palate_loo_observed_celltype_composition.csv"
    )
    selections = select_joint_extremes(scores)
    display_scores, display_values = build_display(
        scores, by_celltype, observed, selections
    )
    selections.to_csv(outputs[7], index=False)
    display_scores.to_csv(outputs[5], index=False)
    display_values.to_csv(outputs[6], index=False)
    make_composition_figure(
        display_scores, display_values, outputs[0]
    )
    make_cy_figure(scores, outputs[3])
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
