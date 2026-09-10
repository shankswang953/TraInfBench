#!/usr/bin/env python
"""Plot the strict-LOO1 Rostral-to-E8.0 off-lineage summary as a figure."""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import pandas as pd

from trainfbench_plot_style import METHOD_STYLES


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results/gastrulation_strict_loo1_rostral_endpoint_offlineage"
INPUT = RESULT_DIR / "strict_loo1_rostral_to_e80_offlineage_k5.csv"
OUTPUT_STEM = RESULT_DIR / "strict_loo1_rostral_to_e80_offlineage_summary"

METHOD_ORDER = (
    "COATI bal.",
    "COATI unbal.",
    "OT(RNA)",
    "UOT(RNA)",
    "CytoBridge bal.",
    "CytoBridge unbal.",
    "MIOFlow (GAGA10D)",
)


def method_artist(method: str) -> tuple[str, str, str, str]:
    """Return colour, marker, marker face, and hatch for one displayed method."""

    if method == "COATI bal.":
        style = METHOD_STYLES["COATI balanced"]
        return style.color, style.marker, style.color, ""
    if method == "COATI unbal.":
        style = METHOD_STYLES["COATI unbalanced"]
        return style.color, style.marker, style.color, "//"
    if method == "OT(RNA)":
        style = METHOD_STYLES["OT baseline RNA"]
        return style.color, style.marker, "white", ""
    if method == "UOT(RNA)":
        style = METHOD_STYLES["COATI unbalanced"]
        return style.color, style.marker, "white", "//"
    if method == "CytoBridge bal.":
        style = METHOD_STYLES["CytoBridge balanced"]
        return style.color, style.marker, style.color, ""
    if method == "CytoBridge unbal.":
        style = METHOD_STYLES["CytoBridge unbalanced"]
        return style.color, style.marker, style.color, "//"
    if method == "MIOFlow (GAGA10D)":
        style = METHOD_STYLES["MIOFlow"]
        return style.color, style.marker, style.color, ""
    raise KeyError(method)


def pct(value: float) -> str:
    value *= 100.0
    return "<0.01%" if 0.0 < value < 0.01 else f"{value:.2f}%"


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "hatch.linewidth": 0.55,
        }
    )

    table = pd.read_csv(INPUT).set_index("method").loc[list(METHOD_ORDER)].copy()

    # A4-width figure: exact values remain readable at 10 pt.
    fig = plt.figure(figsize=(8.15, 3.18), facecolor="white")
    trans = fig.transFigure

    fig.text(
        0.5,
        0.955,
        "Rostral-to-E8.0 gross cross-lineage leakage",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="normal",
    )
    fig.text(
        0.5,
        0.895,
        "E7.5 rostral neurectoderm to held-out E8.0; soft 5-NN readout",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#4D4D4D",
    )

    x_method = 0.037
    x_bar0, x_bar1, x_gross_value = 0.224, 0.323, 0.382
    x_meso, x_blood, x_endo, x_top = 0.470, 0.580, 0.682, 0.765
    header_y = 0.810
    fig.text(x_method, header_y, "Method", ha="left", va="center", fontsize=10)
    fig.text(
        (x_bar0 + x_gross_value) / 2,
        header_y,
        "Gross off-lineage",
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.text(x_meso, header_y, "Mesoderm", ha="center", va="center", fontsize=10)
    fig.text(
        x_blood,
        header_y,
        "Blood/erythroid",
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.text(x_endo, header_y, "Endoderm", ha="center", va="center", fontsize=10)
    fig.text(
        x_top,
        header_y,
        "Dominant destination",
        ha="left",
        va="center",
        fontsize=10,
    )
    fig.add_artist(
        Line2D([0.025, 0.982], [0.775, 0.775], transform=trans, color="#666666", lw=0.65)
    )

    row_ys = [0.714, 0.625, 0.536, 0.447, 0.358, 0.269, 0.180]
    bar_height = 0.029
    gross_max = 0.40
    for row_index, (method, y) in enumerate(zip(METHOD_ORDER, row_ys)):
        row = table.loc[method]
        color, marker, face, hatch = method_artist(method)

        fig.add_artist(
            Line2D(
                [x_method - 0.012],
                [y],
                transform=trans,
                marker=marker,
                markersize=5.5,
                markerfacecolor=face,
                markeredgecolor=color,
                markeredgewidth=0.9,
                linestyle="none",
            )
        )
        fig.text(x_method, y, method, ha="left", va="center", fontsize=10)

        gross = float(row["gross_cross_lineage_leakage_fraction"])
        fig.add_artist(
            Rectangle(
                (x_bar0, y - bar_height / 2),
                x_bar1 - x_bar0,
                bar_height,
                transform=trans,
                facecolor="#F0F0F0",
                edgecolor="none",
            )
        )
        fig.add_artist(
            Rectangle(
                (x_bar0, y - bar_height / 2),
                (x_bar1 - x_bar0) * min(gross / gross_max, 1.0),
                bar_height,
                transform=trans,
                facecolor=color,
                edgecolor=color,
                linewidth=0.45,
                hatch=hatch,
            )
        )
        fig.text(
            x_gross_value,
            y,
            pct(gross),
            ha="right",
            va="center",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15},
        )
        fig.text(
            x_meso,
            y,
            pct(float(row["mesoderm_leakage_fraction"])),
            ha="center",
            va="center",
            fontsize=10,
        )
        fig.text(
            x_blood,
            y,
            pct(float(row["blood_erythroid_leakage_fraction"])),
            ha="center",
            va="center",
            fontsize=10,
        )
        fig.text(
            x_endo,
            y,
            pct(float(row["endoderm_leakage_fraction"])),
            ha="center",
            va="center",
            fontsize=10,
        )
        top_destination = str(row["top_off_lineage_destination"])
        top_value = pct(float(row["top_off_lineage_destination_fraction"]))
        fig.text(
            x_top,
            y,
            f"{top_destination} ({top_value})",
            ha="left",
            va="center",
            fontsize=9.5,
        )

        if row_index < len(row_ys) - 1:
            rule_y = (y + row_ys[row_index + 1]) / 2
            fig.add_artist(
                Line2D(
                    [0.025, 0.982],
                    [rule_y, rule_y],
                    transform=trans,
                    color="#E4E4E4",
                    lw=0.5,
                )
            )

    fig.text(
        0.025,
        0.073,
        "Gross off-lineage = mesoderm + blood/endothelium/erythroid + endoderm.",
        ha="left",
        va="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.text(
        0.982,
        0.073,
        "Bar scale: 0–40%",
        ha="right",
        va="center",
        fontsize=8.5,
        color="#555555",
    )

    for suffix, kwargs in (
        ("pdf", {}),
        ("png", {"dpi": 600}),
    ):
        fig.savefig(
            OUTPUT_STEM.with_suffix(f".{suffix}"),
            bbox_inches=None,
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)
    print(OUTPUT_STEM.with_suffix(".pdf"))
    print(OUTPUT_STEM.with_suffix(".png"))


if __name__ == "__main__":
    main()
