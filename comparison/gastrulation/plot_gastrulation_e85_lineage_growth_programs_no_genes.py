#!/usr/bin/env python
"""Plot compact E8.5 growth-program associations without gene lists."""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from trainfbench_plot_style import NATURE_CUD, apply_nature_rc


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODULES = (
    ROOT
    / "results/gastrulation_all_source_stage_growth_gene_programs/"
    "module_associations_by_setting.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/gastrulation_all_source_stage_growth_gene_programs/figures"
)
DEFAULT_PAPER_OUTPUT = ROOT / "output/pdf/gastrulation_growth_programs"

CELLTYPES = [
    "Cardiomyocytes",
    "Erythroid1",
    "Erythroid2",
    "Forebrain/Midbrain/Hindbrain",
    "Paraxial mesoderm",
    "Spinal cord",
    "Surface ectoderm",
    "Somitic mesoderm",
]
DISPLAY_CELLTYPES = {
    "Forebrain/Midbrain/Hindbrain": "Fore-/mid-/hindbrain",
}
MODULES = [
    ("S phase", "S phase", NATURE_CUD["blue"], "o", -0.10),
    ("G2/M", "G2/M", NATURE_CUD["bluish_green"], "^", 0.10),
    (
        "Lineage maturation (non-cycle)",
        "Maturation",
        NATURE_CUD["vermillion"],
        "D",
        0.0,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", type=Path, default=DEFAULT_MODULES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--paper-output-dir", type=Path, default=DEFAULT_PAPER_OUTPUT
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def signed(value: float) -> str:
    return f"{value:+.2f}"


def configure_style() -> None:
    apply_nature_rc(font_size=10)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "normal",
            "axes.labelsize": 12,
            "axes.labelweight": "normal",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial",
        }
    )


def prepare_values(
    modules: pd.DataFrame, *, mode: str
) -> tuple[pd.DataFrame, str]:
    selected = modules[
        modules["stage"].eq("E8.5")
        & modules["method"].eq("USOT Sync")
        & modules["celltype"].isin(CELLTYPES)
        & modules["module"].isin([item[0] for item in MODULES])
    ].copy()
    c_y_values = np.sort(selected["C_y"].unique().astype(float))
    expected = len(CELLTYPES) * len(MODULES) * len(c_y_values)
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} rows but found {len(selected)}")

    if mode == "cy03":
        values = selected[np.isclose(selected["C_y"], 0.3)].copy()
        if len(values) != len(CELLTYPES) * len(MODULES):
            raise ValueError("The C_y=0.3 table is incomplete")
        values = values.rename(columns={"partial_spearman": "value"})
        values["value_min"] = values["value"]
        values["value_max"] = values["value"]
        title = r"COATI unbal. ($C_y=0.3$)"
    elif mode == "sweep":
        values = (
            selected.groupby(["celltype", "module"], as_index=False)
            .agg(
                value=("partial_spearman", "median"),
                value_min=("partial_spearman", "min"),
                value_max=("partial_spearman", "max"),
            )
        )
        values["C_y"] = np.nan
        title = r"COATI unbal.: $C_y$ median (min--max)"
    else:
        raise ValueError(mode)
    return values, title


def plot(values: pd.DataFrame, *, title: str, show_ranges: bool) -> plt.Figure:
    configure_style()
    fig = plt.figure(figsize=(4.10, 2.82), constrained_layout=False)
    ax = fig.add_axes([0.34, 0.17, 0.64, 0.60])
    y = np.arange(len(CELLTYPES), dtype=float)

    phase_values = values[values["module"].isin(["S phase", "G2/M"])].pivot(
        index="celltype", columns="module", values="value"
    )
    for module, display, color, marker, y_offset in MODULES:
        local = values[values["module"].eq(module)].set_index("celltype")
        point = local.loc[CELLTYPES, "value"].to_numpy(float)
        minimum = local.loc[CELLTYPES, "value_min"].to_numpy(float)
        maximum = local.loc[CELLTYPES, "value_max"].to_numpy(float)
        yy = y + y_offset
        if show_ranges:
            ax.hlines(
                yy,
                minimum,
                maximum,
                color=color,
                linewidth=1.35,
                alpha=0.72,
                zorder=2,
            )
        ax.scatter(
            point,
            yy,
            s=34,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )
        for celltype, value, y_value in zip(CELLTYPES, point, yy):
            if module in {"S phase", "G2/M"}:
                other = "G2/M" if module == "S phase" else "S phase"
                place_left = value < float(phase_values.loc[celltype, other])
            else:
                place_left = value < 0
            ax.text(
                value - 0.018 if place_left else value + 0.018,
                y_value,
                signed(value),
                ha="right" if place_left else "left",
                va="center",
                color="#2B2B2B",
                fontsize=10,
                fontfamily="Arial",
                fontweight="normal",
            )

    ax.axvline(0, color="#5A5A5A", linewidth=0.9, zorder=0)
    ax.set_xlim(-0.86, 0.94)
    ax.set_xticks([-0.8, -0.4, 0.0, 0.4, 0.8])
    ax.set_xlabel(r"Partial Spearman $\rho(g,\mathrm{program})$")
    ax.set_yticks(
        y,
        [DISPLAY_CELLTYPES.get(celltype, celltype) for celltype in CELLTYPES],
    )
    ax.invert_yaxis()
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", width=0.8, length=3)
    ax.set_title(title, pad=7, fontweight="normal")

    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linewidth=1.7,
            markersize=5.5,
            markeredgecolor="white",
            label=display,
        )
        for _, display, color, marker, _ in MODULES
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.66, 0.985),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        columnspacing=0.65,
        handletextpad=0.35,
        prop={"family": "Arial", "size": 10},
    )
    return fig


def save(
    fig: plt.Figure,
    values: pd.DataFrame,
    *,
    stem: str,
    output_dir: Path,
    paper_output_dir: Path,
) -> None:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    csv = output_dir / f"{stem}_values.csv"
    fig.savefig(
        png,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(fig)
    values.to_csv(csv, index=False)
    shutil.copyfile(pdf, paper_output_dir / pdf.name)
    print(png)
    print(pdf)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.paper_output_dir.mkdir(parents=True, exist_ok=True)
    stems = {
        "cy03": "e85_lineage_growth_program_associations_cy03_no_genes",
        "sweep": "e85_lineage_growth_program_associations_cy_sweep_no_genes",
    }
    targets = [
        path
        for stem in stems.values()
        for path in (
            args.output_dir / f"{stem}.png",
            args.output_dir / f"{stem}.pdf",
            args.output_dir / f"{stem}_values.csv",
            args.paper_output_dir / f"{stem}.pdf",
        )
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")

    modules = pd.read_csv(args.modules)
    for mode, stem in stems.items():
        values, title = prepare_values(modules, mode=mode)
        figure = plot(values, title=title, show_ranges=(mode == "sweep"))
        save(
            figure,
            values,
            stem=stem,
            output_dir=args.output_dir,
            paper_output_dir=args.paper_output_dir,
        )


if __name__ == "__main__":
    main()
