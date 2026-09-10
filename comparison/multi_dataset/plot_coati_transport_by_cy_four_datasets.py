#!/usr/bin/env python3
"""Plot balanced and unbalanced transport-only COATI energy for four datasets."""

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
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from trainfbench_plot_style import NATURE_CUD, apply_nature_rc


ROOT = Path(__file__).resolve().parents[2]
CY_GRID = np.round(np.arange(0.1, 1.0, 0.1), 1)
DATASET_ORDER = ("Gastrulation", "Palate", "Pancreas", "Human cerebral")
METHOD_ORDER = ("COATI bal.", "COATI unbal.")
SPACE_ORDER = ("RNA", "ATAC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gastrulation-csv",
        type=Path,
        default=ROOT
        / "results/gastrulation_coati_energy_by_cy/"
        "coati_energy_by_cy_per_seed_seed1.csv",
    )
    parser.add_argument(
        "--palate-csv",
        type=Path,
        default=ROOT
        / "results/palate_full_seed0_energy_by_cy_current/energy_by_cy_seed0.csv",
    )
    parser.add_argument(
        "--pancreas-csv",
        type=Path,
        default=ROOT
        / "results/pancreas_full_seed0_energy_by_cy/energy_by_cy_seed0.csv",
    )
    parser.add_argument(
        "--human-cerebral-csv",
        type=Path,
        default=ROOT
        / "results/human_cerebral_7time_energy_by_cy_30k_40k/plotted_values.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/coati_transport_by_cy_four_datasets",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def standardize(path: Path, dataset: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    if dataset == "Gastrulation":
        required = {
            "display_name",
            "seed",
            "C_y",
            "space",
            "kinetic_energy",
        }
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        table = table.loc[table["display_name"].isin(METHOD_ORDER)].copy()
        table = table.rename(
            columns={
                "display_name": "method",
                "kinetic_energy": "transport_energy",
            }
        )
        table["space"] = table["space"].map(
            {"Primary": "RNA", "Secondary": "ATAC"}
        )
    elif dataset in {"Palate", "Pancreas"}:
        required = {"method", "seed", "C_y", "space", "kinetic_energy"}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        table = table.loc[table["method"].isin(METHOD_ORDER)].copy()
        table = table.rename(columns={"kinetic_energy": "transport_energy"})
    else:
        required = {"method", "seed", "C_y", "space", "transport_energy"}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        table = table.loc[table["method"].isin(METHOD_ORDER)].copy()

    if table.empty:
        raise ValueError(f"No COATI rows in {path}")
    if table["space"].isna().any() or set(table["space"]) != set(SPACE_ORDER):
        raise ValueError(f"{dataset}: unexpected spaces {table['space'].unique()}")
    if not np.isfinite(table["transport_energy"].astype(float)).all():
        raise ValueError(f"{dataset}: non-finite transport energy")

    keep = ["method", "seed", "C_y", "space", "transport_energy"]
    if "iterations" in table.columns:
        keep.insert(2, "iterations")
    result = table[keep].copy()
    if "iterations" not in result.columns:
        result.insert(2, "iterations", np.nan)
    result.insert(0, "dataset", dataset)
    result["source_csv"] = str(path.resolve())
    result = result.sort_values(["method", "space", "C_y"]).reset_index(drop=True)

    keys = ["method", "space", "C_y"]
    if result.duplicated(keys).any():
        raise ValueError(f"{dataset}: duplicate method/space/C_y rows")
    for method in METHOD_ORDER:
        for space in SPACE_ORDER:
            local = result.loc[
                result["method"].eq(method) & result["space"].eq(space)
            ]
            if len(local) != len(CY_GRID) or not np.allclose(local["C_y"], CY_GRID):
                raise ValueError(f"{dataset}, {method}, {space}: incomplete C_y grid")
    return result


def configure_style() -> None:
    apply_nature_rc(font_size=10.0)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 10.0,
            "font.weight": "normal",
            "axes.titlesize": 10.0,
            "axes.titleweight": "normal",
            "axes.labelsize": 10.0,
            "axes.labelweight": "normal",
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.0,
            "figure.titlesize": 10.0,
            "figure.titleweight": "normal",
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot(data: pd.DataFrame, output_dir: Path) -> list[Path]:
    styles = {
        "RNA": (NATURE_CUD["blue"], "o"),
        "ATAC": (NATURE_CUD["vermillion"], "D"),
    }
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 9.2))
    for row, dataset in enumerate(DATASET_ORDER):
        for col, method in enumerate(METHOD_ORDER):
            ax = axes[row, col]
            local = data.loc[
                data["dataset"].eq(dataset) & data["method"].eq(method)
            ]
            for space in SPACE_ORDER:
                values = local.loc[local["space"].eq(space)].sort_values("C_y")
                color, marker = styles[space]
                ax.plot(
                    values["C_y"],
                    values["transport_energy"],
                    color=color,
                    marker=marker,
                    linewidth=1.45,
                    markersize=4.1,
                    markeredgewidth=0.6,
                    label=space,
                )
            ax.set_title(method, pad=4.0, fontweight="normal")
            ax.set_xlabel(r"$C_y$")
            ax.set_ylabel("Energy")
            ax.set_xticks(CY_GRID)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.tick_params(direction="out", pad=2.0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.055,
        top=0.925,
        hspace=0.72,
        wspace=0.28,
    )
    fig.canvas.draw()
    for row, dataset in enumerate(DATASET_ORDER):
        left = axes[row, 0].get_position()
        right = axes[row, 1].get_position()
        fig.text(
            (left.x0 + right.x1) / 2.0,
            left.y1 + 0.028,
            dataset,
            ha="center",
            va="bottom",
            fontsize=10.0,
            fontweight="normal",
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.995),
        frameon=False,
        ncol=2,
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=1.3,
    )

    stem = output_dir / "coati_transport_by_cy_balanced_unbalanced_four_datasets"
    outputs = [stem.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg")]
    save = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.04}
    fig.savefig(outputs[0], dpi=600, **save)
    fig.savefig(outputs[1], **save)
    fig.savefig(outputs[2], **save)
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    sources = {
        "Gastrulation": args.gastrulation_csv.resolve(),
        "Palate": args.palate_csv.resolve(),
        "Pancreas": args.pancreas_csv.resolve(),
        "Human cerebral": args.human_cerebral_csv.resolve(),
    }
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "coati_transport_by_cy_balanced_unbalanced_four_datasets"
    expected = [stem.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg")]
    expected += [output_dir / "plotted_transport_values.csv", output_dir / "manifest.json"]
    existing = [path for path in expected if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs:\n"
            + "\n".join(f"  {path}" for path in existing)
        )

    configure_style()
    tables = {
        dataset: standardize(path, dataset) for dataset, path in sources.items()
    }
    combined = pd.concat(tables.values(), ignore_index=True)
    combined.to_csv(output_dir / "plotted_transport_values.csv", index=False)
    outputs = plot(combined, output_dir)
    manifest = {
        "dataset_order": list(DATASET_ORDER),
        "columns": list(METHOD_ORDER),
        "quantity": "COATI transport/kinetic energy only; growth excluded from both columns",
        "C_y": CY_GRID.tolist(),
        "style": {
            "font": "Arial",
            "font_size_pt": 10,
            "font_weight": "normal",
            "RNA": "blue circles",
            "ATAC": "vermillion diamonds",
            "shared_legend": True,
            "figure_size_inches": [7.2, 9.2],
        },
        "sources": {
            dataset: {"path": str(path), "sha256": sha256(path)}
            for dataset, path in sources.items()
        },
        "seeds": {
            dataset: sorted(table["seed"].astype(int).unique().tolist())
            for dataset, table in tables.items()
        },
        "iterations": {
            dataset: sorted(table["iterations"].dropna().astype(int).unique().tolist())
            for dataset, table in tables.items()
        },
        "qa": {
            "points_per_method_space_dataset": 9,
            "growth_excluded": True,
            "all_values_finite": True,
        },
        "outputs": [str(path.resolve()) for path in outputs],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(combined.groupby(["dataset", "method", "space"]).size().to_string())
    for path in outputs:
        print(path.resolve())


if __name__ == "__main__":
    main()
