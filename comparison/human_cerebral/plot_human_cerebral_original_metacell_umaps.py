#!/usr/bin/env python3
"""Plot seven-time original-cell and paired-metacell RNA/ATAC UMAPs.

RNA original cells and RNA metacells are placed in one frozen raw-PCA30 UMAP:
the reducer is fit on original cells and metacells are transformed.  The ATAC
panel contains paired metacells only because a time-course original-cell ATAC
matrix is not present locally.  A second figure shows the two paired-metacell
modalities in the corrected, normalized spaces used for model training.
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
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/trainfbench-hc-umap-numba")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/trainfbench-hc-umap-xdg")
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import umap

from trainfbench_plot_style import NATURE_CUD, apply_nature_rc


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = Path(
    "external/COATI/humanCerebral/Data/"
    "selected_4_7_9_11_12_18_21"
)
MODEL_DIR = Path(
    "external/COATI/humanCerebral/Data/"
    "processed_syncot_v2_no_d61"
)
ORIGINAL_RNA = (
    ROOT
    / "results/human_cerebral_original_cell_vs_metacell_frozen_pca30/"
    "original_single_cell_frozen_raw_pca30.npz"
)
DEFAULT_OUTPUT = ROOT / "results/human_cerebral_original_metacell_umaps"

DAYS = (4, 7, 9, 11, 12, 18, 21)
TIME_KEYS = tuple(f"age_{day}" for day in DAYS)
ATAC_DIMS_ZERO_BASED = np.asarray([1, 2, 3, 4, 5, 6, 9, 10, 11, 13, 21, 26])
RNA_SCALE = 63.89521587795941
ATAC_SCALE = 34.16607592332852

TIME_COLORS = {
    4: "#440154",
    7: "#414487",
    9: "#2A788E",
    11: "#22A884",
    12: "#7AD151",
    18: "#F6C445",
    21: "#E76F51",
}
CELLTYPE_ORDER = ("RG", "IPC", "EN", "IN", "Other")
CELLTYPE_COLORS = {
    "RG": NATURE_CUD["blue"],
    "IPC": NATURE_CUD["orange"],
    "EN": NATURE_CUD["bluish_green"],
    "IN": NATURE_CUD["vermillion"],
    "Other": "#999999",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coarse_celltype(values: np.ndarray) -> np.ndarray:
    labels = pd.Series(values, dtype="string").str.strip().str.upper()
    return labels.where(labels.isin(CELLTYPE_ORDER[:-1]), "Other").to_numpy(str)


def load_original_rna() -> tuple[np.ndarray, pd.DataFrame]:
    with np.load(ORIGINAL_RNA, allow_pickle=True) as archive:
        coordinates = np.asarray(archive["coordinates"], dtype=np.float32) / RNA_SCALE
        metadata = pd.DataFrame(
            {
                "id": archive["cell_id"].astype(str),
                "day": archive["age"].astype(int),
                "line": np.char.lower(archive["line"].astype(str)),
                "cell_type": coarse_celltype(archive["cell_type"].astype(str)),
            }
        )
    if len(metadata) != len(coordinates) or not np.isfinite(coordinates).all():
        raise ValueError("Invalid original-cell RNA PCA archive")
    return coordinates, metadata


def load_metacell_metadata() -> pd.DataFrame:
    obs = pd.read_csv(REFERENCE_DIR / "paired_obs.csv", low_memory=False)
    parts: list[pd.DataFrame] = []
    for key, day in zip(TIME_KEYS, DAYS, strict=True):
        part = obs.loc[obs["time_key"].astype(str).eq(key)].copy()
        parts.append(
            pd.DataFrame(
                {
                    "id": part["paired_metacell_id"].astype(str).to_numpy(),
                    "day": np.full(len(part), day, dtype=int),
                    "line": part["line"].astype(str).str.lower().to_numpy(),
                    "cell_type": coarse_celltype(
                        part["nowakowski_prediction"].astype(str).to_numpy()
                    ),
                }
            )
        )
    output = pd.concat(parts, ignore_index=True)
    if len(output) != 20_663:
        raise ValueError(f"Expected 20,663 seven-time metacells, got {len(output):,}")
    return output


def load_npz_blocks(path: Path, columns: slice | np.ndarray) -> np.ndarray:
    blocks: list[np.ndarray] = []
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in TIME_KEYS if key not in archive.files]
        if missing:
            raise ValueError(f"Missing time keys in {path}: {missing}")
        for key in TIME_KEYS:
            blocks.append(np.asarray(archive[key][:, columns], dtype=np.float32))
    output = np.vstack(blocks)
    if not np.isfinite(output).all():
        raise ValueError(f"Non-finite coordinates in {path}")
    return output


def make_reducer(args: argparse.Namespace, metric: str = "euclidean") -> umap.UMAP:
    return umap.UMAP(
        n_components=2,
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        metric=metric,
        random_state=args.seed,
        transform_seed=args.seed,
        low_memory=True,
        n_jobs=1,
    )


def fit_embeddings(args: argparse.Namespace) -> pd.DataFrame:
    original_x, original_meta = load_original_rna()
    metacell_meta = load_metacell_metadata()

    raw_rna_meta = load_npz_blocks(
        MODEL_DIR / "rna_pca50_raw_by_age.npz", slice(0, 30)
    ) / RNA_SCALE
    raw_atac_meta = load_npz_blocks(
        MODEL_DIR / "atac_lsi60_raw_by_age.npz", ATAC_DIMS_ZERO_BASED
    ) / ATAC_SCALE
    corrected_rna_meta = load_npz_blocks(
        REFERENCE_DIR / "rna_pca30_normalized_by_time.npz", slice(None)
    )
    corrected_atac_meta = load_npz_blocks(
        REFERENCE_DIR / "atac_lsi12_normalized_by_time.npz", slice(None)
    )
    expected = len(metacell_meta)
    for name, values in {
        "raw RNA metacell": raw_rna_meta,
        "raw ATAC metacell": raw_atac_meta,
        "corrected RNA metacell": corrected_rna_meta,
        "corrected ATAC metacell": corrected_atac_meta,
    }.items():
        if len(values) != expected:
            raise ValueError(f"{name}: {len(values):,} rows != {expected:,} metadata")

    # Reference-preserving RNA visualization: fit once on original cells and
    # transform the paired metacells.  No cell-type labels enter this fit.
    rna_reducer = make_reducer(args)
    original_rna_umap = rna_reducer.fit_transform(original_x)
    metacell_rna_umap = rna_reducer.transform(raw_rna_meta)

    # ATAC original cells are unavailable; this is a standalone metacell map.
    atac_reducer = make_reducer(args)
    metacell_atac_umap = atac_reducer.fit_transform(raw_atac_meta)

    # Actual corrected model spaces, fit independently by modality.
    model_rna_umap = make_reducer(args).fit_transform(corrected_rna_meta)
    model_atac_umap = make_reducer(args).fit_transform(corrected_atac_meta)

    frames: list[pd.DataFrame] = []
    definitions = (
        ("Original RNA", original_rna_umap, original_meta, "frozen_raw_PCA30"),
        ("Metacell RNA", metacell_rna_umap, metacell_meta, "frozen_raw_PCA30"),
        ("Metacell ATAC", metacell_atac_umap, metacell_meta, "frozen_raw_LSI12"),
        ("Model RNA", model_rna_umap, metacell_meta, "corrected_normalized_PCA30"),
        ("Model ATAC", model_atac_umap, metacell_meta, "corrected_normalized_LSI12"),
    )
    for representation, embedding, metadata, space in definitions:
        frame = metadata.copy()
        frame.insert(0, "representation", representation)
        frame.insert(1, "space", space)
        frame["UMAP1"] = embedding[:, 0]
        frame["UMAP2"] = embedding[:, 1]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def setup_style() -> None:
    apply_nature_rc(font_size=7.0)
    mpl.rcParams.update(
        {
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.55,
        }
    )


def style_umap_axis(axis: plt.Axes, title: str, show_y: bool) -> None:
    axis.set_title(title, pad=4)
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2" if show_y else "")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#777777")
        spine.set_linewidth(0.55)


def scatter_time(axis: plt.Axes, frame: pd.DataFrame, size: float) -> None:
    for day in DAYS:
        local = frame.loc[frame["day"].eq(day)]
        axis.scatter(
            local["UMAP1"],
            local["UMAP2"],
            s=size,
            c=TIME_COLORS[day],
            alpha=0.62,
            linewidths=0,
            rasterized=True,
        )


def scatter_celltype(axis: plt.Axes, frame: pd.DataFrame, size: float) -> None:
    # Draw the heterogeneous residual class first so core neural states remain visible.
    for label in ("Other", "IPC", "RG", "EN", "IN"):
        local = frame.loc[frame["cell_type"].eq(label)]
        axis.scatter(
            local["UMAP1"],
            local["UMAP2"],
            s=size,
            c=CELLTYPE_COLORS[label],
            alpha=0.60 if label == "Other" else 0.68,
            linewidths=0,
            rasterized=True,
        )


def time_legend_handles() -> list[Line2D]:
    return [
        Line2D([], [], marker="o", linestyle="none", markersize=4.2,
               markerfacecolor=TIME_COLORS[day], markeredgewidth=0, label=f"D{day}")
        for day in DAYS
    ]


def celltype_legend_handles() -> list[Line2D]:
    return [
        Line2D([], [], marker="o", linestyle="none", markersize=4.2,
               markerfacecolor=CELLTYPE_COLORS[label], markeredgewidth=0, label=label)
        for label in CELLTYPE_ORDER
    ]


def common_limits(frames: list[pd.DataFrame]) -> tuple[tuple[float, float], tuple[float, float]]:
    x = pd.concat([frame["UMAP1"] for frame in frames], ignore_index=True).to_numpy()
    y = pd.concat([frame["UMAP2"] for frame in frames], ignore_index=True).to_numpy()
    dx = max(float(x.max() - x.min()), 1e-6)
    dy = max(float(y.max() - y.min()), 1e-6)
    return (float(x.min() - 0.025 * dx), float(x.max() + 0.025 * dx)), (
        float(y.min() - 0.025 * dy), float(y.max() + 0.025 * dy)
    )


def draw_available_comparison(frame: pd.DataFrame, output: Path) -> None:
    setup_style()
    order = ("Original RNA", "Metacell RNA", "Metacell ATAC")
    titles = (
        "Original RNA cells",
        "Paired metacells · RNA",
        "Paired metacells · ATAC",
    )
    subsets = [frame.loc[frame["representation"].eq(name)].copy() for name in order]
    rna_limits = common_limits(subsets[:2])

    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.45))
    figure.subplots_adjust(left=0.06, right=0.995, bottom=0.15, top=0.91,
                           wspace=0.08, hspace=0.16)
    for column, (local, title) in enumerate(zip(subsets, titles, strict=True)):
        point_size = 1.25 if column == 0 else 1.55
        scatter_time(axes[0, column], local, point_size)
        scatter_celltype(axes[1, column], local, point_size)
        style_umap_axis(axes[0, column], title, show_y=column == 0)
        style_umap_axis(axes[1, column], "", show_y=column == 0)
        if column < 2:
            for axis in axes[:, column]:
                axis.set_xlim(*rna_limits[0])
                axis.set_ylim(*rna_limits[1])
    figure.text(0.013, 0.705, "Culture day", rotation=90, va="center", ha="center")
    figure.text(0.013, 0.315, "Cell type", rotation=90, va="center", ha="center")
    figure.legend(
        handles=time_legend_handles(), loc="lower center", bbox_to_anchor=(0.32, 0.02),
        ncol=len(DAYS), frameon=False, columnspacing=0.8, handletextpad=0.25,
    )
    figure.legend(
        handles=celltype_legend_handles(), loc="lower center", bbox_to_anchor=(0.80, 0.02),
        ncol=len(CELLTYPE_ORDER), frameon=False, columnspacing=0.8, handletextpad=0.25,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"original_metacell_rna_atac_umap.{suffix}", dpi=420)
    plt.close(figure)


def draw_model_spaces(frame: pd.DataFrame, output: Path) -> None:
    setup_style()
    order = ("Model RNA", "Model ATAC")
    titles = ("RNA · corrected PCA30", "ATAC · corrected LSI12")
    subsets = [frame.loc[frame["representation"].eq(name)].copy() for name in order]
    figure, axes = plt.subplots(2, 2, figsize=(5.0, 4.45))
    figure.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.91,
                           wspace=0.10, hspace=0.16)
    for column, (local, title) in enumerate(zip(subsets, titles, strict=True)):
        scatter_time(axes[0, column], local, 1.45)
        scatter_celltype(axes[1, column], local, 1.45)
        style_umap_axis(axes[0, column], title, show_y=column == 0)
        style_umap_axis(axes[1, column], "", show_y=column == 0)
    figure.text(0.015, 0.705, "Culture day", rotation=90, va="center", ha="center")
    figure.text(0.015, 0.315, "Cell type", rotation=90, va="center", ha="center")
    figure.legend(
        handles=time_legend_handles(), loc="lower center", bbox_to_anchor=(0.32, 0.02),
        ncol=4, frameon=False, columnspacing=0.75, handletextpad=0.25,
    )
    figure.legend(
        handles=celltype_legend_handles(), loc="lower center", bbox_to_anchor=(0.79, 0.02),
        ncol=3, frameon=False, columnspacing=0.75, handletextpad=0.25,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"metacell_model_space_rna_atac_umap.{suffix}", dpi=420)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    products = (
        "original_metacell_rna_atac_umap.png",
        "original_metacell_rna_atac_umap.pdf",
        "metacell_model_space_rna_atac_umap.png",
        "metacell_model_space_rna_atac_umap.pdf",
        "umap_coordinates.csv.gz",
        "analysis_manifest.json",
        "README.md",
    )
    existing = [name for name in products if (output / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} existing output files in {output}"
        )

    frame = fit_embeddings(args)
    frame.to_csv(output / "umap_coordinates.csv.gz", index=False)
    draw_available_comparison(frame, output)
    draw_model_spaces(frame, output)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "days": list(DAYS),
        "original_rna_cells": int(frame["representation"].eq("Original RNA").sum()),
        "paired_metacells": int(frame["representation"].eq("Metacell RNA").sum()),
        "rna_umap_fit": "all original RNA cells in frozen raw PCA30",
        "rna_metacell_operation": "transform through original-RNA-fitted UMAP",
        "atac_umap_fit": "paired metacells in frozen raw LSI12",
        "model_space_umaps": "independent fits in corrected normalized PCA30/LSI12",
        "n_neighbors": args.neighbors,
        "min_dist": args.min_dist,
        "metric": "euclidean",
        "seed": args.seed,
        "cell_type": "coarsened Nowakowski RNA-reference class",
        "atac_cell_type_semantics": "label carried by the RNA-matched paired metacell",
        "original_atac_status": (
            "not plotted: no time-course original-cell ATAC matrix is available locally; "
            "humanBrainATAC.h5ad is a paired-metacell matrix"
        ),
        "umap_use": "visualization only; distances must be computed pre-UMAP",
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    readme = """# Human cerebral original-cell and paired-metacell UMAPs

`original_metacell_rna_atac_umap` shows original RNA cells and paired RNA
metacells in one frozen raw-PCA30 UMAP. The UMAP is fit once on all original RNA
cells, then the metacells are transformed; the two RNA panels therefore have
the same coordinates and axis limits. The ATAC panel is a separate raw-LSI12
UMAP because RNA and ATAC do not share features or a common PCA.

`metacell_model_space_rna_atac_umap` shows the actual corrected, normalized
PCA30 and LSI12 spaces used to train the seven-time models. Each modality has
its own UMAP.

Rows are colored by culture day and by coarse Nowakowski RNA-reference class
(`RG`, `IPC`, `EN`, `IN`, `Other`). ATAC cell-type colors are labels inherited
by the RNA-matched paired metacell; they are not an independent ATAC annotation.

There is no original-cell time-course ATAC matrix locally. In particular,
`humanBrainATAC.h5ad` and `atac.mtx` both contain 34,088 paired metacells. The
public original ATAC accession is E-MTAB-11998, but it provides raw FASTQ files
that must be processed and projected through the frozen peak/IDF/SVD model
before a defensible original-ATAC panel can be added. No ATAC metacell has been
mislabelled as an original cell here.

UMAP is visualization only; quantitative original-cell/metacell comparisons
remain in frozen PCA30/LSI12 before UMAP.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    for name in products:
        print(output / name)


if __name__ == "__main__":
    main()
