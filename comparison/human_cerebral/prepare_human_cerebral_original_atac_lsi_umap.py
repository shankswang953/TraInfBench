#!/usr/bin/env python3
"""Project official original-cell ATAC into the frozen cerebral LSI space.

The large RDS is extracted by ``extract_human_cerebral_original_atac.R`` into
CSC slot binaries.  This script then reproduces the metacell TF-IDF transform,
uses the frozen SVD basis and linear correction, selects the benchmark LSI12
coordinates, and creates original-cell versus paired-metacell UMAPs.
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
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/trainfbench-hc-raw-atac-numba")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/trainfbench-hc-raw-atac-xdg")
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier
import umap

from trainfbench_plot_style import NATURE_CUD, apply_nature_rc


ROOT = Path(__file__).resolve().parents[2]
HUMAN_DATA = Path(
    "external/COATI/humanCerebral/Data"
)
MODEL_DIR = HUMAN_DATA / "processed_syncot_v2_no_d61"
SELECTED_DIR = HUMAN_DATA / "selected_4_7_9_11_12_18_21"
DEFAULT_WORK = ROOT / "data/human_cerebral_original_atac"
DEFAULT_OUTPUT = ROOT / "results/human_cerebral_original_atac_frozen_lsi12_umap"
OLD_UMAP_DIR = ROOT / "results/human_cerebral_original_metacell_umaps"

MODEL_DAYS = (4, 7, 9, 11, 12, 18, 21)
# The original scATAC experiment measured D16, not D12/D18.  Compare raw
# scATAC with metacells only at genuinely observed ATAC ages.
RAW_ATAC_DAYS = (4, 7, 9, 11, 16, 21)
DISPLAY_DAYS = (4, 7, 9, 11, 12, 16, 18, 21)
LSI_DIMS_ZERO_BASED = np.asarray([1, 2, 3, 4, 5, 6, 9, 10, 11, 13, 21, 26])
ATAC_SCALE = 34.16607592332852
K_CELLTYPE = 15

TIME_COLORS = {
    4: "#440154",
    7: "#414487",
    9: "#2A788E",
    11: "#22A884",
    12: "#7AD151",
    16: "#B8DE29",
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
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-frozen-peaks-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_model() -> dict[str, np.ndarray]:
    path = MODEL_DIR / "atac_lsi_model.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def write_frozen_peak_names(work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    model = load_frozen_model()
    output = work_dir / "frozen_peak_names.txt"
    with output.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(model["peak_names"].astype(str)))
        handle.write("\n")
    print(output)
    return output


def read_shape(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    return dict(zip(frame["field"], frame["value"], strict=True))


def sparse_binary_chunk(
    indices: np.memmap,
    indptr: np.memmap,
    start: int,
    stop: int,
    n_features: int,
) -> sparse.csr_matrix:
    first = int(indptr[start])
    last = int(indptr[stop])
    local_indptr = np.asarray(indptr[start : stop + 1], dtype=np.int64) - first
    local_indices = np.asarray(indices[first:last], dtype=np.int32)
    data = np.ones(last - first, dtype=np.float32)
    return sparse.csr_matrix(
        (data, local_indices, local_indptr),
        shape=(stop - start, n_features),
    )


def tfidf_svd(
    matrix: sparse.csr_matrix,
    idf: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    matrix = matrix.tocsr(copy=True).astype(np.float32)
    detected = np.diff(matrix.indptr).astype(np.float32)
    if np.any(detected <= 0):
        raise ValueError("At least one cell has no detected frozen peaks")
    matrix.data *= idf[matrix.indices]
    matrix.data *= np.repeat(1.0 / detected, np.diff(matrix.indptr))
    matrix.data = np.log1p(matrix.data * 1.0e4).astype(np.float32)
    return np.asarray(matrix @ components.T, dtype=np.float32)


def project_sparse_export(
    work_dir: Path,
    model: dict[str, np.ndarray],
    chunk_size: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    shape = read_shape(work_dir / "original_atac_sparse_shape.tsv")
    n_cells = int(shape["n_selected_cells"])
    n_features = int(shape["n_selected_peaks"])
    nnz = int(shape["nnz"])
    if n_features != len(model["peak_names"]):
        raise ValueError(
            f"Extracted {n_features:,} peaks, expected {len(model['peak_names']):,}"
        )
    indices = np.memmap(
        work_dir / "original_atac_selected_peaks_csc_i.int32",
        mode="r",
        dtype="<i4",
        shape=(nnz,),
    )
    indptr = np.memmap(
        work_dir / "original_atac_selected_peaks_csc_p.int32",
        mode="r",
        dtype="<i4",
        shape=(n_cells + 1,),
    )
    if int(indptr[-1]) != nnz:
        raise ValueError("Sparse indptr does not terminate at nnz")

    raw60 = np.empty((n_cells, model["components"].shape[0]), dtype=np.float32)
    for start in range(0, n_cells, chunk_size):
        stop = min(start + chunk_size, n_cells)
        block = sparse_binary_chunk(indices, indptr, start, stop, n_features)
        raw60[start:stop] = tfidf_svd(
            block,
            model["idf"].astype(np.float32),
            model["components"].astype(np.float32),
        )
        if start == 0 or stop == n_cells or (start // chunk_size) % 10 == 0:
            print(f"[project] {stop:,}/{n_cells:,}", flush=True)

    metadata = pd.read_csv(
        work_dir / "original_atac_cell_metadata.csv.gz", low_memory=False
    )
    if len(metadata) != n_cells:
        raise ValueError("Original ATAC metadata and sparse export row counts differ")
    metadata["processed_age"] = pd.to_numeric(
        metadata["processed_age"], errors="raise"
    ).astype(int)
    audit = {
        "n_original_atac_cells": n_cells,
        "n_frozen_peaks": n_features,
        "selected_nnz": nnz,
        "mean_detected_frozen_peaks": float(np.diff(indptr).mean()),
    }
    return raw60, metadata, audit


def normalize_line(values: pd.Series) -> pd.Series:
    raw = values.astype("string").str.strip()
    aliases = {
        "409B2": "409b2",
        "409b2": "409b2",
        "H9": "H9",
        "h9": "H9",
        "HOIK1": "Hoik1",
        "Hoik1": "Hoik1",
        "hoik1": "Hoik1",
        "WIBJ2": "Wibj2",
        "Wibj2": "Wibj2",
        "wibj2": "Wibj2",
    }
    return raw.map(lambda value: aliases.get(str(value), str(value)))


def numeric_metadata(metadata: pd.DataFrame, name: str) -> np.ndarray:
    aliases = {
        "nCount_peaks": ("nCount_peaks", "peak_region_fragments"),
        "nFeature_peaks": ("nFeature_peaks",),
        "n_cells_ATAC": ("n_cells_ATAC",),
    }
    for candidate in aliases[name]:
        if candidate in metadata:
            values = pd.to_numeric(metadata[candidate], errors="coerce").to_numpy(float)
            if np.isfinite(values).all():
                return values
    if name == "n_cells_ATAC":
        return np.ones(len(metadata), dtype=float)
    raise ValueError(f"Original ATAC metadata lacks a usable {name} column")


def build_frozen_correction_design(
    metadata: pd.DataFrame,
    training_obs: pd.DataFrame,
    covariate_names: np.ndarray,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    columns: dict[str, np.ndarray] = {
        "intercept": np.ones(len(metadata), dtype=np.float64)
    }
    scaling: dict[str, dict[str, float]] = {}
    for raw_name in ("nCount_peaks", "nFeature_peaks", "n_cells_ATAC"):
        label = f"log1p_{raw_name}"
        train = np.log1p(
            pd.to_numeric(training_obs[raw_name], errors="raise").to_numpy(float)
        )
        mean = float(train.mean())
        sd = float(train.std(ddof=0))
        if sd <= 1e-12:
            raise ValueError(f"Frozen training covariate {raw_name} has zero variance")
        values = np.log1p(np.maximum(numeric_metadata(metadata, raw_name), 0.0))
        columns[label] = (values - mean) / sd
        scaling[label] = {"mean": mean, "sd": sd}

    if "line" not in metadata:
        raise ValueError("Original ATAC metadata lacks source-line labels")
    line = normalize_line(metadata["line"])
    for name in covariate_names.astype(str):
        if name.startswith("line="):
            columns[name] = line.eq(name.split("=", 1)[1]).to_numpy(float)

    missing = [name for name in covariate_names.astype(str) if name not in columns]
    if missing:
        raise ValueError(f"Cannot construct frozen correction columns: {missing}")
    design = np.column_stack([columns[name] for name in covariate_names.astype(str)])
    return design, scaling


def load_metacells() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    metadata_all = pd.read_csv(SELECTED_DIR / "paired_obs.csv", low_memory=False)
    raw_blocks: list[np.ndarray] = []
    corrected_blocks: list[np.ndarray] = []
    metadata_blocks: list[pd.DataFrame] = []
    with np.load(MODEL_DIR / "atac_lsi60_raw_by_age.npz") as raw_archive, np.load(
        SELECTED_DIR / "atac_lsi12_normalized_by_time.npz"
    ) as corrected_archive:
        for day in MODEL_DAYS:
            key = f"age_{day}"
            raw = np.asarray(
                raw_archive[key][:, LSI_DIMS_ZERO_BASED], dtype=np.float32
            )
            corrected = np.asarray(corrected_archive[key], dtype=np.float32)
            local = metadata_all.loc[
                metadata_all["time_key"].astype(str).eq(key)
            ].copy()
            if len(local) != len(raw) or len(local) != len(corrected):
                raise ValueError(f"{key}: metacell coordinate/metadata mismatch")
            local["processed_age"] = day
            raw_blocks.append(raw)
            corrected_blocks.append(corrected)
            metadata_blocks.append(local.reset_index(drop=True))
    return (
        np.vstack(raw_blocks),
        np.vstack(corrected_blocks),
        pd.concat(metadata_blocks, ignore_index=True),
    )


def load_celltype_reference_metacells() -> tuple[np.ndarray, pd.DataFrame]:
    """Load same-day metacells, including D16, only for label transfer."""
    metadata_all = pd.read_csv(MODEL_DIR / "paired_obs.csv", low_memory=False)
    coordinate_blocks: list[np.ndarray] = []
    metadata_blocks: list[pd.DataFrame] = []
    with np.load(MODEL_DIR / "atac_lsi60_corrected_by_age.npz") as archive:
        for day in RAW_ATAC_DAYS:
            key = f"age_{day}"
            coordinates = np.asarray(
                archive[key][:, LSI_DIMS_ZERO_BASED] / ATAC_SCALE,
                dtype=np.float32,
            )
            local = metadata_all.loc[
                pd.to_numeric(metadata_all["processed_age"], errors="coerce").eq(day)
            ].copy()
            if len(local) != len(coordinates):
                raise ValueError(f"{key}: annotation coordinate/metadata mismatch")
            local["processed_age"] = day
            coordinate_blocks.append(coordinates)
            metadata_blocks.append(local.reset_index(drop=True))
    return (
        np.vstack(coordinate_blocks),
        pd.concat(metadata_blocks, ignore_index=True),
    )


def coarse_celltype(values: pd.Series | np.ndarray) -> np.ndarray:
    labels = pd.Series(values, dtype="string").str.strip().str.upper()
    return labels.where(labels.isin(CELLTYPE_ORDER[:-1]), "Other").to_numpy(str)


def assign_raw_celltypes(
    raw_coordinates: np.ndarray,
    raw_metadata: pd.DataFrame,
    metacell_coordinates: np.ndarray,
    metacell_metadata: pd.DataFrame,
) -> tuple[np.ndarray, str]:
    direct_candidates = (
        "nowakowski_prediction",
        "cell_type",
        "celltype",
        "predicted_cell_type",
    )
    for candidate in direct_candidates:
        if candidate in raw_metadata:
            direct = coarse_celltype(raw_metadata[candidate])
            if np.mean(direct != "Other") >= 0.5:
                return direct, f"official original-ATAC metadata column {candidate}"

    target = coarse_celltype(metacell_metadata["nowakowski_prediction"])
    predicted = np.empty(len(raw_metadata), dtype=object)
    for day in RAW_ATAC_DAYS:
        train_mask = metacell_metadata["processed_age"].to_numpy(int) == day
        query_mask = raw_metadata["processed_age"].to_numpy(int) == day
        classifier = KNeighborsClassifier(
            n_neighbors=min(K_CELLTYPE, int(train_mask.sum())),
            weights="distance",
            metric="euclidean",
        )
        classifier.fit(metacell_coordinates[train_mask], target[train_mask])
        predicted[query_mask] = classifier.predict(raw_coordinates[query_mask])
    return predicted.astype(str), (
        "same-day k=15 nearest-neighbour transfer from RNA-matched paired "
        "metacells in corrected normalized LSI12"
    )


def make_reducer(args: argparse.Namespace) -> umap.UMAP:
    return umap.UMAP(
        n_components=2,
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        metric="euclidean",
        random_state=args.seed,
        transform_seed=args.seed,
        low_memory=True,
        n_jobs=1,
    )


def fit_umaps(
    args: argparse.Namespace,
    raw_original: np.ndarray,
    corrected_original: np.ndarray,
    raw_metacell: np.ndarray,
    corrected_metacell: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, umap.UMAP]]:
    # Preserve the established metacell visual reference: each UMAP is fit on
    # exactly the seven-time metacells used in the earlier figure, then the
    # newly projected original ATAC cells are transformed through that model.
    raw_reducer = make_reducer(args)
    raw_metacell_umap = raw_reducer.fit_transform(raw_metacell / ATAC_SCALE)
    raw_original_umap = raw_reducer.transform(raw_original / ATAC_SCALE)
    corrected_reducer = make_reducer(args)
    corrected_metacell_umap = corrected_reducer.fit_transform(corrected_metacell)
    corrected_original_umap = corrected_reducer.transform(corrected_original)
    return (
        {
            "Original ATAC raw": raw_original_umap,
            "Metacell ATAC raw": raw_metacell_umap,
            "Original ATAC model": corrected_original_umap,
            "Metacell ATAC model": corrected_metacell_umap,
        },
        {"raw_lsi12": raw_reducer, "model_lsi12": corrected_reducer},
    )


def setup_style() -> None:
    apply_nature_rc(font_size=7.0)
    mpl.rcParams.update(
        {
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.55,
        }
    )


def style_axis(axis: plt.Axes, title: str, show_y: bool) -> None:
    axis.set_title(title, pad=4)
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2" if show_y else "")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#777777")
        spine.set_linewidth(0.55)


def scatter_time(axis: plt.Axes, xy: np.ndarray, days: np.ndarray, size: float) -> None:
    for day in DISPLAY_DAYS:
        mask = days == day
        axis.scatter(
            xy[mask, 0], xy[mask, 1], s=size, c=TIME_COLORS[day],
            alpha=0.58, linewidths=0, rasterized=True,
        )


def scatter_celltype(
    axis: plt.Axes, xy: np.ndarray, labels: np.ndarray, size: float
) -> None:
    for label in ("Other", "IPC", "RG", "EN", "IN"):
        mask = labels == label
        axis.scatter(
            xy[mask, 0], xy[mask, 1], s=size, c=CELLTYPE_COLORS[label],
            alpha=0.55 if label == "Other" else 0.65,
            linewidths=0, rasterized=True,
        )


def legend_handles(
    days: tuple[int, ...] = DISPLAY_DAYS,
) -> tuple[list[Line2D], list[Line2D]]:
    time = [
        Line2D([], [], marker="o", linestyle="none", markersize=4.0,
               markerfacecolor=TIME_COLORS[day], markeredgewidth=0, label=f"D{day}")
        for day in days
    ]
    celltype = [
        Line2D([], [], marker="o", linestyle="none", markersize=4.0,
               markerfacecolor=CELLTYPE_COLORS[label], markeredgewidth=0, label=label)
        for label in CELLTYPE_ORDER
    ]
    return time, celltype


def common_limits(arrays: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    xy = np.vstack(arrays)
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    dx = max(float(x_max - x_min), 1e-6)
    dy = max(float(y_max - y_min), 1e-6)
    return (
        (float(x_min - 0.025 * dx), float(x_max + 0.025 * dx)),
        (float(y_min - 0.025 * dy), float(y_max + 0.025 * dy)),
    )


def draw_atac_comparison(
    output: Path,
    embeddings: dict[str, np.ndarray],
    original_metadata: pd.DataFrame,
    metacell_metadata: pd.DataFrame,
    original_celltype: np.ndarray,
    metacell_celltype: np.ndarray,
    space: str,
) -> None:
    setup_style()
    keys = (
        ("Original ATAC raw", "Metacell ATAC raw")
        if space == "raw"
        else ("Original ATAC model", "Metacell ATAC model")
    )
    title_suffix = "Common frozen LSI12" if space == "raw" else "Common model LSI12"
    titles = (
        "Original ATAC cells",
        "Paired ATAC metacells",
    )
    figure, axes = plt.subplots(2, 2, figsize=(5.4, 4.55))
    figure.subplots_adjust(
        left=0.075, right=0.995, bottom=0.15, top=0.88, wspace=0.08, hspace=0.16
    )
    figure.suptitle(title_suffix, y=0.975, fontsize=8.5)
    original_days = original_metadata["processed_age"].to_numpy(int)
    metacell_days = metacell_metadata["processed_age"].to_numpy(int)
    limits = common_limits([embeddings[key] for key in keys])
    for column, (key, title) in enumerate(zip(keys, titles, strict=True)):
        is_original = column == 0
        metadata_days = original_days if is_original else metacell_days
        celltype = original_celltype if is_original else metacell_celltype
        point_size = 1.1 if is_original else 1.45
        scatter_time(axes[0, column], embeddings[key], metadata_days, point_size)
        scatter_celltype(axes[1, column], embeddings[key], celltype, point_size)
        style_axis(axes[0, column], title, show_y=column == 0)
        style_axis(axes[1, column], "", show_y=column == 0)
        for axis in axes[:, column]:
            axis.set_xlim(*limits[0])
            axis.set_ylim(*limits[1])
    figure.text(0.015, 0.70, "Culture day", rotation=90, va="center", ha="center")
    figure.text(0.015, 0.31, "Cell type", rotation=90, va="center", ha="center")
    time_handles, celltype_handles = legend_handles(DISPLAY_DAYS)
    figure.legend(
        handles=time_handles, loc="lower center", bbox_to_anchor=(0.31, 0.02),
        ncol=4, frameon=False, columnspacing=0.65, handletextpad=0.2,
    )
    figure.legend(
        handles=celltype_handles, loc="lower center", bbox_to_anchor=(0.79, 0.02),
        ncol=3, frameon=False, columnspacing=0.65, handletextpad=0.2,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(
            output / f"original_vs_metacell_atac_{space}_lsi12_umap.{suffix}", dpi=420
        )
    plt.close(figure)


def draw_four_panel(
    output: Path,
    embeddings: dict[str, np.ndarray],
    original_metadata: pd.DataFrame,
    metacell_metadata: pd.DataFrame,
    original_celltype: np.ndarray,
    metacell_celltype: np.ndarray,
) -> None:
    old = pd.read_csv(OLD_UMAP_DIR / "umap_coordinates.csv.gz", low_memory=False)
    original_rna = old.loc[old["representation"].eq("Original RNA")].copy()
    metacell_rna = old.loc[old["representation"].eq("Metacell RNA")].copy()
    definitions = (
        (
            "Original RNA cells", original_rna[["UMAP1", "UMAP2"]].to_numpy(float),
            original_rna["day"].to_numpy(int), original_rna["cell_type"].to_numpy(str),
        ),
        (
            "Paired metacells · RNA", metacell_rna[["UMAP1", "UMAP2"]].to_numpy(float),
            metacell_rna["day"].to_numpy(int), metacell_rna["cell_type"].to_numpy(str),
        ),
        (
            "Original ATAC cells", embeddings["Original ATAC raw"],
            original_metadata["processed_age"].to_numpy(int), original_celltype,
        ),
        (
            "Paired metacells · ATAC", embeddings["Metacell ATAC raw"],
            metacell_metadata["processed_age"].to_numpy(int), metacell_celltype,
        ),
    )
    setup_style()
    figure, axes = plt.subplots(2, 4, figsize=(9.1, 4.55))
    figure.subplots_adjust(
        left=0.047, right=0.997, bottom=0.15, top=0.91, wspace=0.08, hspace=0.16
    )
    pair_limits = (
        common_limits([definitions[0][1], definitions[1][1]]),
        common_limits([definitions[2][1], definitions[3][1]]),
    )
    for column, (title, xy, days, celltypes) in enumerate(definitions):
        point_size = 0.95 if column in (0, 2) else 1.35
        scatter_time(axes[0, column], xy, days, point_size)
        scatter_celltype(axes[1, column], xy, celltypes, point_size)
        style_axis(axes[0, column], title, show_y=column in (0, 2))
        style_axis(axes[1, column], "", show_y=column in (0, 2))
        limits = pair_limits[0 if column < 2 else 1]
        for axis in axes[:, column]:
            axis.set_xlim(*limits[0])
            axis.set_ylim(*limits[1])
    figure.text(0.007, 0.70, "Culture day", rotation=90, va="center", ha="center")
    figure.text(0.007, 0.31, "Cell type", rotation=90, va="center", ha="center")
    time_handles, celltype_handles = legend_handles()
    figure.legend(
        handles=time_handles, loc="lower center", bbox_to_anchor=(0.34, 0.02),
        ncol=7, frameon=False, columnspacing=0.6, handletextpad=0.2,
    )
    figure.legend(
        handles=celltype_handles, loc="lower center", bbox_to_anchor=(0.80, 0.02),
        ncol=5, frameon=False, columnspacing=0.6, handletextpad=0.2,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"original_vs_metacell_rna_atac_umap.{suffix}", dpi=420)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    frozen_peaks = write_frozen_peak_names(work_dir)
    if args.write_frozen_peaks_only:
        return

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    products = (
        "original_atac_frozen_lsi12.npz",
        "original_atac_cell_metadata_projected.csv.gz",
        "original_vs_metacell_atac_raw_lsi12_umap.png",
        "original_vs_metacell_atac_raw_lsi12_umap.pdf",
        "original_vs_metacell_atac_model_lsi12_umap.png",
        "original_vs_metacell_atac_model_lsi12_umap.pdf",
        "original_vs_metacell_rna_atac_umap.png",
        "original_vs_metacell_rna_atac_umap.pdf",
        "metacell_raw_lsi12_umap_model.joblib",
        "metacell_model_lsi12_umap_model.joblib",
        "umap_coordinates.csv.gz",
        "counts_by_day.csv",
        "analysis_manifest.json",
    )
    existing = [name for name in products if (output / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")

    model = load_frozen_model()
    raw60, original_metadata, projection_audit = project_sparse_export(
        work_dir, model, args.chunk_size
    )
    training_obs = pd.read_csv(MODEL_DIR / "paired_obs.csv", low_memory=False)
    design, correction_scaling = build_frozen_correction_design(
        original_metadata, training_obs, model["correction_covariates"]
    )
    corrected60 = raw60.astype(np.float64) - design @ model["correction_beta"].astype(
        np.float64
    )
    raw12 = raw60[:, LSI_DIMS_ZERO_BASED].astype(np.float32)
    corrected12 = (corrected60[:, LSI_DIMS_ZERO_BASED] / ATAC_SCALE).astype(np.float32)
    if not np.isfinite(raw12).all() or not np.isfinite(corrected12).all():
        raise ValueError("Non-finite original-cell LSI coordinates")

    raw_metacell, corrected_metacell, metacell_metadata = load_metacells()
    metacell_celltype = coarse_celltype(metacell_metadata["nowakowski_prediction"])
    annotation_coordinates, annotation_metadata = load_celltype_reference_metacells()
    original_celltype, celltype_source = assign_raw_celltypes(
        corrected12,
        original_metadata,
        annotation_coordinates,
        annotation_metadata,
    )
    original_metadata["cell_type_for_plot"] = original_celltype
    original_metadata["line_normalized"] = normalize_line(original_metadata["line"])

    embeddings, reducers = fit_umaps(
        args, raw12, corrected12, raw_metacell, corrected_metacell
    )
    joblib.dump(
        reducers["raw_lsi12"],
        output / "metacell_raw_lsi12_umap_model.joblib",
        compress=3,
    )
    joblib.dump(
        reducers["model_lsi12"],
        output / "metacell_model_lsi12_umap_model.joblib",
        compress=3,
    )
    draw_atac_comparison(
        output, embeddings, original_metadata, metacell_metadata,
        original_celltype, metacell_celltype, "raw",
    )
    draw_atac_comparison(
        output, embeddings, original_metadata, metacell_metadata,
        original_celltype, metacell_celltype, "model",
    )
    draw_four_panel(
        output, embeddings, original_metadata, metacell_metadata,
        original_celltype, metacell_celltype,
    )

    np.savez_compressed(
        output / "original_atac_frozen_lsi12.npz",
        raw_lsi12=raw12,
        corrected_normalized_lsi12=corrected12,
        cell_id=np.asarray(original_metadata["cell_id"].astype(str), dtype=str),
        age=original_metadata["processed_age"].to_numpy(int),
        line=np.asarray(original_metadata["line_normalized"].astype(str), dtype=str),
        cell_type=np.asarray(original_celltype, dtype=str),
    )
    original_metadata.to_csv(
        output / "original_atac_cell_metadata_projected.csv.gz", index=False
    )

    coordinate_frames = []
    for key, xy in embeddings.items():
        is_original = key.startswith("Original")
        metadata = original_metadata if is_original else metacell_metadata
        celltype = original_celltype if is_original else metacell_celltype
        coordinate_frames.append(
            pd.DataFrame(
                {
                    "representation": key,
                    "cell_id": metadata[
                        "cell_id" if is_original else "paired_metacell_id"
                    ].astype(str).to_numpy(),
                    "day": metadata["processed_age"].to_numpy(int),
                    "line": normalize_line(metadata["line"]).astype(str).to_numpy(),
                    "cell_type": celltype,
                    "UMAP1": xy[:, 0],
                    "UMAP2": xy[:, 1],
                }
            )
        )
    pd.concat(coordinate_frames, ignore_index=True).to_csv(
        output / "umap_coordinates.csv.gz", index=False
    )

    counts = pd.concat(
        [
            original_metadata.groupby("processed_age").size().rename("original_ATAC"),
            metacell_metadata.groupby("processed_age").size().rename("paired_metacells"),
        ],
        axis=1,
    ).reset_index(names="day")
    counts.to_csv(output / "counts_by_day.csv", index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_source": "https://zenodo.org/records/15371701",
        "official_file": str(work_dir / "ATAC_all_merged_srt.rds"),
        "official_md5": "35b3c7f1c850055611b03129514e21ac",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "r_extractor": str(ROOT / "comparison/human_cerebral/extract_human_cerebral_original_atac.R"),
        "frozen_model": str(MODEL_DIR / "atac_lsi_model.npz"),
        "frozen_peak_list": str(frozen_peaks),
        "model_days": list(MODEL_DAYS),
        "original_atac_days_used": list(RAW_ATAC_DAYS),
        "original_atac_time_note": (
            "The official scATAC experiment has D16 but no D12/D18 libraries; "
            "original cells use D4,D7,D9,D11,D16,D21, while the UMAP reference "
            "is the seven-time metacell set D4,D7,D9,D11,D12,D18,D21."
        ),
        "projection": (
            "binary presence -> per-cell TF -> frozen metacell IDF -> log1p(1e4*x) "
            "-> frozen 60D SVD -> frozen QC/line residualization -> selected LSI12"
        ),
        "selected_lsi_dimensions_1based": (LSI_DIMS_ZERO_BASED + 1).tolist(),
        "atac_normalization_scale": ATAC_SCALE,
        "correction_scaling": correction_scaling,
        "projection_audit": projection_audit,
        "n_paired_metacells": int(len(metacell_metadata)),
        "celltype_source": celltype_source,
        "celltype_transfer_k": K_CELLTYPE if "nearest-neighbour" in celltype_source else None,
        "umap": {
            "fit": (
                "seven-time paired metacells, matching the established metacell "
                "UMAP; original ATAC cells transformed through that reducer"
            ),
            "neighbors": args.neighbors,
            "min_dist": args.min_dist,
            "metric": "euclidean",
            "seed": args.seed,
            "saved_raw_lsi12_reducer": str(
                output / "metacell_raw_lsi12_umap_model.joblib"
            ),
            "saved_model_lsi12_reducer": str(
                output / "metacell_model_lsi12_umap_model.joblib"
            ),
            "use": "visualization only; quantitative distances remain pre-UMAP",
        },
    }
    with (output / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    for name in products:
        print(output / name)


if __name__ == "__main__":
    main()
