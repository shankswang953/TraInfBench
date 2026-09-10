#!/usr/bin/env python3
"""Audit full-data human-cerebral frozen T by cell type and source-line AUROC.

The query set contains all paired metacells at D7--D21.  For every day,
observed ATAC metacells form the reference atlas.  Two query representations
are compared: observed ATAC (leave-one-out reference ceiling) and T(RNA).
The query's own paired ATAC row is excluded from both kNN searches.  This is a
full-data mapping diagnostic, not held-out-time or LOTO validation.
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
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from trainfbench_plot_style import NATURE_CUD, apply_nature_rc


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COORDINATES = (
    ROOT
    / "results/human_cerebral_t_mapping_reference_v1/"
    "t_mapping_predictions_coordinates.npz"
)
DEFAULT_OBS = Path(
    "external/COATI/humanCerebral/"
    "Data/selected_4_7_9_11_12_18_21/paired_obs.csv"
)
DEFAULT_OUTPUT = ROOT / "results/human_cerebral_t_celltype_source_auroc"
DAYS = (7, 9, 11, 12, 18, 21)
CELLTYPE_LEVELS = ("IPC", "RG", "EN", "IN")
SOURCE_LEVELS = ("409b2", "h9", "hoik1", "wibj2")
MODEL_ORDER = ("ATAC reference", "Full T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", type=Path, default=DEFAULT_COORDINATES)
    parser.add_argument("--obs", type=Path, default=DEFAULT_OBS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_celltype(values: np.ndarray) -> np.ndarray:
    values = np.char.upper(np.char.strip(np.asarray(values, dtype=str)))
    return np.asarray(
        [value if value in CELLTYPE_LEVELS else "OTHER" for value in values],
        dtype=str,
    )


def gaussian_weights(distances: np.ndarray) -> np.ndarray:
    scale = float(np.median(distances[:, -1]))
    if not np.isfinite(scale) or scale <= 1e-12:
        positive = distances[distances > 0]
        scale = float(np.median(positive)) if len(positive) else 1.0
    weights = np.exp(-np.square(distances / scale))
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-30)


def paired_excluded_probabilities(
    query: np.ndarray,
    reference: np.ndarray,
    labels: np.ndarray,
    levels: tuple[str, ...],
    excluded_reference_rows: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(reference) <= k:
        raise ValueError("Reference atlas is smaller than k")
    model = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(reference)
    distances, indices = model.kneighbors(query, return_distance=True)
    kept_distances = np.empty((len(query), k), dtype=distances.dtype)
    kept_indices = np.empty((len(query), k), dtype=indices.dtype)
    for row, excluded in enumerate(excluded_reference_rows):
        selection = np.flatnonzero(indices[row] != excluded)[:k]
        if len(selection) != k:
            raise RuntimeError("Could not construct paired-excluded kNN set")
        kept_distances[row] = distances[row, selection]
        kept_indices[row] = indices[row, selection]
    weights = gaussian_weights(kept_distances)
    neighbor_labels = labels[kept_indices]
    probabilities = np.column_stack(
        [
            np.sum(weights * (neighbor_labels == level), axis=1)
            for level in levels
        ]
    )
    return probabilities, kept_distances[:, -1]


def load_inputs(coordinates: Path, obs_path: Path) -> dict[str, np.ndarray]:
    with np.load(coordinates, allow_pickle=False) as archive:
        required = {
            "paired_metacell_id",
            "day",
            "source_line",
            "observed_atac_lsi12_normalized",
            "predicted_atac_lsi12_normalized",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Coordinate archive lacks keys: {sorted(missing)}")
        data = {key: np.asarray(archive[key]) for key in required}
    obs = pd.read_csv(obs_path, low_memory=False)
    if "paired_metacell_id" not in obs or "nowakowski_prediction" not in obs:
        raise ValueError("paired_obs.csv lacks required labels")
    ids = data["paired_metacell_id"].astype(str)
    obs_ids = obs["paired_metacell_id"].astype(str).to_numpy()
    if not np.array_equal(ids, obs_ids):
        raise ValueError("Coordinate archive and paired_obs row order differ")
    data["celltype"] = canonical_celltype(
        obs["nowakowski_prediction"].fillna("OTHER").to_numpy()
    )
    data["source_line"] = np.char.lower(data["source_line"].astype(str))
    if set(data["source_line"]) != set(SOURCE_LEVELS):
        raise ValueError(f"Unexpected source lines: {np.unique(data['source_line'])}")
    for key in ("observed_atac_lsi12_normalized", "predicted_atac_lsi12_normalized"):
        if data[key].ndim != 2 or data[key].shape[1] != 12:
            raise ValueError(f"Unexpected {key} shape: {data[key].shape}")
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Non-finite values in {key}")
    return data


def evaluate(
    data: dict[str, np.ndarray], k: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    day = data["day"].astype(int)
    task_specs = {
        "Cell type": (data["celltype"], CELLTYPE_LEVELS),
        "Cell source": (data["source_line"], SOURCE_LEVELS),
    }
    probability_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    macro_rows: list[dict[str, object]] = []

    for task, (all_labels, levels) in task_specs.items():
        model_probabilities: dict[str, list[np.ndarray]] = {
            model: [] for model in MODEL_ORDER
        }
        true_labels: list[np.ndarray] = []
        query_ids: list[np.ndarray] = []
        query_days: list[np.ndarray] = []
        model_radii: dict[str, list[np.ndarray]] = {model: [] for model in MODEL_ORDER}
        for current_day in DAYS:
            reference_indices = np.flatnonzero(day == current_day)
            query_indices = np.flatnonzero(day == current_day)
            reference_lookup = {
                global_index: local_index
                for local_index, global_index in enumerate(reference_indices)
            }
            excluded = np.asarray(
                [reference_lookup[index] for index in query_indices], dtype=int
            )
            reference = data["observed_atac_lsi12_normalized"][reference_indices]
            reference_labels = all_labels[reference_indices]
            queries = {
                "ATAC reference": data["observed_atac_lsi12_normalized"][query_indices],
                "Full T": data["predicted_atac_lsi12_normalized"][query_indices],
            }
            for model, query in queries.items():
                probabilities, radius = paired_excluded_probabilities(
                    query,
                    reference,
                    reference_labels,
                    levels,
                    excluded,
                    k,
                )
                model_probabilities[model].append(probabilities)
                model_radii[model].append(radius)
            true_labels.append(all_labels[query_indices])
            query_ids.append(data["paired_metacell_id"][query_indices].astype(str))
            query_days.append(np.full(len(query_indices), current_day, dtype=int))

        y_all = np.concatenate(true_labels)
        ids_all = np.concatenate(query_ids)
        days_all = np.concatenate(query_days)
        if task == "Cell type":
            keep = np.isin(y_all, levels)
        else:
            keep = np.ones(len(y_all), dtype=bool)
        y = y_all[keep]
        ids = ids_all[keep]
        days_kept = days_all[keep]

        for model in MODEL_ORDER:
            probabilities = np.concatenate(model_probabilities[model])[keep]
            radii = np.concatenate(model_radii[model])[keep]
            aucs = []
            for column, level in enumerate(levels):
                positive = y == level
                if not positive.any() or positive.all():
                    raise ValueError(f"{task} {level} lacks both AUROC classes")
                auc = float(
                    roc_auc_score(positive.astype(np.int8), probabilities[:, column])
                )
                aucs.append(auc)
                detail_rows.append(
                    {
                        "task": task,
                        "model": model,
                        "class": level,
                        "support": int(positive.sum()),
                        "n_queries": int(len(y)),
                        "one_vs_rest_auroc": auc,
                    }
                )
            macro_rows.append(
                {
                    "task": task,
                    "model": model,
                    "n_queries": int(len(y)),
                    "n_classes": len(levels),
                    "macro_ovr_auroc": float(np.mean(aucs)),
                    "mean_k15_radius": float(np.mean(radii)),
                }
            )
            for row, query_id in enumerate(ids):
                record: dict[str, object] = {
                    "task": task,
                    "model": model,
                    "paired_metacell_id": query_id,
                    "day": int(days_kept[row]),
                    "true_class": y[row],
                    "k15_radius": float(radii[row]),
                }
                for column, level in enumerate(levels):
                    record[f"probability_{level}"] = float(probabilities[row, column])
                probability_rows.append(record)

    return (
        pd.DataFrame(detail_rows),
        pd.DataFrame(macro_rows),
        pd.DataFrame(probability_rows),
    )


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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot(detail: pd.DataFrame, macro: pd.DataFrame, output_dir: Path) -> list[Path]:
    styles = {
        "ATAC reference": {
            "marker": "x",
            "color": "#1A1A1A",
            "face": "none",
            "label": "ATAC reference",
        },
        "Full T": {
            "marker": "o",
            "color": NATURE_CUD["sky_blue"],
            "face": "none",
            "label": "Full T",
        },
    }
    task_levels = {
        "Cell type": CELLTYPE_LEVELS,
        "Cell source": SOURCE_LEVELS,
    }
    display = {
        "409b2": "409B2",
        "h9": "H9",
        "hoik1": "HOIK1",
        "wibj2": "WIBJ2",
    }
    x_limits = {"Cell type": (0.62, 0.90), "Cell source": (0.72, 0.98)}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    offsets = {"ATAC reference": -0.09, "Full T": 0.09}
    for ax, task in zip(axes, task_levels, strict=True):
        levels = task_levels[task]
        y = np.arange(len(levels))
        for model in MODEL_ORDER:
            local = detail.loc[
                detail["task"].eq(task) & detail["model"].eq(model)
            ].set_index("class")
            values = np.asarray(
                [local.loc[level, "one_vs_rest_auroc"] for level in levels]
            )
            style = styles[model]
            scatter_kwargs = {
                "marker": style["marker"],
                "s": 30,
                "linewidths": 1.0,
                "label": style["label"],
                "zorder": 3,
            }
            if model == "ATAC reference":
                scatter_kwargs["color"] = style["color"]
            else:
                scatter_kwargs["facecolors"] = "none"
                scatter_kwargs["edgecolors"] = style["color"]
            ax.scatter(values, y + offsets[model], **scatter_kwargs)
        full_macro = float(
            macro.loc[
                macro["task"].eq(task) & macro["model"].eq("Full T"),
                "macro_ovr_auroc",
            ].iloc[0]
        )
        ax.set_title(f"{task}\nFull T macro AUROC = {full_macro:.3f}", pad=5.0)
        ax.set_yticks(y, [display.get(level, level) for level in levels])
        ax.invert_yaxis()
        ax.set_xlim(*x_limits[task])
        ax.set_xlabel("One-vs-rest AUROC")
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.98),
        ncol=2,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.0,
    )
    fig.suptitle(
        "Human cerebral: RNA-to-ATAC mapping",
        y=1.035,
        fontsize=10.0,
        fontweight="normal",
    )
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.19, top=0.70, wspace=0.34)
    stem = output_dir / "human_cerebral_t_celltype_source_auroc"
    outputs = [stem.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg")]
    save = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.04}
    fig.savefig(outputs[0], dpi=600, **save)
    fig.savefig(outputs[1], **save)
    fig.savefig(outputs[2], **save)
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    args.coordinates = args.coordinates.resolve()
    args.obs = args.obs.resolve()
    args.output_dir = args.output_dir.resolve()
    for path in (args.coordinates, args.obs):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "human_cerebral_t_celltype_source_auroc"
    expected = [stem.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg")]
    expected += [
        args.output_dir / "auroc_by_class.csv",
        args.output_dir / "macro_auroc.csv",
        args.output_dir / "query_probabilities.csv.gz",
        args.output_dir / "analysis_manifest.json",
    ]
    existing = [path for path in expected if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs:\n"
            + "\n".join(f"  {path}" for path in existing)
        )

    data = load_inputs(args.coordinates, args.obs)
    detail, macro, probabilities = evaluate(data, args.knn_k)
    detail.to_csv(args.output_dir / "auroc_by_class.csv", index=False)
    macro.to_csv(args.output_dir / "macro_auroc.csv", index=False)
    probabilities.to_csv(
        args.output_dir / "query_probabilities.csv.gz",
        index=False,
        compression="gzip",
    )
    configure_style()
    outputs = plot(detail, macro, args.output_dir)
    manifest = {
        "analysis": "Frozen full T cell-type and source-line preservation",
        "query_unit": "all paired metacells at D7-D21",
        "days": list(DAYS),
        "D4_excluded": "ATAC metacell cell-type reference is degenerate at D4; the same D7-D21 scope is used for both panels",
        "reference": "same-day observed normalized ATAC LSI12 metacells",
        "models": {
            "ATAC reference": "observed ATAC query with its own row excluded",
            "Full T": "predicted T(RNA,time) query with its paired ATAC row excluded",
        },
        "classifier": {
            "method": "same-day Gaussian-distance-weighted kNN",
            "k": args.knn_k,
            "source_blind_for_celltype": True,
            "self_or_paired_target_excluded": True,
        },
        "metrics": "full-data pooled D7-D21 one-vs-rest AUROC and unweighted macro mean over four classes",
        "celltype_levels": list(CELLTYPE_LEVELS),
        "source_levels": list(SOURCE_LEVELS),
        "inputs": {
            "coordinates": {
                "path": str(args.coordinates),
                "sha256": sha256(args.coordinates),
            },
            "metadata": {"path": str(args.obs), "sha256": sha256(args.obs)},
        },
        "limitations": [
            "ATAC metacell cell-type labels are inherited from paired RNA and are not an independent ATAC annotation.",
            "Source-line AUROC measures preservation of source-associated state and may include line or batch signatures; it is not lineage tracing.",
            "This is a descriptive full-data mapping diagnostic, not held-out or leave-one-time-out validation.",
        ],
        "outputs": [str(path.resolve()) for path in outputs],
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(detail.to_string(index=False))
    print("\nMacro AUROC")
    print(macro.to_string(index=False))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
