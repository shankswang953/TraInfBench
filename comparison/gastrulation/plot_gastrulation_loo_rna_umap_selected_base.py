#!/usr/bin/env python
"""Compare selected Gastrulation held-out RNA predictions on one fixed UMAP."""

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

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/trainfbench-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import anndata as ad
import joblib
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numba
import numpy as np
import pandas as pd
import torch

from trainfbench_plot_style import apply_nature_rc, method_style


ROOT = Path(__file__).resolve().parents[2]
GASTRULATION_DATA = Path(
    "external/COATI/Gastrulation/data"
)
RNA_H5AD = ROOT / "data" / "gastrulation_rna_cytobridge.h5ad"
RNA_UMAP_MODEL = ROOT / "data" / "gastrulation_rna_umap_model.joblib"
RNA_NORM = GASTRULATION_DATA / "primal_norm_params.pt"
BASE_DIR = (
    ROOT / "results" / "gastrulation_loo_shared_t_trajectorynet_base"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "gastrulation_loo_rna_umap_selected_trajectorynet_base"
)

METHODS = (
    (
        "TrajectoryNet 20k",
        "TrajectoryNet",
        "TrajectoryNet",
    ),
    (
        "CytoBridge balanced 20k",
        "CytoBridge bal",
        "CytoBridge balanced",
    ),
    (
        "CytoBridge unbalanced 20k",
        "CytoBridge unbal",
        "CytoBridge unbalanced",
    ),
    (
        "BSOT C_y=0.3",
        "COATI bal",
        "COATI balanced",
    ),
    (
        "USOT C_y=0.3",
        "COATI unbal",
        "COATI unbalanced",
    ),
)

SCENARIOS = {
    "loo_time1": {
        "stage": "E8.0",
        "stage_tag": "e80",
        "physical_time": 1.0,
        "prediction_cache": (
            ROOT
            / "results"
            / "gastrulation_loo_time1_coverage_validity"
            / "loo_time1_shared_t_atac_predictions.npz"
        ),
        "base_prediction": BASE_DIR / "loo_time1_trajectorynet_base_predictions.npz",
        "scores": BASE_DIR / "loo_time1_shared_t_scores_trajectorynet_base.csv",
    },
    "loo_time2": {
        "stage": "E8.5",
        "stage_tag": "e85",
        "physical_time": 2.0,
        "prediction_cache": (
            ROOT
            / "results"
            / "gastrulation_loo_time2_shared_t_atac"
            / "loo_time2_shared_t_predictions.npz"
        ),
        "base_prediction": BASE_DIR / "loo_time2_trajectorynet_base_predictions.npz",
        "scores": BASE_DIR / "loo_time2_shared_t_scores_trajectorynet_base.csv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--umap-batch-size", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def disable_numba_disk_cache() -> None:
    """Allow the frozen UMAP pickle to load in the shared environment."""

    original_njit = numba.njit
    original_jit = numba.jit

    def njit_no_cache(*args, **kwargs):
        kwargs["cache"] = False
        return original_njit(*args, **kwargs)

    def jit_no_cache(*args, **kwargs):
        kwargs["cache"] = False
        return original_jit(*args, **kwargs)

    numba.njit = njit_no_cache
    numba.jit = jit_no_cache


def load_scale(path: Path) -> float:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return float(np.asarray(state["scale"]).reshape(-1)[0])


def normalized_weights(log_mass: np.ndarray) -> np.ndarray:
    values = np.asarray(log_mass, dtype=np.float64)
    values = np.exp(np.clip(values - np.max(values), -700.0, 0.0))
    values /= values.sum()
    return values


def systematic_resample(log_mass: np.ndarray, n: int) -> np.ndarray:
    cumulative = np.cumsum(normalized_weights(log_mass))
    cumulative[-1] = 1.0
    positions = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    return np.searchsorted(cumulative, positions, side="left").astype(np.int64)


def batched_transform(model, values: np.ndarray, batch_size: int) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        pieces.append(
            np.asarray(model.transform(values[start:stop]), dtype=np.float32)
        )
    return np.concatenate(pieces, axis=0)


def load_predictions(
    config: dict[str, object],
    rna_scale: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, bool]]:
    cache_path = Path(config["prediction_cache"])
    with np.load(cache_path, allow_pickle=True) as cached:
        cached_methods = cached["methods"].astype(str).tolist()
        cached_predictions = np.asarray(
            cached["rna_predictions"], dtype=np.float32
        )
        cached_log_masses = np.asarray(
            cached["native_log_masses"], dtype=np.float32
        )
        cached_has_mass = np.asarray(cached["has_native_mass"], dtype=bool)
    method_to_index = {
        method: index for index, method in enumerate(cached_methods)
    }
    required = {source for source, _, _ in METHODS if not source.startswith("TrajectoryNet")}
    missing = sorted(required - set(method_to_index))
    if missing:
        raise KeyError(f"Missing methods in {cache_path}: {missing}")

    predictions: dict[str, np.ndarray] = {}
    log_masses: dict[str, np.ndarray] = {}
    has_mass: dict[str, bool] = {}
    for source, _, _ in METHODS:
        if source == "TrajectoryNet 20k":
            with np.load(Path(config["base_prediction"]), allow_pickle=True) as base:
                prediction_raw = np.asarray(
                    base["prediction_rna_raw"], dtype=np.float32
                )
            predictions[source] = prediction_raw
            log_masses[source] = np.zeros(
                len(prediction_raw), dtype=np.float32
            )
            has_mass[source] = False
            continue
        index = method_to_index[source]
        predictions[source] = (
            cached_predictions[index] * rna_scale
        ).astype(np.float32, copy=False)
        log_masses[source] = cached_log_masses[index]
        has_mass[source] = bool(cached_has_mass[index])
    return predictions, log_masses, has_mass


def shared_limits(
    atlas: np.ndarray,
    real: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> tuple[tuple[float, float], tuple[float, float]]:
    pooled = np.concatenate([atlas, real, *predictions.values()], axis=0)
    low = np.quantile(pooled, 0.001, axis=0)
    high = np.quantile(pooled, 0.999, axis=0)
    padding = 0.035 * np.maximum(high - low, 1e-6)
    return (
        (float(low[0] - padding[0]), float(high[0] + padding[0])),
        (float(low[1] - padding[1]), float(high[1] + padding[1])),
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(RNA_H5AD)
    times = pd.to_numeric(
        adata.obs["time_point_processed"], errors="raise"
    ).to_numpy(float)
    raw_pca = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    atlas_umap = np.asarray(adata.obsm["X_umap"], dtype=np.float32)

    disable_numba_disk_cache()
    umap_model = joblib.load(RNA_UMAP_MODEL)
    umap_training = np.asarray(umap_model._raw_data, dtype=np.float32)
    if umap_training.shape != raw_pca.shape:
        raise ValueError(
            "Frozen RNA UMAP was not trained on the complete reference atlas"
        )
    umap_training_max_abs = float(np.max(np.abs(umap_training - raw_pca)))
    if umap_training_max_abs > 1e-5:
        raise ValueError(
            "Frozen RNA UMAP training data mismatch: "
            f"max_abs={umap_training_max_abs:g}"
        )
    rna_scale = load_scale(RNA_NORM)

    apply_nature_rc(font_size=9.0)
    mpl.rcParams.update(
        {
            "axes.titlesize": 10.0,
            "legend.fontsize": 9.0,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "figure.titleweight": "normal",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
        }
    )
    real_style = method_style("Observed / real")
    manifest: dict[str, object] = {
        "task": (
            "Gastrulation held-out E8.0/E8.5 RNA UMAP comparison using "
            "one frozen all-real-stage UMAP model"
        ),
        "umap_model": str(RNA_UMAP_MODEL.resolve()),
        "umap_training_data_max_abs_check": umap_training_max_abs,
        "coati_selection": "C_y=0.3 for balanced and unbalanced COATI",
        "trajectorynet_origin": "standard-normal base",
        "unbalanced_display": (
            "Deterministic systematic resampling from normalized native mass"
        ),
        "scenarios": {},
    }

    for scenario, config in SCENARIOS.items():
        stage = str(config["stage"])
        stage_tag = str(config["stage_tag"])
        stem = (
            output_dir
            / f"gastrulation_loo_{stage_tag}_rna_umap_selected_trajectorynet_base"
        )
        outputs = [
            stem.with_suffix(extension)
            for extension in (".png", ".pdf", ".svg")
        ]
        coordinates_path = output_dir / f"{stem.name}_coordinates.npz"
        scores_path = output_dir / f"{stem.name}_scores.csv"
        outputs.extend([coordinates_path, scores_path])
        existing = [path for path in outputs if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                "Refusing to overwrite existing outputs; pass --overwrite:\n"
                + "\n".join(str(path) for path in existing)
            )

        target_mask = np.isclose(times, float(config["physical_time"]))
        real_umap = atlas_umap[target_mask]
        predictions_raw, log_masses, has_mass = load_predictions(
            config, rna_scale
        )
        prediction_umap: dict[str, np.ndarray] = {}
        display_indices: dict[str, np.ndarray] = {}
        for source, _, _ in METHODS:
            print(f"[{stage}] UMAP {source}", flush=True)
            prediction_umap[source] = batched_transform(
                umap_model,
                predictions_raw[source],
                args.umap_batch_size,
            )
            display_indices[source] = (
                systematic_resample(
                    log_masses[source], len(prediction_umap[source])
                )
                if has_mass[source]
                else np.arange(len(prediction_umap[source]), dtype=np.int64)
            )

        score_frame = pd.read_csv(Path(config["scores"])).set_index("method")
        score_rows: list[dict[str, object]] = []
        w2: dict[str, float] = {}
        for source, display, canonical in METHODS:
            divergence = float(
                score_frame.loc[source, "rna_sinkhorn_divergence"]
            )
            w2[source] = float(np.sqrt(2.0 * max(divergence, 0.0)))
            score_rows.append(
                {
                    "scenario": scenario,
                    "heldout_stage": stage,
                    "method": canonical,
                    "display_name": display.replace("$", ""),
                    "source_method": source,
                    "rna_effective_coverage": float(
                        score_frame.loc[source, "rna_effective_coverage"]
                    ),
                    "rna_sinkhorn_divergence": divergence,
                    "rna_w2": w2[source],
                    "native_mass_resampled": has_mass[source],
                }
            )
        pd.DataFrame(score_rows).to_csv(scores_path, index=False)

        xlim, ylim = shared_limits(atlas_umap, real_umap, prediction_umap)
        fig = plt.figure(figsize=(3.6, 6.25), facecolor="white")
        grid = fig.add_gridspec(
            3,
            2,
            left=0.005,
            right=0.995,
            top=0.83,
            bottom=0.01,
            hspace=0.28,
            wspace=0.03,
        )
        axes = [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[1, 1]),
            fig.add_subplot(grid[2, 0]),
            fig.add_subplot(grid[2, 1]),
        ]

        observed_ax = axes[0]
        observed_ax.scatter(
            atlas_umap[:, 0],
            atlas_umap[:, 1],
            s=1.2,
            c="#B3B3B3",
            marker="o",
            alpha=0.09,
            linewidths=0,
            rasterized=True,
        )
        observed_ax.scatter(
            real_umap[:, 0],
            real_umap[:, 1],
            s=2.5,
            c=real_style.color,
            marker="o",
            alpha=0.68,
            linewidths=0,
            rasterized=True,
        )
        observed_ax.set_title(
            f"Observed {stage}\n"
            "in global RNA atlas",
            pad=3,
            linespacing=1.12,
        )
        observed_ax.set_xlim(xlim)
        observed_ax.set_ylim(ylim)
        observed_ax.set_aspect("equal", adjustable="box")
        observed_ax.set_xticks([])
        observed_ax.set_yticks([])
        for spine in observed_ax.spines.values():
            spine.set_visible(False)

        for ax, (source, display, canonical) in zip(axes[1:], METHODS):
            style = method_style(canonical)
            predicted = prediction_umap[source][display_indices[source]]
            ax.scatter(
                real_umap[:, 0],
                real_umap[:, 1],
                s=1.5,
                c=real_style.color,
                marker="o",
                alpha=0.16,
                linewidths=0,
                rasterized=True,
            )
            scatter_kwargs: dict[str, object] = {
                "s": 2.4,
                "c": style.color,
                "marker": style.marker,
                "alpha": 0.66,
                "linewidths": 0,
                "rasterized": True,
            }
            if canonical == "TrajectoryNet":
                scatter_kwargs.update(
                    {
                        "s": 3.2,
                        "edgecolors": style.markeredgecolor,
                        "linewidths": 0.10,
                    }
                )
            ax.scatter(
                predicted[:, 0],
                predicted[:, 1],
                **scatter_kwargs,
            )
            ax.set_title(
                display + "\n" + rf"RNA $W_2={w2[source]:.3f}$",
                pad=3,
                linespacing=1.12,
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color="#B3B3B3",
                markersize=5.5,
                label="All real stages",
            ),
            Line2D(
                [0],
                [0],
                marker=real_style.marker,
                linestyle="",
                color=real_style.color,
                markersize=5.5,
                label=f"Real {stage}",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color="#777777",
                markersize=5.5,
                label="Predicted",
            ),
        ]
        fig.suptitle(
            f"Held-out {stage} RNA distribution",
            fontsize=12,
            fontweight="normal",
            y=0.985,
        )
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=3,
            frameon=False,
            handletextpad=0.35,
            columnspacing=1.1,
        )
        for extension, options in (
            ("png", {"dpi": 600}),
            ("pdf", {}),
            ("svg", {"dpi": 600}),
        ):
            for collection in fig.axes[0].figure.findobj(
                match=mpl.collections.PathCollection
            ):
                collection.set_rasterized(extension != "pdf")
            fig.savefig(
                stem.with_suffix(f".{extension}"),
                facecolor="white",
                **options,
            )
        plt.close(fig)

        coordinate_payload: dict[str, np.ndarray] = {
            "methods": np.asarray(
                [source for source, _, _ in METHODS], dtype=str
            ),
            "atlas_umap": atlas_umap,
            "real_umap": real_umap,
            "xlim": np.asarray(xlim, dtype=np.float32),
            "ylim": np.asarray(ylim, dtype=np.float32),
        }
        for index, (source, _, _) in enumerate(METHODS):
            coordinate_payload[f"prediction_umap_{index}"] = prediction_umap[
                source
            ]
            coordinate_payload[f"display_indices_{index}"] = display_indices[
                source
            ]
        np.savez_compressed(coordinates_path, **coordinate_payload)
        manifest["scenarios"][scenario] = {
            "heldout_stage": stage,
            "prediction_cache": str(Path(config["prediction_cache"])),
            "trajectorynet_base_prediction": str(
                Path(config["base_prediction"])
            ),
            "scores": str(Path(config["scores"])),
            "figure": str(stem.with_suffix(".png")),
            "coordinates": str(coordinates_path),
            "displayed_rna_w2": w2,
            "observed_panel": (
                f"All-stage real RNA atlas in gray with observed {stage} "
                "highlighted in black"
            ),
            "shared_xlim": xlim,
            "shared_ylim": ylim,
        }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest_path, flush=True)


if __name__ == "__main__":
    main()
