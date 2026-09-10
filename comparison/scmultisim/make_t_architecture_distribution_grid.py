#!/usr/bin/env python3
"""Same observed-time distribution grid, but the columns are T architectures.

All four models are Balanced Sync at Cy=0.1 with an identical budget; only the
frozen cross-modal map T differs.  The point of the figure is whether the sync
result depends on how T is parameterised.

Projection note: the original grid embeds predictions with the fitted UMAP
reducer's `transform`.  That joblib was written under a newer Python and no
longer unpickles in any environment here, so predictions are placed on the
same observed embedding by inverse-distance 15-NN instead -- the projection
already used elsewhere in this project for exactly this purpose.  Every
column in this figure uses that same projection, so they stay comparable to
each other; they are not pixel-comparable to the original four-method grid.
The correct rates are unaffected: they are computed in raw RNA10, not UMAP.
"""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import importlib.util
import json
import os
import sys
from pathlib import Path


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

import joblib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


HERE = (_REPO / "external/COATI/Synthetic/UnbalancedSyncSweep")
REPO_ROOT = HERE.parents[1]
SYNTHETIC = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from src.Neural import MLPVectorField
from src.utility import forward_ode


OUT = HERE / "outputs/t_architecture_distribution"
DATA = SYNTHETIC / "5scRNA/all_time_scRNA_pca10.npz"
LABELS = SYNTHETIC / "5scRNA/all_time_scRNA_label.npz"
NORM = SYNTHETIC / "5scRNA/primal_norm_params_rna10_w2.pt"
UMAP = Path(
    "results/"
    "cytobridge_synthetic_rna10_n128_i3000_comparison/"
    "rna10_umap/rna10_umap_model.joblib"
)
SYNC = SYNTHETIC / "BalancedSyncSweep/outputs/runs"
MODELS = {
    "film": {
        "label": "FiLM",
        "unbalanced": False,
        "run": SYNC / "density_10_energy_1_cy_0p1",
    },
    "mlp": {
        "label": "MLP",
        "unbalanced": False,
        "run": SYNC / "density_10_energy_1_cy_0p1_Tmlp",
    },
    "tmlp": {
        "label": "temporal MLP",
        "unbalanced": False,
        "run": SYNC / "density_10_energy_1_cy_0p1_Ttmlp",
    },
    "tridge": {
        "label": "temporal ridge",
        "unbalanced": False,
        "run": SYNC / "density_10_energy_1_cy_0p1_Ttridge",
    },
    "ridge": {
        "label": "ridge",
        "unbalanced": False,
        "run": SYNC / "density_10_energy_1_cy_0p1_Tridge",
    },
}
# Paired by family: static map first, then its time-varying counterpart.
MODEL_ORDER = ("mlp", "tmlp", "ridge", "tridge")
WIDTH_MM = 207.6
HEIGHT_MM = 132.0
OBSERVED_TIMES = (("time2", 20, 1.0), ("time3", 40, 2.0), ("time4", 60, 3.0))
ALLOWED = {"4_1": {"4_1"}, "4_5": {"4_5", "5_2", "5_3"}}
# Preserve the colors used for the two source labels in the original trajectory plots.
# Colours come from the shared TraInfBench standard rather than being fixed
# here, so this figure matches the method-comparison strips instead of keeping
# the older hardcoded Paul Tol pair.
_STYLE_PATH = Path(
    "common/trainfbench_plot_style.py"
)
_style_spec = importlib.util.spec_from_file_location(
    "trainfbench_plot_style", _STYLE_PATH
)
_style = importlib.util.module_from_spec(_style_spec)
sys.modules["trainfbench_plot_style"] = _style
_style_spec.loader.exec_module(_style)
SOURCE_COLORS = {
    "4_1": _style.synthetic_population_color("4_1"),
    "4_5": _style.synthetic_population_color("4_5"),
}
OBSERVED_GREY = "#B3B3B3"


def checkpoint_path(run: Path) -> Path:
    path = run / "checkpoint/path_param.pth"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def integrate_observed_times(
    checkpoint: Path,
    initial: np.ndarray,
    scale: float,
    unbalanced: bool,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint_payload.get("iteration", -1)) != 3000:
        raise ValueError(f"Expected iteration 3000: {checkpoint}")
    func = MLPVectorField(
        dim=10,
        hidden_dim=400,
        n_layers=3,
        activation="leaky_relu",
        unbalanced=unbalanced,
        alpha_growth=100.0 if unbalanced else 0.0,
    )
    func.load_state_dict(checkpoint_payload["func_state_dict"])
    func.eval()

    points = {time_name: [] for time_name, _, _ in OBSERVED_TIMES}
    weights = {time_name: [] for time_name, _, _ in OBSERVED_TIMES}
    with torch.no_grad():
        for start in range(0, len(initial), batch_size):
            initial_batch = torch.from_numpy(initial[start : start + batch_size] / scale)
            result = forward_ode(
                func,
                initial_batch,
                torch.device("cpu"),
                unbalancedModel=unbalanced,
                viz_timesteps=61,
                t_start=0.0,
                t_end=3.0,
                method="rk4",
            )
            trajectory = result[0].detach().cpu().numpy() * scale
            log_weights = result[1].detach().cpu().numpy() if unbalanced else None
            for time_name, step, _ in OBSERVED_TIMES:
                points[time_name].append(trajectory[step])
                if unbalanced:
                    weights[time_name].append(
                        np.exp(log_weights[step, :, 0].astype(np.float64))
                    )

    output = {
        f"{time_name}_rna": np.concatenate(values).astype(np.float32)
        for time_name, values in points.items()
    }
    if unbalanced:
        output.update({
            f"{time_name}_weights": np.concatenate(values).astype(np.float64)
            for time_name, values in weights.items()
        })
    return output


def transform_chunks(reducer, points: np.ndarray, chunk_size: int = 10000) -> np.ndarray:
    return np.concatenate(
        [reducer.transform(points[i : i + chunk_size]) for i in range(0, len(points), chunk_size)]
    ).astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data_archive = np.load(DATA)
    observed_arrays = {
        f"time{i}": np.asarray(data_archive[f"time{i}"], dtype=np.float32)
        for i in range(1, 5)
    }
    initial = observed_arrays["time1"]
    label_archive = np.load(LABELS, allow_pickle=True)
    observed_labels = {
        f"time{i}": label_archive[f"time{i}"].astype(str) for i in range(1, 5)
    }
    initial_source = observed_labels["time1"]
    scale = float(torch.load(NORM, map_location="cpu", weights_only=False)["scale"])
    observed_all = np.concatenate(list(observed_arrays.values()))
    cache_path = OUT / "all_initial_observed_time_predictions.npz"
    # The fitted reducer is only needed to embed predictions that are not
    # cached yet.  Loading it unconditionally makes a pure re-plot fail: the
    # stored joblib was written by a different numba build and no longer
    # unpickles here.  The observed embedding comes from its own cache, whose
    # row order is checked against the labels below.
    if True:
        observed_cache = np.load(UMAP.parent / "observed_rna10_umap.npz", allow_pickle=True)
        observed_embedding = np.asarray(observed_cache["observed_umap"], dtype=np.float32)
        expected = np.concatenate([observed_labels[f"time{i}"] for i in range(1, 5)])
        if len(observed_embedding) != len(observed_all) or not (
            observed_cache["observed_population"].astype(str) == expected
        ).all():
            raise ValueError(
                "The cached observed UMAP is not the time1..time4 concatenation "
                "of the RNA10 labels"
            )
    offsets = np.cumsum([0] + [len(observed_arrays[f"time{i}"]) for i in range(1, 5)])
    observed_umap = {
        f"time{i}": observed_embedding[offsets[i - 1] : offsets[i]] for i in range(1, 5)
    }

    if cache_path.exists():
        cached = np.load(cache_path)
        arrays = {key: cached[key] for key in cached.files}
    else:
        arrays = {}
        for model_key in MODEL_ORDER:
            config = MODELS[model_key]
            result = integrate_observed_times(
                checkpoint_path(config["run"]),
                initial,
                scale,
                config["unbalanced"],
            )
            for key, value in result.items():
                arrays[f"{model_key}_{key}"] = value

        # Replace terminal states and weights with the already audited all-initial evaluation files.
        for model_key in MODEL_ORDER:
            config = MODELS[model_key]
            endpoints = np.load(config["run"] / "all_initial/all_initial_endpoints.npz")
            arrays[f"{model_key}_time4_rna"] = np.asarray(endpoints["endpoint"], dtype=np.float32)
            if config["unbalanced"]:
                arrays[f"{model_key}_time4_weights"] = np.asarray(
                    endpoints["final_weights"], dtype=np.float64
                )

        # Inverse-distance 15-NN onto the observed embedding (see module docstring).
        neighbour_model = NearestNeighbors(n_neighbors=15, n_jobs=-1).fit(observed_all)
        for model_key in MODEL_ORDER:
            for time_name, _, _ in OBSERVED_TIMES:
                distances, indices = neighbour_model.kneighbors(
                    arrays[f"{model_key}_{time_name}_rna"]
                )
                inverse = 1.0 / np.maximum(distances, 1e-6)
                inverse /= inverse.sum(axis=1, keepdims=True)
                arrays[f"{model_key}_{time_name}_umap"] = np.sum(
                    observed_embedding[indices] * inverse[:, :, None], axis=1
                ).astype(np.float32)
        np.savez_compressed(cache_path, **arrays)

    classifiers = {
        time_name: KNeighborsClassifier(n_neighbors=15, weights="distance").fit(
            observed_arrays[time_name], observed_labels[time_name]
        )
        for time_name, _, _ in OBSERVED_TIMES
    }
    metrics = {
        "definition": {
            "n_initial": int(len(initial)),
            "initial_source_counts": {
                label: int(np.sum(initial_source == label))
                for label in sorted(np.unique(initial_source))
            },
            "classifier": "distance-weighted 15-NN against the corresponding observed snapshot",
            "allowed_populations": {key: sorted(value) for key, value in ALLOWED.items()},
            "balanced": "unweighted correct rate",
            "unbalanced": "particle-weighted correct rate at each time",
        },
        "models": {},
    }
    correct_masks: dict[tuple[str, str], np.ndarray] = {}
    for model_key in MODEL_ORDER:
        config = MODELS[model_key]
        model_metrics = {"label": config["label"], "times": {}}
        for time_name, _, model_time in OBSERVED_TIMES:
            predicted = classifiers[time_name].predict(
                arrays[f"{model_key}_{time_name}_rna"]
            ).astype(str)
            correct = np.asarray([
                prediction in ALLOWED[source]
                for source, prediction in zip(initial_source, predicted)
            ])
            correct_masks[(model_key, time_name)] = correct
            count_rate = float(correct.mean())
            if config["unbalanced"]:
                weights = arrays[f"{model_key}_{time_name}_weights"]
                rate = float(weights[correct].sum() / (weights.sum() + 1e-12))
            else:
                rate = count_rate
            model_metrics["times"][time_name] = {
                "model_time": model_time,
                "correct_rate": rate,
                "unweighted_correct_rate": count_rate,
            }
        metrics["models"][model_key] = model_metrics
    (OUT / "observed_time_distribution_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    font_manager.findfont("Arial", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "font.weight": "normal",
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        # Without these the text is emitted as Type 3 glyph procedures, which
        # are neither selectable nor editable and are rejected by most journals.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # Mathtext ignores font.family; without these any $...$ would silently
        # fall back to DejaVu Sans and embed a second typeface.
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    })
    figure, axes = plt.subplots(
        3,
        len(MODEL_ORDER),
        # 207.6 mm canvas so this figure needs no rescaling next to the
        # Cy tables; at the old ~297 mm width a 12 pt label shrank to ~8 pt
        # once the figure was placed at text width.
        figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
        sharex="row",
        sharey="row",
    )
    figure.subplots_adjust(
        left=0.075, right=0.995, top=0.89, bottom=0.09, wspace=0.035, hspace=0.13
    )
    rng = np.random.default_rng(0)
    order = rng.permutation(len(initial_source))
    point_colors = np.asarray([SOURCE_COLORS[label] for label in initial_source])
    for row_index, (time_name, _, _) in enumerate(OBSERVED_TIMES):
        background = observed_umap[time_name]
        row_points = np.concatenate(
            [background]
            + [arrays[f"{model_key}_{time_name}_umap"] for model_key in MODEL_ORDER],
            axis=0,
        )
        lower = row_points.min(axis=0)
        upper = row_points.max(axis=0)
        if row_index > 0:
            # Sparse upper-tail UMAP outliers otherwise create large empty bands.
            upper[1] = np.quantile(row_points[:, 1], 0.995)
        padding = np.maximum((upper - lower) * 0.035, 0.25)
        for column_index, model_key in enumerate(MODEL_ORDER):
            axis = axes[row_index, column_index]
            axis.scatter(
                background[:, 0],
                background[:, 1],
                s=4,
                color=OBSERVED_GREY,
                alpha=0.18,
                edgecolors="none",
                rasterized=True,
            )
            predicted_umap = arrays[f"{model_key}_{time_name}_umap"]
            axis.scatter(
                predicted_umap[order, 0],
                predicted_umap[order, 1],
                s=11,
                c=point_colors[order],
                alpha=0.46,
                edgecolors="none",
                rasterized=True,
            )
            rate = metrics["models"][model_key]["times"][time_name]["correct_rate"]
            plot_lower = lower - padding
            plot_upper = upper + padding
            plot_lower[1] -= 0.10 * (plot_upper[1] - plot_lower[1])
            axis.text(
                0.03,
                0.015,
                f"Correct rate = {rate:.1%}",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=12,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xlim(plot_lower[0], plot_upper[0])
            axis.set_ylim(plot_lower[1], plot_upper[1])
            axis.set_aspect("auto")
            if row_index == 0:
                axis.set_title(MODELS[model_key]["label"], fontsize=12, fontweight="normal", pad=8)
        axes[row_index, 0].set_ylabel(time_name, fontsize=12, fontweight="normal")
    figure.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="", color=SOURCE_COLORS["4_1"],
                label="4_1 branch", markersize=7,
            ),
            Line2D(
                [0], [0], marker="o", linestyle="", color=SOURCE_COLORS["4_5"],
                label="4_5 branch", markersize=7,
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.006),
        fontsize=12,
    )
    png_path = OUT / "observed_time_distribution_grid.png"
    figure.savefig(png_path, dpi=300, facecolor="white")
    figure.savefig(png_path.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    print(json.dumps(metrics, indent=2))
    print(png_path)


if __name__ == "__main__":
    main()
