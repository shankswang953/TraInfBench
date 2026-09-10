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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
TRAINF_DATA = Path("external/COATI/Gastrulation/data")
STAGE_KEYS = ["time0", "time1", "time2", "time3"]
REAL_COLOR = "#2F6FB0"
PREDICTED_COLOR = "#D84A3A"
OTHER_COLOR = "#B8B8B8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot selected LOO ATAC distributions on the fixed ATAC UMAP."
    )
    parser.add_argument("--time", type=int, choices=(1, 2), required=True)
    parser.add_argument("--c-y", type=float, default=0.5)
    parser.add_argument("--umap-batch-size", type=int, default=4096)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--publication-compact",
        action="store_true",
        help=(
            "Write a compact publication subfigure with only held-out real and "
            "predicted ATAC points. Existing legacy outputs are left untouched."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_scale(path: Path) -> float:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return float(np.asarray(state["scale"]).reshape(-1)[0])


def batched_transform(model, values: np.ndarray, batch_size: int) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(model.transform(values[start : start + batch_size]), dtype=np.float32)
            for start in range(0, values.shape[0], batch_size)
        ],
        axis=0,
    )


def systematic_resample(log_mass: np.ndarray, n: int) -> np.ndarray:
    mass = np.asarray(log_mass, dtype=np.float64)
    mass = np.exp(np.clip(mass - np.max(mass), -700.0, 0.0))
    mass /= mass.sum()
    cumulative = np.cumsum(mass)
    cumulative[-1] = 1.0
    positions = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    return np.searchsorted(cumulative, positions, side="left").astype(np.int64)


def short_name(method: str) -> str:
    names = {
        "CytoBridge balanced 20k": "CytoBridge bal.",
        "CytoBridge unbalanced 20k": "CytoBridge unbal.",
        "MIOFlow 20k": "MIOFlow",
        "TIGON 20k": "TIGON",
    }
    return names.get(method, method)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12.0,
            "axes.titlesize": 12.0,
            "axes.labelsize": 12.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 12.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
        }
    )


def main() -> None:
    args = parse_args()
    stage_index = args.time
    stage = "E8.0" if args.time == 1 else "E8.5"
    cache = (
        ROOT
        / "results/gastrulation_loo_time1_coverage_validity/loo_time1_shared_t_atac_predictions.npz"
        if args.time == 1
        else ROOT
        / "results/gastrulation_loo_time2_shared_t_atac/loo_time2_shared_t_predictions.npz"
    )
    scores_path = (
        ROOT
        / f"results/gastrulation_loo_time{args.time}_shared_t_atac"
        / f"loo_time{args.time}_shared_t_scores.csv"
    )
    output_dir = args.output_dir or (
        ROOT / f"results/gastrulation_loo_time{args.time}_atac_umap_selected"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem_suffix = "_publication_compact_v3" if args.publication_compact else ""
    stem = output_dir / (
        f"loo_time{args.time}_atac_umap_selected_methods{stem_suffix}"
    )
    outputs = [stem.with_suffix(extension) for extension in (".png", ".pdf", ".svg")]
    outputs.extend(
        [
            output_dir
            / f"loo_time{args.time}_atac_umap_selected_coordinates{stem_suffix}.npz",
            output_dir / f"plot_manifest{stem_suffix}.json",
        ]
    )
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise FileExistsError(f"Outputs already exist in {output_dir}; pass --overwrite")

    raw_atac = np.load(TRAINF_DATA / "atac_lsi_by_time_14D.npz", allow_pickle=True)
    stage_values = [np.asarray(raw_atac[key], dtype=np.float32) for key in STAGE_KEYS]
    stage_sizes = [values.shape[0] for values in stage_values]
    all_real = np.concatenate(stage_values, axis=0)
    offsets = np.cumsum([0, *stage_sizes])
    heldout_mask = np.zeros(all_real.shape[0], dtype=bool)
    heldout_mask[offsets[stage_index] : offsets[stage_index + 1]] = True

    umap_model = joblib.load(TRAINF_DATA / "atac_umap_model_14D.joblib")
    raw_training = np.asarray(umap_model._raw_data)
    if raw_training.shape != all_real.shape:
        raise ValueError("ATAC UMAP training data shape mismatch")
    max_abs = float(np.max(np.abs(raw_training - all_real)))
    if max_abs > 1e-6:
        raise ValueError(f"ATAC UMAP training data mismatch: max_abs={max_abs}")
    real_umap = np.asarray(umap_model.embedding_, dtype=np.float32)

    with np.load(cache, allow_pickle=True) as cached:
        methods = cached["methods"].astype(str).tolist()
        predictions = np.asarray(cached["atac_predictions"], dtype=np.float32)
        log_masses = np.asarray(cached["native_log_masses"], dtype=np.float32)
        has_native = np.asarray(cached["has_native_mass"], dtype=bool)
    method_to_index = {method: index for index, method in enumerate(methods)}
    selected = [
        f"BSOT C_y={args.c_y:.1f}",
        f"USOT C_y={args.c_y:.1f}",
        "MIOFlow 20k",
        "CytoBridge balanced 20k",
        "CytoBridge unbalanced 20k",
        "TIGON 20k",
    ]
    missing = sorted(set(selected) - set(method_to_index))
    if missing:
        raise KeyError(f"Missing predictions for {missing} in {cache}")

    scores = pd.read_csv(scores_path).set_index("method")
    missing_scores = sorted(set(selected) - set(scores.index.astype(str)))
    if missing_scores:
        raise KeyError(f"Missing Sinkhorn values for {missing_scores}")
    sinkhorn = scores["atac_sinkhorn_divergence"].astype(float).to_dict()
    w2_distance = {
        method: float(np.sqrt(2.0 * max(value, 0.0)))
        for method, value in sinkhorn.items()
    }

    atac_scale = load_scale(TRAINF_DATA / "secondary_norm_params.pt")
    prediction_umap: dict[str, np.ndarray] = {}
    display_indices: dict[str, np.ndarray] = {}
    for method in selected:
        index = method_to_index[method]
        prediction_umap[method] = batched_transform(
            umap_model,
            predictions[index] * atac_scale,
            args.umap_batch_size,
        )
        display_indices[method] = (
            systematic_resample(log_masses[index], predictions.shape[1])
            if has_native[index]
            else np.arange(predictions.shape[1], dtype=np.int64)
        )
        print(f"[UMAP] {method}", flush=True)

    heldout = real_umap[heldout_mask]
    real_for_limits = heldout if args.publication_compact else real_umap
    pooled = np.concatenate([real_for_limits, *prediction_umap.values()], axis=0)
    x_min, y_min = np.min(pooled, axis=0)
    x_max, y_max = np.max(pooled, axis=0)
    x_pad = max((x_max - x_min) * 0.025, 0.2)
    y_pad = max((y_max - y_min) * 0.025, 0.2)

    configure_style()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.2, 5.2) if args.publication_compact else (12.0, 8.0),
        sharex=True,
        sharey=True,
        facecolor="white",
    )
    other = real_umap[~heldout_mask]
    for panel_index, (ax, method) in enumerate(zip(axes.ravel(), selected)):
        predicted = prediction_umap[method][display_indices[method]]
        if not args.publication_compact:
            ax.scatter(
                other[:, 0],
                other[:, 1],
                s=0.22,
                c=OTHER_COLOR,
                alpha=0.12,
                linewidths=0,
                rasterized=True,
            )
        ax.scatter(
            heldout[:, 0],
            heldout[:, 1],
            s=5.0 if args.publication_compact else 3.8,
            c=REAL_COLOR,
            alpha=0.54,
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            predicted[:, 0],
            predicted[:, 1],
            s=5.0 if args.publication_compact else 3.8,
            c=PREDICTED_COLOR,
            alpha=0.54,
            linewidths=0,
            rasterized=True,
        )
        title = (
            rf"BSOT ($C_y={args.c_y:g}$)"
            if method.startswith("BSOT")
            else rf"USOT ($C_y={args.c_y:g}$)"
            if method.startswith("USOT")
            else short_name(method)
        )
        if args.publication_compact:
            ax.set_title(
                title
                + "\n"
                + rf"$W_2={w2_distance[method]:.3f}$",
                pad=3,
                linespacing=1.18,
            )
        else:
            ax.set_title(title, pad=24)
            ax.text(
                0.5,
                1.01,
                f"ATAC weighted Sinkhorn ↓  {sinkhorn[method]:.4f}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
            )
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if not args.publication_compact and panel_index // 3 == 1:
            ax.set_xlabel("ATAC UMAP 1")
        if not args.publication_compact and panel_index % 3 == 0:
            ax.set_ylabel("ATAC UMAP 2")

    legend = []
    if not args.publication_compact:
        legend.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=OTHER_COLOR,
                markersize=7,
                label="Other-stage real ATAC",
            )
        )
    legend.extend([
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=REAL_COLOR,
            markersize=7,
            label=f"Real {stage}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=PREDICTED_COLOR,
            markersize=7,
            label=f"Predicted {stage}",
        ),
    ])
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(legend),
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.5,
    )
    if args.publication_compact:
        fig.subplots_adjust(
            left=0.015,
            right=0.995,
            top=0.80,
            bottom=0.01,
            hspace=0.25,
            wspace=0.065,
        )
    else:
        fig.subplots_adjust(
            left=0.05,
            right=0.99,
            top=0.88,
            bottom=0.07,
            hspace=0.30,
            wspace=0.08,
        )
    for extension, options in (("png", {"dpi": 500}), ("pdf", {}), ("svg", {})):
        fig.savefig(stem.with_suffix(f".{extension}"), facecolor="white", bbox_inches="tight", **options)
    plt.close(fig)

    payload: dict[str, np.ndarray] = {
        "real_atac_umap": real_umap,
        "heldout_mask": heldout_mask,
        "methods": np.asarray(selected, dtype=str),
    }
    for index, method in enumerate(selected):
        payload[f"prediction_umap_{index}"] = prediction_umap[method]
        payload[f"display_indices_{index}"] = display_indices[method]
    np.savez_compressed(
        output_dir
        / f"loo_time{args.time}_atac_umap_selected_coordinates{stem_suffix}.npz",
        **payload,
    )
    manifest = {
        "task": f"Gastrulation LOO time{args.time}/{stage} selected-method ATAC UMAP",
        "excluded": ["T(real RNA) diagnostic", "TrajectoryNet"],
        "methods": selected,
        "sync_selection": f"pre-specified C_y={args.c_y:.1f} for both BSOT and USOT",
        "prediction_cache": str(cache),
        "sinkhorn_source": str(scores_path),
        "displayed_distance": (
            "d_W2=(2*S_epsilon)^0.5, where S_epsilon is the existing debiased "
            "p=2 Sinkhorn divergence on weighted ATAC microclusters."
        ),
        "w2_distance_estimates": {
            method: w2_distance[method] for method in selected
        },
        "sinkhorn_divergence_source_values": {
            method: sinkhorn[method] for method in selected
        },
        "umap_reference": str(TRAINF_DATA / "atac_umap_model_14D.joblib"),
        "umap_training_data_max_abs_check": max_abs,
        "native_mass_display": "Deterministic systematic resampling for unbalanced methods; uniform display for balanced methods.",
        "publication_compact": args.publication_compact,
        "point_layers": (
            [f"real {stage} ATAC", f"predicted {stage} ATAC"]
            if args.publication_compact
            else ["other-stage real ATAC", f"real {stage} ATAC", f"predicted {stage} ATAC"]
        ),
        "typography": "Arial, 12 pt throughout",
    }
    (output_dir / f"plot_manifest{stem_suffix}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[write] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
