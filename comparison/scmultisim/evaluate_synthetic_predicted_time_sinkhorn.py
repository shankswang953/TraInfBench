#!/usr/bin/env python
"""Compare fitted-time RNA10 distributions with a shared Sinkhorn metric.

The script evaluates all 2,006 cells propagated from time1 against the observed
time2, time3, and time4 snapshots. Every method is scored in the same normalized
RNA10 space. Unbalanced methods are reported both with their native particle
mass and with equal particle weights so that the metric can be compared with an
ordinary equal-point trajectory plot.
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
import json
import math
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from geomloss import SamplesLoss


ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = Path("external/COATI/Synthetic")
DATA_DIR = SYNTHETIC / "5scRNA"
DEFAULT_EXTERNAL_CACHE = (
    ROOT
    / "results/synthetic_rna10_all_method_coati_protocol_comparison"
    / "aligned_trajectory_cache.npz"
)
DEFAULT_COATI_CACHE = (
    SYNTHETIC
    / "UnbalancedSyncSweep/outputs/bio_all1_alpha100_d10_e1"
    / "observed_time_distribution/all_initial_observed_time_predictions.npz"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/synthetic_rna10_same_space_predicted_time_sinkhorn"
)

TIME_KEYS = ("time2", "time3", "time4")
METHOD_ORDER = (
    "CytoBridge balanced",
    "CytoBridge unbalanced",
    "COATI balanced",
    "COATI unbalanced",
    "TrajectoryNet",
    "MIOFlow",
    "TIGON",
    "Initial cells (no movement)",
)
METHOD_COLORS = {
    "CytoBridge balanced": "#4C78A8",
    "CytoBridge unbalanced": "#72B7B2",
    "COATI balanced": "#F58518",
    "COATI unbalanced": "#E45756",
    "TrajectoryNet": "#7A5195",
    "MIOFlow": "#54A24B",
    "TIGON": "#B279A2",
    "Initial cells (no movement)": "#8A8A8A",
}
NATIVE_MASS_METHODS = {
    "CytoBridge unbalanced",
    "COATI unbalanced",
    "TIGON",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-cache", type=Path, default=DEFAULT_EXTERNAL_CACHE)
    parser.add_argument(
        "--cytobridge-cache",
        type=Path,
        default=None,
        help="Optional cache containing CytoBridge predictions; defaults to --external-cache.",
    )
    parser.add_argument("--coati-cache", type=Path, default=DEFAULT_COATI_CACHE)
    parser.add_argument(
        "--rna-data",
        type=Path,
        default=DATA_DIR / "all_time_scRNA_pca10.npz",
    )
    parser.add_argument(
        "--rna-norm",
        type=Path,
        default=DATA_DIR / "primal_norm_params_rna10_w2.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sinkhorn-blur", type=float, default=0.05)
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda"), default="cpu"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _normalise_weights(values: np.ndarray, *, are_log_weights: bool) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(weights).all():
        raise ValueError("Particle weights contain NaN or infinity")
    if are_log_weights:
        weights = np.exp(weights - float(weights.max()))
    if np.any(weights < 0):
        raise ValueError("Particle weights must be non-negative")
    total = float(weights.sum())
    if not total > 0:
        raise ValueError("Particle weights have zero total mass")
    return weights / total


def _validate_points(name: str, values: np.ndarray, n_expected: int = 2006) -> np.ndarray:
    points = np.asarray(values, dtype=np.float32)
    if points.shape != (n_expected, 10):
        raise ValueError(f"Unexpected {name} shape: {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return points


def _load_predictions(
    external_path: Path,
    cytobridge_path: Path,
    coati_path: Path,
) -> tuple[
    dict[str, list[np.ndarray]],
    dict[str, list[np.ndarray | None]],
    dict[str, str],
]:
    external = np.load(external_path, allow_pickle=False)
    cytobridge = np.load(cytobridge_path, allow_pickle=False)
    coati = np.load(coati_path, allow_pickle=False)

    predictions: dict[str, list[np.ndarray]] = {}
    native_weights: dict[str, list[np.ndarray | None]] = {}
    sources: dict[str, str] = {}

    external_specs = (
        ("CytoBridge balanced", "cytobridge_balanced", False, cytobridge, cytobridge_path),
        ("CytoBridge unbalanced", "cytobridge_unbalanced", True, cytobridge, cytobridge_path),
        ("TrajectoryNet", "trajectorynet", False, external, external_path),
        ("MIOFlow", "mioflow", False, external, external_path),
        ("TIGON", "tigon", True, external, external_path),
    )
    for display, prefix, uses_mass, archive, source_path in external_specs:
        point_key = f"{prefix}_all_initial_selected_rna10_raw"
        weight_key = f"{prefix}_all_initial_selected_log_weights"
        points = np.asarray(archive[point_key], dtype=np.float32)
        if points.shape != (3, 2006, 10):
            raise ValueError(f"Unexpected {point_key} shape: {points.shape}")
        predictions[display] = [
            _validate_points(f"{display} {time_key}", points[index])
            for index, time_key in enumerate(TIME_KEYS)
        ]
        if uses_mass:
            log_weights = np.asarray(archive[weight_key], dtype=np.float64)
            if log_weights.shape != (3, 2006):
                raise ValueError(f"Unexpected {weight_key} shape: {log_weights.shape}")
            native_weights[display] = [
                _normalise_weights(log_weights[index], are_log_weights=True)
                for index in range(3)
            ]
        else:
            native_weights[display] = [None, None, None]
        sources[display] = str(source_path.resolve())

    for display, prefix, uses_mass in (
        ("COATI balanced", "coati_bal", False),
        ("COATI unbalanced", "coati_unbal", True),
    ):
        predictions[display] = [
            _validate_points(
                f"{display} {time_key}",
                coati[f"{prefix}_{time_key}_rna"],
            )
            for time_key in TIME_KEYS
        ]
        if uses_mass:
            native_weights[display] = [
                _normalise_weights(
                    coati[f"{prefix}_{time_key}_weights"], are_log_weights=False
                )
                for time_key in TIME_KEYS
            ]
        else:
            native_weights[display] = [None, None, None]
        sources[display] = str(coati_path.resolve())

    return predictions, native_weights, sources


def _sinkhorn_divergence(
    loss_fn: SamplesLoss,
    predicted: np.ndarray,
    observed: np.ndarray,
    predicted_weights: np.ndarray | None,
    device: torch.device,
) -> float:
    x = torch.as_tensor(predicted, dtype=torch.float32, device=device)
    y = torch.as_tensor(observed, dtype=torch.float32, device=device)
    with torch.inference_mode():
        if predicted_weights is None:
            value = loss_fn(x, y)
        else:
            a = torch.as_tensor(predicted_weights, dtype=torch.float32, device=device)
            b = torch.full(
                (len(observed),),
                1.0 / len(observed),
                dtype=torch.float32,
                device=device,
            )
            value = loss_fn(a, x, b, y)
    return max(float(value.detach().cpu()), 0.0)


def _effective_sample_size_fraction(weights: np.ndarray | None, n: int) -> float:
    if weights is None:
        return 1.0
    ess = 1.0 / float(np.sum(np.square(weights)))
    return ess / n


def _make_plot(scores: pd.DataFrame, output_dir: Path, blur: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
    x = np.arange(3)
    for axis, weighting, title in zip(
        axes,
        ("native_mass", "equal_particle"),
        ("Native mass", "Equal particle weights"),
    ):
        subset = scores[scores["weighting"] == weighting]
        for method in METHOD_ORDER:
            values = (
                subset[subset["method"] == method]
                .set_index("time_key")
                .reindex(TIME_KEYS)["sinkhorn_divergence"]
                .to_numpy()
            )
            if not np.isfinite(values).all():
                continue
            is_baseline = method == "Initial cells (no movement)"
            axis.plot(
                x,
                values,
                color=METHOD_COLORS[method],
                marker="o" if not is_baseline else None,
                markersize=5,
                linewidth=2.0 if not is_baseline else 1.4,
                linestyle="--" if is_baseline else "-",
                alpha=0.9 if not is_baseline else 0.7,
                label=method,
            )
        axis.set_title(title)
        axis.set_xticks(x, ("time2", "time3", "time4"))
        axis.set_xlabel("Predicted time")
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    axes[0].set_ylabel("Debiased Sinkhorn divergence (lower is better)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(f"Observed-time fit in normalized RNA10 space (blur={blur:g})", y=0.99)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.84, bottom=0.26, wspace=0.16)
    fig.savefig(output_dir / "predicted_time_sinkhorn.png", dpi=300, facecolor="white")
    fig.savefig(output_dir / "predicted_time_sinkhorn.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    outputs = (
        output_dir / "predicted_time_sinkhorn.csv",
        output_dir / "predicted_time_sinkhorn_summary.csv",
        output_dir / "predicted_time_sinkhorn.json",
        output_dir / "predicted_time_sinkhorn.png",
        output_dir / "predicted_time_sinkhorn.pdf",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs; pass --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    norm_payload = torch.load(args.rna_norm, map_location="cpu", weights_only=False)
    scale = float(norm_payload["scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid RNA scale: {scale}")

    rna_archive = np.load(args.rna_data, allow_pickle=False)
    initial_raw = _validate_points("observed time1", rna_archive["time1"])
    observed_raw = [
        np.asarray(rna_archive[time_key], dtype=np.float32) for time_key in TIME_KEYS
    ]
    for time_key, values in zip(TIME_KEYS, observed_raw):
        if values.ndim != 2 or values.shape[1] != 10 or not np.isfinite(values).all():
            raise ValueError(f"Invalid observed {time_key} array: {values.shape}")

    cytobridge_cache = args.cytobridge_cache or args.external_cache
    predictions_raw, native_weights, sources = _load_predictions(
        args.external_cache, cytobridge_cache, args.coati_cache
    )
    predictions_raw["Initial cells (no movement)"] = [initial_raw] * 3
    native_weights["Initial cells (no movement)"] = [None, None, None]
    sources["Initial cells (no movement)"] = str(args.rna_data.resolve())

    device = torch.device(args.device)
    loss_fn = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=args.sinkhorn_blur,
        debias=True,
        backend="tensorized",
    )

    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        print(f"[sinkhorn] {method}", flush=True)
        for time_index, time_key in enumerate(TIME_KEYS):
            predicted = predictions_raw[method][time_index] / scale
            observed = observed_raw[time_index] / scale
            weights = native_weights[method][time_index]
            native_value = _sinkhorn_divergence(
                loss_fn, predicted, observed, weights, device
            )
            weightings = [("native_mass", weights, native_value)]
            if method in NATIVE_MASS_METHODS:
                equal_value = _sinkhorn_divergence(
                    loss_fn, predicted, observed, None, device
                )
            else:
                equal_value = native_value
            weightings.append(("equal_particle", None, equal_value))
            for weighting, effective_weights, divergence in weightings:
                rows.append(
                    {
                        "method": method,
                        "time_key": time_key,
                        "physical_time": float(time_index + 1),
                        "weighting": weighting,
                        "sinkhorn_divergence": divergence,
                        "w2_like_sqrt_2s": math.sqrt(2.0 * divergence),
                        "n_predicted": len(predicted),
                        "n_observed": len(observed),
                        "particle_ess_fraction": _effective_sample_size_fraction(
                            effective_weights, len(predicted)
                        ),
                        "rna_scale": scale,
                        "sinkhorn_blur": args.sinkhorn_blur,
                    }
                )

    scores = pd.DataFrame(rows)
    scores.to_csv(outputs[0], index=False, float_format="%.10g")

    summary_rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        method_scores = scores[scores["method"] == method]
        row: dict[str, object] = {"method": method}
        for weighting in ("native_mass", "equal_particle"):
            selected = method_scores[method_scores["weighting"] == weighting]
            row[f"{weighting}_mean_sinkhorn"] = float(
                selected["sinkhorn_divergence"].mean()
            )
            row[f"{weighting}_terminal_sinkhorn"] = float(
                selected.loc[
                    selected["time_key"] == "time4", "sinkhorn_divergence"
                ].iloc[0]
            )
            row[f"{weighting}_mean_w2_like"] = float(
                selected["w2_like_sqrt_2s"].mean()
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outputs[1], index=False, float_format="%.10g")

    payload = {
        "metric": {
            "name": "debiased Sinkhorn divergence",
            "loss": "sinkhorn",
            "p": 2,
            "blur": args.sinkhorn_blur,
            "backend": "tensorized",
            "space": "RNA10 raw PCA divided by the shared primal W2 scale",
            "rna_scale": scale,
            "target_weights": "uniform over each observed snapshot",
            "w2_like_definition": "sqrt(2 * debiased Sinkhorn divergence)",
        },
        "protocol": {
            "source": "all 2,006 observed time1 cells",
            "predicted_times": list(TIME_KEYS),
            "native_mass_methods": sorted(NATIVE_MASS_METHODS),
            "equal_particle_purpose": (
                "Sensitivity analysis matching ordinary equal-point visualizations"
            ),
            "no_movement_baseline": "observed time1 cells compared with each later snapshot",
        },
        "inputs": {
            "external_cache": str(args.external_cache.resolve()),
            "cytobridge_cache": str(cytobridge_cache.resolve()),
            "coati_cache": str(args.coati_cache.resolve()),
            "rna_data": str(args.rna_data.resolve()),
            "rna_norm": str(args.rna_norm.resolve()),
            "method_sources": sources,
        },
        "outputs": {path.name: str(path.resolve()) for path in outputs},
        "summary": summary.to_dict(orient="records"),
    }
    outputs[2].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _make_plot(scores, output_dir, args.sinkhorn_blur)
    print(f"Saved results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
