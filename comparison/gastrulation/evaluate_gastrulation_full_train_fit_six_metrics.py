#!/usr/bin/env python
"""Evaluate full-train Gastrulation distribution fitting with six shared metrics.

All fitted time points are present during training. RNA predictions are evaluated
in the shared normalized RNA PCA50 space; predicted RNA is mapped through the
same frozen time-dependent T model before ATAC evaluation. The metric pipeline
is identical to the held-one-out six-metric analysis.
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
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from geomloss import SamplesLoss


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

from evaluate_gastrulation_full_cmcc import (  # noqa: E402
    STAGE_NAMES,
    UNKNOWN,
    _apply_t,
    _load_references,
    _load_t_model,
)
from evaluate_gastrulation_loo_time1_paired_1nn_atac import (  # noqa: E402
    _load_separate_modality_labels,
)
from evaluate_gastrulation_loo_time1_shared_t_atac import (  # noqa: E402
    _evaluate_distribution,
    _prepare_reference,
)
from evaluate_terminal_push import _load_trajectorynet_model  # noqa: E402
from plot_gastrulation_loo_terminal_floor_six_metrics import (  # noqa: E402
    METRICS,
    output_paths,
    plot_panel,
)
from trainfbench_plot_style import apply_nature_rc  # noqa: E402


TRAINF_ROOT = Path("external/COATI")
GASTRULATION_ROOT = TRAINF_ROOT / "Gastrulation"
DEFAULT_EXTERNAL_CACHE = (
    ROOT
    / "results/gastrulation_nonloo_all_rna_only_sinkhorn_pareto"
    / "all_rna_only_predictions.npz"
)
DEFAULT_FLOOR = (
    ROOT
    / "results/gastrulation_loo_shared_t_trajectorynet_terminal_floor"
    / "sampling_floor_summary.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "results/gastrulation_full_train_fit_six_metrics"

COATI_BALANCED = (
    GASTRULATION_ROOT
    / "FilmSyncEnergy/trajectory/primary_trajectory_s0_a0.3_iter20000.pt"
)
COATI_UNBALANCED = (
    GASTRULATION_ROOT
    / "UnbalancedSync_biological_num/trajectory_hpc_iter20000"
    / "primary_trajectory_s0_a0.3_iter20000.pt"
)
COATI_UNBALANCED_MASS = (
    GASTRULATION_ROOT
    / "UnbalancedSync_biological_num/trajectory_hpc_iter20000"
    / "mass_lnw_trajectory_s0_a0.3_iter20000.pt"
)
TRAJECTORYNET_CHECKPOINT = (
    ROOT / "results/trajectorynet_gastrulation_rna_20000/checkpt-20000.pth"
)

EXTERNAL_METHODS = (
    "TrajectoryNet",
    "MIOFlow",
    "TIGON",
    "CytoBridge balanced",
    "CytoBridge unbalanced",
)
METHOD_NAME = {
    "TrajectoryNet": "TrajectoryNet 20k",
    "MIOFlow": "MIOFlow 20k",
    "TIGON": "TIGON 20k",
    "CytoBridge balanced": "CytoBridge balanced 20k",
    "CytoBridge unbalanced": "CytoBridge unbalanced 20k",
}
NATIVE_MASS_METHODS = {"TIGON", "CytoBridge unbalanced"}

# reference index, external-cache index, COATI trajectory step, T time, tag,
# and TrajectoryNet density-direction intervals from real terminal E8.75.
STAGES = (
    (1, 0, 10, 1.0, "e80", ((1.5, 2.0), (1.0, 1.5))),
    (2, 1, 20, 2.0, "e85", ((1.5, 2.0),)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-cache", type=Path, default=DEFAULT_EXTERNAL_CACHE)
    parser.add_argument("--floor-summary", type=Path, default=DEFAULT_FLOOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--sinkhorn-clusters", type=int, default=512)
    parser.add_argument("--sinkhorn-blur", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_external_cache(
    path: Path,
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    cached = np.load(path, allow_pickle=False)
    methods = tuple(cached["methods"].astype(str).tolist())
    stage_keys = tuple(cached["stage_keys"].astype(str).tolist())
    if methods != EXTERNAL_METHODS:
        raise ValueError(f"Unexpected external methods: {methods}")
    if stage_keys != ("time1", "time2", "time3"):
        raise ValueError(f"Unexpected external stages: {stage_keys}")

    predictions: dict[str, list[np.ndarray]] = {}
    log_masses: dict[str, list[np.ndarray]] = {}
    for method_index, method in enumerate(methods):
        predictions[method] = [
            np.asarray(cached[f"prediction_{method_index}_{stage_index}"], dtype=np.float32)
            for stage_index in range(len(stage_keys))
        ]
        log_masses[method] = [
            np.asarray(cached[f"log_mass_{method_index}_{stage_index}"], dtype=np.float32)
            for stage_index in range(len(stage_keys))
        ]
    return predictions, log_masses


def _load_coati() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    balanced = torch.load(COATI_BALANCED, map_location="cpu")
    unbalanced = torch.load(COATI_UNBALANCED, map_location="cpu")
    unbalanced_mass = torch.load(COATI_UNBALANCED_MASS, map_location="cpu")
    expected = (26, 9018, 50)
    if tuple(balanced.shape) != expected or tuple(unbalanced.shape) != expected:
        raise ValueError(
            f"Unexpected COATI trajectory shapes: {balanced.shape}, {unbalanced.shape}"
        )
    if tuple(unbalanced_mass.shape) != (26, 9018, 1):
        raise ValueError(f"Unexpected COATI mass shape: {unbalanced_mass.shape}")
    return balanced, unbalanced, unbalanced_mass


def _trajectorynet_from_terminal(
    source_norm: np.ndarray,
    *,
    scale: float,
    intervals: tuple[tuple[float, float], ...],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Push real E8.75 cells through the full TrajectoryNet density direction."""

    model, model_args = _load_trajectorynet_model(
        TRAJECTORYNET_CHECKPOINT,
        source_norm.shape[1],
        device,
        "rk4",
        0.1,
    )
    if not np.isclose(float(model_args.time_scale), 0.5):
        raise ValueError(f"Expected time_scale=0.5, got {model_args.time_scale}")

    pieces: list[np.ndarray] = []
    for start in range(0, len(source_norm), batch_size):
        stop = min(start + batch_size, len(source_norm))
        values = torch.as_tensor(
            source_norm[start:stop] * scale,
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            for lower, upper in intervals:
                zero = torch.zeros((len(values), 1), dtype=values.dtype, device=device)
                values, _ = model(
                    values,
                    zero,
                    integration_times=torch.tensor(
                        [lower, upper], dtype=torch.float32, device=device
                    ),
                    reverse=False,
                )
        pieces.append((values / scale).detach().cpu().numpy())
    return np.concatenate(pieces).astype(np.float32, copy=False)


def _configure_plot_style() -> None:
    apply_nature_rc(font_size=10.0)
    mpl.rcParams.update(
        {
            "axes.titlesize": 12.0,
            "axes.labelsize": 12.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.0,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "figure.titleweight": "normal",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
        }
    )


def _plot_stage(
    scores: pd.DataFrame,
    floor: pd.DataFrame,
    *,
    stage: str,
    stage_tag: str,
    output_dir: Path,
    overwrite: bool,
) -> None:
    stem = f"gastrulation_full_train_fit_{stage_tag}_six_metrics_with_floor"
    outputs = output_paths(output_dir, stem)
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs; pass --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )

    fig, axes = plt.subplots(3, 2, figsize=(3.45, 4.05), facecolor="white")
    display_rows: list[dict[str, object]] = []
    for index, (ax, metric) in enumerate(zip(axes.ravel(), METRICS)):
        display_rows.extend(
            plot_panel(
                ax,
                scores,
                floor,
                metric,
                stage=stage,
                show_ylabels=index % 2 == 0,
                exclude_displays=frozenset({"TIGON"}),
            )
        )
    fig.suptitle(
        f"Full-train fit {stage}",
        fontsize=12,
        fontweight="normal",
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.31,
        right=0.995,
        top=0.91,
        bottom=0.025,
        hspace=0.54,
        wspace=0.16,
    )
    png_path, pdf_path, svg_path, csv_path = outputs
    save_options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.02}
    fig.savefig(png_path, dpi=600, **save_options)
    fig.savefig(pdf_path, **save_options)
    fig.savefig(svg_path, **save_options)
    pd.DataFrame(display_rows).to_csv(csv_path, index=False)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    score_paths = {
        stage_tag: output_dir / f"full_train_fit_{stage_tag}_shared_t_scores.csv"
        for _, _, _, _, stage_tag, _ in STAGES
    }
    required_outputs = [
        *score_paths.values(),
        output_dir / "full_train_fit_celltype_composition.csv",
        output_dir / "sampling_floor_summary.csv",
        output_dir / "trajectorynet_terminal_predictions.npz",
        output_dir / "evaluation_manifest.json",
    ]
    existing = [path for path in required_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs; pass --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    rna_ref, atac_ref, shared_labels, rna_scale, atac_scale = _load_references()
    rna_labels, atac_labels, label_audit = _load_separate_modality_labels(shared_labels)
    classes = sorted(
        set(np.concatenate(rna_labels).tolist())
        | set(np.concatenate(atac_labels).tolist())
        | {UNKNOWN}
    )
    external_predictions, external_log_masses = _load_external_cache(
        args.external_cache
    )
    coati_balanced, coati_unbalanced, coati_unbalanced_mass = _load_coati()
    t_model = _load_t_model(device)
    sinkhorn = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=args.sinkhorn_blur,
        debias=True,
        backend="tensorized",
    )

    all_scores: list[pd.DataFrame] = []
    all_composition: list[dict[str, object]] = []
    terminal_predictions: dict[str, np.ndarray] = {}
    terminal_rna = np.asarray(rna_ref[3], dtype=np.float32)
    for (
        stage_index,
        cache_index,
        trajectory_step,
        physical_time,
        stage_tag,
        trajectorynet_intervals,
    ) in STAGES:
        stage = STAGE_NAMES[stage_index]
        print(f"[full-train fit] {stage}", flush=True)
        rna_prepared = _prepare_reference(rna_ref[stage_index], args.sinkhorn_clusters)
        atac_prepared = _prepare_reference(atac_ref[stage_index], args.sinkhorn_clusters)

        predictions: dict[str, np.ndarray] = {
            "BSOT C_y=0.3": coati_balanced[trajectory_step].numpy().astype(
                np.float32, copy=False
            ),
            "USOT C_y=0.3": coati_unbalanced[trajectory_step].numpy().astype(
                np.float32, copy=False
            ),
        }
        log_masses: dict[str, np.ndarray] = {
            "USOT C_y=0.3": coati_unbalanced_mass[trajectory_step, :, 0]
            .numpy()
            .astype(np.float32, copy=False)
        }
        for external_method in EXTERNAL_METHODS:
            if external_method == "TrajectoryNet":
                continue
            method = METHOD_NAME[external_method]
            predictions[method] = external_predictions[external_method][cache_index]
            if external_method in NATIVE_MASS_METHODS:
                log_masses[method] = external_log_masses[external_method][cache_index]
        trajectorynet_prediction = _trajectorynet_from_terminal(
            terminal_rna,
            scale=rna_scale,
            intervals=trajectorynet_intervals,
            device=device,
            batch_size=args.batch_size,
        )
        predictions["TrajectoryNet 20k"] = trajectorynet_prediction
        terminal_predictions[f"prediction_{stage_tag}"] = trajectorynet_prediction

        stage_rows: list[dict[str, object]] = []
        for method, prediction_rna in predictions.items():
            native_log_mass = log_masses.get(method)
            prediction_atac = _apply_t(
                t_model,
                prediction_rna,
                physical_time,
                device,
                args.batch_size,
            ).astype(np.float32, copy=False)
            rna_metrics, rna_composition = _evaluate_distribution(
                method,
                "RNA",
                prediction_rna,
                rna_ref[stage_index],
                rna_labels[stage_index],
                rna_prepared,
                classes,
                sinkhorn,
                native_log_mass,
                args.sinkhorn_clusters,
                stage_name=stage,
            )
            atac_metrics, atac_composition = _evaluate_distribution(
                method,
                "ATAC",
                prediction_atac,
                atac_ref[stage_index],
                atac_labels[stage_index],
                atac_prepared,
                classes,
                sinkhorn,
                native_log_mass,
                args.sinkhorn_clusters,
                stage_name=stage,
            )
            stage_rows.append(
                {
                    "method": method,
                    "stage": stage,
                    "physical_time": physical_time,
                    "training_regime": "full train; fitted time point observed",
                    "rna_prediction_origin": (
                        "real E8.75 terminal distribution"
                        if method == "TrajectoryNet 20k"
                        else "real E7.5 starting distribution"
                    ),
                    "query_weighting": (
                        "native integrated growth"
                        if method in {"USOT C_y=0.3", "TIGON 20k", "CytoBridge unbalanced 20k"}
                        else "uniform"
                    ),
                    **rna_metrics,
                    **atac_metrics,
                }
            )
            all_composition.extend(rna_composition)
            all_composition.extend(atac_composition)

        stage_frame = pd.DataFrame(stage_rows)
        stage_frame.to_csv(score_paths[stage_tag], index=False)
        all_scores.append(stage_frame)

    combined_scores = pd.concat(all_scores, ignore_index=True)
    combined_scores.to_csv(output_dir / "full_train_fit_all_scores.csv", index=False)
    pd.DataFrame(all_composition).to_csv(
        output_dir / "full_train_fit_celltype_composition.csv", index=False
    )
    np.savez_compressed(
        output_dir / "trajectorynet_terminal_predictions.npz",
        source_e875=terminal_rna,
        **terminal_predictions,
    )
    floor = pd.read_csv(args.floor_summary)
    floor.to_csv(output_dir / "sampling_floor_summary.csv", index=False)

    _configure_plot_style()
    for stage_index, _, _, _, stage_tag, _ in STAGES:
        stage = STAGE_NAMES[stage_index]
        _plot_stage(
            combined_scores[combined_scores["stage"].eq(stage)],
            floor,
            stage=stage,
            stage_tag=stage_tag,
            output_dir=output_dir,
            overwrite=args.overwrite,
        )

    manifest = {
        "task": "Gastrulation full-train in-sample distribution fitting",
        "heldout": False,
        "evaluated_stages": [STAGE_NAMES[item[0]] for item in STAGES],
        "spaces": {
            "RNA": "normalized shared RNA PCA50",
            "ATAC": "normalized shared ATAC LSI14 after frozen RNA-to-ATAC T mapping",
            "rna_scale": rna_scale,
            "atac_scale": atac_scale,
        },
        "metric_protocol": {
            "effective_coverage": "1 / sum(transferred_reference_mass^2) / n_reference",
            "w2": "sqrt(2 * 512-microcluster debiased Sinkhorn divergence), blur=0.05",
            "composition_jsd": "base-2 JSD after modality-specific 1-NN cell-type readout",
            "sampling_floor": str(args.floor_summary.resolve()),
        },
        "trajectorynet": {
            "checkpoint": str(TRAJECTORYNET_CHECKPOINT),
            "source": "real E8.75 terminal RNA cells",
            "direction": "training/density direction",
            "solver": "rk4; step_size=0.1",
            "intervals": {
                "E8.0": [[1.5, 2.0], [1.0, 1.5]],
                "E8.5": [[1.5, 2.0]],
            },
        },
        "atac_prediction": "all methods: predicted RNA mapped through the same frozen full-data T",
        "external_prediction_cache": str(args.external_cache.resolve()),
        "coati": {
            "balanced_C_y_0.3": str(COATI_BALANCED),
            "unbalanced_C_y_0.3": str(COATI_UNBALANCED),
            "unbalanced_log_mass": str(COATI_UNBALANCED_MASS),
        },
        "modality_label_audit": label_audit,
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
