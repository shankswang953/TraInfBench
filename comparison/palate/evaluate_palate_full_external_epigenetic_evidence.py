#!/usr/bin/env python
"""Evaluate full-train palate trajectories on fixed external ATAC peak sets.

This is the training-stage counterpart of
``evaluate_palate_loo_external_epigenetic_evidence.py``.  At E13.5 and E14.0,
each method's predicted RNA particles are mapped to ATAC with the same frozen
full-data FiLM map.  A common stage-restricted 5NN readout assigns RNA branch
identity and decodes raw binary peak accessibility.  The reported statistic is
the across-peak Spearman correlation between predicted and observed
anterior-minus-posterior accessibility contrasts.

Unlike the strict-LOO analysis, the evaluated stage is present in the kNN
reference and in model training.  These numbers therefore measure full-train
reconstruction, not held-out interpolation.
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
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch

from analyze_palate_cnc_rna_atac_fate_concordance import rollout_mioflow_common
from analyze_palate_shox2_1nn_regulation import (
    ANTERIOR,
    ATAC_NORM,
    ATAC_RAW,
    ATAC_REFERENCE,
    POSTERIOR,
    RNA_NORM,
    RNA_REFERENCE,
)
from evaluate_palate_loo_external_epigenetic_evidence import (
    family_metrics,
    majority_branch,
    observed_binary_profile,
    weighted_binary_profile,
)
from evaluate_palate_loo_same_space import _load_film, _load_scale, _map_to_atac
from recalculate_palate_loo_trajectorynet_from_base import generate_native_marginal
from trainfbench_plot_style import apply_nature_rc, method_style


ROOT = Path(__file__).resolve().parents[2]
PALATE = Path("external/COATI/MouseBrain")
ARCHIVE = ROOT / "results/palate_trajectory_archive"
COMMON_INITIAL = ARCHIVE / "manifest/common_initial_indices.npy"
COMMON_X0 = ARCHIVE / "manifest/common_initial_rna_norm.npy"
CB_U_WEIGHTS = (
    ROOT
    / "results/palate_ap_fate_separation"
    / "cytobridge_unbalanced_common_initial_weights.npz"
)
PEAK_ANNOTATION = (
    ROOT
    / "results/palate_strict_loo_external_epigenetic_evidence_with_rna_only"
    / "external_peak_annotations_with_meox2.csv.gz"
)
TRAJECTORYNET_CHECKPOINT = (
    ROOT / "results/trajectorynet_palate_rna_20000/checkpt-20000.pth"
)
DEFAULT_OUTPUT = ROOT / "results/palate_full_external_epigenetic_evidence"

STAGES = (("E13.5", 1.0, 1.0), ("E14.0", 1.5, 2.0))
METHOD_ORDER = (
    "COATI-B",
    "COATI-U",
    "CytoBridge-B",
    "CytoBridge-U",
    "MIOFlow",
    "TrajectoryNet",
    "RNAonly-B",
    "RNAonly-U",
)
METHOD_LABELS = {
    "COATI-B": "COATI bal.",
    "COATI-U": "COATI unbal.",
    "CytoBridge-B": "CytoBridge bal.",
    "CytoBridge-U": "CytoBridge unbal.",
    "MIOFlow": "MIOFlow",
    "TrajectoryNet": "TrajectoryNet",
    "RNAonly-B": "OT(RNA)",
    "RNAonly-U": "UOT(RNA)",
}
STYLE_NAMES = {
    "COATI-B": "COATI balanced",
    "COATI-U": "COATI unbalanced",
    "CytoBridge-B": "CytoBridge balanced",
    "CytoBridge-U": "CytoBridge unbalanced",
    "MIOFlow": "MIOFlow",
    "TrajectoryNet": "TrajectoryNet",
    "RNAonly-B": "Balanced RNA-only",
    "RNAonly-U": "Unbalanced RNA-only",
}


@dataclass
class MethodTrajectory:
    rna: np.ndarray
    weights: np.ndarray
    source: str
    cohort: str = "common observed E12.5 particles"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--trajectorynet-particles", type=int, default=2048)
    parser.add_argument("--trajectorynet-seed", type=int, default=20260727)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_tensor(path: Path) -> np.ndarray:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 3 or not np.isfinite(result).all():
        raise ValueError(f"Invalid trajectory tensor {path}: {result.shape}")
    return result


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as saved:
        return {key: np.asarray(saved[key]) for key in saved.files}


def normalize_log_mass(log_mass: np.ndarray) -> np.ndarray:
    values = np.asarray(log_mass, dtype=float).reshape(-1)
    values -= np.max(values)
    weights = np.exp(np.clip(values, -80.0, 0.0))
    return weights / weights.sum()


def uniform_weights(n_stage: int, n_particles: int) -> np.ndarray:
    return np.full((n_stage, n_particles), 1.0 / n_particles, dtype=np.float64)


def tensor_method(
    path: Path,
    common: np.ndarray,
    *,
    mass_path: Path | None = None,
) -> MethodTrajectory:
    trajectory = load_tensor(path)
    positions = np.asarray([10, 15], dtype=int)
    rna = trajectory[positions][:, common]
    if mass_path is None:
        weights = uniform_weights(len(positions), len(common))
    else:
        mass = load_tensor(mass_path)[positions][:, common]
        weights = np.stack([normalize_log_mass(value) for value in mass])
    return MethodTrajectory(rna=rna, weights=weights, source=str(path))


def load_full_methods(
    common: np.ndarray,
    rna_scale: float,
    film: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    trajectorynet_particles: int,
    trajectorynet_seed: int,
) -> dict[str, MethodTrajectory]:
    methods: dict[str, MethodTrajectory] = {}
    methods["COATI-B"] = tensor_method(
        PALATE
        / "FilmBalanced/trajectory/iter20000/primary_trajectory_a0.5_iter20000.pt",
        common,
    )
    methods["COATI-U"] = tensor_method(
        PALATE
        / "FilmUnbalancedSync/trajectory/primary_trajectory_filmunbalanced_s0_a0.5_iter20000.pt",
        common,
        mass_path=PALATE
        / "FilmUnbalancedSync/trajectory/mass_lnw_filmunbalanced_s0_a0.5_iter20000.pt",
    )
    methods["RNAonly-B"] = tensor_method(
        PALATE
        / "Balanced/trajectory/primary_trajectory_balanced_rnaonly_s0_iter20000.pt",
        common,
    )
    methods["RNAonly-U"] = tensor_method(
        PALATE
        / "UnbalancedRNAOnly/trajectory/primary_trajectory_unbalanced_rnaonly_s0_iter20000.pt",
        common,
        mass_path=PALATE
        / "UnbalancedRNAOnly/trajectory/mass_lnw_trajectory_unbalanced_rnaonly_s0_iter20000.pt",
    )

    cb_weights = load_npz(CB_U_WEIGHTS)
    for method, filename in (
        ("CytoBridge-B", "cytobridge_balanced.npz"),
        ("CytoBridge-U", "cytobridge_unbalanced.npz"),
    ):
        path = ARCHIVE / "trajectories/external" / filename
        saved = load_npz(path)
        if not np.array_equal(saved["initial_indices"], common):
            raise ValueError(f"{method} does not use the common initial indices")
        positions = np.asarray(
            [int(np.argmin(np.abs(saved["time"] - time))) for _, time, _ in STAGES]
        )
        rna = np.asarray(saved["rna_norm"][positions], dtype=np.float32)
        if method == "CytoBridge-U":
            stage_positions = np.asarray(
                [
                    int(np.argmin(np.abs(cb_weights["stage_time"] - time)))
                    for _, time, _ in STAGES
                ]
            )
            weights = np.asarray(cb_weights["weights"][stage_positions], dtype=float)
            weights /= weights.sum(axis=1, keepdims=True)
        else:
            weights = uniform_weights(len(STAGES), len(common))
        methods[method] = MethodTrajectory(rna=rna, weights=weights, source=str(path))

    mioflow = rollout_mioflow_common(
        np.load(COMMON_X0).astype(np.float32),
        rna_scale,
        film,
        device,
        batch_size,
    )
    positions = np.asarray(
        [int(np.argmin(np.abs(mioflow["time"] - time))) for _, time, _ in STAGES]
    )
    methods["MIOFlow"] = MethodTrajectory(
        rna=np.asarray(mioflow["rna_norm"][positions], dtype=np.float32),
        weights=uniform_weights(len(STAGES), len(common)),
        source=str(np.asarray(mioflow["source"]).item()),
    )

    trajectorynet_states = []
    clocks: list[dict[str, object]] = []
    for _, _, rank in STAGES:
        raw, clock = generate_native_marginal(
            TRAJECTORYNET_CHECKPOINT,
            n_particles=trajectorynet_particles,
            dimension=40,
            target_rank=rank,
            seed=trajectorynet_seed,
            device=device,
            batch_size=batch_size,
            steps_per_interval=20,
        )
        trajectorynet_states.append(raw / rna_scale)
        clocks.append(clock)
    methods["TrajectoryNet"] = MethodTrajectory(
        rna=np.asarray(trajectorynet_states, dtype=np.float32),
        weights=uniform_weights(len(STAGES), trajectorynet_particles),
        source=str(TRAJECTORYNET_CHECKPOINT),
        cohort="native standard-normal base",
    )
    methods["TrajectoryNet"].clocks = clocks  # type: ignore[attr-defined]
    return methods


def plot_table(metrics: pd.DataFrame, output_dir: Path) -> None:
    columns = (
        ("E13.5", "H3K27ac", "E13.5\nH3K27ac"),
        ("E14.0", "H3K27ac", "E14.0\nH3K27ac"),
        ("E13.5", "TF-bound active CRE", "E13.5\nTF CRE"),
        ("E14.0", "TF-bound active CRE", "E14.0\nTF CRE"),
    )
    values = np.empty((len(METHOD_ORDER), len(columns)), dtype=float)
    for row_index, method in enumerate(METHOD_ORDER):
        for column_index, (stage, family, _) in enumerate(columns):
            selected = metrics[
                metrics["method"].eq(method)
                & metrics["stage"].eq(stage)
                & metrics["family"].eq(family)
            ]
            if len(selected) != 1:
                raise ValueError(f"Missing plot value for {method}, {stage}, {family}")
            values[row_index, column_index] = float(
                selected.iloc[0]["peak_contrast_spearman"]
            )

    apply_nature_rc(font_size=10)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
        }
    )
    figure = plt.figure(figsize=(3.75, 2.35), facecolor="white")
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(
        0.5,
        0.970,
        "External regulatory peak contrast recovery",
        ha="center",
        va="top",
        color="#222222",
        fontweight="normal",
    )
    axis.text(
        0.5,
        0.875,
        r"Spearman $\rho$(predicted, observed) $\uparrow$",
        ha="center",
        va="top",
        color="#222222",
        fontweight="normal",
    )
    metric_centers = np.asarray([0.420, 0.595, 0.770, 0.945])
    for center, (_, _, label) in zip(metric_centers, columns):
        axis.text(
            center,
            0.725,
            label,
            ha="center",
            va="center",
            color="#222222",
            fontweight="normal",
            linespacing=0.95,
        )
    axis.plot([0.005, 0.995], [0.620, 0.620], color="#777777", lw=0.65)

    row_centers = np.linspace(0.555, 0.105, len(METHOD_ORDER))
    for row_index, (center_y, method) in enumerate(zip(row_centers, METHOD_ORDER)):
        if row_index % 2 == 0:
            axis.add_patch(
                Rectangle(
                    (0.005, center_y - 0.028),
                    0.99,
                    0.056,
                    facecolor="#F5F5F5",
                    edgecolor="none",
                )
            )
        style = method_style(STYLE_NAMES[method])
        axis.add_patch(
            Rectangle(
                (0.012, center_y - 0.016),
                0.026,
                0.032,
                facecolor=style.color,
                edgecolor=style.color,
                linewidth=0.6,
            )
        )
        axis.text(
            0.050,
            center_y,
            METHOD_LABELS[method],
            ha="left",
            va="center",
            color="#222222",
            fontweight="normal",
        )
        for center_x, value in zip(metric_centers, values[row_index]):
            axis.text(
                center_x,
                center_y,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="#222222",
                fontweight="normal",
            )
    stem = output_dir / "palate_full_external_peak_contrast_spearman_table"
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        kwargs: dict[str, object] = {"bbox_inches": "tight", "pad_inches": 0.0}
        if suffix == "png":
            kwargs["dpi"] = 450
        figure.savefig(path, **kwargs)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    metrics_path = output_dir / "full_external_epigenetic_metrics.csv"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {metrics_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    rna = ad.read_h5ad(RNA_REFERENCE, backed="r")
    atac = ad.read_h5ad(ATAC_REFERENCE, backed="r")
    raw_atac = ad.read_h5ad(ATAC_RAW, backed="r")
    if not np.array_equal(rna.obs_names, atac.obs_names) or not np.array_equal(
        rna.obs_names, raw_atac.obs_names
    ):
        raise ValueError("Paired RNA, ATAC, and raw ATAC cells are not aligned")
    labels = rna.obs["celltype_sub"].astype(str).to_numpy()
    atlas_times = pd.to_numeric(
        rna.obs["time_point_processed"], errors="raise"
    ).to_numpy(float)
    rna_scale = _load_scale(RNA_NORM)
    rna_norm = np.asarray(rna.obsm["X_latent"], dtype=np.float32) / rna_scale
    atac_norm = np.asarray(atac.obsm["X_lsi15"], dtype=np.float32) / _load_scale(
        ATAC_NORM
    )

    peaks = pd.read_csv(
        PEAK_ANNOTATION,
        usecols=(
            "peak",
            "h3_anterior_specific",
            "h3_posterior_specific",
            "shox2_bound_anterior_active",
            "meox2_bound_posterior_active",
        ),
    )
    if not np.array_equal(peaks["peak"].astype(str), raw_atac.var_names.astype(str)):
        raise ValueError("External peak table does not align to raw ATAC")
    masks = {
        "H3K27ac": (
            peaks["h3_anterior_specific"].astype(bool).to_numpy(),
            peaks["h3_posterior_specific"].astype(bool).to_numpy(),
        ),
        "TF-bound active CRE": (
            peaks["shox2_bound_anterior_active"].astype(bool).to_numpy(),
            peaks["meox2_bound_posterior_active"].astype(bool).to_numpy(),
        ),
    }
    original_positions = np.unique(
        np.concatenate([np.flatnonzero(mask) for pair in masks.values() for mask in pair])
    )
    lookup = np.full(len(peaks), -1, dtype=int)
    lookup[original_positions] = np.arange(len(original_positions))
    families = {
        family: (lookup[np.flatnonzero(anterior)], lookup[np.flatnonzero(posterior)])
        for family, (anterior, posterior) in masks.items()
    }

    film, film_source, film_checkpoint, _ = _load_film(device)
    common = np.load(COMMON_INITIAL).astype(int)
    methods = load_full_methods(
        common,
        rna_scale,
        film,
        device,
        args.batch_size,
        args.trajectorynet_particles,
        args.trajectorynet_seed,
    )
    if set(methods) != set(METHOD_ORDER):
        raise ValueError(f"Unexpected method inventory: {sorted(methods)}")

    metric_rows: list[dict[str, object]] = []
    classifier_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    for stage_index, (stage, stage_time, _) in enumerate(STAGES):
        stage_rows = np.flatnonzero(np.isclose(atlas_times, stage_time))
        rna_nn = NearestNeighbors(n_neighbors=args.knn_k, n_jobs=-1).fit(
            rna_norm[stage_rows]
        )
        atac_nn = NearestNeighbors(n_neighbors=args.knn_k, n_jobs=-1).fit(
            atac_norm[stage_rows]
        )
        observed_anterior = observed_binary_profile(
            raw_atac,
            stage_rows[labels[stage_rows] == ANTERIOR],
            original_positions,
        )
        observed_posterior = observed_binary_profile(
            raw_atac,
            stage_rows[labels[stage_rows] == POSTERIOR],
            original_positions,
        )
        for method in METHOD_ORDER:
            trajectory = methods[method]
            predicted_rna = trajectory.rna[stage_index]
            predicted_atac = _map_to_atac(
                film, predicted_rna, stage_time, device, args.batch_size
            )
            weights = trajectory.weights[stage_index]
            rna_neighbors = stage_rows[
                rna_nn.kneighbors(predicted_rna, return_distance=False)
            ]
            predicted_branch = majority_branch(labels[rna_neighbors])
            atac_neighbors = stage_rows[
                atac_nn.kneighbors(predicted_atac, return_distance=False)
            ]
            anterior_mask = predicted_branch == "anterior"
            posterior_mask = predicted_branch == "posterior"
            classifier_rows.append(
                {
                    "stage": stage,
                    "method": method,
                    "n_particles": len(predicted_branch),
                    "n_anterior": int(anterior_mask.sum()),
                    "n_posterior": int(posterior_mask.sum()),
                    "n_unassigned": int(np.sum(predicted_branch == "unassigned")),
                    "effective_sample_size": float(1.0 / np.sum(weights**2)),
                }
            )
            predicted_anterior = weighted_binary_profile(
                raw_atac,
                atac_neighbors,
                anterior_mask,
                weights,
                original_positions,
            )
            predicted_posterior = weighted_binary_profile(
                raw_atac,
                atac_neighbors,
                posterior_mask,
                weights,
                original_positions,
            )
            for family, (anterior_positions, posterior_positions) in families.items():
                metric_rows.append(
                    {
                        "stage": stage,
                        "time": stage_time,
                        "method": method,
                        "family": family,
                        **family_metrics(
                            predicted_anterior,
                            predicted_posterior,
                            observed_anterior,
                            observed_posterior,
                            anterior_positions,
                            posterior_positions,
                        ),
                    }
                )
            source_rows.append(
                {
                    "stage": stage,
                    "method": method,
                    "source": trajectory.source,
                    "cohort": trajectory.cohort,
                }
            )

    metrics = pd.DataFrame(metric_rows)
    classifier = pd.DataFrame(classifier_rows)
    sources = pd.DataFrame(source_rows)
    metrics.to_csv(metrics_path, index=False)
    classifier.to_csv(output_dir / "predicted_branch_audit.csv", index=False)
    sources.to_csv(output_dir / "method_source_audit.csv", index=False)
    plot_table(metrics, output_dir)

    manifest = {
        "analysis": "full-train external regulatory peak contrast recovery",
        "full_train": True,
        "validation_boundary": (
            "The evaluated stage is present in both model training and the stage-restricted "
            "kNN readout; this is training-stage reconstruction, not held-out interpolation."
        ),
        "stages": [stage for stage, _, _ in STAGES],
        "knn_k": args.knn_k,
        "branch_assignment": (
            "strict majority of stage-restricted RNA kNN labels; only anterior/posterior "
            "majorities are assigned"
        ),
        "peak_readout": "stage-restricted ATAC LSI15 kNN to raw binary accessibility",
        "contrast": "anterior mean accessibility minus posterior mean accessibility per peak",
        "primary_metric": "Spearman across external peaks: predicted contrast vs observed contrast",
        "peak_sets": {
            family: {
                "anterior": int(pair[0].sum()),
                "posterior": int(pair[1].sum()),
            }
            for family, pair in masks.items()
        },
        "common_initial_particles": len(common),
        "trajectorynet": {
            "particle_origin": "native standard-normal base",
            "particles": args.trajectorynet_particles,
            "seed": args.trajectorynet_seed,
            "checkpoint": str(TRAJECTORYNET_CHECKPOINT),
            "comparison_boundary": (
                "TrajectoryNet does not share the observed E12.5 particle identities; "
                "it is included as a native marginal reconstruction baseline."
            ),
        },
        "unbalanced_weighting": (
            "COATI-U and UOT(RNA) use stage-specific learned mass; CytoBridge-U uses "
            "saved stage-specific destination mass; each is renormalized within particles."
        ),
        "film_source": str(film_source),
        "film_checkpoint": str(film_checkpoint),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    rna.file.close()
    atac.file.close()
    raw_atac.file.close()

    table = metrics.pivot_table(
        index="method", columns=["stage", "family"], values="peak_contrast_spearman"
    ).reindex(METHOD_ORDER)
    print(table.to_string(float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
