#!/usr/bin/env python
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
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from compare_gastrulation_unbalanced_nmp_decomposition import (
    _rollout_cytobridge,
    _stage_arrays_from_usot,
)
from decompose_gastrulation_nmp_mass_change import (
    STAGE_TIME,
    clean_celltype,
    hard_waterfall,
    load_tensor,
    normalized_weights,
)
from evaluate_gastrulation_flow_methods_normalized import (
    _load_cytobridge_model,
)
from tigon_checkpoint_evaluation_common import (
    file_sha256,
    load_checkpoint_model,
    rollout as rollout_tigon_checkpoint,
)
from trainfbench_plot_style import METHOD_STYLES, apply_nature_rc


ROOT = Path(__file__).resolve().parents[2]
GAST_ROOT = Path("external/COATI/Gastrulation")
STAGES = ("E7.5", "E8.0", "E8.5", "E8.75")
TARGET_STAGES = STAGES[1:]
INTERVALS = tuple(zip(STAGES[:-1], STAGES[1:]))
METHODS = ("COATI unbalanced", "CytoBridge unbalanced", "TIGON")
METHOD_LABELS = {
    "COATI unbalanced": "COATI unbal.",
    "CytoBridge unbalanced": "CytoBridge unbal.",
    "TIGON": "TIGON",
}
COMPONENTS = (
    ("retained_nmp_weight_change", "Retained reweighting", "#0072B2"),
    ("incoming_final_mass", "Incoming", "#E69F00"),
    ("outgoing_plot", "Outgoing", "#CC79A7"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continuous full-data NMP mass decomposition for COATI unbalanced, "
            "unbalanced CytoBridge, and official AE10 TIGON."
        )
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "data/gastrulation_rna_cytobridge.h5ad",
    )
    parser.add_argument(
        "--scale-file",
        type=Path,
        default=GAST_ROOT / "data/primal_norm_params.pt",
    )
    parser.add_argument(
        "--coati-dir",
        type=Path,
        default=GAST_ROOT / "UnbalancedSync_biological_num/trajectory_hpc_iter20000",
    )
    parser.add_argument(
        "--cytobridge-adata",
        type=Path,
        default=(
            ROOT
            / "results/cytobridge_gastrulation_rna_20000_unbalanced/adata.h5ad"
        ),
    )
    parser.add_argument(
        "--tigon-checkpoint",
        type=Path,
        default=(
            ROOT
            / "results/tigon_gastrulation_full_upstream_public_exact_v2_rngexact_"
            "ae10_dopri5pack_n1024_20000_seed1/tigon_checkpoint_iter020000.pt"
        ),
    )
    parser.add_argument(
        "--tigon-ae-cache",
        type=Path,
        default=(
            ROOT
            / "results/tigon_official_ae_embeddings/"
            "gastrulation_ae10_upstream_public_exact_v2"
        ),
    )
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--cytobridge-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results/gastrulation_full_nmp_mass_decomposition_three_unbalanced_"
            "tigon_upstream_exact_v2_iter20000"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class NativeStageClassifier:
    def __init__(
        self,
        coordinates: np.ndarray,
        stages: np.ndarray,
        labels: np.ndarray,
        neighbors: int,
    ):
        self.models: dict[str, tuple[NearestNeighbors, np.ndarray]] = {}
        for stage in STAGES:
            index = np.flatnonzero(stages == stage)
            model = NearestNeighbors(
                n_neighbors=min(neighbors, len(index)), n_jobs=-1
            ).fit(coordinates[index])
            self.models[stage] = (model, labels[index])

    def label(self, points: np.ndarray, stage: str) -> np.ndarray:
        model, labels = self.models[stage]
        distances, local = model.kneighbors(points, return_distance=True)
        neighbor_labels = labels[local]
        positive = distances[distances > 0]
        floor = float(np.median(positive)) * 1e-3 if positive.size else 1e-8
        weights = 1.0 / np.maximum(distances, max(floor, 1e-8))
        hard: list[str] = []
        for row_labels, row_weights in zip(neighbor_labels, weights):
            scores: dict[str, float] = {}
            for label, weight in zip(row_labels, row_weights):
                scores[str(label)] = scores.get(str(label), 0.0) + float(weight)
            hard.append(max(scores.items(), key=lambda item: item[1])[0])
        return np.asarray(hard, dtype=object)


def rollout_official_ae_tigon(
    x0: np.ndarray,
    model,
    saved_args: dict,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    positions, log_masses = rollout_tigon_checkpoint(
        model=model,
        saved_args=saved_args,
        source=np.asarray(x0, dtype=np.float32),
        evaluation_times=np.asarray(
            [STAGE_TIME[stage] for stage in STAGES], dtype=np.float32
        ),
        device=device,
        batch_size=batch_size,
    )
    return np.stack(positions), np.stack(log_masses)


def evaluate_rollout(
    method: str,
    c_y: float | None,
    positions: np.ndarray,
    log_masses: np.ndarray,
    classifier: NativeStageClassifier,
) -> list[dict[str, object]]:
    labels = {
        stage: classifier.label(positions[index], stage)
        for index, stage in enumerate(STAGES)
    }
    weights = {
        stage: normalized_weights(log_masses[index])
        for index, stage in enumerate(STAGES)
    }
    rows: list[dict[str, object]] = []
    for stage0, stage1 in INTERVALS:
        hard, _ = hard_waterfall(
            weights[stage0],
            weights[stage1],
            labels[stage0],
            labels[stage1],
        )
        start_share = float(weights[stage0][labels[stage0] == "NMP"].sum())
        end_share = float(weights[stage1][labels[stage1] == "NMP"].sum())
        rows.append(
            {
                "method": method,
                "c_y": c_y,
                "interval": f"{stage0}→{stage1}",
                "source_stage": stage0,
                "target_stage": stage1,
                "nmp_share_start": start_share,
                "nmp_share_end": end_share,
                "delta_nmp_share": end_share - start_share,
                **hard,
                "closure_error": (
                    end_share - start_share - float(hard["waterfall_sum"])
                ),
            }
        )
    return rows


def observed_shares(reference: ad.AnnData) -> pd.DataFrame:
    stages = reference.obs["stage"].astype(str).to_numpy()
    labels = clean_celltype(reference.obs["celltype"])
    rows = []
    for stage in STAGES:
        local = stages == stage
        rows.append(
            {
                "stage": stage,
                "observed_nmp_share": float(np.mean(labels[local] == "NMP")),
                "n_cells": int(local.sum()),
                "n_nmp": int(np.sum(labels[local] == "NMP")),
            }
        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = (
        "nmp_share_start",
        "nmp_share_end",
        "delta_nmp_share",
        "retained_nmp_weight_change",
        "incoming_final_mass",
        "outgoing_initial_mass",
    )
    for (method, interval), local in frame.groupby(
        ["method", "interval"], sort=False
    ):
        row: dict[str, object] = {
            "method": method,
            "interval": interval,
            "source_stage": local["source_stage"].iloc[0],
            "target_stage": local["target_stage"].iloc[0],
            "n_settings": int(len(local)),
        }
        for metric in metrics:
            values = local[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows)


def apply_plot_style() -> None:
    apply_nature_rc(font_size=10.0)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "font.weight": "normal",
            "axes.titlesize": 12,
            "axes.titleweight": "normal",
            "axes.labelsize": 12,
            "axes.labelweight": "normal",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_decomposition(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    observed: pd.DataFrame,
    output_dir: Path,
) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 5.40), sharey=False)
    y = np.arange(len(METHODS))
    observed_lookup = observed.set_index("stage")["observed_nmp_share"]
    for ax, (stage0, stage1) in zip(axes, INTERVALS):
        interval = f"{stage0}→{stage1}"
        local = summary[summary["interval"] == interval].set_index("method").loc[
            list(METHODS)
        ]
        positive = np.zeros(len(METHODS))
        negative = np.zeros(len(METHODS))
        plot_values = {
            "retained_nmp_weight_change": local[
                "retained_nmp_weight_change_mean"
            ].to_numpy(float)
            * 100.0,
            "incoming_final_mass": local["incoming_final_mass_mean"].to_numpy(float)
            * 100.0,
            "outgoing_plot": -local["outgoing_initial_mass_mean"].to_numpy(float)
            * 100.0,
        }
        for column, label, color in COMPONENTS:
            values = plot_values[column]
            left = np.where(values >= 0, positive, negative)
            ax.barh(
                y,
                values,
                left=left,
                height=0.26,
                color=color,
                label=label,
                zorder=2,
            )
            positive += np.where(values >= 0, values, 0.0)
            negative += np.where(values < 0, values, 0.0)
        net = local["delta_nmp_share_mean"].to_numpy(float) * 100.0
        ax.scatter(net, y, marker="D", s=34, color="#222222", zorder=4)
        for value, index in zip(net, y):
            offsets = ((5, 0, "left"), (-5, 0, "right"), (0, -10, "center"))
            x_offset, y_offset, horizontal_alignment = offsets[int(index)]
            ax.annotate(
                f"{value:+.2f}",
                (value, index),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha=horizontal_alignment,
                va="center",
                fontsize=10,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "pad": 0.3,
                },
            )
        observed_delta = (
            float(observed_lookup.loc[stage1])
            - float(observed_lookup.loc[stage0])
        ) * 100.0
        ax.axvline(
            observed_delta,
            color="#222222",
            linestyle="--",
            linewidth=1.15,
            zorder=1,
        )
        ax.axvline(0.0, color="#555555", linewidth=0.8, zorder=1)
        ax.set_title(
            interval,
            loc="left",
            # Align with the complete subplot, including the method labels,
            # rather than only with the numerical plotting area.
            x=-0.67,
            ha="left",
            fontsize=12,
            fontfamily="Arial",
            fontweight="normal",
            y=1.14,
            pad=0,
        )
        ax.text(
            1.0,
            1.025,
            f"Observed: {observed_delta:+.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            fontfamily="Arial",
            fontweight="normal",
        )
        ax.set_yticks(
            y,
            [METHOD_LABELS[method] for method in METHODS],
            fontsize=10,
            fontweight="normal",
        )
        ax.invert_yaxis()
        ax.set_ylim(len(METHODS) - 0.40, -0.72)
        ax.grid(axis="x", color="#E1E1E1", linewidth=0.7, zorder=0)
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min, x_max + 0.12 * (x_max - x_min))
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=10)
        ax.tick_params(axis="x", labelsize=10)
    fig.supxlabel(
        "Δ NMP proportion (percentage points)",
        x=0.62,
        y=0.135,
        fontsize=12,
        fontweight="normal",
    )
    handles = [Patch(facecolor=color, label=label) for _, label, color in COMPONENTS]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor="#222222",
                markersize=6,
                label="Predicted net change",
            ),
            Line2D(
                [0],
                [0],
                color="#222222",
                linestyle="--",
                linewidth=1.15,
                label="Observed",
            ),
        ]
    )
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.60, 0.004),
        handlelength=1.0,
        columnspacing=0.65,
        handletextpad=0.35,
        prop={"family": "Arial", "size": 10, "weight": "normal"},
    )
    fig.subplots_adjust(
        left=0.44,
        right=0.995,
        top=0.975,
        bottom=0.245,
        hspace=0.78,
    )
    for stem in (
        "nmp_full_mass_change_decomposition",
        "nmp_full_mass_change_decomposition_vertical",
    ):
        for suffix in ("png", "pdf"):
            fig.savefig(
                output_dir / f"{stem}.{suffix}",
                dpi=400 if suffix == "png" else None,
                bbox_inches="tight",
                pad_inches=0.01,
                facecolor="white",
            )
    plt.close(fig)


def plot_endpoint_shares(
    frame: pd.DataFrame,
    observed: pd.DataFrame,
    output_dir: Path,
) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.35))
    x = np.arange(len(TARGET_STAGES), dtype=float)
    observed_values = (
        observed.set_index("stage").loc[list(TARGET_STAGES), "observed_nmp_share"]
        .to_numpy(float)
        * 100.0
    )
    observed_style = METHOD_STYLES["Observed / real"]
    ax.plot(
        x,
        observed_values,
        color=observed_style.color,
        marker=observed_style.marker,
        linewidth=observed_style.linewidth,
        markersize=observed_style.markersize,
        label="Observed",
        zorder=4,
    )
    for method in METHODS:
        local = frame[frame["method"] == method]
        pivot = local.pivot_table(
            index="c_y",
            columns="target_stage",
            values="nmp_share_end",
            aggfunc="first",
            dropna=False,
        )
        if method == "COATI unbalanced":
            values = np.stack(
                [
                    local[local["target_stage"] == stage][
                        "nmp_share_end"
                    ].to_numpy(float)
                    for stage in TARGET_STAGES
                ],
                axis=1,
            )
            mean = np.mean(values, axis=0) * 100.0
            lower = np.min(values, axis=0) * 100.0
            upper = np.max(values, axis=0) * 100.0
            style = METHOD_STYLES[method]
            ax.fill_between(
                x, lower, upper, color=style.color, alpha=0.18, linewidth=0
            )
            ax.plot(
                x,
                mean,
                color=style.color,
                marker=style.marker,
                linewidth=style.linewidth,
                markersize=style.markersize,
                label="COATI unbal.",
                zorder=3,
            )
        else:
            values = np.asarray(
                [
                    local[local["target_stage"] == stage][
                        "nmp_share_end"
                    ].iloc[0]
                    for stage in TARGET_STAGES
                ]
            ) * 100.0
            style = METHOD_STYLES[method]
            ax.plot(
                x,
                values,
                color=style.color,
                marker=style.marker,
                linewidth=style.linewidth,
                markersize=style.markersize,
                label=METHOD_LABELS[method],
                zorder=3,
            )
    ax.set_xticks(x, TARGET_STAGES)
    ax.set_ylabel("NMP relative mass (%)")
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.54, -0.01),
        columnspacing=0.9,
        handlelength=1.5,
    )
    fig.subplots_adjust(left=0.17, right=0.99, top=0.98, bottom=0.32)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"nmp_full_endpoint_mass_share.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_table = args.output_dir / "nmp_full_mass_decomposition_per_setting.csv"
    if output_table.exists() and not args.overwrite:
        raise FileExistsError(f"{output_table} exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    reference = ad.read_h5ad(args.reference)
    scale = float(
        np.asarray(
            torch.load(
                args.scale_file, map_location="cpu", weights_only=False
            )["scale"]
        ).reshape(-1)[0]
    )
    reference_stages = reference.obs["stage"].astype(str).to_numpy()
    reference_labels = clean_celltype(reference.obs["celltype"])
    common_coordinates = np.asarray(reference.obsm["X_latent"], dtype=np.float32)
    common_classifier = NativeStageClassifier(
        common_coordinates,
        reference_stages,
        reference_labels,
        args.neighbors,
    )

    stage_steps = {"E7.5": 0, "E8.0": 10, "E8.5": 20, "E8.75": 25}
    expected_time = np.arange(0.0, 2.5 + 1e-8, 0.1)
    time_grid = load_tensor(
        args.coati_dir / "t_grid_s0_iter20000.pt"
    ).reshape(-1)
    if not np.allclose(time_grid, expected_time):
        raise ValueError("Unexpected COATI time grid")

    rows: list[dict[str, object]] = []
    common_x0: np.ndarray | None = None
    for c_y in np.round(np.arange(0.1, 1.0, 0.1), 1):
        trajectory = load_tensor(
            args.coati_dir
            / f"primary_trajectory_s0_a{c_y:.1f}_iter20000.pt"
        )
        log_mass = load_tensor(
            args.coati_dir
            / f"mass_lnw_trajectory_s0_a{c_y:.1f}_iter20000.pt"
        )
        positions_norm, masses = _stage_arrays_from_usot(trajectory, log_mass)
        positions = positions_norm * scale
        if common_x0 is None:
            common_x0 = positions_norm[0].copy()
        elif not np.allclose(common_x0, positions_norm[0], atol=0.0, rtol=0.0):
            raise ValueError(f"COATI C_y={c_y:.1f} source particles differ")
        rows.extend(
            evaluate_rollout(
                "COATI unbalanced",
                float(c_y),
                positions,
                masses,
                common_classifier,
            )
        )
    assert common_x0 is not None

    source_index = np.flatnonzero(reference_stages == "E7.5")
    if not np.allclose(
        common_x0 * scale,
        common_coordinates[source_index],
        atol=5e-6,
        rtol=1e-6,
    ):
        raise ValueError("COATI source does not match the full E7.5 reference")

    cytobridge = _load_cytobridge_model(args.cytobridge_adata, device)
    cb_positions_norm, cb_masses = _rollout_cytobridge(
        common_x0,
        scale,
        cytobridge,
        steps=args.cytobridge_steps,
        batch_size=args.batch_size,
        device=device,
    )
    rows.extend(
        evaluate_rollout(
            "CytoBridge unbalanced",
            None,
            cb_positions_norm * scale,
            cb_masses,
            common_classifier,
        )
    )

    tigon_config_path = args.tigon_checkpoint.parent / "config.json"
    tigon_config = json.loads(tigon_config_path.read_text())
    configured_ae_cache = Path(str(tigon_config["frozen_full_ae_cache"]))
    if configured_ae_cache.resolve() != args.tigon_ae_cache.resolve():
        raise ValueError(
            "TIGON checkpoint/config requires a different native AE cache: "
            f"{configured_ae_cache}"
        )
    latent_path = args.tigon_ae_cache / "ae_latent_scaled_minus2_2.npy"
    latent_sha256 = file_sha256(latent_path)
    expected_latent_sha256 = str(tigon_config["frozen_latent_sha256"])
    if latent_sha256 != expected_latent_sha256:
        raise ValueError(
            "TIGON AE latent/checkpoint mismatch: "
            f"{latent_sha256} != {expected_latent_sha256}"
        )
    ae_latent = np.load(latent_path).astype(np.float32, copy=False)
    ae_obs_names = pd.Index(
        np.load(args.tigon_ae_cache / "obs_names.npy", allow_pickle=False).astype(str)
    )
    indexer = ae_obs_names.get_indexer(reference.obs_names.astype(str))
    if np.any(indexer < 0):
        raise ValueError("TIGON AE cache is missing benchmark cells")
    ae_latent = ae_latent[indexer]
    ae_classifier = NativeStageClassifier(
        ae_latent,
        reference_stages,
        reference_labels,
        args.neighbors,
    )
    tigon, tigon_args, tigon_state = load_checkpoint_model(
        args.tigon_checkpoint, device
    )
    if int(tigon_state.get("iter", -1)) != 20_000:
        raise ValueError(
            f"Expected TIGON iter=20000, found {tigon_state.get('iter')}"
        )
    tigon_positions, tigon_masses = rollout_official_ae_tigon(
        ae_latent[source_index],
        tigon,
        tigon_args,
        device=device,
        batch_size=args.batch_size,
    )
    rows.extend(
        evaluate_rollout(
            "TIGON",
            None,
            tigon_positions,
            tigon_masses,
            ae_classifier,
        )
    )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_table, index=False)
    summary = summarize(frame)
    summary.to_csv(
        args.output_dir / "nmp_full_mass_decomposition_summary.csv", index=False
    )
    observed = observed_shares(reference)
    observed.to_csv(args.output_dir / "observed_nmp_mass_share.csv", index=False)
    observed_lookup = observed.set_index("stage")["observed_nmp_share"]
    endpoint_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    for method, local in frame.groupby("method", sort=False):
        for stage in TARGET_STAGES:
            values = local.loc[
                local["target_stage"] == stage, "nmp_share_end"
            ].to_numpy(float)
            predicted = float(np.mean(values))
            truth = float(observed_lookup.loc[stage])
            endpoint_rows.append(
                {
                    "method": method,
                    "stage": stage,
                    "predicted_nmp_share": predicted,
                    "predicted_nmp_share_min": float(np.min(values)),
                    "predicted_nmp_share_max": float(np.max(values)),
                    "observed_nmp_share": truth,
                    "error": predicted - truth,
                    "absolute_error": abs(predicted - truth),
                }
            )
        for stage0, stage1 in INTERVALS:
            interval = f"{stage0}→{stage1}"
            values = local.loc[
                local["interval"] == interval, "delta_nmp_share"
            ].to_numpy(float)
            predicted = float(np.mean(values))
            truth = float(
                observed_lookup.loc[stage1] - observed_lookup.loc[stage0]
            )
            interval_rows.append(
                {
                    "method": method,
                    "interval": interval,
                    "predicted_delta_nmp_share": predicted,
                    "predicted_delta_min": float(np.min(values)),
                    "predicted_delta_max": float(np.max(values)),
                    "observed_delta_nmp_share": truth,
                    "error": predicted - truth,
                    "absolute_error": abs(predicted - truth),
                }
            )
    pd.DataFrame(endpoint_rows).to_csv(
        args.output_dir / "nmp_full_endpoint_comparison.csv", index=False
    )
    pd.DataFrame(interval_rows).to_csv(
        args.output_dir / "nmp_full_interval_change_comparison.csv", index=False
    )
    plot_decomposition(frame, summary, observed, args.output_dir)
    plot_endpoint_shares(frame, observed, args.output_dir)

    max_closure = float(np.max(np.abs(frame["closure_error"].to_numpy(float))))
    manifest = {
        "analysis": "Continuous full-data NMP mass-change decomposition",
        "methods": list(METHODS),
        "source_particles": int(len(source_index)),
        "source": "same complete real E7.5 cohort in each method's native preprocessing space",
        "intervals": [f"{a}→{b}" for a, b in INTERVALS],
        "no_interval_restart": True,
        "mass_weighting": "native particle mass normalized within method at each stage",
        "classifier": (
            f"stage-conditioned distance-weighted kNN hard label, k={args.neighbors}; "
            "COATI/CytoBridge in shared 50D RNA PCA and TIGON in official frozen AE10"
        ),
        "coati_summary": "mean and min-max across C_y=0.1,...,0.9",
        "identity": (
            "Delta NMP share = retained-NMP reweighting + incoming final mass "
            "- outgoing initial mass"
        ),
        "max_absolute_closure_error": max_closure,
        "tigon": {
            "checkpoint": str(args.tigon_checkpoint),
            "checkpoint_sha256": file_sha256(args.tigon_checkpoint),
            "checkpoint_iteration": int(tigon_state["iter"]),
            "ae_cache": str(args.tigon_ae_cache),
            "ae_latent": str(latent_path),
            "ae_latent_sha256": latent_sha256,
            "training_status": tigon_config.get("status"),
            "latent_dimension": 10,
            "preprocessing": (
                "official frozen 3000-HVG expression AE; scaled_minus2_2; "
                "no post-AE standardization"
            ),
            "comparison_space": (
                "TIGON predictions and stage-specific kNN references are both "
                "in the same native frozen AE10 coordinate system"
            ),
            "ode_backend": tigon_args.get("density_ode_backend"),
            "ode_solver": tigon_args.get("ode_solver"),
        },
        "limitation": (
            "Retained reweighting is an inferred relative mass source/sink; "
            "it is not a direct assay of proliferation or death."
        ),
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"max closure error: {max_closure:.3e}")
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
