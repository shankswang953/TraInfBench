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
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors


BENCH_ROOT = Path(__file__).resolve().parents[2]
GAST_ROOT = Path("external/COATI/Gastrulation")
STAGE_TIME = {"E7.5": 0.0, "E8.0": 1.0, "E8.5": 2.0, "E8.75": 2.5}
BIO_TOTAL = {"E7.5": 1.0, "E8.0": 4.0, "E8.5": 6.0, "E8.75": 11.0}
INTERVALS = [("E8.0", "E8.5"), ("E8.5", "E8.75")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose USOT NMP mass change into cell-state transport and local "
            "particle reweighting without drawing a geometric boundary."
        )
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=BENCH_ROOT / "data/gastrulation_rna_cytobridge.h5ad",
    )
    parser.add_argument(
        "--scale-file",
        type=Path,
        default=GAST_ROOT / "data/primal_norm_params.pt",
    )
    parser.add_argument(
        "--rna-trajectory",
        type=Path,
        default=GAST_ROOT
        / "UnbalancedRNAOnly_biological_num/trajectory/"
        "primary_trajectory_unbalanced_rnaonly_bionum_s0_iter20000.pt",
    )
    parser.add_argument(
        "--rna-mass",
        type=Path,
        default=GAST_ROOT
        / "UnbalancedRNAOnly_biological_num/trajectory/"
        "mass_lnw_trajectory_unbalanced_rnaonly_bionum_s0_iter20000.pt",
    )
    parser.add_argument(
        "--sync-dir",
        type=Path,
        default=GAST_ROOT / "UnbalancedSync_biological_num/trajectory_hpc_iter20000",
    )
    parser.add_argument("--neighbors", nargs="+", type=int, default=[1, 10, 20])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCH_ROOT / "results/gastrulation_nmp_mass_decomposition",
    )
    return parser.parse_args()


def clean_celltype(values: pd.Series) -> np.ndarray:
    return (
        values.astype(object)
        .where(values.notna(), "Unannotated")
        .astype(str)
        .replace("nan", "Unannotated")
        .to_numpy()
    )


def load_tensor(path: Path) -> np.ndarray:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    x = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    x = x - np.max(x)
    weight = np.exp(x)
    return weight / np.sum(weight)


class StageClassifier:
    def __init__(self, reference: ad.AnnData, neighbors: int):
        self.neighbors = neighbors
        latent = np.asarray(reference.obsm["X_latent"], dtype=np.float32)
        stages = reference.obs["stage"].astype(str).to_numpy()
        labels = clean_celltype(reference.obs["celltype"])
        self.models: dict[str, tuple[NearestNeighbors, np.ndarray]] = {}
        for stage in STAGE_TIME:
            idx = np.flatnonzero(stages == stage)
            k = min(neighbors, len(idx))
            model = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(latent[idx])
            self.models[stage] = (model, labels[idx])

    def membership_and_label(self, points: np.ndarray, stage: str) -> tuple[np.ndarray, np.ndarray]:
        model, labels = self.models[stage]
        distances, local = model.kneighbors(points, return_distance=True)
        neighbor_labels = labels[local]
        if self.neighbors == 1:
            hard = neighbor_labels[:, 0]
            return (hard == "NMP").astype(np.float64), hard
        positive = distances[distances > 0]
        floor = float(np.median(positive)) * 1e-3 if positive.size else 1e-8
        weights = 1.0 / np.maximum(distances, max(floor, 1e-8))
        nmp = np.sum(weights * (neighbor_labels == "NMP"), axis=1) / np.sum(weights, axis=1)
        hard: list[str] = []
        for row_labels, row_weights in zip(neighbor_labels, weights):
            score: dict[str, float] = {}
            for label, weight in zip(row_labels, row_weights):
                score[str(label)] = score.get(str(label), 0.0) + float(weight)
            hard.append(max(score.items(), key=lambda item: item[1])[0])
        return nmp, np.asarray(hard, dtype=object)


def soft_decomposition(
    weight0: np.ndarray,
    weight1: np.ndarray,
    probability0: np.ndarray,
    probability1: np.ndarray,
) -> dict[str, float]:
    start = float(weight0 @ probability0)
    end = float(weight1 @ probability1)
    state_transport = float(np.sum(0.5 * (weight0 + weight1) * (probability1 - probability0)))
    local_reweighting = float(np.sum(0.5 * (probability0 + probability1) * (weight1 - weight0)))
    return {
        "nmp_mass_start": start,
        "nmp_mass_end": end,
        "nmp_mass_change": end - start,
        "state_transport": state_transport,
        "local_reweighting": local_reweighting,
        "closure_error": (end - start) - state_transport - local_reweighting,
    }


def hard_waterfall(
    weight0: np.ndarray,
    weight1: np.ndarray,
    label0: np.ndarray,
    label1: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    is0 = label0 == "NMP"
    is1 = label1 == "NMP"
    retained = is0 & is1
    incoming = ~is0 & is1
    outgoing = is0 & ~is1
    result = {
        "retained_nmp_weight_change": float(np.sum(weight1[retained] - weight0[retained])),
        "incoming_final_mass": float(np.sum(weight1[incoming])),
        "outgoing_initial_mass": float(np.sum(weight0[outgoing])),
    }
    result["waterfall_sum"] = (
        result["retained_nmp_weight_change"]
        + result["incoming_final_mass"]
        - result["outgoing_initial_mass"]
    )
    transition_rows: list[dict[str, object]] = []
    for source, destination in sorted(set(zip(label0.astype(str), label1.astype(str)))):
        idx = (label0.astype(str) == source) & (label1.astype(str) == destination)
        if source != "NMP" and destination != "NMP":
            continue
        transition_rows.append(
            {
                "source": source,
                "destination": destination,
                "n_particles": int(np.sum(idx)),
                "initial_mass": float(np.sum(weight0[idx])),
                "final_mass": float(np.sum(weight1[idx])),
                "midpoint_mass": float(np.sum(0.5 * (weight0[idx] + weight1[idx]))),
            }
        )
    return result, transition_rows


def evaluate_method(
    method: str,
    c_y: float | None,
    trajectory: np.ndarray,
    log_mass: np.ndarray,
    time_grid: np.ndarray,
    scale: float,
    classifiers: dict[int, StageClassifier],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    soft_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    for stage0, stage1 in INTERVALS:
        step0 = int(np.argmin(np.abs(time_grid - STAGE_TIME[stage0])))
        step1 = int(np.argmin(np.abs(time_grid - STAGE_TIME[stage1])))
        x0 = np.asarray(trajectory[step0], dtype=np.float32) * scale
        x1 = np.asarray(trajectory[step1], dtype=np.float32) * scale
        q0 = normalized_weights(log_mass[step0])
        q1 = normalized_weights(log_mass[step1])
        weightings = {
            "relative_share": (q0, q1),
            "common_biological_total": (q0 * BIO_TOTAL[stage0], q1 * BIO_TOTAL[stage1]),
        }
        for k, classifier in classifiers.items():
            p0, label0 = classifier.membership_and_label(x0, stage0)
            p1, label1 = classifier.membership_and_label(x1, stage1)
            for weighting, (weight0, weight1) in weightings.items():
                soft = soft_decomposition(weight0, weight1, p0, p1)
                waterfall, transitions = hard_waterfall(weight0, weight1, label0, label1)
                base = {
                    "method": method,
                    "c_y": c_y,
                    "neighbors": k,
                    "interval": f"{stage0}→{stage1}",
                    "weighting": weighting,
                }
                soft_rows.append({**base, **soft, **waterfall})
                transition_rows.extend({**base, **row} for row in transitions)
    return soft_rows, transition_rows


def plot_decomposition(frame: pd.DataFrame, output_dir: Path) -> None:
    selected = frame[
        (frame["neighbors"] == 20)
        & (frame["weighting"] == "common_biological_total")
    ].copy()
    intervals = [f"{a}→{b}" for a, b in INTERVALS]
    components = ["state_transport", "local_reweighting"]
    labels = ["Cell-state transport", "Local net source/sink"]
    colors = ["#4C78A8", "#F58518"]
    fig, axes = plt.subplots(1, len(intervals), figsize=(7.0, 3.3), sharey=False)
    if len(intervals) == 1:
        axes = [axes]
    for ax, interval in zip(axes, intervals):
        local = selected[selected["interval"] == interval]
        x = np.arange(2, dtype=float)
        width = 0.32
        for method_index, method in enumerate(["USOT RNA-only", "USOT Sync"]):
            method_frame = local[local["method"] == method]
            means = method_frame[components].mean(axis=0).to_numpy(float)
            errors = method_frame[components].std(axis=0, ddof=1).fillna(0).to_numpy(float)
            offset = (method_index - 0.5) * width
            bars = ax.bar(
                x + offset,
                means,
                width,
                yerr=errors if method == "USOT Sync" else None,
                capsize=3,
                color="#1679B8" if method == "USOT RNA-only" else "#E67817",
                label=method,
                zorder=2,
            )
            if method == "USOT Sync":
                for component_index, component in enumerate(components):
                    values = method_frame[component].to_numpy(float)
                    jitter = np.linspace(-0.07, 0.07, len(values)) if len(values) > 1 else np.zeros(1)
                    ax.scatter(
                        np.full(len(values), x[component_index] + offset) + jitter,
                        values,
                        s=12,
                        facecolors="none",
                        edgecolors="#7A3B00",
                        linewidths=0.7,
                        zorder=3,
                    )
            for bar, value in zip(bars, means):
                if method == "USOT RNA-only":
                    label_y = value - 0.018 if value >= 0 else value + 0.018
                    va = "top" if value >= 0 else "bottom"
                    color = "white"
                else:
                    label_y = value + 0.045 if value >= 0 else value - 0.045
                    va = "bottom" if value >= 0 else "top"
                    color = "0.15"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    f"{value:+.2f}",
                    ha="center",
                    va=va,
                    color=color,
                )
        ax.axhline(0, color="0.25", linewidth=0.8)
        ax.set_xticks(x, labels, rotation=22, ha="right")
        ax.set_title(interval)
        ax.set_ylabel("Change in NMP biological mass")
        ax.margins(y=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(top=0.79, bottom=0.30, left=0.10, right=0.98, wspace=0.35)
    for suffix in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"nmp_mass_change_decomposition_usot.{suffix}",
            dpi=360 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    reference = ad.read_h5ad(args.reference)
    scale = float(torch.load(args.scale_file, map_location="cpu", weights_only=False)["scale"])
    classifiers = {k: StageClassifier(reference, k) for k in sorted(set(args.neighbors))}

    time_path = args.sync_dir / "t_grid_s0_iter20000.pt"
    time_grid = load_tensor(time_path).astype(float).reshape(-1)
    methods: list[tuple[str, float | None, Path, Path]] = [
        ("USOT RNA-only", None, args.rna_trajectory, args.rna_mass)
    ]
    for c_y in np.arange(0.1, 1.0, 0.1):
        methods.append(
            (
                "USOT Sync",
                float(np.round(c_y, 1)),
                args.sync_dir / f"primary_trajectory_s0_a{c_y:.1f}_iter20000.pt",
                args.sync_dir / f"mass_lnw_trajectory_s0_a{c_y:.1f}_iter20000.pt",
            )
        )

    all_soft: list[dict[str, object]] = []
    all_transitions: list[dict[str, object]] = []
    for method, c_y, trajectory_path, mass_path in methods:
        trajectory = load_tensor(trajectory_path)
        log_mass = load_tensor(mass_path)
        soft, transitions = evaluate_method(
            method, c_y, trajectory, log_mass, time_grid, scale, classifiers
        )
        all_soft.extend(soft)
        all_transitions.extend(transitions)

    soft_frame = pd.DataFrame(all_soft)
    transition_frame = pd.DataFrame(all_transitions)
    soft_frame.to_csv(args.output_dir / "nmp_soft_mass_change_decomposition.csv", index=False)
    transition_frame.to_csv(args.output_dir / "nmp_hard_transition_breakdown.csv", index=False)

    sync_summary = (
        soft_frame[soft_frame["method"] == "USOT Sync"]
        .groupby(["neighbors", "interval", "weighting"], observed=False)
        .agg(
            n_c_y=("c_y", "size"),
            state_transport_mean=("state_transport", "mean"),
            state_transport_sd=("state_transport", "std"),
            local_reweighting_mean=("local_reweighting", "mean"),
            local_reweighting_sd=("local_reweighting", "std"),
            nmp_mass_change_mean=("nmp_mass_change", "mean"),
            nmp_mass_change_sd=("nmp_mass_change", "std"),
        )
        .reset_index()
    )
    sync_summary.to_csv(args.output_dir / "nmp_sync_c_y_summary.csv", index=False)
    plot_decomposition(soft_frame, args.output_dir)

    manifest = {
        "analysis": "Lagrangian NMP mass-change decomposition",
        "identity": "same 9,018 E7.5 particles for USOT RNA-only and all Sync C_y",
        "soft_membership": (
            "stage-conditioned distance-weighted RNA kNN probability of NMP; "
            "k=1 is the hard-label sensitivity"
        ),
        "weightings": {
            "relative_share": "q_i=w_i/sum_j w_j",
            "common_biological_total": "q_i multiplied by [1,4,6,11] at observed stages",
        },
        "identity_check": "delta_NMP = state_transport + local_reweighting exactly",
        "limitation": (
            "the source/sink term is model-inferred particle reweighting, not direct "
            "measurement of cell division or death"
        ),
    }
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2))
    show = soft_frame[
        (soft_frame["neighbors"] == 20)
        & (soft_frame["weighting"] == "common_biological_total")
    ]
    print(show[["method", "c_y", "interval", "nmp_mass_change", "state_transport", "local_reweighting", "closure_error"]].to_string(index=False))


if __name__ == "__main__":
    main()
