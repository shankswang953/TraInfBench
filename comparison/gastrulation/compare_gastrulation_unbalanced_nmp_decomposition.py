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
import numpy as np
import pandas as pd
import torch

from decompose_gastrulation_nmp_mass_change import (
    BIO_TOTAL,
    INTERVALS,
    STAGE_TIME,
    StageClassifier,
    hard_waterfall,
    load_tensor,
    normalized_weights,
    soft_decomposition,
)
from evaluate_gastrulation_flow_methods_normalized import (
    _cytobridge_velocity,
    _load_cytobridge_model,
)
from evaluate_tigon_gastrulation_normalized import (
    _infer_model_input_space,
    _integrate_normalized,
    _load_model as _load_tigon_model,
    _load_tigon_module,
)


BENCH_ROOT = Path(__file__).resolve().parents[2]
GAST_ROOT = Path("external/COATI/Gastrulation")
STAGES = ["E7.5", "E8.0", "E8.5", "E8.75"]
STAGE_STEPS = {"E7.5": 0, "E8.0": 10, "E8.5": 20, "E8.75": 25}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare NMP state transport and local mass reweighting for USOT, "
            "unbalanced CytoBridge, and TIGON on the same E7.5 particles."
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
        "--sync-dir",
        type=Path,
        default=GAST_ROOT / "UnbalancedSync_biological_num/trajectory_hpc_iter20000",
    )
    parser.add_argument(
        "--cytobridge-adata",
        type=Path,
        default=BENCH_ROOT / "results/cytobridge_gastrulation_rna_20000_unbalanced/adata.h5ad",
    )
    parser.add_argument(
        "--tigon-model",
        type=Path,
        default=BENCH_ROOT / "results/tigon_gastrulation_rna_20000/tigon.pt",
    )
    parser.add_argument(
        "--tigon-config",
        type=Path,
        default=BENCH_ROOT / "results/tigon_gastrulation_rna_20000/config.json",
    )
    parser.add_argument("--neighbors", nargs="+", type=int, default=[1, 10, 20])
    parser.add_argument("--cytobridge-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCH_ROOT / "results/gastrulation_unbalanced_nmp_decomposition",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _stage_arrays_from_usot(
    trajectory: np.ndarray, log_mass: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.stack([trajectory[STAGE_STEPS[stage]] for stage in STAGES])
    masses = np.stack([log_mass[STAGE_STEPS[stage]] for stage in STAGES])
    if masses.ndim == 3 and masses.shape[-1] == 1:
        masses = masses[..., 0]
    return positions.astype(np.float32, copy=False), masses.astype(np.float64, copy=False)


def _rollout_cytobridge(
    x0: np.ndarray,
    scale: float,
    model,
    *,
    steps: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    positions = [x0.astype(np.float32, copy=True)]
    masses = [np.zeros(x0.shape[0], dtype=np.float64)]
    current = x0.astype(np.float32, copy=True)
    cumulative = np.zeros(x0.shape[0], dtype=np.float64)
    for stage0, stage1 in zip(STAGES[:-1], STAGES[1:]):
        t0 = STAGE_TIME[stage0]
        t1 = STAGE_TIME[stage1]
        dt = (t1 - t0) / float(steps)
        next_position = np.empty_like(current)
        next_growth = np.empty(current.shape[0], dtype=np.float64)
        for start in range(0, current.shape[0], batch_size):
            stop = min(start + batch_size, current.shape[0])
            x = torch.as_tensor(current[start:stop], dtype=torch.float32, device=device)
            growth_sum = torch.zeros(x.shape[0], dtype=x.dtype, device=device)
            with torch.no_grad():
                for step in range(steps):
                    t_mid = torch.tensor(
                        [t0 + (step + 0.5) * dt], dtype=x.dtype, device=device
                    )
                    x_raw = x * scale
                    velocity = _cytobridge_velocity(model, t_mid, x_raw)
                    growth = model.growth_net(
                        torch.cat([x_raw, t_mid.expand(x.shape[0], 1)], dim=1)
                    )[:, 0]
                    x = x + dt * velocity / scale
                    growth_sum = growth_sum + dt * growth
            next_position[start:stop] = x.detach().cpu().numpy()
            next_growth[start:stop] = growth_sum.detach().cpu().numpy()
        current = next_position
        cumulative = cumulative + next_growth
        positions.append(current.copy())
        masses.append(cumulative.copy())
    return np.stack(positions), np.stack(masses)


def _rollout_tigon(
    x0: np.ndarray,
    scale: float,
    model,
    config: dict,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, str, int]:
    model_input_space = _infer_model_input_space(config, "auto")
    ode_steps = int(config.get("ode_steps", 8))
    positions = [x0.astype(np.float32, copy=True)]
    masses = [np.zeros(x0.shape[0], dtype=np.float64)]
    current = x0.astype(np.float32, copy=True)
    cumulative = np.zeros(x0.shape[0], dtype=np.float64)
    for stage0, stage1 in zip(STAGES[:-1], STAGES[1:]):
        next_position = np.empty_like(current)
        next_growth = np.empty(current.shape[0], dtype=np.float64)
        for start in range(0, current.shape[0], batch_size):
            stop = min(start + batch_size, current.shape[0])
            x = torch.as_tensor(current[start:stop], dtype=torch.float32, device=device)
            with torch.no_grad():
                endpoint, log_growth, _, _ = _integrate_normalized(
                    model,
                    x,
                    STAGE_TIME[stage0],
                    STAGE_TIME[stage1],
                    ode_steps,
                    scale,
                    model_input_space,
                )
            next_position[start:stop] = endpoint.detach().cpu().numpy()
            next_growth[start:stop] = log_growth.detach().cpu().numpy()
        current = next_position
        cumulative = cumulative + next_growth
        positions.append(current.copy())
        masses.append(cumulative.copy())
    return np.stack(positions), np.stack(masses), model_input_space, ode_steps


def _evaluate(
    method: str,
    c_y: float | None,
    positions: np.ndarray,
    log_mass: np.ndarray,
    scale: float,
    classifiers: dict[int, StageClassifier],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    transitions: list[dict[str, object]] = []
    stage_index = {stage: index for index, stage in enumerate(STAGES)}
    for stage0, stage1 in INTERVALS:
        idx0 = stage_index[stage0]
        idx1 = stage_index[stage1]
        x0 = positions[idx0] * scale
        x1 = positions[idx1] * scale
        q0 = normalized_weights(log_mass[idx0])
        q1 = normalized_weights(log_mass[idx1])
        for k, classifier in classifiers.items():
            p0, label0 = classifier.membership_and_label(x0, stage0)
            p1, label1 = classifier.membership_and_label(x1, stage1)
            for weighting, weight0, weight1 in (
                ("relative_share", q0, q1),
                (
                    "common_biological_total",
                    q0 * BIO_TOTAL[stage0],
                    q1 * BIO_TOTAL[stage1],
                ),
            ):
                soft = soft_decomposition(weight0, weight1, p0, p1)
                hard, transition_rows = hard_waterfall(weight0, weight1, label0, label1)
                denominator = abs(soft["state_transport"]) + abs(soft["local_reweighting"])
                base = {
                    "method": method,
                    "c_y": c_y,
                    "neighbors": k,
                    "interval": f"{stage0}→{stage1}",
                    "weighting": weighting,
                }
                rows.append(
                    {
                        **base,
                        **soft,
                        **hard,
                        "local_fraction_of_absolute_components": (
                            abs(soft["local_reweighting"]) / denominator
                            if denominator > 0
                            else np.nan
                        ),
                    }
                )
                transitions.extend({**base, **row} for row in transition_rows)
    return rows, transitions


def _observed_nmp_share(reference: ad.AnnData) -> pd.DataFrame:
    stages = reference.obs["stage"].astype(str)
    labels = reference.obs["celltype"].astype("string").fillna("Unannotated")
    rows = []
    for stage in STAGES:
        mask = stages.eq(stage)
        rows.append(
            {
                "stage": stage,
                "n_cells": int(mask.sum()),
                "n_nmp": int((mask & labels.eq("NMP")).sum()),
                "observed_nmp_share": float((mask & labels.eq("NMP")).sum() / mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def _plot(frame: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update(
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
    selected = frame[
        frame["neighbors"].eq(20) & frame["weighting"].eq("relative_share")
    ].copy()
    summary_rows = []
    for method in ("USOT Sync", "CytoBridge unbalanced", "TIGON"):
        local = selected[selected["method"].eq(method)]
        for interval in [f"{a}→{b}" for a, b in INTERVALS]:
            part = local[local["interval"].eq(interval)]
            row = {"method": method, "interval": interval}
            for column in ("state_transport", "local_reweighting", "nmp_mass_change"):
                row[f"{column}_mean"] = float(part[column].mean())
                row[f"{column}_sd"] = float(part[column].std(ddof=1)) if len(part) > 1 else 0.0
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "nmp_relative_share_method_summary.csv", index=False)

    colors = {
        "USOT Sync": "#D55E00",
        "CytoBridge unbalanced": "#7F7F7F",
        "TIGON": "#4D4D4D",
    }
    components = ["state_transport", "local_reweighting"]
    labels = ["State transport", "Local reweighting"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25), sharey=True)
    methods = list(colors)
    width = 0.23
    for ax, interval in zip(axes, [f"{a}→{b}" for a, b in INTERVALS]):
        local = summary[summary["interval"].eq(interval)]
        x = np.arange(len(components), dtype=float)
        for method_index, method in enumerate(methods):
            row = local[local["method"].eq(method)].iloc[0]
            values = np.asarray([row[f"{c}_mean"] for c in components])
            errors = np.asarray([row[f"{c}_sd"] for c in components])
            offset = (method_index - 1) * width
            ax.bar(
                x + offset,
                values,
                width,
                yerr=errors if method == "USOT Sync" else None,
                capsize=2.5,
                color=colors[method],
                label=method,
                zorder=2,
            )
        ax.axhline(0, color="#333333", linewidth=0.9)
        ax.set_xticks(x, labels)
        ax.set_title(interval, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#E2E2E2", linewidth=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Change in NMP relative mass share")
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.17, top=0.79, wspace=0.25)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"unbalanced_nmp_decomposition.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_table = args.output_dir / "nmp_mass_decomposition_all_methods.csv"
    if output_table.exists() and not args.overwrite:
        raise FileExistsError(f"{output_table} exists; pass --overwrite to replace it")

    device = torch.device(args.device)
    reference = ad.read_h5ad(args.reference)
    scale = float(
        np.asarray(
            torch.load(args.scale_file, map_location="cpu", weights_only=False)["scale"]
        ).reshape(-1)[0]
    )
    classifiers = {k: StageClassifier(reference, k) for k in sorted(set(args.neighbors))}

    time_grid = load_tensor(args.sync_dir / "t_grid_s0_iter20000.pt").reshape(-1)
    expected_time = np.arange(0.0, 2.5 + 1e-8, 0.1)
    if not np.allclose(time_grid, expected_time):
        raise ValueError("Unexpected USOT time grid")

    trajectories: list[tuple[str, float | None, np.ndarray, np.ndarray]] = []
    x0: np.ndarray | None = None
    for c_y in np.round(np.arange(0.1, 1.0, 0.1), 1):
        trajectory = load_tensor(
            args.sync_dir / f"primary_trajectory_s0_a{c_y:.1f}_iter20000.pt"
        )
        log_mass = load_tensor(
            args.sync_dir / f"mass_lnw_trajectory_s0_a{c_y:.1f}_iter20000.pt"
        )
        positions, masses = _stage_arrays_from_usot(trajectory, log_mass)
        if x0 is None:
            x0 = positions[0].copy()
        elif not np.array_equal(x0, positions[0]):
            raise ValueError(f"USOT C_y={c_y:.1f} has a different initial particle set")
        trajectories.append(("USOT Sync", float(c_y), positions, masses))
    assert x0 is not None

    cytobridge = _load_cytobridge_model(args.cytobridge_adata, device)
    cb_positions, cb_masses = _rollout_cytobridge(
        x0,
        scale,
        cytobridge,
        steps=args.cytobridge_steps,
        batch_size=args.batch_size,
        device=device,
    )
    trajectories.append(("CytoBridge unbalanced", None, cb_positions, cb_masses))

    tigon_module = _load_tigon_module()
    tigon, tigon_config, _ = _load_tigon_model(
        tigon_module, args.tigon_model, args.tigon_config, device
    )
    tigon_positions, tigon_masses, tigon_space, tigon_steps = _rollout_tigon(
        x0,
        scale,
        tigon,
        tigon_config,
        batch_size=args.batch_size,
        device=device,
    )
    trajectories.append(("TIGON", None, tigon_positions, tigon_masses))

    all_rows: list[dict[str, object]] = []
    all_transitions: list[dict[str, object]] = []
    for method, c_y, positions, masses in trajectories:
        rows, transition_rows = _evaluate(
            method, c_y, positions, masses, scale, classifiers
        )
        all_rows.extend(rows)
        all_transitions.extend(transition_rows)
    frame = pd.DataFrame(all_rows)
    transition_frame = pd.DataFrame(all_transitions)
    frame.to_csv(output_table, index=False)
    transition_frame.to_csv(
        args.output_dir / "nmp_hard_transition_breakdown_all_methods.csv", index=False
    )
    _observed_nmp_share(reference).to_csv(
        args.output_dir / "observed_nmp_share.csv", index=False
    )
    _plot(frame, args.output_dir)

    np.savez_compressed(
        args.output_dir / "external_method_stage_rollouts.npz",
        stages=np.asarray(STAGES, dtype=object),
        source_x0_norm=x0,
        cytobridge_positions_norm=cb_positions,
        cytobridge_log_mass=cb_masses,
        tigon_positions_norm=tigon_positions,
        tigon_log_mass=tigon_masses,
    )
    manifest = {
        "analysis": "NMP mass-change decomposition across unbalanced trajectory models",
        "common_particles": int(x0.shape[0]),
        "common_source": "the same 9,018 real E7.5 RNA PCA states",
        "primary_weighting": "within-method relative mass share q_i=w_i/sum_j w_j",
        "secondary_weighting": "the same external biological totals [1,4,6,11] applied to every method",
        "classifier": "common stage-conditioned RNA kNN; k=20 primary, k=1/10 sensitivity",
        "identity": "Delta NMP mass = state transport + local reweighting (midpoint product identity)",
        "usot_summary": "mean and SD across all C_y=0.1,...,0.9; no best parameter",
        "cytobridge_steps_per_interval": args.cytobridge_steps,
        "tigon_steps_per_interval": tigon_steps,
        "tigon_model_input_space": tigon_space,
        "limitation": "local reweighting is a model-inferred net source/sink, not direct cell division or death",
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(frame[(frame["neighbors"] == 20) & (frame["weighting"] == "relative_share")][
        ["method", "c_y", "interval", "nmp_mass_change", "state_transport", "local_reweighting", "closure_error"]
    ].to_string(index=False))
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
