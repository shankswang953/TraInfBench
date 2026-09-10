#!/usr/bin/env python3
"""Compare full-trajectory RNA states after a common frozen RNA-to-ATAC map.

For every method, the 501 common D4-origin particles are represented in the
same corrected normalized PCA30 space.  MIOFlow uses its frozen decoded PCA30
coordinates.  At each observed time, the same full-data FiLM map T converts
RNA coordinates to corrected normalized ATAC LSI12.  The mapped distribution
for each D4 source line is compared directly with the observed paired-metacell
ATAC distribution for the matching time and source line using sliced W2.

This is a common T readout for all methods, including COATI. Sync changes the
training constraints; saved Sync secondary coordinates may themselves be
T(RNA), not independently integrated ATAC dynamics.
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
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import ot
import pandas as pd
import torch

from analyze_human_cerebral_source_line_group_w2 import (
    LINES,
    OBSERVED_INDICES,
    PHYSICAL_TIMES,
    PRIMARY_MODELS,
    TIME_KEYS,
    TIME_LABELS,
    load_primary_rollouts,
    load_reference,
    observed_line_mask,
)
from trainfbench_plot_style import apply_nature_rc, method_style


ROOT = Path(__file__).resolve().parents[2]
HUMAN = Path("external/COATI/humanCerebral")
T_DIR = HUMAN / "Data/TrainT_7time_D4_D21_no_D16_lsi12"
T_CHECKPOINT = T_DIR / "T_FiLM.pt"
T_MODEL_SOURCE = HUMAN / "Data/TrainT/map_models.py"
DEFAULT_OUTPUT = (
    ROOT
    / "results/human_cerebral_full_biological_interpretability"
    / "11_common_t_atac_distribution_recovery"
)
N_PROJECTIONS = 512
RANDOM_SEED = 90217

DISPLAY_NAMES = {
    "COATI bal.": "COATI bal. (Sync)",
    "COATI unbal.": "COATI unbal. (Sync)",
    "CytoBridge bal.": "CytoBridge bal.",
    "CytoBridge unbal.": "CytoBridge unbal.",
    "MIOFlow": "MIOFlow",
    "TrajectoryNet": "TrajectoryNet",
    "OT(RNA)": "RNA-only bal.",
    "UOT(RNA)": "RNA-only unbal.",
}
STYLE_NAMES = {
    "COATI bal. (Sync)": "COATI balanced",
    "COATI unbal. (Sync)": "COATI unbalanced",
    "CytoBridge bal.": "CytoBridge balanced",
    "CytoBridge unbal.": "CytoBridge unbalanced",
    "MIOFlow": "MIOFlow",
    "TrajectoryNet": "TrajectoryNet",
    "RNA-only bal.": "Balanced RNA-only",
    "RNA-only unbal.": "Unbalanced RNA-only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--t-checkpoint", type=Path, default=T_CHECKPOINT)
    parser.add_argument("--n-projections", type=int, default=N_PROJECTIONS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-d4", action="store_true",
                        help="Include the initial D4 mapping diagnostic as well as all later times")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_t(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    spec = importlib.util.spec_from_file_location("human_cerebral_common_t", T_MODEL_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import T architecture from {T_MODEL_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = module.FiLMMLP(**state["config"]).to(device)
    model.load_state_dict(state["state_dict"])
    return model.eval().requires_grad_(False), state


@torch.inference_mode()
def apply_t(
    model: torch.nn.Module,
    points: np.ndarray,
    physical_time: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    outputs: list[np.ndarray] = []
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        x = torch.as_tensor(points[start:stop], dtype=torch.float32, device=device)
        t = torch.full((stop - start,), physical_time, dtype=torch.float32, device=device)
        outputs.append(model(x, t).detach().cpu().numpy())
    result = np.concatenate(outputs).astype(np.float32, copy=False)
    if result.shape != (len(points), 12) or not np.isfinite(result).all():
        raise ValueError(f"Unexpected T output: {result.shape}")
    return result


def unit_projections(dim: int, n_projections: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(dim, n_projections))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    return directions.astype(np.float64)


def sliced_w2(
    query: np.ndarray,
    target_projected_sorted: np.ndarray,
    projections: np.ndarray,
) -> float:
    query_projected = np.asarray(query, dtype=np.float64) @ projections
    losses = ot.wasserstein_1d(
        query_projected,
        target_projected_sorted,
        p=2,
        require_sort=True,
    )
    return float(np.sqrt(np.mean(np.asarray(losses, dtype=float))))


def decoded_pca30(row: pd.Series, rollout_rna: np.ndarray) -> np.ndarray:
    if not str(row["model_id"]).startswith("mioflow"):
        result = np.asarray(rollout_rna, dtype=np.float32)
    else:
        path = Path(str(row["trajectory_path"]))
        with np.load(path, allow_pickle=False) as saved:
            result = np.asarray(
                saved["trajectory_decoded_normalized_pca30"], dtype=np.float32
            )
    if result.shape != (66, 501, 30) or not np.isfinite(result).all():
        raise ValueError(f"Unexpected normalized PCA30 trajectory: {result.shape}")
    return result


def evaluate(
    registry: pd.DataFrame,
    rollouts: list,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    projections: np.ndarray,
    include_d4: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = load_reference()
    stages = range(0 if include_d4 else 1, len(TIME_KEYS))
    target_cache: dict[tuple[int, str], np.ndarray] = {}
    for stage in stages:
        for line in LINES:
            mask = observed_line_mask(reference, stage, line)
            target_cache[(stage, line)] = np.sort(
                np.asarray(reference.atac[stage][mask], dtype=np.float64) @ projections,
                axis=0,
            )

    floor_rows: list[dict[str, object]] = []
    for stage in stages:
        mapped_observed = apply_t(
            model,
            reference.pca[stage],
            float(PHYSICAL_TIMES[stage]),
            device,
            batch_size,
        )
        stage_frame = reference.obs.loc[
            reference.obs["time_key"].astype(str).eq(TIME_KEYS[stage])
        ].reset_index(drop=True)
        for line in LINES:
            mask = stage_frame["line"].astype(str).str.lower().eq(line).to_numpy()
            floor_rows.append(
                {
                    "time": TIME_LABELS[stage],
                    "physical_time": float(PHYSICAL_TIMES[stage]),
                    "source_line": line,
                    "query_n": int(mask.sum()),
                    "target_n": int(mask.sum()),
                    "sliced_w2": sliced_w2(
                        mapped_observed[mask], target_cache[(stage, line)], projections
                    ),
                }
            )

    rows: list[dict[str, object]] = []
    for method_order, (((label, rollout), registry_row), expected) in enumerate(
        zip(zip(rollouts, registry.itertuples(index=False)), PRIMARY_MODELS, strict=True)
    ):
        expected_id, expected_label = expected
        if rollout.model_id != expected_id or label != expected_label:
            raise RuntimeError("Primary model order changed unexpectedly")
        row_series = pd.Series(registry_row._asdict())
        trajectory = decoded_pca30(row_series, rollout.rna)
        display = DISPLAY_NAMES[label]
        for stage in stages:
            dense_index = int(OBSERVED_INDICES[stage])
            mapped = apply_t(
                model,
                trajectory[dense_index],
                float(PHYSICAL_TIMES[stage]),
                device,
                batch_size,
            )
            for line in LINES:
                source_mask = reference.source_lines == line
                target_mask = observed_line_mask(reference, stage, line)
                rows.append(
                    {
                        "method_order": method_order,
                        "method": display,
                        "registry_method": label,
                        "model_id": rollout.model_id,
                        "balance_mode": rollout.balance_mode,
                        "sync_mode": rollout.sync_mode,
                        "C_y": rollout.c_y,
                        "time": TIME_LABELS[stage],
                        "physical_time": float(PHYSICAL_TIMES[stage]),
                        "source_line": line,
                        "query_n": int(source_mask.sum()),
                        "observed_n": int(target_mask.sum()),
                        "sliced_w2": sliced_w2(
                            mapped[source_mask], target_cache[(stage, line)], projections
                        ),
                        "rna_coordinate_semantics": (
                            "frozen decoded normalized PCA30"
                            if rollout.model_id.startswith("mioflow")
                            else "native corrected normalized PCA30"
                        ),
                        "atac_coordinate_semantics": "common frozen full-T normalized LSI12",
                    }
                )
        print(f"scored {display}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(floor_rows)


def summarize(scores: pd.DataFrame, include_d4: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = list(TIME_LABELS[0 if include_d4 else 1:])
    by_time = (
        scores.groupby(
            ["method_order", "method", "time", "physical_time"],
            sort=False,
            as_index=False,
        )["sliced_w2"]
        .mean()
        .rename(columns={"sliced_w2": "mean_sliced_w2_across_source_lines"})
    )
    wide = (
        by_time.pivot(index=["method_order", "method"], columns="time", values="mean_sliced_w2_across_source_lines")
        .reindex(columns=times)
        .reset_index()
        .sort_values("method_order")
    )
    wide["Mean"] = wide.loc[:, times].mean(axis=1)
    if include_d4:
        wide["Mean_excluding_D4"] = wide.loc[:, list(TIME_LABELS[1:])].mean(axis=1)
    wide = wide.rename(columns={"method": "Method"}).drop(columns="method_order")
    return by_time, wide


def plot(wide: pd.DataFrame, output: Path, include_d4: bool = False) -> None:
    table = wide.set_index("Method")
    order = [DISPLAY_NAMES[label] for _, label in PRIMARY_MODELS]
    times = list(TIME_LABELS[0 if include_d4 else 1:])
    apply_nature_rc(font_size=8.0)
    fig, ax = plt.subplots(figsize=(7.15, 3.5))
    centers = np.arange(len(times), dtype=float)
    offsets = np.linspace(-0.22, 0.22, len(order))
    handles: list[Line2D] = []
    for method_index, method in enumerate(order):
        style = method_style(STYLE_NAMES[method])
        is_coati = method.startswith("COATI")
        face = style.color if style.markerfacecolor is None else style.markerfacecolor
        edge = style.color if style.markeredgecolor is None else style.markeredgecolor
        ax.scatter(
            centers + offsets[method_index],
            table.loc[method, times].to_numpy(dtype=float),
            s=49 if is_coati else 40,
            marker=style.marker,
            facecolors=face,
            edgecolors=edge,
            linewidths=0.9 if is_coati else 0.75,
            zorder=4 if is_coati else 3,
        )
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=style.marker,
                markersize=5.5,
                markerfacecolor=face,
                markeredgecolor=edge,
                markeredgewidth=0.8,
                label=method,
            )
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(r"ATAC sliced $W_2$ to metacells $\downarrow$")
    ax.set_xlim(-0.48, len(times) - 0.52)
    ax.set_xticks(centers, times)
    values = table.loc[:, times].to_numpy(dtype=float)
    lo, hi = float(values.min()), float(values.max())
    pad = max(0.08 * (hi - lo), 1e-4)
    ax.set_ylim(max(0.0, lo - pad), hi + pad)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#666666")
        spine.set_linewidth(0.75)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=7.2,
        handletextpad=0.45,
        columnspacing=1.15,
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.79, bottom=0.17)

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    stem = figures / "all_methods_common_t_atac_w2_by_time_scatter"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    key_output = output / "method_summary_by_time.csv"
    if key_output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {key_output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    t_model, t_state = load_t(args.t_checkpoint.resolve(), device)
    registry, rollouts = load_primary_rollouts()
    projections = unit_projections(12, args.n_projections, args.seed)
    scores, floor = evaluate(
        registry,
        rollouts,
        t_model,
        device,
        args.batch_size,
        projections,
        include_d4=args.include_d4,
    )
    by_time, wide = summarize(scores, include_d4=args.include_d4)
    times = list(TIME_LABELS[0 if args.include_d4 else 1:])
    expected_rows = len(PRIMARY_MODELS) * len(times) * len(LINES)
    keys = ["model_id", "time", "source_line"]
    if len(scores) != expected_rows or scores.duplicated(keys).any() or not np.isfinite(scores.sliced_w2).all():
        raise ValueError("Missing, duplicate or nonfinite method/time/source scores")
    reproduction_error = None
    previous_path = DEFAULT_OUTPUT / "model_time_source_line_scores.csv"
    if args.include_d4 and previous_path.exists() and previous_path.parent.resolve() != output:
        previous = pd.read_csv(previous_path)
        compared = scores.loc[scores.time.ne("D4")].merge(
            previous[keys + ["sliced_w2"]], on=keys, how="left", validate="one_to_one",
            suffixes=("_new", "_old"))
        if compared.sliced_w2_old.isna().any():
            raise ValueError("Previous six-day reference lacks some scores")
        reproduction_error = float((compared.sliced_w2_new - compared.sliced_w2_old).abs().max())
        if reproduction_error > 1e-7:
            raise ValueError(f"Previous six-day scores changed: {reproduction_error}")
    floor_by_time = (
        floor.groupby(["time", "physical_time"], sort=False, as_index=False)["sliced_w2"]
        .mean()
        .rename(columns={"sliced_w2": "mean_sliced_w2_across_source_lines"})
    )
    method_summary = (
        scores.groupby(["method_order", "method", "model_id"], sort=False, as_index=False)["sliced_w2"]
        .mean()
        .rename(columns={"sliced_w2": "mean_sliced_w2_equal_time_source_line"})
        .sort_values(["mean_sliced_w2_equal_time_source_line", "method_order"])
    )

    scores.to_csv(output / "model_time_source_line_scores.csv", index=False)
    by_time.to_csv(output / "method_by_time_long.csv", index=False)
    wide.to_csv(output / "method_summary_by_time.csv", index=False, float_format="%.9f")
    method_summary.to_csv(output / "method_summary_equal_time_source_line.csv", index=False)
    floor.to_csv(output / "observed_rna_through_t_floor_by_time_source_line.csv", index=False)
    floor_by_time.to_csv(output / "observed_rna_through_t_floor_by_time.csv", index=False)
    registry.to_csv(output / "frozen_model_subset.csv", index=False)
    plot(wide, output, include_d4=args.include_d4)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "t_checkpoint": str(args.t_checkpoint.resolve()),
        "t_checkpoint_sha256": sha256(args.t_checkpoint.resolve()),
        "t_config": t_state["config"],
        "methods": list(wide["Method"]),
        "times": times,
        "excluded_time": None if args.include_d4 else "D4 initial condition",
        "d4_interpretation": "Initial RNA/decoder-to-T mapping diagnostic, not trajectory evolution; no value forced to zero",
        "source_particles": 501,
        "source_lines": list(LINES),
        "aggregation": f"uniform within each D4 source line; equal mean over four lines within time; equal mean over {len(times)} times",
        "six_time_replication_max_abs_error": reproduction_error,
        "metric": "sliced W2 in corrected normalized LSI12",
        "n_random_projections": args.n_projections,
        "projection_seed": args.seed,
        "prediction_readout": "all methods' normalized PCA30 RNA trajectories passed directly through the same frozen full-data FiLM T; no kNN snapping",
        "mioflow_readout": "frozen decoded normalized PCA30 trajectory passed through the same T",
        "target": "observed paired-metacell ATAC distribution from matching time and source line",
        "raw_atac_used": False,
        "claim_limit": "full-training common-T metacell ATAC reconstruction; not original-cell ATAC validation or held-out prediction",
        "checkpoint_policy": {
            "COATI balanced Sync": 30000,
            "COATI unbalanced Sync": 40000,
            "all other methods": 30000,
        },
        "hpc_accessed": False,
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    best = method_summary.iloc[0]
    floor_mean = float(floor["sliced_w2"].mean())
    readme = f"""# Common-T ATAC distribution recovery by time

Every method's RNA trajectory is represented in the same corrected normalized
PCA30 space and passed directly through the same frozen full-data FiLM T.  The
resulting normalized LSI12 distribution is compared with observed paired-
metacell ATAC at the matching time and source line.  No kNN geometric snapping
is used.  Values are sliced W2; lower is better.

The primary plot averages the four D4 source-line groups equally within each
time. Times: {', '.join(times)}. Best equal-time/source-line mean: `{best['method']}`
({best['mean_sliced_w2_equal_time_source_line']:.6f}).  The observed-RNA-through-T
reference mean is {floor_mean:.6f}.

This is intentionally a common post-hoc T readout for every method, including
COATI. It tests whether each RNA trajectory maps to the expected ATAC
distribution under a shared frozen map. Sync changes the training constraints;
the saved Sync secondary may itself equal T(RNA), not an independent ATAC ODE.
The target is paired metacell ATAC, not original-cell
ATAC, so the result is full-training reconstruction/coherence rather than an
independent biological validation.

D4, when included, is the initial RNA/decoder-to-T diagnostic. It is not forced
to zero or used as evidence of trajectory evolution. `Mean` uses all displayed
times; `Mean_excluding_D4`, when present, preserves the previous six-time mean.
The MIOFlow input uses its frozen decoded GAGA coordinates even at D4, so any
initial reconstruction error is retained rather than silently replacing points.
The observed-RNA-through-T reference is not a mathematical lower bound.

Previous six-time replication maximum absolute difference: {reproduction_error}.

![ATAC metacell recovery](figures/all_methods_common_t_atac_w2_by_time_scatter.png)
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(wide.to_string(index=False), flush=True)
    print("\nObserved RNA -> T reference by time", flush=True)
    print(floor_by_time.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
