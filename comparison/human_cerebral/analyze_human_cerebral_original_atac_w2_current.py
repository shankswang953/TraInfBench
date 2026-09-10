#!/usr/bin/env python3
"""Full-train current-checkpoint ATAC reconstruction against original scATAC."""

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
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import ot
import pandas as pd
import torch

from analyze_human_cerebral_common_t_atac_recovery_by_time import (
    T_CHECKPOINT, apply_t, decoded_pca30, load_t, sha256, unit_projections,
)
from analyze_human_cerebral_source_line_group_w2 import (
    LINES, OBSERVED_INDICES, PHYSICAL_TIMES, TIME_LABELS,
    load_primary_rollouts, load_reference, normalized_weights, observed_line_mask,
)
from plot_human_cerebral_raw_cell_rna_w2_all_methods_by_time import (
    ORDER, STYLE_NAMES,
)
from trainfbench_plot_style import apply_nature_rc, method_style


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/human_cerebral_full_biological_interpretability"
OUTPUT = BASE / "13_original_atac_w2_current"
ORIGINAL = ROOT / "results/human_cerebral_original_atac_frozen_lsi12_umap/original_atac_frozen_lsi12.npz"
PREVIOUS = BASE / "11_common_t_atac_distribution_recovery/model_time_source_line_scores.csv"
STAGES = (1, 2, 3, 6)
TIMES = tuple(TIME_LABELS[x] for x in STAGES)


def distance(query, projected_target, directions, weights=None) -> float:
    values = ot.wasserstein_1d(
        np.asarray(query, dtype=np.float64) @ directions, projected_target,
        u_weights=weights, p=2, require_sort=True,
    )
    return float(np.sqrt(np.mean(values)))


def make_table(frame: pd.DataFrame) -> pd.DataFrame:
    sizes = frame.groupby(["method", "time"]).size()
    if len(sizes) != 32 or not sizes.eq(4).all():
        raise ValueError("Expected eight methods x four days x four lines")
    table = frame.groupby(["method", "time"]).sliced_w2.mean().unstack("time")
    table = table.reindex(index=ORDER, columns=TIMES)
    table["Mean"] = table.mean(axis=1)
    table.index.name = "Method"
    return table


def plot(table: pd.DataFrame, reference: pd.Series, stem: Path,
         limits: tuple[float, float]) -> None:
    apply_nature_rc(font_size=8)
    fig, ax = plt.subplots(figsize=(7.15, 3.65))
    centers = np.arange(4, dtype=float)
    offsets = np.linspace(-0.24, 0.16, 8)
    handles = []
    for i, method in enumerate(ORDER):
        style = method_style(STYLE_NAMES[method])
        face = style.color if style.markerfacecolor is None else style.markerfacecolor
        edge = style.color if style.markeredgecolor is None else style.markeredgecolor
        ax.scatter(centers + offsets[i], table.loc[method, list(TIMES)],
                   marker=style.marker, facecolors=face, edgecolors=edge,
                   s=49 if method.startswith("COATI") else 40,
                   linewidths=0.8, zorder=3)
        handles.append(Line2D([], [], linestyle="none", marker=style.marker,
                              markerfacecolor=face, markeredgecolor=edge,
                              markersize=5.5, label=method))
    ax.scatter(centers + 0.25, reference.reindex(TIMES), facecolors="none",
               edgecolors="#777777", marker="o", s=37, linewidths=1, zorder=4)
    handles.append(Line2D([], [], linestyle="none", marker="o", markersize=5,
                          markerfacecolor="none", markeredgecolor="#777777",
                          label="Observed metacells"))
    ax.set_xticks(centers, TIMES)
    ax.set_xlim(-0.45, 3.45)
    ax.set_ylim(*limits)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"ATAC sliced $W_2$ to original cells $\downarrow$")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#666666")
        spine.set_linewidth(0.75)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=3, frameon=False, fontsize=7.2, handletextpad=0.4,
               columnspacing=1.15)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.755, bottom=0.16)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=600,
                    bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    with np.load(ORIGINAL, allow_pickle=False) as saved:
        points = np.asarray(saved["corrected_normalized_lsi12"], dtype=np.float64)
        days = saved["age"].astype(int)
        lines = np.char.lower(saved["line"].astype(str))
        ids = saved["cell_id"].astype(str)
    if points.shape != (22692, 12) or not np.isfinite(points).all():
        raise ValueError("Unexpected original ATAC coordinates")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate original ATAC cell IDs")
    reference = load_reference()
    registry, rollouts = load_primary_rollouts()
    device = torch.device("cpu")
    model, _ = load_t(T_CHECKPOINT, device)
    directions = unit_projections(12, 512, 90217)
    targets, counts, refs, representation_samples = {}, [], [], []
    for stage in STAGES:
        day = int(TIME_LABELS[stage][1:])
        for line in LINES:
            mask = (days == day) & (lines == line)
            meta = reference.atac[stage][observed_line_mask(reference, stage, line)]
            raw = points[mask]
            if not len(raw) or not len(meta):
                raise ValueError(f"Missing target {day}/{line}")
            for name, array in (("original_cell", raw), ("metacell", meta)):
                targets[(stage, line, name)] = np.sort(
                    np.asarray(array, dtype=np.float64) @ directions, axis=0)
            counts.append(dict(time=TIME_LABELS[stage], source_line=line,
                               original_cells=len(raw), metacells=len(meta),
                               particles=int((reference.source_lines == line).sum())))
            refs.append(dict(time=TIME_LABELS[stage], source_line=line,
                             sliced_w2=distance(meta, targets[(stage, line, "original_cell")], directions)))
            representation_samples.extend(ids[mask].tolist())

    rows, map_audit = [], []
    for order, ((label, rollout), (_, model_row)) in enumerate(zip(rollouts, registry.iterrows(), strict=True)):
        print(f"Scoring {label}", flush=True)
        rna = decoded_pca30(model_row, rollout.rna)
        for stage in STAGES:
            index = int(OBSERVED_INDICES[stage])
            common = apply_t(model, rna[index], float(PHYSICAL_TIMES[stage]), device, 2048)
            # Audit saved Sync secondary coordinates against a fresh common-T
            # readout; this is NOT an independently integrated ATAC ODE.
            native = rollout.atac[index] if rollout.sync_mode == "RNA_ATAC_sync" else common
            map_audit.append(dict(
                model_id=rollout.model_id, time=TIME_LABELS[stage],
                atac_semantics=rollout.atac_semantics,
                saved_vs_recomputed_t_max_abs=float(np.max(np.abs(rollout.atac[index] - common))),
            ))
            mass = None if rollout.log_mass is None else rollout.log_mass[index]
            for line in LINES:
                mask = reference.source_lines == line
                weightings = {"uniform": np.full(mask.sum(), 1.0 / mask.sum()),
                              "native_mass": normalized_weights(mass, mask)}
                for route, query in (("common_T_all_methods", common),
                                     ("native_COATI_vs_common_T_baselines", native)):
                    for weighting, weights in weightings.items():
                        for target in ("original_cell", "metacell"):
                            rows.append(dict(method=ORDER[order], model_id=rollout.model_id,
                                             time=TIME_LABELS[stage], source_line=line,
                                             route=route, query_weighting=weighting,
                                             target=target, sliced_w2=distance(
                                                 query[mask], targets[(stage, line, target)],
                                                 directions, weights)))
    scores = pd.DataFrame(rows)
    if len(scores) != 8 * 4 * 4 * 2 * 2 * 2 or not np.isfinite(scores.sliced_w2).all():
        raise ValueError("Incomplete or nonfinite scores")

    # Prove the common-T pipeline reproduces the existing metacell benchmark.
    current = scores.loc[scores.route.eq("common_T_all_methods") &
                         scores.query_weighting.eq("uniform") & scores.target.eq("metacell")]
    old = pd.read_csv(PREVIOUS)
    keys = ["model_id", "time", "source_line"]
    check = current.merge(old[keys + ["sliced_w2"]], on=keys, how="left",
                          validate="one_to_one", suffixes=("_new", "_old"))
    if check.sliced_w2_old.isna().any():
        raise ValueError("Missing original common-T comparison rows")
    reproduction_error = float((check.sliced_w2_new - check.sliced_w2_old).abs().max())
    if reproduction_error > 1e-7:
        raise ValueError(f"Common-T metacell scores changed: {reproduction_error}")

    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "model_time_source_target_scores.csv", index=False)
    pd.DataFrame(counts).to_csv(output / "target_and_particle_counts.csv", index=False)
    pd.DataFrame(map_audit).to_csv(output / "t_readout_audit.csv", index=False)
    registry.to_csv(output / "frozen_model_subset.csv", index=False)
    ref = pd.DataFrame(refs)
    ref.to_csv(output / "observed_metacell_to_original_atac_reference.csv", index=False)
    ref_by_time = ref.groupby("time").sliced_w2.mean().reindex(TIMES)
    ref_by_time.to_csv(output / "observed_metacell_to_original_atac_by_time.csv")
    tables = {}
    for route in pd.unique(scores.route):
        for weighting in ("uniform", "native_mass"):
            for target in ("original_cell", "metacell"):
                table = make_table(scores.loc[scores.route.eq(route) &
                                             scores.query_weighting.eq(weighting) &
                                             scores.target.eq(target)])
                table.to_csv(output / f"{route}_{target}_{weighting}_by_time.csv",
                             float_format="%.10f")
                if weighting == "uniform" and target == "original_cell":
                    tables[route] = table
    values = np.concatenate([t[list(TIMES)].to_numpy().ravel() for t in tables.values()]
                            + [ref_by_time.to_numpy()])
    lo, hi = float(values.min()), float(values.max())
    limits = (max(0, lo - .08 * (hi - lo)), hi + .08 * (hi - lo))
    for route, table in tables.items():
        plot(table, ref_by_time, output / f"{route}_original_atac_w2_by_time_scatter", limits)
    mean_table = pd.DataFrame({route: t.Mean for route, t in tables.items()})
    mean_table.to_csv(output / "mean_w2_by_readout.csv", float_format="%.10f")
    manifest = dict(
        original_source=str(ORIGINAL), original_source_sha256=sha256(ORIGINAL),
        original_preprocess_manifest=str(ORIGINAL.parent / "analysis_manifest.json"),
        t_checkpoint=str(T_CHECKPOINT), t_sha256=sha256(T_CHECKPOINT),
        target="Real scATAC cells, not RNA mapped by T; corrected_normalized_lsi12",
        model_space="Frozen corrected normalized LSI12, 12 dimensions; no UMAP distances",
        normalization_scale=34.16607592332852, no_knn_for_w2=True,
        fixed_readout_k_when_needed=15, projection_seed=90217, n_projections=512,
        primary_weighting="Uniform query and target per line; four lines equal; four days equal",
        checkpoint_policy="COATI unbalanced Sync Cy0.5 40k; others30k; not selecting by this endpoint",
        times=list(TIMES), original_reference_total=len(points),
        evaluable_original_cells=len(representation_samples),
        missing_line_cells_at_evaluable_days=int((np.isin(days,[7,9,11,21]) & ~np.isin(lines,LINES)).sum()),
        excluded_times="D4 shared initial; D16 outside chosen benchmark; D12/D18 no raw ATAC",
        prior_metacell_score_max_absolute_error=reproduction_error,
        prior_metacell_score_comparisons=len(check),
        caveat="Full-train reconstruction, not independent external or LOO validation",
        representation_reference="Observed metacell->original; not a mathematical lower bound",
        saved_secondary_semantics="Legacy native_COATI route names mean saved Sync secondary; not an independent ATAC ODE. The primary comparison recomputes common T for all methods.",
        original_peak_alignment="Genomic-overlap binary union to frozen regions, not exact fragment recounts; 184572/184915 mapped peaks,343 absent.",
        representation_gap_caveat="Aggregation, peak-boundary alignment and metacell-trained QC residualization contribute to the original/metacell gap, not purely biology.",
    )
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "README.md").write_text(f"""# Current-checkpoint original ATAC distribution recovery

The target is the official original scATAC cell distribution projected with the
frozen training TF/IDF/SVD, QC/line residualization, selected LSI12 dimensions,
and scale 34.16607592332852. It is **not** metacell LSI or RNA transformed by T.
Evaluable days: D7, D9, D11, D21. D12/D18 have no original scATAC libraries;
D16 is not part of this requested seven-time benchmark; D4 is the shared start.
{len(representation_samples)} original cells with known source line enter the comparison.

All methods use the same 501 D4-origin particles and the same source-line strata.
Primary weights are uniform; native model-mass sensitivity is separately saved.
Scores are sliced W2 in corrected normalized LSI12, 512 directions, seed90217.
No kNN replaces trajectory points and no classifier labels enter the distances.
Four lines equal per day, four days equal for the overall mean.

COATI unbalanced Sync uses converged Cy0.5 40k; balanced Sync Cy0.5 and all other
methods use 30k. These are the same model choices as the previous RNA analysis.
Previous common-T metacell scores reproduce to {reproduction_error:.3g} max error.

## 1. Same frozen T for all methods

Every predicted RNA state, including COATI, is mapped with the identical full-T.
MIOFlow first uses its saved frozen decoder to normalized PCA30. This isolates
the RNA-trajectory-through-T readout. Sync changes training constraints, not the
fact that the saved secondary can itself be T(RNA).

![Common T](common_T_all_methods_original_atac_w2_by_time_scatter.png)

## 2. Saved COATI secondary audit (legacy filename contains native_COATI)

Only the two COATI Sync arms switch to their saved secondary coordinates.
All six other methods are unchanged. These are not independently integrated ATAC
trajectories: the materializer computes secondary=T(RNA). If the same T was
used, this audit duplicates the common-T result up to numerical precision and
must not be treated as additional biological evidence.

![Saved secondary audit](native_COATI_vs_common_T_baselines_original_atac_w2_by_time_scatter.png)

The grey hollow circles are **observed metacells vs original scATAC**, a
representation-gap reference, not a guaranteed attainable lower bound.
Both plots have identical axis limits and no connecting lines.

These original cells contributed to metacell construction. This remains
full-train high-resolution reconstruction, not independent biological validation.
The original-cell ATAC cell-type labels were transferred from RNA and are not
used here. Raw-vs-metacell representation differences can dominate the score;
do not interpret small ranking changes as proven biological superiority.

Original peaks are aligned by overlap and binary union, not fragment recounting
in exact frozen regions. 184,572/184,915 peaks overlap; 343 are absent. The
frozen QC correction was fit on metacells and original cells use n_cells_ATAC=1.
Aggregation, peak-boundary and QC effects all contribute to the target gap.

Mean representation reference: {ref.sliced_w2.mean():.6f}.
Tables retain both targets, both readout routes and both weighting choices.
""")
    print(mean_table.to_string(float_format=lambda x: f"{x:.6f}"))
    print("Representation reference:", ref.sliced_w2.mean())
    print("Common-T old-score max difference:", reproduction_error)
    print(output)


if __name__ == "__main__":
    main()
