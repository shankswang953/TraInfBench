#!/usr/bin/env python
"""Evaluate palate strict-LOO ATAC states against external epigenetic evidence.

All methods are read from the shared-full-T strict-LOO prediction manifest.  In
particular, the ATAC LSI15 state of every baseline is the output of the same
frozen full-data RNA-to-ATAC map T.  A common training-stage-only kNN readout is
used only to convert the predicted LSI15 state into accessibility of fixed
external regions; held-out peak values never enter that readout.

The external regions are frozen before method scoring:

* broad anterior/posterior H3K27ac-specific atlas peaks from GSE138721;
* SHOX2/MEOX2 ChIP-supported active CREs, defined by replicate-consistent
  GSE250247 ChIP signal intersected with the corresponding H3K27ac direction.

The broad ChIP peak sets alone are retained in the audit but are not used as a
primary benchmark because they do not distinguish anterior from posterior in
the observed held-out scATAC data.
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
import gzip
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from analyze_palate_shox2_1nn_regulation import (
    ANTERIOR,
    ATAC_NORM,
    ATAC_RAW,
    ATAC_REFERENCE,
    GTF,
    POSTERIOR,
    RNA_NORM,
    RNA_REFERENCE,
)
from evaluate_palate_loo_same_space import _load_scale
from trainfbench_plot_style import apply_nature_rc, method_style


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT / "results/palate_strict_sync_loo_full_t_gaga10_tn_forward_native"
)
DEFAULT_OUTPUT = ROOT / "results/palate_strict_loo_external_epigenetic_evidence"
EXTERNAL_SOURCE = ROOT / "results/palate_shox2_external_validation"

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
METHOD_NAMES = {
    "COATI-B": "COATI balanced",
    "COATI-U": "COATI unbalanced",
    "CytoBridge-B": "CytoBridge balanced",
    "CytoBridge-U": "CytoBridge unbalanced",
    "MIOFlow": "MIOFlow",
    "TrajectoryNet": "TrajectoryNet",
    "RNAonly-B": "Balanced RNA-only",
    "RNAonly-U": "Unbalanced RNA-only",
}
MANIFEST_NAMES = {
    "COATI-B": "COATI-B",
    "COATI-U": "COATI-U",
    "CytoBridge balanced": "CytoBridge-B",
    "CytoBridge unbalanced": "CytoBridge-U",
    "MIOFlow": "MIOFlow",
    "TrajectoryNet": "TrajectoryNet",
    "Balanced RNA-only": "RNAonly-B",
    "Unbalanced RNA-only": "RNAonly-U",
}
SCENARIOS = {
    "loo_time1": {"stage": "E13.5", "time": 1.0},
    "loo_time2": {"stage": "E14.0", "time": 1.5},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--h3-quantile", type=float, default=0.80)
    parser.add_argument("--chip-quantile", type=float, default=0.90)
    parser.add_argument("--matched-negatives", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def rank_fraction(values: np.ndarray) -> np.ndarray:
    return stats.rankdata(np.asarray(values, dtype=float), method="average") / len(values)


def bigwig_mean(path: Path, peaks: pd.DataFrame) -> np.ndarray:
    try:
        import pyBigWig
    except ImportError as error:
        raise ImportError(
            "pyBigWig is needed only when the cached MEOX2 peak annotations do not exist"
        ) from error
    handle = pyBigWig.open(str(path))
    result = np.zeros(len(peaks), dtype=np.float32)
    for index, (chrom, start, end) in enumerate(
        peaks[["chrom", "start", "end"]].itertuples(index=False, name=None)
    ):
        value = handle.stats(chrom, int(start), int(end), type="mean")[0]
        result[index] = 0.0 if value is None or not np.isfinite(value) else value
    handle.close()
    return result


def parse_gene_tss(path: Path) -> dict[str, np.ndarray]:
    result: dict[str, list[int]] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            chrom, start, end, strand = fields[0], int(fields[3]), int(fields[4]), fields[6]
            tss = start - 1 if strand == "+" else end
            result.setdefault(chrom, []).append(tss)
    return {chrom: np.sort(np.asarray(values, dtype=int)) for chrom, values in result.items()}


def nearest_tss_distance(peaks: pd.DataFrame, tss: dict[str, np.ndarray]) -> np.ndarray:
    distance = np.full(len(peaks), np.nan, dtype=float)
    centers = ((peaks["start"].to_numpy(int) + peaks["end"].to_numpy(int)) // 2)
    for chrom, group_indices in peaks.groupby("chrom", sort=False).groups.items():
        coordinates = tss.get(str(chrom))
        if coordinates is None or len(coordinates) == 0:
            continue
        indices = np.asarray(list(group_indices), dtype=int)
        query = centers[indices]
        insertion = np.searchsorted(coordinates, query)
        left = coordinates[np.maximum(insertion - 1, 0)]
        right = coordinates[np.minimum(insertion, len(coordinates) - 1)]
        distance[indices] = np.minimum(np.abs(query - left), np.abs(query - right))
    finite = np.isfinite(distance)
    fill = float(np.nanmedian(distance[finite])) if finite.any() else 0.0
    distance[~finite] = fill
    return distance


def standardized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / max(values.std(), 1e-8)


def match_negatives(
    positive: np.ndarray,
    candidate: np.ndarray,
    peaks: pd.DataFrame,
    ratio: int,
) -> np.ndarray:
    positive_indices = np.flatnonzero(positive)
    candidate_indices = np.flatnonzero(candidate & ~positive)
    features = np.column_stack(
        [
            standardized(np.log1p(peaks["atlas_prevalence"].to_numpy(float) * 1000.0)),
            standardized(np.log1p(peaks["width"].to_numpy(float))),
            standardized(np.log1p(peaks["nearest_tss_distance"].to_numpy(float))),
        ]
    )
    neighbours = NearestNeighbors(
        n_neighbors=min(max(ratio * 8, 24), len(candidate_indices)), n_jobs=-1
    ).fit(features[candidate_indices])
    proposed = neighbours.kneighbors(features[positive_indices], return_distance=False)
    used: set[int] = set()
    chosen: list[int] = []
    for row in proposed:
        local_count = 0
        for local_index in row:
            index = int(candidate_indices[local_index])
            if index in used:
                continue
            used.add(index)
            chosen.append(index)
            local_count += 1
            if local_count == ratio:
                break
    return np.asarray(chosen, dtype=int)


def majority_branch(neighbour_labels: np.ndarray) -> np.ndarray:
    anterior_count = np.sum(neighbour_labels == ANTERIOR, axis=1)
    posterior_count = np.sum(neighbour_labels == POSTERIOR, axis=1)
    threshold = neighbour_labels.shape[1] // 2 + 1
    result = np.full(len(neighbour_labels), "unassigned", dtype=object)
    result[anterior_count >= threshold] = "anterior"
    result[posterior_count >= threshold] = "posterior"
    return result.astype(str)


def weighted_binary_profile(
    raw_atac: ad.AnnData,
    neighbour_rows: np.ndarray,
    particle_mask: np.ndarray,
    particle_weights: np.ndarray,
    peak_positions: np.ndarray,
) -> np.ndarray:
    particle_mask = np.asarray(particle_mask, dtype=bool)
    if not particle_mask.any():
        return np.full(len(peak_positions), np.nan)
    selected_weights = np.asarray(particle_weights[particle_mask], dtype=float)
    selected_weights = np.maximum(selected_weights, 0.0)
    if selected_weights.sum() <= 0:
        selected_weights = np.ones_like(selected_weights)
    selected_weights /= selected_weights.sum()
    k = neighbour_rows.shape[1]
    rows = np.asarray(neighbour_rows[particle_mask], dtype=int).ravel()
    row_weights = np.repeat(selected_weights / k, k)
    unique_rows, inverse = np.unique(rows, return_inverse=True)
    aggregate_weights = np.bincount(inverse, weights=row_weights, minlength=len(unique_rows))
    block = raw_atac.X[unique_rows, :][:, peak_positions]
    if sparse.issparse(block):
        block = block.copy()
        block.data = np.ones_like(block.data)
        return np.asarray(aggregate_weights @ block).ravel()
    return np.asarray(aggregate_weights @ (np.asarray(block) > 0)).ravel()


def observed_binary_profile(
    raw_atac: ad.AnnData,
    rows: np.ndarray,
    peak_positions: np.ndarray,
) -> np.ndarray:
    block = raw_atac.X[np.asarray(rows, dtype=int), :][:, peak_positions]
    if sparse.issparse(block):
        block = block.copy()
        block.data = np.ones_like(block.data)
        return np.asarray(block.mean(axis=0)).ravel()
    return np.asarray(np.asarray(block) > 0).mean(axis=0)


def family_metrics(
    anterior_profile: np.ndarray,
    posterior_profile: np.ndarray,
    observed_anterior: np.ndarray,
    observed_posterior: np.ndarray,
    anterior_positions: np.ndarray,
    posterior_positions: np.ndarray,
) -> dict[str, float]:
    predicted_contrast = anterior_profile - posterior_profile
    observed_contrast = observed_anterior - observed_posterior
    indices = np.concatenate([anterior_positions, posterior_positions])
    labels = np.concatenate(
        [np.ones(len(anterior_positions), dtype=int), np.zeros(len(posterior_positions), dtype=int)]
    )
    predicted_values = predicted_contrast[indices]
    observed_values = observed_contrast[indices]
    predicted_margin = float(
        predicted_contrast[anterior_positions].mean()
        - predicted_contrast[posterior_positions].mean()
    )
    observed_margin = float(
        observed_contrast[anterior_positions].mean()
        - observed_contrast[posterior_positions].mean()
    )
    return {
        "external_auroc": float(roc_auc_score(labels, predicted_values)),
        "observed_auroc": float(roc_auc_score(labels, observed_values)),
        "predicted_margin": predicted_margin,
        "observed_margin": observed_margin,
        "margin_absolute_error": abs(predicted_margin - observed_margin),
        "peak_contrast_spearman": float(stats.spearmanr(predicted_values, observed_values).statistic),
        "peak_contrast_rmse": float(np.sqrt(np.mean((predicted_values - observed_values) ** 2))),
    }


def matched_background_metrics(
    anterior_profile: np.ndarray,
    posterior_profile: np.ndarray,
    observed_anterior: np.ndarray,
    observed_posterior: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    branch: str,
) -> dict[str, float]:
    correct = anterior_profile if branch == "anterior" else posterior_profile
    opposite = posterior_profile if branch == "anterior" else anterior_profile
    observed_correct = observed_anterior if branch == "anterior" else observed_posterior
    observed_opposite = observed_posterior if branch == "anterior" else observed_anterior
    labels = np.concatenate(
        [np.ones(len(positive), dtype=int), np.zeros(len(negative), dtype=int)]
    )
    selected = np.concatenate([positive, negative])
    correct_enrichment = float(correct[positive].mean() - correct[negative].mean())
    opposite_enrichment = float(opposite[positive].mean() - opposite[negative].mean())
    observed_correct_enrichment = float(
        observed_correct[positive].mean() - observed_correct[negative].mean()
    )
    observed_opposite_enrichment = float(
        observed_opposite[positive].mean() - observed_opposite[negative].mean()
    )
    specificity = correct_enrichment - opposite_enrichment
    observed_specificity = observed_correct_enrichment - observed_opposite_enrichment
    return {
        "correct_branch_auroc": float(roc_auc_score(labels, correct[selected])),
        "correct_branch_enrichment": correct_enrichment,
        "opposite_branch_enrichment": opposite_enrichment,
        "branch_specificity": specificity,
        "observed_branch_specificity": observed_specificity,
        "branch_specificity_absolute_error": abs(specificity - observed_specificity),
    }


def configure_plot() -> None:
    apply_nature_rc(font_size=10)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("pdf", "png", "svg"):
        path = output / f"{stem}.{suffix}"
        kwargs: dict[str, object] = {"bbox_inches": "tight", "pad_inches": 0.02}
        if suffix == "png":
            kwargs["dpi"] = 400
        fig.savefig(path, **kwargs)
    plt.close(fig)


def shared_plot_limits(metrics: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    limits: dict[tuple[str, str], tuple[float, float]] = {}
    for family in ("H3K27ac", "TF-bound active CRE"):
        local = metrics[metrics["family"].eq(family)]
        auroc_values = local["external_auroc"].to_numpy(float)
        observed_values = local["observed_auroc"].to_numpy(float)
        lower = max(0.0, min(0.45, float(np.nanmin(auroc_values)) - 0.03))
        upper = min(
            1.0,
            max(float(np.nanmax(auroc_values)), float(np.nanmax(observed_values))) + 0.10,
        )
        limits[(family, "external_auroc")] = (lower, upper)
        error_values = local["margin_absolute_error"].to_numpy(float)
        limits[(family, "margin_absolute_error")] = (
            0.0,
            max(float(np.nanmax(error_values)) * 1.35, 0.005),
        )
    return limits


def plot_scenario(
    metrics: pd.DataFrame,
    scenario: str,
    output: Path,
    limits: dict[tuple[str, str], tuple[float, float]],
    method_order: tuple[str, ...],
) -> None:
    configure_plot()
    stage = SCENARIOS[scenario]["stage"]
    panels = (
        ("H3K27ac", "external_auroc", "H3K27ac AUROC ↑"),
        ("TF-bound active CRE", "external_auroc", "TF CRE AUROC ↑"),
        ("H3K27ac", "margin_absolute_error", "H3K27ac Δ error ↓"),
        ("TF-bound active CRE", "margin_absolute_error", "TF CRE Δ error ↓"),
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(4.13, 3.65),
        gridspec_kw={"hspace": 0.58, "wspace": 0.42},
    )
    y = np.arange(len(method_order))
    for panel_index, (axis, (family, metric, title)) in enumerate(zip(axes.ravel(), panels)):
        local = metrics[
            metrics["scenario"].eq(scenario)
            & metrics["family"].eq(family)
        ].set_index("method")
        values = np.asarray([local.loc[method, metric] for method in method_order], dtype=float)
        for index, (method, value) in enumerate(zip(method_order, values)):
            style = method_style(METHOD_NAMES[method])
            axis.barh(index, value, color=style.color, height=0.58)
            axis.text(value, index, f" {value:.3f}", va="center", ha="left", color="#222222", fontsize=9)
        if metric == "external_auroc":
            observed = float(local["observed_auroc"].iloc[0])
            axis.axvline(observed, color="#555555", lw=0.9, ls="--")
        axis.set_xlim(*limits[(family, metric)])
        axis.set_title(title, pad=3)
        axis.invert_yaxis()
        axis.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
        axis.spines[["top", "right"]].set_visible(False)
        if panel_index % 2 == 0:
            axis.set_yticks(y, method_order)
        else:
            axis.set_yticks(y, [])
    fig.suptitle(f"Held-out {stage}", y=0.995, fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.31, right=0.95, bottom=0.10, top=0.90)
    save_figure(fig, output, f"palate_{scenario}_external_epigenetic_evidence")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    required_outputs = (
        args.output_dir / "external_peak_set_audit.csv",
        args.output_dir / "loo_external_epigenetic_metrics.csv",
        args.output_dir / "matched_background_metrics.csv",
        args.output_dir / "analysis_manifest.json",
    )
    if not args.overwrite and any(path.exists() for path in required_outputs):
        raise FileExistsError("Refusing to overwrite existing external epigenetic outputs")

    prediction_manifest = pd.read_csv(args.input_dir / "prediction_manifest.csv")
    prediction_manifest["canonical_method"] = prediction_manifest["method"].map(MANIFEST_NAMES)
    available_methods = set(prediction_manifest["canonical_method"].dropna())
    required_methods = set(METHOD_ORDER) - {"RNAonly-B"}
    if not required_methods.issubset(available_methods):
        raise ValueError(
            "Prediction manifest does not contain all required methods: "
            f"expected={sorted(required_methods)}, found={sorted(available_methods)}"
        )
    method_order = tuple(method for method in METHOD_ORDER if method in available_methods)
    prediction_manifest = prediction_manifest[
        prediction_manifest["canonical_method"].isin(method_order)
    ].copy()

    rna = ad.read_h5ad(RNA_REFERENCE, backed="r")
    atac = ad.read_h5ad(ATAC_REFERENCE, backed="r")
    raw_atac = ad.read_h5ad(ATAC_RAW, backed="r")
    if not np.array_equal(rna.obs_names, atac.obs_names) or not np.array_equal(
        rna.obs_names, raw_atac.obs_names
    ):
        raise ValueError("Paired RNA, ATAC, and raw ATAC cells are not aligned")
    rna_norm = np.asarray(rna.obsm["X_latent"], dtype=np.float32) / _load_scale(RNA_NORM)
    atac_norm = np.asarray(atac.obsm["X_lsi15"], dtype=np.float32) / _load_scale(ATAC_NORM)
    labels = rna.obs["celltype_sub"].astype(str).to_numpy()
    atlas_times = pd.to_numeric(rna.obs["time_point_processed"], errors="raise").to_numpy(float)

    peaks = pd.read_csv(EXTERNAL_SOURCE / "external_peak_annotations.csv.gz")
    if not np.array_equal(peaks["peak"].astype(str).to_numpy(), raw_atac.var_names.astype(str)):
        raise ValueError("External peak annotation is not aligned to raw ATAC peak order")
    meox_paths = (
        EXTERNAL_SOURCE / "external/GSM7976214_Meox2_1_treat.bigwig",
        EXTERNAL_SOURCE / "external/GSM7976215_Meox2_2_treat.bigwig",
    )
    annotation_cache = args.output_dir / "external_peak_annotations_with_meox2.csv.gz"
    required_external = (GTF,) if annotation_cache.is_file() else (*meox_paths, GTF)
    missing = [str(path) for path in required_external if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing external inputs:\n" + "\n".join(missing))
    if annotation_cache.is_file():
        cached = pd.read_csv(
            annotation_cache,
            usecols=["peak", "meox2_chip_rep1_mean", "meox2_chip_rep2_mean"],
        )
        if not np.array_equal(cached["peak"].astype(str).to_numpy(), peaks["peak"].astype(str).to_numpy()):
            raise ValueError("Cached MEOX2 annotation is not aligned to the raw ATAC peaks")
        meox_values = [
            cached["meox2_chip_rep1_mean"].to_numpy(float),
            cached["meox2_chip_rep2_mean"].to_numpy(float),
        ]
    else:
        meox_values = [bigwig_mean(path, peaks) for path in meox_paths]
    peaks["meox2_chip_rep1_mean"] = meox_values[0]
    peaks["meox2_chip_rep2_mean"] = meox_values[1]
    peaks["nearest_tss_distance"] = nearest_tss_distance(peaks, parse_gene_tss(GTF))

    h3_difference = (
        peaks["gse138721_ap_h3k27ac_mean_rank"].to_numpy(float)
        - peaks["gse138721_pp_h3k27ac_mean_rank"].to_numpy(float)
    )
    h3_upper = float(np.quantile(h3_difference, args.h3_quantile))
    h3_lower = float(np.quantile(h3_difference, 1.0 - args.h3_quantile))
    h3_anterior = (h3_difference >= h3_upper) & (
        peaks["gse138721_ap_h3k27ac_mean_rank"].to_numpy(float) >= 0.5
    )
    h3_posterior = (h3_difference <= h3_lower) & (
        peaks["gse138721_pp_h3k27ac_mean_rank"].to_numpy(float) >= 0.5
    )
    shox_rank = np.minimum(
        rank_fraction(peaks["chip_rep1_mean"].to_numpy(float)),
        rank_fraction(peaks["chip_rep2_mean"].to_numpy(float)),
    )
    meox_rank = np.minimum(rank_fraction(meox_values[0]), rank_fraction(meox_values[1]))
    shox_broad = shox_rank >= args.chip_quantile
    meox_broad = meox_rank >= args.chip_quantile
    shox_active = shox_broad & h3_anterior
    meox_active = meox_broad & h3_posterior

    neutral_candidates = ~(h3_anterior | h3_posterior | shox_broad | meox_broad)
    shox_negative = match_negatives(
        shox_active, neutral_candidates, peaks, args.matched_negatives
    )
    meox_negative = match_negatives(
        meox_active, neutral_candidates & ~np.isin(np.arange(len(peaks)), shox_negative), peaks, args.matched_negatives
    )

    peak_set_rows: list[dict[str, object]] = []
    for family, branch, mask in (
        ("H3K27ac", "anterior", h3_anterior),
        ("H3K27ac", "posterior", h3_posterior),
        ("TF-bound active CRE", "anterior", shox_active),
        ("TF-bound active CRE", "posterior", meox_active),
        ("broad ChIP audit", "anterior", shox_broad),
        ("broad ChIP audit", "posterior", meox_broad),
    ):
        peak_set_rows.append(
            {
                "family": family,
                "branch": branch,
                "role": "positive",
                "n_peaks": int(mask.sum()),
            }
        )
    peak_set_rows.extend(
        [
            {"family": "TF-bound active CRE", "branch": "anterior", "role": "matched_negative", "n_peaks": len(shox_negative)},
            {"family": "TF-bound active CRE", "branch": "posterior", "role": "matched_negative", "n_peaks": len(meox_negative)},
        ]
    )
    pd.DataFrame(peak_set_rows).to_csv(
        args.output_dir / "external_peak_set_audit.csv", index=False
    )

    all_original_positions = np.unique(
        np.concatenate(
            [
                np.flatnonzero(h3_anterior),
                np.flatnonzero(h3_posterior),
                np.flatnonzero(shox_active),
                np.flatnonzero(meox_active),
                shox_negative,
                meox_negative,
                np.flatnonzero(shox_broad),
                np.flatnonzero(meox_broad),
            ]
        )
    )
    local_lookup = np.full(len(peaks), -1, dtype=int)
    local_lookup[all_original_positions] = np.arange(len(all_original_positions))
    families = {
        "H3K27ac": (
            local_lookup[np.flatnonzero(h3_anterior)],
            local_lookup[np.flatnonzero(h3_posterior)],
        ),
        "TF-bound active CRE": (
            local_lookup[np.flatnonzero(shox_active)],
            local_lookup[np.flatnonzero(meox_active)],
        ),
        "broad ChIP audit": (
            local_lookup[np.flatnonzero(shox_broad)],
            local_lookup[np.flatnonzero(meox_broad)],
        ),
    }
    shox_negative_local = local_lookup[shox_negative]
    meox_negative_local = local_lookup[meox_negative]

    metric_rows: list[dict[str, object]] = []
    background_rows: list[dict[str, object]] = []
    classifier_rows: list[dict[str, object]] = []
    for scenario, scenario_info in SCENARIOS.items():
        heldout_time = float(scenario_info["time"])
        heldout_rows = np.flatnonzero(np.isclose(atlas_times, heldout_time))
        training_rows = np.flatnonzero(~np.isclose(atlas_times, heldout_time))
        rna_nn = NearestNeighbors(n_neighbors=args.knn_k, n_jobs=-1).fit(rna_norm[training_rows])
        atac_nn = NearestNeighbors(n_neighbors=args.knn_k, n_jobs=-1).fit(atac_norm[training_rows])

        observed_anterior_rows = heldout_rows[labels[heldout_rows] == ANTERIOR]
        observed_posterior_rows = heldout_rows[labels[heldout_rows] == POSTERIOR]
        observed_anterior = observed_binary_profile(
            raw_atac, observed_anterior_rows, all_original_positions
        )
        observed_posterior = observed_binary_profile(
            raw_atac, observed_posterior_rows, all_original_positions
        )

        local_manifest = prediction_manifest[prediction_manifest["scenario"].eq(scenario)]
        for method in method_order:
            row = local_manifest[local_manifest["canonical_method"].eq(method)].iloc[0]
            prediction_path = Path(row["prediction_file"])
            if not prediction_path.is_absolute():
                prediction_path = args.input_dir / prediction_path
            with np.load(prediction_path, allow_pickle=False) as saved:
                predicted_rna = np.asarray(saved["rna_norm"], dtype=np.float32)
                predicted_atac = np.asarray(saved["atac_norm"], dtype=np.float32)
                weights = np.asarray(saved["weights"], dtype=float)
            rna_neighbours_local = rna_nn.kneighbors(predicted_rna, return_distance=False)
            rna_neighbours = training_rows[rna_neighbours_local]
            predicted_branch = majority_branch(labels[rna_neighbours])
            atac_neighbours_local = atac_nn.kneighbors(predicted_atac, return_distance=False)
            atac_neighbours = training_rows[atac_neighbours_local]
            anterior_mask = predicted_branch == "anterior"
            posterior_mask = predicted_branch == "posterior"
            classifier_rows.append(
                {
                    "scenario": scenario,
                    "heldout_stage": scenario_info["stage"],
                    "method": method,
                    "n_particles": len(predicted_branch),
                    "n_anterior": int(anterior_mask.sum()),
                    "n_posterior": int(posterior_mask.sum()),
                    "n_unassigned": int(np.sum(predicted_branch == "unassigned")),
                }
            )
            predicted_anterior = weighted_binary_profile(
                raw_atac,
                atac_neighbours,
                anterior_mask,
                weights,
                all_original_positions,
            )
            predicted_posterior = weighted_binary_profile(
                raw_atac,
                atac_neighbours,
                posterior_mask,
                weights,
                all_original_positions,
            )
            for family, (anterior_positions, posterior_positions) in families.items():
                values = family_metrics(
                    predicted_anterior,
                    predicted_posterior,
                    observed_anterior,
                    observed_posterior,
                    anterior_positions,
                    posterior_positions,
                )
                metric_rows.append(
                    {
                        "scenario": scenario,
                        "heldout_stage": scenario_info["stage"],
                        "method": method,
                        "family": family,
                        **values,
                    }
                )
            for regulator, positive, negative, branch in (
                (
                    "SHOX2",
                    families["TF-bound active CRE"][0],
                    shox_negative_local,
                    "anterior",
                ),
                (
                    "MEOX2",
                    families["TF-bound active CRE"][1],
                    meox_negative_local,
                    "posterior",
                ),
            ):
                values = matched_background_metrics(
                    predicted_anterior,
                    predicted_posterior,
                    observed_anterior,
                    observed_posterior,
                    positive,
                    negative,
                    branch,
                )
                background_rows.append(
                    {
                        "scenario": scenario,
                        "heldout_stage": scenario_info["stage"],
                        "method": method,
                        "regulator": regulator,
                        "expected_branch": branch,
                        **values,
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    background = pd.DataFrame(background_rows)
    classifier = pd.DataFrame(classifier_rows)
    metrics.to_csv(args.output_dir / "loo_external_epigenetic_metrics.csv", index=False)
    background.to_csv(args.output_dir / "matched_background_metrics.csv", index=False)
    classifier.to_csv(args.output_dir / "predicted_branch_audit.csv", index=False)
    peaks.assign(
        h3_anterior_specific=h3_anterior,
        h3_posterior_specific=h3_posterior,
        shox2_chip_rank=shox_rank,
        meox2_chip_rank=meox_rank,
        shox2_bound_anterior_active=shox_active,
        meox2_bound_posterior_active=meox_active,
    ).to_csv(
        args.output_dir / "external_peak_annotations_with_meox2.csv.gz",
        index=False,
        compression="gzip",
    )

    manifest = {
        "input_prediction_manifest": str((args.input_dir / "prediction_manifest.csv").resolve()),
        "evaluation_policy": "ATAC LSI15 for every method comes from the same frozen full-data T; peak readout and RNA-only branch classifier use only non-held-out atlas cells",
        "knn_k": args.knn_k,
        "method_order": list(method_order),
        "h3_quantile": args.h3_quantile,
        "chip_quantile": args.chip_quantile,
        "peak_sets": {
            "h3_anterior": int(h3_anterior.sum()),
            "h3_posterior": int(h3_posterior.sum()),
            "shox2_broad_chip": int(shox_broad.sum()),
            "meox2_broad_chip": int(meox_broad.sum()),
            "shox2_bound_anterior_active": int(shox_active.sum()),
            "meox2_bound_posterior_active": int(meox_active.sum()),
        },
        "background_matching": {
            "ratio": args.matched_negatives,
            "features": ["atlas detection prevalence", "peak width", "distance to nearest gene TSS"],
            "GC_content": "not present in the supplied peak metadata; not used",
        },
        "important_boundaries": [
            "Broad SHOX2/MEOX2 ChIP peaks are an audit only because their observed held-out scATAC branch AUROC is close to 0.5.",
            "The TF-bound active CRE benchmark intersects ChIP support with an independently fixed H3K27ac branch direction.",
            "T mapping supplies ATAC latent states; kNN is only a common molecular readout from LSI15 to fixed peaks.",
            "This validates branch-specific chromatin-state recovery, not clonal ancestry or causal regulation.",
        ],
    }
    with (args.output_dir / "analysis_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)

    plot_limits = shared_plot_limits(metrics)
    for scenario in SCENARIOS:
        plot_scenario(metrics, scenario, args.output_dir, plot_limits, method_order)

    display = metrics[metrics["family"].isin(["H3K27ac", "TF-bound active CRE"])].pivot_table(
        index=["heldout_stage", "method"],
        columns="family",
        values=["external_auroc", "margin_absolute_error"],
    )
    print(display.to_string(float_format=lambda value: f"{value:.3f}"), flush=True)


if __name__ == "__main__":
    main()
