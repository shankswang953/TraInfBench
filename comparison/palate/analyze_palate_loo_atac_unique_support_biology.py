#!/usr/bin/env python
"""Biological annotation of ATAC states uniquely covered in palate strict LOO.

For every real held-out ATAC cell, this script recomputes the same mass-aware
15NN target-support hit probability used by the revised six-metric benchmark.
It then asks which biological regulatory programs characterize cells covered
by COATI but missed by each external method.

Coverage is not defined from a cell-type label or biological program.  At the
primary threshold, a target cell is covered when the expected probability of
at least one hit from an effective-size sample of the predicted distribution is
at least 0.5.  Thresholds 0.25 and 0.75 are retained as sensitivity analyses.

Biological programs are frozen independently of method coverage:

* anterior/posterior H3K27ac-specific peaks;
* SHOX2/MEOX2-bound active CREs;
* author peak--gene links for five palate lineage marker programs;
* available literature-defined TF motif programs.

For each COATI-versus-baseline comparison, COATI-only cells are matched 1:1 to
cells covered by both methods, exactly by cell type and nearest in ATAC depth
and local LSI density.  The primary summary is the matched difference in a
within-stage, within-cell-type standardized regulatory-program score.
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
import json
import os
import zlib
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from analyze_palate_shox2_1nn_regulation import ATAC_RAW
from analyze_palate_usot_fixed_lineage_multiome import (
    LINEAGE_PROGRAMS,
    MOTIF_FILE,
    PEAK_LINKS,
)
from evaluate_palate_loo_same_space import _load_scale
from plot_palate_strict_loo_six_metrics_revised import (
    local_predicted_mass,
    normalized_weights,
    sha256,
    target_radii,
)
from trainfbench_plot_style import (
    PALATE_CELLTYPE_COLORS,
    PALATE_CELLTYPE_LABELS,
    PALATE_CELLTYPE_ORDER,
    apply_nature_rc,
)


ROOT = Path(__file__).resolve().parents[2]
ATAC_BENCHMARK = ROOT / "data/palate_atac_benchmark.h5ad"
ATAC_NORM = ROOT / "data/palate_atac_secondary_norm_params_lsi15.pt"
DEFAULT_METRICS = (
    ROOT
    / "results/palate_strict_loo_six_metrics_revised_tn_forward_native"
    / "revised_six_metric_scores.csv"
)
EXTERNAL_ANNOTATION = (
    ROOT
    / "results/palate_strict_loo_external_epigenetic_evidence"
    / "external_peak_annotations_with_meox2.csv.gz"
)
DEFAULT_OUTPUT = ROOT / "results/palate_loo_atac_unique_support_biology"

SCENARIOS = {
    "loo_time1": {"stage": "E13.5", "time": 1.0},
    "loo_time2": {"stage": "E14.0", "time": 1.5},
}
METHOD_CANONICAL = {
    "COATI-B": "COATI-B",
    "COATI-U": "COATI-U",
    "CytoBridge balanced": "CytoBridge-B",
    "CytoBridge unbalanced": "CytoBridge-U",
    "MIOFlow": "MIOFlow",
    "TrajectoryNet": "TrajectoryNet",
}
METHOD_ORDER = (
    "COATI-B",
    "COATI-U",
    "CytoBridge-B",
    "CytoBridge-U",
    "MIOFlow",
    "TrajectoryNet",
)
EXTERNAL_METHODS = ("CytoBridge-B", "CytoBridge-U", "MIOFlow", "TrajectoryNet")
COATI_METHODS = ("COATI-B", "COATI-U")
PRIMARY_COMPARISONS = (
    ("COATI-B", "CytoBridge-B"),
    ("COATI-B", "MIOFlow"),
    ("COATI-B", "TrajectoryNet"),
    ("COATI-B", "External union"),
    ("COATI-U", "CytoBridge-U"),
    ("COATI-U", "MIOFlow"),
    ("COATI-U", "TrajectoryNet"),
    ("COATI-U", "External union"),
)
ALL_PAIRWISE_COMPARISONS = tuple(
    (coati, baseline) for coati in COATI_METHODS for baseline in EXTERNAL_METHODS
) + tuple((coati, "External union") for coati in COATI_METHODS)
INTERMEDIATE = "intermidiate cells"
CNC = "CNC-derived progenitors"

CATEGORY_ORDER = ("COATI only", "both", "baseline only", "neither")
CATEGORY_COLORS = {
    "COATI only": "#0072B2",
    "both": "#8C8C8C",
    "baseline only": "#D55E00",
    "neither": "#D9D9D9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--distance-batch-size", type=int, default=256)
    parser.add_argument("--primary-threshold", type=float, default=0.5)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=(0.25, 0.5, 0.75)
    )
    parser.add_argument("--program-high-quantile", type=float, default=0.8)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_prediction_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def support_hit_probability(local_mass: np.ndarray, effective_n: float) -> np.ndarray:
    clipped = np.minimum(np.maximum(local_mass, 0.0), 1.0 - 1e-15)
    return -np.expm1(effective_n * np.log1p(-clipped))


def binary_fraction(matrix: object, positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=int)
    if len(positions) == 0:
        return np.full(matrix.shape[0], np.nan, dtype=np.float32)
    block = matrix[:, positions]
    if sparse.issparse(block):
        block = block.tocsr(copy=True)
        block.data = np.ones_like(block.data)
        return np.asarray(block.mean(axis=1)).ravel().astype(np.float32)
    return np.asarray(np.asarray(block) > 0).mean(axis=1).astype(np.float32)


def dense_columns(matrix: object, positions: list[int]) -> np.ndarray:
    block = matrix[:, positions]
    if sparse.issparse(block):
        return np.asarray(block.toarray(), dtype=np.float32)
    return np.asarray(block, dtype=np.float32)


def load_support_scores(
    metrics_path: Path,
    knn_k: int,
    batch_size: int,
) -> tuple[pd.DataFrame, ad.AnnData]:
    metrics = pd.read_csv(metrics_path)
    metrics["canonical_method"] = metrics["method"].map(METHOD_CANONICAL)
    metrics = metrics[metrics["canonical_method"].isin(METHOD_ORDER)].copy()
    atac = ad.read_h5ad(ATAC_BENCHMARK)
    time = pd.to_numeric(atac.obs["time_point_processed"], errors="raise").to_numpy(float)
    scale = _load_scale(ATAC_NORM)
    atac_norm = np.asarray(atac.obsm["X_lsi15"], dtype=np.float32) / scale
    frames: list[pd.DataFrame] = []
    for scenario, info in SCENARIOS.items():
        heldout = np.flatnonzero(np.isclose(time, float(info["time"])))
        reference = atac_norm[heldout]
        radii = target_radii(reference, knn_k)
        frame = pd.DataFrame(
            {
                "scenario": scenario,
                "stage": str(info["stage"]),
                "global_row": heldout,
                "cell_id": atac.obs_names.to_numpy(str)[heldout],
                "celltype": atac.obs["celltype_sub"].astype(str).to_numpy()[heldout],
                "atac_depth": pd.to_numeric(
                    atac.obs["nCount_ATAC"], errors="coerce"
                ).to_numpy(float)[heldout],
                "local_radius": radii,
            }
        )
        local_metrics = metrics[metrics["scenario"].eq(scenario)]
        for method in METHOD_ORDER:
            row = local_metrics[local_metrics["canonical_method"].eq(method)]
            if len(row) != 1:
                raise ValueError(
                    f"Expected one metric row for {scenario}, {method}; got {len(row)}"
                )
            item = row.iloc[0]
            prediction_path = resolve_prediction_path(str(item["prediction_file"]))
            if sha256(prediction_path) != str(item["prediction_sha256"]):
                raise ValueError(f"Prediction checksum mismatch: {prediction_path}")
            with np.load(prediction_path, allow_pickle=False) as saved:
                prediction = np.asarray(saved["atac_norm"], dtype=np.float32)
                weights = normalized_weights(saved["weights"])
            effective_n = float(1.0 / np.sum(np.square(weights)))
            local_mass = local_predicted_mass(
                reference,
                prediction,
                weights,
                radii,
                batch_size,
            )
            frame[f"support_mass__{method}"] = local_mass
            frame[f"hit_probability__{method}"] = support_hit_probability(
                local_mass, effective_n
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), atac


def build_program_scores(
    atac: ad.AnnData,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = ad.read_h5ad(ATAC_RAW)
    motif = ad.read_h5ad(MOTIF_FILE)
    expected_names = atac.obs_names.to_numpy(str)
    if not np.array_equal(expected_names, raw.obs_names.to_numpy(str)):
        raise ValueError("Raw ATAC cells are not aligned to benchmark ATAC cells")
    if not np.array_equal(expected_names, motif.obs_names.to_numpy(str)):
        raise ValueError("Motif cells are not aligned to benchmark ATAC cells")

    raw_peak_names = raw.var_names.to_numpy(str)
    peak_lookup = {name: index for index, name in enumerate(raw_peak_names)}
    external = pd.read_csv(EXTERNAL_ANNOTATION)
    if not np.array_equal(external["peak"].astype(str).to_numpy(), raw_peak_names):
        raise ValueError("External peak annotation is not aligned to raw ATAC peaks")

    definitions: list[dict[str, object]] = []
    scores: dict[str, np.ndarray] = {}

    h3_anterior = np.flatnonzero(
        external["h3_anterior_specific"].astype(bool).to_numpy()
    )
    h3_posterior = np.flatnonzero(
        external["h3_posterior_specific"].astype(bool).to_numpy()
    )
    shox_cre = np.flatnonzero(
        external["shox2_bound_anterior_active"].astype(bool).to_numpy()
    )
    meox_cre = np.flatnonzero(
        external["meox2_bound_posterior_active"].astype(bool).to_numpy()
    )

    h3_a = binary_fraction(raw.X, h3_anterior)
    h3_p = binary_fraction(raw.X, h3_posterior)
    shox = binary_fraction(raw.X, shox_cre)
    meox = binary_fraction(raw.X, meox_cre)
    external_programs = {
        "H3K27ac anterior": (h3_a - h3_p, len(h3_anterior) + len(h3_posterior)),
        "H3K27ac posterior": (h3_p - h3_a, len(h3_anterior) + len(h3_posterior)),
        "SHOX2-bound CRE": (shox - meox, len(shox_cre) + len(meox_cre)),
        "MEOX2-bound CRE": (meox - shox, len(shox_cre) + len(meox_cre)),
    }
    for name, (values, n_features) in external_programs.items():
        scores[name] = values
        definitions.append(
            {
                "program": name,
                "family": "external epigenetic",
                "n_features": n_features,
                "definition": "branch-specific positive peak fraction minus opposite-branch peak fraction",
                "method_independent": True,
            }
        )

    links = pd.read_csv(PEAK_LINKS)
    links = links[
        (pd.to_numeric(links["zscore"], errors="coerce") > 0)
        & (pd.to_numeric(links["pvalue"], errors="coerce") < 0.05)
        & links["peak"].astype(str).isin(peak_lookup)
    ].copy()
    for lineage, spec in LINEAGE_PROGRAMS.items():
        gene_scores: list[np.ndarray] = []
        used_genes: list[str] = []
        used_peaks: set[int] = set()
        for gene in spec["genes"]:
            gene_links = links[links["gene"].astype(str).eq(str(gene))]
            positions = np.asarray(
                [peak_lookup[name] for name in gene_links["peak"].astype(str)],
                dtype=int,
            )
            positions = np.unique(positions)
            if len(positions) == 0:
                continue
            gene_scores.append(binary_fraction(raw.X, positions))
            used_genes.append(str(gene))
            used_peaks.update(positions.tolist())
        if not gene_scores:
            continue
        name = f"Linked peaks: {lineage}"
        scores[name] = np.mean(np.column_stack(gene_scores), axis=1).astype(np.float32)
        definitions.append(
            {
                "program": name,
                "family": "author peak-gene links",
                "n_features": len(used_peaks),
                "n_genes": len(used_genes),
                "genes": ";".join(used_genes),
                "definition": "mean peak accessibility per gene, then equal-weight mean across fixed lineage genes",
                "method_independent": True,
            }
        )

    motif_lookup = {name.lower(): index for index, name in enumerate(motif.var_names.astype(str))}
    for lineage, spec in LINEAGE_PROGRAMS.items():
        requested = [str(name) for name in spec["motifs"]]
        positions = [motif_lookup[name.lower()] for name in requested if name.lower() in motif_lookup]
        if not positions:
            continue
        name = f"Motifs: {lineage}"
        scores[name] = dense_columns(motif.X, positions).mean(axis=1).astype(np.float32)
        definitions.append(
            {
                "program": name,
                "family": "TF motif accessibility",
                "n_features": len(positions),
                "motifs": ";".join(motif.var_names[positions].astype(str)),
                "definition": "equal-weight mean of available fixed lineage-relevant motif scores",
                "method_independent": True,
            }
        )

    frame = pd.DataFrame({"cell_id": expected_names, **scores})
    return frame, pd.DataFrame(definitions)


def add_standardized_programs(
    cell_frame: pd.DataFrame,
    program_names: list[str],
    high_quantile: float,
) -> pd.DataFrame:
    result = cell_frame.copy()
    for program in program_names:
        result[f"z__{program}"] = np.nan
        result[f"high__{program}"] = False
        for (_, _), indices in result.groupby(["stage", "celltype"], sort=False).groups.items():
            index = np.asarray(list(indices), dtype=int)
            values = result.loc[index, program].to_numpy(float)
            finite = np.isfinite(values)
            if not finite.any():
                continue
            mean = float(np.mean(values[finite]))
            sd = float(np.std(values[finite]))
            z = np.zeros(len(values), dtype=float) if sd < 1e-8 else (values - mean) / sd
            threshold = float(np.quantile(values[finite], high_quantile))
            result.loc[index, f"z__{program}"] = z
            result.loc[index, f"high__{program}"] = values >= threshold
    return result


def baseline_probability(frame: pd.DataFrame, baseline: str) -> np.ndarray:
    if baseline == "External union":
        return np.max(
            np.column_stack(
                [frame[f"hit_probability__{method}"].to_numpy(float) for method in EXTERNAL_METHODS]
            ),
            axis=1,
        )
    return frame[f"hit_probability__{baseline}"].to_numpy(float)


def coverage_categories(
    frame: pd.DataFrame,
    coati: str,
    baseline: str,
    threshold: float,
) -> np.ndarray:
    coati_covered = frame[f"hit_probability__{coati}"].to_numpy(float) >= threshold
    baseline_covered = baseline_probability(frame, baseline) >= threshold
    result = np.full(len(frame), "neither", dtype=object)
    result[coati_covered & ~baseline_covered] = "COATI only"
    result[coati_covered & baseline_covered] = "both"
    result[~coati_covered & baseline_covered] = "baseline only"
    return result.astype(str)


def build_group_summaries(
    cells: pd.DataFrame,
    program_names: list[str],
    thresholds: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    count_rows: list[dict[str, object]] = []
    celltype_rows: list[dict[str, object]] = []
    program_rows: list[dict[str, object]] = []
    for scenario, stage_frame in cells.groupby("scenario", sort=False):
        stage_frame = stage_frame[~stage_frame["celltype"].eq(INTERMEDIATE)].copy()
        for threshold in thresholds:
            for coati, baseline in ALL_PAIRWISE_COMPARISONS:
                categories = coverage_categories(stage_frame, coati, baseline, threshold)
                n_total = len(stage_frame)
                for category in CATEGORY_ORDER:
                    member = categories == category
                    count_rows.append(
                        {
                            "scenario": scenario,
                            "stage": stage_frame["stage"].iloc[0],
                            "threshold": threshold,
                            "coati": coati,
                            "baseline": baseline,
                            "category": category,
                            "n_cells": int(member.sum()),
                            "fraction": float(member.mean()),
                        }
                    )
                    for celltype in PALATE_CELLTYPE_ORDER:
                        if celltype == INTERMEDIATE:
                            continue
                        local = member & stage_frame["celltype"].eq(celltype).to_numpy()
                        celltype_rows.append(
                            {
                                "scenario": scenario,
                                "stage": stage_frame["stage"].iloc[0],
                                "threshold": threshold,
                                "coati": coati,
                                "baseline": baseline,
                                "category": category,
                                "celltype": celltype,
                                "n_cells": int(local.sum()),
                                "fraction_within_category": (
                                    float(local.sum() / member.sum()) if member.sum() else np.nan
                                ),
                            }
                        )
                    if member.sum() == 0:
                        continue
                    for program in program_names:
                        values = stage_frame.loc[member, f"z__{program}"].to_numpy(float)
                        high = stage_frame.loc[member, f"high__{program}"].to_numpy(bool)
                        program_rows.append(
                            {
                                "scenario": scenario,
                                "stage": stage_frame["stage"].iloc[0],
                                "threshold": threshold,
                                "coati": coati,
                                "baseline": baseline,
                                "category": category,
                                "program": program,
                                "n_cells": int(member.sum()),
                                "mean_program_z": float(np.nanmean(values)),
                                "program_high_fraction": float(np.mean(high)),
                            }
                        )
    return pd.DataFrame(count_rows), pd.DataFrame(celltype_rows), pd.DataFrame(program_rows)


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pooled = np.sqrt((np.var(a) + np.var(b)) / 2.0)
    return 0.0 if pooled < 1e-12 else float((np.mean(a) - np.mean(b)) / pooled)


def match_coati_only_to_common(
    frame: pd.DataFrame,
    categories: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    treated_all: list[int] = []
    control_all: list[int] = []
    balance_rows: list[dict[str, object]] = []
    log_depth = np.log1p(frame["atac_depth"].to_numpy(float))
    log_radius = np.log1p(frame["local_radius"].to_numpy(float))
    for celltype in pd.unique(frame["celltype"]):
        type_mask = frame["celltype"].eq(celltype).to_numpy()
        treated = np.flatnonzero(type_mask & (categories == "COATI only"))
        controls = np.flatnonzero(type_mask & (categories == "both"))
        if len(treated) == 0 or len(controls) == 0:
            continue
        combined = np.concatenate([treated, controls])
        features = np.column_stack([log_depth[combined], log_radius[combined]])
        mean = features.mean(axis=0)
        sd = features.std(axis=0)
        sd[sd < 1e-8] = 1.0
        standardized = (features - mean) / sd
        treated_features = standardized[: len(treated)]
        control_features = standardized[len(treated) :]
        neighbour = NearestNeighbors(n_neighbors=1).fit(control_features)
        distance, local_index = neighbour.kneighbors(treated_features, return_distance=True)
        matched = controls[local_index.ravel()]
        treated_all.extend(treated.tolist())
        control_all.extend(matched.tolist())
        for covariate, values in (
            ("log1p_atac_depth", log_depth),
            ("log1p_local_radius", log_radius),
        ):
            balance_rows.append(
                {
                    "celltype": celltype,
                    "covariate": covariate,
                    "n_treated": len(treated),
                    "n_common_pool": len(controls),
                    "n_unique_matched_controls": len(np.unique(matched)),
                    "smd_before_matching": standardized_mean_difference(
                        values[treated], values[controls]
                    ),
                    "smd_after_matching": standardized_mean_difference(
                        values[treated], values[matched]
                    ),
                    "median_match_distance": float(np.median(distance)),
                    "maximum_match_distance": float(np.max(distance)),
                }
            )
    return (
        np.asarray(treated_all, dtype=int),
        np.asarray(control_all, dtype=int),
        balance_rows,
    )


def bootstrap_matched_effects(
    treated_values: np.ndarray,
    control_values: np.ndarray,
    treated_high: np.ndarray,
    control_high: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    treated_values = np.asarray(treated_values, dtype=float)
    control_values = np.asarray(control_values, dtype=float)
    treated_high = np.asarray(treated_high, dtype=float)
    control_high = np.asarray(control_high, dtype=float)
    finite = np.isfinite(treated_values) & np.isfinite(control_values)
    treated_values = treated_values[finite]
    control_values = control_values[finite]
    treated_high = treated_high[finite]
    control_high = control_high[finite]
    n = len(treated_values)
    if n == 0:
        return {
            "n_pairs": 0,
            "delta_z": np.nan,
            "delta_z_ci_low": np.nan,
            "delta_z_ci_high": np.nan,
            "risk_difference": np.nan,
            "risk_difference_ci_low": np.nan,
            "risk_difference_ci_high": np.nan,
            "enrichment_ratio": np.nan,
            "enrichment_ratio_ci_low": np.nan,
            "enrichment_ratio_ci_high": np.nan,
        }
    delta = treated_values - control_values
    risk_delta = treated_high - control_high
    treated_fraction = float(np.mean(treated_high))
    control_fraction = float(np.mean(control_high))
    ratio = ((treated_high.sum() + 0.5) / (n + 1.0)) / (
        (control_high.sum() + 0.5) / (n + 1.0)
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(repeats, n))
    boot_delta = delta[indices].mean(axis=1)
    boot_risk = risk_delta[indices].mean(axis=1)
    boot_treated = treated_high[indices].sum(axis=1)
    boot_control = control_high[indices].sum(axis=1)
    boot_ratio = ((boot_treated + 0.5) / (n + 1.0)) / (
        (boot_control + 0.5) / (n + 1.0)
    )
    return {
        "n_pairs": n,
        "n_unique_treated": n,
        "coati_only_high_fraction": treated_fraction,
        "matched_common_high_fraction": control_fraction,
        "delta_z": float(np.mean(delta)),
        "delta_z_ci_low": float(np.quantile(boot_delta, 0.025)),
        "delta_z_ci_high": float(np.quantile(boot_delta, 0.975)),
        "risk_difference": float(np.mean(risk_delta)),
        "risk_difference_ci_low": float(np.quantile(boot_risk, 0.025)),
        "risk_difference_ci_high": float(np.quantile(boot_risk, 0.975)),
        "enrichment_ratio": float(ratio),
        "enrichment_ratio_ci_low": float(np.quantile(boot_ratio, 0.025)),
        "enrichment_ratio_ci_high": float(np.quantile(boot_ratio, 0.975)),
    }


def matched_enrichment(
    cells: pd.DataFrame,
    program_names: list[str],
    thresholds: tuple[float, ...],
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    effect_rows: list[dict[str, object]] = []
    balance_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []
    for scenario, stage_all in cells.groupby("scenario", sort=False):
        stage_all = stage_all[~stage_all["celltype"].eq(INTERMEDIATE)].copy()
        for scope, scope_frame in (
            ("all non-intermediate", stage_all),
            ("CNC only", stage_all[stage_all["celltype"].eq(CNC)].copy()),
        ):
            scope_frame = scope_frame.reset_index(drop=True)
            if len(scope_frame) == 0:
                continue
            for threshold in thresholds:
                for coati, baseline in ALL_PAIRWISE_COMPARISONS:
                    categories = coverage_categories(scope_frame, coati, baseline, threshold)
                    treated, control, local_balance = match_coati_only_to_common(
                        scope_frame, categories
                    )
                    base = {
                        "scenario": scenario,
                        "stage": scope_frame["stage"].iloc[0],
                        "scope": scope,
                        "threshold": threshold,
                        "coati": coati,
                        "baseline": baseline,
                    }
                    for row in local_balance:
                        balance_rows.append({**base, **row})
                    for treated_index, control_index in zip(treated, control):
                        match_rows.append(
                            {
                                **base,
                                "coati_only_cell_id": scope_frame.iloc[treated_index]["cell_id"],
                                "matched_common_cell_id": scope_frame.iloc[control_index]["cell_id"],
                                "celltype": scope_frame.iloc[treated_index]["celltype"],
                            }
                        )
                    if len(treated) == 0:
                        continue
                    for program in program_names:
                        local_seed = (
                            seed
                            + zlib.crc32(
                                f"{scenario}|{scope}|{threshold}|{coati}|{baseline}|{program}".encode()
                            )
                        ) % (2**32)
                        effects = bootstrap_matched_effects(
                            scope_frame.iloc[treated][f"z__{program}"].to_numpy(float),
                            scope_frame.iloc[control][f"z__{program}"].to_numpy(float),
                            scope_frame.iloc[treated][f"high__{program}"].to_numpy(bool),
                            scope_frame.iloc[control][f"high__{program}"].to_numpy(bool),
                            repeats,
                            local_seed,
                        )
                        effect_rows.append(
                            {
                                **base,
                                "program": program,
                                **effects,
                                "positive_delta_supported": bool(
                                    int(effects["n_pairs"]) >= 20
                                    and np.isfinite(effects["delta_z_ci_low"])
                                    and effects["delta_z_ci_low"] > 0
                                ),
                            }
                        )
    return pd.DataFrame(effect_rows), pd.DataFrame(balance_rows), pd.DataFrame(match_rows)


def matched_enrichment_by_celltype(
    cells: pd.DataFrame,
    matches: pd.DataFrame,
    program_names: list[str],
    primary_threshold: float,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    lookup = cells.drop_duplicates("cell_id").set_index("cell_id")
    local_matches = matches[
        matches["scope"].eq("all non-intermediate")
        & np.isclose(matches["threshold"], primary_threshold)
    ].copy()
    rows: list[dict[str, object]] = []
    keys = ["scenario", "stage", "coati", "baseline", "celltype"]
    for key, group in local_matches.groupby(keys, sort=False):
        base = dict(zip(keys, key))
        treated_ids = group["coati_only_cell_id"].astype(str).to_numpy()
        control_ids = group["matched_common_cell_id"].astype(str).to_numpy()
        for program in program_names:
            local_seed = (
                seed
                + zlib.crc32(
                    ("|".join(str(value) for value in key) + f"|{program}").encode()
                )
            ) % (2**32)
            effects = bootstrap_matched_effects(
                lookup.loc[treated_ids, f"z__{program}"].to_numpy(float),
                lookup.loc[control_ids, f"z__{program}"].to_numpy(float),
                lookup.loc[treated_ids, f"high__{program}"].to_numpy(bool),
                lookup.loc[control_ids, f"high__{program}"].to_numpy(bool),
                repeats,
                local_seed,
            )
            rows.append(
                {
                    **base,
                    "threshold": primary_threshold,
                    "program": program,
                    **effects,
                    "positive_delta_supported": bool(
                        int(effects["n_pairs"]) >= 20
                        and np.isfinite(effects["delta_z_ci_low"])
                        and effects["delta_z_ci_low"] > 0
                    ),
                }
            )
    return pd.DataFrame(rows)


def configure_plot() -> None:
    apply_nature_rc(font_size=10.0)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "font.weight": "normal",
            "axes.titlesize": 10.0,
            "axes.titleweight": "normal",
            "axes.labelsize": 10.0,
            "axes.labelweight": "normal",
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("pdf", "png", "svg"):
        kwargs: dict[str, object] = {"bbox_inches": "tight", "pad_inches": 0.03}
        if suffix == "png":
            kwargs["dpi"] = 400
        fig.savefig(output / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def comparison_label(coati: str, baseline: str) -> str:
    short = {
        "CytoBridge-B": "CB-B",
        "CytoBridge-U": "CB-U",
        "MIOFlow": "MIO",
        "TrajectoryNet": "TN",
        "External union": "all ext.",
    }
    return f"{coati.replace('COATI-', '')}/{short[baseline]}"


def plot_group_fractions(
    counts: pd.DataFrame,
    output: Path,
    threshold: float,
) -> None:
    configure_plot()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25), sharex=True, sharey=True)
    labels = [comparison_label(*item) for item in PRIMARY_COMPARISONS]
    y = np.arange(len(PRIMARY_COMPARISONS))
    for axis, (scenario, info) in zip(axes, SCENARIOS.items()):
        local = counts[
            counts["scenario"].eq(scenario) & np.isclose(counts["threshold"], threshold)
        ]
        left = np.zeros(len(PRIMARY_COMPARISONS), dtype=float)
        for category in CATEGORY_ORDER:
            values = []
            for coati, baseline in PRIMARY_COMPARISONS:
                row = local[
                    local["coati"].eq(coati)
                    & local["baseline"].eq(baseline)
                    & local["category"].eq(category)
                ]
                values.append(float(row.iloc[0]["fraction"]))
            values_array = np.asarray(values)
            axis.barh(
                y,
                values_array,
                left=left,
                height=0.62,
                color=CATEGORY_COLORS[category],
                edgecolor="white",
                linewidth=0.35,
                label=category,
            )
            left += values_array
        axis.set_title(str(info["stage"]))
        axis.set_xlim(0, 1)
        axis.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.25))
        axis.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
        axis.grid(axis="x", color="#DDDDDD", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0, pad=2)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.supxlabel("Fraction of real held-out ATAC cells", y=0.01)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.15, top=0.82, wspace=0.12)
    save_figure(fig, output, "palate_atac_unique_support_group_fractions")


def program_label(name: str) -> str:
    return (
        name.replace("Linked peaks: ", "Peaks: ")
        .replace("Motifs: ", "Motif: ")
        .replace("H3K27ac ", "H3: ")
        .replace("-bound CRE", " CRE")
    )


def plot_enrichment_heatmap(
    effects: pd.DataFrame,
    output: Path,
    threshold: float,
    scope: str,
    stem: str,
) -> None:
    configure_plot()
    primary_programs = [
        "H3K27ac anterior",
        "H3K27ac posterior",
        "SHOX2-bound CRE",
        "MEOX2-bound CRE",
        "Linked peaks: anterior",
        "Linked peaks: posterior",
        "Linked peaks: dental",
        "Linked peaks: osteogenic",
        "Linked peaks: perimysial",
        "Motifs: anterior",
        "Motifs: posterior",
        "Motifs: dental",
        "Motifs: osteogenic",
        "Motifs: perimysial",
    ]
    available = set(effects["program"].astype(str))
    programs = [name for name in primary_programs if name in available]
    comparisons = list(PRIMARY_COMPARISONS)
    subset = effects[
        effects["scope"].eq(scope) & np.isclose(effects["threshold"], threshold)
    ]
    matrices: list[np.ndarray] = []
    support_matrices: list[np.ndarray] = []
    n_matrices: list[np.ndarray] = []
    for scenario in SCENARIOS:
        matrix = np.full((len(programs), len(comparisons)), np.nan)
        supported = np.zeros_like(matrix, dtype=bool)
        n_pairs = np.zeros_like(matrix, dtype=int)
        local = subset[subset["scenario"].eq(scenario)]
        for row_index, program in enumerate(programs):
            for column_index, (coati, baseline) in enumerate(comparisons):
                row = local[
                    local["program"].eq(program)
                    & local["coati"].eq(coati)
                    & local["baseline"].eq(baseline)
                ]
                if len(row) != 1:
                    continue
                n_pairs[row_index, column_index] = int(row.iloc[0]["n_pairs"])
                if n_pairs[row_index, column_index] < 20:
                    continue
                matrix[row_index, column_index] = float(row.iloc[0]["delta_z"])
                supported[row_index, column_index] = bool(
                    row.iloc[0]["positive_delta_supported"]
                )
        matrices.append(matrix)
        support_matrices.append(supported)
        n_matrices.append(n_pairs)
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    limit = max(0.25, float(np.quantile(np.abs(finite), 0.95))) if len(finite) else 1.0
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, max(4.7, 0.29 * len(programs) + 1.6)),
        sharex=True,
        sharey=True,
    )
    image = None
    for axis, (scenario, info), matrix, supported, n_pairs in zip(
        axes, SCENARIOS.items(), matrices, support_matrices, n_matrices
    ):
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        for row_index in range(len(programs)):
            for column_index in range(len(comparisons)):
                value = matrix[row_index, column_index]
                if n_pairs[row_index, column_index] > 0 and n_pairs[row_index, column_index] < 20:
                    text = f"n={n_pairs[row_index, column_index]}"
                elif not np.isfinite(value):
                    text = "NA"
                else:
                    text = f"{value:.2f}{'*' if supported[row_index, column_index] else ''}"
                axis.text(
                    column_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="#222222",
                )
        axis.set_title(str(info["stage"]))
        axis.set_xticks(
            np.arange(len(comparisons)),
            [comparison_label(*item) for item in comparisons],
            rotation=55,
            ha="right",
        )
        axis.set_yticks(np.arange(len(programs)), [program_label(name) for name in programs])
        axis.set_xticks(np.arange(-0.5, len(comparisons), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(programs), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.7)
        axis.tick_params(which="minor", bottom=False, left=False)
    color_axis = fig.add_axes([0.925, 0.22, 0.015, 0.55])
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_label("Matched difference in program z-score", rotation=90, labelpad=7)
    fig.text(
        0.51,
        0.015,
        "COATI-only minus matched common-covered cells; * 95% bootstrap CI > 0",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.25, right=0.90, bottom=0.19, top=0.94, wspace=0.08)
    save_figure(fig, output, stem)


def plot_external_union_celltypes(
    celltypes: pd.DataFrame,
    output: Path,
    threshold: float,
) -> None:
    configure_plot()
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 3.8), sharex=True, sharey=True)
    celltype_order = [name for name in PALATE_CELLTYPE_ORDER if name != INTERMEDIATE]
    for row_index, coati in enumerate(COATI_METHODS):
        for column_index, (scenario, info) in enumerate(SCENARIOS.items()):
            axis = axes[row_index, column_index]
            local = celltypes[
                celltypes["scenario"].eq(scenario)
                & np.isclose(celltypes["threshold"], threshold)
                & celltypes["coati"].eq(coati)
                & celltypes["baseline"].eq("External union")
                & celltypes["category"].isin(["COATI only", "both"])
            ]
            y = np.arange(2)
            left = np.zeros(2, dtype=float)
            categories = ("COATI only", "both")
            for celltype in celltype_order:
                values = []
                for category in categories:
                    item = local[
                        local["category"].eq(category)
                        & local["celltype"].eq(celltype)
                    ]
                    values.append(
                        0.0
                        if len(item) == 0 or not np.isfinite(item.iloc[0]["fraction_within_category"])
                        else float(item.iloc[0]["fraction_within_category"])
                    )
                values_array = np.asarray(values)
                axis.barh(
                    y,
                    values_array,
                    left=left,
                    height=0.62,
                    color=PALATE_CELLTYPE_COLORS[celltype],
                    edgecolor="white",
                    linewidth=0.35,
                    label=PALATE_CELLTYPE_LABELS[celltype],
                )
                left += values_array
            axis.set_title(f"{coati} · {info['stage']}")
            axis.set_yticks(y, ["COATI only", "Common"])
            axis.set_xlim(0, 1)
            axis.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
            axis.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.tick_params(axis="y", length=0, pad=2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.0,
    )
    fig.supxlabel("Cell-type fraction", y=0.01)
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.14, top=0.78, hspace=0.55, wspace=0.15)
    save_figure(fig, output, "palate_atac_unique_support_celltypes_vs_all_external")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    required = (
        args.output_dir / "per_cell_atac_support_and_programs.csv.gz",
        args.output_dir / "coverage_group_counts.csv",
        args.output_dir / "matched_program_enrichment.csv",
        args.output_dir / "analysis_manifest.json",
    )
    if not args.overwrite and any(path.exists() for path in required):
        raise FileExistsError("Refusing to overwrite existing unique-support analysis")

    support, atac = load_support_scores(
        args.metrics, args.knn_k, args.distance_batch_size
    )
    programs, definitions = build_program_scores(atac)
    cells = support.merge(programs, on="cell_id", how="left", validate="many_to_one")
    program_names = definitions["program"].astype(str).tolist()
    cells = add_standardized_programs(
        cells, program_names, args.program_high_quantile
    )

    thresholds = tuple(float(value) for value in args.thresholds)
    counts, celltypes, group_programs = build_group_summaries(
        cells, program_names, thresholds
    )
    effects, balance, matches = matched_enrichment(
        cells,
        program_names,
        thresholds,
        args.bootstrap_repeats,
        args.seed,
    )
    celltype_effects = matched_enrichment_by_celltype(
        cells,
        matches,
        program_names,
        args.primary_threshold,
        args.bootstrap_repeats,
        args.seed,
    )

    cells.to_csv(
        args.output_dir / "per_cell_atac_support_and_programs.csv.gz",
        index=False,
        compression="gzip",
    )
    definitions.to_csv(args.output_dir / "program_definitions.csv", index=False)
    counts.to_csv(args.output_dir / "coverage_group_counts.csv", index=False)
    celltypes.to_csv(args.output_dir / "coverage_group_celltypes.csv", index=False)
    group_programs.to_csv(
        args.output_dir / "coverage_group_program_summary.csv", index=False
    )
    effects.to_csv(args.output_dir / "matched_program_enrichment.csv", index=False)
    celltype_effects.to_csv(
        args.output_dir / "matched_program_enrichment_by_celltype.csv", index=False
    )
    balance.to_csv(args.output_dir / "matching_balance.csv", index=False)
    matches.to_csv(args.output_dir / "matched_cell_pairs.csv.gz", index=False, compression="gzip")

    plot_group_fractions(counts, args.output_dir, args.primary_threshold)
    plot_enrichment_heatmap(
        effects,
        args.output_dir,
        args.primary_threshold,
        "all non-intermediate",
        "palate_atac_unique_support_program_enrichment_all_cells",
    )
    plot_enrichment_heatmap(
        effects,
        args.output_dir,
        args.primary_threshold,
        "CNC only",
        "palate_atac_unique_support_program_enrichment_cnc",
    )
    plot_external_union_celltypes(celltypes, args.output_dir, args.primary_threshold)

    primary = effects[
        np.isclose(effects["threshold"], args.primary_threshold)
        & effects["scope"].eq("all non-intermediate")
    ].copy()
    primary["comparison"] = primary.apply(
        lambda row: comparison_label(str(row["coati"]), str(row["baseline"])), axis=1
    )
    primary.sort_values(
        ["stage", "coati", "baseline", "delta_z"],
        ascending=[True, True, True, False],
    ).to_csv(args.output_dir / "primary_enrichment_ranked.csv", index=False)

    manifest = {
        "analysis": "biological annotation of ATAC states uniquely covered in strict LOO",
        "metrics_source": str(args.metrics.resolve()),
        "metrics_sha256": sha256(args.metrics),
        "knn_k": args.knn_k,
        "coverage_score": "per-cell expected hit probability from the same mass-aware 15NN support definition as the revised six-metric benchmark",
        "primary_coverage_threshold": args.primary_threshold,
        "sensitivity_thresholds": list(thresholds),
        "program_high_quantile_within_stage_celltype": args.program_high_quantile,
        "matching": "exact cell type; nearest with replacement in standardized log1p ATAC depth and log1p 15NN radius",
        "bootstrap_repeats": args.bootstrap_repeats,
        "excluded_celltype": INTERMEDIATE,
        "coati_settings": "COATI-B and COATI-U at fixed C_y=0.5 from the exact latest six-metric prediction files",
        "external_methods": list(EXTERNAL_METHODS),
        "important_boundaries": [
            "COATI-only coverage is geometric and mass-aware; biological programs are not used to define coverage.",
            "Program enrichment describes which real held-out ATAC states are additionally represented; it does not establish clonal ancestry or causal regulation.",
            "External-union coverage is the maximum per-cell hit probability across CytoBridge-B, CytoBridge-U, MIOFlow, and TrajectoryNet.",
            "Motif programs use only motifs present in the supplied CNC_motif matrix.",
        ],
    }
    with (args.output_dir / "analysis_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)

    print("Primary matched enrichment, strongest positive programs", flush=True)
    display = primary[
        [
            "stage",
            "comparison",
            "program",
            "n_pairs",
            "delta_z",
            "delta_z_ci_low",
            "delta_z_ci_high",
            "positive_delta_supported",
        ]
    ].sort_values(["stage", "delta_z"], ascending=[True, False])
    print(display.groupby("stage", sort=False).head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
