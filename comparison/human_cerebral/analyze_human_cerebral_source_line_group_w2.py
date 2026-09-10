#!/usr/bin/env python3
"""Source-line-conditioned distribution recovery for human cerebral rollouts.

The analysis follows all 501 common D4 particles, keeps their frozen stem-cell
line label, and evaluates D7--D21 against observed metacells from the matching
line.  Every method is decoded through the same stage-conditioned, line-blind
k=15 barycentric readout before scoring.  MIOFlow neighbours are queried only
in its frozen GAGA10 space; every other RNA model is queried in normalized
PCA30.  ATAC is queried in normalized LSI12.

The reported W2-like score is sqrt(2 * debiased p=2 Sinkhorn divergence), with
blur=0.05 and at most 512 deterministic microclusters, matching the benchmark's
six-metric convention.  It is a regularized surrogate rather than exact W2.
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
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd
import torch
from geomloss import SamplesLoss
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from analyze_human_cerebral_peak_gene_max_correlation import (
    Rollout,
    load_rollout,
    neighbor_weights,
)


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "results/human_cerebral_full_biological_interpretability"
REGISTRY = RESULT_ROOT / "00_input_audit/model_registry.csv"
DEFAULT_OUTPUT = RESULT_ROOT / "09_source_line_group_w2"
HUMAN = Path("external/COATI/humanCerebral")
DATA = HUMAN / "Data/selected_4_7_9_11_12_18_21"
GAGA = (
    ROOT
    / "results/mioflow_human_cerebral_7time_d4_d21_no_d16_full_official_gaga10_n256_30000"
    / "gaga10_embedding.npz"
)

TIME_KEYS = ("age_4", "age_7", "age_9", "age_11", "age_12", "age_18", "age_21")
TIME_LABELS = ("D4", "D7", "D9", "D11", "D12", "D18", "D21")
PHYSICAL_TIMES = np.asarray((0.0, 0.3, 0.5, 0.7, 0.8, 1.4, 1.7), dtype=float)
OBSERVED_INDICES = np.asarray((0, 11, 19, 26, 30, 53, 65), dtype=int)
LINES = ("409b2", "h9", "hoik1", "wibj2")
KNN_K = 15

PRIMARY_MODELS = (
    ("coati_sync_balanced_cy0.5_s0_i30000", "COATI bal."),
    ("coati_sync_unbalanced_cy0.5_s0_i40000", "COATI unbal."),
    ("cytobridge_balanced_s42_i30000", "CytoBridge bal."),
    ("cytobridge_unbalanced_biological_prior_s42_i30000", "CytoBridge unbal."),
    ("mioflow_gaga10_balanced_s42_i30000", "MIOFlow"),
    ("trajectorynet_forward_balanced_s0_i30000", "TrajectoryNet"),
    ("coati_rna_only_balanced_s0_i30000", "OT(RNA)"),
    ("coati_rna_only_unbalanced_biological_prior_s0_i30000", "UOT(RNA)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sinkhorn-clusters", type=int, default=512)
    parser.add_argument("--sinkhorn-blur", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Reference:
    obs: pd.DataFrame
    ids: np.ndarray
    source_lines: np.ndarray
    pca: list[np.ndarray]
    atac: list[np.ndarray]
    gaga: list[np.ndarray]
    pca_nn: list[NearestNeighbors]
    atac_nn: list[NearestNeighbors]
    gaga_nn: list[NearestNeighbors]


def load_reference() -> Reference:
    obs = pd.read_csv(DATA / "paired_obs.csv", low_memory=False)
    with np.load(DATA / "paired_metacell_ids_by_time.npz", allow_pickle=True) as saved:
        ids_by_time = [saved[key].astype(str) for key in TIME_KEYS]
    ids = np.concatenate(ids_by_time)
    if not np.array_equal(ids, obs["paired_metacell_id"].astype(str).to_numpy()):
        raise ValueError("paired_obs and paired_metacell_ids_by_time order differ")

    with np.load(DATA / "rna_pca30_normalized_by_time.npz") as saved:
        pca = [np.asarray(saved[key], dtype=np.float32) for key in TIME_KEYS]
    with np.load(DATA / "atac_lsi12_normalized_by_time.npz") as saved:
        atac = [np.asarray(saved[key], dtype=np.float32) for key in TIME_KEYS]
    with np.load(GAGA, allow_pickle=True) as saved:
        if not np.array_equal(saved["cell_ids"].astype(str), ids):
            raise ValueError("GAGA and paired reference IDs differ")
        gaga_all = np.asarray(saved["embedding_model_input"], dtype=np.float32)
        gaga_times = np.asarray(saved["time_labels"], dtype=float)
    gaga = [gaga_all[np.isclose(gaga_times, time)] for time in PHYSICAL_TIMES]

    stage_rows = [np.flatnonzero(obs["time_key"].astype(str).eq(key)) for key in TIME_KEYS]
    for index, rows in enumerate(stage_rows):
        expected = len(ids_by_time[index])
        if len(rows) != expected or len(pca[index]) != expected or len(atac[index]) != expected:
            raise ValueError(f"Stage reference length mismatch for {TIME_KEYS[index]}")
        if len(gaga[index]) != expected:
            raise ValueError(f"GAGA stage length mismatch for {TIME_KEYS[index]}")

    source_lines = (
        obs.loc[obs["time_key"].astype(str).eq(TIME_KEYS[0]), "line"]
        .astype(str)
        .str.lower()
        .to_numpy()
    )
    if len(source_lines) != 501:
        raise ValueError(f"Expected 501 D4 source particles, found {len(source_lines)}")
    observed_lines = tuple(sorted(pd.unique(source_lines)))
    if observed_lines != tuple(sorted(LINES)):
        raise ValueError(f"Unexpected source lines: {observed_lines}")

    return Reference(
        obs=obs,
        ids=ids,
        source_lines=source_lines,
        pca=pca,
        atac=atac,
        gaga=gaga,
        pca_nn=[NearestNeighbors(n_neighbors=KNN_K, n_jobs=-1).fit(x) for x in pca],
        atac_nn=[NearestNeighbors(n_neighbors=KNN_K, n_jobs=-1).fit(x) for x in atac],
        gaga_nn=[NearestNeighbors(n_neighbors=KNN_K, n_jobs=-1).fit(x) for x in gaga],
    )


def barycentric_readout(
    query: np.ndarray,
    neighbors: NearestNeighbors,
    observed_values: np.ndarray,
) -> tuple[np.ndarray, float]:
    distances, indices = neighbors.kneighbors(query, n_neighbors=KNN_K)
    weights = neighbor_weights(distances)
    mapped = np.sum(observed_values[indices] * weights[..., None], axis=1)
    return np.asarray(mapped, dtype=np.float32), float(np.median(distances[:, -1]))


def normalized_weights(log_mass: np.ndarray | None, mask: np.ndarray) -> np.ndarray:
    if log_mass is None:
        return np.full(int(mask.sum()), 1.0 / int(mask.sum()), dtype=np.float32)
    values = np.asarray(log_mass, dtype=np.float64).reshape(-1)[mask]
    if not np.isfinite(values).all():
        raise ValueError("Non-finite log mass")
    weights = np.exp(values - float(np.max(values)))
    weights /= float(weights.sum())
    return weights.astype(np.float32)


def microcluster(
    points: np.ndarray,
    weights: np.ndarray,
    max_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights /= float(weights.sum())
    n_clusters = min(int(max_clusters), len(points))
    if n_clusters == len(points):
        return weights.astype(np.float32), points
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=0,
        batch_size=2048,
        n_init=3,
        max_iter=200,
    )
    cluster_ids = model.fit_predict(points, sample_weight=weights)
    cluster_weights = np.bincount(
        cluster_ids, weights=weights, minlength=n_clusters
    ).astype(np.float64)
    keep = cluster_weights > 0
    cluster_weights = cluster_weights[keep]
    cluster_weights /= float(cluster_weights.sum())
    centers = np.asarray(model.cluster_centers_[keep], dtype=np.float32)
    return cluster_weights.astype(np.float32), centers


def sinkhorn_w2(
    points_x: np.ndarray,
    weights_x: np.ndarray,
    points_y: np.ndarray,
    weights_y: np.ndarray,
    loss: SamplesLoss,
    max_clusters: int,
) -> tuple[float, float]:
    wx, x = microcluster(points_x, weights_x, max_clusters)
    wy, y = microcluster(points_y, weights_y, max_clusters)
    with torch.no_grad():
        divergence = loss(
            torch.as_tensor(wx),
            torch.as_tensor(x),
            torch.as_tensor(wy),
            torch.as_tensor(y),
        )
    value = max(float(divergence.detach().cpu()), 0.0)
    return value, float(np.sqrt(2.0 * value))


def sinkhorn_w2_prepared(
    points_x: np.ndarray,
    weights_x: np.ndarray,
    prepared_y: tuple[np.ndarray, np.ndarray],
    loss: SamplesLoss,
    max_clusters: int,
) -> tuple[float, float]:
    wx, x = microcluster(points_x, weights_x, max_clusters)
    wy, y = prepared_y
    with torch.no_grad():
        divergence = loss(
            torch.as_tensor(wx),
            torch.as_tensor(x),
            torch.as_tensor(wy),
            torch.as_tensor(y),
        )
    value = max(float(divergence.detach().cpu()), 0.0)
    return value, float(np.sqrt(2.0 * value))


def load_primary_rollouts() -> tuple[pd.DataFrame, list[tuple[str, Rollout]]]:
    registry = pd.read_csv(REGISTRY)
    lookup = registry.set_index("model_id", drop=False)
    rows: list[pd.Series] = []
    rollouts: list[tuple[str, Rollout]] = []
    for model_id, label in PRIMARY_MODELS:
        if model_id not in lookup.index:
            raise KeyError(f"Missing frozen model: {model_id}")
        row = lookup.loc[model_id]
        if not str(row["completion_status"]).startswith("analysis_ready"):
            raise ValueError(f"Frozen model is not analysis-ready: {model_id}")
        rollout = load_rollout(row)
        if not np.allclose(rollout.time[OBSERVED_INDICES], PHYSICAL_TIMES, atol=1e-6):
            raise ValueError(f"Observed time grid mismatch for {model_id}")
        if not np.isfinite(rollout.rna).all() or not np.isfinite(rollout.atac).all():
            raise ValueError(f"Non-finite rollout: {model_id}")
        rows.append(row)
        rollouts.append((label, rollout))
    return pd.DataFrame(rows).reset_index(drop=True), rollouts


def observed_line_mask(reference: Reference, stage: int, line: str) -> np.ndarray:
    frame = reference.obs.loc[
        reference.obs["time_key"].astype(str).eq(TIME_KEYS[stage])
    ].reset_index(drop=True)
    return frame["line"].astype(str).str.lower().eq(line).to_numpy()


def evaluate(
    rollouts: list[tuple[str, Rollout]],
    reference: Reference,
    sinkhorn: SamplesLoss,
    max_clusters: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    matching_rows: list[dict[str, object]] = []
    prepared_targets: dict[tuple[int, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for stage in range(1, len(TIME_KEYS)):
        for line in LINES:
            target_mask = observed_line_mask(reference, stage, line)
            for modality, observed in (
                ("RNA", reference.pca[stage][target_mask]),
                ("ATAC", reference.atac[stage][target_mask]),
            ):
                observed_uniform = np.full(
                    len(observed), 1.0 / len(observed), dtype=np.float32
                )
                prepared_targets[(stage, line, modality)] = microcluster(
                    observed, observed_uniform, max_clusters
                )
        print(f"prepared observed {TIME_LABELS[stage]}", flush=True)

    for method_order, (label, rollout) in enumerate(rollouts):
        for stage in range(1, len(TIME_KEYS)):
            dense_index = int(OBSERVED_INDICES[stage])
            rna_neighbors = (
                reference.gaga_nn[stage]
                if rollout.rna_space == "GAGA10_zscore"
                else reference.pca_nn[stage]
            )
            mapped_rna, rna_kth = barycentric_readout(
                rollout.rna[dense_index], rna_neighbors, reference.pca[stage]
            )
            mapped_atac, atac_kth = barycentric_readout(
                rollout.atac[dense_index], reference.atac_nn[stage], reference.atac[stage]
            )
            mapping_rows.append(
                {
                    "method_order": method_order,
                    "method": label,
                    "model_id": rollout.model_id,
                    "time": TIME_LABELS[stage],
                    "rna_native_space": rollout.rna_space,
                    "rna_median_k15_radius": rna_kth,
                    "atac_median_k15_radius": atac_kth,
                    "atac_semantics": rollout.atac_semantics,
                }
            )
            stage_mass = None if rollout.log_mass is None else rollout.log_mass[dense_index]
            for line in LINES:
                source_mask = reference.source_lines == line
                target_mask = observed_line_mask(reference, stage, line)
                uniform_query = np.full(
                    int(source_mask.sum()), 1.0 / int(source_mask.sum()), dtype=np.float32
                )
                native_query = normalized_weights(stage_mass, source_mask)
                for modality, predicted, observed in (
                    ("RNA", mapped_rna[source_mask], reference.pca[stage][target_mask]),
                    ("ATAC", mapped_atac[source_mask], reference.atac[stage][target_mask]),
                ):
                    prepared_target = prepared_targets[(stage, line, modality)]
                    previous: tuple[float, float] | None = None
                    for weighting, query_weights in (
                        ("uniform", uniform_query),
                        ("native_mass", native_query),
                    ):
                        if weighting == "native_mass" and stage_mass is None:
                            if previous is None:
                                raise RuntimeError("Uniform score cache was not populated")
                            divergence, w2 = previous
                        else:
                            divergence, w2 = sinkhorn_w2_prepared(
                                predicted,
                                query_weights,
                                prepared_target,
                                sinkhorn,
                                max_clusters,
                            )
                            if weighting == "uniform":
                                previous = (divergence, w2)
                        rows.append(
                            {
                                "method_order": method_order,
                                "method": label,
                                "model_id": rollout.model_id,
                                "balance_mode": rollout.balance_mode,
                                "sync_mode": rollout.sync_mode,
                                "C_y": rollout.c_y,
                                "time": TIME_LABELS[stage],
                                "physical_time": PHYSICAL_TIMES[stage],
                                "source_line": line,
                                "modality": modality,
                                "query_weighting": weighting,
                                "query_n": int(source_mask.sum()),
                                "observed_n": int(target_mask.sum()),
                                "sinkhorn_divergence": divergence,
                                "w2_score": w2,
                                "atac_semantics": rollout.atac_semantics,
                            }
                        )
                    if previous is None:
                        raise RuntimeError("Uniform score cache was not populated")
                    for target_line in LINES:
                        if target_line == line:
                            match_divergence, match_w2 = previous
                        else:
                            match_divergence, match_w2 = sinkhorn_w2_prepared(
                                predicted,
                                uniform_query,
                                prepared_targets[(stage, target_line, modality)],
                                sinkhorn,
                                max_clusters,
                            )
                        matching_rows.append(
                            {
                                "method_order": method_order,
                                "method": label,
                                "model_id": rollout.model_id,
                                "time": TIME_LABELS[stage],
                                "physical_time": PHYSICAL_TIMES[stage],
                                "source_line": line,
                                "target_line": target_line,
                                "is_matching_line": target_line == line,
                                "modality": modality,
                                "query_weighting": "uniform",
                                "sinkhorn_divergence": match_divergence,
                                "w2_score": match_w2,
                            }
                        )
        print(f"scored {label}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(mapping_rows), pd.DataFrame(matching_rows)


def summarize(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        scores.groupby(
            ["method_order", "method", "model_id", "modality", "query_weighting"],
            sort=False,
            as_index=False,
        )
        .agg(
            mean_w2=("w2_score", "mean"),
            median_w2=("w2_score", "median"),
            std_across_24_units=("w2_score", "std"),
            n_line_time_units=("w2_score", "size"),
        )
        .sort_values(["modality", "query_weighting", "mean_w2", "method_order"])
    )
    by_line = (
        scores.groupby(
            ["method_order", "method", "modality", "query_weighting", "source_line"],
            sort=False,
            as_index=False,
        )["w2_score"]
        .mean()
        .rename(columns={"w2_score": "mean_w2_across_times"})
    )
    by_time = (
        scores.groupby(
            ["method_order", "method", "modality", "query_weighting", "time", "physical_time"],
            sort=False,
            as_index=False,
        )["w2_score"]
        .mean()
        .rename(columns={"w2_score": "mean_w2_across_lines"})
    )
    return summary, by_line, by_time


def summarize_line_matching(matching: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unit_rows: list[dict[str, object]] = []
    keys = ["method_order", "method", "model_id", "modality", "time", "source_line"]
    for key, group in matching.groupby(keys, sort=False):
        group = group.sort_values("target_line")
        matched = group.loc[group["is_matching_line"]]
        wrong = group.loc[~group["is_matching_line"]]
        if len(matched) != 1 or len(wrong) != 3:
            raise ValueError("Expected one matching and three mismatching target lines")
        best = group.sort_values(["w2_score", "target_line"]).iloc[0]
        unit_rows.append(
            {
                **dict(zip(keys, key)),
                "matched_w2": float(matched.iloc[0]["w2_score"]),
                "mean_wrong_w2": float(wrong["w2_score"].mean()),
                "nearest_target_line": str(best["target_line"]),
                "top1_same_line": str(best["target_line"]) == str(key[-1]),
            }
        )
    units = pd.DataFrame(unit_rows)
    units["wrong_minus_matched_w2"] = units["mean_wrong_w2"] - units["matched_w2"]
    units["matched_to_wrong_ratio"] = units["matched_w2"] / units["mean_wrong_w2"]
    summary = (
        units.groupby(
            ["method_order", "method", "model_id", "modality"],
            sort=False,
            as_index=False,
        )
        .agg(
            mean_matched_w2=("matched_w2", "mean"),
            mean_wrong_w2=("mean_wrong_w2", "mean"),
            mean_wrong_minus_matched_w2=("wrong_minus_matched_w2", "mean"),
            mean_matched_to_wrong_ratio=("matched_to_wrong_ratio", "mean"),
            top1_same_line_accuracy=("top1_same_line", "mean"),
            n_line_time_units=("top1_same_line", "size"),
        )
        .sort_values(["modality", "mean_matched_w2", "method_order"])
    )
    return units, summary


def write_readme(output: Path, summary: pd.DataFrame) -> None:
    primary = summary.loc[summary["query_weighting"].eq("uniform")]
    rna = primary.loc[primary["modality"].eq("RNA")].sort_values("mean_w2")
    atac = primary.loc[primary["modality"].eq("ATAC")].sort_values("mean_w2")
    text = f"""# Source-line group distribution recovery

All 501 common D4 particles were grouped by their recorded stem-cell line.
At D7, D9, D11, D12, D18 and D21, each predicted group was compared with
observed metacells from the matching line.  The reported mean gives equal
weight to each of four lines and six times; D4 is excluded.

Primary analysis uses uniform particle and observed-metacell weights.  All
methods use the same line-blind, stage-conditioned k=15 common readout.  The
distance is `sqrt(2 * debiased p=2 Sinkhorn divergence)` with blur 0.05 and at
most 512 microclusters, so it is a regularized W2 surrogate rather than exact
empirical W2.

Best uniform RNA mean: `{rna.iloc[0]['method']}` ({rna.iloc[0]['mean_w2']:.6f}).
Best uniform ATAC mean: `{atac.iloc[0]['method']}` ({atac.iloc[0]['mean_w2']:.6f}).

ATAC interpretation is asymmetric: only COATI Sync has model-coupled native
ATAC trajectories.  ATAC for all other methods is a fixed-T post-hoc readout.
These are full-training reconstruction/coherence results, not held-out
generalization results.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite non-empty directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    registry, rollouts = load_primary_rollouts()
    reference = load_reference()
    sinkhorn = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=args.sinkhorn_blur,
        debias=True,
        backend="tensorized",
    )
    scores, mapping, matching = evaluate(
        rollouts, reference, sinkhorn, args.sinkhorn_clusters
    )
    summary, by_line, by_time = summarize(scores)
    matching_units, matching_summary = summarize_line_matching(matching)
    scores.to_csv(output / "source_line_time_scores.csv", index=False)
    summary.to_csv(output / "method_summary.csv", index=False)
    by_line.to_csv(output / "method_by_source_line.csv", index=False)
    by_time.to_csv(output / "method_by_time.csv", index=False)
    mapping.to_csv(output / "common_readout_mapping_audit.csv", index=False)
    matching.to_csv(output / "source_to_target_line_w2_matrix.csv", index=False)
    matching_units.to_csv(output / "line_matching_unit_summary.csv", index=False)
    matching_summary.to_csv(output / "line_matching_method_summary.csv", index=False)
    registry.to_csv(output / "frozen_model_subset.csv", index=False)
    write_readme(output, summary)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "registry": str(REGISTRY),
        "registry_sha256": sha256(REGISTRY),
        "source_data": str(DATA),
        "source_particles": 501,
        "source_lines": list(LINES),
        "evaluated_times": list(TIME_LABELS[1:]),
        "excluded_time": "D4 (the observed starting distribution)",
        "aggregation": "equal mean over four source lines and six later times",
        "primary_weighting": "uniform query particles and uniform observed metacells",
        "sensitivity_weighting": "native predicted mass normalized within line and time",
        "knn_k": KNN_K,
        "knn_line_blind": True,
        "rna_mapping": "PCA30 models query observed normalized PCA30; MIOFlow queries frozen GAGA10; all values are observed normalized PCA30 barycenters",
        "atac_mapping": "all methods query observed normalized LSI12 and return normalized LSI12 barycenters",
        "w2_definition": "sqrt(2 * debiased p=2 Sinkhorn divergence), a regularized W2 surrogate",
        "sinkhorn_blur": args.sinkhorn_blur,
        "max_microclusters": args.sinkhorn_clusters,
        "coati_cy": 0.5,
        "atac_caveat": "COATI Sync is native model-coupled; all other ATAC trajectories are fixed-T post-hoc readouts",
        "claim_limit": "full-training source-line reconstruction/coherence, not held-out prediction or lineage tracing",
        "hpc_accessed": False,
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
