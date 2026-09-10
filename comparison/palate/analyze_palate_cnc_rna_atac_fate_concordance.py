#!/usr/bin/env python
"""Particle-level RNA--ATAC anterior/posterior fate concordance in palate.

The analysis follows the same E12.5 CNC-derived particle through RNA and the
shared full-data FiLM RNA->ATAC map.  Fate axes are fixed before looking at any
trajectory result:

* original-paper top-five anterior/posterior markers;
* genes retained only when the author-provided peak--gene catalogue contains
  at least one positive, significant and assayed linked peak;
* every eligible linked peak is retained, first averaged within gene and then
  across genes so genes with more catalogued peaks do not dominate;
* an independent motif sensitivity analysis uses Shox2/Alx1 versus Dlx1/Dlx2.

At each observed embryonic stage, a common kNN decoder transfers the fixed RNA
and ATAC program logits from real paired cells to every method's particle.  The
primary k is 5; k=10 and 20 are prespecified sensitivity analyses.  Unbalanced
methods are weighted by native learned mass renormalized within the initial
CNC cohort.  The outputs include continuous fate correlation, probability JSD,
binary branch discordance, definite (both-confident) discordance and Cohen's
kappa, which is the same-label concordance above the value explained by branch
marginals alone.

TrajectoryNet is intentionally absent: the currently validated palate model is
terminal-origin, so its particle IDs do not index the common E12.5 CNC cohort.
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
import math
import os
import sys
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
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torchdiffeq import odeint


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

from evaluate_gastrulation_flow_methods_normalized import (  # noqa: E402
    _load_mioflow_bundle,
    _mioflow_model_path_to_xnorm,
    _mioflow_xnorm_to_model_space,
)
from evaluate_palate_loo_same_space import (  # noqa: E402
    _load_film,
    _load_scale,
    _map_to_atac,
)
from trainfbench_plot_style import (  # noqa: E402
    METHOD_STYLES,
    NATURE_CUD,
    apply_nature_rc,
)


PALATE_ROOT = Path("external/COATI/MouseBrain")
RNA_BENCH = ROOT / "data/palate_rna_cytobridge.h5ad"
ATAC_BENCH = ROOT / "data/palate_atac_benchmark.h5ad"
RNA_RAW = PALATE_ROOT / "data/rna_dimReduced.h5ad"
ATAC_RAW = PALATE_ROOT / "data/atac_dimReduced.h5ad"
MOTIF_RAW = PALATE_ROOT / "data/CNC_motif.h5ad"
PEAK_LINKS = PALATE_ROOT / "DataGen/Mouse/marker_gene_peak_links.csv"
RNA_NORM = ROOT / "data/palate_rna_primal_norm_params.pt"
ATAC_NORM = ROOT / "data/palate_atac_secondary_norm_params_lsi15.pt"
ARCHIVE = ROOT / "results/palate_trajectory_archive"
COMMON_INITIAL = ARCHIVE / "manifest/common_initial_indices.npy"
COMMON_X0 = ARCHIVE / "manifest/common_initial_rna_norm.npy"
CB_U_WEIGHTS = (
    ROOT
    / "results/palate_ap_fate_separation"
    / "cytobridge_unbalanced_common_initial_weights.npz"
)
MIOFLOW_FULL = ROOT / "results/mioflow_palate_full_pca_gaga10_n1024_20000"
LOO_ROOT = ROOT / "results/palate_strict_sync_loo_full_t_gaga10"
DEFAULT_OUTPUT = ROOT / "results/palate_cnc_rna_atac_fate_concordance"

STAGES = ((0.0, "E12.5"), (1.0, "E13.5"), (1.5, "E14.0"), (2.0, "E14.5"))
STAGE_TIMES = np.asarray([value[0] for value in STAGES], dtype=float)
K_VALUES = (5, 10, 20)
PRIMARY_K = 5
PRIMARY_CY = 0.5
CONFIDENCE_THRESHOLD = 0.65
CNC = "CNC-derived progenitors"
ANTERIOR = "anterior palatal mesenchymal"
POSTERIOR = "posterior palatal mesenchymal"

# Yan et al. 2024 top-five regional markers.  The primary matched program later
# intersects these lists with the author peak--gene catalogue.
PAPER_MARKERS = {
    "anterior": ("Shox2", "Satb2", "Inhba", "Cyp26b1", "Nrp1"),
    "posterior": ("Meox2", "Prickle1", "Sim2", "Efnb2", "Trps1"),
}
MOTIF_PROGRAMS = {
    "anterior": ("Shox2", "Alx1"),
    "posterior": ("Dlx1", "Dlx2"),
}

METHODS_FULL = (
    "COATI balanced",
    "COATI unbalanced",
    "CytoBridge balanced",
    "CytoBridge unbalanced",
    "MIOFlow",
)
DISPLAY = {
    "COATI balanced": "COATI bal",
    "COATI unbalanced": "COATI unbal",
    "CytoBridge balanced": "CytoBridge bal",
    "CytoBridge unbalanced": "CytoBridge unbal",
    "MIOFlow": "MIOFlow",
    "Unbalanced RNA-only": "RNA-only unbal",
    "Observed / real": "Observed",
}
RNA_ONLY_COLOR = "#777777"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cy", type=float, default=PRIMARY_CY)
    parser.add_argument("--k-values", type=int, nargs="+", default=list(K_VALUES))
    parser.add_argument("--primary-k", type=int, default=PRIMARY_K)
    parser.add_argument("--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dense(value: object) -> np.ndarray:
    if sp.issparse(value):
        return value.toarray().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


def zscore_columns(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    mean = np.nanmean(value, axis=0, keepdims=True)
    sd = np.nanstd(value, axis=0, keepdims=True)
    sd[~np.isfinite(sd) | (sd < 1e-10)] = 1.0
    return np.nan_to_num((value - mean) / sd)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sigmoid(logit: np.ndarray) -> np.ndarray:
    value = np.asarray(logit, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def normalize_weight(weight: np.ndarray) -> np.ndarray:
    value = np.asarray(weight, dtype=np.float64).reshape(-1)
    value = np.where(np.isfinite(value) & (value >= 0), value, 0.0)
    if value.sum() <= 0:
        raise ValueError("No positive particle mass")
    return value / value.sum()


def weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    keep = np.isfinite(value) & np.isfinite(weight) & (weight >= 0)
    if not np.any(keep) or weight[keep].sum() <= 0:
        return np.nan
    return float(np.average(value[keep], weights=weight[keep]))


def weighted_pearson(left: np.ndarray, right: np.ndarray, weight: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    keep = np.isfinite(left) & np.isfinite(right) & np.isfinite(weight) & (weight >= 0)
    if keep.sum() < 3 or weight[keep].sum() <= 0:
        return np.nan
    x = left[keep]
    y = right[keep]
    w = weight[keep] / weight[keep].sum()
    xc = x - np.sum(w * x)
    yc = y - np.sum(w * y)
    denominator = math.sqrt(float(np.sum(w * xc * xc) * np.sum(w * yc * yc)))
    if denominator <= 1e-12:
        return np.nan
    return float(np.sum(w * xc * yc) / denominator)


def weighted_spearman(left: np.ndarray, right: np.ndarray, weight: np.ndarray) -> float:
    return weighted_pearson(rankdata(left), rankdata(right), weight)


def jsd_binary(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(left, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    q = np.clip(np.asarray(right, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    p2 = np.column_stack((p, 1.0 - p))
    q2 = np.column_stack((q, 1.0 - q))
    midpoint = 0.5 * (p2 + q2)
    return 0.5 * np.sum(p2 * np.log2(p2 / midpoint), axis=1) + 0.5 * np.sum(
        q2 * np.log2(q2 / midpoint), axis=1
    )


def calibrate_logit(
    contrast: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    terminal_ap = np.isclose(times, 2.0) & np.isin(labels, [ANTERIOR, POSTERIOR])
    response = labels[terminal_ap] == ANTERIOR
    model = LogisticRegression(
        class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=0
    ).fit(np.asarray(contrast)[terminal_ap, None], response)
    logits = model.decision_function(np.asarray(contrast)[:, None])
    auc = roc_auc_score(response, logits[terminal_ap])
    return logits, {
        "training_stage": "E14.5",
        "training_celltypes": [ANTERIOR, POSTERIOR],
        "n_training_cells": int(terminal_ap.sum()),
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
        "terminal_auc": float(auc),
    }


def build_fixed_programs(
    obs_names: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, object]]:
    raw_rna = ad.read_h5ad(RNA_RAW, backed="r")
    raw_atac = ad.read_h5ad(ATAC_RAW, backed="r")
    raw_motif = ad.read_h5ad(MOTIF_RAW, backed="r")
    for name, obj in (("RNA", raw_rna), ("ATAC", raw_atac), ("motif", raw_motif)):
        if not np.array_equal(obj.obs_names.astype(str).to_numpy(), obs_names):
            raise ValueError(f"{name} raw cells are not aligned to benchmark cells")
    if raw_rna.raw is None:
        raise ValueError("Raw RNA object has no .raw matrix")

    links = pd.read_csv(PEAK_LINKS)
    branch_by_gene = {
        gene: branch for branch, genes in PAPER_MARKERS.items() for gene in genes
    }
    links = links[
        links["gene"].astype(str).isin(branch_by_gene)
        & (pd.to_numeric(links["zscore"], errors="coerce") > 0)
        & (pd.to_numeric(links["pvalue"], errors="coerce") < 0.05)
        & links["peak"].astype(str).isin(set(raw_atac.var_names.astype(str)))
    ].copy()
    links["branch"] = links["gene"].map(branch_by_gene)
    links = links.sort_values(
        ["branch", "gene", "zscore", "pvalue"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    linked_genes = {
        branch: tuple(
            gene for gene in PAPER_MARKERS[branch] if gene in set(links["gene"])
        )
        for branch in PAPER_MARKERS
    }
    if any(len(value) == 0 for value in linked_genes.values()):
        raise ValueError("A branch has no linked original-paper marker genes")

    genes = list(linked_genes["anterior"] + linked_genes["posterior"])
    missing_genes = [gene for gene in genes if gene not in raw_rna.raw.var_names]
    if missing_genes:
        raise ValueError(f"Missing RNA genes: {missing_genes}")
    gene_indices = [int(raw_rna.raw.var_names.get_loc(gene)) for gene in genes]
    expression = zscore_columns(dense(raw_rna.raw.X[:, gene_indices]))
    gene_column = {gene: column for column, gene in enumerate(genes)}
    rna_branch = {
        branch: expression[:, [gene_column[g] for g in linked_genes[branch]]].mean(axis=1)
        for branch in PAPER_MARKERS
    }
    rna_contrast = rna_branch["anterior"] - rna_branch["posterior"]

    peaks = list(dict.fromkeys(links["peak"].astype(str)))
    peak_indices = [int(raw_atac.var_names.get_loc(peak)) for peak in peaks]
    accessibility = zscore_columns(dense(raw_atac.X[:, peak_indices]))
    peak_column = {peak: column for column, peak in enumerate(peaks)}
    atac_gene_score: dict[str, np.ndarray] = {}
    for gene in genes:
        selected = links.loc[links["gene"].eq(gene), "peak"].astype(str).tolist()
        atac_gene_score[gene] = accessibility[:, [peak_column[p] for p in selected]].mean(axis=1)
    atac_branch = {
        branch: np.column_stack([atac_gene_score[g] for g in linked_genes[branch]]).mean(axis=1)
        for branch in PAPER_MARKERS
    }
    peak_contrast = atac_branch["anterior"] - atac_branch["posterior"]

    motifs = list(MOTIF_PROGRAMS["anterior"] + MOTIF_PROGRAMS["posterior"])
    missing_motifs = [motif for motif in motifs if motif not in raw_motif.var_names]
    if missing_motifs:
        raise ValueError(f"Missing motif features: {missing_motifs}")
    motif_indices = [int(raw_motif.var_names.get_loc(motif)) for motif in motifs]
    motif_values = zscore_columns(dense(raw_motif.X[:, motif_indices]))
    motif_column = {motif: column for column, motif in enumerate(motifs)}
    motif_branch = {
        branch: motif_values[:, [motif_column[m] for m in MOTIF_PROGRAMS[branch]]].mean(axis=1)
        for branch in PAPER_MARKERS
    }
    motif_contrast = motif_branch["anterior"] - motif_branch["posterior"]

    outputs: dict[str, np.ndarray] = {}
    calibration: dict[str, object] = {}
    for name, contrast in (
        ("RNA", rna_contrast),
        ("ATAC linked peaks", peak_contrast),
        ("ATAC motifs", motif_contrast),
    ):
        outputs[name], calibration[name] = calibrate_logit(contrast, labels, times)

    ap = np.isin(labels, [ANTERIOR, POSTERIOR])
    stage_controls: list[dict[str, object]] = []
    for stage_time, stage_name in STAGES:
        subset = ap & np.isclose(times, stage_time)
        response = labels[subset] == ANTERIOR
        for name in ("RNA", "ATAC linked peaks", "ATAC motifs"):
            stage_controls.append(
                {
                    "stage_time": stage_time,
                    "stage": stage_name,
                    "program": name,
                    "n_anterior_posterior": int(subset.sum()),
                    "observed_ap_auc": float(roc_auc_score(response, outputs[name][subset])),
                }
            )

    audit = {
        "paper_markers": {key: list(value) for key, value in PAPER_MARKERS.items()},
        "matched_linked_genes": {key: list(value) for key, value in linked_genes.items()},
        "motif_programs": {key: list(value) for key, value in MOTIF_PROGRAMS.items()},
        "peak_selection": (
            "original-paper top-five anterior/posterior marker intersected with "
            "author marker_gene_peak_links.csv; zscore>0, p<0.05, assayed peak; "
            "all eligible peaks retained"
        ),
        "atac_aggregation": (
            "z-score each peak across the paired atlas; average peaks within gene; "
            "average genes within branch"
        ),
        "rna_aggregation": (
            "z-score each matched linked marker across the paired atlas; average "
            "genes within branch"
        ),
        "calibration": calibration,
    }
    raw_rna.file.close()
    raw_atac.file.close()
    raw_motif.file.close()
    return outputs, pd.DataFrame(stage_controls), audit


class StageDecoder:
    def __init__(
        self,
        latent: np.ndarray,
        times: np.ndarray,
        target: np.ndarray,
        k: int,
    ) -> None:
        self.models: dict[float, NearestNeighbors] = {}
        self.targets: dict[float, np.ndarray] = {}
        for stage_time, _ in STAGES:
            subset = np.isclose(times, stage_time)
            self.models[stage_time] = NearestNeighbors(
                n_neighbors=min(k, int(subset.sum())), n_jobs=-1
            ).fit(np.asarray(latent[subset], dtype=np.float32))
            self.targets[stage_time] = np.asarray(target[subset], dtype=np.float64)

    def __call__(self, query: np.ndarray, time: float) -> np.ndarray:
        if time not in self.models:
            raise ValueError(f"Only observed stage anchors are evaluated, got {time}")
        indices = self.models[time].kneighbors(
            np.asarray(query, dtype=np.float32), return_distance=False
        )
        return self.targets[time][indices].mean(axis=1)


def rollout_mioflow_common(
    x0_norm: np.ndarray,
    rna_scale: float,
    film: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model_path = MIOFLOW_FULL / "model.pt"
    metadata = json.loads((MIOFLOW_FULL / "metadata.json").read_text(encoding="utf-8"))
    bundle = _load_mioflow_bundle(model_path, device)
    if not bundle.use_gaga or bundle.mean_vals.shape != (10,):
        raise ValueError("Expected the full palate MIOFlow GAGA10 checkpoint")
    physical_training = np.asarray(metadata["time_values"], dtype=float)
    if not np.allclose(physical_training, STAGE_TIMES):
        raise ValueError(f"Unexpected MIOFlow physical times: {physical_training}")
    z0 = _mioflow_xnorm_to_model_space(x0_norm, bundle, rna_scale, device, batch_size)
    paths: list[np.ndarray] = []
    ts = torch.as_tensor(bundle.model_times, dtype=torch.float32, device=device)
    for start in range(0, len(z0), batch_size):
        state = torch.as_tensor(z0[start : start + batch_size], dtype=torch.float32, device=device)
        if hasattr(bundle.model, "reset_momentum"):
            bundle.model.reset_momentum()
        with torch.no_grad():
            paths.append(odeint(bundle.model, state, ts).detach().cpu().numpy())
    path_model = np.concatenate(paths, axis=1).astype(np.float32, copy=False)
    rna_path = _mioflow_model_path_to_xnorm(
        path_model, bundle, rna_scale, device, batch_size
    )
    # GAGA is not an exactly invertible autoencoder.  The biological initial
    # condition is nevertheless the shared observed E12.5 particle set, so the
    # displayed/decoded state at t=0 must be x0 itself rather than its AE
    # reconstruction.  All later states remain native GAGA10 rollouts.
    rna_path[0] = x0_norm
    atac_path = np.stack(
        [
            _map_to_atac(film, state, float(time), device, batch_size)
            for state, time in zip(rna_path, STAGE_TIMES)
        ]
    ).astype(np.float32, copy=False)
    return {
        "time": STAGE_TIMES.astype(np.float32),
        "rna_norm": rna_path,
        "atac_norm": atac_path,
        "initial_indices": np.load(COMMON_INITIAL),
        "source": np.asarray(str(model_path)),
    }


def load_full_methods(
    cy: float,
    x0_norm: np.ndarray,
    rna_scale: float,
    film: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    cy_text = f"{cy:.1f}"
    paths = {
        "COATI balanced": ARCHIVE
        / f"trajectories/sync/BalancedSync/seed_0/cy_{cy_text}_iter_20000.npz",
        "COATI unbalanced": ARCHIVE
        / f"trajectories/sync/FiLMUnbalancedSync/seed_0/cy_{cy_text}_iter_20000.npz",
        "CytoBridge balanced": ARCHIVE / "trajectories/external/cytobridge_balanced.npz",
        "CytoBridge unbalanced": ARCHIVE / "trajectories/external/cytobridge_unbalanced.npz",
    }
    result: dict[str, dict[str, np.ndarray]] = {}
    common = np.load(COMMON_INITIAL)
    for method, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=True) as saved:
            result[method] = {key: np.asarray(saved[key]) for key in saved.files}
        if not np.array_equal(result[method]["initial_indices"], common):
            raise ValueError(f"{method} does not use the common initial particles")
        result[method]["source"] = np.asarray(str(path))
    result["MIOFlow"] = rollout_mioflow_common(
        x0_norm, rna_scale, film, device, batch_size
    )
    return result


def full_weight(
    method: str,
    saved: dict[str, np.ndarray],
    stage_time: float,
    position: int,
    cohort: np.ndarray,
    cb_weights: dict[str, np.ndarray],
) -> np.ndarray:
    if method == "COATI unbalanced":
        log_weight = np.asarray(saved["log_weight"][position], dtype=float).reshape(-1)[cohort]
        log_weight -= np.max(log_weight)
        return normalize_weight(np.exp(np.clip(log_weight, -80.0, 0.0)))
    if method == "CytoBridge unbalanced" and stage_time > 0:
        cb_time = np.asarray(cb_weights["stage_time"], dtype=float)
        cb_position = int(np.argmin(np.abs(cb_time - stage_time)))
        if not np.isclose(cb_time[cb_position], stage_time):
            raise ValueError(f"No CytoBridge-U mass at {stage_time}")
        return normalize_weight(np.asarray(cb_weights["weights"][cb_position], dtype=float)[cohort])
    return np.full(int(cohort.sum()), 1.0 / int(cohort.sum()), dtype=np.float64)


def concordance_metrics(
    rna_logit: np.ndarray,
    atac_logit: np.ndarray,
    weight: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    weight = normalize_weight(weight)
    rna_probability = sigmoid(rna_logit)
    atac_probability = sigmoid(atac_logit)
    rna_label = rna_probability >= 0.5
    atac_label = atac_probability >= 0.5
    agreement = weighted_mean(rna_label == atac_label, weight)
    p_rna_a = weighted_mean(rna_label, weight)
    p_atac_a = weighted_mean(atac_label, weight)
    chance = p_rna_a * p_atac_a + (1.0 - p_rna_a) * (1.0 - p_atac_a)
    kappa = (agreement - chance) / (1.0 - chance) if chance < 1.0 - 1e-12 else np.nan
    rna_confident = (rna_probability >= threshold) | (rna_probability <= 1.0 - threshold)
    atac_confident = (atac_probability >= threshold) | (atac_probability <= 1.0 - threshold)
    both_confident = rna_confident & atac_confident
    confident_mass = weighted_mean(both_confident, weight)
    opposite_confident = both_confident & (rna_label != atac_label)
    opposite_total_mass = weighted_mean(opposite_confident, weight)
    confident_discordance = (
        opposite_total_mass / confident_mass if confident_mass > 1e-12 else np.nan
    )
    return {
        "weighted_pearson": weighted_pearson(rna_logit, atac_logit, weight),
        "weighted_spearman": weighted_spearman(rna_logit, atac_logit, weight),
        "weighted_mean_probability_jsd": weighted_mean(
            jsd_binary(rna_probability, atac_probability), weight
        ),
        "binary_agreement": agreement,
        "binary_discordance": 1.0 - agreement,
        "chance_binary_agreement": chance,
        "cohens_kappa": kappa,
        "rna_anterior_mass": weighted_mean(rna_probability, weight),
        "atac_anterior_mass": weighted_mean(atac_probability, weight),
        "both_confident_mass": confident_mass,
        "opposite_confident_total_mass": opposite_total_mass,
        "confident_discordance_conditional": confident_discordance,
    }


def score_one_state(
    *,
    scope: str,
    scenario: str,
    method: str,
    stage_time: float,
    stage_name: str,
    rna_points: np.ndarray,
    atac_points: np.ndarray,
    weight: np.ndarray,
    particle_ids: np.ndarray,
    decoders: dict[tuple[int, str], StageDecoder],
    k: int,
    threshold: float,
    save_detail: bool,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    rna_logit = decoders[(k, "RNA")](rna_points, stage_time)
    rows: list[dict[str, object]] = []
    details: list[pd.DataFrame] = []
    for atac_program in ("ATAC linked peaks", "ATAC motifs"):
        atac_logit = decoders[(k, atac_program)](atac_points, stage_time)
        metrics = concordance_metrics(rna_logit, atac_logit, weight, threshold)
        rows.append(
            {
                "scope": scope,
                "scenario": scenario,
                "stage_time": stage_time,
                "stage": stage_name,
                "method": method,
                "display_method": DISPLAY[method],
                "atac_program": atac_program,
                "k": k,
                "n_initial_cnc": len(rna_points),
                "confidence_threshold": threshold,
                **metrics,
            }
        )
        if save_detail:
            rna_probability = sigmoid(rna_logit)
            atac_probability = sigmoid(atac_logit)
            rna_label = rna_probability >= 0.5
            atac_label = atac_probability >= 0.5
            rna_confident = (rna_probability >= threshold) | (
                rna_probability <= 1.0 - threshold
            )
            atac_confident = (atac_probability >= threshold) | (
                atac_probability <= 1.0 - threshold
            )
            details.append(
                pd.DataFrame(
                    {
                        "scope": scope,
                        "scenario": scenario,
                        "stage_time": stage_time,
                        "stage": stage_name,
                        "method": method,
                        "display_method": DISPLAY[method],
                        "atac_program": atac_program,
                        "k": k,
                        "particle": particle_ids,
                        "weight": normalize_weight(weight),
                        "rna_fate_logit": rna_logit,
                        "atac_fate_logit": atac_logit,
                        "rna_anterior_probability": rna_probability,
                        "atac_anterior_probability": atac_probability,
                        "rna_branch": np.where(rna_label, "anterior", "posterior"),
                        "atac_branch": np.where(atac_label, "anterior", "posterior"),
                        "both_confident": rna_confident & atac_confident,
                        "branch_discordant": rna_label != atac_label,
                    }
                )
            )
    return rows, details


def score_real_controls(
    program_logits: dict[str, np.ndarray],
    labels: np.ndarray,
    times: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage_time, stage_name in STAGES:
        subset = np.isclose(times, stage_time) & np.isin(labels, [ANTERIOR, POSTERIOR])
        weight = np.full(int(subset.sum()), 1.0 / int(subset.sum()))
        for atac_program in ("ATAC linked peaks", "ATAC motifs"):
            rows.append(
                {
                    "scope": "observed paired A/P cells",
                    "scenario": "observed",
                    "stage_time": stage_time,
                    "stage": stage_name,
                    "method": "Observed / real",
                    "display_method": DISPLAY["Observed / real"],
                    "atac_program": atac_program,
                    "k": 0,
                    "n_initial_cnc": np.nan,
                    "n_observed_ap": int(subset.sum()),
                    "confidence_threshold": threshold,
                    **concordance_metrics(
                        program_logits["RNA"][subset],
                        program_logits[atac_program][subset],
                        weight,
                        threshold,
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_full(summary: pd.DataFrame, output: Path, primary_k: int) -> None:
    frame = summary[
        summary["scope"].eq("full")
        & summary["atac_program"].eq("ATAC linked peaks")
        & summary["k"].eq(primary_k)
    ].copy()
    apply_nature_rc(font_size=10)
    fig, axes = plt.subplots(2, 1, figsize=(4.13, 4.35), sharex=True)
    metrics = (
        ("weighted_spearman", "RNA–ATAC fate Spearman ↑"),
        ("binary_discordance", "Branch discordance (%) ↓"),
    )
    for axis, (metric, label) in zip(axes, metrics):
        for method in METHODS_FULL:
            local = frame[frame["method"].eq(method)].sort_values("stage_time")
            style = METHOD_STYLES[method]
            values = local[metric].to_numpy(float)
            if metric in ("opposite_confident_total_mass", "binary_discordance"):
                values = 100.0 * values
            axis.plot(
                local["stage_time"],
                values,
                label=DISPLAY[method],
                color=style.color,
                marker=style.marker,
                linestyle=style.linestyle,
                linewidth=1.4,
                markersize=4.5,
            )
        axis.set_ylabel(label)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        axis.margins(x=0.03)
    axes[0].axhline(0, color="#999999", linewidth=0.6, linestyle=":")
    axes[1].set_xticks(STAGE_TIMES, [stage for _, stage in STAGES])
    axes[1].set_xlabel("Embryonic stage")
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.005),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.20, right=0.98, top=0.86, bottom=0.12, hspace=0.32)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"full_cnc_rna_atac_fate_concordance_k{primary_k}.{suffix}", dpi=400)
    plt.close(fig)


def method_style_values(method: str) -> tuple[str, str]:
    if method == "Unbalanced RNA-only":
        return RNA_ONLY_COLOR, "o"
    style = METHOD_STYLES[method]
    return style.color, style.marker


def plot_loo(summary: pd.DataFrame, output: Path, primary_k: int) -> None:
    frame = summary[
        summary["scope"].eq("strict LOO")
        & summary["atac_program"].eq("ATAC linked peaks")
        & summary["k"].eq(primary_k)
    ].copy()
    methods = list(METHODS_FULL) + ["Unbalanced RNA-only"]
    apply_nature_rc(font_size=10)
    fig, axes = plt.subplots(2, 2, figsize=(4.13, 3.85), sharex="col")
    for column, (scenario, stage_name) in enumerate(
        (("loo_time1", "Held-out E13.5"), ("loo_time2", "Held-out E14.0"))
    ):
        local = frame[frame["scenario"].eq(scenario)].set_index("method")
        y = np.arange(len(methods))
        for row, (metric, label) in enumerate(
            (
                ("weighted_spearman", "Spearman ↑"),
                ("binary_discordance", "Discordance (%) ↓"),
            )
        ):
            axis = axes[row, column]
            for position, method in enumerate(methods):
                color, marker = method_style_values(method)
                value = float(local.loc[method, metric])
                if metric in ("opposite_confident_total_mass", "binary_discordance"):
                    value *= 100.0
                axis.scatter(value, position, color=color, marker=marker, s=24, zorder=3)
            axis.set_yticks(y)
            if column == 0:
                axis.set_yticklabels([DISPLAY[m] for m in methods])
            else:
                axis.set_yticklabels([])
                axis.tick_params(axis="y", length=0)
            axis.invert_yaxis()
            axis.grid(axis="x", color="#DDDDDD", linewidth=0.6)
            axis.set_xlabel(label)
            if row == 0:
                axis.set_title(stage_name, pad=5)
    fig.subplots_adjust(left=0.34, right=0.98, top=0.92, bottom=0.12, wspace=0.24, hspace=0.42)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"strict_loo_cnc_rna_atac_fate_concordance_k{primary_k}.{suffix}", dpi=400)
    plt.close(fig)


def plot_loo_stage(
    summary: pd.DataFrame,
    output: Path,
    primary_k: int,
    scenario: str,
    stage_name: str,
) -> None:
    frame = summary[
        summary["scope"].eq("strict LOO")
        & summary["scenario"].eq(scenario)
        & summary["atac_program"].eq("ATAC linked peaks")
        & summary["k"].eq(primary_k)
    ].set_index("method")
    methods = list(METHODS_FULL) + ["Unbalanced RNA-only"]
    apply_nature_rc(font_size=10)
    fig, axes = plt.subplots(1, 2, figsize=(4.13, 2.55), sharey=True)
    y = np.arange(len(methods))
    for axis, (metric, title) in zip(
        axes,
        (
            ("weighted_spearman", "Fate Spearman ↑"),
            ("binary_discordance", "Discordance (%) ↓"),
        ),
    ):
        values: list[float] = []
        for position, method in enumerate(methods):
            color, marker = method_style_values(method)
            value = float(frame.loc[method, metric])
            if metric == "binary_discordance":
                value *= 100.0
            values.append(value)
            axis.scatter(value, position, color=color, marker=marker, s=25, zorder=3)
        span = max(max(values) - min(values), 1e-3)
        offset = 0.035 * span
        for position, value in enumerate(values):
            text_value = f"{value:.2f}" if metric == "binary_discordance" else f"{value:.3f}"
            axis.text(value + offset, position, text_value, va="center", ha="left", color="#222222")
        axis.set_xlim(min(values) - 0.10 * span, max(values) + 0.38 * span)
        axis.set_title(title, pad=4)
        axis.grid(axis="x", color="#DDDDDD", linewidth=0.6)
        axis.set_yticks(y)
    axes[0].invert_yaxis()
    axes[0].set_yticklabels([DISPLAY[method] for method in methods])
    axes[1].tick_params(axis="y", labelleft=False, length=0)
    fig.suptitle(stage_name, y=0.985)
    fig.subplots_adjust(left=0.34, right=0.99, top=0.80, bottom=0.16, wspace=0.25)
    slug = "e13p5" if scenario == "loo_time1" else "e14p0"
    for suffix in ("pdf", "png"):
        fig.savefig(
            output / f"strict_loo_{slug}_cnc_rna_atac_fate_concordance_k{primary_k}.{suffix}",
            dpi=400,
        )
    plt.close(fig)


def write_results_note(summary: pd.DataFrame, output: Path, primary_k: int) -> None:
    primary = summary[
        summary["atac_program"].eq("ATAC linked peaks") & summary["k"].eq(primary_k)
    ].copy()
    lines = [
        "# Palate CNC particle RNA–ATAC fate concordance",
        "",
        f"Primary molecular decoder: k={primary_k}.",
        "",
        "The continuous fate axis is anterior minus posterior. RNA and linked-peak axes use the same six original-paper marker genes with assayed author links. Lower binary branch discordance and higher Spearman/kappa indicate better same-particle coupling. Definite discordance is retained as a sensitivity metric but not used for ranking because both-confident mass is low.",
        "",
        "## Full trajectories",
        "",
    ]
    full = primary[primary["scope"].eq("full")]
    for _, row in full.sort_values(["stage_time", "weighted_spearman"], ascending=[True, False]).iterrows():
        lines.append(
            f"- {row['stage']} — {row['display_method']}: Spearman {row['weighted_spearman']:.3f}; "
            f"branch discordance {100*row['binary_discordance']:.2f}%; "
            f"both-confident mass {100*row['both_confident_mass']:.1f}%; kappa {row['cohens_kappa']:.3f}."
        )
    lines.extend(["", "## Strict LOO endpoints", ""])
    loo = primary[primary["scope"].eq("strict LOO")]
    for _, row in loo.sort_values(["stage_time", "weighted_spearman"], ascending=[True, False]).iterrows():
        lines.append(
            f"- {row['stage']} — {row['display_method']}: Spearman {row['weighted_spearman']:.3f}; "
            f"branch discordance {100*row['binary_discordance']:.2f}%; "
            f"both-confident mass {100*row['both_confident_mass']:.1f}%; kappa {row['cohens_kappa']:.3f}."
        )
    lines.extend(
        [
            "",
            "TrajectoryNet is not listed because its validated palate trajectory starts from E14.5 terminal cells; treating those IDs as E12.5 CNC particles would not be a same-particle ancestry test.",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.primary_k not in args.k_values:
        raise ValueError("primary-k must be included in k-values")
    if not 0.5 < args.confidence_threshold < 1.0:
        raise ValueError("confidence-threshold must be between 0.5 and 1")
    required = [
        RNA_BENCH,
        ATAC_BENCH,
        RNA_RAW,
        ATAC_RAW,
        MOTIF_RAW,
        PEAK_LINKS,
        RNA_NORM,
        ATAC_NORM,
        COMMON_INITIAL,
        COMMON_X0,
        CB_U_WEIGHTS,
        MIOFLOW_FULL / "model.pt",
        MIOFLOW_FULL / "gaga_model.pt",
        MIOFLOW_FULL / "metadata.json",
        LOO_ROOT / "prediction_manifest.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    targets = (
        args.output_dir / "fate_concordance_summary.csv",
        args.output_dir / "fate_concordance_particle_scores_k5.csv.gz",
        args.output_dir / "observed_program_auc_by_stage.csv",
        args.output_dir / "program_peak_audit.csv",
        args.output_dir / "analysis_manifest.json",
        args.output_dir / "RESULTS.md",
    )
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.set_num_threads(1)
    rna = ad.read_h5ad(RNA_BENCH)
    atac = ad.read_h5ad(ATAC_BENCH)
    if not np.array_equal(rna.obs_names.to_numpy(str), atac.obs_names.to_numpy(str)):
        raise ValueError("Benchmark RNA and ATAC cells are not paired")
    obs_names = rna.obs_names.to_numpy(str)
    labels = rna.obs["celltype_sub"].astype(str).to_numpy()
    times = pd.to_numeric(rna.obs["time_point_processed"], errors="raise").to_numpy(float)
    rna_scale = _load_scale(RNA_NORM)
    atac_scale = _load_scale(ATAC_NORM)
    x_rna = np.asarray(rna.obsm["X_latent"], dtype=np.float32) / rna_scale
    x_atac = np.asarray(atac.obsm["X_lsi15"], dtype=np.float32) / atac_scale

    common_local = np.load(COMMON_INITIAL)
    initial_population = np.flatnonzero(np.isclose(times, 0.0))
    common_global = initial_population[common_local]
    cohort = labels[common_global] == CNC
    if int(cohort.sum()) != 1048:
        raise ValueError(f"Expected 1048 common initial CNC particles, found {cohort.sum()}")
    x0_norm = np.load(COMMON_X0).astype(np.float32, copy=False)
    if not np.array_equal(x0_norm, x_rna[common_global]):
        raise ValueError("Archived common initial RNA differs from benchmark input")

    print("Building fixed RNA, linked-peak and motif fate programs", flush=True)
    program_logits, observed_auc, program_audit = build_fixed_programs(
        obs_names, labels, times
    )
    observed_auc.to_csv(args.output_dir / "observed_program_auc_by_stage.csv", index=False)
    links = pd.read_csv(PEAK_LINKS)
    matched = set(
        program_audit["matched_linked_genes"]["anterior"]
        + program_audit["matched_linked_genes"]["posterior"]
    )
    frozen_links = links[
        links["gene"].isin(matched)
        & (pd.to_numeric(links["zscore"], errors="coerce") > 0)
        & (pd.to_numeric(links["pvalue"], errors="coerce") < 0.05)
    ].copy()
    atac_audit = ad.read_h5ad(ATAC_RAW, backed="r")
    assayed_peaks = set(atac_audit.var_names.astype(str))
    atac_audit.file.close()
    frozen_links = frozen_links[
        frozen_links["peak"].astype(str).isin(assayed_peaks)
    ]
    frozen_links["branch"] = frozen_links["gene"].map(
        {gene: branch for branch, genes in PAPER_MARKERS.items() for gene in genes}
    )
    frozen_links.sort_values(["branch", "gene", "zscore"], ascending=[True, True, False]).to_csv(
        args.output_dir / "program_peak_audit.csv", index=False
    )

    decoders: dict[tuple[int, str], StageDecoder] = {}
    for k in args.k_values:
        decoders[(k, "RNA")] = StageDecoder(x_rna, times, program_logits["RNA"], k)
        for name in ("ATAC linked peaks", "ATAC motifs"):
            decoders[(k, name)] = StageDecoder(x_atac, times, program_logits[name], k)

    film, film_source, film_checkpoint, _ = _load_film(device)
    print("Rolling out the common initial particles with full GAGA10 MIOFlow", flush=True)
    full_methods = load_full_methods(
        args.cy, x0_norm, rna_scale, film, device, args.batch_size
    )
    cb_saved = np.load(CB_U_WEIGHTS, allow_pickle=True)
    cb_weights = {key: np.asarray(cb_saved[key]) for key in cb_saved.files}

    summary_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    source_audit: dict[str, object] = {}
    for method in METHODS_FULL:
        saved = full_methods[method]
        grid = np.asarray(saved["time"], dtype=float)
        source_audit[method] = str(np.asarray(saved["source"]).item())
        for stage_time, stage_name in STAGES:
            position = int(np.argmin(np.abs(grid - stage_time)))
            if not np.isclose(grid[position], stage_time, atol=1e-6):
                raise ValueError(f"{method} has no state at {stage_time}")
            weight = full_weight(method, saved, stage_time, position, cohort, cb_weights)
            for k in args.k_values:
                rows, details = score_one_state(
                    scope="full",
                    scenario="full",
                    method=method,
                    stage_time=stage_time,
                    stage_name=stage_name,
                    rna_points=np.asarray(saved["rna_norm"][position], dtype=np.float32)[cohort],
                    atac_points=np.asarray(saved["atac_norm"][position], dtype=np.float32)[cohort],
                    weight=weight,
                    particle_ids=np.flatnonzero(cohort),
                    decoders=decoders,
                    k=k,
                    threshold=args.confidence_threshold,
                    save_detail=k == args.primary_k,
                )
                summary_rows.extend(rows)
                detail_frames.extend(details)

    print("Scoring strict-LOO held-out endpoints with the shared full T", flush=True)
    loo_manifest = pd.read_csv(LOO_ROOT / "prediction_manifest.csv")
    method_map = {
        "COATI-B": "COATI balanced",
        "COATI-U": "COATI unbalanced",
        "CytoBridge balanced": "CytoBridge balanced",
        "CytoBridge unbalanced": "CytoBridge unbalanced",
        "MIOFlow": "MIOFlow",
        "Unbalanced RNA-only": "Unbalanced RNA-only",
    }
    scenario_stage = {"loo_time1": (1.0, "E13.5"), "loo_time2": (1.5, "E14.0")}
    loo_sources: dict[str, object] = {}
    for item in loo_manifest.itertuples(index=False):
        if str(item.method) not in method_map:
            continue
        method = method_map[str(item.method)]
        scenario = str(item.scenario)
        stage_time, stage_name = scenario_stage[scenario]
        path = LOO_ROOT / str(item.prediction_file)
        with np.load(path, allow_pickle=True) as saved:
            if not np.array_equal(np.asarray(saved["initial_indices"]), common_local):
                raise ValueError(f"{method} {scenario} is not common-initial")
            rna_endpoint = np.asarray(saved["rna_norm"], dtype=np.float32)[cohort]
            atac_endpoint = np.asarray(saved["atac_norm"], dtype=np.float32)[cohort]
            weight = normalize_weight(np.asarray(saved["weights"], dtype=float)[cohort])
        loo_sources[f"{scenario}:{method}"] = {
            "prediction": str(path),
            "sha256": sha256(path),
            "model_artifact": str(item.model_artifact),
            "particle_origin": str(item.particle_origin),
        }
        for k in args.k_values:
            rows, details = score_one_state(
                scope="strict LOO",
                scenario=scenario,
                method=method,
                stage_time=stage_time,
                stage_name=stage_name,
                rna_points=rna_endpoint,
                atac_points=atac_endpoint,
                weight=weight,
                particle_ids=np.flatnonzero(cohort),
                decoders=decoders,
                k=k,
                threshold=args.confidence_threshold,
                save_detail=k == args.primary_k,
            )
            summary_rows.extend(rows)
            detail_frames.extend(details)

    summary = pd.DataFrame(summary_rows)
    observed_controls = score_real_controls(
        program_logits, labels, times, args.confidence_threshold
    )
    summary = pd.concat([summary, observed_controls], ignore_index=True, sort=False)
    summary.to_csv(args.output_dir / "fate_concordance_summary.csv", index=False)
    pd.concat(detail_frames, ignore_index=True).to_csv(
        args.output_dir / f"fate_concordance_particle_scores_k{args.primary_k}.csv.gz",
        index=False,
    )

    plot_full(summary, args.output_dir, args.primary_k)
    plot_loo(summary, args.output_dir, args.primary_k)
    plot_loo_stage(summary, args.output_dir, args.primary_k, "loo_time1", "Held-out E13.5")
    plot_loo_stage(summary, args.output_dir, args.primary_k, "loo_time2", "Held-out E14.0")
    write_results_note(summary, args.output_dir, args.primary_k)
    manifest = {
        "primary_k": args.primary_k,
        "k_sensitivity": list(args.k_values),
        "confidence_threshold": args.confidence_threshold,
        "cohort": "1048 common observed E12.5 particles annotated CNC-derived progenitors",
        "COATI_C_y": args.cy,
        "mass_weighting": (
            "native learned particle mass for COATI-U and CytoBridge-U, renormalized "
            "within the initial CNC cohort at each stage; uniform otherwise"
        ),
        "mioflow_initial_state": (
            "t=0 is explicitly the shared observed E12.5 x0; later states are the "
            "decoded GAGA10 ODE rollout, avoiding an AE-reconstruction mismatch at the source"
        ),
        "atac_evaluation": (
            "all method RNA states use the same frozen full-data time-conditioned "
            "FiLM RNA->ATAC map; strict-LOO COATI checkpoints were trained with "
            "strict LOO T but evaluated here with full T"
        ),
        "program_audit": program_audit,
        "full_sources": source_audit,
        "loo_sources": loo_sources,
        "film_source": str(film_source),
        "film_checkpoint": str(film_checkpoint),
        "trajectorynet_exclusion": (
            "validated current palate TrajectoryNet trajectories originate from observed "
            "E14.5 terminal cells and therefore do not index the common E12.5 CNC cohort"
        ),
        "interpretation_limits": [
            "The metric evaluates program concordance on predicted particles; it is not lineage-tracing ground truth.",
            "ATAC is produced by the common FiLM map, so the result tests whether a method visits states where the shared RNA->ATAC map preserves branch programs.",
            "The ATAC program is noisier than RNA; observed paired-cell AUC and concordance are reported as an assay ceiling/context.",
            "Particle resampling would quantify computational particle variability, not biological replicate uncertainty; no significance claim is made.",
        ],
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
