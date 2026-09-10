#!/usr/bin/env python3
"""HC07 seven-time growth programs with realized and instantaneous growth.

Primary inference uses realized interval log-mass change.  Instantaneous growth
is reported separately as a secondary model diagnostic.  The four cell lines,
not particles or C_y settings, are the uncertainty units.  kNN is frozen at 15
in the shared seven-time normalized RNA PCA30 space.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
HUMAN = Path("external/COATI/humanCerebral")
DATA = HUMAN / "Data/selected_4_7_9_11_12_18_21"
MAT = ROOT / "results/human_cerebral_full_biological_interpretability/01_trajectory_materialization"
OUTPUT = ROOT / "results/human_cerebral_full_biological_interpretability/03_mass_growth"
COATI = HUMAN / "UnbalancedSync_7time_D4_D21_no_D16_lsi12_FiLM_biological_num/hpc/trajectory_iter40000"
K = 15
STAGES = ("D4", "D7", "D9", "D11", "D12", "D18", "D21")
TIMES = np.asarray((0.0, 0.3, 0.5, 0.7, 0.8, 1.4, 1.7), float)
INTERVALS = list(zip(STAGES[:-1], STAGES[1:]))

# Frozen before opening growth scores.  These are programme-level hypotheses;
# absent genes are reported rather than replaced.
MODULES = {
    "S phase": ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"],
    "G2/M": ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"],
    "Proliferation": ["MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1", "CCNB2", "CDC20", "UBE2C", "BIRC5", "TYMS", "MCM2", "MCM4", "MCM5", "MCM6"],
    "Apoptosis/stress": ["BAX", "BAK1", "BBC3", "PMAIP1", "CASP3", "CASP7", "FAS", "DDIT3", "ATF4", "HSPA1A", "JUN", "FOS"],
    "Early neural ectoderm": ["SIX3", "SOX3", "HES5", "CDH2"],
    "Telencephalic": ["FGF8", "FOXG1", "PAX6", "GLI3"],
    "Non-telencephalic": ["WLS", "WNT8B", "IRX5"],
    "Dorsal": ["EMX1", "EMX2", "BMP7", "NEUROD6", "PAX6"],
    "Ventral": ["DLX1", "DLX2", "GSX2", "ASCL1", "NKX2-1"],
    "Transient pre-DV": ["DCT", "DIO3", "SIX6"],
    "Radial glia/progenitor": ["SOX2", "PAX6", "HES1", "HES5", "VIM", "NES", "FABP7", "GLI3", "HOPX", "SLC1A3"],
    "IPC/neurogenesis": ["EOMES", "NEUROG2", "ASCL1", "HES6", "INSM1", "NEUROD1", "NEUROD4", "ELAVL4", "DCX", "TUBB3"],
    "Excitatory maturation": ["NEUROD2", "TBR1", "BCL11B", "SATB2", "CUX1", "CUX2", "STMN2", "RBFOX3", "MAP2", "SYT1"],
    "Inhibitory maturation": ["DLX1", "DLX2", "DLX5", "DLX6", "GAD1", "GAD2", "LHX6", "ERBB4", "SST", "CALB2"],
}
CELL_CYCLE_MODULES = {"S phase", "G2/M", "Proliferation"}
LATE_LINEAGE_MODULES = {"Telencephalic", "Non-telencephalic", "Dorsal", "Ventral", "IPC/neurogenesis", "Excitatory maturation", "Inhibitory maturation", "Transient pre-DV"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--module-permutations", type=int, default=200)
    parser.add_argument("--min-cohort", type=int, default=30)
    return parser.parse_args()


def load_tensor(path: Path) -> np.ndarray:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise ValueError(path)
    return value


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    result = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return result
    values = p[valid]
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    result[valid] = np.minimum(adjusted[inverse], 1.0)
    return result


def covariate_matrix(cov: pd.DataFrame, include_line: bool) -> np.ndarray:
    numeric = cov[["log_library", "S_score", "G2M_score"]].to_numpy(float)
    columns = [np.ones(len(cov)), *[numeric[:, i] for i in range(numeric.shape[1])]]
    if include_line:
        lines = cov["line"].astype(str).to_numpy()
        levels = sorted(np.unique(lines).tolist())
        columns.extend((lines == value).astype(float) for value in levels[:-1])
    matrix = np.column_stack(columns)
    keep = np.std(matrix, axis=0) > 1e-10
    keep[0] = True
    return matrix[:, keep]


def residual_rank(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    ranked = stats.rankdata(values, axis=0)
    q, _ = np.linalg.qr(design, mode="reduced")
    return ranked - q @ (q.T @ ranked)


def partial_corr_vector(x: np.ndarray, y: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xr = residual_rank(np.asarray(x, float), design)
    yr = residual_rank(np.asarray(y, float).reshape(-1, 1), design).reshape(-1)
    denominator = np.sqrt(np.sum(xr * xr, axis=0) * np.sum(yr * yr))
    rho = np.divide(xr.T @ yr, denominator, out=np.full(xr.shape[1], np.nan), where=denominator > 0)
    df = max(len(y) - design.shape[1] - 2, 1)
    t_value = rho * np.sqrt(df / np.maximum(1.0 - rho * rho, 1e-12))
    p = 2 * stats.t.sf(np.abs(t_value), df)
    return rho, p


def mapped_values(
    query: np.ndarray,
    reference: np.ndarray,
    expression: sp.csr_matrix,
    covariates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distances, indices = NearestNeighbors(n_neighbors=K, n_jobs=1).fit(reference).kneighbors(query)
    positive = distances[distances > 0]
    floor = float(np.median(positive)) * 1e-3 if positive.size else 1e-8
    weights = 1.0 / np.maximum(distances, max(floor, 1e-8))
    weights /= weights.sum(axis=1, keepdims=True)
    mapped = None
    for neighbor in range(K):
        part = expression[indices[:, neighbor]].multiply(weights[:, neighbor, None])
        mapped = part if mapped is None else mapped + part
    mapped_cov = np.sum(weights[..., None] * covariates[indices], axis=1)
    return np.asarray(mapped.toarray(), np.float32), mapped_cov


def model_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for cy in np.arange(0.1, 0.9, 0.1):
        tag = f"{cy:.1f}"
        records.append(
            {
                "model_id": f"coati_sync_unbalanced_cy{tag}", "method": "COATI-U Sync", "c_y": float(cy),
                "rna": COATI / f"primary_trajectory_sync_7time_s0_a{tag}_iter40000.pt",
                "mass": COATI / f"mass_lnw_trajectory_sync_7time_s0_a{tag}_iter40000.pt",
                "growth": MAT / "coati_unbalanced_sync_40000_audit" / f"instantaneous_growth_rate_sync_7time_s0_a{tag}_iter40000.pt",
                "grid": COATI / "t_grid_s0_iter40000.pt", "gene_level": True,
            }
        )
    rna = MAT / "coati_rna_only_unbalanced_biological_prior_30000"
    records.append(
        {"model_id": "coati_rna_only_unbalanced", "method": "COATI-U RNA-only", "c_y": np.nan,
         "rna": rna / "primary_trajectory_rna_only_full_s0_iter30000.pt",
         "mass": rna / "mass_lnw_trajectory_rna_only_full_s0_iter30000.pt",
         "growth": rna / "instantaneous_growth_rate_rna_only_full_s0_iter30000.pt",
         "grid": rna / "t_grid_rna_only_full_s0_iter30000.pt", "gene_level": True}
    )
    cyto = MAT / "cytobridge_unbalanced_biological_prior_30000"
    records.append(
        {"model_id": "cytobridge_unbalanced", "method": "CytoBridge-U", "c_y": np.nan,
         "rna": cyto / "trajectory_native_normalized_pca30.pt",
         "mass": cyto / "trajectory_native_log_mass.pt",
         "growth": cyto / "trajectory_native_instantaneous_growth_rate.pt",
         "grid": cyto / "physical_time_grid.pt", "gene_level": False}
    )
    for record in records:
        for key in ("rna", "mass", "growth", "grid"):
            if not Path(record[key]).is_file():
                raise FileNotFoundError(record[key])
    return records


def module_scores(expression: np.ndarray, genes: list[str]) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    lookup = {gene: i for i, gene in enumerate(genes)}
    standardized = (expression - expression.mean(axis=0, keepdims=True)) / np.maximum(expression.std(axis=0, keepdims=True), 1e-6)
    scores: dict[str, np.ndarray] = {}
    present: dict[str, list[str]] = {}
    for module, requested in MODULES.items():
        selected = [gene for gene in requested if gene in lookup]
        present[module] = selected
        scores[module] = standardized[:, [lookup[gene] for gene in selected]].mean(axis=1) if selected else np.full(len(expression), np.nan)
    return scores, present


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(DATA / "paired_obs.csv")
    source = paired[paired.processed_age.eq(4)].reset_index(drop=True)
    source_lines = source.line.astype(str).to_numpy()
    assignments_path = args.output_dir / "particle_in_stay_go_assignments.csv.gz"
    assignments = pd.read_csv(assignments_path) if assignments_path.is_file() else pd.DataFrame()
    with np.load(DATA / "rna_pca30_normalized_by_time.npz") as payload:
        ref_pca = {key: np.asarray(payload[key], np.float32) for key in payload.files}

    hvg = ad.read_h5ad(HUMAN / "Data/rna_dimReduced.h5ad", backed="r")
    hvg_genes = [str(gene) for gene in hvg.var_names]
    hvg.file.close()
    requested_modules = sorted({gene for genes in MODULES.values() for gene in genes})
    expression_genes = list(dict.fromkeys([*hvg_genes, *requested_modules]))
    brain = ad.read_h5ad(HUMAN / "Data/humanBrainRNA.h5ad", backed="r")
    expression_genes = [gene for gene in expression_genes if gene in brain.var_names]
    hvg_indices = np.asarray([expression_genes.index(gene) for gene in hvg_genes if gene in expression_genes], int)
    hvg_kept = [gene for gene in hvg_genes if gene in expression_genes]

    growth_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    gene_rows: list[dict[str, Any]] = []
    model_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for record in model_records():
        model_cache[record["model_id"]] = (
            load_tensor(record["rna"]),
            load_tensor(record["mass"]).reshape(66, 501),
            load_tensor(record["growth"]).reshape(66, 501),
            load_tensor(record["grid"]).reshape(-1),
        )

    for interval_index, (stage0, stage1) in enumerate(INTERVALS):
        age0 = int(stage0[1:])
        stage_obs = paired[paired.processed_age.eq(age0)].reset_index(drop=True)
        ids = stage_obs.paired_metacell_id.astype(str).tolist()
        obs_positions = brain.obs_names.get_indexer(ids)
        if (obs_positions < 0).any():
            raise ValueError(f"Expression IDs missing for {stage0}")
        stage_expression = brain[obs_positions, expression_genes].X
        if hasattr(stage_expression, "to_memory"):
            stage_expression = stage_expression.to_memory()
        stage_expression = sp.csr_matrix(stage_expression)
        stage_cov = np.column_stack(
            [
                np.log1p(pd.to_numeric(stage_obs.nCount_RNA, errors="coerce").fillna(0).to_numpy(float)),
                pd.to_numeric(stage_obs["S.Score"], errors="coerce").fillna(0).to_numpy(float),
                pd.to_numeric(stage_obs["G2M.Score"], errors="coerce").fillna(0).to_numpy(float),
            ]
        )
        print(f"[{stage0}→{stage1}] reference {stage_expression.shape}", flush=True)
        for record in model_records():
            trajectory, log_mass, growth, grid = model_cache[record["model_id"]]
            i0 = int(np.argmin(abs(grid - TIMES[interval_index])))
            i1 = int(np.argmin(abs(grid - TIMES[interval_index + 1])))
            mapped_expression, mapped_cov = mapped_values(
                trajectory[i0], ref_pca[f"age_{age0}"], stage_expression, stage_cov
            )
            realized = log_mass[i1] - log_mass[i0]
            instantaneous = growth[i0 : i1 + 1].mean(axis=0)
            scores, present = module_scores(mapped_expression, expression_genes)
            interval_name = f"{stage0}→{stage1}"
            group_specs: list[tuple[str, str | None, np.ndarray]] = [("all", None, np.ones(501, bool))]
            if not assignments.empty and interval_name in {"D12→D18", "D18→D21"}:
                local_assign = assignments[
                    assignments.model_id.eq(record["model_id"])
                    & assignments.interval.eq(interval_name)
                ]
                for target in sorted(local_assign.target.unique()):
                    target_frame = local_assign[local_assign.target.eq(target)].sort_values("source_particle")
                    if len(target_frame) != 501:
                        continue
                    groups = target_frame.hard_group.to_numpy()
                    for group in ("stay", "in", "go"):
                        group_specs.append((group, target, groups == group))
            for particle in range(501):
                growth_rows.append(
                    {
                        "model_id": record["model_id"], "method": record["method"], "c_y": record["c_y"],
                        "interval": interval_name, "source_particle": particle,
                        "source_metacell_id": source.loc[particle, "paired_metacell_id"],
                        "source_line": source_lines[particle],
                        "realized_log_mass_change": float(realized[particle]),
                        "mean_instantaneous_growth": float(instantaneous[particle]),
                        "mapped_log_library": float(mapped_cov[particle, 0]),
                        "mapped_S_score": float(mapped_cov[particle, 1]),
                        "mapped_G2M_score": float(mapped_cov[particle, 2]),
                    }
                )
            for group, target, selector in group_specs:
                line_levels = ["ALL", *sorted(np.unique(source_lines[selector]).tolist())]
                for line in line_levels:
                    cohort = selector & (np.ones(501, bool) if line == "ALL" else source_lines == line)
                    if cohort.sum() < args.min_cohort:
                        continue
                    cov = pd.DataFrame(
                        {
                            "log_library": mapped_cov[cohort, 0], "S_score": mapped_cov[cohort, 1],
                            "G2M_score": mapped_cov[cohort, 2], "line": source_lines[cohort],
                        }
                    )
                    design = covariate_matrix(cov, include_line=line == "ALL")
                    for outcome_name, outcome in [("realized", realized), ("instantaneous", instantaneous)]:
                        y = outcome[cohort]
                        for module, values in scores.items():
                            x = values[cohort]
                            if not np.isfinite(x).all() or np.std(x) < 1e-10:
                                rho, p = np.nan, np.nan
                            else:
                                rho_v, p_v = partial_corr_vector(x[:, None], y, design)
                                rho, p = float(rho_v[0]), float(p_v[0])
                            module_rows.append(
                                {
                                    "model_id": record["model_id"], "method": record["method"], "c_y": record["c_y"],
                                    "interval": interval_name, "phase": "early" if interval_index <= 1 else ("late" if interval_index >= 4 else "middle"),
                                    "target_fate": target, "in_stay_go": group, "line": line,
                                    "outcome": outcome_name, "module": module,
                                    "module_class": "cell_cycle" if module in CELL_CYCLE_MODULES else ("late_lineage" if module in LATE_LINEAGE_MODULES else "other"),
                                    "genes_present": ";".join(present[module]), "n_genes_present": len(present[module]),
                                    "n_particles": int(cohort.sum()), "partial_spearman": rho, "p_value": p,
                                }
                            )
                        if record["gene_level"] and group == "all" and line == "ALL":
                            rho, p_value = partial_corr_vector(mapped_expression[cohort][:, hvg_indices], y, design)
                            for gene, rho_value, p_value_one in zip(hvg_kept, rho, p_value):
                                gene_rows.append(
                                    {
                                        "model_id": record["model_id"], "method": record["method"], "c_y": record["c_y"],
                                        "interval": interval_name, "outcome": outcome_name, "gene": gene,
                                        "n_particles": int(cohort.sum()), "partial_spearman": float(rho_value),
                                        "p_value": float(p_value_one),
                                    }
                                )
        del stage_expression
    brain.file.close()

    growth_frame = pd.DataFrame(growth_rows)
    modules = pd.DataFrame(module_rows)
    genes = pd.DataFrame(gene_rows)
    modules["q_value_within_setting"] = modules.groupby(
        ["model_id", "interval", "target_fate", "in_stay_go", "line", "outcome"], dropna=False
    )["p_value"].transform(lambda x: bh_fdr(x.to_numpy(float)))
    genes["q_value_within_setting"] = genes.groupby(
        ["model_id", "interval", "outcome"], dropna=False
    )["p_value"].transform(lambda x: bh_fdr(x.to_numpy(float)))
    growth_frame.to_csv(args.output_dir / "instantaneous_and_realized_growth.csv.gz", index=False)
    modules.to_csv(args.output_dir / "growth_module_associations_by_setting.csv.gz", index=False)
    genes.to_csv(args.output_dir / "growth_gene_associations_by_setting.csv.gz", index=False)

    coati_modules = modules[
        modules.method.eq("COATI-U Sync") & modules.line.eq("ALL") & modules.in_stay_go.eq("all")
    ]
    module_summary = (
        coati_modules.groupby(["interval", "outcome", "module", "module_class"], as_index=False)
        .agg(
            median_partial_spearman=("partial_spearman", "median"),
            min_partial_spearman=("partial_spearman", "min"),
            max_partial_spearman=("partial_spearman", "max"),
            positive_cy_fraction=("partial_spearman", lambda x: float(np.mean(np.asarray(x) > 0))),
            significant_cy_fraction=("q_value_within_setting", lambda x: float(np.mean(np.asarray(x) <= 0.10))),
        )
    )
    module_summary.to_csv(args.output_dir / "growth_gene_summary_across_cy.csv.gz", index=False)

    # Line-level early/late contrasts. C_y is first collapsed within line, so it
    # cannot masquerade as replication.
    line_modules = modules[
        modules.method.eq("COATI-U Sync") & modules.in_stay_go.eq("all") & modules.line.ne("ALL")
    ].copy()
    phase = (
        line_modules[line_modules.phase.isin(["early", "late"])]
        .groupby(["line", "c_y", "outcome", "module", "module_class", "phase"], as_index=False)
        .partial_spearman.median()
        .pivot(index=["line", "c_y", "outcome", "module", "module_class"], columns="phase", values="partial_spearman")
        .reset_index()
    )
    phase["late_minus_early"] = phase.get("late", np.nan) - phase.get("early", np.nan)
    phase["expected_contrast"] = np.where(
        phase.module_class.eq("cell_cycle"), -phase.late_minus_early,
        np.where(phase.module_class.eq("late_lineage"), phase.late_minus_early, np.nan),
    )
    phase.to_csv(args.output_dir / "growth_module_early_late_contrasts.csv", index=False)

    # Gene-set enrichment against the complete HVG ranking; RNA-only is the
    # same-prior comparator.  This is an enrichment diagnostic, not a replicate test.
    enrichment_rows: list[dict[str, Any]] = []
    late_gene_set = set(gene for module in LATE_LINEAGE_MODULES for gene in MODULES[module])
    for outcome in ("realized", "instantaneous"):
        late = genes[genes.interval.isin(["D12→D18", "D18→D21"]) & genes.outcome.eq(outcome)]
        for method in ("COATI-U Sync", "COATI-U RNA-only"):
            local = late[late.method.eq(method)].groupby("gene", as_index=False).partial_spearman.median()
            labels = local.gene.isin(late_gene_set).astype(int).to_numpy()
            auc = float(roc_auc_score(labels, local.partial_spearman)) if np.unique(labels).size == 2 else np.nan
            enrichment_rows.append({"outcome": outcome, "method": method, "late_lineage_gene_enrichment_auroc": auc, "n_positive_genes": int(labels.sum()), "n_genes": len(labels)})
    enrichment = pd.DataFrame(enrichment_rows)
    enrichment.to_csv(args.output_dir / "growth_module_enrichment_comparison.csv", index=False)

    # Compact heatmap of median COATI-U programme associations.
    heat = module_summary[module_summary.outcome.eq("realized")].pivot(index="module", columns="interval", values="median_partial_spearman")
    heat = heat.reindex(columns=[f"{a}→{b}" for a, b in INTERVALS])
    fig, ax = plt.subplots(figsize=(8.0, 5.6), constrained_layout=True)
    image = ax.imshow(heat.to_numpy(float), aspect="auto", cmap="RdBu_r", vmin=-0.4, vmax=0.4)
    ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set_title("COATI-U realized growth: partial Spearman (median across $C_y$)")
    fig.colorbar(image, ax=ax, label="partial Spearman")
    fig.savefig(args.output_dir / "growth_gene_program_heatmap.pdf")
    fig.savefig(args.output_dir / "growth_gene_program_heatmap.png", dpi=200)
    plt.close(fig)

    manifest = {
        "analysis": "HC07 seven-time growth-associated programmes",
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "knn_k": K,
        "mapping": "stage-conditioned inverse-distance kNN in normalized RNA PCA30",
        "primary_outcome": "realized interval log-mass change",
        "secondary_outcome": "mean instantaneous growth head over the same interval",
        "covariates": ["D4 source line", "mapped log RNA library size", "mapped S score", "mapped G2M score"],
        "uncertainty_unit": "biological line; C_y is sensitivity only",
        "expression": "observed log-normalized humanBrainRNA expression barycentrically read out from 15 neighbors",
        "genes": {"hvg_tested": len(hvg_kept), "modules": MODULES},
        "claim_ceiling": "model-associated allocation programme; not measured proliferation or causal growth genes",
        "note": "module analytical q-values are particle-level screens; line-direction stability is required for any biological interpretation",
    }
    (args.output_dir / "growth_analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote HC07 to {args.output_dir}")


if __name__ == "__main__":
    main()
