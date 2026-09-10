#!/usr/bin/env python
"""Validate palate trajectories against the fixed SHOX2 regulatory program.

Every trajectory method is decoded in exactly the same way: at each observed
stage, each predicted particle is assigned to its single nearest real cell in
the normalized RNA PCA40 space and independently in the normalized ATAC LSI15
space.  No method-specific smoothing is used.

The primary biological reference is the fixed 11-target SHOX2 network from
Figure 6 of Xu et al. (2024).  KO effect sizes are recomputed from the six
E14.5 anterior hard-palate bulk RNA-seq samples in GSE129821.  The paper's
reported eight significant, direction-consistent targets are kept fixed as the
primary target set; exploratory p-values below are not substituted for the
paper's DESeq2 analysis.
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
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

from evaluate_palate_loo_same_space import _load_scale  # noqa: E402


PALATE_ROOT = Path("external/COATI/MouseBrain")
RNA_REFERENCE = ROOT / "data/palate_rna_cytobridge.h5ad"
ATAC_REFERENCE = ROOT / "data/palate_atac_benchmark.h5ad"
RNA_RAW = PALATE_ROOT / "data/rna_dimReduced.h5ad"
ATAC_RAW = PALATE_ROOT / "data/atac_dimReduced.h5ad"
RNA_NORM = ROOT / "data/palate_rna_primal_norm_params.pt"
ATAC_NORM = ROOT / "data/palate_atac_secondary_norm_params_lsi15.pt"
FULL_LINKS = PALATE_ROOT / "DataGen/Mouse/raw/peak_gene_pairs_500kb.csv"
GTF = PALATE_ROOT / "DataGen/Mouse/gencode.vM25.annotation.gtf.gz"
DEFAULT_OUTPUT = ROOT / "results/palate_shox2_regulatory_validation"

CNC = "CNC-derived progenitors"
ANTERIOR = "anterior palatal mesenchymal"
POSTERIOR = "posterior palatal mesenchymal"
STAGES = ("E12.5", "E13.5", "E14.0", "E14.5")
TIMES = (0.0, 1.0, 1.5, 2.0)

# CellOracle signs in Figure 6e.  These genes are fixed before looking at any
# trajectory method.  paper_significant is the eight-gene subset in Figure 6f.
TARGETS = (
    ("Satb2", "positive", True),
    ("Ncam1", "positive", True),
    ("Pde7b", "positive", True),
    ("Phldb2", "positive", True),
    ("Marcks", "positive", True),
    ("Prrx1", "positive", True),
    ("Nfia", "positive", False),
    ("Kif26b", "positive", False),
    ("Prickle1", "negative", True),
    ("Lingo2", "negative", True),
    ("Rora", "negative", False),
)
GENES = tuple(row[0] for row in TARGETS)
PAPER_SIGNIFICANT = {row[0] for row in TARGETS if row[2]}
EXPECTED_SIGN = {gene: 1 if relation == "positive" else -1 for gene, relation, _ in TARGETS}


@dataclass(frozen=True)
class Method:
    name: str
    path: Path


METHODS = (
    Method(
        "COATI balanced",
        ROOT
        / "results/palate_trajectory_archive/trajectories/sync/BalancedSync/seed_0/cy_0.3_iter_20000.npz",
    ),
    Method(
        "COATI unbalanced",
        ROOT
        / "results/palate_trajectory_archive/trajectories/sync/FiLMUnbalancedSync/seed_0/cy_0.3_iter_20000.npz",
    ),
    Method(
        "CytoBridge balanced",
        ROOT / "results/palate_trajectory_archive/trajectories/external/cytobridge_balanced.npz",
    ),
    Method(
        "CytoBridge unbalanced",
        ROOT / "results/palate_trajectory_archive/trajectories/external/cytobridge_unbalanced.npz",
    ),
    Method(
        "MIOFlow",
        ROOT / "results/palate_cnc_timecourse_composition/mioflow_common_initial_timecourse.npz",
    ),
    Method(
        "TrajectoryNet",
        ROOT / "results/palate_trajectory_archive/trajectories/external/trajectorynet_terminal_origin.npz",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--links-per-gene", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dense(matrix: object) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def load_ko_counts(external: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(external.glob("GSM*_counts.tab.gz"))
    if len(paths) != 6:
        raise FileNotFoundError(f"Expected six GSE129821 count files in {external}")
    columns: list[pd.Series] = []
    sample_names: list[str] = []
    for path in paths:
        with gzip.open(path, "rt") as handle:
            frame = pd.read_csv(handle, sep="\t", index_col=0)
        sample = "WT" if "_WT_" in path.name else "KO"
        replicate = path.name.split("_")[0]
        columns.append(frame.iloc[:, 0].rename(replicate))
        sample_names.append(sample)
    counts = pd.concat(columns, axis=1).fillna(0).astype(float)
    positive = (counts > 0).all(axis=1)
    log_geo_mean = np.log(counts.loc[positive]).mean(axis=1)
    ratios = np.exp(np.log(counts.loc[positive]).sub(log_geo_mean, axis=0))
    size_factors = ratios.median(axis=0)
    normalized = counts.divide(size_factors, axis=1)
    records: list[dict[str, object]] = []
    for gene in GENES:
        values = normalized.loc[gene].to_numpy(float)
        wt = values[np.asarray(sample_names) == "WT"]
        ko = values[np.asarray(sample_names) == "KO"]
        log2fc = float(np.log2((ko.mean() + 0.5) / (wt.mean() + 0.5)))
        test = stats.ttest_ind(
            np.log2(ko + 0.5), np.log2(wt + 0.5), equal_var=False
        )
        relation = next(row[1] for row in TARGETS if row[0] == gene)
        records.append(
            {
                "gene": gene,
                "celloracle_relation": relation,
                "paper_significant": gene in PAPER_SIGNIFICANT,
                "wt_normalized_mean": float(wt.mean()),
                "ko_normalized_mean": float(ko.mean()),
                "ko_vs_wt_log2fc": log2fc,
                "development_reference": -log2fc,
                "exploratory_welch_pvalue": float(test.pvalue),
            }
        )
    effects = pd.DataFrame(records)
    effects["exploratory_welch_fdr"] = bh_adjust(
        effects["exploratory_welch_pvalue"].to_numpy(float)
    )
    sample = pd.DataFrame(
        {
            "sample": counts.columns,
            "genotype": sample_names,
            "median_ratio_size_factor": size_factors.to_numpy(float),
        }
    )
    return effects, sample


def choose_peak_links(raw_atac: ad.AnnData, n: int) -> pd.DataFrame:
    links = pd.read_csv(FULL_LINKS)
    links = links[
        links["gene"].isin(GENES)
        & (pd.to_numeric(links["score"], errors="coerce") > 0)
        & (pd.to_numeric(links["pvalue"], errors="coerce") < 0.05)
        & links["peak"].isin(raw_atac.var_names)
    ].copy()
    links["source"] = "positive peak-gene correlation"
    selected = (
        links.sort_values(["gene", "score"], ascending=[True, False])
        .groupby("gene", sort=False)
        .head(n)
    )

    # Rora has no significant pair in the archived peak-gene table.  Use the
    # closest assayed peak to the TSS as a clearly marked promoter-proximal
    # fallback rather than silently dropping the target.
    missing = set(GENES) - set(selected["gene"])
    if missing:
        import re

        pattern = re.compile(r'gene_name "([^"]+)"')
        coords: dict[str, tuple[str, int]] = {}
        with gzip.open(GTF, "rt") as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                fields = line.rstrip().split("\t")
                if len(fields) != 9 or fields[2] != "gene":
                    continue
                match = pattern.search(fields[8])
                if match and match.group(1) in missing:
                    tss = int(fields[3]) if fields[6] == "+" else int(fields[4])
                    coords[match.group(1)] = (fields[0], tss)
        peaks = pd.DataFrame(
            {
                "peak": raw_atac.var_names.astype(str),
                "chrom": raw_atac.var["seqnames"].astype(str).to_numpy(),
                "center": (
                    pd.to_numeric(raw_atac.var["start"]).to_numpy(float)
                    + pd.to_numeric(raw_atac.var["end"]).to_numpy(float)
                )
                / 2,
            }
        )
        fallback: list[dict[str, object]] = []
        for gene in sorted(missing):
            chrom, tss = coords[gene]
            local = peaks[peaks["chrom"].eq(chrom)].copy()
            local["distance"] = np.abs(local["center"] - tss)
            for row in local.nsmallest(n, "distance").itertuples(index=False):
                fallback.append(
                    {
                        "gene": gene,
                        "peak": row.peak,
                        "score": np.nan,
                        "pvalue": np.nan,
                        "source": "nearest TSS fallback",
                    }
                )
        selected = pd.concat([selected, pd.DataFrame(fallback)], ignore_index=True)
    return selected.sort_values(["gene", "source", "score"], na_position="last").reset_index(drop=True)


def gene_accessibility(
    raw_atac: ad.AnnData, obs_names: pd.Index, links: pd.DataFrame
) -> np.ndarray:
    row_index = raw_atac.obs_names.get_indexer(obs_names)
    if np.any(row_index < 0):
        raise ValueError("ATAC raw cells do not align to benchmark cells")
    result = np.zeros((len(obs_names), len(GENES)), dtype=np.float32)
    for gene_index, gene in enumerate(GENES):
        peaks = links.loc[links["gene"].eq(gene), "peak"].tolist()
        positions = raw_atac.var_names.get_indexer(peaks)
        block = dense(raw_atac.X[row_index, :][:, positions])
        result[:, gene_index] = np.mean(block > 0, axis=1)
    return result


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return np.nan
    return float(stats.spearmanr(left, right).statistic)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_files = (
        args.output_dir / "ko_rna_effects.csv",
        args.output_dir / "selected_peak_gene_links.csv",
        args.output_dir / "trajectory_1nn_dynamics.csv",
        args.output_dir / "trajectory_1nn_changes.csv",
        args.output_dir / "trajectory_1nn_summary.csv",
        args.output_dir / "trajectory_selection_audit.csv",
        args.output_dir / "analysis_manifest.json",
    )
    existing = [path for path in output_files if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")
    required = [
        RNA_REFERENCE,
        ATAC_REFERENCE,
        RNA_RAW,
        ATAC_RAW,
        RNA_NORM,
        ATAC_NORM,
        FULL_LINKS,
        GTF,
        *(method.path for method in METHODS),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))

    ko, ko_samples = load_ko_counts(args.output_dir / "external")
    ko.to_csv(args.output_dir / "ko_rna_effects.csv", index=False)
    ko_samples.to_csv(args.output_dir / "ko_rna_samples.csv", index=False)

    rna = ad.read_h5ad(RNA_REFERENCE, backed="r")
    atac = ad.read_h5ad(ATAC_REFERENCE, backed="r")
    raw_rna = ad.read_h5ad(RNA_RAW, backed="r")
    raw_atac = ad.read_h5ad(ATAC_RAW, backed="r")
    if not np.array_equal(rna.obs_names, atac.obs_names):
        raise ValueError("RNA and ATAC benchmark cells are not aligned")
    rna_norm = np.asarray(rna.obsm["X_latent"], dtype=np.float32) / _load_scale(RNA_NORM)
    atac_norm = np.asarray(atac.obsm["X_lsi15"], dtype=np.float32) / _load_scale(ATAC_NORM)
    labels = rna.obs["celltype_sub"].astype(str).to_numpy()
    times = pd.to_numeric(rna.obs["time_point_processed"], errors="raise").to_numpy(float)

    raw_rows = raw_rna.obs_names.get_indexer(rna.obs_names)
    gene_positions = raw_rna.raw.var_names.get_indexer(GENES)
    if np.any(raw_rows < 0) or np.any(gene_positions < 0):
        raise ValueError("RNA raw cells or fixed SHOX2 genes are missing")
    expression = dense(raw_rna.raw.X[raw_rows, :][:, gene_positions]).astype(np.float32)
    expression_sd = np.maximum(expression.std(axis=0), 1e-6)
    links = choose_peak_links(raw_atac, args.links_per_gene)
    links.to_csv(args.output_dir / "selected_peak_gene_links.csv", index=False)
    accessibility = gene_accessibility(raw_atac, rna.obs_names, links)

    stage_rows = [np.flatnonzero(np.isclose(times, time)) for time in TIMES]
    rna_nn = [NearestNeighbors(n_neighbors=1).fit(rna_norm[rows]) for rows in stage_rows]
    atac_nn = [NearestNeighbors(n_neighbors=1).fit(atac_norm[rows]) for rows in stage_rows]
    ko_lookup = ko.set_index("gene")

    dynamic_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for method in METHODS:
        print(f"Decoding {method.name} with stage-matched 1NN", flush=True)
        with np.load(method.path, allow_pickle=False) as saved:
            trajectory_time = np.asarray(saved["time"], dtype=float)
            rna_trajectory = np.asarray(saved["rna_norm"], dtype=np.float32)
            atac_trajectory = np.asarray(saved["atac_norm"], dtype=np.float32)
        trajectory_indices = [int(np.argmin(np.abs(trajectory_time - time))) for time in TIMES]
        rna_indices: list[np.ndarray] = []
        atac_indices: list[np.ndarray] = []
        for stage_index, trajectory_index in enumerate(trajectory_indices):
            local_rna = rna_nn[stage_index].kneighbors(
                rna_trajectory[trajectory_index], return_distance=False
            )[:, 0]
            local_atac = atac_nn[stage_index].kneighbors(
                atac_trajectory[trajectory_index], return_distance=False
            )[:, 0]
            rna_indices.append(stage_rows[stage_index][local_rna])
            atac_indices.append(stage_rows[stage_index][local_atac])

        initial_cnc = (labels[rna_indices[0]] == CNC) & (labels[atac_indices[0]] == CNC)
        terminal_masks = {
            "anterior": (labels[rna_indices[-1]] == ANTERIOR)
            & (labels[atac_indices[-1]] == ANTERIOR),
            "posterior": (labels[rna_indices[-1]] == POSTERIOR)
            & (labels[atac_indices[-1]] == POSTERIOR),
        }
        for branch, terminal_mask in terminal_masks.items():
            selected = initial_cnc & terminal_mask
            audit_rows.append(
                {
                    "method": method.name,
                    "branch": branch,
                    "n_particles": int(len(selected)),
                    "n_initial_consensus_cnc": int(initial_cnc.sum()),
                    "n_terminal_consensus": int(terminal_mask.sum()),
                    "n_selected_trajectories": int(selected.sum()),
                    "selected_fraction": float(selected.mean()),
                }
            )
            for stage_index, (stage, time) in enumerate(zip(STAGES, TIMES)):
                rna_cells = rna_indices[stage_index][selected]
                atac_cells = atac_indices[stage_index][selected]
                if len(rna_cells) == 0:
                    continue
                rna_mean = expression[rna_cells].mean(axis=0)
                atac_mean = accessibility[atac_cells].mean(axis=0)
                for gene_index, gene in enumerate(GENES):
                    dynamic_rows.append(
                        {
                            "method": method.name,
                            "branch": branch,
                            "stage": stage,
                            "time": time,
                            "gene": gene,
                            "paper_significant": gene in PAPER_SIGNIFICANT,
                            "celloracle_relation": ko_lookup.loc[gene, "celloracle_relation"],
                            "expected_development_sign": EXPECTED_SIGN[gene],
                            "n_selected_trajectories": int(selected.sum()),
                            "rna_1nn_mean": float(rna_mean[gene_index]),
                            "rna_1nn_mean_z": float(rna_mean[gene_index] / expression_sd[gene_index]),
                            "atac_1nn_linked_accessibility": float(atac_mean[gene_index]),
                        }
                    )

    dynamics = pd.DataFrame(dynamic_rows)
    audit = pd.DataFrame(audit_rows)
    dynamics.to_csv(args.output_dir / "trajectory_1nn_dynamics.csv", index=False)
    audit.to_csv(args.output_dir / "trajectory_selection_audit.csv", index=False)

    first = dynamics[dynamics["stage"].eq(STAGES[0])].set_index(
        ["method", "branch", "gene"]
    )
    last = dynamics[dynamics["stage"].eq(STAGES[-1])].set_index(
        ["method", "branch", "gene"]
    )
    changes = last[
        ["paper_significant", "celloracle_relation", "expected_development_sign", "n_selected_trajectories"]
    ].copy()
    changes["rna_change"] = last["rna_1nn_mean"] - first["rna_1nn_mean"]
    changes["rna_change_z"] = last["rna_1nn_mean_z"] - first["rna_1nn_mean_z"]
    changes["atac_change"] = (
        last["atac_1nn_linked_accessibility"] - first["atac_1nn_linked_accessibility"]
    )
    changes = changes.reset_index().merge(
        ko[["gene", "ko_vs_wt_log2fc", "development_reference"]], on="gene", how="left"
    )
    changes["ko_direction_correct"] = (
        np.sign(changes["rna_change_z"]) == changes["expected_development_sign"]
    )
    changes["rna_atac_sign_concordant"] = (
        np.sign(changes["rna_change_z"]) == np.sign(changes["atac_change"])
    )
    changes.to_csv(args.output_dir / "trajectory_1nn_changes.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    for (method, branch), local in changes.groupby(["method", "branch"], sort=False):
        primary = local[local["paper_significant"]].copy()
        linked = local[np.isfinite(local["atac_change"])].copy()
        summary_rows.append(
            {
                "method": method,
                "branch": branch,
                "n_selected_trajectories": int(local["n_selected_trajectories"].iloc[0]),
                "ko_sign_accuracy_8": float(primary["ko_direction_correct"].mean()),
                "ko_spearman_8": spearman(
                    primary["rna_change_z"].to_numpy(float),
                    primary["development_reference"].to_numpy(float),
                ),
                "ko_spearman_11": spearman(
                    local["rna_change_z"].to_numpy(float),
                    local["development_reference"].to_numpy(float),
                ),
                "rna_atac_sign_concordance_8": float(
                    primary["rna_atac_sign_concordant"].mean()
                ),
                "rna_atac_sign_concordance_11": float(
                    linked["rna_atac_sign_concordant"].mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "trajectory_1nn_summary.csv", index=False)

    manifest = {
        "decoder": (
            "Exactly one nearest real cell per particle, separately in stage-matched "
            "normalized RNA PCA40 and ATAC LSI15 spaces; no smoothing and uniform "
            "particle contribution for every method."
        ),
        "trajectory_selection": (
            "Same particle must be consensus CNC at E12.5 and consensus anterior or "
            "posterior at E14.5, where consensus means the independent RNA and ATAC "
            "1NN labels agree. Particle ID is then followed through all four stages."
        ),
        "ko_rna": {
            "accession": "GSE129821",
            "design": "E14.5 anterior hard palate; 3 WT and 3 Shox2 knockout bulk RNA-seq samples",
            "effect": "median-of-ratios normalized KO-versus-WT log2 fold change",
            "primary_significance": (
                "Fixed eight targets reported significant and direction-consistent in "
                "Xu et al. Figure 6f; exploratory Welch FDR is an audit only."
            ),
        },
        "atac_readout": (
            f"Mean binary accessibility of the top {args.links_per_gene} positive, "
            "p<0.05 archived peak-gene correlations per target. Rora, which has no "
            "archived significant pair, uses the nearest TSS peaks and is marked as fallback."
        ),
        "methods": [method.name for method in METHODS],
        "genes": list(GENES),
        "paper_significant_genes": sorted(PAPER_SIGNIFICANT),
        "caveat": (
            "The 1NN readout validates where trajectories travel in observed molecular "
            "space; it does not prove that a decoded real cell is the particle's true lineage descendant."
        ),
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
