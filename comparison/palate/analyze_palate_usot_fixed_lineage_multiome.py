#!/usr/bin/env python
"""Fixed external lineage-program validation of palate USOT growth.

The same lineage programs are used at every observed stage.  Within each
stage-by-current-cell-type cohort, the script tests whether USOT native
instantaneous growth is positively associated with three paired readouts:

1. RNA expression of fixed lineage marker genes;
2. accessibility of positively linked ATAC peaks for those genes;
3. accessibility of available lineage-relevant TF motifs.

The programs are frozen from the palate literature and do not depend on any
USOT growth result or on stage-specific differential-expression selection.
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
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import fisher_exact, rankdata, t as student_t


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

from analyze_gastrulation_preserved_growth_genes import bh_fdr  # noqa: E402


PALATE_ROOT = Path("external/COATI/MouseBrain")
DATA_ROOT = PALATE_ROOT / "data"
RNA_FILE = DATA_ROOT / "rna_dimReduced.h5ad"
ATAC_FILE = DATA_ROOT / "atac_dimReduced.h5ad"
MOTIF_FILE = DATA_ROOT / "CNC_motif.h5ad"
PEAK_LINKS = (
    PALATE_ROOT / "DataGen/Mouse/marker_gene_peak_links.csv"
)
GROWTH_FILE = (
    ROOT
    / "results/palate_usot_growth_programs_by_stage/"
    "per_cell_usot_instantaneous_growth.csv.gz"
)
COHORT_FILE = (
    ROOT
    / "results/palate_usot_growth_programs_by_stage/"
    "stage_celltype_cohorts.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "results/palate_usot_fixed_lineage_multiome"
)

STAGES = ("E12.5", "E13.5", "E14.0", "E14.5")
CY_VALUES = tuple(round(i / 10.0, 1) for i in range(1, 10))
GROWTH_COLUMNS = tuple(
    f"usot_sync_c_y_{str(c_y).replace('.', '_')}__growth_raw"
    for c_y in CY_VALUES
)

LINEAGE_PROGRAMS: dict[str, dict[str, object]] = {
    "anterior": {
        "celltype": "anterior palatal mesenchymal",
        "genes": [
            "Shox2",
            "Satb2",
            "Inhba",
            "Cyp26b1",
            "Nrp1",
            "Alx1",
            "Msx1",
        ],
        "motifs": ["Shox2", "Alx1"],
        "source": (
            "Yan et al. Nature Communications 2024: anterior markers "
            "validated by bulk RNA, qRT-PCR and RNAscope; Alx1 subtype marker."
        ),
        "source_url": "https://www.nature.com/articles/s41467-024-45199-x",
    },
    "posterior": {
        "celltype": "posterior palatal mesenchymal",
        "genes": [
            "Meox2",
            "Prickle1",
            "Sim2",
            "Efnb2",
            "Trps1",
            "Tbx22",
        ],
        "motifs": ["Dlx1", "Dlx2"],
        "source": (
            "Yan et al. Nature Communications 2024: posterior markers "
            "validated by bulk RNA, qRT-PCR and RNAscope; Dlx1/2 reported "
            "as posterior-trajectory regulators."
        ),
        "source_url": "https://www.nature.com/articles/s41467-024-45199-x",
    },
    "dental": {
        "celltype": "dental mesenchymal",
        "genes": ["Dlx2", "Sostdc1", "Tfap2b", "Msx1", "Runx2"],
        "motifs": ["Dlx2"],
        "source": (
            "Yan et al. Nature Communications 2024: dental subtype markers "
            "Dlx2/Sostdc1/Tfap2b and dental-trajectory regulators Msx1/Runx2."
        ),
        "source_url": "https://www.nature.com/articles/s41467-024-45199-x",
    },
    "osteogenic": {
        "celltype": "osteogenic",
        "genes": ["Runx2", "Sp7", "Sox6", "Alpl", "Satb2"],
        "motifs": ["Sox6"],
        "source": (
            "Yan et al. Nature Communications 2024 and palate osteogenesis "
            "literature: Runx2/Sp7 osteoblast markers with Sox6/Alpl/Satb2."
        ),
        "source_url": "https://www.nature.com/articles/s41467-024-45199-x",
    },
    "perimysial": {
        "celltype": "perimysial",
        "genes": [
            "Aldh1a2",
            "Hic1",
            "Tbx15",
            "Smoc2",
            "Creb5",
            "Fgf18",
        ],
        "motifs": ["Hic1", "Creb5"],
        "source": (
            "Independent developing soft-palate studies: Aldh1a2/Hic1/Tbx15/"
            "Smoc2 perimysial markers and CREB5-FGF18 regulatory axis."
        ),
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9771365/",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna", type=Path, default=RNA_FILE)
    parser.add_argument("--atac", type=Path, default=ATAC_FILE)
    parser.add_argument("--motif", type=Path, default=MOTIF_FILE)
    parser.add_argument("--peak-links", type=Path, default=PEAK_LINKS)
    parser.add_argument("--growth", type=Path, default=GROWTH_FILE)
    parser.add_argument("--cohorts", type=Path, default=COHORT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-fraction", type=float, default=0.20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dense(value) -> np.ndarray:
    if sp.issparse(value):
        return value.toarray().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


def standardize_columns(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    sd = np.std(matrix, axis=0)
    informative = np.isfinite(sd) & (sd > 1e-10)
    if not np.any(informative):
        return np.empty((len(matrix), 0), dtype=np.float64), informative
    selected = matrix[:, informative]
    selected = (
        selected - np.mean(selected, axis=0, keepdims=True)
    ) / np.std(selected, axis=0, keepdims=True)
    return selected, informative


def gene_module_score(
    expression: sp.spmatrix | np.ndarray,
    idx: np.ndarray,
    columns: list[int],
) -> tuple[np.ndarray, int]:
    values, informative = standardize_columns(
        dense(expression[idx][:, columns])
    )
    if not values.shape[1]:
        return np.full(len(idx), np.nan), 0
    return np.mean(values, axis=1), int(np.sum(informative))


def linked_peak_module_score(
    peak_matrix: sp.spmatrix | np.ndarray,
    idx: np.ndarray,
    gene_to_columns: dict[str, list[int]],
) -> tuple[np.ndarray, int, int]:
    gene_scores: list[np.ndarray] = []
    informative_peaks = 0
    for columns in gene_to_columns.values():
        values, informative = standardize_columns(
            dense(peak_matrix[idx][:, columns])
        )
        if not values.shape[1]:
            continue
        informative_peaks += int(np.sum(informative))
        score = np.mean(values, axis=1)
        score_sd = float(np.std(score))
        if score_sd <= 1e-10:
            continue
        gene_scores.append((score - np.mean(score)) / score_sd)
    if not gene_scores:
        return np.full(len(idx), np.nan), 0, informative_peaks
    return (
        np.mean(np.stack(gene_scores, axis=1), axis=1),
        len(gene_scores),
        informative_peaks,
    )


def motif_module_score(
    motif_matrix: sp.spmatrix | np.ndarray,
    idx: np.ndarray,
    columns: list[int],
) -> tuple[np.ndarray, int]:
    if not columns:
        return np.full(len(idx), np.nan), 0
    values, informative = standardize_columns(
        dense(motif_matrix[idx][:, columns])
    )
    if not values.shape[1]:
        return np.full(len(idx), np.nan), 0
    return np.mean(values, axis=1), int(np.sum(informative))


def standardize_vector(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    sd = float(np.std(value))
    return (
        (value - np.mean(value)) / sd
        if sd > 1e-12
        else np.zeros_like(value)
    )


def build_q(obs: pd.DataFrame, modality: str) -> np.ndarray:
    pieces: list[np.ndarray] = [
        np.ones((len(obs), 1), dtype=np.float64)
    ]
    sample = pd.get_dummies(
        obs["library"].astype(str), drop_first=True, dtype=float
    )
    if sample.shape[1]:
        pieces.append(sample.to_numpy(np.float64))
    if modality == "RNA":
        pieces.append(
            standardize_vector(
                np.log1p(
                    pd.to_numeric(
                        obs["nCount_RNA"], errors="coerce"
                    ).fillna(0).to_numpy(float)
                )
            )[:, None]
        )
        pieces.append(
            standardize_vector(
                pd.to_numeric(
                    obs["percent.mt"], errors="coerce"
                ).fillna(0).to_numpy(float)
            )[:, None]
        )
    else:
        pieces.append(
            standardize_vector(
                np.log1p(
                    pd.to_numeric(
                        obs["nCount_ATAC"], errors="coerce"
                    ).fillna(0).to_numpy(float)
                )
            )[:, None]
        )
        pieces.append(
            standardize_vector(
                pd.to_numeric(
                    obs["TSS.enrichment"], errors="coerce"
                ).fillna(0).to_numpy(float)
            )[:, None]
        )
    design = np.concatenate(pieces, axis=1)
    keep = np.std(design, axis=0) > 1e-12
    keep[0] = True
    q, _ = np.linalg.qr(design[:, keep], mode="reduced")
    return q


def partial_spearman(
    growth: np.ndarray,
    score: np.ndarray,
    q: np.ndarray,
) -> tuple[float, float]:
    growth = np.asarray(growth, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    valid = np.isfinite(growth) & np.isfinite(score)
    if np.sum(valid) < 10:
        return np.nan, np.nan
    growth = growth[valid]
    score = score[valid]
    local_q = q[valid]
    if np.std(growth) <= 1e-12 or np.std(score) <= 1e-12:
        return np.nan, np.nan
    x = rankdata(growth, method="average")
    y = rankdata(score, method="average")
    x = x - local_q @ (local_q.T @ x)
    y = y - local_q @ (local_q.T @ y)
    x -= np.mean(x)
    y -= np.mean(y)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator <= 1e-12:
        return np.nan, np.nan
    rho = float(x @ y / denominator)
    df = max(len(x) - local_q.shape[1] - 1, 1)
    clipped = float(np.clip(rho, -0.999999, 0.999999))
    statistic = clipped * np.sqrt(
        df / max(1.0 - clipped * clipped, 1e-12)
    )
    p_value = float(2.0 * student_t.sf(abs(statistic), df=df))
    return rho, p_value


def high_low_delta(
    growth: np.ndarray,
    score: np.ndarray,
    fraction: float,
) -> float:
    valid = np.isfinite(growth) & np.isfinite(score)
    growth = np.asarray(growth)[valid]
    score = np.asarray(score)[valid]
    if len(growth) < 10:
        return np.nan
    order = np.argsort(growth)
    tail = max(1, int(np.floor(fraction * len(order))))
    return float(np.mean(score[order[-tail:]]) - np.mean(score[order[:tail]]))


def exact_top_mask(value: np.ndarray, fraction: float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(value))
    mask = np.zeros(len(value), dtype=bool)
    if not len(valid):
        return mask
    tail = max(1, int(np.floor(fraction * len(valid))))
    order = valid[np.argsort(value[valid], kind="stable")]
    mask[order[-tail:]] = True
    return mask


def joint_high_enrichment(
    growth: np.ndarray,
    scores: list[np.ndarray],
    fraction: float,
) -> tuple[float, float, float, float, int, int]:
    valid = np.isfinite(growth)
    for score in scores:
        valid &= np.isfinite(score)
    if np.sum(valid) < 20:
        return np.nan, np.nan, np.nan, np.nan, 0, 0
    growth_valid = np.asarray(growth, dtype=float)[valid]
    score_valid = [np.asarray(score, dtype=float)[valid] for score in scores]
    growth_high = exact_top_mask(growth_valid, fraction)
    program_high = np.ones(len(growth_valid), dtype=bool)
    for score in score_valid:
        program_high &= exact_top_mask(score, fraction)
    a = int(np.sum(growth_high & program_high))
    b = int(np.sum(growth_high & ~program_high))
    c = int(np.sum(~growth_high & program_high))
    d = int(np.sum(~growth_high & ~program_high))
    observed = float(a / max(a + b, 1))
    baseline = float((a + c) / len(growth_valid))
    enrichment = observed / baseline if baseline > 0 else np.nan
    _, p_value = fisher_exact(
        [[a, b], [c, d]], alternative="greater"
    )
    return (
        observed,
        baseline,
        float(enrichment),
        float(p_value),
        a,
        int(a + b),
    )


def library_sign_agreement(
    growth: np.ndarray,
    score: np.ndarray,
    obs: pd.DataFrame,
    modality: str,
    expected_sign: float,
) -> tuple[bool, float, float]:
    values: list[float] = []
    samples = obs["library"].astype(str).to_numpy()
    for sample in pd.unique(samples):
        member = samples == sample
        if np.sum(member) < 10:
            continue
        q = build_q(obs.iloc[np.flatnonzero(member)], modality)
        rho, _ = partial_spearman(
            growth[member], score[member], q
        )
        if np.isfinite(rho):
            values.append(rho)
    if not values:
        return False, np.nan, np.nan
    array = np.asarray(values, dtype=float)
    return (
        bool(
            len(array) >= 2
            and np.all(np.sign(array) == expected_sign)
        ),
        float(np.min(array)),
        float(np.max(array)),
    )


def definitions_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lineage, spec in LINEAGE_PROGRAMS.items():
        rows.append(
            {
                "lineage": lineage,
                "matched_celltype": spec["celltype"],
                "fixed_RNA_genes": ";".join(spec["genes"]),
                "fixed_TF_motifs": ";".join(spec["motifs"]),
                "source": spec["source"],
                "source_url": spec["source_url"],
                "stage_invariant": True,
                "growth_independent_definition": True,
            }
        )
    return pd.DataFrame(rows)


def summarize_settings(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["analysis", "stage", "celltype", "lineage"]
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(keys, observed=True, sort=False):
        row = dict(zip(keys, key))
        row["n_C_y"] = int(group["C_y"].nunique())
        for prefix in ("rna", "peak", "motif"):
            values = group[f"{prefix}_partial_spearman"].to_numpy(float)
            finite = values[np.isfinite(values)]
            row[f"{prefix}_rho_mean"] = (
                float(np.mean(finite)) if len(finite) else np.nan
            )
            row[f"{prefix}_rho_median"] = (
                float(np.median(finite)) if len(finite) else np.nan
            )
            row[f"{prefix}_rho_min"] = (
                float(np.min(finite)) if len(finite) else np.nan
            )
            row[f"{prefix}_rho_max"] = (
                float(np.max(finite)) if len(finite) else np.nan
            )
            row[f"{prefix}_positive_C_y_fraction"] = (
                float(np.mean(finite > 0)) if len(finite) else np.nan
            )
            delta = group[f"{prefix}_high_minus_low"].to_numpy(float)
            delta = delta[np.isfinite(delta)]
            row[f"{prefix}_high_minus_low_mean"] = (
                float(np.mean(delta)) if len(delta) else np.nan
            )
            agreement = group[
                f"{prefix}_library_sign_agreement"
            ].astype(bool)
            row[f"{prefix}_library_agreement_C_y_fraction"] = (
                float(np.mean(agreement)) if len(finite) else np.nan
            )
        rna_positive = group["rna_partial_spearman"].to_numpy(float) > 0
        peak_positive = group["peak_partial_spearman"].to_numpy(float) > 0
        row["RNA_peak_joint_positive_C_y_fraction"] = float(
            np.mean(rna_positive & peak_positive)
        )
        motif_values = group["motif_partial_spearman"].to_numpy(float)
        motif_available = np.isfinite(motif_values)
        row["RNA_peak_motif_joint_positive_C_y_fraction"] = (
            float(
                np.mean(
                    rna_positive[motif_available]
                    & peak_positive[motif_available]
                    & (motif_values[motif_available] > 0)
                )
            )
            if np.any(motif_available)
            else np.nan
        )
        row["all_C_y_RNA_peak_positive"] = bool(
            np.all(rna_positive & peak_positive)
        )
        for name in ("RNA_peak", "RNA_peak_motif"):
            enrichment = group[
                f"{name}_high_enrichment"
            ].to_numpy(float)
            enrichment = enrichment[np.isfinite(enrichment)]
            row[f"{name}_high_enrichment_mean"] = (
                float(np.mean(enrichment))
                if len(enrichment)
                else np.nan
            )
            row[f"{name}_high_enrichment_min"] = (
                float(np.min(enrichment))
                if len(enrichment)
                else np.nan
            )
            row[f"{name}_high_enrichment_max"] = (
                float(np.max(enrichment))
                if len(enrichment)
                else np.nan
            )
            row[f"{name}_high_enrichment_gt1_C_y_fraction"] = (
                float(np.mean(enrichment > 1))
                if len(enrichment)
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "definitions": args.output_dir
        / "fixed_lineage_program_definitions.csv",
        "availability": args.output_dir
        / "fixed_lineage_feature_availability.csv",
        "matched_setting": args.output_dir
        / "matched_lineage_multiome_by_setting.csv",
        "matched_summary": args.output_dir
        / "matched_lineage_multiome_summary.csv",
        "progenitor_setting": args.output_dir
        / "progenitor_fate_multiome_by_setting.csv",
        "progenitor_summary": args.output_dir
        / "progenitor_fate_multiome_summary.csv",
        "manifest": args.output_dir / "analysis_manifest.json",
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")
    required = (
        args.rna,
        args.atac,
        args.motif,
        args.peak_links,
        args.growth,
        args.cohorts,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))

    definitions_frame().to_csv(targets["definitions"], index=False)
    print("Loading aligned paired multiome readouts", flush=True)
    rna = ad.read_h5ad(args.rna, backed="r")
    atac = ad.read_h5ad(args.atac, backed="r")
    motif = ad.read_h5ad(args.motif, backed="r")
    if rna.raw is None:
        raise ValueError("RNA AnnData has no raw expression")
    if not np.array_equal(
        rna.obs_names.astype(str), atac.obs_names.astype(str)
    ):
        raise ValueError("RNA and ATAC cells are not row-aligned")
    if not np.array_equal(
        rna.obs_names.astype(str), motif.obs_names.astype(str)
    ):
        raise ValueError("RNA and motif cells are not row-aligned")
    obs = rna.obs.copy()
    growth = pd.read_csv(args.growth)
    if not np.array_equal(
        rna.obs_names.astype(str).to_numpy(),
        growth["cell"].astype(str).to_numpy(),
    ):
        raise ValueError("Growth table and paired multiome cells are not aligned")
    cohort_table = pd.read_csv(args.cohorts)

    all_genes = list(
        dict.fromkeys(
            gene
            for spec in LINEAGE_PROGRAMS.values()
            for gene in spec["genes"]
        )
    )
    present_genes = [
        gene for gene in all_genes if gene in rna.raw.var_names
    ]
    rna_indices = [
        int(rna.raw.var_names.get_loc(gene)) for gene in present_genes
    ]
    rna_matrix = rna.raw.X[:, rna_indices].tocsr()
    rna_column = {gene: i for i, gene in enumerate(present_genes)}

    links = pd.read_csv(args.peak_links)
    links = links[
        links["gene"].astype(str).isin(all_genes)
        & (pd.to_numeric(links["score"], errors="coerce") > 0)
        & (pd.to_numeric(links["pvalue"], errors="coerce") < 0.05)
    ].copy()
    atac_names = set(atac.var_names.astype(str))
    links = links[links["peak"].astype(str).isin(atac_names)].copy()
    selected_peaks = list(dict.fromkeys(links["peak"].astype(str)))
    atac_indices = [
        int(atac.var_names.get_loc(peak)) for peak in selected_peaks
    ]
    peak_matrix = atac.X[:, atac_indices].tocsr()
    peak_column = {peak: i for i, peak in enumerate(selected_peaks)}

    all_motifs = list(
        dict.fromkeys(
            value
            for spec in LINEAGE_PROGRAMS.values()
            for value in spec["motifs"]
        )
    )
    present_motifs = [
        value for value in all_motifs if value in motif.var_names
    ]
    motif_indices = [
        int(motif.var_names.get_loc(value)) for value in present_motifs
    ]
    motif_matrix = motif.X[:, motif_indices].tocsr()
    motif_column = {
        value: i for i, value in enumerate(present_motifs)
    }

    program_features: dict[str, dict[str, object]] = {}
    availability_rows: list[dict[str, object]] = []
    for lineage, spec in LINEAGE_PROGRAMS.items():
        genes = [gene for gene in spec["genes"] if gene in rna_column]
        motifs = [
            value for value in spec["motifs"] if value in motif_column
        ]
        local_links = links[links["gene"].astype(str).isin(genes)]
        gene_to_peaks = {
            gene: [
                peak_column[peak]
                for peak in group["peak"].astype(str)
                if peak in peak_column
            ]
            for gene, group in local_links.groupby(
                "gene", observed=True, sort=False
            )
        }
        program_features[lineage] = {
            "gene_columns": [rna_column[gene] for gene in genes],
            "genes": genes,
            "gene_to_peak_columns": gene_to_peaks,
            "motif_columns": [motif_column[value] for value in motifs],
            "motifs": motifs,
        }
        availability_rows.append(
            {
                "lineage": lineage,
                "matched_celltype": spec["celltype"],
                "RNA_genes_requested": len(spec["genes"]),
                "RNA_genes_present": len(genes),
                "RNA_genes": ";".join(genes),
                "genes_with_positive_linked_peaks": len(gene_to_peaks),
                "linked_peaks_present": sum(
                    len(value) for value in gene_to_peaks.values()
                ),
                "linked_peak_genes": ";".join(gene_to_peaks),
                "motifs_requested": len(spec["motifs"]),
                "motifs_present": len(motifs),
                "motifs": ";".join(motifs),
            }
        )
    pd.DataFrame(availability_rows).to_csv(
        targets["availability"], index=False
    )

    labels = obs["celltype_sub"].astype(str).to_numpy()
    stages = obs["stage"].astype(str).to_numpy()
    eligible = set(
        zip(
            cohort_table["stage"].astype(str),
            cohort_table["celltype"].astype(str),
        )
    )

    matched_rows: list[dict[str, object]] = []
    progenitor_rows: list[dict[str, object]] = []

    def analyze_cohort(
        *,
        analysis: str,
        stage: str,
        celltype: str,
        lineage: str,
        output: list[dict[str, object]],
    ) -> None:
        idx = np.flatnonzero((stages == stage) & (labels == celltype))
        features = program_features[lineage]
        rna_score, n_rna = gene_module_score(
            rna_matrix, idx, features["gene_columns"]
        )
        peak_score, n_peak_genes, n_peaks = linked_peak_module_score(
            peak_matrix,
            idx,
            features["gene_to_peak_columns"],
        )
        motif_score, n_motifs = motif_module_score(
            motif_matrix, idx, features["motif_columns"]
        )
        local_obs = obs.iloc[idx]
        q_rna = build_q(local_obs, "RNA")
        q_atac = build_q(local_obs, "ATAC")
        for c_y, column in zip(CY_VALUES, GROWTH_COLUMNS):
            local_growth = growth.iloc[idx][column].to_numpy(float)
            rna_rho, rna_p = partial_spearman(
                local_growth, rna_score, q_rna
            )
            peak_rho, peak_p = partial_spearman(
                local_growth, peak_score, q_atac
            )
            motif_rho, motif_p = partial_spearman(
                local_growth, motif_score, q_atac
            )
            rna_agree, rna_rep_min, rna_rep_max = (
                library_sign_agreement(
                    local_growth,
                    rna_score,
                    local_obs,
                    "RNA",
                    np.sign(rna_rho),
                )
            )
            peak_agree, peak_rep_min, peak_rep_max = (
                library_sign_agreement(
                    local_growth,
                    peak_score,
                    local_obs,
                    "ATAC",
                    np.sign(peak_rho),
                )
            )
            motif_agree, motif_rep_min, motif_rep_max = (
                library_sign_agreement(
                    local_growth,
                    motif_score,
                    local_obs,
                    "ATAC",
                    np.sign(motif_rho),
                )
                if n_motifs
                else (False, np.nan, np.nan)
            )
            (
                rna_peak_observed,
                rna_peak_baseline,
                rna_peak_enrichment,
                rna_peak_overlap_p,
                rna_peak_overlap_cells,
                growth_high_cells,
            ) = joint_high_enrichment(
                local_growth,
                [rna_score, peak_score],
                args.top_fraction,
            )
            (
                rna_peak_motif_observed,
                rna_peak_motif_baseline,
                rna_peak_motif_enrichment,
                rna_peak_motif_overlap_p,
                rna_peak_motif_overlap_cells,
                _,
            ) = joint_high_enrichment(
                local_growth,
                [rna_score, peak_score, motif_score],
                args.top_fraction,
            )
            output.append(
                {
                    "analysis": analysis,
                    "stage": stage,
                    "celltype": celltype,
                    "lineage": lineage,
                    "C_y": c_y,
                    "n_cells": len(idx),
                    "n_libraries": local_obs["library"].nunique(),
                    "n_informative_RNA_genes": n_rna,
                    "n_informative_peak_genes": n_peak_genes,
                    "n_informative_peaks": n_peaks,
                    "n_informative_motifs": n_motifs,
                    "rna_partial_spearman": rna_rho,
                    "rna_p_value": rna_p,
                    "rna_high_minus_low": high_low_delta(
                        local_growth,
                        rna_score,
                        args.top_fraction,
                    ),
                    "rna_library_sign_agreement": rna_agree,
                    "rna_library_rho_min": rna_rep_min,
                    "rna_library_rho_max": rna_rep_max,
                    "peak_partial_spearman": peak_rho,
                    "peak_p_value": peak_p,
                    "peak_high_minus_low": high_low_delta(
                        local_growth,
                        peak_score,
                        args.top_fraction,
                    ),
                    "peak_library_sign_agreement": peak_agree,
                    "peak_library_rho_min": peak_rep_min,
                    "peak_library_rho_max": peak_rep_max,
                    "motif_partial_spearman": motif_rho,
                    "motif_p_value": motif_p,
                    "motif_high_minus_low": high_low_delta(
                        local_growth,
                        motif_score,
                        args.top_fraction,
                    ),
                    "motif_library_sign_agreement": motif_agree,
                    "motif_library_rho_min": motif_rep_min,
                    "motif_library_rho_max": motif_rep_max,
                    "RNA_peak_joint_positive": bool(
                        rna_rho > 0 and peak_rho > 0
                    ),
                    "RNA_peak_motif_joint_positive": (
                        bool(
                            rna_rho > 0
                            and peak_rho > 0
                            and motif_rho > 0
                        )
                        if n_motifs
                        else pd.NA
                    ),
                    "growth_high_cells": growth_high_cells,
                    "RNA_peak_high_in_growth_high_fraction": (
                        rna_peak_observed
                    ),
                    "RNA_peak_high_baseline_fraction": rna_peak_baseline,
                    "RNA_peak_high_enrichment": rna_peak_enrichment,
                    "RNA_peak_high_overlap_p_value": rna_peak_overlap_p,
                    "RNA_peak_high_overlap_cells": (
                        rna_peak_overlap_cells
                    ),
                    "RNA_peak_motif_high_in_growth_high_fraction": (
                        rna_peak_motif_observed
                    ),
                    "RNA_peak_motif_high_baseline_fraction": (
                        rna_peak_motif_baseline
                    ),
                    "RNA_peak_motif_high_enrichment": (
                        rna_peak_motif_enrichment
                    ),
                    "RNA_peak_motif_high_overlap_p_value": (
                        rna_peak_motif_overlap_p
                    ),
                    "RNA_peak_motif_high_overlap_cells": (
                        rna_peak_motif_overlap_cells
                    ),
                }
            )

    for lineage, spec in LINEAGE_PROGRAMS.items():
        celltype = str(spec["celltype"])
        for stage in STAGES:
            if (stage, celltype) not in eligible:
                continue
            print(
                f"Matched {lineage}: {stage} {celltype}",
                flush=True,
            )
            analyze_cohort(
                analysis="matched current celltype",
                stage=stage,
                celltype=celltype,
                lineage=lineage,
                output=matched_rows,
            )

    progenitor = "CNC-derived progenitors"
    for stage in STAGES:
        if (stage, progenitor) not in eligible:
            continue
        for lineage in LINEAGE_PROGRAMS:
            print(
                f"Progenitor priming {lineage}: {stage}", flush=True
            )
            analyze_cohort(
                analysis="CNC progenitor fate priming",
                stage=stage,
                celltype=progenitor,
                lineage=lineage,
                output=progenitor_rows,
            )

    matched = pd.DataFrame(matched_rows)
    progenitor_frame = pd.DataFrame(progenitor_rows)
    p_columns = [
        "rna_p_value",
        "peak_p_value",
        "motif_p_value",
        "RNA_peak_high_overlap_p_value",
        "RNA_peak_motif_high_overlap_p_value",
    ]
    for frame in (matched, progenitor_frame):
        for column in p_columns:
            frame[column.replace("_p_value", "_fdr")] = bh_fdr(
                frame[column].to_numpy(float)
            )
    matched_summary = summarize_settings(matched)
    progenitor_summary = summarize_settings(progenitor_frame)

    matched.to_csv(targets["matched_setting"], index=False)
    matched_summary.to_csv(targets["matched_summary"], index=False)
    progenitor_frame.to_csv(targets["progenitor_setting"], index=False)
    progenitor_summary.to_csv(targets["progenitor_summary"], index=False)

    manifest = {
        "analysis": (
            "Fixed, stage-invariant, growth-independent lineage programs tested "
            "against paired RNA, linked ATAC peak accessibility, and available "
            "TF motif accessibility on the same observed cells."
        ),
        "growth_input": str(args.growth),
        "growth_definition": (
            "USOT native instantaneous d log mass / dt evaluated at each real "
            "cell's normalized RNA PCA40 state and observed time."
        ),
        "paired_alignment": (
            "RNA, ATAC peaks, motif accessibility, and growth were required to "
            "have identical cell barcode order."
        ),
        "RNA_score": (
            "Within each stage x current-celltype cohort, z-score each fixed "
            "marker gene and average informative genes."
        ),
        "ATAC_peak_score": (
            "Positive marker gene--peak links with p<0.05, restricted to peaks "
            "present in the filtered ATAC matrix. Peaks are averaged within "
            "gene and genes are weighted equally."
        ),
        "motif_score": (
            "Within-cohort z-score and mean of only pre-specified TF motifs "
            "available in CNC_motif.h5ad. Missing motifs are reported as "
            "unavailable rather than substituted."
        ),
        "association": (
            "Partial Spearman. RNA adjusts for library, log RNA depth and "
            "mitochondrial fraction; ATAC adjusts for library, log ATAC depth "
            "and TSS enrichment."
        ),
        "high_low_contrast": (
            f"Mean module z-score in the top {args.top_fraction:.0%} growth "
            f"cells minus the bottom {args.top_fraction:.0%}."
        ),
        "joint_high_enrichment": (
            f"Exact top {args.top_fraction:.0%} cells are selected separately "
            "for growth and each module. The observed fraction of joint "
            "RNA-high/peak-high (and RNA-high/peak-high/motif-high) cells "
            "inside growth-high is divided by its cohort-wide baseline; a "
            "one-sided Fisher exact test assesses enrichment."
        ),
        "setting_aggregation": (
            "Each C_y is analyzed independently. Summaries report mean, median, "
            "min, max and positive fraction across the nine settings."
        ),
        "programs": LINEAGE_PROGRAMS,
        "outputs": {key: path.name for key, path in targets.items()},
        "limitations": [
            "Positive peak--gene links were estimated in the original palate "
            "dataset and are independent of USOT growth, but are not an "
            "external experimental enhancer map.",
            "Only motifs available in the provided 107-feature motif matrix can "
            "be tested; absence is not evidence of biological inactivity.",
            "C_y settings measure hyperparameter stability, not biological "
            "replicate uncertainty.",
            "Within-current-celltype association does not establish causal "
            "lineage progression or future fate.",
        ],
    }
    targets["manifest"].write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    rna.file.close()
    atac.file.close()
    motif.file.close()
    print(f"Wrote fixed-lineage multiome analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
