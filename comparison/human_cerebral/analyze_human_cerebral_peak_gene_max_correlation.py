#!/usr/bin/env python
"""Null-calibrated peak--gene timing for the frozen human-cerebral rollouts.

The candidate set is frozen from the Pando table before trajectory scores are
opened.  Pando regions are joined to the assayed ATAC atlas by GRCh38 interval
overlap, never by string equality.  RNA and ATAC features are decoded with the
same stage-conditioned Gaussian barycentric kNN readout (fixed k=15).  Native
COATI ATAC and fixed-T post-hoc ATAC are retained as different quantities.

Pairs are ranked only by their maximum lagged derivative cosine.  The old
winning margin is written as a diagnostic and is never a gate or ranking key.
Every matched/circular/particle-shuffle null is subjected to the identical
max-over-lag operation.
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
import hashlib
import json
import math
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.signal import savgol_filter
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "results/human_cerebral_full_biological_interpretability"
AUDIT = RESULT_ROOT / "00_input_audit"
EXTERNAL = AUDIT / "external_evidence"
MATERIALIZED = RESULT_ROOT / "01_trajectory_materialization"
DEFAULT_OUTPUT = RESULT_ROOT / "02_peak_gene_max_correlation"

HUMAN = Path("external/COATI/humanCerebral")
DATA = HUMAN / "Data/selected_4_7_9_11_12_18_21"
RNA_H5AD = HUMAN / "Data/humanBrainRNA.h5ad"
ATAC_H5AD = HUMAN / "Data/humanBrainATAC.h5ad"
GTF = EXTERNAL / "reference_genome/gencode.v48.annotation.gtf.gz"

PANDO_CANDIDATES = EXTERNAL / "pando_priority_peak_gene_candidates.tsv.gz"
PANDO_GLI3 = EXTERNAL / "pando_gli3_prespecified_edges.tsv"
PUBLIC = EXTERNAL / "public_support"
GLI3_KO_DA_ALL = PUBLIC / "gli3_ko_early_telencephalic_da_granges.tsv.gz"
GLI3_KO_DA_PAPER5000 = PUBLIC / "gli3_ko_da_fdr1e4_lowest_coef_top5000.bed.gz"
GLI3_KO_DA_STRICT_NEGATIVE = PUBLIC / "gli3_ko_da_fdr1e4_strict_negative.bed.gz"
GLI3_KO_DE_EARLY = PUBLIC / "gli3_ko_early_telencephalic_de.tsv.gz"
GLI3_KO_DE_VENTRAL = PUBLIC / "gli3_ko_ventral_telencephalon_de.tsv.gz"

MODEL_REGISTRY = AUDIT / "model_registry.csv"
TIME_KEYS = ("age_4", "age_7", "age_9", "age_11", "age_12", "age_18", "age_21")
PHYSICAL_TIMES = np.asarray((0.0, 0.3, 0.5, 0.7, 0.8, 1.4, 1.7), dtype=float)
OBSERVED_INDICES = np.asarray((0, 11, 19, 26, 30, 53, 65), dtype=int)
KNN_K = 15
MAX_LAG_STEPS = 5
PRIMARY_SMOOTHING = 5
SMOOTHING_GRID = (5, 7)
MIN_GROUP_N = 8
NULLS_PER_PAIR = 10
RANDOM_SEED = 73191
GENE_LOCUS_RADIUS_BP = 100_000

CORE_GENES = (
    "HES4", "HES5", "PAX6", "CREB5", "OTX2", "ID1", "LHX8", "BCL11A",
    "SIX3", "SOX3", "DLX5", "TFAP2A", "FGF8", "FOXG1", "WLS", "WNT8B",
    "IRX5", "EMX1", "EMX2", "BMP7", "NEUROD6", "DLX1", "DLX2", "GSX2",
    "ASCL1", "NKX2-1", "DCT", "DIO3", "SIX6",
)
GLI3_DIRECT_TARGETS = ("HES4", "HES5", "PAX6", "CREB5", "OTX2", "ID1", "LHX8", "BCL11A")
PRIMING_TARGETS = ("NKX2-1", "ID1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Use the frozen preregistered core genes but evaluate all 23 rollouts.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return out
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.minimum(ranked, 1.0)
    return out


def parse_assayed_peak(value: str) -> tuple[str, int, int]:
    chrom, start, end = str(value).rsplit("-", 2)
    return chrom, int(start), int(end)


def parse_pando_peak(value: str) -> tuple[str, int, int]:
    chrom, start, end = str(value).rsplit("_", 2)
    return chrom, int(start), int(end)


def overlap_mapping(
    query_names: list[str], reference_names: list[str], *, query_kind: str
) -> pd.DataFrame:
    by_chrom: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    records: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for index, name in enumerate(reference_names):
        chrom, start, end = parse_assayed_peak(name)
        records[chrom].append((start, end, index))
    for chrom, values in records.items():
        values.sort()
        by_chrom[chrom] = (
            np.asarray([v[0] for v in values], dtype=np.int64),
            np.asarray([v[1] for v in values], dtype=np.int64),
            np.asarray([v[2] for v in values], dtype=np.int64),
        )
    rows: list[dict[str, object]] = []
    parser = parse_pando_peak if query_kind == "pando" else parse_assayed_peak
    for name in query_names:
        chrom, start, end = parser(name)
        if chrom not in by_chrom:
            rows.append({"query_peak": name, "reference_peak": None, "overlap_bp": 0})
            continue
        starts, ends, order = by_chrom[chrom]
        stop = int(np.searchsorted(starts, end, side="left"))
        eligible = np.flatnonzero(ends[:stop] > start)
        if not len(eligible):
            rows.append({"query_peak": name, "reference_peak": None, "overlap_bp": 0})
        for local in eligible:
            ref = reference_names[int(order[local])]
            _, rstart, rend = parse_assayed_peak(ref)
            rows.append(
                {
                    "query_peak": name,
                    "reference_peak": ref,
                    "overlap_bp": int(min(end, rend) - max(start, rstart)),
                }
            )
    return pd.DataFrame(rows)


def aggregate_strings(values: pd.Series) -> str:
    return ";".join(sorted(set(values.dropna().astype(str))))


def freeze_candidates(output: Path, pilot: bool) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_csv(PANDO_CANDIDATES, sep="\t")
    if pilot:
        raw = raw.loc[raw["target"].isin(CORE_GENES)].copy()
    atac = ad.read_h5ad(ATAC_H5AD, backed="r")
    assayed = atac.var_names.astype(str).tolist()
    atac.file.close()
    mapping = overlap_mapping(
        sorted(raw["pando_peak"].astype(str).unique()), assayed, query_kind="pando"
    ).rename(columns={"query_peak": "pando_peak", "reference_peak": "peak"})
    mapped = raw.drop(columns=["trajectory_peaks"], errors="ignore").merge(
        mapping.loc[mapping["peak"].notna()], on="pando_peak", how="inner"
    )
    mapped["is_gli3_pando_edge"] = mapped["tf"].astype(str).eq("GLI3")
    candidates = (
        mapped.groupby(["target", "peak"], as_index=False, observed=True)
        .agg(
            pando_tfs=("tf", aggregate_strings),
            pando_regions=("pando_peak", aggregate_strings),
            pando_min_padj=("padj", "min"),
            pando_max_abs_estimate=("estimate", lambda x: float(np.max(np.abs(x)))),
            pando_corr_median=("corr", "median"),
            pando_overlap_bp_max=("overlap_bp", "max"),
            is_gli3_pando_edge=("is_gli3_pando_edge", "max"),
        )
        .rename(columns={"target": "gene"})
        .sort_values(["gene", "peak"])
        .reset_index(drop=True)
    )
    candidates["pair_id"] = [f"PG{i:05d}" for i in range(len(candidates))]
    candidates["is_core_gene"] = candidates["gene"].isin(CORE_GENES)
    candidates["is_gli3_direct_target_gene"] = candidates["gene"].isin(GLI3_DIRECT_TARGETS)
    candidates["is_priming_gene"] = candidates["gene"].isin(PRIMING_TARGETS)
    candidates["candidate_selection"] = "Pando priority table; GRCh38 interval-overlap to assayed peak"
    candidates["selection_uses_trajectory_score"] = False
    path = output / "frozen_peak_gene_candidates.csv.gz"
    candidates.to_csv(path, index=False, compression="gzip")
    return candidates, {
        "raw_rows": int(len(raw)),
        "unique_pando_regions": int(raw["pando_peak"].nunique()),
        "interval_mapped_pando_regions": int(
            mapping.loc[mapping["peak"].notna(), "pando_peak"].nunique()
        ),
        "assayed_peaks": len(assayed),
        "frozen_unique_pairs": int(len(candidates)),
        "frozen_genes": int(candidates["gene"].nunique()),
        "frozen_peaks": int(candidates["peak"].nunique()),
        "candidate_sha256": sha256(path),
        "exact_string_matching_used": False,
    }


ATTRIBUTE_PATTERN = re.compile(r'(\w+) "([^"]+)"')


def load_gtf() -> tuple[pd.DataFrame, pd.DataFrame]:
    genes: list[dict[str, object]] = []
    transcripts: list[dict[str, object]] = []
    with gzip.open(GTF, "rt") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] not in {"gene", "transcript"}:
                continue
            attrs = dict(ATTRIBUTE_PATTERN.findall(fields[8]))
            name = attrs.get("gene_name")
            if not name:
                continue
            chrom = fields[0] if fields[0].startswith("chr") else f"chr{fields[0]}"
            start = int(fields[3]) - 1
            end = int(fields[4])
            strand = fields[6]
            record = {
                "chrom": chrom,
                "start": start,
                "end": end,
                "strand": strand,
                "tss": start if strand == "+" else end - 1,
                "gene": name,
                "gene_id": attrs.get("gene_id", ""),
                "gene_type": attrs.get("gene_type", attrs.get("gene_biotype", "")),
            }
            if fields[2] == "gene":
                genes.append(record)
            else:
                record["transcript_id"] = attrs.get("transcript_id", "")
                record["transcript_type"] = attrs.get(
                    "transcript_type", attrs.get("transcript_biotype", "")
                )
                transcripts.append(record)
    gene_frame = pd.DataFrame(genes).drop_duplicates(["gene_id", "chrom", "start", "end"])
    transcript_frame = pd.DataFrame(transcripts).drop_duplicates("transcript_id")
    return gene_frame, transcript_frame


def annotate_loci(
    candidates: pd.DataFrame, genes: pd.DataFrame, transcripts: pd.DataFrame
) -> pd.DataFrame:
    protein = genes.loc[genes["gene_type"].eq("protein_coding")].copy()
    protein_tx = transcripts.loc[
        transcripts["gene_type"].eq("protein_coding")
        | transcripts["transcript_type"].eq("protein_coding")
    ].copy()
    gene_lookup = {
        gene: frame for gene, frame in genes.groupby("gene", sort=False, observed=True)
    }
    protein_by_chrom = {
        chrom: frame.reset_index(drop=True)
        for chrom, frame in protein.groupby("chrom", sort=False, observed=True)
    }
    tx_by_chrom = {
        chrom: frame.reset_index(drop=True)
        for chrom, frame in protein_tx.groupby("chrom", sort=False, observed=True)
    }
    rows: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        chrom, start, end = parse_assayed_peak(row.peak)
        midpoint = 0.5 * (start + end)
        target = gene_lookup.get(str(row.gene), pd.DataFrame())
        target_same = target.loc[target["chrom"].eq(chrom)] if not target.empty else target
        if target_same.empty:
            target_tss = np.nan
            target_distance = np.nan
            target_gene_body = False
        else:
            distance = np.abs(target_same["tss"].to_numpy(float) - midpoint)
            best = target_same.iloc[int(np.argmin(distance))]
            target_tss = int(best["tss"])
            target_distance = float(midpoint - target_tss)
            target_gene_body = bool(
                ((target_same["start"] < end) & (target_same["end"] > start)).any()
            )
        local = protein_by_chrom.get(chrom, pd.DataFrame())
        if local.empty:
            overlaps: list[str] = []
            nearest = ""
            nearest_other = ""
            nearest_distance = np.nan
            nearest_other_distance = np.nan
        else:
            overlap_frame = local.loc[(local["start"] < end) & (local["end"] > start)]
            overlaps = sorted(set(overlap_frame["gene"].astype(str)))
            dist = np.abs(local["tss"].to_numpy(float) - midpoint)
            order = np.argsort(dist)
            nearest = str(local.iloc[int(order[0])]["gene"])
            nearest_distance = float(midpoint - float(local.iloc[int(order[0])]["tss"]))
            other_order = [i for i in order if str(local.iloc[int(i)]["gene"]) != str(row.gene)]
            nearest_other = str(local.iloc[int(other_order[0])]["gene"]) if other_order else ""
            nearest_other_distance = (
                float(midpoint - float(local.iloc[int(other_order[0])]["tss"]))
                if other_order
                else np.nan
            )
        local_tx = tx_by_chrom.get(chrom, pd.DataFrame())
        if local_tx.empty:
            promoter2 = []
            promoter5 = []
        else:
            tss = local_tx["tss"].to_numpy(int)
            promoter2 = sorted(
                set(local_tx.loc[(tss + 2000 >= start) & (tss - 2000 < end), "gene"].astype(str))
            )
            promoter5 = sorted(
                set(local_tx.loc[(tss + 5000 >= start) & (tss - 5000 < end), "gene"].astype(str))
            )
        other_bodies = [value for value in overlaps if value != str(row.gene)]
        other_promoters2 = [value for value in promoter2 if value != str(row.gene)]
        rows.append(
            {
                "pair_id": row.pair_id,
                "chrom": chrom,
                "peak_start": start,
                "peak_end": end,
                "peak_midpoint": midpoint,
                "linked_target_tss": target_tss,
                "signed_target_tss_distance_bp": target_distance,
                "overlaps_linked_gene_body": target_gene_body,
                "protein_coding_gene_body_overlaps": ";".join(overlaps),
                "other_gene_body_overlaps": ";".join(other_bodies),
                "protein_coding_promoter_2kb_overlaps": ";".join(promoter2),
                "other_gene_promoter_2kb_overlaps": ";".join(other_promoters2),
                "protein_coding_promoter_5kb_overlaps_sensitivity": ";".join(promoter5),
                "overlaps_other_gene_body_or_promoter": bool(other_bodies or other_promoters2),
                "nearest_protein_coding_gene": nearest,
                "nearest_protein_coding_gene_tss_offset_bp": nearest_distance,
                "nearest_other_protein_coding_gene": nearest_other,
                "nearest_other_protein_coding_gene_tss_offset_bp": nearest_other_distance,
            }
        )
    return pd.DataFrame(rows)


def annotate_external(candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    candidate_peaks = candidates["peak"].astype(str).drop_duplicates().tolist()

    # The source supplementary table uses 1-based inclusive coordinates. The
    # prepared audit table includes the explicit BED conversion used below.
    da = pd.read_csv(GLI3_KO_DA_ALL, sep="\t").copy()
    da["da_peak"] = (
        da["chrom"].astype(str)
        + "-"
        + da["start_0based_bed"].astype(int).astype(str)
        + "-"
        + da["end_0based_bed"].astype(int).astype(str)
    )
    mapping = overlap_mapping(
        da["da_peak"].astype(str).tolist(),
        candidate_peaks,
        query_kind="assayed",
    ).rename(columns={"query_peak": "da_peak", "reference_peak": "peak"})
    da_mapped = mapping.loc[mapping["peak"].notna()].merge(da, on="da_peak", how="left")
    da_best = (
        da_mapped.sort_values(["peak", "padj", "pval"])
        .groupby("peak", as_index=False, observed=True)
        .first()[["peak", "da_peak", "coef", "pval", "padj", "overlap_bp"]]
        .rename(
            columns={
                "coef": "gli3_ko_early_tel_da_coef",
                "pval": "gli3_ko_early_tel_da_pval",
                "padj": "gli3_ko_early_tel_da_padj",
                "overlap_bp": "gli3_ko_early_tel_da_overlap_bp",
            }
        )
    )

    def map_frozen_bed(path: Path) -> tuple[set[str], int]:
        frozen = pd.read_csv(path, sep="\t")
        frozen["source_peak"] = (
            frozen["chrom"].astype(str)
            + "-"
            + frozen["start"].astype(int).astype(str)
            + "-"
            + frozen["end"].astype(int).astype(str)
        )
        frozen_mapping = overlap_mapping(
            frozen["source_peak"].astype(str).tolist(),
            candidate_peaks,
            query_kind="assayed",
        )
        mapped = set(
            frozen_mapping.loc[
                frozen_mapping["reference_peak"].notna(), "reference_peak"
            ].astype(str)
        )
        return mapped, int(len(frozen))

    paper5000_peaks, paper5000_n = map_frozen_bed(GLI3_KO_DA_PAPER5000)
    strict_negative_peaks, strict_negative_n = map_frozen_bed(GLI3_KO_DA_STRICT_NEGATIVE)

    de_ventral = pd.read_csv(GLI3_KO_DE_VENTRAL, sep="\t").rename(
        columns={"gene": "gene", "coef": "gli3_ko_d45_ventral_de_coef", "padj": "gli3_ko_d45_ventral_de_padj"}
    )[["gene", "gli3_ko_d45_ventral_de_coef", "gli3_ko_d45_ventral_de_padj"]]
    de_early = pd.read_csv(GLI3_KO_DE_EARLY, sep="\t")
    de_early = de_early.loc[de_early["group"].astype(str).eq("telen")].rename(
        columns={"feature": "gene", "fc": "gli3_ko_early_tel_de_fc", "padj": "gli3_ko_early_tel_de_padj"}
    )[["gene", "gli3_ko_early_tel_de_fc", "gli3_ko_early_tel_de_padj"]]
    annotated = candidates.merge(da_best, on="peak", how="left").merge(
        de_early, on="gene", how="left"
    ).merge(de_ventral, on="gene", how="left")
    annotated["in_gli3_ko_da_paper_ranked5000"] = annotated["peak"].isin(paper5000_peaks)
    annotated["in_gli3_ko_da_strict_negative4286"] = annotated["peak"].isin(strict_negative_peaks)
    annotated["early_ko_da_supported"] = annotated["in_gli3_ko_da_paper_ranked5000"]
    annotated["early_ko_de_supported"] = annotated["gli3_ko_early_tel_de_padj"].le(0.10)
    annotated["independent_perturbation_supported"] = (
        annotated["in_gli3_ko_da_paper_ranked5000"] & annotated["early_ko_de_supported"]
    )
    annotated["independent_perturbation_supported_strict"] = (
        annotated["in_gli3_ko_da_strict_negative4286"] & annotated["early_ko_de_supported"]
    )
    annotated["ko_da_de_sign_consistent"] = (
        annotated["gli3_ko_early_tel_da_coef"] * annotated["gli3_ko_early_tel_de_fc"] > 0
    )
    annotated["strong_gli3_locus_support"] = (
        annotated["is_gli3_pando_edge"]
        & annotated["independent_perturbation_supported"]
        & annotated["ko_da_de_sign_consistent"]
    )
    return annotated, {
        "supplementary_table_10_DA_rows": int(len(da)),
        "supplementary_table_10_mapped_candidate_peaks": int(da_best["peak"].nunique()),
        "frozen_primary_DA_rows": paper5000_n,
        "frozen_primary_DA_mapped_candidate_peaks": int(len(paper5000_peaks)),
        "frozen_strict_negative_DA_rows": strict_negative_n,
        "frozen_strict_negative_DA_mapped_candidate_peaks": int(len(strict_negative_peaks)),
        "independent_perturbation_supported_pairs": int(annotated["independent_perturbation_supported"].sum()),
        "independent_perturbation_supported_strict_pairs": int(annotated["independent_perturbation_supported_strict"].sum()),
        "strong_gli3_locus_supported_pairs": int(annotated["strong_gli3_locus_support"].sum()),
        "primary_DA_support_rule": "overlap with frozen paper-ranked5000 set (FDR<1e-4, lowest coefficients)",
        "strict_DA_sensitivity_rule": "overlap with frozen strict-negative4286 set (FDR<1e-4 and coefficient<0)",
        "da_coefficient_sign": "positive means more accessible in GLI3 KO; negative means depleted in KO",
        "early_tel_de_fc_sign": "positive means higher in early-telencephalic GLI3 KO; negative means lower in KO",
    }


def target_tss_table(genes: pd.DataFrame, target_names: set[str]) -> pd.DataFrame:
    local = genes.loc[genes["gene"].isin(target_names)].copy()
    rows: list[dict[str, object]] = []
    for name, frame in local.groupby("gene", observed=True, sort=False):
        # GENCODE can contain PAR/alternate entries. Prefer canonical chr1--22/X/Y.
        canonical = frame.loc[frame["chrom"].str.fullmatch(r"chr(?:[0-9]+|X|Y)")]
        use = canonical if not canonical.empty else frame
        best = use.sort_values(["chrom", "start"]).iloc[0]
        rows.append({"gene": name, "gene_chrom": best.chrom, "gene_tss": int(best.tss)})
    return pd.DataFrame(rows)


def construct_matched_nulls(
    candidates: pd.DataFrame, genes: pd.DataFrame, n_null: int
) -> pd.DataFrame:
    rna = ad.read_h5ad(RNA_H5AD, backed="r")
    available_rna_genes = set(rna.var_names.astype(str))
    rna.file.close()
    candidate_tss = target_tss_table(genes, set(candidates["gene"].astype(str)))
    candidate = candidates.merge(candidate_tss, on="gene", how="left")
    # Null targets are drawn from every measured protein-coding gene on the
    # same chromosome, not only from the frozen candidate target list. This
    # provides the preregistered number of nonlinks without changing the peak.
    protein_coding = genes.loc[genes["gene_type"].eq("protein_coding"), "gene"].astype(str)
    null_gene_pool = available_rna_genes & set(protein_coding)
    all_tss = target_tss_table(genes, null_gene_pool)
    linked = set(zip(candidate["peak"].astype(str), candidate["gene"].astype(str)))
    genes_by_chrom = {
        chrom: frame.drop_duplicates("gene").reset_index(drop=True)
        for chrom, frame in all_tss.groupby("gene_chrom", observed=True, sort=False)
    }
    rows: list[dict[str, object]] = []
    for row in candidate.itertuples(index=False):
        chrom, start, end = parse_assayed_peak(row.peak)
        midpoint = 0.5 * (start + end)
        if not isinstance(row.gene_chrom, str) or row.gene_chrom != chrom:
            continue
        local = genes_by_chrom.get(chrom)
        if local is None:
            continue
        target_distance = abs(midpoint - float(row.gene_tss))
        local = local.loc[
            local["gene"].astype(str).ne(str(row.gene))
            & ~local["gene"].astype(str).map(lambda gene: (str(row.peak), gene) in linked)
        ].copy()
        if local.empty:
            continue
        distance = np.abs(midpoint - local["gene_tss"].to_numpy(float))
        # Log-distance mismatch is the matching metric; peak GC/accessibility/length
        # are exact because the same peak is reused in the null pair.
        mismatch = np.abs(np.log1p(distance) - np.log1p(target_distance))
        local["null_distance_bp"] = distance
        local["distance_log_mismatch"] = mismatch
        local = local.sort_values(["distance_log_mismatch", "gene"]).head(n_null)
        for rank, null in enumerate(local.itertuples(index=False), start=1):
            rows.append(
                {
                    "pair_id": row.pair_id,
                    "true_gene": row.gene,
                    "peak": row.peak,
                    "null_rank": rank,
                    "null_gene": null.gene,
                    "true_abs_tss_distance_bp": target_distance,
                    "null_abs_tss_distance_bp": null.null_distance_bp,
                    "distance_log_mismatch": null.distance_log_mismatch,
                    "peak_gc_match": "exact_same_peak",
                    "peak_accessibility_match": "exact_same_peak",
                    "peak_length_match": "exact_same_peak",
                    "known_pando_link_excluded": True,
                }
            )
    return pd.DataFrame(rows)


@dataclass
class MolecularReference:
    paired_obs: pd.DataFrame
    ids: np.ndarray
    rna_pca: list[np.ndarray]
    atac_lsi: list[np.ndarray]
    gaga: list[np.ndarray]
    rna_values: list[sp.csr_matrix]
    atac_values: list[sp.csr_matrix]
    atac_depth: list[np.ndarray]
    genes: np.ndarray
    peaks: np.ndarray


def load_molecular_reference(
    candidate: pd.DataFrame,
    nulls: pd.DataFrame,
    loci: pd.DataFrame,
) -> tuple[MolecularReference, pd.DataFrame, dict[str, object]]:
    paired_obs = pd.read_csv(DATA / "paired_obs.csv")
    with np.load(DATA / "paired_metacell_ids_by_time.npz", allow_pickle=True) as saved:
        ids = np.concatenate([saved[key].astype(str) for key in TIME_KEYS])
    if not np.array_equal(ids, paired_obs["paired_metacell_id"].astype(str).to_numpy()):
        raise ValueError("paired IDs and paired_obs order differ")
    with np.load(DATA / "rna_pca30_normalized_by_time.npz") as saved:
        rna_pca = [np.asarray(saved[key], dtype=np.float32) for key in TIME_KEYS]
    with np.load(DATA / "atac_lsi12_normalized_by_time.npz") as saved:
        atac_lsi = [np.asarray(saved[key], dtype=np.float32) for key in TIME_KEYS]
    gaga_path = ROOT / "results/mioflow_human_cerebral_7time_d4_d21_no_d16_full_official_gaga10_n256_30000/gaga10_embedding.npz"
    with np.load(gaga_path, allow_pickle=True) as saved:
        if not np.array_equal(saved["cell_ids"].astype(str), ids):
            raise ValueError("GAGA embedding and paired IDs differ")
        full_gaga = np.asarray(saved["embedding_model_input"], dtype=np.float32)
        labels = np.asarray(saved["time_labels"], dtype=float)
    unique_labels = np.unique(labels)
    if len(unique_labels) != len(PHYSICAL_TIMES) or not np.allclose(
        unique_labels, PHYSICAL_TIMES, atol=1e-6, rtol=0.0
    ):
        raise ValueError(
            f"Unexpected GAGA physical time labels: {unique_labels.tolist()}"
        )
    gaga = [full_gaga[np.isclose(labels, value)] for value in PHYSICAL_TIMES]

    requested_genes = set(candidate["gene"].astype(str)) | set(nulls["null_gene"].astype(str))
    for column in ("other_gene_body_overlaps", "other_gene_promoter_2kb_overlaps"):
        for value in loci[column].dropna().astype(str):
            requested_genes.update(item for item in value.split(";") if item)
    requested_peaks = set(candidate["peak"].astype(str))

    rna = ad.read_h5ad(RNA_H5AD, backed="r")
    atac = ad.read_h5ad(ATAC_H5AD, backed="r")
    rna_names = pd.Index(rna.var_names.astype(str))
    atac_names = pd.Index(atac.var_names.astype(str))
    gene_names = np.asarray(sorted(requested_genes & set(rna_names)), dtype=object)
    peak_names = np.asarray(sorted(requested_peaks & set(atac_names)), dtype=object)
    row_index = pd.Index(rna.obs_names.astype(str)).get_indexer(ids)
    atac_rows = pd.Index(atac.obs_names.astype(str)).get_indexer(ids)
    if np.any(row_index < 0) or np.any(atac_rows < 0) or not np.array_equal(row_index, atac_rows):
        raise ValueError("Molecular matrices do not contain the paired cells in the same order")
    gene_index = rna_names.get_indexer(gene_names)
    peak_index = atac_names.get_indexer(peak_names)
    rna_subset = rna[row_index, gene_index].to_memory().X
    atac_subset = atac[atac_rows, peak_index].to_memory().X
    rna.file.close()
    atac.file.close()
    rna_subset = sp.csr_matrix(rna_subset, dtype=np.float32)
    rna_subset.data = np.expm1(rna_subset.data).astype(np.float32, copy=False)
    atac_subset = sp.csr_matrix(atac_subset, dtype=np.float32)
    stage_rows = [np.flatnonzero(paired_obs["time_key"].astype(str).eq(key)) for key in TIME_KEYS]
    rna_values = [rna_subset[index] for index in stage_rows]
    atac_values = [atac_subset[index] for index in stage_rows]
    atac_depth = [paired_obs.iloc[index]["nCount_peaks"].to_numpy(float) for index in stage_rows]

    usable = candidate.loc[
        candidate["gene"].isin(gene_names) & candidate["peak"].isin(peak_names)
    ].copy()
    audit = {
        "requested_genes": len(requested_genes),
        "available_requested_genes": len(gene_names),
        "requested_peaks": len(requested_peaks),
        "available_requested_peaks": len(peak_names),
        "usable_pairs": len(usable),
        "stage_rows": [len(x) for x in stage_rows],
        "fixed_knn_k": KNN_K,
        "rna_value_semantics": "100 * expm1(source log-normalized counts) = CPM-like readout",
        "atac_value_semantics": "source raw peak counts / source nCount_peaks * 1e4",
    }
    return (
        MolecularReference(
            paired_obs=paired_obs,
            ids=ids,
            rna_pca=rna_pca,
            atac_lsi=atac_lsi,
            gaga=gaga,
            rna_values=rna_values,
            atac_values=atac_values,
            atac_depth=atac_depth,
            genes=gene_names,
            peaks=peak_names,
        ),
        usable,
        audit,
    )


@dataclass
class Rollout:
    model_id: str
    method: str
    balance_mode: str
    sync_mode: str
    c_y: float | None
    rna: np.ndarray
    atac: np.ndarray
    time: np.ndarray
    log_mass: np.ndarray | None
    rna_space: str
    atac_semantics: str


def load_rollout(row: pd.Series) -> Rollout:
    path = Path(str(row["trajectory_path"]))
    model_id = str(row["model_id"])
    if path.suffix == ".pt":
        rna = torch.load(path, map_location="cpu", weights_only=False).detach().cpu().numpy()
    else:
        with np.load(path, allow_pickle=False) as saved:
            if model_id.startswith("mioflow"):
                rna = np.asarray(saved["trajectory_native_gaga10_zscore"], dtype=np.float32)
                time = np.asarray(saved["physical_time_grid"], dtype=float)
            else:
                rna = np.asarray(saved["trajectory_normalized_pca30"], dtype=np.float32)
                time = np.asarray(saved["physical_time_grid"], dtype=float)
    if path.suffix == ".pt":
        # All PT materializations use the canonical grid stored alongside the rollout.
        if "coati_sync_balanced" in model_id:
            tpath = path.parent / "t_grid_s0_iter30000.pt"
        elif "coati_sync_unbalanced" in model_id:
            tpath = path.parent / "t_grid_s0_iter40000.pt"
        elif "coati_rna_only" in model_id:
            tpath = path.parent / "t_grid_rna_only_full_s0_iter30000.pt"
        else:
            tpath = path.parent / "physical_time_grid.pt"
        time = torch.load(tpath, map_location="cpu", weights_only=False).detach().cpu().numpy()
    if "RNA_ATAC_sync" == str(row["sync_mode"]):
        atac_path = Path(str(path).replace("primary_trajectory", "secondary_trajectory"))
        atac = torch.load(atac_path, map_location="cpu", weights_only=False).detach().cpu().numpy()
        atac_semantics = "native_model_coupled_sync"
    else:
        if model_id.startswith("coati_rna_only"):
            atac_path = path.parent / "posthoc_secondary_trajectory_rna_only_full_s0_iter30000.pt"
        else:
            atac_path = path.parent / "trajectory_posthoc_T_normalized_lsi12.pt"
        atac = torch.load(atac_path, map_location="cpu", weights_only=False).detach().cpu().numpy()
        atac_semantics = "post_hoc_fixed_T"
    log_mass = None
    mass_value = row.get("mass_artifact")
    if isinstance(mass_value, str) and mass_value and Path(mass_value).is_file():
        log_mass = torch.load(Path(mass_value), map_location="cpu", weights_only=False)
        if torch.is_tensor(log_mass):
            log_mass = log_mass.detach().cpu().numpy()
        log_mass = np.asarray(log_mass, dtype=np.float32).reshape(len(time), len(rna[0]))
    if rna.shape[:2] != (66, 501) or atac.shape[:2] != (66, 501):
        raise ValueError(f"Unexpected trajectory shape for {model_id}: {rna.shape}, {atac.shape}")
    c_y = float(row["C_y"]) if pd.notna(row["C_y"]) else None
    return Rollout(
        model_id=model_id,
        method=str(row["method"]),
        balance_mode=str(row["balance_mode"]),
        sync_mode=str(row["sync_mode"]),
        c_y=c_y,
        rna=np.asarray(rna, dtype=np.float32),
        atac=np.asarray(atac, dtype=np.float32),
        time=np.asarray(time, dtype=float),
        log_mass=log_mass,
        rna_space="GAGA10_zscore" if model_id.startswith("mioflow") else "normalized_PCA30",
        atac_semantics=atac_semantics,
    )


def neighbor_weights(distances: np.ndarray) -> np.ndarray:
    scale = float(np.median(distances[:, -1]))
    if not np.isfinite(scale) or scale <= 1e-12:
        positive = distances[distances > 0]
        scale = float(np.median(positive)) if len(positive) else 1.0
    values = np.exp(-np.square(distances / scale))
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-30)


def stage_bracket(time: float) -> tuple[int, int, float]:
    if time <= PHYSICAL_TIMES[0]:
        return 0, 0, 0.0
    if time >= PHYSICAL_TIMES[-1]:
        return len(PHYSICAL_TIMES) - 1, len(PHYSICAL_TIMES) - 1, 0.0
    high = int(np.searchsorted(PHYSICAL_TIMES, time, side="right"))
    low = high - 1
    alpha = float((time - PHYSICAL_TIMES[low]) / (PHYSICAL_TIMES[high] - PHYSICAL_TIMES[low]))
    return low, high, alpha


def weighted_hard_labels(
    query: np.ndarray, model: NearestNeighbors, labels: np.ndarray
) -> np.ndarray:
    distances, indices = model.kneighbors(query, n_neighbors=KNN_K)
    weights = neighbor_weights(distances)
    levels = np.asarray(sorted(pd.unique(labels.astype(str))), dtype=object)
    votes = np.stack(
        [np.sum(weights * (labels[indices] == level), axis=1) for level in levels], axis=1
    )
    return levels[np.argmax(votes, axis=1)]


def group_matrix(
    source_lines: np.ndarray,
    terminal_fates: np.ndarray,
    mass: np.ndarray | None,
    time_index: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    masks: list[tuple[str, np.ndarray]] = [("all", np.ones(len(source_lines), dtype=bool))]
    for line in sorted(pd.unique(source_lines)):
        masks.append((f"line:{line}", source_lines == line))
    for fate in ("ctx", "ge", "nt", "telencephalon"):
        masks.append((f"fate:{fate}", terminal_fates == fate))
    names: list[str] = []
    matrix: list[np.ndarray] = []
    sizes: list[int] = []
    for name, mask in masks:
        if int(mask.sum()) < MIN_GROUP_N:
            continue
        if mass is None:
            weight = mask.astype(float)
        else:
            values = np.asarray(mass[time_index], dtype=float)
            values = values - float(np.max(values[mask]))
            weight = np.zeros(len(mask), dtype=float)
            weight[mask] = np.exp(values[mask])
        weight /= max(float(weight.sum()), 1e-30)
        names.append(name)
        matrix.append(weight)
        sizes.append(int(mask.sum()))
    return names, np.stack(matrix), np.asarray(sizes, dtype=int)


def barycentric_coefficients(
    query: np.ndarray, neighbor_model: NearestNeighbors, group_weights: np.ndarray
) -> np.ndarray:
    distances, indices = neighbor_model.kneighbors(query, n_neighbors=KNN_K)
    weights = neighbor_weights(distances)
    rows = np.repeat(np.arange(len(query)), KNN_K)
    graph = sp.csr_matrix(
        (weights.reshape(-1), (rows, indices.reshape(-1))),
        shape=(len(query), neighbor_model.n_samples_fit_),
    )
    return np.asarray(group_weights @ graph)


def decode_rollout(
    rollout: Rollout,
    reference: MolecularReference,
    output: Path,
) -> tuple[Path, pd.DataFrame]:
    rna_reference = reference.gaga if rollout.rna_space == "GAGA10_zscore" else reference.rna_pca
    rna_models = [NearestNeighbors(n_neighbors=KNN_K, n_jobs=-1).fit(x) for x in rna_reference]
    atac_models = [NearestNeighbors(n_neighbors=KNN_K, n_jobs=-1).fit(x) for x in reference.atac_lsi]
    terminal_labels = reference.paired_obs.loc[
        reference.paired_obs["time_key"].eq(TIME_KEYS[-1]), "lineage"
    ].astype(str).to_numpy()
    terminal_fates = weighted_hard_labels(rollout.rna[-1], rna_models[-1], terminal_labels)
    source_lines = reference.paired_obs.loc[
        reference.paired_obs["time_key"].eq(TIME_KEYS[0]), "line"
    ].astype(str).to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    permutation = np.arange(len(source_lines))
    for line in pd.unique(source_lines):
        idx = np.flatnonzero(source_lines == line)
        permutation[idx] = rng.permutation(idx)

    first_names, _, first_sizes = group_matrix(source_lines, terminal_fates, rollout.log_mass, 0)
    n_groups = len(first_names)
    rna_curves = np.full((n_groups, len(rollout.time), len(reference.genes)), np.nan, dtype=np.float32)
    atac_curves = np.full((n_groups, len(rollout.time), len(reference.peaks)), np.nan, dtype=np.float32)
    atac_shuffle = np.full_like(atac_curves, np.nan)
    for ti, time in enumerate(rollout.time):
        names, group_weights, _ = group_matrix(source_lines, terminal_fates, rollout.log_mass, ti)
        if names != first_names:
            raise ValueError("Group order changed across time")
        low, high, alpha = stage_bracket(float(time))

        def decode_rna(stage: int) -> np.ndarray:
            coeff = barycentric_coefficients(rollout.rna[ti], rna_models[stage], group_weights)
            return np.asarray(coeff @ reference.rna_values[stage], dtype=float) * 100.0

        def decode_atac(stage: int, shuffled: bool) -> tuple[np.ndarray, np.ndarray]:
            query = rollout.atac[ti][permutation] if shuffled else rollout.atac[ti]
            coeff = barycentric_coefficients(query, atac_models[stage], group_weights)
            counts = np.asarray(coeff @ reference.atac_values[stage], dtype=float)
            depth = np.asarray(coeff @ reference.atac_depth[stage], dtype=float).reshape(-1, 1)
            return counts, depth

        rna_low = decode_rna(low)
        atac_low, depth_low = decode_atac(low, False)
        shuffle_low, shuffle_depth_low = decode_atac(low, True)
        if low == high:
            rna_value = rna_low
            atac_count, atac_depth = atac_low, depth_low
            shuffle_count, shuffle_depth = shuffle_low, shuffle_depth_low
        else:
            rna_value = (1.0 - alpha) * rna_low + alpha * decode_rna(high)
            atac_high, depth_high = decode_atac(high, False)
            shuffle_high, shuffle_depth_high = decode_atac(high, True)
            atac_count = (1.0 - alpha) * atac_low + alpha * atac_high
            atac_depth = (1.0 - alpha) * depth_low + alpha * depth_high
            shuffle_count = (1.0 - alpha) * shuffle_low + alpha * shuffle_high
            shuffle_depth = (1.0 - alpha) * shuffle_depth_low + alpha * shuffle_depth_high
        rna_curves[:, ti] = rna_value.astype(np.float32)
        atac_curves[:, ti] = (atac_count / np.maximum(atac_depth, 1e-30) * 1e4).astype(np.float32)
        atac_shuffle[:, ti] = (
            shuffle_count / np.maximum(shuffle_depth, 1e-30) * 1e4
        ).astype(np.float32)

    curve_dir = output / "decoded_feature_curves"
    curve_dir.mkdir(exist_ok=True)
    curve_path = curve_dir / f"{rollout.model_id}.npz"
    np.savez_compressed(
        curve_path,
        time=rollout.time.astype(np.float32),
        group_names=np.asarray(first_names, dtype=object),
        group_particle_n=first_sizes,
        gene_names=reference.genes,
        peak_names=reference.peaks,
        rna_cpm=rna_curves,
        atac_cp10k=atac_curves,
        atac_cp10k_paired_particle_shuffle=atac_shuffle,
        terminal_fate=terminal_fates,
        source_line=source_lines,
        atac_semantics=np.asarray(rollout.atac_semantics),
        rna_space=np.asarray(rollout.rna_space),
    )
    audit = pd.DataFrame(
        {
            "model_id": rollout.model_id,
            "group": first_names,
            "n_particles": first_sizes,
            "rna_space": rollout.rna_space,
            "atac_semantics": rollout.atac_semantics,
            "curve_file": str(curve_path),
        }
    )
    return curve_path, audit


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right) or not np.isfinite(left).all() or not np.isfinite(right).all():
        return np.nan
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else np.nan


def lag_score(
    rna: np.ndarray,
    atac: np.ndarray,
    time: np.ndarray,
    smoothing: int,
    *,
    circular_shift: int = 0,
) -> dict[str, float]:
    rna = np.log1p(np.asarray(rna, dtype=float))
    atac = np.log1p(np.asarray(atac, dtype=float))
    if circular_shift:
        atac = np.roll(atac, int(circular_shift))
    rna_s = savgol_filter(rna, smoothing, 2, mode="interp")
    atac_s = savgol_filter(atac, smoothing, 2, mode="interp")
    dr = np.gradient(rna_s, time)
    da = np.gradient(atac_s, time)
    scores: dict[int, float] = {}
    for lag in range(-MAX_LAG_STEPS, MAX_LAG_STEPS + 1):
        if lag < 0:
            scores[lag] = cosine(dr[:lag], da[-lag:])
        elif lag > 0:
            scores[lag] = cosine(da[:-lag], dr[lag:])
        else:
            scores[lag] = cosine(dr, da)
    valid = [(lag, score) for lag, score in scores.items() if np.isfinite(score)]
    if not valid:
        return {
            "max_derivative_cosine": np.nan,
            "lag_steps_at_max": np.nan,
            "lag_model_time_at_max": np.nan,
            "second_best_score": np.nan,
            "winning_margin_diagnostic_only": np.nan,
            "rna_log_dynamic_range": float(np.ptp(rna)),
            "atac_log_dynamic_range": float(np.ptp(atac)),
        }
    ordered = sorted(valid, key=lambda x: x[1], reverse=True)
    lag, best = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else np.nan
    lag_time = (
        float(np.median(time[lag:] - time[:-lag]))
        if lag > 0
        else -float(np.median(time[-lag:] - time[:lag]))
        if lag < 0
        else 0.0
    )
    return {
        "max_derivative_cosine": float(best),
        "lag_steps_at_max": int(lag),
        "lag_model_time_at_max": lag_time,
        "second_best_score": float(second),
        "winning_margin_diagnostic_only": float(best - second),
        "rna_log_dynamic_range": float(np.ptp(rna)),
        "atac_log_dynamic_range": float(np.ptp(atac)),
    }


def score_curves(
    curve_paths: list[Path],
    candidates: pd.DataFrame,
    nulls: pd.DataFrame,
    registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    true_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    line_rows: list[dict[str, object]] = []
    particle_rows: list[dict[str, object]] = []
    pair_records = list(candidates[["pair_id", "gene", "peak"]].itertuples(index=False))
    null_records = list(nulls.itertuples(index=False))
    meta = registry.set_index("model_id")
    for curve_path in curve_paths:
        model_id = curve_path.stem
        row = meta.loc[model_id]
        with np.load(curve_path, allow_pickle=True) as saved:
            time = np.asarray(saved["time"], dtype=float)
            groups = saved["group_names"].astype(str).tolist()
            gene_names = saved["gene_names"].astype(str)
            peak_names = saved["peak_names"].astype(str)
            rna = np.asarray(saved["rna_cpm"], dtype=float)
            atac = np.asarray(saved["atac_cp10k"], dtype=float)
            atac_shuffle = np.asarray(saved["atac_cp10k_paired_particle_shuffle"], dtype=float)
        gene_index = {name: i for i, name in enumerate(gene_names)}
        peak_index = {name: i for i, name in enumerate(peak_names)}
        all_index = groups.index("all")
        for smoothing in SMOOTHING_GRID:
            for pair in pair_records:
                if pair.gene not in gene_index or pair.peak not in peak_index:
                    continue
                score = lag_score(
                    rna[all_index, :, gene_index[pair.gene]],
                    atac[all_index, :, peak_index[pair.peak]],
                    time,
                    smoothing,
                )
                circular = [
                    lag_score(
                        rna[all_index, :, gene_index[pair.gene]],
                        atac[all_index, :, peak_index[pair.peak]],
                        time,
                        smoothing,
                        circular_shift=shift,
                    )["max_derivative_cosine"]
                    for shift in (13, 22, 33)
                ]
                finite_circular = [value for value in circular if np.isfinite(value)]
                true_rows.append(
                    {
                        "model_id": model_id,
                        "method": row["method"],
                        "balance_mode": row["balance_mode"],
                        "sync_mode": row["sync_mode"],
                        "C_y": row["C_y"],
                        "atac_semantics": "native_model_coupled_sync" if row["sync_mode"] == "RNA_ATAC_sync" else "post_hoc_fixed_T",
                        "group": "all",
                        "smoothing_window": smoothing,
                        "pair_id": pair.pair_id,
                        "gene": pair.gene,
                        "peak": pair.peak,
                        **score,
                        "circular_shift_13_max_derivative_cosine": circular[0],
                        "circular_shift_22_max_derivative_cosine": circular[1],
                        "circular_shift_33_max_derivative_cosine": circular[2],
                        "circular_shift_null_median": (
                            float(np.median(finite_circular)) if finite_circular else np.nan
                        ),
                        "circular_shift_null_max": (
                            float(np.max(finite_circular)) if finite_circular else np.nan
                        ),
                    }
                )
            for null in null_records:
                if null.null_gene not in gene_index or null.peak not in peak_index:
                    continue
                score = lag_score(
                    rna[all_index, :, gene_index[null.null_gene]],
                    atac[all_index, :, peak_index[null.peak]],
                    time,
                    smoothing,
                )
                null_rows.append(
                    {
                        "model_id": model_id,
                        "smoothing_window": smoothing,
                        "pair_id": null.pair_id,
                        "true_gene": null.true_gene,
                        "null_gene": null.null_gene,
                        "peak": null.peak,
                        "null_rank": null.null_rank,
                        "distance_log_mismatch": null.distance_log_mismatch,
                        **score,
                    }
                )
        # Biological-line stability and paired-particle shuffle are evaluated
        # for core/prespecified pairs only under the primary smoothing.
        core = candidates.loc[
            candidates["is_core_gene"] | candidates["is_gli3_pando_edge"]
        ][["pair_id", "gene", "peak"]]
        for group_index, group in enumerate(groups):
            if group.startswith("line:"):
                for pair in core.itertuples(index=False):
                    if pair.gene in gene_index and pair.peak in peak_index:
                        line_rows.append(
                            {
                                "model_id": model_id,
                                "group": group,
                                "pair_id": pair.pair_id,
                                "gene": pair.gene,
                                "peak": pair.peak,
                                **lag_score(
                                    rna[group_index, :, gene_index[pair.gene]],
                                    atac[group_index, :, peak_index[pair.peak]],
                                    time,
                                    PRIMARY_SMOOTHING,
                                ),
                            }
                        )
            if group.startswith("fate:"):
                for pair in core.itertuples(index=False):
                    if pair.gene in gene_index and pair.peak in peak_index:
                        observed = lag_score(
                            rna[group_index, :, gene_index[pair.gene]],
                            atac[group_index, :, peak_index[pair.peak]],
                            time,
                            PRIMARY_SMOOTHING,
                        )
                        shuffled = lag_score(
                            rna[group_index, :, gene_index[pair.gene]],
                            atac_shuffle[group_index, :, peak_index[pair.peak]],
                            time,
                            PRIMARY_SMOOTHING,
                        )
                        particle_rows.append(
                            {
                                "model_id": model_id,
                                "group": group,
                                "pair_id": pair.pair_id,
                                "gene": pair.gene,
                                "peak": pair.peak,
                                "observed_max_derivative_cosine": observed["max_derivative_cosine"],
                                "paired_shuffle_max_derivative_cosine": shuffled["max_derivative_cosine"],
                                "observed_minus_paired_shuffle": observed["max_derivative_cosine"] - shuffled["max_derivative_cosine"],
                            }
                        )
    true = pd.DataFrame(true_rows)
    null = pd.DataFrame(null_rows)
    lines = pd.DataFrame(line_rows)
    particles = pd.DataFrame(particle_rows)
    return true, null, lines, particles


def calibrate_and_rank(true: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    summaries = (
        null.groupby(["model_id", "smoothing_window", "pair_id"], observed=True)[
            "max_derivative_cosine"
        ]
        .agg(matched_null_n="count", matched_null_median="median", matched_null_max="max")
        .reset_index()
    )
    merged = true.merge(summaries, on=["model_id", "smoothing_window", "pair_id"], how="left")
    null_groups = {
        key: frame["max_derivative_cosine"].to_numpy(float)
        for key, frame in null.groupby(["model_id", "smoothing_window", "pair_id"], observed=True)
    }
    p_values: list[float] = []
    percentiles: list[float] = []
    for row in merged.itertuples(index=False):
        values = null_groups.get((row.model_id, row.smoothing_window, row.pair_id), np.empty(0))
        values = values[np.isfinite(values)]
        if not len(values) or not np.isfinite(row.max_derivative_cosine):
            p_values.append(np.nan)
            percentiles.append(np.nan)
        else:
            p_values.append((1.0 + np.sum(values >= row.max_derivative_cosine)) / (1.0 + len(values)))
            percentiles.append(float(np.mean(values < row.max_derivative_cosine)))
    merged["matched_empirical_p"] = p_values
    merged["matched_null_percentile"] = percentiles
    merged["true_minus_matched_null_median"] = (
        merged["max_derivative_cosine"] - merged["matched_null_median"]
    )
    merged["matched_empirical_q"] = np.nan
    merged["rank_by_max_correlation"] = np.nan
    merged["top_decile_by_max_correlation"] = False
    for _, index in merged.groupby(["model_id", "smoothing_window"], observed=True).groups.items():
        idx = np.asarray(list(index), dtype=int)
        merged.loc[idx, "matched_empirical_q"] = bh(merged.loc[idx, "matched_empirical_p"].to_numpy(float))
        merged.loc[idx, "rank_by_max_correlation"] = merged.loc[idx, "max_derivative_cosine"].rank(
            method="min", ascending=False
        )
        cutoff = max(1, int(math.ceil(len(idx) * 0.10)))
        merged.loc[idx, "top_decile_by_max_correlation"] = merged.loc[idx, "rank_by_max_correlation"].le(cutoff)
    merged["ranking_uses_margin"] = False
    return merged


def stability_summary(ranked: pd.DataFrame, line_scores: pd.DataFrame) -> pd.DataFrame:
    primary = ranked.loc[ranked["smoothing_window"].eq(PRIMARY_SMOOTHING)].copy()
    alternate = ranked.loc[ranked["smoothing_window"].eq(SMOOTHING_GRID[1]), [
        "model_id", "pair_id", "max_derivative_cosine", "rank_by_max_correlation"
    ]].rename(columns={
        "max_derivative_cosine": "alternate_smoothing_score",
        "rank_by_max_correlation": "alternate_smoothing_rank",
    })
    primary = primary.merge(alternate, on=["model_id", "pair_id"], how="left")
    line_summary = (
        line_scores.groupby(["model_id", "pair_id"], observed=True)
        .agg(
            biological_line_n=("group", "nunique"),
            biological_line_score_median=("max_derivative_cosine", "median"),
            biological_line_positive_lag_fraction=("lag_steps_at_max", lambda x: float(np.mean(np.asarray(x) > 0))),
        )
        .reset_index()
    )
    primary = primary.merge(line_summary, on=["model_id", "pair_id"], how="left")
    return primary


def odds_ratio_top_decile(frame: pd.DataFrame, label: str) -> tuple[float, float, float]:
    supported = frame[label].fillna(False).to_numpy(bool)
    top = frame["top_decile_by_max_correlation"].to_numpy(bool)
    a = int(np.sum(top & supported))
    b = int(np.sum(top & ~supported))
    c = int(np.sum(~top & supported))
    d = int(np.sum(~top & ~supported))
    odds = (a + 0.5) * (d + 0.5) / ((b + 0.5) * (c + 0.5))
    se = math.sqrt(sum(1.0 / (value + 0.5) for value in (a, b, c, d)))
    low, high = math.exp(math.log(odds) - 1.96 * se), math.exp(math.log(odds) + 1.96 * se)
    return float(odds), float(low), float(high)


def model_summaries(
    ranked: pd.DataFrame,
    annotated: pd.DataFrame,
    particles: pd.DataFrame,
) -> pd.DataFrame:
    primary = ranked.loc[ranked["smoothing_window"].eq(PRIMARY_SMOOTHING)].merge(
        annotated[["pair_id", "independent_perturbation_supported", "strong_gli3_locus_support"]],
        on="pair_id",
        how="left",
    )
    rows: list[dict[str, object]] = []
    for model_id, frame in primary.groupby("model_id", observed=True, sort=False):
        label = frame["independent_perturbation_supported"].fillna(False).to_numpy(int)
        score = frame["max_derivative_cosine"].to_numpy(float)
        valid = np.isfinite(score)
        auprc = average_precision_score(label[valid], score[valid]) if len(np.unique(label[valid])) > 1 else np.nan
        odds, low, high = odds_ratio_top_decile(frame, "independent_perturbation_supported")
        paired = particles.loc[particles["model_id"].eq(model_id), "observed_minus_paired_shuffle"]
        first = frame.iloc[0]
        rows.append(
            {
                "model_id": model_id,
                "method": first["method"],
                "balance_mode": first["balance_mode"],
                "sync_mode": first["sync_mode"],
                "C_y": first["C_y"],
                "atac_semantics": first["atac_semantics"],
                "n_pairs": len(frame),
                "n_independent_supported_pairs": int(label.sum()),
                "supported_link_AUPRC": auprc,
                "top_decile_supported_odds_ratio": odds,
                "top_decile_supported_or_ci_low": low,
                "top_decile_supported_or_ci_high": high,
                "median_true_minus_matched_null": float(frame["true_minus_matched_null_median"].median()),
                "fraction_positive_ATAC_lead": float(np.mean(frame["lag_steps_at_max"] > 0)),
                "median_core_fate_score_minus_paired_shuffle": float(paired.median()) if len(paired) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_gene_locus_evidence(
    candidates: pd.DataFrame,
    genes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """One allowed evidence-tier revision fixed before gene-level scores are opened."""
    tss = target_tss_table(genes, set(candidates["gene"].astype(str))).set_index("gene")
    primary_da = pd.read_csv(GLI3_KO_DA_PAPER5000, sep="\t").copy()
    strict_da = pd.read_csv(GLI3_KO_DA_STRICT_NEGATIVE, sep="\t").copy()
    for frame in (primary_da, strict_da):
        frame["midpoint"] = 0.5 * (frame["start"].to_numpy(float) + frame["end"].to_numpy(float))
    early_de = pd.read_csv(GLI3_KO_DE_EARLY, sep="\t")
    early_de = early_de.loc[early_de["group"].astype(str).eq("telen")].set_index("feature")

    rows: list[dict[str, object]] = []
    for gene, frame in candidates.groupby("gene", observed=True, sort=True):
        if gene not in tss.index:
            continue
        locus = tss.loc[gene]
        chrom = str(locus["gene_chrom"])
        gene_tss = int(locus["gene_tss"])
        primary = primary_da.loc[
            primary_da["chrom"].astype(str).eq(chrom)
            & primary_da["midpoint"].sub(gene_tss).abs().le(GENE_LOCUS_RADIUS_BP)
        ].copy()
        strict = strict_da.loc[
            strict_da["chrom"].astype(str).eq(chrom)
            & strict_da["midpoint"].sub(gene_tss).abs().le(GENE_LOCUS_RADIUS_BP)
        ].copy()
        if gene in early_de.index:
            de_row = early_de.loc[gene]
            de_fc = float(de_row["fc"])
            de_padj = float(de_row["padj"])
        else:
            de_fc = np.nan
            de_padj = np.nan
        primary_coef = float(primary["coef"].median()) if len(primary) else np.nan
        strict_coef = float(strict["coef"].median()) if len(strict) else np.nan
        is_gli3_target = bool(frame["is_gli3_pando_edge"].max())
        supported = bool(is_gli3_target and de_padj <= 0.10 and len(primary))
        supported_strict = bool(is_gli3_target and de_padj <= 0.10 and len(strict))
        rows.append(
            {
                "gene": gene,
                "gene_chrom": chrom,
                "canonical_gene_tss": gene_tss,
                "locus_radius_bp": GENE_LOCUS_RADIUS_BP,
                "frozen_candidate_pair_n": int(len(frame)),
                "pando_gli3_target": is_gli3_target,
                "pando_gli3_direct_pair_n": int(frame["is_gli3_pando_edge"].sum()),
                "early_tel_gli3_ko_de_fc": de_fc,
                "early_tel_gli3_ko_de_padj": de_padj,
                "paper_ranked5000_locus_da_n": int(len(primary)),
                "paper_ranked5000_locus_da_coef_median": primary_coef,
                "paper_ranked5000_locus_da_peaks": ";".join(primary["name"].astype(str)),
                "strict_negative4286_locus_da_n": int(len(strict)),
                "strict_negative4286_locus_da_coef_median": strict_coef,
                "strict_negative4286_locus_da_peaks": ";".join(strict["name"].astype(str)),
                "gene_locus_perturbation_supported": supported,
                "gene_locus_perturbation_supported_strict": supported_strict,
                "gene_locus_ko_da_de_sign_consistent": bool(
                    supported and primary_coef * de_fc > 0
                ),
                "revision_rule": (
                    "Pando GLI3->target exists; early-tel GLI3-KO DE padj<=0.10; "
                    "at least one frozen paper-ranked DA peak within canonical TSS +/-100kb"
                ),
            }
        )
    evidence = pd.DataFrame(rows)
    supported = evidence.loc[evidence["gene_locus_perturbation_supported"]]
    return evidence, {
        "revision_number": 1,
        "revision_is_only_allowed_evidence_tier_fallback": True,
        "gene_set_changed": False,
        "time_grid_changed": False,
        "locus_definition": f"canonical GENCODE TSS +/-{GENE_LOCUS_RADIUS_BP} bp",
        "gene_score_aggregation": (
            "maximum pair max_derivative_cosine across every frozen Pando candidate peak for the gene"
        ),
        "gene_null_aggregation": (
            "for each of 10 null ranks, identical lag scan per frozen peak followed by the identical cross-peak maximum"
        ),
        "supported_gene_n": int(len(supported)),
        "supported_genes": supported["gene"].astype(str).tolist(),
        "strict_supported_gene_n": int(
            evidence["gene_locus_perturbation_supported_strict"].sum()
        ),
        "ko_sign_consistent_fraction": (
            float(supported["gene_locus_ko_da_de_sign_consistent"].mean())
            if len(supported)
            else np.nan
        ),
    }


def gene_locus_revision(
    ranked: pd.DataFrame,
    null_scores: pd.DataFrame,
    line_scores: pd.DataFrame,
    particle_scores: pd.DataFrame,
    evidence: pd.DataFrame,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    primary = ranked.loc[ranked["smoothing_window"].eq(PRIMARY_SMOOTHING)].copy()
    keys = [
        "model_id", "method", "balance_mode", "sync_mode", "C_y", "atac_semantics", "gene"
    ]
    winners = (
        primary.sort_values("max_derivative_cosine", na_position="first")
        .groupby(keys, dropna=False, observed=True, sort=False)
        .tail(1)
        .rename(
            columns={
                "pair_id": "gene_winning_pair_id",
                "peak": "gene_winning_peak",
                "max_derivative_cosine": "gene_max_derivative_cosine",
                "lag_steps_at_max": "gene_lag_steps_at_max",
                "lag_model_time_at_max": "gene_lag_model_time_at_max",
            }
        )
    )
    keep = keys + [
        "gene_winning_pair_id",
        "gene_winning_peak",
        "gene_max_derivative_cosine",
        "gene_lag_steps_at_max",
        "gene_lag_model_time_at_max",
    ]
    circular_columns = [
        f"circular_shift_{shift}_max_derivative_cosine" for shift in (13, 22, 33)
    ]
    gene_circular_rows: list[dict[str, object]] = []
    if all(column in primary.columns for column in circular_columns):
        for (model_id, gene), frame in primary.groupby(
            ["model_id", "gene"], observed=True, sort=False
        ):
            values = [float(frame[column].max()) for column in circular_columns]
            finite = [value for value in values if np.isfinite(value)]
            gene_circular_rows.append(
                {
                    "model_id": model_id,
                    "gene": gene,
                    **{f"gene_{column}": value for column, value in zip(circular_columns, values)},
                    "gene_circular_shift_null_median": (
                        float(np.median(finite)) if finite else np.nan
                    ),
                    "gene_circular_shift_null_max": float(np.max(finite)) if finite else np.nan,
                }
            )
    gene_circular = pd.DataFrame(gene_circular_rows)

    null_primary = null_scores.loc[
        null_scores["smoothing_window"].eq(PRIMARY_SMOOTHING)
    ].copy()
    null_maxima = (
        null_primary.groupby(
            ["model_id", "true_gene", "null_rank"], observed=True, sort=False
        )
        .agg(
            gene_null_max_derivative_cosine=("max_derivative_cosine", "max"),
            candidate_peak_n=("pair_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"true_gene": "gene"})
    )
    null_summary = (
        null_maxima.groupby(["model_id", "gene"], observed=True, sort=False)
        .agg(
            gene_matched_null_n=("gene_null_max_derivative_cosine", "count"),
            gene_matched_null_median=("gene_null_max_derivative_cosine", "median"),
            gene_matched_null_max=("gene_null_max_derivative_cosine", "max"),
        )
        .reset_index()
    )
    scores = winners[keep].merge(null_summary, on=["model_id", "gene"], how="left")
    if not gene_circular.empty:
        scores = scores.merge(gene_circular, on=["model_id", "gene"], how="left")
    scores = scores.merge(evidence, on="gene", how="left")
    empirical_p: list[float] = []
    for row in scores.itertuples(index=False):
        values = null_maxima.loc[
            null_maxima["model_id"].astype(str).eq(str(row.model_id))
            & null_maxima["gene"].astype(str).eq(str(row.gene)),
            "gene_null_max_derivative_cosine",
        ].dropna().to_numpy(float)
        observed = float(row.gene_max_derivative_cosine)
        empirical_p.append(
            float((1 + np.sum(values >= observed)) / (1 + len(values)))
            if len(values) and np.isfinite(observed)
            else np.nan
        )
    scores["gene_matched_empirical_p"] = empirical_p
    scores["gene_matched_empirical_q"] = np.nan
    scores["gene_rank_by_max_correlation"] = np.nan
    for _, index in scores.groupby("model_id", observed=True).groups.items():
        idx = np.asarray(list(index), dtype=int)
        scores.loc[idx, "gene_matched_empirical_q"] = bh(
            scores.loc[idx, "gene_matched_empirical_p"].to_numpy(float)
        )
        scores.loc[idx, "gene_rank_by_max_correlation"] = scores.loc[
            idx, "gene_max_derivative_cosine"
        ].rank(method="min", ascending=False)
    scores["gene_true_minus_matched_null_median"] = (
        scores["gene_max_derivative_cosine"] - scores["gene_matched_null_median"]
    )
    scores["ranking_uses_margin"] = False

    line = (
        line_scores.sort_values("max_derivative_cosine", na_position="first")
        .groupby(["model_id", "group", "gene"], observed=True, sort=False)
        .tail(1)[
            ["model_id", "group", "gene", "pair_id", "peak", "max_derivative_cosine", "lag_steps_at_max"]
        ]
        .rename(
            columns={
                "pair_id": "line_gene_winning_pair_id",
                "peak": "line_gene_winning_peak",
                "max_derivative_cosine": "line_gene_max_derivative_cosine",
                "lag_steps_at_max": "line_gene_lag_steps_at_max",
            }
        )
        .merge(
            evidence[["gene", "gene_locus_perturbation_supported"]],
            on="gene",
            how="left",
        )
    )
    particle = (
        particle_scores.groupby(["model_id", "group", "gene"], observed=True, sort=False)
        .agg(
            gene_observed_max_derivative_cosine=("observed_max_derivative_cosine", "max"),
            gene_paired_shuffle_max_derivative_cosine=(
                "paired_shuffle_max_derivative_cosine", "max"
            ),
        )
        .reset_index()
        .merge(
            evidence[["gene", "gene_locus_perturbation_supported"]],
            on="gene",
            how="left",
        )
    )
    particle["gene_observed_minus_paired_shuffle"] = (
        particle["gene_observed_max_derivative_cosine"]
        - particle["gene_paired_shuffle_max_derivative_cosine"]
    )

    particle_gene = (
        particle.groupby(["model_id", "gene"], observed=True, sort=False)
        .agg(
            observed=("gene_observed_max_derivative_cosine", "mean"),
            shuffled=("gene_paired_shuffle_max_derivative_cosine", "mean"),
        )
        .reset_index()
        .merge(
            evidence[["gene", "gene_locus_perturbation_supported"]],
            on="gene",
            how="left",
        )
    )
    rows: list[dict[str, object]] = []
    for model_id, frame in scores.groupby("model_id", observed=True, sort=False):
        supported = frame.loc[frame["gene_locus_perturbation_supported"].fillna(False)]
        valid = np.isfinite(frame["gene_max_derivative_cosine"])
        labels = frame.loc[valid, "gene_locus_perturbation_supported"].fillna(False).to_numpy(int)
        values = frame.loc[valid, "gene_max_derivative_cosine"].to_numpy(float)
        auprc = (
            average_precision_score(labels, values) if len(np.unique(labels)) > 1 else np.nan
        )
        local_particle = particle_gene.loc[particle_gene["model_id"].eq(model_id)]
        pvalid = np.isfinite(local_particle["observed"]) & np.isfinite(local_particle["shuffled"])
        plabel = local_particle.loc[
            pvalid, "gene_locus_perturbation_supported"
        ].fillna(False).to_numpy(int)
        observed_auprc = (
            average_precision_score(plabel, local_particle.loc[pvalid, "observed"])
            if len(np.unique(plabel)) > 1
            else np.nan
        )
        shuffled_auprc = (
            average_precision_score(plabel, local_particle.loc[pvalid, "shuffled"])
            if len(np.unique(plabel)) > 1
            else np.nan
        )
        supported_particle = particle.loc[
            particle["model_id"].eq(model_id)
            & particle["gene_locus_perturbation_supported"].fillna(False)
        ]
        supported_line = line.loc[
            line["model_id"].eq(model_id)
            & line["gene_locus_perturbation_supported"].fillna(False)
        ]
        first = frame.iloc[0]
        rows.append(
            {
                "model_id": model_id,
                "method": first["method"],
                "balance_mode": first["balance_mode"],
                "sync_mode": first["sync_mode"],
                "C_y": first["C_y"],
                "atac_semantics": first["atac_semantics"],
                "gene_n": int(len(frame)),
                "supported_gene_n": int(len(supported)),
                "supported_gene_AUPRC": auprc,
                "supported_gene_median_true_minus_matched_null": float(
                    supported["gene_true_minus_matched_null_median"].median()
                ),
                "supported_gene_q_le_0p1_n": int(
                    supported["gene_matched_empirical_q"].le(0.10).sum()
                ),
                "supported_gene_positive_ATAC_lead_fraction": float(
                    np.mean(supported["gene_lag_steps_at_max"] > 0)
                ),
                "supported_gene_line_positive_ATAC_lead_fraction": float(
                    np.mean(supported_line["line_gene_lag_steps_at_max"] > 0)
                ),
                "supported_gene_median_observed_minus_paired_shuffle": float(
                    supported_particle["gene_observed_minus_paired_shuffle"].median()
                ),
                "paired_fate_observed_AUPRC": observed_auprc,
                "paired_fate_shuffle_AUPRC": shuffled_auprc,
                "paired_fate_AUPRC_delta": observed_auprc - shuffled_auprc,
            }
        )
    summary = pd.DataFrame(rows)
    posthoc_all = float(
        summary.loc[~summary["sync_mode"].eq("RNA_ATAC_sync"), "supported_gene_AUPRC"].max()
    )
    summary["AUPRC_delta_vs_best_posthoc_all"] = summary["supported_gene_AUPRC"] - posthoc_all
    summary["AUPRC_delta_vs_best_posthoc_balance_matched"] = np.nan
    for balance_mode in ("balanced", "unbalanced_biological_prior"):
        comparator = summary.loc[
            ~summary["sync_mode"].eq("RNA_ATAC_sync")
            & summary["balance_mode"].eq(balance_mode),
            "supported_gene_AUPRC",
        ].max()
        mask = summary["sync_mode"].eq("RNA_ATAC_sync") & summary["balance_mode"].eq(balance_mode)
        summary.loc[mask, "AUPRC_delta_vs_best_posthoc_balance_matched"] = (
            summary.loc[mask, "supported_gene_AUPRC"] - comparator
        )

    sync = summary.loc[summary["sync_mode"].eq("RNA_ATAC_sync")].copy()
    supported_evidence = evidence.loc[evidence["gene_locus_perturbation_supported"]]
    biology_gate = bool(
        len(supported_evidence) >= 3
        and supported_evidence["gene_locus_ko_da_de_sign_consistent"].mean() >= 0.65
        and sync["supported_gene_median_true_minus_matched_null"].median() >= 0.10
        and sync["supported_gene_q_le_0p1_n"].sum() > 0
    )
    per_setting_advantage = (
        sync["AUPRC_delta_vs_best_posthoc_all"].ge(0.05)
        & sync["supported_gene_median_observed_minus_paired_shuffle"].ge(0.05)
        & sync["supported_gene_median_true_minus_matched_null"].gt(0)
    )
    advantage_fraction = float(per_setting_advantage.mean()) if len(sync) else np.nan
    coati_gate = bool(advantage_fraction >= 0.75)
    decision = pd.DataFrame(
        [
            {
                "revision": "one_allowed_gene_locus_evidence_tier_revision",
                "decision": "PASS_COATI" if biology_gate and coati_gate else "PASS_BIOLOGY_ONLY" if biology_gate else "STOP_NEGATIVE",
                "failure_class": "" if biology_gate else "METRIC_OVERESTIMATED",
                "coati_advantage": coati_gate,
                "supported_genes": ";".join(supported_evidence["gene"].astype(str)),
                "supported_gene_n": int(len(supported_evidence)),
                "ko_sign_consistent_fraction": float(
                    supported_evidence["gene_locus_ko_da_de_sign_consistent"].mean()
                ) if len(supported_evidence) else np.nan,
                "median_sync_true_minus_matched_null": float(
                    sync["supported_gene_median_true_minus_matched_null"].median()
                ),
                "sync_settings_with_any_q_le_0p1": int(
                    sync["supported_gene_q_le_0p1_n"].gt(0).sum()
                ),
                "median_sync_AUPRC_delta_vs_best_posthoc_all": float(
                    sync["AUPRC_delta_vs_best_posthoc_all"].median()
                ),
                "median_sync_observed_minus_paired_shuffle": float(
                    sync["supported_gene_median_observed_minus_paired_shuffle"].median()
                ),
                "coati_advantage_setting_fraction": advantage_fraction,
                "stop_rule": (
                    "stop HC01/HC03 headline after this sole revision if matched-null biology "
                    "or native-ATAC comparator gate fails"
                ),
            }
        ]
    )

    scores.to_csv(
        output / "gene_locus_revision_scores.csv.gz", index=False, compression="gzip"
    )
    null_maxima.to_csv(
        output / "gene_locus_revision_matched_null_maxima.csv.gz",
        index=False,
        compression="gzip",
    )
    line.to_csv(
        output / "gene_locus_revision_line_stability.csv.gz", index=False, compression="gzip"
    )
    particle.to_csv(
        output / "gene_locus_revision_paired_shuffle.csv.gz", index=False, compression="gzip"
    )
    summary.to_csv(output / "gene_locus_revision_model_summary.csv", index=False)
    decision.to_csv(output / "gene_locus_revision_decision.csv", index=False)

    order = summary.sort_values("supported_gene_AUPRC")
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.26 * len(order))))
    colors = ["#0072B2" if value == "RNA_ATAC_sync" else "#999999" for value in order["sync_mode"]]
    ax.barh(np.arange(len(order)), order["supported_gene_AUPRC"], color=colors)
    ax.axvline(
        float(evidence["gene_locus_perturbation_supported"].mean()),
        color="black",
        ls=":",
        lw=1,
        label="support prevalence",
    )
    ax.set_yticks(np.arange(len(order)), order["model_id"], fontsize=6)
    ax.set_xlabel("AUPRC for gene/locus perturbation support")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "gene_locus_revision_auprc.pdf")
    fig.savefig(output / "gene_locus_revision_auprc.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.26 * len(order))))
    ax.barh(
        np.arange(len(order)),
        order["supported_gene_median_true_minus_matched_null"],
        color=colors,
    )
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_yticks(np.arange(len(order)), order["model_id"], fontsize=6)
    ax.set_xlabel("Supported-gene max correlation minus matched-null median")
    fig.tight_layout()
    fig.savefig(output / "gene_locus_revision_matched_null_delta.pdf")
    fig.savefig(output / "gene_locus_revision_matched_null_delta.png", dpi=220)
    plt.close(fig)

    audit = {
        "supported_gene_n": int(len(supported_evidence)),
        "supported_genes": supported_evidence["gene"].astype(str).tolist(),
        "best_posthoc_all_AUPRC": posthoc_all,
        "native_sync_setting_n": int(len(sync)),
        "coati_advantage_setting_fraction": advantage_fraction,
        "biology_gate": biology_gate,
        "coati_gate": coati_gate,
        "final_stop": not (biology_gate and coati_gate),
    }
    return scores, summary, decision, audit


def experiment_decisions(
    ranked: pd.DataFrame,
    annotated: pd.DataFrame,
    summaries: pd.DataFrame,
    loci: pd.DataFrame,
) -> pd.DataFrame:
    primary = ranked.loc[ranked["smoothing_window"].eq(PRIMARY_SMOOTHING)].merge(
        annotated, on=["pair_id", "gene", "peak"], how="left", suffixes=("", "_ann")
    ).merge(loci, on="pair_id", how="left", suffixes=("", "_locus"))
    coati = primary.loc[primary["sync_mode"].eq("RNA_ATAC_sync")]
    posthoc = primary.loc[~primary["sync_mode"].eq("RNA_ATAC_sync")]
    gli3 = coati.loc[coati["strong_gli3_locus_support"].fillna(False)]
    genes = int(gli3["gene"].nunique())
    pairs = int(gli3[["gene", "peak"]].drop_duplicates().shape[0])
    gli3_delta = float(gli3["true_minus_matched_null_median"].median()) if len(gli3) else np.nan
    gli3_sign = float(gli3["ko_da_de_sign_consistent"].mean()) if len(gli3) else np.nan
    gli3_atac_lead = float(np.mean(gli3["lag_steps_at_max"] > 0)) if len(gli3) else np.nan
    h3 = summaries
    coati_auprc = h3.loc[h3["sync_mode"].eq("RNA_ATAC_sync"), "supported_link_AUPRC"].median()
    posthoc_auprc = h3.loc[~h3["sync_mode"].eq("RNA_ATAC_sync"), "supported_link_AUPRC"].max()
    atlas_odds = h3.loc[h3["sync_mode"].eq("RNA_ATAC_sync"), "top_decile_supported_odds_ratio"].median()
    atlas_low = h3.loc[h3["sync_mode"].eq("RNA_ATAC_sync"), "top_decile_supported_or_ci_low"].median()
    host = primary.loc[
        primary["overlaps_other_gene_body_or_promoter"].fillna(False)
        & primary["independent_perturbation_supported"].fillna(False)
    ]
    host_loci = int(host["peak"].nunique())
    host_programs = int(host["gene"].nunique())

    hc01_biology = pairs >= 10 and genes >= 3 and gli3_delta >= 0.10 and gli3_sign >= 0.65
    hc01_advantage = bool(
        np.isfinite(coati_auprc)
        and np.isfinite(posthoc_auprc)
        and coati_auprc - posthoc_auprc >= 0.05
    )
    hc03_biology = bool(atlas_odds >= 1.5 and atlas_low > 1.0)
    hc03_advantage = hc01_advantage
    rows = [
        {
            "experiment_id": "HC01",
            "priority": "P0-1",
            "decision": "PASS_COATI" if hc01_biology and hc01_advantage else "PASS_BIOLOGY_ONLY" if hc01_biology else "NEGATIVE_INFORMATIVE",
            "failure_class": "" if hc01_biology else "METRIC_OVERESTIMATED" if pairs >= 10 else "DATA_MISSING",
            "coati_advantage": hc01_advantage,
            "observed": f"strong GLI3 pairs={pairs}, genes={genes}, median true-null={gli3_delta:.3f}, KO sign={gli3_sign:.2f}, ATAC-leading={gli3_atac_lead:.2f}",
            "stop_or_continue": "continue as headline" if hc01_biology else "retain supported loci only; do not claim general GLI3 timing",
        },
        {
            "experiment_id": "HC02",
            "priority": "P0-2",
            "decision": "FAIL_NON_IDENTIFYING",
            "failure_class": "NON_IDENTIFYING_DESIGN",
            "coati_advantage": False,
            "observed": "D18 single-cell multiome was not memory-approved; D45 perturbation does not provide line-matched early ATAC for leave-one-line-out prediction",
            "stop_or_continue": "stop predictive priming claim; retain NKX2-1/ID1 timing as descriptive loci",
        },
        {
            "experiment_id": "HC03",
            "priority": "P0-3",
            "decision": "PASS_COATI" if hc03_biology and hc03_advantage else "PASS_BIOLOGY_ONLY" if hc03_biology else "NEGATIVE_INFORMATIVE",
            "failure_class": "" if hc03_biology else "METRIC_OVERESTIMATED",
            "coati_advantage": hc03_advantage,
            "observed": f"median COATI enrichment OR={atlas_odds:.2f} (median lower bound={atlas_low:.2f}); AUPRC delta vs best post-hoc={coati_auprc-posthoc_auprc:.3f}",
            "stop_or_continue": "continue atlas" if hc03_biology else "drop max-correlation as headline; use diagnostic curves only",
        },
        {
            "experiment_id": "HC04",
            "priority": "P0-4",
            "decision": "PASS_BIOLOGY_ONLY" if host_loci >= 3 and host_programs >= 2 else "NEGATIVE_INFORMATIVE",
            "failure_class": "" if host_loci >= 3 and host_programs >= 2 else "DATA_MISSING",
            "coati_advantage": False,
            "observed": f"independently supported other-gene host loci={host_loci}, target programs={host_programs}",
            "stop_or_continue": "show labelled shared-neighborhood loci" if host_loci >= 3 else "do not generalize; at most labelled anecdotes",
        },
    ]
    return pd.DataFrame(rows)


def export_core_curves(
    curve_paths: list[Path], candidate: pd.DataFrame, output: Path
) -> None:
    core = candidate.loc[candidate["is_core_gene"]].copy()
    rows: list[dict[str, object]] = []
    for path in curve_paths:
        with np.load(path, allow_pickle=True) as saved:
            groups = saved["group_names"].astype(str).tolist()
            gi = groups.index("all")
            genes = saved["gene_names"].astype(str)
            peaks = saved["peak_names"].astype(str)
            gene_index = {name: i for i, name in enumerate(genes)}
            peak_index = {name: i for i, name in enumerate(peaks)}
            time = saved["time"]
            rna = saved["rna_cpm"]
            atac = saved["atac_cp10k"]
            for pair in core.itertuples(index=False):
                if pair.gene not in gene_index or pair.peak not in peak_index:
                    continue
                for ti, value in enumerate(time):
                    rows.append(
                        {
                            "model_id": path.stem,
                            "pair_id": pair.pair_id,
                            "gene": pair.gene,
                            "peak": pair.peak,
                            "model_time": float(value),
                            "rna_cpm": float(rna[gi, ti, gene_index[pair.gene]]),
                            "atac_cp10k": float(atac[gi, ti, peak_index[pair.peak]]),
                        }
                    )
    pd.DataFrame(rows).to_csv(
        output / "trajectory_peak_gene_curves.csv.gz", index=False, compression="gzip"
    )


def plots(
    ranked: pd.DataFrame,
    annotated: pd.DataFrame,
    summaries: pd.DataFrame,
    curve_paths: list[Path],
    output: Path,
) -> None:
    mpl.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    order = summaries.sort_values("supported_link_AUPRC", ascending=True)
    colors = ["#0072B2" if value == "RNA_ATAC_sync" else "#999999" for value in order["sync_mode"]]
    ax.barh(np.arange(len(order)), order["supported_link_AUPRC"], color=colors)
    ax.set_yticks(np.arange(len(order)), order["model_id"])
    ax.set_xlabel("AUPRC for perturbation-supported peak–gene pairs")
    ax.set_title("Native Sync ATAC (blue) versus fixed-T post-hoc ATAC (gray)")
    fig.tight_layout()
    fig.savefig(output / "model_supported_pair_auprc.pdf")
    fig.savefig(output / "model_supported_pair_auprc.png", dpi=220)
    plt.close(fig)

    representative = "coati_sync_unbalanced_cy0.5_s0_i40000"
    rep = ranked.loc[
        ranked["model_id"].eq(representative)
        & ranked["smoothing_window"].eq(PRIMARY_SMOOTHING)
    ].merge(annotated, on=["pair_id", "gene", "peak"], how="left")
    selected = rep.loc[rep["independent_perturbation_supported"].fillna(False)].nlargest(
        12, "max_derivative_cosine"
    )
    if selected.empty:
        selected = rep.nlargest(12, "max_derivative_cosine")
    path = next((x for x in curve_paths if x.stem == representative), None)
    if path is not None and len(selected):
        with np.load(path, allow_pickle=True) as saved:
            groups = saved["group_names"].astype(str).tolist()
            gi = groups.index("all")
            genes = saved["gene_names"].astype(str)
            peaks = saved["peak_names"].astype(str)
            gene_index = {name: i for i, name in enumerate(genes)}
            peak_index = {name: i for i, name in enumerate(peaks)}
            time = saved["time"]
            rna = saved["rna_cpm"]
            atac = saved["atac_cp10k"]
            ncol = 4
            nrow = int(math.ceil(len(selected) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.4 * nrow), squeeze=False)
            for ax, row in zip(axes.ravel(), selected.itertuples(index=False)):
                rv = np.log1p(rna[gi, :, gene_index[row.gene]])
                av = np.log1p(atac[gi, :, peak_index[row.peak]])
                rz = (rv - rv.mean()) / max(rv.std(), 1e-12)
                az = (av - av.mean()) / max(av.std(), 1e-12)
                ax.plot(time, rz, color="#0072B2", label="RNA")
                ax.plot(time, az, color="#D55E00", ls="--", label="ATAC")
                ax.scatter(PHYSICAL_TIMES, np.interp(PHYSICAL_TIMES, time, rz), s=10, color="#0072B2")
                ax.scatter(PHYSICAL_TIMES, np.interp(PHYSICAL_TIMES, time, az), s=10, color="#D55E00")
                ax.set_title(f"{row.gene} | r={row.max_derivative_cosine:.2f}, lag={row.lag_model_time_at_max:+.2f}")
                ax.set_xticks(PHYSICAL_TIMES, ["D4", "D7", "D9", "D11", "D12", "D18", "D21"], rotation=45)
            for ax in axes.ravel()[len(selected):]:
                ax.axis("off")
            axes[0, 0].legend(frameon=False)
            fig.suptitle("COATI-U C_y=0.5: ranked perturbation-supported pairs", y=1.0)
            fig.tight_layout()
            fig.savefig(output / "supported_pair_panels.pdf", bbox_inches="tight")
            fig.savefig(output / "supported_pair_panels.png", dpi=220, bbox_inches="tight")
            plt.close(fig)

    top = rep.nlargest(40, "max_derivative_cosine").sort_values("max_derivative_cosine")
    fig, ax = plt.subplots(figsize=(7.5, 8.0))
    labels = top["gene"].astype(str) + " | " + top["peak"].astype(str)
    ax.barh(np.arange(len(top)), top["max_derivative_cosine"], color="#0072B2")
    ax.scatter(top["matched_null_median"], np.arange(len(top)), color="#D55E00", s=12, label="matched-null median")
    ax.set_yticks(np.arange(len(top)), labels, fontsize=6)
    ax.set_xlabel("Maximum derivative cosine (margin not used)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "all_ranked_pairs.pdf")
    fig.savefig(output / "all_ranked_pairs.png", dpi=220)
    plt.close(fig)


def write_readme(
    output: Path,
    decisions: pd.DataFrame,
    revision_decision: pd.DataFrame,
    manifest: dict[str, object],
) -> None:
    lines = [
        "# Human cerebral peak–gene max-correlation analysis",
        "",
        "## Outcome",
        "",
        "This analysis used only the 23 frozen local full rollouts. No model was trained and HPC was not accessed.",
        "Pando regions were mapped to the assayed ATAC atlas by GRCh38 interval overlap. The biological readout used stage-conditioned Gaussian barycentric kNN with fixed `k=15` in each native space.",
        "",
        "Pairs are ranked only by `max_derivative_cosine`. `winning_margin_diagnostic_only` is retained for audit and never used for filtering, ranking, or labels.",
        "",
        "## P0 decisions",
        "",
    ]
    for row in decisions.itertuples(index=False):
        lines.extend(
            [
                f"- **{row.priority} / {row.experiment_id}: {row.decision}.** {row.observed}",
                f"  Next: {row.stop_or_continue}.",
            ]
        )
    revision = revision_decision.iloc[0]
    lines.extend(
        [
            "",
            "## One allowed evidence-tier revision",
            "",
            (
                "The only revision lifted independent support to a fixed gene/locus endpoint: "
                "a Pando GLI3→target edge, significant early-telencephalic GLI3-KO DE, and at "
                "least one frozen paper-ranked KO DA peak within the canonical GENCODE TSS "
                "±100 kb. The gene score is the maximum across every frozen Pando candidate "
                "peak; every matched null uses the identical lag scan and identical cross-peak maximum."
            ),
            "",
            (
                f"Final revision decision: **{revision['decision']}**. Supported genes: "
                f"{revision['supported_genes']}. Median native-Sync true-minus-null: "
                f"{revision['median_sync_true_minus_matched_null']:.3f}; median native-Sync "
                f"AUPRC delta versus the best fixed post-hoc model: "
                f"{revision['median_sync_AUPRC_delta_vs_best_posthoc_all']:.3f}; native setting "
                f"pass fraction: {revision['coati_advantage_setting_fraction']:.2f}."
            ),
            "",
            "The preregistered stop rule is therefore final: do not use HC01/HC03 as a COATI-advantage headline. HC04 remains two labelled supplementary loci; its threshold was not relaxed.",
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Pando is a same-study candidate prior, not independent validation.",
            "- Supplementary Tables 9–10 provide perturbation direction; CUT&Tag occupancy is not available as a complete local peak table.",
            "- A peak inside another gene locus is called a shared-neighborhood association, never direct regulation by overlap alone.",
            "- Native synchronized ATAC is shown separately from fixed-T post-hoc ATAC for RNA-only methods.",
            "- Full-training trajectories support reconstruction and model-implied interpolation, not held-out generalization.",
            "- The requested fixed `k=15` contract supersedes the older two-k sensitivity suggestion; smoothing windows 5 and 7 are both reported.",
            "",
            "## Main files",
            "",
            "- `frozen_peak_gene_candidates.csv.gz`",
            "- `trajectory_peak_gene_curves.csv.gz` (core-gene compact curves)",
            "- `decoded_feature_curves/*.npz` (complete wide feature curves)",
            "- `shifted_derivative_correlations.csv.gz`",
            "- `matched_null_scores.csv.gz`",
            "- `max_correlation_rankings.csv.gz`",
            "- `nearest_and_host_gene_annotations.csv`",
            "- `external_evidence_annotations.csv.gz`",
            "- `gene_locus_external_evidence_revision.csv`",
            "- `gene_locus_revision_scores.csv.gz`",
            "- `gene_locus_revision_model_summary.csv`",
            "- `gene_locus_revision_decision.csv`",
            "- `experiment_decisions.csv`",
            "- `analysis_manifest.json`",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    required = [
        PANDO_CANDIDATES,
        MODEL_REGISTRY,
        RNA_H5AD,
        ATAC_H5AD,
        GTF,
        GLI3_KO_DA_ALL,
        GLI3_KO_DA_PAPER5000,
        GLI3_KO_DA_STRICT_NEGATIVE,
        GLI3_KO_DE_EARLY,
        GLI3_KO_DE_VENTRAL,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    candidates, candidate_audit = freeze_candidates(
        args.output_dir, args.pilot or args.core_only
    )
    genes, transcripts = load_gtf()
    loci = annotate_loci(candidates, genes, transcripts)
    loci.to_csv(args.output_dir / "nearest_and_host_gene_annotations.csv", index=False)
    annotated, external_audit = annotate_external(candidates)
    annotated.to_csv(
        args.output_dir / "external_evidence_annotations.csv.gz", index=False, compression="gzip"
    )
    gene_locus_evidence, gene_locus_evidence_audit = build_gene_locus_evidence(
        candidates, genes
    )
    gene_locus_evidence.to_csv(
        args.output_dir / "gene_locus_external_evidence_revision.csv", index=False
    )
    nulls = construct_matched_nulls(annotated, genes, NULLS_PER_PAIR)
    nulls.to_csv(args.output_dir / "matched_null_pairs.csv.gz", index=False, compression="gzip")
    reference, candidates, molecular_audit = load_molecular_reference(annotated, nulls, loci)

    registry = pd.read_csv(MODEL_REGISTRY)
    registry = registry.loc[registry["completion_status"].astype(str).str.startswith("analysis_ready")].copy()
    if args.pilot:
        representatives = {
            "coati_sync_balanced_cy0.5_s0_i30000",
            "coati_sync_unbalanced_cy0.5_s0_i40000",
            "coati_rna_only_balanced_s0_i30000",
            "coati_rna_only_unbalanced_biological_prior_s0_i30000",
            "cytobridge_balanced_s42_i30000",
            "cytobridge_unbalanced_biological_prior_s42_i30000",
            "mioflow_gaga10_balanced_s42_i30000",
            "trajectorynet_forward_balanced_s0_i30000",
        }
        registry = registry.loc[registry["model_id"].isin(representatives)].copy()
    if len(registry) != (8 if args.pilot else 23):
        raise ValueError(f"Unexpected number of registered rollouts: {len(registry)}")

    curve_paths: list[Path] = []
    audits: list[pd.DataFrame] = []
    for number, row in enumerate(registry.itertuples(index=False), start=1):
        print(f"[{number}/{len(registry)}] decoding {row.model_id}", flush=True)
        rollout = load_rollout(pd.Series(row._asdict()))
        path, audit = decode_rollout(rollout, reference, args.output_dir)
        curve_paths.append(path)
        audits.append(audit)
    pd.concat(audits, ignore_index=True).to_csv(args.output_dir / "readout_group_audit.csv", index=False)
    export_core_curves(curve_paths, candidates, args.output_dir)

    print("Scoring observed pairs and identical max-over-lag nulls", flush=True)
    true, null_scores, line_scores, particle_scores = score_curves(
        curve_paths, candidates, nulls, registry
    )
    true.to_csv(
        args.output_dir / "shifted_derivative_correlations.csv.gz", index=False, compression="gzip"
    )
    null_scores.to_csv(
        args.output_dir / "matched_null_scores.csv.gz", index=False, compression="gzip"
    )
    line_scores.to_csv(
        args.output_dir / "biological_line_stability_scores.csv.gz", index=False, compression="gzip"
    )
    particle_scores.to_csv(
        args.output_dir / "paired_particle_shuffle_scores.csv.gz", index=False, compression="gzip"
    )
    ranked = calibrate_and_rank(true, null_scores)
    ranked.to_csv(
        args.output_dir / "max_correlation_rankings.csv.gz", index=False, compression="gzip"
    )
    stable = stability_summary(ranked, line_scores)
    stable.to_csv(args.output_dir / "max_correlation_primary_with_stability.csv.gz", index=False, compression="gzip")
    summaries = model_summaries(ranked, annotated, particle_scores)
    summaries.to_csv(args.output_dir / "model_level_peak_gene_summary.csv", index=False)
    decisions = experiment_decisions(ranked, annotated, summaries, loci)
    decisions.to_csv(args.output_dir / "experiment_decisions.csv", index=False)
    _, gene_locus_summary, revision_decision, gene_locus_revision_audit = gene_locus_revision(
        ranked,
        null_scores,
        line_scores,
        particle_scores,
        gene_locus_evidence,
        args.output_dir,
    )
    plots(ranked, annotated, summaries, curve_paths, args.output_dir)

    valid_null_fraction = float(np.mean(ranked["matched_null_n"].fillna(0) >= 5))
    manifest = {
        "task": "human cerebral P0-1 through P0-4 peak-gene max-correlation analysis",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot": args.pilot,
        "core_only": args.core_only,
        "input_policy": "only frozen local 01_trajectory_materialization rollouts; no HPC",
        "n_models": len(registry),
        "model_ids": registry["model_id"].astype(str).tolist(),
        "candidate_audit": candidate_audit,
        "molecular_audit": molecular_audit,
        "external_audit": external_audit,
        "gene_locus_evidence_revision_audit": gene_locus_evidence_audit,
        "gene_locus_revision_result_audit": gene_locus_revision_audit,
        "gtf": {"path": str(GTF), "sha256": sha256(GTF), "release": "GENCODE v48 GRCh38.p14"},
        "analysis_contract": {
            "knn_k": KNN_K,
            "knn": "stage-conditioned Gaussian-distance barycentric readout",
            "different_space_rule": "MIOFlow neighbors in frozen full GAGA10; all other RNA neighbors in normalized PCA30; molecular output through fixed observed-reference barycenter",
            "native_posthoc_rule": "COATI Sync native/model-coupled ATAC separated from post-hoc fixed-T ATAC",
            "smoothing_windows": list(SMOOTHING_GRID),
            "primary_smoothing": PRIMARY_SMOOTHING,
            "max_lag_steps": MAX_LAG_STEPS,
            "ranking": "max_derivative_cosine only",
            "margin": "diagnostic only; never gates or ranks",
            "matched_null": "same peak paired to non-Pando gene on same chromosome with nearest TSS-distance; exact GC/accessibility/length matching by construction",
            "nulls_per_pair_requested": NULLS_PER_PAIR,
            "circular_shifts": [13, 22, 33],
            "paired_particle_shuffle": "fixed deterministic permutation within source line; fate groups assigned by terminal RNA k=15",
        },
        "quality": {
            "valid_matched_null_fraction_at_least_5": valid_null_fraction,
            "all_scores_ranked_without_margin": bool((~ranked["ranking_uses_margin"]).all()),
            "finite_primary_score_fraction": float(
                np.mean(np.isfinite(ranked.loc[ranked["smoothing_window"].eq(PRIMARY_SMOOTHING), "max_derivative_cosine"]))
            ),
        },
        "decisions": decisions.to_dict(orient="records"),
        "one_allowed_revision_decision": revision_decision.to_dict(orient="records"),
        "limitations": [
            "Pando is same-study candidate evidence, not independent validation.",
            "Supplementary GLI3 perturbation DA/DE supplies direction but not a complete CUT&Tag occupancy table.",
            "C_y values are sensitivity settings, not replicates.",
            "Full-train trajectories do not establish held-out generalization.",
            "k=15 is fixed by the user contract; no alternate-k result is used.",
        ],
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_readme(args.output_dir, decisions, revision_decision, manifest)
    print(decisions.to_string(index=False), flush=True)
    print(revision_decision.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
