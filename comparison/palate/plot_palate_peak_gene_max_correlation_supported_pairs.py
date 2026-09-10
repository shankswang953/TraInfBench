#!/usr/bin/env python
"""Rank palate peak--gene trajectories by maximum shifted derivative cosine.

The ranking intentionally ignores the historical winning-margin gate.  The
displayed examples are selected from genes with independent palate-function
evidence and peaks with branch-matched external H3K27ac support.  Observed
stage pseudobulk points are not drawn; every curve in the figure is decoded
from a COATI trajectory.
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
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results/palate_coati_peak_gene_trajectory_time_classes_cy05"
DEFAULT_EXTERNAL = (
    ROOT
    / "results/palate_strict_loo_external_epigenetic_evidence"
    / "external_peak_annotations_with_meox2.csv.gz"
)
DEFAULT_OUTPUT = ROOT / "results/palate_peak_gene_max_correlation_supported_pairs"
DEFAULT_GTF = Path(
    "external/COATI/MouseBrain/"
    "DataGen/Mouse/gencode.vM25.annotation.gtf.gz"
)

METHODS = ("COATI-B", "COATI-U")
RNA_COLOR = "#0072B2"
ATAC_COLOR = "#D55E00"
FUNCTIONALLY_SUPPORTED_GENES = ("Satb2", "Sim2", "Shox2", "Prickle1")
GENE_EVIDENCE = {
    "Satb2": (
        "Anterior marker in the palate atlas; palatal-mesenchyme expression and "
        "cleft-palate/craniofacial phenotypes support developmental relevance."
    ),
    "Sim2": (
        "Posterior marker in the palate atlas; Sim2-null mice show secondary "
        "cleft palate and altered palatal mesenchyme."
    ),
    "Shox2": (
        "Anterior lineage regulator in the palate atlas; anterior-restricted "
        "cleft and hard-palate patterning defects follow Shox2 loss."
    ),
    "Prickle1": (
        "Posterior marker in the palate atlas; Prickle1 mutant mice develop a "
        "complete secondary cleft palate."
    ),
}
EVIDENCE_URL = {
    "Satb2": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3058410/",
    "Sim2": "https://doi.org/10.1002/dvdy.10116",
    "Shox2": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6885637/",
    "Prickle1": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3960056/",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--external-annotations", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--gtf", type=Path, default=DEFAULT_GTF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_gene_bounds(gtf_path: Path, genes: set[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    opener = gzip.open if gtf_path.suffix == ".gz" else open
    with opener(gtf_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            match = re.search(r'gene_name "([^"]+)"', fields[8])
            if match is None or match.group(1) not in genes:
                continue
            records.append(
                {
                    "gene": match.group(1),
                    "gene_chrom": fields[0],
                    "gene_start": int(fields[3]),
                    "gene_end": int(fields[4]),
                    "gene_strand": fields[6],
                }
            )
    bounds = pd.DataFrame(records).drop_duplicates("gene")
    missing = genes - set(bounds["gene"].astype(str))
    if missing:
        raise RuntimeError(f"Genes absent from {gtf_path}: {sorted(missing)}")
    return bounds


def configure_plot() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "font.weight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def trajectory_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = float(np.std(values, ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / scale


def build_rankings(
    classes: pd.DataFrame,
    links: pd.DataFrame,
    external: pd.DataFrame,
    gene_bounds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    link_columns = [
        "gene",
        "peak",
        "signed_offset_bp",
        "region_type",
        "zscore",
        "pvalue",
        "static_rank_within_gene",
    ]
    annotation_columns = [
        "peak",
        "h3_anterior_specific",
        "h3_posterior_specific",
        "shox2_bound_anterior_active",
        "meox2_bound_posterior_active",
        "shox2_cis_linked",
    ]
    ranked = classes.merge(
        links[link_columns], on=["gene", "peak"], how="left", validate="many_to_one"
    ).merge(
        external[annotation_columns], on="peak", how="left", validate="many_to_one"
    ).merge(
        gene_bounds, on="gene", how="left", validate="many_to_one"
    )
    boolean_columns = annotation_columns[1:]
    ranked[boolean_columns] = ranked[boolean_columns].fillna(False).astype(bool)
    peak_coordinates = ranked["peak"].str.extract(
        r"^(?P<peak_chrom>[^-]+)-(?P<peak_start>\d+)-(?P<peak_end>\d+)$"
    )
    peak_start = peak_coordinates["peak_start"].astype(int)
    peak_end = peak_coordinates["peak_end"].astype(int)
    ranked["overlaps_gene_body"] = (
        peak_coordinates["peak_chrom"].eq(ranked["gene_chrom"])
        & peak_end.ge(ranked["gene_start"])
        & peak_start.le(ranked["gene_end"])
    )
    ranked["strict_distal"] = (
        ~ranked["overlaps_gene_body"]
        & ranked["signed_offset_bp"].abs().gt(2000)
    )
    ranked["branch_matched_h3k27ac"] = (
        ranked["branch"].eq("anterior") & ranked["h3_anterior_specific"]
    ) | (
        ranked["branch"].eq("posterior") & ranked["h3_posterior_specific"]
    )
    ranked["branch_matched_tf_cre"] = (
        ranked["branch"].eq("anterior")
        & ranked["shox2_bound_anterior_active"]
    ) | (
        ranked["branch"].eq("posterior")
        & ranked["meox2_bound_posterior_active"]
    )
    ranked = ranked.rename(
        columns={
            "winning_score": "max_derivative_cosine",
            "chosen_lag_days": "lag_at_max_days",
        }
    )
    ranked["rank_within_model"] = ranked.groupby("model")[
        "max_derivative_cosine"
    ].rank(method="first", ascending=False).astype(int)
    ranked["rank_within_model_and_cohort"] = ranked.groupby(
        ["model", "cohort"]
    )["max_derivative_cosine"].rank(method="first", ascending=False).astype(int)
    ranked = ranked.sort_values(
        ["model", "max_derivative_cosine"], ascending=[True, False]
    ).reset_index(drop=True)

    keys = [
        "cohort",
        "branch",
        "gene",
        "peak",
        "signed_offset_bp",
        "region_type",
        "overlaps_gene_body",
        "strict_distal",
        "branch_matched_h3k27ac",
        "branch_matched_tf_cre",
        "shox2_cis_linked",
    ]
    summary = (
        ranked.groupby(keys, as_index=False, observed=True)
        .agg(
            mean_max_correlation=("max_derivative_cosine", "mean"),
            minimum_max_correlation=("max_derivative_cosine", "min"),
            maximum_max_correlation=("max_derivative_cosine", "max"),
            mean_lag_days=("lag_at_max_days", "mean"),
            n_models=("model", "nunique"),
        )
        .sort_values(
            ["mean_max_correlation", "minimum_max_correlation"], ascending=False
        )
        .reset_index(drop=True)
    )
    summary["overall_rank"] = np.arange(1, len(summary) + 1)
    summary["gene_functionally_supported"] = summary["gene"].isin(
        FUNCTIONALLY_SUPPORTED_GENES
    )
    summary["gene_evidence"] = summary["gene"].map(GENE_EVIDENCE).fillna("")
    summary["gene_evidence_url"] = summary["gene"].map(EVIDENCE_URL).fillna("")
    return ranked, summary


def choose_supported_pairs(summary: pd.DataFrame) -> pd.DataFrame:
    eligible = summary.loc[
        summary["gene_functionally_supported"]
        & summary["branch_matched_h3k27ac"]
        & summary["strict_distal"]
        & summary["n_models"].eq(len(METHODS))
    ].copy()
    selected = (
        eligible.sort_values("mean_max_correlation", ascending=False)
        .drop_duplicates("gene", keep="first")
        .sort_values("mean_max_correlation", ascending=False)
        .head(len(FUNCTIONALLY_SUPPORTED_GENES))
        .reset_index(drop=True)
    )
    missing = set(FUNCTIONALLY_SUPPORTED_GENES) - set(selected["gene"])
    if missing:
        raise RuntimeError(f"No branch-matched H3K27ac-supported pair for {sorted(missing)}")
    selected["display_order"] = np.arange(1, len(selected) + 1)
    return selected


def plot_method(
    method: str,
    selected: pd.DataFrame,
    ranked: pd.DataFrame,
    curves: pd.DataFrame,
    output: Path,
) -> tuple[Path, Path]:
    configure_plot()
    figure, axes = plt.subplots(2, 2, figsize=(3.55, 3.40), sharex=True, sharey=True)
    axes = axes.ravel()
    for axis, pair in zip(axes, selected.itertuples(index=False)):
        score_row = ranked.loc[
            ranked["model"].eq(method)
            & ranked["cohort"].eq(pair.cohort)
            & ranked["branch"].eq(pair.branch)
            & ranked["gene"].eq(pair.gene)
            & ranked["peak"].eq(pair.peak)
        ]
        if len(score_row) != 1:
            raise RuntimeError(f"Expected one score row for {method} {pair.gene}")
        score_row = score_row.iloc[0]
        local = curves.loc[
            curves["model"].eq(method)
            & curves["cohort"].eq(pair.cohort)
            & curves["branch"].eq(pair.branch)
            & curves["gene"].eq(pair.gene)
            & curves["peak"].eq(pair.peak)
        ].sort_values("time_index")
        x = local["embryonic_day"].to_numpy(float)
        rna = savgol_filter(np.log1p(local["rna_cpm"]), 5, 2, mode="interp")
        atac = savgol_filter(np.log1p(local["atac_cp10k"]), 5, 2, mode="interp")
        axis.plot(x, trajectory_zscore(rna), color=RNA_COLOR, lw=1.35, label="RNA")
        axis.plot(
            x,
            trajectory_zscore(atac),
            color=ATAC_COLOR,
            lw=1.35,
            ls="--",
            label="ATAC",
        )
        offset_kb = float(pair.signed_offset_bp) / 1000.0
        target_celltype = "Anterior" if pair.branch == "anterior" else "Posterior"
        source_celltype = (
            "CNC progenitor"
            if pair.cohort == "CNC_to_branch"
            else target_celltype
        )
        axis.set_title(f"{pair.gene}  {offset_kb:+.1f} kb", pad=2, fontsize=9.5)
        axis.text(
            0.03,
            0.94,
            (
                f"{source_celltype} → {target_celltype}\n"
                f"lag {float(score_row.lag_at_max_days):+.2f} d"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.4,
            linespacing=1.0,
        )
        axis.axhline(0, color="0.72", lw=0.5)
        axis.grid(axis="y", color="0.92", lw=0.45)
        axis.set_xlim(12.45, 14.60)
        axis.set_xticks([12.5, 13.5, 14.5], ["E12.5", "E13.5", "E14.5"])
        tick_labels = axis.get_xticklabels()
        if tick_labels:
            tick_labels[0].set_ha("left")
            tick_labels[-1].set_ha("right")
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.0))
    figure.supylabel("Trajectory z-score", x=0.025, fontsize=9.5)
    figure.supxlabel("Embryonic stage", y=0.035, fontsize=9.5)
    figure.subplots_adjust(left=0.16, right=0.985, top=0.855, bottom=0.17, wspace=0.05, hspace=0.34)
    stem = output / f"palate_{method.lower().replace('-', '_')}_max_correlation_supported_pairs"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(
        png,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    figure.savefig(
        pdf,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)
    return png, pdf


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected = [
        args.output_dir / "all_peak_gene_max_correlation_rankings.csv",
        args.output_dir / "biologically_supported_pair_ranking.csv",
        args.output_dir / "strict_distal_h3_supported_pair_ranking.csv",
        *[
            args.output_dir
            / f"palate_{method.lower().replace('-', '_')}_max_correlation_supported_pairs.{extension}"
            for method in METHODS
            for extension in ("png", "pdf")
        ],
    ]
    if not args.overwrite and any(path.exists() for path in expected):
        raise FileExistsError("Outputs exist; pass --overwrite")

    classes = pd.read_csv(args.input_dir / "shifted_correlation_timing_classes.csv")
    curves = pd.read_csv(args.input_dir / "trajectory_time_curves.csv.gz")
    links = pd.read_csv(args.input_dir / "frozen_paper_marker_full_catalogue_links.csv")
    external = pd.read_csv(args.external_annotations)
    gene_bounds = load_gene_bounds(args.gtf, set(classes["gene"].astype(str)))
    ranked, summary = build_rankings(classes, links, external, gene_bounds)
    selected = choose_supported_pairs(summary)
    strict_distal_h3 = summary.loc[
        summary["strict_distal"]
        & summary["branch_matched_h3k27ac"]
        & summary["n_models"].eq(len(METHODS))
    ].sort_values(
        ["mean_max_correlation", "minimum_max_correlation"], ascending=False
    )

    ranked.to_csv(expected[0], index=False)
    selected.to_csv(expected[1], index=False)
    strict_distal_h3.to_csv(expected[2], index=False)
    for method in METHODS:
        plot_method(method, selected, ranked, curves, args.output_dir)

    print(selected[
        [
            "overall_rank",
            "gene",
            "peak",
            "cohort",
            "branch",
            "mean_max_correlation",
            "minimum_max_correlation",
            "mean_lag_days",
        ]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
