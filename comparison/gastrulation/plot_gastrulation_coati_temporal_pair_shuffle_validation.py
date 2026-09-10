#!/usr/bin/env python3
"""Validate COATI temporal peak--gene concordance against pair-identity shuffling.

The null keeps the exact gene and peak trajectories used by the observed
candidate set, but breaks their identities by pairing every selected gene with
every selected peak that was not one of the 187 frozen observed pairs.  The
same smoothing, derivative, lag search, and maximum-score rule used by the
primary analysis is applied to observed and mismatched pairs.

This script also recomputes the descriptive E8.5 branch-preference audit at the
time-matched COATI time u=0.8 (E8.5 on the E7.5--E8.75 display scale).
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
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results/gastrulation_coati_b_objective_lineage_temporal"
DEFAULT_OUTPUT = ROOT / "results/gastrulation_coati_b_temporal_pair_shuffle_validation"

LINEAGES = ("NMP-retaining", "Neural", "Mesoderm")
N_GRID = 41
MAX_LAG_STEPS = 5
SMOOTHING_WINDOW = 5
SMOOTHING_ORDER = 2
E85_NORMALIZED_TIME = 0.8
N_PAIRING_PERMUTATIONS = 10_000
PERMUTATION_SEED = 1

COATI_BLUE = "#0072B2"
ATAC_ORANGE = "#E69F00"
NULL_GREY = "#B8B8B8"
GRID_GREY = "#D9D9D9"
TEXT_GREY = "#4D4D4D"
FGF3_CANDIDATE = "Fgf3|chr7:144884955-144885555"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def derivative(values: np.ndarray, native_time: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, N_GRID)
    interpolated = np.interp(grid, native_time, np.log1p(values))
    return savgol_filter(
        interpolated,
        SMOOTHING_WINDOW,
        SMOOTHING_ORDER,
        deriv=1,
        delta=1.0 / (N_GRID - 1),
    )


def lagged_score_matrix(
    gene_derivatives: np.ndarray, peak_derivatives: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return gene x peak maximum cosine and the corresponding lag."""
    best = np.full((len(gene_derivatives), len(peak_derivatives)), -np.inf)
    best_lag = np.zeros_like(best, dtype=int)
    for lag in range(-MAX_LAG_STEPS, MAX_LAG_STEPS + 1):
        if lag < 0:
            gene = gene_derivatives[:, :lag]
            peak = peak_derivatives[:, -lag:]
        elif lag > 0:
            gene = gene_derivatives[:, lag:]
            peak = peak_derivatives[:, :-lag]
        else:
            gene = gene_derivatives
            peak = peak_derivatives
        numerator = gene @ peak.T
        denominator = np.linalg.norm(gene, axis=1)[:, None] * np.linalg.norm(
            peak, axis=1
        )[None, :]
        score = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=float),
            where=denominator > 1e-12,
        )
        update = np.isfinite(score) & (
            (score > best)
            | (
                np.isclose(score, best)
                & ((abs(lag) < np.abs(best_lag)) | ((abs(lag) == np.abs(best_lag)) & (lag < best_lag)))
            )
        )
        best[update] = score[update]
        best_lag[update] = lag
    best[~np.isfinite(best)] = np.nan
    return best, best_lag


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def build_pair_shuffle_scores(
    native: pd.DataFrame, audit: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed_pairs = set(map(tuple, audit[["gene", "peak"]].drop_duplicates().to_numpy()))
    all_rows: list[pd.DataFrame] = []
    observed_rows: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    permutation_rows: list[dict[str, object]] = []

    for lineage_index, lineage in enumerate(LINEAGES):
        frame = native.loc[native["lineage"].eq(lineage)]
        gene_names = sorted(frame["gene"].astype(str).unique())
        peak_names = sorted(frame["peak"].astype(str).unique())
        gene_derivatives = []
        peak_derivatives = []

        for gene in gene_names:
            local = (
                frame.loc[frame["gene"].eq(gene)]
                .drop_duplicates("time_index_native")
                .sort_values("normalized_time_native")
            )
            gene_derivatives.append(
                derivative(
                    local["rna_cpm"].to_numpy(float),
                    local["normalized_time_native"].to_numpy(float),
                )
            )
        for peak in peak_names:
            local = (
                frame.loc[frame["peak"].eq(peak)]
                .drop_duplicates("time_index_native")
                .sort_values("normalized_time_native")
            )
            peak_derivatives.append(
                derivative(
                    local["atac_cp10k"].to_numpy(float),
                    local["normalized_time_native"].to_numpy(float),
                )
            )

        scores, lags = lagged_score_matrix(
            np.asarray(gene_derivatives), np.asarray(peak_derivatives)
        )
        gene_index, peak_index = np.indices(scores.shape)
        local_scores = pd.DataFrame(
            {
                "lineage": lineage,
                "gene": np.asarray(gene_names)[gene_index.ravel()],
                "peak": np.asarray(peak_names)[peak_index.ravel()],
                "S_max": scores.ravel(),
                "lag_at_max_steps": lags.ravel(),
            }
        )
        local_scores["candidate_id"] = (
            local_scores["gene"] + "|" + local_scores["peak"]
        )
        local_scores["pairing"] = np.where(
            [tuple(row) in observed_pairs for row in local_scores[["gene", "peak"]].to_numpy()],
            "Observed pair",
            "Mismatched pair",
        )
        observed = local_scores.loc[local_scores["pairing"].eq("Observed pair")].copy()
        mismatched = local_scores.loc[local_scores["pairing"].eq("Mismatched pair")].copy()

        reference = audit.loc[audit["lineage"].eq(lineage)].set_index("candidate_id")["S_max"]
        reproduced = observed.set_index("candidate_id")["S_max"].reindex(reference.index)
        maximum_error = float(np.nanmax(np.abs(reproduced.to_numpy() - reference.to_numpy())))
        if maximum_error > 1e-10:
            raise RuntimeError(f"Observed score reproduction failed for {lineage}: {maximum_error}")

        null_values = mismatched["S_max"].dropna().to_numpy(float)
        observed_values = observed["S_max"].dropna().to_numpy(float)
        empirical_p = np.asarray(
            [(1.0 + np.sum(null_values >= value)) / (1.0 + len(null_values)) for value in observed_values]
        )
        observed.loc[observed["S_max"].notna(), "empirical_p"] = empirical_p
        observed["empirical_BH_q"] = bh_adjust(observed["empirical_p"].to_numpy(float))

        u_stat, p_value = mannwhitneyu(
            observed_values, null_values, alternative="greater"
        )
        gene_lookup = {gene: index for index, gene in enumerate(gene_names)}
        peak_lookup = {peak: index for index, peak in enumerate(peak_names)}
        pair_order = audit.loc[audit["lineage"].eq(lineage), ["gene", "peak"]].drop_duplicates()
        observed_gene_indices = pair_order["gene"].map(gene_lookup).to_numpy(int)
        observed_peak_indices = pair_order["peak"].map(peak_lookup).to_numpy(int)
        rng = np.random.default_rng(PERMUTATION_SEED + lineage_index)
        permutation_medians = np.empty(N_PAIRING_PERMUTATIONS, dtype=float)
        for permutation in range(N_PAIRING_PERMUTATIONS):
            permuted_peak_indices = rng.permutation(observed_peak_indices)
            permutation_medians[permutation] = float(
                np.nanmedian(scores[observed_gene_indices, permuted_peak_indices])
            )
        observed_median = float(np.nanmedian(scores[observed_gene_indices, observed_peak_indices]))
        permutation_p = float(
            (1.0 + np.sum(permutation_medians >= observed_median))
            / (1.0 + N_PAIRING_PERMUTATIONS)
        )
        permutation_rows.extend(
            {
                "lineage": lineage,
                "permutation": permutation + 1,
                "permuted_median_S_max": value,
            }
            for permutation, value in enumerate(permutation_medians)
        )
        summaries.append(
            {
                "lineage": lineage,
                "n_observed_pairs": len(observed_values),
                "n_mismatched_pairs": len(null_values),
                "observed_median_S_max": float(np.median(observed_values)),
                "mismatched_median_S_max": float(np.median(null_values)),
                "common_language_AUC": float(u_stat / (len(observed_values) * len(null_values))),
                "mannwhitney_one_sided_p": float(p_value),
                "pairing_permutations": N_PAIRING_PERMUTATIONS,
                "pairing_permutation_seed": PERMUTATION_SEED + lineage_index,
                "permuted_median_S_max": float(np.median(permutation_medians)),
                "permuted_median_S_max_q025": float(np.quantile(permutation_medians, 0.025)),
                "permuted_median_S_max_q975": float(np.quantile(permutation_medians, 0.975)),
                "median_pairing_permutation_p": permutation_p,
                "observed_fraction_above_0_5": float(np.mean(observed_values > 0.5)),
                "mismatched_fraction_above_0_5": float(np.mean(null_values > 0.5)),
                "observed_fraction_above_0_8": float(np.mean(observed_values > 0.8)),
                "mismatched_fraction_above_0_8": float(np.mean(null_values > 0.8)),
                "n_empirical_BH_q_below_0_05": int(np.sum(observed["empirical_BH_q"] < 0.05)),
                "n_empirical_BH_q_below_0_10": int(np.sum(observed["empirical_BH_q"] < 0.10)),
                "maximum_reproduction_error": maximum_error,
            }
        )
        all_rows.append(local_scores)
        observed_rows.append(observed)

    return (
        pd.concat(all_rows, ignore_index=True),
        pd.concat(observed_rows, ignore_index=True),
        pd.DataFrame(summaries),
        pd.DataFrame(permutation_rows),
    )


def time_matched_e85_audit(curves: pd.DataFrame, audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    e85 = curves.loc[np.isclose(curves["normalized_time"], E85_NORMALIZED_TIME)].copy()
    peak = e85.pivot(index="candidate_id", columns="lineage", values="atac_log1p_cp10k_smoothed")
    gene = e85.pivot(index="candidate_id", columns="lineage", values="rna_log1p_cpm_smoothed")
    result = audit.drop_duplicates("candidate_id").set_index("candidate_id").copy()
    result["COATI_E85_peak_dominant_compartment"] = peak.idxmax(axis=1)
    result["COATI_E85_gene_dominant_compartment"] = gene.idxmax(axis=1)
    result = result.rename(
        columns={
            "Koch_WT_peak_region": "Argelaguet_E85_control_peak_dominant_compartment",
            "Koch_WT_gene_region": "Argelaguet_E85_control_gene_dominant_compartment",
            "Koch_WT_peak_region_log2_margin": "Argelaguet_E85_control_peak_log2_margin",
            "Koch_WT_gene_region_log2_margin": "Argelaguet_E85_control_gene_log2_margin",
        }
    )
    result["E85_peak_compartment_match"] = result[
        "COATI_E85_peak_dominant_compartment"
    ].eq(result["Argelaguet_E85_control_peak_dominant_compartment"])
    result["E85_gene_compartment_match"] = result[
        "COATI_E85_gene_dominant_compartment"
    ].eq(result["Argelaguet_E85_control_gene_dominant_compartment"])
    summary = pd.DataFrame(
        [
            {
                "comparison_time": "E8.5",
                "COATI_normalized_time": E85_NORMALIZED_TIME,
                "n_pairs": len(result),
                "peak_compartment_matches": int(result["E85_peak_compartment_match"].sum()),
                "peak_compartment_match_fraction": float(result["E85_peak_compartment_match"].mean()),
                "gene_compartment_matches": int(result["E85_gene_compartment_match"].sum()),
                "gene_compartment_match_fraction": float(result["E85_gene_compartment_match"].mean()),
            }
        ]
    )
    return result.reset_index(), summary


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = float(np.std(values))
    return np.zeros_like(values) if scale <= 1e-12 else (values - np.mean(values)) / scale


def p_label(value: float, permutations: int | None = None) -> str:
    if permutations is not None and np.isclose(value, 1.0 / (permutations + 1)):
        return rf"$P_{{perm}}<10^{{-{int(np.log10(permutations))}}}$"
    if value < 1e-99:
        return r"$P<10^{-99}$"
    exponent = int(np.floor(np.log10(value)))
    coefficient = value / (10.0**exponent)
    return rf"$P={coefficient:.1f}\times10^{{{exponent}}}$"


def plot_figure(
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    curves: pd.DataFrame,
    audit: pd.DataFrame,
    output_dir: Path,
) -> None:
    configure_plotting()
    figure = plt.figure(figsize=(7.2, 5.05))
    grid = figure.add_gridspec(
        2,
        3,
        left=0.085,
        right=0.985,
        bottom=0.11,
        top=0.84,
        hspace=0.72,
        wspace=0.28,
    )

    figure.suptitle(
        "Lineage-resolved temporal concordance of observed peak–gene pairs",
        x=0.535,
        y=0.965,
        fontsize=10,
        fontweight="normal",
    )
    figure.text(
        0.085,
        0.895,
        "a",
        fontsize=9,
        fontweight="bold",
        va="center",
    )
    figure.text(
        0.112,
        0.895,
        "Observed pair identities versus mismatched-pair null",
        fontsize=8,
        va="center",
    )
    figure.legend(
        handles=[
            Patch(facecolor=NULL_GREY, edgecolor="none", label="Mismatched pair"),
            Patch(facecolor=COATI_BLUE, edgecolor="none", label="Observed pair"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.913),
        frameon=False,
        ncol=2,
        handlelength=1.1,
        columnspacing=1.1,
    )

    for column, lineage in enumerate(LINEAGES):
        axis = figure.add_subplot(grid[0, column])
        local = scores.loc[scores["lineage"].eq(lineage)]
        mismatched = local.loc[local["pairing"].eq("Mismatched pair"), "S_max"].dropna().to_numpy()
        observed = local.loc[local["pairing"].eq("Observed pair"), "S_max"].dropna().to_numpy()
        violin = axis.violinplot(
            [mismatched, observed],
            positions=[0, 1],
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.18,
        )
        for body, color in zip(violin["bodies"], [NULL_GREY, COATI_BLUE]):
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.82)
        medians = [float(np.median(mismatched)), float(np.median(observed))]
        axis.scatter([0, 1], medians, marker="D", s=13, c=[TEXT_GREY, "black"], zorder=4)
        for x_value, median in enumerate(medians):
            axis.text(x_value, median + 0.10, f"{median:.2f}", ha="center", va="bottom", fontsize=6.5)
        info = summary.loc[summary["lineage"].eq(lineage)].iloc[0]
        axis.set_title(lineage, pad=4)
        axis.text(
            0.5,
            0.98,
            f"AUC={info.common_language_AUC:.3f}; "
            f"{p_label(info.median_pairing_permutation_p, int(info.pairing_permutations))}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=6.5,
        )
        axis.set_xlim(-0.6, 1.6)
        axis.set_ylim(-1.05, 1.18)
        axis.set_xticks([0, 1], ["Mismatched\n(n=17,957)", "Observed\n(n=187)"])
        axis.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        axis.axhline(0, color=GRID_GREY, lw=0.6, zorder=0)
        axis.grid(axis="y", color=GRID_GREY, lw=0.45, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylabel(r"Maximum temporal concordance, $S_{\max}$" if column == 0 else "")
        if column > 0:
            axis.tick_params(labelleft=False)

    figure.text(0.085, 0.455, "b", fontsize=9, fontweight="bold", va="center")
    figure.text(
        0.112,
        0.455,
        "Same Fgf3-linked pair across descendants of E7.5 caudal epiblast",
        fontsize=8,
        va="center",
    )
    figure.text(
        0.985,
        0.455,
        "T-KO direction + Brachyury ChIP",
        ha="right",
        va="center",
        fontsize=7,
        color=TEXT_GREY,
    )
    figure.legend(
        handles=[
            Line2D([0], [0], color=COATI_BLUE, lw=1.4, label="RNA"),
            Line2D([0], [0], color=ATAC_ORANGE, lw=1.4, ls="--", label="ATAC"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.535, 0.014),
        frameon=False,
        ncol=2,
        handlelength=1.8,
        columnspacing=1.2,
    )

    candidate_audit = audit.loc[audit["candidate_id"].eq(FGF3_CANDIDATE)]
    for column, lineage in enumerate(LINEAGES):
        axis = figure.add_subplot(grid[1, column])
        local = curves.loc[
            curves["candidate_id"].eq(FGF3_CANDIDATE) & curves["lineage"].eq(lineage)
        ].sort_values("normalized_time")
        rna = zscore(local["rna_log1p_cpm_smoothed"].to_numpy(float))
        atac = zscore(local["atac_log1p_cp10k_smoothed"].to_numpy(float))
        day = local["embryonic_day_display"].to_numpy(float)
        axis.plot(day, rna, color=COATI_BLUE, lw=1.4)
        axis.plot(day, atac, color=ATAC_ORANGE, lw=1.4, ls="--")
        row = candidate_audit.loc[candidate_audit["lineage"].eq(lineage)].iloc[0]
        axis.set_title(
            f"{lineage}\n$S_{{\max}}$={row.S_max:.2f}; {row.timing_label}; rank {int(row.rank_within_lineage)}",
            pad=3,
            fontsize=7.5,
        )
        axis.set_xlim(7.5, 8.75)
        axis.set_ylim(-2.5, 2.5)
        axis.set_xticks([7.5, 8.0, 8.5, 8.75], ["E7.5", "E8.0", "E8.5", "E8.75"])
        axis.set_yticks([-2, -1, 0, 1, 2])
        axis.grid(color=GRID_GREY, lw=0.45, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("Embryonic stage")
        axis.set_ylabel("Trajectory z-score" if column == 0 else "")
        if column > 0:
            axis.tick_params(labelleft=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "coati_temporal_pair_shuffle_validation"
    for suffix in ("pdf", "svg"):
        figure.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    native = pd.read_csv(args.input_dir / "trajectory_native_grid_curves.csv.gz")
    curves = pd.read_csv(args.input_dir / "trajectory_time_curves.csv.gz")
    audit = pd.read_csv(args.input_dir / "external_support_postranking_audit.csv")

    scores, observed, summary, permutation_medians = build_pair_shuffle_scores(native, audit)
    e85_audit, e85_summary = time_matched_e85_audit(curves, audit)
    scores.to_csv(args.output_dir / "observed_and_mismatched_pair_scores.csv.gz", index=False)
    observed.to_csv(args.output_dir / "observed_pair_empirical_significance.csv", index=False)
    summary.to_csv(args.output_dir / "pair_shuffle_lineage_summary.csv", index=False)
    permutation_medians.to_csv(
        args.output_dir / "pairing_permutation_median_scores.csv.gz", index=False
    )
    e85_audit.to_csv(args.output_dir / "e85_time_matched_branch_concordance.csv", index=False)
    e85_summary.to_csv(args.output_dir / "e85_time_matched_branch_concordance_summary.csv", index=False)
    plot_figure(scores, summary, curves, audit, args.output_dir)

    print(summary.to_string(index=False))
    print(e85_summary.to_string(index=False))
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
