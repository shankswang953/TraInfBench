#!/usr/bin/env python3
"""Replot the existing programme-mass coefficients for Cy=0.5 only."""
from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results/human_cerebral_full_biological_interpretability/03_mass_growth/growth_module_associations_by_setting.csv.gz"
OLD = INPUT.with_name("growth_gene_summary_across_cy.csv.gz")
METHOD_MANIFEST = INPUT.with_name("growth_analysis_manifest.json")
STYLE_SOURCE = ROOT / "comparison/human_cerebral/plot_human_cerebral_growth_publication_redraw_10pt.py"
ANALYSIS_SOURCE = ROOT / "comparison/human_cerebral/analyze_human_cerebral_growth_genes_7time.py"
FONT = Path(__import__('matplotlib.font_manager', fromlist=['findfont']).findfont('Arial'))
OUT = ROOT / "results/human_cerebral_programme_mass_heatmap_cy0p5_v1"
STEM = "growth_programme_partial_spearman_heatmap_cy0p5"
MODULES = ("Apoptosis/stress", "Dorsal", "Early neural ectoderm", "Excitatory maturation", "G2/M", "IPC/neurogenesis", "Inhibitory maturation", "Non-telencephalic", "Proliferation", "Radial glia/progenitor", "S phase", "Telencephalic", "Transient pre-DV", "Ventral")
INTERVALS = ("D4→D7", "D7→D9", "D9→D11", "D11→D12", "D12→D18", "D18→D21")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    data = pd.read_csv(INPUT)
    mask = (data.model_id.eq("coati_sync_unbalanced_cy0.5") & data.method.eq("COATI-U Sync") & data.c_y.eq(0.5) & data.line.eq("ALL") & data.in_stay_go.eq("all") & data.outcome.eq("realized") & data.target_fate.isna())
    chosen = data.loc[mask].copy()
    assert len(chosen) == 84 and not chosen.duplicated(["module", "interval"]).any()
    assert chosen.n_particles.eq(501).all() and chosen.c_y.nunique() == 1
    assert set(chosen.module) == set(MODULES) and set(chosen.interval) == set(INTERVALS)
    chosen.insert(0, "source_csv_row_1based_including_header", chosen.index + 2)
    chosen.insert(0, "column_order", chosen.interval.map({x: i + 1 for i, x in enumerate(INTERVALS)}))
    chosen.insert(0, "row_order", chosen.module.map({x: i + 1 for i, x in enumerate(MODULES)}))
    chosen = chosen.sort_values(["row_order", "column_order"]).reset_index(drop=True)
    values = chosen.pivot(index="module", columns="interval", values="partial_spearman").reindex(index=MODULES, columns=INTERVALS)
    assert np.isfinite(values.to_numpy()).all()
    old = pd.read_csv(OLD)
    old = old.loc[old.outcome.eq("realized"), ["module", "interval", "median_partial_spearman"]]
    comparison = chosen[["module", "interval", "partial_spearman"]].merge(old, on=["module", "interval"], validate="one_to_one")
    comparison["single_cy_minus_old_median"] = comparison.partial_spearman - comparison.median_partial_spearman
    low = chosen.partial_spearman.lt(-0.4)
    high = chosen.partial_spearman.gt(0.4)
    extend = "both" if low.any() and high.any() else "min" if low.any() else "max" if high.any() else "neither"
    inputs = {str(p): sha(p) for p in (INPUT, OLD, METHOD_MANIFEST, STYLE_SOURCE, ANALYSIS_SOURCE, FONT, Path(__file__).resolve())}
    output.mkdir(parents=True)
    chosen.to_csv(output / "plotted_values.csv", index=False, float_format="%.17g")
    comparison.to_csv(output / "comparison_to_previous_across_cy_median.csv", index=False, float_format="%.17g")
    font_manager.fontManager.addfont(str(FONT))
    mpl.rcParams.update({"font.family": "Arial", "font.size": 10, "font.weight": "normal", "axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    assert Path(font_manager.findfont("Arial")).resolve() == FONT.resolve()
    fig = plt.figure(figsize=(4.05, 4.35), facecolor="white")
    axis = fig.add_axes((0.43, 0.15, 0.54, 0.70))
    image = axis.imshow(values.to_numpy(float), aspect="auto", cmap="RdBu_r", vmin=-0.4, vmax=0.4, interpolation="nearest")
    axis.set_xticks(np.arange(6), INTERVALS, rotation=42, ha="right")
    axis.set_yticks(np.arange(14), MODULES)
    axis.tick_params(length=2.5, width=0.65, pad=2)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#666666")
        spine.set_linewidth(0.75)
    caxis = fig.add_axes((0.55, 0.90, 0.40, 0.022))
    colorbar = fig.colorbar(image, cax=caxis, orientation="horizontal", extend=extend)
    colorbar.set_ticks((-0.4, -0.2, 0.0, 0.2, 0.4))
    colorbar.outline.set_linewidth(0.75)
    colorbar.outline.set_edgecolor("#666666")
    fig.text(0.75, 0.985, "Partial Spearman", ha="center", va="top")
    # Pin every text artist to the actual Arial face, not a substitute family.
    from matplotlib.text import Text
    for artist in fig.findobj(Text):
        artist.set_fontproperties(font_manager.FontProperties(fname=str(FONT), size=10))
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(output / f"{STEM}.{suffix}", dpi=600 if suffix == "png" else None, facecolor="white", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    assert all(sha(Path(p)) == h for p, h in inputs.items())
    command = f"MPLCONFIGDIR=/tmp/matplotlib-cache python {Path(__file__).resolve()} --output-dir /path/to/new/output_directory"
    manifest = {
        "status": "rendered_pending_visual_review",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "Replace only the across-Cy median with a direct Cy=0.5 selection; no new scientific analysis",
        "selection": {"model_id": "coati_sync_unbalanced_cy0.5", "method": "COATI-U Sync", "c_y": 0.5, "line": "ALL", "in_stay_go": "all", "outcome": "realized", "target_fate": None, "aggregation_across_Cy": "none", "rows": 84, "unique_module_interval_cells": 84, "particles_per_cell": 501},
        "model": {"balance_tier": "unbalanced", "seed": 0, "iteration": 40000, "Cy": 0.5, "provenance": "Existing HC07 per-setting coefficients; seed and iteration independently checked against canonical model registry by coordinator"},
        "unchanged_statistic": {"outcome": "log_mass(t1) - log_mass(t0)", "module_readout": "Mean of source-stage mapped gene expression z-scores across module genes", "mapping": "stage-conditioned inverse-distance kNN, k=15, normalized RNA PCA30", "partial_correlation": "Pearson correlation of residualized ranks (rank outcomes and module scores, QR residualization against original covariate design)", "covariates": ["intercept", "D4 source-line fixed effects", "mapped log RNA library size", "mapped S score", "mapped G2M score"], "source_aggregation": "Pooled ALL 501 D4-origin particles, not mean across four lines"},
        "display": {"row_order": list(MODULES), "column_order": list(INTERVALS), "cmap": "RdBu_r", "vmin": -0.4, "vmax": 0.4, "data_min": float(chosen.partial_spearman.min()), "data_max": float(chosen.partial_spearman.max()), "below_vmin_count": int(low.sum()), "above_vmax_count": int(high.sum()), "colorbar_extend": extend, "saturated_cells_unclipped_in_CSV": chosen.loc[low | high, ["module", "interval", "partial_spearman"]].to_dict("records"), "font": "Arial", "font_file": str(FONT), "all_text_pt": 10, "configured_size_inches_before_tight_crop": [4.05, 4.35], "figure_title": None},
        "comparison": {"old_statistic": "median_partial_spearman across Cy settings", "max_absolute_change": float(comparison.single_cy_minus_old_median.abs().max())},
        "limitations": ["Full-train model-associated allocation, not measured cell proliferation, causal growth genes, or held-out validation", "S-phase and G2/M rows condition on their related cell-cycle scores; interpret as residual conditional associations, not total cycling-growth coupling", "Pooled particle correlation is not equal-source-line aggregation and particles are not independent biological replicates", "Only Cy selection changed; covariates, modules, intervals, statistic, and colour limits remain unchanged"],
        "input_sha256": inputs,
        "input_hashes_unchanged_after_plotting": True,
        "reproduce_command": command,
    }
    (output / "render_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    readme = f"""# Programme-mass heatmap: Cy=0.5 only

![Cy=0.5 programme-mass heatmap]({STEM}.png)

The new panel uses COATI-U Sync, **Cy=0.5, seed=0, 40,000 iterations**. All 84 cells are copied directly from the existing per-setting coefficient table (`model_id=coati_sync_unbalanced_cy0.5`, `method=COATI-U Sync`, `c_y=0.5`, `line=ALL`, `in_stay_go=all`, `outcome=realized`, and no target-fate restriction). There is **no averaging or median across Cy settings** and no coefficient recomputation.

## Unchanged method and interpretation

Each cell is a partial rank correlation between source-stage programme expression and subsequent interval log-mass change, pooled over the same 501 D4-origin particles. Expression is read out with stage-conditioned, inverse-distance kNN (k=15) in normalized RNA PCA30. Gene expression is standardized across particles before equal-gene module averaging. Module and mass ranks are residualized against source-line fixed effects, mapped log RNA library size, mapped S score, and mapped G2M score (plus intercept), and residuals are Pearson-correlated. This is **pooled particle correlation, not four-source equal weighting**.

Only the Cy selector changed. The 14 programmes, six intervals, adjustment, statistic, row/column ordering, red-blue colour scale, Arial 10 pt styling, and compact layout are retained from the old panel. The colour scale remains -0.4 to 0.4: {int(low.sum())} value is below and {int(high.sum())} values above those limits. Both-end colourbar triangles explicitly mark saturation. All original coefficients are retained unclipped in `plotted_values.csv`; their range is {chosen.partial_spearman.min():.9f} to {chosen.partial_spearman.max():.9f}.

This is full-train **model-associated allocation**, not observed proliferation, a causal growth-gene assay, or held-out prediction. S-phase and G2/M rows are conditional on related cell-cycle covariates and must not be interpreted as total cycling-growth coupling. Correlations across particles are not independent biological-replicate evidence. The original analysis and all previous outputs are unchanged.

## Files and reproduction

- `{STEM}.pdf`, `.png`, `.svg`: publication panel; no extra title.
- `plotted_values.csv`: 84 exact selected rows, including source-table row and original fields.
- `comparison_to_previous_across_cy_median.csv`: cellwise audit of the selector change, not a new analysis.
- `analysis_manifest.json`: final provenance, checks, and hashes; `render_manifest.json` preserves initial rendering-stage metadata.
- `qa/`: actual PDF rendering and font/layout checks.

Input: `{INPUT}`. Original method manifest: `{METHOD_MANIFEST}`. Original style: `{STYLE_SOURCE}`.

Run into a new non-existing directory (the script refuses overwrites):

```sh
{command}
```

## Suggested caption

Programme-mass partial rank correlations for COATI-U (Cy=0.5, seed 0, 40,000 iterations), calculated from 501 D4-origin particles at each source-stage interval and adjusted for source line, RNA library size, S score and G2M score. Values are single-setting coefficients, not medians across Cy. Colourbar extensions indicate values outside the displayed ±0.4 range. These are full-train allocation associations; cell-cycle rows reflect conditional associations after adjustment for related scores.
"""
    (output / "README.md").write_text(readme)
    print(json.dumps({"output_dir": str(output), "pdf": str(output / f"{STEM}.pdf"), "rows": len(chosen), "min": float(chosen.partial_spearman.min()), "max": float(chosen.partial_spearman.max()), "extend": extend, "max_change_vs_old": manifest["comparison"]["max_absolute_change"]}, indent=2))


if __name__ == "__main__":
    main()
