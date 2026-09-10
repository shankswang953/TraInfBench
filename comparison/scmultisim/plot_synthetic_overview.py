"""Synthetic data overview: the two observed spaces, the frozen T, and the UMAP.

Six panels in a 2x3 grid:
  row 1 -- observed RNA10 (PC1/PC2), observed selected-ATAC8 (PC1/PC3), and the
           same ATAC cloud with T(RNA) drawn on top.  The residual in the third
           panel is the error floor that any trajectory read through T inherits.
  row 2 -- the shared RNA10 UMAP coloured by snapshot time, the same UMAP
           coloured by population, and the key.

Everything is shown in normalized (W2-scaled) coordinates -- that is the space
the ODE and T actually operate in.  Styling follows
TraInfBench/FIGURE_COLOR_STANDARD.md; colours come from the shared module
rather than being defined here.
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
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

# Force Agg before pyplot: the interactive macOS backend snaps the canvas to
# whole device pixels, which silently shrinks the saved page by a fraction of a
# millimetre and breaks alignment with the comparison strip.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

HERE = (_REPO / "external/COATI/Synthetic")
BENCH = Path("common/trainfbench_plot_style.py")
_spec = importlib.util.spec_from_file_location("trainfbench_plot_style", BENCH)
style = importlib.util.module_from_spec(_spec)
sys.modules["trainfbench_plot_style"] = style
_spec.loader.exec_module(style)

TIME_LABELS = ("time1", "time2", "time3", "time4")
TIME_POINTS = (0.0, 1.0, 2.0, 3.0)
OBSERVED_GREY = "#B3B3B3"
# Column 0/1 of the selected-ATAC8 archive are the original ATAC PC1 and PC3
# (PC2 is deliberately dropped -- see prepare_atac_dim.py --drop-pc2).
ATAC_AXIS_NAMES = ("ATAC PC1", "ATAC PC3")

DEFAULT_UMAP = Path(
    "results/"
    "cytobridge_synthetic_rna10_n128_i3000_comparison/rna10_umap/"
    "observed_rna10_umap.npz"
)


def load_film(checkpoint: Path):
    spec = importlib.util.spec_from_file_location("film", HERE / "TrainT/film_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = module.FiLMMLP(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("meta", {})


def scatter_by_population(axis, coords, populations, order, size, alpha):
    """Plot populations back-to-front so no single label hides the others."""
    for name in order:
        mask = populations == name
        if not mask.any():
            continue
        axis.scatter(
            coords[mask, 0], coords[mask, 1], s=size, alpha=alpha, linewidths=0,
            color=style.synthetic_population_color(name), rasterized=True,
        )


def tidy(axis, title, xlabel, ylabel):
    axis.set_title(title, pad=3)
    axis.set_xlabel(xlabel, labelpad=2)
    axis.set_ylabel(ylabel, labelpad=2)
    axis.set_xticks([])
    axis.set_yticks([])
    for side in ("left", "bottom"):
        axis.spines[side].set_linewidth(0.6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna", default="5scRNA/all_time_scRNA_pca10.npz")
    parser.add_argument("--atac", default="5scRNA/all_time_scATAC_pca8_no_pc2.npz")
    parser.add_argument("--labels", default="5scRNA/all_time_scRNA_label.npz")
    parser.add_argument("--rna_norm", default="5scRNA/primal_norm_params_rna10_w2.pt")
    parser.add_argument(
        "--atac_norm", default="5scRNA/secondary_norm_params_atac8_no_pc2_w2.pt"
    )
    parser.add_argument(
        "--t_checkpoint", default="TrainT/outputs/T_FiLM_rna10_atac8_no_pc2.pt"
    )
    parser.add_argument("--umap", type=Path, default=DEFAULT_UMAP)
    # 207.6 mm aligns with the existing figures and with the comparison strip
    # this one stacks against; 105 mm is the single column.
    parser.add_argument("--width_mm", type=float, default=207.6)
    parser.add_argument("--height_mm", type=float, default=74.0)
    parser.add_argument("--key_left", type=float, default=6.0,
                        help="Left edge of the first key column, in mm.")
    parser.add_argument("--key_column_mm", type=float, default=67.0)
    parser.add_argument("--key_entry_mm", type=float, default=30.0,
                        help="Horizontal step between entries within a section.")
    parser.add_argument("--key_per_row", type=int, default=2)
    parser.add_argument("--key_top", type=float, default=18.0,
                        help="Baseline of the key section titles, in mm.")
    parser.add_argument("--font_size", type=float, default=10.0,
                        help="Ticks, legend, and key text.")
    parser.add_argument("--title_font_size", type=float, default=12.0,
                        help="Panel titles and axis labels.")
    parser.add_argument("--point_size", type=float, default=1.6)
    parser.add_argument("--output", default="outputs/synthetic_overview.png")
    args = parser.parse_args()

    rna = np.load(HERE / args.rna)
    atac = np.load(HERE / args.atac)
    labels = np.load(HERE / args.labels, allow_pickle=True)
    rna_scale = torch.load(HERE / args.rna_norm, map_location="cpu",
                           weights_only=False)["scale"]
    atac_scale = torch.load(HERE / args.atac_norm, map_location="cpu",
                            weights_only=False)["scale"]

    populations = np.concatenate([labels[name] for name in TIME_LABELS]).astype(str)
    counts = [rna[name].shape[0] for name in TIME_LABELS]
    time_index = np.repeat(np.arange(len(TIME_LABELS)), counts)

    rna_all = np.concatenate([rna[name] for name in TIME_LABELS]) / rna_scale
    atac_all = np.concatenate([atac[name] for name in TIME_LABELS]) / atac_scale

    model, meta = load_film(HERE / args.t_checkpoint)
    eval_times = meta.get("time_points", list(TIME_POINTS))
    with torch.no_grad():
        pushed = np.concatenate([
            model(torch.tensor(rna[name] / rna_scale, dtype=torch.float32),
                  float(time_point)).numpy()
            for name, time_point in zip(TIME_LABELS, eval_times)
        ])

    umap_payload = np.load(args.umap, allow_pickle=True)
    umap = umap_payload["observed_umap"]
    # The cached embedding is only reusable if its row order is the same
    # time1..time4 concatenation used everywhere else.
    cached_populations = umap_payload["observed_population"].astype(str)
    if umap.shape[0] != populations.shape[0] or not (
        cached_populations == populations
    ).all():
        raise ValueError(
            f"{args.umap} does not match the time1..time4 concatenation of "
            f"{args.labels}; refit the UMAP instead of reusing this cache."
        )

    # TraInfBench font convention: Arial throughout, all weights normal, titles
    # and axis labels one step up from ticks / legend / key text.  The mathtext
    # keys are restated here on purpose -- without them every $...$ silently
    # renders in DejaVu Sans and the file ships with two typefaces.
    style.apply_nature_rc(font_size=args.font_size)
    plt.rcParams.update({
        "font.weight": "normal",
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "axes.titlesize": args.title_font_size,
        "axes.labelsize": args.title_font_size,
        "xtick.labelsize": args.font_size,
        "ytick.labelsize": args.font_size,
        "legend.fontsize": args.font_size,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    })
    # One strip of five panels with the key as a horizontal band underneath, so
    # this block stacks flush against the method-comparison strip, which has the
    # same width and the same legend-below-panels shape.  The old 2x3 grid left
    # a panel-sized hole beside the key.
    figure, axes = plt.subplots(
        1, 5, figsize=(args.width_mm / 25.4, args.height_mm / 25.4),
    )
    # Fixed-millimetre header and footer bands: as fractions they would swell or
    # collapse whenever the strip height changes.
    figure.subplots_adjust(
        left=0.058, right=0.995,
        top=1.0 - 11.0 / args.height_mm, bottom=26.0 / args.height_mm,
        wspace=0.30,
    )

    order = style.SYNTHETIC_POPULATION_ORDER
    size = args.point_size
    pushforward_colour = style.NATURE_CUD["vermillion"]

    scatter_by_population(axes[0], rna_all, populations, order, size, 0.65)
    tidy(axes[0], "Primary (RNA)", "RNA PC1", "RNA PC2")

    scatter_by_population(axes[1], atac_all, populations, order, size, 0.65)
    tidy(axes[1], "Secondary (ATAC)", *ATAC_AXIS_NAMES)

    axes[2].scatter(atac_all[:, 0], atac_all[:, 1], s=size * 1.3, alpha=0.55,
                    linewidths=0, color=OBSERVED_GREY, rasterized=True)
    axes[2].scatter(pushed[:, 0], pushed[:, 1], s=size, marker="x", alpha=0.35,
                    linewidths=0.25, color=pushforward_colour, rasterized=True)
    tidy(axes[2], "$T$(RNA) vs\nobserved ATAC", *ATAC_AXIS_NAMES)
    # The two ATAC panels must share limits, otherwise the residual in the
    # third panel is not readable against the second.
    axes[2].set_xlim(*axes[1].get_xlim())
    axes[2].set_ylim(*axes[1].get_ylim())

    time_palette = style.time_colors(len(TIME_LABELS))
    for index in range(len(TIME_LABELS)):
        mask = time_index == index
        axes[3].scatter(umap[mask, 0], umap[mask, 1], s=size, alpha=0.65,
                        linewidths=0, color=time_palette[index], rasterized=True)
    tidy(axes[3], "RNA UMAP\nsnapshot time", "UMAP1", "UMAP2")

    scatter_by_population(axes[4], umap, populations, order, size, 0.65)
    tidy(axes[4], "RNA UMAP\npopulation", "UMAP1", "UMAP2")
    axes[4].set_xlim(*axes[3].get_xlim())
    axes[4].set_ylim(*axes[3].get_ylim())

    # Three legends laid out by hand in one band: matplotlib sizes each legend
    # independently, so stacked or side-by-side legend artists drift apart
    # whenever the font size or the number of entries changes.
    sections = [
        ("Population",
         [(name, style.synthetic_population_color(name), "o") for name in order]),
        ("Snapshot time",
         [(f"{name}  ($t$ = {point:.0f})", colour, "o")
          for name, point, colour in zip(TIME_LABELS, eval_times, time_palette)]),
        ("Cross-modal map",
         [("Observed ATAC", OBSERVED_GREY, "o"),
          ("$T$(RNA)", pushforward_colour, "x")]),
    ]
    # Entries run across before they run down, so a four-entry section is two
    # short rows rather than one tall column and the band stays shallow.
    line_mm = 1.55 * args.font_size / 2.835
    marker_size = 0.45 * args.font_size
    for column, (title, entries) in enumerate(sections):
        x0 = args.key_left + column * args.key_column_mm
        figure.text(x0 / args.width_mm, args.key_top / args.height_mm, title,
                    va="center", ha="left")
        for index, (label, colour, marker) in enumerate(entries):
            x = x0 + (index % args.key_per_row) * args.key_entry_mm
            y = args.key_top - (1 + index // args.key_per_row) * line_mm
            figure.add_artist(Line2D(
                [(x + 0.17 * args.font_size) / args.width_mm],
                [y / args.height_mm],
                marker=marker, markersize=marker_size, color=colour,
                linestyle="none", markeredgewidth=0.9,
                transform=figure.transFigure,
            ))
            figure.text((x + 0.46 * args.font_size) / args.width_mm,
                        y / args.height_mm, label, va="center", ha="left")

    out = HERE / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    # PDF first: the raster save snaps the canvas to a whole number of pixels,
    # and a PDF written afterwards inherits that slightly shrunken page, which
    # would leave this strip a fraction of a millimetre narrower than the
    # comparison strip it is meant to stack against.
    figure.savefig(out.with_suffix(".pdf"))
    figure.savefig(out, dpi=400)
    plt.close(figure)
    print(f"saved {out} (+ .pdf)")
    print(f"cells={len(populations)} rna_scale={rna_scale:.6f} "
          f"atac_scale={atac_scale:.6f} T_epoch={meta.get('epoch')}")


if __name__ == "__main__":
    main()
