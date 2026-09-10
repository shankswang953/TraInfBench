#!/usr/bin/env python
"""Ground truth plus five methods, by observed snapshot, on one A4 landscape page.

Column 1 is the observed snapshot itself coloured by lineage -- the reference
any method column is read against.  Columns 2-6 are the predicted clouds from
the same 2,006 time1 initials, coloured by their *source* lineage, over a grey
copy of the matching observed snapshot.

Nothing is recomputed here: coordinates and metrics come from the caches written
by plot_synthetic_coati_cytobridge_reversed_tn_comparison.py.  Styling follows
FIGURE_COLOR_STANDARD.md; colours come from trainfbench_plot_style rather than
being defined locally.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import trainfbench_plot_style as style

ROOT = Path(__file__).resolve().parents[2]
METHODS = (
    "TrajectoryNet reversed",
    "COATI balanced",
    "COATI unbalanced",
    "CytoBridge balanced",
    "CytoBridge unbalanced",
)
# At 10 pt a one-line method name is wider than its own column, so headings are
# wrapped rather than allowed to collide with the neighbouring column.
HEADINGS = {
    "TrajectoryNet reversed": "TrajectoryNet",
    "COATI balanced": "COATI\nbalanced",
    "COATI unbalanced": "COATI\nunbalanced",
    "CytoBridge balanced": "CytoBridge\nbalanced",
    "CytoBridge unbalanced": "CytoBridge\nunbalanced",
}
# Cache key stems differ from the display names.
METHOD_KEYS = {
    "TrajectoryNet reversed": "trajectorynet_reversed_predicted_umap",
    "COATI balanced": "coati_balanced_predicted_umap",
    "COATI unbalanced": "coati_unbalanced_predicted_umap",
    "CytoBridge balanced": "cytobridge_balanced_predicted_umap",
    "CytoBridge unbalanced": "cytobridge_unbalanced_predicted_umap",
}
ALL_TIME_KEYS = ("time2", "time3", "time4")
# Cache keys stay as they are; only the row label changes.
TIME_LABELS = {"time2": "time2", "time3": "time3", "time4": "terminal"}
# `4_5` splits into `5_2`/`5_3`, so the observed panel is coloured by lineage of
# origin, matching how the method panels colour by source lineage.
LINEAGE_OF = {"4_1": "4_1", "4_5": "4_5", "5_2": "4_5", "5_3": "4_5"}
OBSERVED_GREY = "#B3B3B3"

# A4 landscape is 297 x 210 mm; leave a printable margin on every side.
A4_LANDSCAPE_MM = (297.0, 210.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/synthetic_rna10_trajectorynet_reversed.npz",
    )
    parser.add_argument(
        "--coordinates",
        type=Path,
        default=(
            ROOT
            / "results/synthetic_rna10_coati_cytobridge_trajectorynet_reversed_method_columns/"
            "comparison_coordinates.npz"
        ),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=(
            ROOT
            / "results/synthetic_rna10_coati_cytobridge_trajectorynet_reversed_method_columns/"
            "metrics.csv"
        ),
    )
    parser.add_argument(
        "--times", nargs="+", default=["time4"], choices=list(ALL_TIME_KEYS),
        help="Observed snapshots to show, one row each.",
    )
    # 207.6 mm aligns with the existing figures; 105 mm is the single column.
    parser.add_argument("--width_mm", type=float, default=207.6)
    parser.add_argument("--height_mm", type=float, default=76.0)
    parser.add_argument("--font_size", type=float, default=10.0,
                        help="Ticks, legend, and in-panel annotation.")
    parser.add_argument("--title_font_size", type=float, default=12.0,
                        help="Panel titles and axis labels.")
    parser.add_argument("--metric_font_size", type=float, default=10.0)
    parser.add_argument("--no-axis-labels", dest="axis_labels",
                        action="store_false",
                        help="Drop the shared UMAP1/UMAP2 labels and reclaim "
                             "the margins they occupied.")
    parser.add_argument("--point_size", type=float, default=1.6)
    parser.add_argument(
        "--clip_percentile", type=float, default=0.2,
        help="Per-row display limits span this to (100 - this) percentile. "
             "Use 0.0 to show every point at the cost of large empty margins.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results/synthetic_rna10_coati_cytobridge_trajectorynet_reversed_method_columns/"
            "truth_and_method_columns_a4.png"
        ),
    )
    args = parser.parse_args()

    for path in (args.dataset, args.coordinates, args.metrics):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.width_mm > A4_LANDSCAPE_MM[0] or args.height_mm > A4_LANDSCAPE_MM[1]:
        raise ValueError(
            f"{args.width_mm} x {args.height_mm} mm does not fit A4 landscape "
            f"({A4_LANDSCAPE_MM[0]} x {A4_LANDSCAPE_MM[1]} mm)"
        )

    data = np.load(args.dataset, allow_pickle=True)
    coordinates = np.load(args.coordinates, allow_pickle=False)
    metrics = pd.read_csv(args.metrics)

    rank = np.asarray(data["original_sample_labels"], dtype=np.int64)
    population = np.asarray(data["population"]).astype(str)
    source = np.asarray(coordinates["initial_source"]).astype(str)

    # Keep the canonical order regardless of the order the flag was given in.
    time_keys = tuple(key for key in ALL_TIME_KEYS if key in set(args.times))
    # `rank` counts observed snapshots from time1 = 0, so time2 is rank 1.
    rank_of = {key: index for index, key in enumerate(ALL_TIME_KEYS, start=1)}

    observed_umap = [coordinates[f"observed_{time}"] for time in time_keys]
    observed_lineage = [
        np.asarray([LINEAGE_OF[name] for name in population[rank == rank_of[time]]])
        for time in time_keys
    ]
    for coords, lineage in zip(observed_umap, observed_lineage):
        if len(coords) != len(lineage):
            raise ValueError(
                "Observed UMAP rows do not match the observed labels of the same "
                "snapshot; the two caches were built from different orderings."
            )

    # The cached prediction arrays are always stacked over all three observed
    # times, so subset them to the requested rows before anything indexes by row.
    selected_rows = [rank_of[time] - 1 for time in time_keys]
    predicted_umap = {
        method: np.asarray(
            coordinates[METHOD_KEYS[method]], dtype=np.float32
        )[selected_rows]
        for method in METHODS
    }

    # One fixed random draw order shared by every panel: plotting 4_1 then 4_5
    # would let whichever lineage is drawn last dominate the overlap regions.
    permutation = np.random.default_rng(0).permutation(len(source))
    source_colors = np.asarray(
        [style.synthetic_population_color(name) for name in source]
    )

    # TraInfBench font convention: Arial throughout, all weights normal, titles
    # and axis labels one step up from ticks / legend / in-panel annotation.
    # The mathtext keys are not redundant with apply_nature_rc -- they are
    # restated here because without them every $...$ silently renders in
    # DejaVu Sans and the file ships with two typefaces.
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
    figure, axes = plt.subplots(
        len(time_keys), len(METHODS) + 1,
        figsize=(args.width_mm / 25.4, args.height_mm / 25.4),
        squeeze=False,
    )
    # Explicit margins instead of bbox_inches="tight": the shared axis labels
    # and the legend live in reserved bands, and the saved page is then exactly
    # width_mm x height_mm rather than whatever the tight bbox happens to be.
    # The header and footer bands are fixed in millimetres so they do not swell
    # when fewer rows make the page shorter.
    # Dropping the shared axis labels frees a band under the panels and a strip
    # beside them, so the margins shrink with them rather than leaving a hole.
    header_mm = 23.0
    footer_mm = 21.0 if args.axis_labels else 13.0
    figure.subplots_adjust(
        left=0.098 if args.axis_labels else 0.062, right=0.985,
        top=1.0 - header_mm / args.height_mm, bottom=footer_mm / args.height_mm,
        wspace=0.08, hspace=0.46,
    )

    # A handful of TrajectoryNet outliers otherwise stretch every panel and
    # leave most of the canvas empty (time4 gains ~16 dead UMAP units at the
    # top).  Limits are therefore robust per row, and any predicted point left
    # outside the view is counted and reported rather than silently dropped.
    # This is display only: W2 and branch correctness are computed in 10D.
    def robust_limits(values: np.ndarray) -> tuple[float, float]:
        low, high = np.percentile(values, [args.clip_percentile,
                                           100.0 - args.clip_percentile])
        pad = 0.04 * (high - low)
        return float(low - pad), float(high + pad)

    row_stacks = [
        np.concatenate(
            [observed_umap[row]] + [predicted_umap[m][row] for m in METHODS]
        )
        for row in range(len(time_keys))
    ]
    x_limits = robust_limits(np.concatenate(row_stacks)[:, 0])
    y_limits = [robust_limits(stack[:, 1]) for stack in row_stacks]

    size = args.point_size
    for row, time_key in enumerate(time_keys):
        truth_axis = axes[row, 0]
        for lineage in ("4_1", "4_5"):
            mask = observed_lineage[row] == lineage
            truth_axis.scatter(
                observed_umap[row][mask, 0], observed_umap[row][mask, 1],
                s=size, alpha=0.55, linewidths=0, rasterized=True,
                color=style.synthetic_population_color(lineage),
            )
        truth_axis.set_ylabel(TIME_LABELS[time_key], labelpad=3)

        for column, method in enumerate(METHODS, start=1):
            axis = axes[row, column]
            axis.scatter(
                observed_umap[row][:, 0], observed_umap[row][:, 1],
                s=size, alpha=0.16, linewidths=0, color=OBSERVED_GREY,
                rasterized=True,
            )
            axis.scatter(
                predicted_umap[method][row, permutation, 0],
                predicted_umap[method][row, permutation, 1],
                s=size, alpha=0.40, linewidths=0,
                c=source_colors[permutation], rasterized=True,
            )
            record = metrics[
                (metrics["method"] == method) & (metrics["time"] == time_key)
            ]
            if len(record) != 1:
                raise ValueError(f"Expected one metric row for {method} / {time_key}")
            record = record.iloc[0]
            # Metrics go above the panel, not inside it: the cloud sits low at
            # time4 and high at time2, so no in-panel corner is free in every row.
            # A small pad plus a generous hspace keeps each block visibly
            # attached to the panel it describes, which is the one below it.
            axis.set_title(
                rf"$W_2$ = {record['rna_w2']:.3f}" + "\n"
                + f"Correct = {record['branch_correct_rate']:.1%}",
                pad=2, linespacing=1.3, fontsize=args.metric_font_size,
            )

        for column in range(len(METHODS) + 1):
            axis = axes[row, column]
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits[row])
            axis.tick_params(length=2.0, width=0.6)
            if row != len(time_keys) - 1:
                axis.set_xticklabels([])
            if column:
                axis.set_yticklabels([])

    # Column headings are annotated rather than folded into the title so that
    # they can take the title size while the metrics below them stay at the
    # annotation size.  annotate() inherits font.size, not axes.titlesize, so
    # the title size has to be passed explicitly.
    # The offset clears a two-line metric block, so all headings share a
    # baseline even though the truth column carries no metrics.
    heading_offset = 2.6 * args.metric_font_size
    for column, method in enumerate(("Observed (truth)",) + METHODS):
        axes[0, column].annotate(
            HEADINGS.get(method, method),
            xy=(0.5, 1.0), xycoords="axes fraction",
            xytext=(0, heading_offset), textcoords="offset points",
            ha="center", va="bottom", linespacing=1.2,
            fontsize=args.title_font_size,
        )

    outside = 0
    for row in range(len(time_keys)):
        for method in METHODS:
            coords = predicted_umap[method][row]
            outside += int(
                (
                    (coords[:, 0] < x_limits[0]) | (coords[:, 0] > x_limits[1])
                    | (coords[:, 1] < y_limits[row][0])
                    | (coords[:, 1] > y_limits[row][1])
                ).sum()
            )
    total = len(source) * len(METHODS) * len(time_keys)

    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=3.2,
               color=style.synthetic_population_color("4_1"),
               label="4_1 lineage / source"),
        Line2D([], [], linestyle="none", marker="o", markersize=3.2,
               color=style.synthetic_population_color("4_5"),
               label="4_5 lineage / source"),
        Line2D([], [], linestyle="none", marker="o", markersize=3.2,
               color=OBSERVED_GREY, label="Observed target snapshot"),
    ]
    if args.axis_labels:
        # One shared pair of axis labels: repeating them under all six columns
        # and beside all three rows costs a band of canvas and says nothing
        # extra.  Both sit in the fixed-millimetre footer band, so they are
        # placed in millimetres too -- as height fractions they collide on a
        # short page.  figure.text rather than supxlabel/supylabel: those
        # helpers treat the given position as a hint and nudge it, which on a
        # short page pushes the x label back up into the tick row.  They are
        # axis labels, so they take axes.labelsize; figure.text would otherwise
        # inherit the smaller font.size and undercut the shared spec.
        figure.text(0.5, 12.0 / args.height_mm, "UMAP1", ha="center",
                    va="center", fontsize=args.title_font_size)
        # Centred on the axes band, not on the page: with three rows the header
        # and footer are no longer a large fraction of the height and a
        # page-centred label drifts away from the panels.
        figure.text(0.013,
                    0.5 * (footer_mm + args.height_mm - header_mm) / args.height_mm,
                    "UMAP2", ha="center", va="center", rotation=90,
                    fontsize=args.title_font_size)
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                  bbox_to_anchor=(0.5, 0.6 / args.height_mm),
                  handletextpad=0.25, columnspacing=1.1, borderpad=0.0,
                  borderaxespad=0.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=400, facecolor="white")
    figure.savefig(args.output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    print(f"saved {args.output} (+ .pdf)")
    print(f"panels={len(time_keys)}x{len(METHODS) + 1} "
          f"predicted_cells={len(source)} "
          f"canvas={args.width_mm:.0f}x{args.height_mm:.0f} mm")
    print(f"predicted points outside the plotted limits: {outside}/{total} "
          f"({outside / total:.2%}) at clip_percentile={args.clip_percentile}")


if __name__ == "__main__":
    main()
