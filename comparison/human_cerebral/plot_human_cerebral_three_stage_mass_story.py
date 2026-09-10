from __future__ import annotations

import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results" / "human_cerebral_three_stage_mass_story_v1"

METHODS = ["COATI unbal.", "UOT(RNA)", "CytoBridge unbal."]
COLORS = {
    "COATI unbal.": "#D55E00",
    "UOT(RNA)": "#111111",
    "CytoBridge unbal.": "#CC79A7",
}


ROWS = [
    {
        "stage": "D7→D11",
        "process": "Neural induction with\ncycling-coupled allocation",
        "metric": "S-phase partial ρ, D7→D9",
        "COATI unbal.": "0.468",
        "UOT(RNA)": "0.305",
        "CytoBridge unbal.": "0.169",
        "group": 0,
    },
    {
        "stage": "",
        "process": "",
        "metric": "S-phase partial ρ, D9→D11",
        "COATI unbal.": "0.354",
        "UOT(RNA)": "0.357",
        "CytoBridge unbal.": "−0.021",
        "group": 0,
    },
    {
        "stage": "D12→D18",
        "process": "Shift toward telencephalic\nregionalization",
        "metric": "S-phase partial ρ",
        "COATI unbal.": "−0.083",
        "UOT(RNA)": "−0.070",
        "CytoBridge unbal.": "0.207",
        "group": 1,
    },
    {
        "stage": "",
        "process": "",
        "metric": "Non-S regionalization-gene AUROC",
        "COATI unbal.": "0.730",
        "UOT(RNA)": "0.689",
        "CytoBridge unbal.": "0.672",
        "group": 1,
    },
    {
        "stage": "D18→D21",
        "process": "Regional fate\nconsolidation",
        "metric": "Late-lineage gene AUROC (pooled)",
        "COATI unbal.": "0.678",
        "UOT(RNA)": "0.620",
        "CytoBridge unbal.": "0.537",
        "group": 2,
    },
    {
        "stage": "",
        "process": "",
        "metric": "Ventral | dorsal partial ρ",
        "COATI unbal.": "0.520",
        "UOT(RNA)": "0.089",
        "CytoBridge unbal.": "0.232",
        "group": 2,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the three-stage human-cerebral modeled-mass story."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def put_text(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    ha: str = "left",
    va: str = "center",
    size: float = 10.0,
    weight: str = "normal",
    color: str = "#222222",
) -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        fontweight=weight,
        color=color,
        linespacing=1.18,
        transform=ax.transAxes,
    )


def draw(output_dir: Path) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(13.1, 3.8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_axis_off()

    left, right = 0.018, 0.988
    bottom, top = 0.025, 0.975
    header_h = 0.105
    body_top = top - header_h

    # Stage, biological interpretation, evidence, and three method columns.
    widths = [0.105, 0.245, 0.285, 0.115, 0.105, 0.140]
    scale = (right - left) / sum(widths)
    widths = [w * scale for w in widths]
    xs = [left]
    for width in widths:
        xs.append(xs[-1] + width)

    group_weights = [2.18, 2.18, 2.18]
    total_weight = sum(group_weights)
    group_heights = [(body_top - bottom) * w / total_weight for w in group_weights]
    group_bounds: list[tuple[float, float]] = []
    y_cursor = body_top
    for height in group_heights:
        group_bounds.append((y_cursor - height, y_cursor))
        y_cursor -= height

    # Background bands and outer frame.
    fills = ["#FAFAFA", "#F3F5F6", "#FAFAFA"]
    for group, (y0, y1) in enumerate(group_bounds):
        ax.add_patch(
            Rectangle(
                (left, y0),
                right - left,
                y1 - y0,
                transform=ax.transAxes,
                facecolor=fills[group],
                edgecolor="none",
                zorder=0,
            )
        )

    ax.add_patch(
        Rectangle(
            (left, bottom),
            right - left,
            top - bottom,
            transform=ax.transAxes,
            facecolor="none",
            edgecolor="#4A4A4A",
            linewidth=1.05,
            zorder=5,
        )
    )
    ax.add_patch(
        Rectangle(
            (left, body_top),
            right - left,
            header_h,
            transform=ax.transAxes,
            facecolor="#FFFFFF",
            edgecolor="none",
            zorder=0,
        )
    )

    # Major column and row rules.
    for x in xs[1:-1]:
        ax.plot([x, x], [bottom, top], color="#C6C6C6", lw=0.72, transform=ax.transAxes)
    ax.plot([left, right], [body_top, body_top], color="#555555", lw=1.0, transform=ax.transAxes)
    for group in range(1, len(group_bounds)):
        y = group_bounds[group][1]
        ax.plot([left, right], [y, y], color="#777777", lw=0.95, transform=ax.transAxes)

    # Header.
    headers = ["Stage", "Biological process", "Evidence"]
    for index, label in enumerate(headers):
        put_text(ax, xs[index] + 0.010, body_top + header_h / 2, label, weight="semibold", size=10.6)
    for method_index, method in enumerate(METHODS, start=3):
        center = (xs[method_index] + xs[method_index + 1]) / 2
        ax.scatter(
            center - 0.034,
            body_top + header_h / 2,
            s=50,
            color=COLORS[method],
            edgecolor="none",
            transform=ax.transAxes,
            clip_on=False,
            zorder=6,
        )
        label = method.replace("CytoBridge unbal.", "CytoBridge\nunbal.")
        put_text(
            ax,
            center - 0.024,
            body_top + header_h / 2,
            label,
            size=9.5,
            weight="semibold",
            color=COLORS[method],
        )

    rows_by_group = [[r for r in ROWS if r["group"] == group] for group in range(3)]
    for group, rows in enumerate(rows_by_group):
        y0, y1 = group_bounds[group]
        row_h = (y1 - y0) / len(rows)

        first = rows[0]
        put_text(
            ax,
            xs[0] + 0.010,
            (y0 + y1) / 2,
            str(first["stage"]),
            size=11.2,
            weight="semibold",
        )
        put_text(
            ax,
            xs[1] + 0.013,
            (y0 + y1) / 2,
            str(first["process"]),
            size=10.0,
        )

        for row_index, row in enumerate(rows):
            row_top = y1 - row_index * row_h
            row_bottom = row_top - row_h
            y = (row_top + row_bottom) / 2
            if row_index > 0:
                ax.plot(
                    [xs[2], right],
                    [row_top, row_top],
                    color="#DDDDDD",
                    lw=0.58,
                    transform=ax.transAxes,
                )
            put_text(ax, xs[2] + 0.012, y, str(row["metric"]), size=9.75)
            for method_index, method in enumerate(METHODS, start=3):
                center = (xs[method_index] + xs[method_index + 1]) / 2
                put_text(
                    ax,
                    center,
                    y,
                    str(row[method]),
                    ha="center",
                    size=11.1,
                    color="#222222",
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "three_stage_mass_allocation_story.pdf", bbox_inches="tight")
    fig.savefig(
        output_dir / "three_stage_mass_allocation_story.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    pd.DataFrame(ROWS).drop(columns="group").to_csv(
        output_dir / "three_stage_mass_allocation_values.csv", index=False
    )


def main() -> None:
    args = parse_args()
    targets = [
        args.output_dir / "three_stage_mass_allocation_story.pdf",
        args.output_dir / "three_stage_mass_allocation_story.png",
        args.output_dir / "three_stage_mass_allocation_values.csv",
    ]
    if not args.overwrite and any(path.exists() for path in targets):
        existing = ", ".join(str(path) for path in targets if path.exists())
        raise FileExistsError(f"Refusing to overwrite: {existing}. Pass --overwrite.")
    draw(args.output_dir)


if __name__ == "__main__":
    main()
