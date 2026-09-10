#!/usr/bin/env python3
"""Render the whole-trajectory branch jump rates as a compact comparison table."""

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


HERE = (_REPO / "external/COATI/Synthetic/UnbalancedSyncSweep")
OUT = HERE / "outputs/bio_all1_alpha100_d10_e1/ever_branch_jump"
INPUT = OUT / "ever_branch_jump.json"


def main() -> None:
    payload = json.loads(INPUT.read_text())
    results = payload["results"]
    baseline = {result["model"]: result for result in results if result["cy"] is None}
    balanced = {
        float(result["cy"]): result
        for result in results
        if result["model"] == "COATI bal"
    }
    unbalanced = {
        float(result["cy"]): result
        for result in results
        if result["model"] == "COATI unbal"
    }
    cy_values = [round(value / 10, 1) for value in range(10)]
    balanced_values = [baseline["OT"]["count_jump_rate"]] + [
        balanced[cy]["count_jump_rate"] for cy in cy_values
    ]
    unbalanced_values = [baseline["UOT"]["terminal_mass_weighted_jump_rate"]] + [
        unbalanced[cy]["terminal_mass_weighted_jump_rate"] for cy in cy_values
    ]

    columns = ["Model", "OT / UOT\n(RNA only)"] + [f"Cy={cy:.1f}" for cy in cy_values]
    rows = [
        ["COATI bal"] + [f"{100 * value:.2f}%" for value in balanced_values],
        ["COATI unbal"] + [f"{100 * value:.2f}%" for value in unbalanced_values],
    ]
    with (OUT / "ever_branch_jump_table.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)

    font_manager.findfont("Arial", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Arial", "font.size": 12})
    figure = plt.figure(figsize=(11.0, 2.15), dpi=300)
    figure.patch.set_facecolor("white")
    axis = figure.add_axes([0.01, 0.04, 0.98, 0.76])
    axis.axis("off")
    figure.text(
        0.5,
        0.91,
        "Whole-trajectory branch jump rate",
        ha="center",
        va="center",
        fontsize=12,
        fontfamily="Arial",
        fontweight="normal",
    )
    table = axis.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.12, 0.14] + [0.074] * 10,
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.set_linewidth(0.6)
        cell.PAD = 0.025
        cell.get_text().set_fontfamily("Arial")
        cell.get_text().set_fontsize(12)
        if row_index == 0 or column_index == 0:
            cell.get_text().set_fontweight("bold")

    output = OUT / "ever_branch_jump_table.png"
    figure.savefig(output, dpi=300, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
