#!/usr/bin/env python3
"""Calculate and tabulate fixed40 correctness across all observed follow-up times."""

from __future__ import annotations

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
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier


HERE = (_REPO / "external/COATI/Synthetic/UnbalancedSyncSweep")
SYNTHETIC = HERE.parent
OUT = HERE / "outputs/bio_all1_alpha100_d10_e1/observed_time_correctness"
DATA = SYNTHETIC / "5scRNA/all_time_scRNA_pca10.npz"
LABELS = SYNTHETIC / "5scRNA/all_time_scRNA_label.npz"
NORM = SYNTHETIC / "5scRNA/primal_norm_params_rna10_w2.pt"
BALANCED_BASELINE = SYNTHETIC / "BalancedRNAOnly/runs/rna10_n128_i3000_d10_e1_w1000"
UNBALANCED_BASELINE = (
    SYNTHETIC
    / "UnbalancedRNAOnly/runs/"
    "rna10_n128_i3000_d10_e1_w1000_alpha_growth100_bio_all1_dlogw_g"
)
ALLOWED = {"4_1": {"4_1"}, "4_5": {"4_5", "5_2", "5_3"}}


def cy_tag(cy: float) -> str:
    return f"{cy:.1f}".replace(".", "p")


def balanced_run(cy: float) -> Path:
    name = "density_10_energy_1" if cy == 0.5 else f"density_10_energy_1_cy_{cy_tag(cy)}"
    return SYNTHETIC / "BalancedSyncSweep/outputs/runs" / name


def unbalanced_run(cy: float) -> Path:
    return (
        HERE
        / "outputs/runs"
        / f"density_10_energy_1_cy_{cy_tag(cy)}_alpha_growth_100_bio_all1_dlogw_g"
    )


def evaluate_run(
    run: Path,
    classifiers: dict[int, KNeighborsClassifier],
    scale: float,
    weighted: bool,
    expected_indices: np.ndarray | None,
) -> tuple[dict, np.ndarray]:
    trajectory_dir = run / "trajectory/rna10_umap"
    trajectory = (
        torch.load(
            trajectory_dir / "primary_trajectory_40.pt",
            map_location="cpu",
            weights_only=False,
        ).numpy()
        * scale
    )
    archive = np.load(trajectory_dir / "trajectory_umap.npz")
    source = archive["initial_population"].astype(str)
    indices = archive["fixed_initial_indices"]
    if trajectory.shape != (61, 40, 10):
        raise ValueError(f"Unexpected trajectory shape for {run}: {trajectory.shape}")
    if {label: int(np.sum(source == label)) for label in np.unique(source)} != {
        "4_1": 20,
        "4_5": 20,
    }:
        raise ValueError(f"Expected 20 cells from each source population: {run}")
    if expected_indices is not None and not np.array_equal(indices, expected_indices):
        raise ValueError(f"The fixed40 selection differs for {run}")
    log_weights = archive["log_weights"][:, :, 0] if weighted else None

    per_time = {}
    count_per_time = {}
    for observed_time, step in zip((2, 3, 4), (20, 40, 60)):
        predicted = classifiers[observed_time].predict(trajectory[step]).astype(str)
        correct = np.asarray(
            [prediction in ALLOWED[initial] for initial, prediction in zip(source, predicted)]
        )
        count_per_time[f"time{observed_time}"] = float(correct.mean())
        if weighted:
            weights = np.exp(log_weights[step].astype(np.float64))
            rate = float(weights[correct].sum() / (weights.sum() + 1e-12))
        else:
            rate = float(correct.mean())
        per_time[f"time{observed_time}"] = rate
    return (
        {
            "run": str(run.resolve()),
            "weighted": weighted,
            "per_time": per_time,
            "mean_over_time2_time3_time4": float(np.mean(list(per_time.values()))),
            "unweighted_per_time": count_per_time,
            "unweighted_mean": float(np.mean(list(count_per_time.values()))),
        },
        indices,
    )


def save_csv(payload: dict, path: Path) -> None:
    cy_values = payload["cy_values"]
    header = ["model", "OT/UOT (RNA only)"] + [f"Cy={cy:.1f}" for cy in cy_values]
    rows = [
        ["COATI bal", payload["baseline"]["balanced"]["mean_over_time2_time3_time4"]]
        + [payload["balanced"][f"{cy:.1f}"]["mean_over_time2_time3_time4"] for cy in cy_values],
        ["COATI unbal", payload["baseline"]["unbalanced"]["mean_over_time2_time3_time4"]]
        + [payload["unbalanced"][f"{cy:.1f}"]["mean_over_time2_time3_time4"] for cy in cy_values],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def draw_table(payload: dict, path: Path) -> None:
    font_manager.findfont("Arial", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Arial", "font.size": 12})
    cy_values = payload["cy_values"]
    columns = ["Model", "OT / UOT\n(RNA only)"] + [f"Cy={cy:.1f}" for cy in cy_values]
    balanced = [payload["baseline"]["balanced"]["mean_over_time2_time3_time4"]] + [
        payload["balanced"][f"{cy:.1f}"]["mean_over_time2_time3_time4"] for cy in cy_values
    ]
    unbalanced = [payload["baseline"]["unbalanced"]["mean_over_time2_time3_time4"]] + [
        payload["unbalanced"][f"{cy:.1f}"]["mean_over_time2_time3_time4"] for cy in cy_values
    ]
    cell_text = [
        ["COATI bal"] + [f"{100 * value:.2f}%" for value in balanced],
        ["COATI unbal"] + [f"{100 * value:.2f}%" for value in unbalanced],
    ]

    fig = plt.figure(figsize=(8.27, 2.15), dpi=300)
    fig.patch.set_facecolor("white")
    axis = fig.add_axes([0.01, 0.04, 0.98, 0.76])
    axis.axis("off")
    fig.text(
        0.5,
        0.91,
        "Mean correct rate across observed times",
        ha="center",
        va="center",
        fontsize=12,
        fontfamily="Arial",
    )
    table = axis.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.13, 0.15] + [0.08] * 9,
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
    fig.savefig(path, dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(DATA)
    labels = np.load(LABELS, allow_pickle=True)
    classifiers = {
        time_index: KNeighborsClassifier(n_neighbors=15, weights="distance").fit(
            np.asarray(data[f"time{time_index}"], dtype=np.float32),
            labels[f"time{time_index}"].astype(str),
        )
        for time_index in (2, 3, 4)
    }
    scale = float(torch.load(NORM, map_location="cpu", weights_only=False)["scale"])
    cy_values = [round(i / 10, 1) for i in range(1, 10)]
    payload = {
        "definition": {
            "observed_followup_times": ["time2 (t=1)", "time3 (t=2)", "time4 (t=3)"],
            "classifier": "distance-weighted 15-NN against the corresponding observed snapshot",
            "allowed_populations": {key: sorted(value) for key, value in ALLOWED.items()},
            "fixed_selection": "40 cells total: 20 initial 4_1 and 20 initial 4_5",
            "balanced_aggregation": "unweighted correctness at each time, then mean across three times",
            "unbalanced_aggregation": "particle-weighted correctness at each time, then mean across three times",
        },
        "cy_values": cy_values,
        "baseline": {},
        "balanced": {},
        "unbalanced": {},
    }
    balanced_baseline, fixed = evaluate_run(
        BALANCED_BASELINE, classifiers, scale, weighted=False, expected_indices=None
    )
    unbalanced_baseline, _ = evaluate_run(
        UNBALANCED_BASELINE, classifiers, scale, weighted=True, expected_indices=fixed
    )
    payload["baseline"] = {
        "balanced": balanced_baseline,
        "unbalanced": unbalanced_baseline,
    }
    for cy in cy_values:
        balanced, _ = evaluate_run(
            balanced_run(cy), classifiers, scale, weighted=False, expected_indices=fixed
        )
        unbalanced, _ = evaluate_run(
            unbalanced_run(cy), classifiers, scale, weighted=True, expected_indices=fixed
        )
        payload["balanced"][f"{cy:.1f}"] = balanced
        payload["unbalanced"][f"{cy:.1f}"] = unbalanced

    json_path = OUT / "observed_time_correctness.json"
    csv_path = OUT / "observed_time_correctness.csv"
    png_path = OUT / "observed_time_correctness.png"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    save_csv(payload, csv_path)
    draw_table(payload, png_path)
    print(json.dumps({
        "balanced": {key: value["mean_over_time2_time3_time4"] for key, value in payload["balanced"].items()},
        "unbalanced": {key: value["mean_over_time2_time3_time4"] for key, value in payload["unbalanced"].items()},
        "baseline": {
            key: value["mean_over_time2_time3_time4"] for key, value in payload["baseline"].items()
        },
    }, indent=2))
    print(png_path)


if __name__ == "__main__":
    main()
