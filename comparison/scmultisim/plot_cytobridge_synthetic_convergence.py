#!/usr/bin/env python
"""Recover CytoBridge convergence curves from the synthetic run logs."""

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
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BALANCED_LOG = (
    ROOT / "results/cytobridge_synthetic_rna10_balanced_d10_e1_n128_i3000_stdout.log"
)
DEFAULT_UNBALANCED_LOG = (
    ROOT / "results/cytobridge_synthetic_rna10_unbalanced_d10_e1_n128_i3000_stdout.log"
)
DEFAULT_OUTPUT_DIR = ROOT / "results/cytobridge_synthetic_rna10_d10_e1_convergence"

STAGE_RE = re.compile(r"Starting Stage: (Pretrain|Train)")
LOGGED_RE = re.compile(
    r"Stage '(Pretrain|Train)', Epoch\s+(\d+)/(\d+), Loss: ([0-9.eE+-]+)"
)
BEST_RE = re.compile(r"Epoch\s+(\d+) has a lower loss\| all_loss ([0-9.eE+-]+)")
SAVED_RE = re.compile(
    r"(Best|Last) model \(loss=([0-9.eE+-]+)\) saved"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balanced-log", type=Path, default=DEFAULT_BALANCED_LOG)
    parser.add_argument("--unbalanced-log", type=Path, default=DEFAULT_UNBALANCED_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_log(path: Path, method: str) -> tuple[list[dict], list[dict], list[dict]]:
    stage: str | None = None
    logged: list[dict] = []
    best: list[dict] = []
    saved: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stage_match = STAGE_RE.search(line)
        if stage_match:
            stage = stage_match.group(1)
        logged_match = LOGGED_RE.search(line)
        if logged_match:
            logged.append(
                {
                    "method": method,
                    "stage": logged_match.group(1),
                    "epoch": int(logged_match.group(2)),
                    "total_epochs": int(logged_match.group(3)),
                    "loss": float(logged_match.group(4)),
                }
            )
        best_match = BEST_RE.search(line)
        if best_match:
            if stage is None:
                raise ValueError(f"Best-loss update appeared before a stage in {path}")
            best.append(
                {
                    "method": method,
                    "stage": stage,
                    "epoch": int(best_match.group(1)) + 1,
                    "best_loss": float(best_match.group(2)),
                }
            )
        saved_match = SAVED_RE.search(line)
        if saved_match:
            if stage is None:
                raise ValueError(f"Saved-loss entry appeared before a stage in {path}")
            total_epochs = max(
                row["total_epochs"] for row in logged if row["stage"] == stage
            )
            saved.append(
                {
                    "method": method,
                    "stage": stage,
                    "epoch": total_epochs,
                    "checkpoint_strategy": saved_match.group(1).lower(),
                    "saved_checkpoint_loss": float(saved_match.group(2)),
                }
            )
    return logged, best, saved


def make_plot(
    logged: pd.DataFrame,
    best: pd.DataFrame,
    saved: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharex=True, sharey=True)
    colors = {"Balanced": "#4C78A8", "Unbalanced": "#E45756"}
    for axis, method in zip(axes, ("Balanced", "Unbalanced")):
        curve = logged[(logged["method"] == method) & (logged["stage"] == "Train")]
        curve = curve.sort_values("epoch").copy()
        curve["rolling_median"] = curve["loss"].rolling(11, center=True, min_periods=3).median()
        best_curve = best[(best["method"] == method) & (best["stage"] == "Train")]
        last_row = saved[(saved["method"] == method) & (saved["stage"] == "Train")].iloc[-1]

        axis.plot(
            curve["epoch"],
            curve["loss"],
            color=colors[method],
            alpha=0.20,
            linewidth=0.8,
            label="Logged minibatch total loss",
        )
        axis.plot(
            curve["epoch"],
            curve["rolling_median"],
            color=colors[method],
            linewidth=2.1,
            label="110-epoch rolling median",
        )
        axis.step(
            best_curve["epoch"],
            best_curve["best_loss"],
            where="post",
            color="#222222",
            linewidth=1.4,
            linestyle="--",
            label="Best observed loss",
        )
        axis.scatter(
            [last_row["epoch"]],
            [last_row["saved_checkpoint_loss"]],
            s=42,
            facecolors="white",
            edgecolors=colors[method],
            linewidths=1.8,
            zorder=5,
            label="Saved checkpoint",
        )
        axis.set_title(method)
        axis.set_xlabel("Main-train epoch")
        axis.grid(axis="y", color="#E7E7E7", linewidth=0.8)
    axes[0].set_ylabel("Reported total loss")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("CytoBridge synthetic RNA10 convergence", y=0.99)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.24, wspace=0.14)
    fig.savefig(output_dir / "cytobridge_convergence.png", dpi=300, facecolor="white")
    fig.savefig(output_dir / "cytobridge_convergence.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    outputs = (
        output_dir / "cytobridge_logged_losses.csv",
        output_dir / "cytobridge_best_loss_updates.csv",
        output_dir / "cytobridge_saved_checkpoints.csv",
        output_dir / "cytobridge_convergence_summary.json",
        output_dir / "cytobridge_convergence.png",
        output_dir / "cytobridge_convergence.pdf",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs; pass --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    logged_rows: list[dict] = []
    best_rows: list[dict] = []
    saved_rows: list[dict] = []
    for method, path in (
        ("Balanced", args.balanced_log),
        ("Unbalanced", args.unbalanced_log),
    ):
        logged, best, saved = parse_log(path, method)
        logged_rows.extend(logged)
        best_rows.extend(best)
        saved_rows.extend(saved)

    logged_frame = pd.DataFrame(logged_rows)
    best_frame = pd.DataFrame(best_rows)
    saved_frame = pd.DataFrame(saved_rows)
    logged_frame.to_csv(outputs[0], index=False)
    best_frame.to_csv(outputs[1], index=False)
    saved_frame.to_csv(outputs[2], index=False)

    summary: dict[str, dict[str, float | int]] = {}
    for method in ("Balanced", "Unbalanced"):
        train_best = best_frame[
            (best_frame["method"] == method) & (best_frame["stage"] == "Train")
        ].iloc[-1]
        train_saved = saved_frame[
            (saved_frame["method"] == method) & (saved_frame["stage"] == "Train")
        ].iloc[-1]
        summary[method] = {
            "best_epoch": int(train_best["epoch"]),
            "best_loss": float(train_best["best_loss"]),
            "saved_checkpoint": f"{train_saved['checkpoint_strategy']}_model.pth",
            "saved_checkpoint_loss": float(train_saved["saved_checkpoint_loss"]),
            "saved_minus_best": float(
                train_saved["saved_checkpoint_loss"] - train_best["best_loss"]
            ),
        }
    outputs[3].write_text(
        json.dumps(
            {
                "logs": {
                    "Balanced": str(args.balanced_log.resolve()),
                    "Unbalanced": str(args.unbalanced_log.resolve()),
                },
                "logging_granularity": (
                    "Total loss every 10 epochs plus every new best total loss; "
                    "no individual OT/mass/density/energy components"
                ),
                "checkpoint_inventory": saved_frame.to_dict(orient="records"),
                "main_train_summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    make_plot(logged_frame, best_frame, saved_frame, output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
