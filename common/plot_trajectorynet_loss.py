#!/usr/bin/env python

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[1]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TRAIN_RE = re.compile(
    r"^Iter\s+(\d+)\s+\|.*?\|\s+Loss\s+([-+0-9.eE]+)\(([-+0-9.eE]+)\)"
)
EVAL_RE = re.compile(r"^\s*(\d+)\s*,\s*(?:tensor\()?([-+0-9.eE]+)")


def parse_train_log(path: Path):
    iters, losses, avg_losses = [], [], []
    if not path.exists():
        return iters, losses, avg_losses
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = TRAIN_RE.search(line)
            if match is None:
                continue
            iters.append(int(match.group(1)))
            losses.append(float(match.group(2)))
            avg_losses.append(float(match.group(3)))
    return iters, losses, avg_losses


def parse_eval_csv(path: Path):
    iters, losses = [], []
    if not path.exists():
        return iters, losses
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = EVAL_RE.search(line)
            if match is None:
                continue
            iters.append(int(match.group(1)))
            losses.append(float(match.group(2)))
    return iters, losses


def read_loss_terms(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = {field: [] for field in reader.fieldnames or []}
        for row in reader:
            for field in columns:
                value = row.get(field, "")
                if value == "":
                    columns[field].append(np.nan)
                else:
                    columns[field].append(float(value))
    return columns


def numeric_term_columns(table):
    columns = []
    for key, values in table.items():
        if key == "iter":
            continue
        arr = np.asarray(values, dtype=float)
        if np.isfinite(arr).any():
            columns.append(key)
    return columns


def prefixed_terms_path(output_prefix: Path):
    if output_prefix.name.endswith("_curve"):
        return output_prefix.with_name(output_prefix.name[:-6] + "_terms")
    return output_prefix.with_name(output_prefix.name + "_terms")


def _plot_common(ax):
    ax.grid(True, color="#D7DCE2", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_structured_losses(result_dir: Path, output_prefix: Path):
    train = read_loss_terms(result_dir / "loss_terms_train.csv")
    if train is None:
        return False
    eval_terms = read_loss_terms(result_dir / "loss_terms_eval.csv")
    term_columns = numeric_term_columns(train)
    if not term_columns:
        raise RuntimeError(f"No numeric terms found in {result_dir / 'loss_terms_train.csv'}")

    train_iter = np.asarray(train["iter"], dtype=float)
    eval_iter = None if eval_terms is None else np.asarray(eval_terms["iter"], dtype=float)

    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    objective_cols = [
        col for col in ["total_loss", "nll_loss", "loss_before_cnf_regularization"]
        if col in term_columns
    ]
    colors = ["#1B4E7A", "#4C78A8", "#54A24B"]
    for color, col in zip(colors, objective_cols):
        ax.plot(train_iter, train[col], linewidth=1.8, color=color, label=f"train {col}")
        if eval_terms is not None and col in eval_terms:
            eval_arr = np.asarray(eval_terms[col], dtype=float)
            if np.isfinite(eval_arr).any():
                ax.plot(
                    eval_iter,
                    eval_arr,
                    marker="o",
                    markersize=4.5,
                    linewidth=1.5,
                    linestyle="--",
                    label=f"eval {col}",
                )
    ax.set_title("TrajectoryNet Objective Curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    _plot_common(ax)
    ax.legend(loc="best", frameon=False)
    fig.savefig(output_prefix.with_suffix(".png"))
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")

    terms_prefix = prefixed_terms_path(output_prefix)
    ncols = 2 if len(term_columns) <= 4 else 3
    nrows = math.ceil(len(term_columns) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols, 3.4 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, col in zip(axes.ravel(), term_columns):
        ax.plot(train_iter, train[col], color="#1B4E7A", linewidth=1.5, label="train")
        if eval_terms is not None and col in eval_terms:
            eval_arr = np.asarray(eval_terms[col], dtype=float)
            if np.isfinite(eval_arr).any():
                ax.plot(
                    eval_iter,
                    eval_arr,
                    color="#F58518",
                    marker="o",
                    markersize=3.8,
                    linewidth=1.2,
                    linestyle="--",
                    label="eval",
                )
        ax.set_title(col, fontsize=12)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        _plot_common(ax)
    for ax in axes.ravel()[len(term_columns):]:
        ax.axis("off")
    fig.savefig(terms_prefix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(terms_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {terms_prefix.with_suffix('.png')}")
    print(f"Wrote {terms_prefix.with_suffix('.pdf')}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/trajectorynet_moscot_rna_10000"),
    )
    parser.add_argument("--stdout-log", type=Path, default=None)
    parser.add_argument("--output-prefix", type=Path, default=None)
    args = parser.parse_args()

    result_dir = args.result_dir
    stdout_log = args.stdout_log
    if stdout_log is None:
        stdout_log = result_dir.parent / f"{result_dir.name}_stdout.log"

    output_prefix = args.output_prefix or (result_dir / "trajectorynet_loss_curve")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 160,
            "savefig.dpi": 220,
        }
    )

    if plot_structured_losses(result_dir, output_prefix):
        return

    train_iters, train_losses, train_avg_losses = parse_train_log(stdout_log)
    eval_iters, eval_losses = parse_eval_csv(result_dir / "train_eval.csv")
    if not train_iters and not eval_iters:
        raise RuntimeError(f"No losses found in {stdout_log} or {result_dir / 'train_eval.csv'}")

    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    if train_iters:
        ax.plot(
            train_iters,
            train_losses,
            color="#4C78A8",
            alpha=0.28,
            linewidth=0.9,
            label="train NLL",
        )
        ax.plot(
            train_iters,
            train_avg_losses,
            color="#1B4E7A",
            linewidth=1.8,
            label="train NLL running avg",
        )
    if eval_iters:
        ax.plot(
            eval_iters,
            eval_losses,
            color="#F58518",
            marker="o",
            markersize=4.8,
            linewidth=2.0,
            label="test NLL",
        )

    ax.set_title("TrajectoryNet Training Curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Negative log-likelihood")
    ax.grid(True, color="#D7DCE2", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False)

    fig.savefig(output_prefix.with_suffix(".png"))
    fig.savefig(output_prefix.with_suffix(".pdf"))
    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
