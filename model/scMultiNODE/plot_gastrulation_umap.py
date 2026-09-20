#!/usr/bin/env python
"""Compare initial-push decoded trajectories to true stages using frozen UMAPs.

No UMAP refitting. Normalized predictions are multiplied by the original frozen
scale ONLY to match the raw-coordinate input expected by the saved UMAP models.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from run_gastrulation import sha256, TIMES, STAGES


def draw(modality, observed, predicted, dest, run_label="short-training test", heldout_stage=None):
    plt.rcParams.update({"font.family": "Arial", "font.size": 10,
        "axes.titlesize": 10, "axes.labelsize": 10, "legend.fontsize": 10,
        "font.weight": "normal", "axes.titleweight": "normal"})
    fig, axes = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True)
    all_xy = np.concatenate([*observed, *predicted])
    lo, hi = all_xy.min(axis=0), all_xy.max(axis=0)
    pad = (hi - lo) * .055
    for i, stage in enumerate(STAGES):
        true, pred = observed[i], predicted[i]
        axes[0, i].scatter(true[:, 0], true[:, 1], s=2.5, c="#666666", alpha=.45,
                           linewidths=0, rasterized=True)
        axes[1, i].scatter(true[:, 0], true[:, 1], s=2.5, c="#CFCFCF", alpha=.4,
                           linewidths=0, rasterized=True)
        axes[1, i].scatter(pred[:, 0], pred[:, 1], s=3 if len(pred)>1000 else 13,
                           c="#D55E00", alpha=.4 if len(pred)>1000 else .85,
                           linewidths=0, rasterized=True)
        suffix = " (held out)" if stage == heldout_stage else ""
        axes[0, i].set_title(f"{stage}{suffix}\nObserved n={len(true):,}", pad=7)
        for ax in axes[:, i]:
            ax.set_xlim(lo[0]-pad[0], hi[0]+pad[0])
            ax.set_ylim(lo[1]-pad[1], hi[1]+pad[1])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    axes[0, 0].set_ylabel("Observed", labelpad=8)
    axes[1, 0].set_ylabel("Initial → time", labelpad=8)
    fig.suptitle(f"Gastrulation — scMultiNODE {modality.upper()} ({run_label})", fontsize=10, y=.97)
    legend = [Line2D([], [], color="#999999", marker="o", linestyle="", markersize=4, label="Observed"),
              Line2D([], [], color="#D55E00", marker="o", linestyle="", markersize=4,
                     label=f"Initial-push prediction (n={len(predicted[0])})")]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5,.035))
    fig.subplots_adjust(left=.065, right=.99, bottom=.14, top=.85, hspace=.02, wspace=.05)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(dest / f"{modality}_initial_push_vs_observed_umap.{ext}", dpi=240)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--trajectory", default="trajectories.npz")
    p.add_argument("--output-dir", type=Path)
    a = p.parse_args()
    dest = a.output_dir or a.run_dir / "umap_by_time"
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite {dest}")
    manifest = json.loads((a.run_dir / "manifest.json").read_text())
    scenario = manifest.get("scenario")
    run_label = {"full": "full", "loo1": "LOO E8.0", "loo2": "LOO E8.5"}.get(scenario, "short-training test")
    if manifest.get("engineering_smoke"):
        run_label += "; engineering smoke"
    caveats = ["E7.5 prediction is decoded initial latent, not exact observed input",
               "UMAP visualization is not a distribution metric; compare in normalized source spaces"]
    if scenario:
        caveats += ["Native training-grid Euler with piecewise-linear latent dense output",
                    "Frozen common PCA/LSI/scalars; not end-to-end inductive raw-data LOO"]
    else:
        caveats += ["Short full-data subsampled training; not converged or LOO",
                    "Legacy dense-query Euler may differ from training-grid Euler"]
    traj_path = a.run_dir / a.trajectory
    outputs, audit = {}, {}
    with np.load(traj_path) as traj:
        time = traj["time"]
        indices = [int(np.argmin(abs(time - t))) for t in TIMES]
        if not np.allclose(time[indices], TIMES, atol=1e-6):
            raise ValueError("Trajectory is missing observed times")
        for modality, model_name in (("rna", "rna_umap_model.joblib"),
                                     ("atac", "atac_umap_model_14D.joblib")):
            info = manifest["inputs"][modality]
            source = Path(info["source"])
            if sha256(source) != info["source_sha256"]:
                raise ValueError("Input fixture changed since training")
            with np.load(source) as z:
                raw_stages = [np.asarray(z[f"time{i}"], dtype=np.float32) for i in range(4)]
            raw = np.concatenate(raw_stages)
            model_path = source.parent / model_name
            print("Load frozen", modality, "UMAP", model_path, flush=True)
            model = joblib.load(model_path)
            training = np.asarray(model._raw_data, dtype=np.float32)
            if raw.shape != training.shape:
                raise ValueError(f"{modality}: raw atlas shape does not match frozen UMAP")
            error = float(np.max(abs(raw - training)))
            if error > 1e-5:
                raise ValueError(f"{modality}: raw atlas mismatch {error}")
            coords = np.asarray(model.embedding_, dtype=np.float32)
            bounds = np.cumsum([0, *[len(x) for x in raw_stages]])
            observed = [coords[bounds[i]:bounds[i+1]] for i in range(4)]
            normalized = np.asarray(traj[f"{modality}_norm"][indices], dtype=np.float32)
            # Keep existing UMAP geometry: its fit was on unnormalized PCA/LSI.
            raw_pred = normalized * float(info["scale"])
            model.n_jobs = 1
            predicted = []
            for i, stage in enumerate(STAGES):
                print("Project", modality, stage, raw_pred[i].shape, flush=True)
                xy = np.asarray(model.transform(raw_pred[i]), dtype=np.float32)
                if not np.isfinite(xy).all():
                    raise FloatingPointError("Nonfinite UMAP projection")
                predicted.append(xy)
                outputs[f"{modality}_observed_time{i}"] = observed[i]
                outputs[f"{modality}_predicted_time{i}"] = xy
                outputs[f"{modality}_predicted_norm_time{i}"] = normalized[i]
            audit[modality] = {"umap": str(model_path), "umap_sha256": sha256(model_path),
                "fit_data_max_abs_difference": error, "fit_data_shape": list(training.shape),
                "prediction_normalization_inverse_scale": info["scale"],
                "transform_seed": getattr(model, "transform_seed", None),
                "reference_counts": [len(x) for x in observed], "predicted_counts": [len(x) for x in predicted]}
            if not dest.exists():
                dest.mkdir(parents=True)
            draw(modality, observed, predicted, dest, run_label, manifest.get("heldout_stage"))
    outputs["time"] = TIMES
    np.savez_compressed(dest / "umap_coordinates.npz", **outputs)
    (dest / "manifest.json").write_text(json.dumps({"training_run": str(a.run_dir.resolve()),
        "trajectory": str(traj_path.resolve()), "trajectory_sha256": sha256(traj_path),
        "time": TIMES.tolist(), "source": "initial RNA -> shared latent ODE -> separate modality decoders",
        "reference": "all original cells, saved embedding_; no UMAP refit",
        "display": "top: observed; bottom: same observed gray with prediction orange",
        "scenario": scenario, "caveats": caveats,
        "audit": audit}, indent=2))
    print("Saved", dest, flush=True)


if __name__ == "__main__":
    main()
