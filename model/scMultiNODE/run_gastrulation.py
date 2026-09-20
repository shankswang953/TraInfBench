#!/usr/bin/env python
"""Official scMultiNODE training with normalized Gastrulation input/output adapters.

Small, explicitly subsampled smoke defaults; not a publication training protocol.
The upstream checkout is unmodified. Only decoder final activations are adapted
to signed embeddings. See README.md for time-grid and initial-state caveats.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_COMMIT = "c4e444d24849393bc923ffb5373e5cea35476013"
TIMES = np.array([0., 1., 2., 2.5], dtype=np.float32)
STAGES = ["E7.5", "E8.0", "E8.5", "E8.75"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--upstream", type=Path, default=ROOT / "external/scMultiNODE")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cells-per-time", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=10)
    p.add_argument("--ae-iters", type=int, default=200)
    p.add_argument("--fusion-iters", type=int, default=100)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ae-batch-size", type=int, default=128)
    p.add_argument("--fusion-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--align-coeff", type=float, default=0.1)
    p.add_argument("--dyn-reg-coeff", type=float, default=0.1)
    p.add_argument("--n-neighbors", type=int, default=10)
    p.add_argument("--n-particles", type=int, default=128)
    p.add_argument("--dt", type=float, default=0.025)
    p.add_argument("--threads", type=int, default=1)
    return p.parse_args()


def make_model(running, rna_dim, atac_dim, latent_dim):
    model = running.constructscMultiNODEModel(
        rna_dim, atac_dim, latent_dim, "rna", rna_enc_latent=[50],
        rna_dec_latent=[50], atac_enc_latent=[50], atac_dec_latent=[50],
        fusion_latent=[50], drift_latent=[50], act_name="relu", ode_method="euler",
    )
    # ReLU outputs cannot reconstruct negative normalized PCA/LSI coordinates.
    # Hidden layers, encoders, fusion and drift remain exactly as upstream.
    for decoder in (model.rna_dec, model.atac_dec):
        assert isinstance(decoder.net[-1], torch.nn.ReLU)
        decoder.net[-1] = torch.nn.Identity()
    return model


class LossRecorder:
    """tqdm-compatible logging only: preserve official updates and RNG calls.

    Upstream formats losses BEFORE set_postfix, so these are rounded display
    losses, explicitly not full-precision raw tensors.
    """
    rows = []
    def __init__(self, iterable, desc="", **kwargs):
        self.iterable, self.desc, self.step = iterable, desc, 0
    def __iter__(self):
        for step in self.iterable:
            self.step = step + 1
            yield step
    def set_postfix(self, values):
        values = {key: float(value) for key, value in values.items()}
        if not all(np.isfinite(list(values.values()))):
            raise FloatingPointError(f"Nonfinite displayed loss: {self.desc} {values}")
        self.rows.append({"phase": self.desc, "iteration": self.step, **values})
        if self.step == 1 or self.step % 25 == 0:
            print(self.desc, self.step, values, flush=True)


def diagnostic_plot(output, refs, dense, sparse):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 10,
                         "axes.titlesize": 10, "axes.labelsize": 10})
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), constrained_layout=True)
    for row, modality in enumerate(("rna", "atac")):
        start, end = refs[modality][0], refs[modality][-1]
        for col in range(3):
            ax = axes[row, col]
            true = start if col == 0 else end
            pred = dense[modality][0 if col == 0 else -1] if col != 2 else sparse[modality][-1]
            ax.scatter(true[:, 0], true[:, 1], s=7, alpha=.5, c="#777777", label="Observed")
            ax.scatter(pred[:, 0], pred[:, 1], s=7, alpha=.5, c="#D55E00", label="Decoded")
            ax.set_title(["Initial reconstruction", "Terminal: dense Euler grid", "Terminal: training Euler grid"][col])
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel(modality.upper() + " dimension 2")
            ax.spines[["top", "right"]].set_visible(False)
        axes[row, 0].legend(frameon=False, fontsize=10)
    fig.suptitle("Gastrulation scMultiNODE — pipeline smoke test (not converged)", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"trajectory_diagnostic.{ext}", dpi=200)
    plt.close(fig)


def main():
    a = parse_args()
    if min(a.cells_per_time, a.n_particles, a.latent_dim, a.ae_iters,
           a.fusion_iters, a.iters, a.batch_size, a.n_neighbors) < 1:
        raise ValueError("Smoke-test sizes and iteration counts must be positive")
    if a.cells_per_time > 2048:
        raise ValueError("QGW builds dense all-cell matrices. Review memory before using >2048 cells/time.")
    if a.dt <= 0 or not np.isclose(round(2.5 / a.dt) * a.dt, 2.5):
        raise ValueError("dt must be positive and divide 2.5")
    if a.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {a.output_dir}; use a new output directory")
    commit = subprocess.check_output(["git", "-C", str(a.upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(a.upstream), "status", "--porcelain"], text=True).strip()
    if commit != UPSTREAM_COMMIT or dirty:
        raise RuntimeError(f"Expected clean pinned official checkout; got {commit}, {dirty}")
    sys.path.insert(0, str(a.upstream.resolve()))
    from optim import running
    torch.set_num_threads(a.threads)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    source = {
        "rna": (a.data_dir / "rna_pca_by_time.npz", a.data_dir / "primal_norm_params.pt"),
        "atac": (a.data_dir / "atac_lsi_by_time_14D.npz", a.data_dir / "secondary_norm_params.pt"),
    }
    refs, tensors, provenance, saved_inputs = {}, {}, {}, {"time": TIMES}
    for modality, (npz_path, norm_path) in source.items():
        normalization = torch.load(norm_path, map_location="cpu", weights_only=False)
        scale = float(normalization["scale"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid frozen normalization scale")
        arrays, counts = [], []
        with np.load(npz_path) as data:
            for i in range(4):
                # Exactly the existing shared-space convention; no zscore or refit.
                all_cells = np.asarray(data[f"time{i}"], dtype=np.float32) / scale
                if not np.isfinite(all_cells).all():
                    raise ValueError("Input contains nonfinite coordinates")
                idx = rng.choice(len(all_cells), min(a.cells_per_time, len(all_cells)), replace=False)
                arrays.append(all_cells[idx])
                counts.append(len(all_cells))
                saved_inputs[f"{modality}_time{i}"] = arrays[-1]
                saved_inputs[f"{modality}_indices_time{i}"] = idx
        refs[modality] = arrays
        tensors[modality] = [torch.from_numpy(x) for x in arrays]
        provenance[modality] = {"source": str(npz_path.resolve()), "source_sha256": sha256(npz_path),
            "normalization": str(norm_path.resolve()), "normalization_sha256": sha256(norm_path),
            "scale": scale, "dimension": arrays[0].shape[1], "full_counts": counts,
            "selected_counts": [len(x) for x in arrays], "minimum": min(float(x.min()) for x in arrays),
            "maximum": max(float(x.max()) for x in arrays)}
    for modality in tensors:
        total = sum(map(len, tensors[modality]))
        if max(a.ae_batch_size, a.fusion_batch_size, a.n_neighbors + 1, 10) > total:
            raise ValueError("Not enough selected cells for official no-replacement sampling/QGW")
    a.output_dir.mkdir(parents=True)
    np.savez_compressed(a.output_dir / "smoke_inputs.npz", **saved_inputs)
    model = make_model(running, refs["rna"][0].shape[1], refs["atac"][0].shape[1], a.latent_dim)
    def finite_gradient(grad):
        if not torch.isfinite(grad).all():
            raise FloatingPointError("Nonfinite gradient; stopping without clipping/modifying training")
        return grad
    hooks = [p.register_hook(finite_gradient) for p in model.parameters()]
    old_tqdm, running.tqdm = running.tqdm, LossRecorder
    LossRecorder.rows = []
    start = time.monotonic()
    try:
        model, ae_model, fusion_model, coupling, c1, c2 = running.scMultiNODETrain(
            tensors["rna"], tensors["atac"], torch.from_numpy(TIMES), torch.from_numpy(TIMES),
            model, a.iters, a.batch_size, a.lr,
            ae_iters=a.ae_iters, ae_lr=a.lr, ae_batch_size=a.ae_batch_size,
            fusion_iters=a.fusion_iters, fusion_lr=a.lr, fusion_batch_size=a.fusion_batch_size,
            align_coeff=a.align_coeff, dyn_reg_coeff=a.dyn_reg_coeff, train_all=True,
            n_neighbors=a.n_neighbors, qgw_sample_ratio=.1, gw_type="gw", epsilon=.01,
        )
    finally:
        running.tqdm = old_tqdm
        for h in hooks:
            h.remove()
    elapsed = time.monotonic() - start
    fields = list(dict.fromkeys(k for row in LossRecorder.rows for k in row))
    with (a.output_dir / "losses_display_precision.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(LossRecorder.rows)
    model.eval()
    initial_idx = rng.choice(len(refs["rna"][0]), a.n_particles,
                             replace=a.n_particles > len(refs["rna"][0]))
    initial = tensors["rna"][0][initial_idx]
    dense_times = np.unique(np.concatenate([np.linspace(0, 2.5, round(2.5 / a.dt) + 1, dtype=np.float32), TIMES]))
    dense, sparse = {}, {}
    with torch.no_grad():
        for time_grid, target in ((dense_times, dense), (TIMES, sparse)):
            # Official forward avoids predict()'s repeat_interleave/truncate bias.
            rna, atac, passed_initial, latent_rna, latent_atac = model(
                initial, torch.from_numpy(time_grid), torch.from_numpy(time_grid), batch_size=None)
            assert torch.equal(initial, passed_initial)
            assert torch.equal(latent_rna, latent_atac)
            target.update(rna=rna.numpy().transpose(1, 0, 2), atac=atac.numpy().transpose(1, 0, 2),
                          latent=latent_rna.numpy().transpose(1, 0, 2))
    for name, target, t in (("trajectories", dense, dense_times), ("trajectories_training_grid", sparse, TIMES)):
        for values in target.values():
            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite generated trajectory")
        np.savez_compressed(a.output_dir / f"{name}.npz", time=t, rna_norm=target["rna"],
            atac_norm=target["atac"], latent=target["latent"], initial_rna_norm=initial.numpy(),
            initial_indices=saved_inputs["rna_indices_time0"][initial_idx],
            weights=np.full((len(t), a.n_particles), 1 / a.n_particles, dtype=np.float64))
    checkpoint = {"model_state_dict": model.state_dict(), "rna_dim": initial.shape[1],
                  "atac_dim": refs["atac"][0].shape[1], "latent_dim": a.latent_dim,
                  "upstream_commit": commit, "decoder_output": "identity", "training_times": TIMES.tolist()}
    torch.save(checkpoint, a.output_dir / "model.pt")
    torch.save(ae_model.state_dict(), a.output_dir / "ae_model.pt")
    torch.save(fusion_model.state_dict(), a.output_dir / "fusion_model.pt")
    from scipy.sparse import save_npz
    qgw_stored_entries_before_compaction = coupling.nnz
    # Upstream thresholding materializes explicit zeros in CSR; compact only
    # the saved artifact AFTER training, with no change to fitted values.
    coupling.eliminate_zeros()
    save_npz(a.output_dir / "qgw_correspondence.npz", coupling)
    loaded = torch.load(a.output_dir / "model.pt", map_location="cpu", weights_only=False)
    restored = make_model(running, loaded["rna_dim"], loaded["atac_dim"], loaded["latent_dim"])
    restored.load_state_dict(loaded["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        rerun = restored(initial, torch.from_numpy(dense_times), torch.from_numpy(dense_times), batch_size=None)
    errors = [float(np.max(np.abs(rerun[i].numpy().transpose(1, 0, 2) - dense[m])))
              for i, m in enumerate(("rna", "atac"))]
    if max(errors) != 0:
        raise AssertionError(f"Reload changed trajectories: {errors}")
    checks = {"finite_parameters": all(bool(torch.isfinite(p).all()) for p in model.parameters()),
        "finite_qgw_distances": bool(np.isfinite(c1).all() and np.isfinite(c2).all()),
        "qgw_nonzero_entries": int(coupling.count_nonzero()),
        "qgw_stored_entries_before_compaction": int(qgw_stored_entries_before_compaction),
        "checkpoint_reload_max_abs_error": errors,
        "rna_shape": list(dense["rna"].shape), "atac_shape": list(dense["atac"].shape),
        "latent_shape": list(dense["latent"].shape), "training_seconds": elapsed,
        "no_additional_input_scaling": True, "native_mass_growth": False,
        "decoder_initial_rna_rmse": float(np.sqrt(np.mean((dense["rna"][0] - initial.numpy()) ** 2)))}
    for modality in ("rna", "atac"):
        checks[f"{modality}_terminal_dense_vs_training_grid_rmse"] = float(np.sqrt(np.mean(
            (dense[modality][-1] - sparse[modality][-1]) ** 2)))
        checks[f"{modality}_initial_to_terminal_rms_displacement"] = float(np.sqrt(np.mean(
            (dense[modality][-1] - dense[modality][0]) ** 2)))
    if not checks["finite_parameters"] or not checks["finite_qgw_distances"] or coupling.nnz == 0:
        raise AssertionError(checks)
    manifest = {"status": "smoke_completed_not_convergence_validated", "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        "upstream_commit": commit, "upstream_unmodified": True, "inputs": provenance,
        "stage_names": STAGES, "time": TIMES.tolist(), "checks": checks,
        "adaptations": ["Decoder final ReLU -> Identity for signed normalized embeddings",
                        "Official forward(batch_size=None) on explicit initial sample instead of biased predict repetition",
                        "tqdm display loss recorder and nonfinite gradient check; no objective changes"],
        "caveats": ["Subsampled full-data smoke; not strict LOO, not a full training result",
                    "Initial decoded state is an autoencoder reconstruction, not forced to equal observed RNA",
                    "Official Euler uses query times as integration steps; dense and training-grid outputs differ",
                    "Uniform particle weights: no birth/death or inferred growth"],
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchdiffeq", "numpy", "scipy", "POT", "geomloss", "scikit-learn")}}
    (a.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    diagnostic_plot(a.output_dir, refs, dense, sparse)
    print(json.dumps(checks, indent=2), flush=True)
    print("Saved", a.output_dir, flush=True)


if __name__ == "__main__":
    main()
