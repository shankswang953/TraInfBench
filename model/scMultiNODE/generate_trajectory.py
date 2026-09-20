#!/usr/bin/env python
"""Generate decoded trajectories from ALL original initial cells by default."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

from run_gastrulation import ROOT, TIMES, UPSTREAM_COMMIT, make_model, sha256
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--upstream", type=Path, default=ROOT / "external/scMultiNODE")
    p.add_argument("--dt", type=float, default=.025)
    p.add_argument("--initial-source", choices=("all", "saved"), default="all",
                   help="All original earliest-stage RNA cells (default); saved replays an older exported subset.")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Inference memory batch only; never subsamples initial cells.")
    a = p.parse_args()
    if a.output.exists() or a.output.with_suffix(".json").exists():
        raise FileExistsError("Output exists; choose a new path")
    if a.output.suffix != ".npz" or a.batch_size < 1:
        raise ValueError("Use an .npz output and positive inference batch size")
    if not np.isfinite(a.dt) or a.dt <= 0:
        raise ValueError("dt must be positive")
    commit = subprocess.check_output(["git", "-C", str(a.upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(a.upstream), "status", "--porcelain"], text=True).strip()
    if commit != UPSTREAM_COMMIT or dirty:
        raise ValueError("Upstream revision mismatch")
    sys.path.insert(0, str(a.upstream.resolve()))
    from optim import running
    torch.set_num_threads(1)
    ckpt = torch.load(a.run_dir / "model.pt", map_location="cpu", weights_only=False)
    model = make_model(running, ckpt["rna_dim"], ckpt["atac_dim"], ckpt["latent_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    manifest = json.loads((a.run_dir / "manifest.json").read_text())
    all_times = np.asarray(ckpt.get("all_times", manifest.get("time", TIMES)), dtype=np.float32)
    if (all_times.ndim != 1 or len(all_times) < 2 or not np.isfinite(all_times).all()
            or all_times[0] != 0 or np.any(np.diff(all_times) <= 0)):
        raise ValueError("Invalid original dataset clock")
    end_time = float(all_times[-1])
    if not np.isclose(round(end_time / a.dt) * a.dt, end_time):
        raise ValueError(f"dt must divide dataset terminal time {end_time}")
    source_info = manifest["inputs"]["rna"]
    source = Path(source_info["source"])
    if sha256(source) != source_info["source_sha256"]:
        raise ValueError("Original source changed since training")
    if a.initial_source == "all":
        with np.load(source) as z:
            raw_initial = np.asarray(z[source_info.get("stage_keys", ["time0"])[0]], dtype=np.float32)
        initial = torch.from_numpy(raw_initial / float(source_info["scale"]))
        indices = np.arange(len(initial), dtype=np.int64)
        if len(initial) != source_info["full_counts"][0]:
            raise ValueError("Full initial cell count mismatch")
    else:
        with np.load(a.run_dir / "trajectories.npz") as z:
            initial = torch.from_numpy(z["initial_rna_norm"].copy())
            indices = z["initial_indices"].copy()
    from trajectory_export import query_time_grid
    times = query_time_grid(np.unique(np.concatenate([
        all_times, np.asarray(ckpt["training_times"], dtype=np.float32)])), a.dt)
    # New benchmark checkpoints explicitly use native Euler knot interpolation.
    # Never silently replay those checkpoints with the legacy dense-query solve.
    if "scenario" in ckpt:
        from trajectory_export import export_trajectories
        audit = export_trajectories(model, initial, indices, ckpt["training_times"], times,
                                    a.output, batch_size=a.batch_size)
        a.output.with_suffix(".json").write_text(json.dumps({
            "checkpoint_sha256": sha256(a.run_dir / "model.pt"), "upstream_commit": commit,
            "scenario": ckpt["scenario"], "initial_source": a.initial_source,
            "initial_source_sha256": source_info["source_sha256"], "dt": a.dt,
            "solver": "native training-grid Euler with piecewise-linear latent dense output",
            "audit": audit, "model_retrained": False}, indent=2))
        print(a.output)
        return
    arrays = {key: np.empty((len(times), len(initial), dim), dtype=np.float32)
              for key, dim in (("rna_norm", ckpt["rna_dim"]), ("atac_norm", ckpt["atac_dim"]),
                               ("latent", ckpt["latent_dim"]))}
    with torch.no_grad():
        for start in range(0, len(initial), a.batch_size):
            stop = min(start + a.batch_size, len(initial))
            rna, atac, returned_initial, latent, latent_atac = model(
                initial[start:stop], torch.from_numpy(times), torch.from_numpy(times), batch_size=None)
            assert torch.equal(returned_initial, initial[start:stop])
            assert torch.equal(latent, latent_atac)
            for key, tensor in (("rna_norm", rna), ("atac_norm", atac), ("latent", latent)):
                arrays[key][:, start:stop] = tensor.numpy().transpose(1, 0, 2)
            print(f"Pushed initial cells {start}:{stop} / {len(initial)}", flush=True)
    assert all(np.isfinite(x).all() for x in arrays.values())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, time=times, **arrays, initial_indices=indices,
                        initial_rna_norm=initial.numpy(), weights=np.full((len(times), len(initial)), 1/len(initial)))
    a.output.with_suffix(".json").write_text(json.dumps({"checkpoint_sha256": sha256(a.run_dir / "model.pt"),
        "upstream_commit": commit, "dt": a.dt, "solver": "official euler, query-grid-dependent",
        "initial_source": a.initial_source, "initial_source_file": str(source),
        "initial_source_sha256": source_info["source_sha256"], "initial_scale": source_info["scale"],
        "n_initial": len(initial), "n_unique_initial_indices": len(np.unique(indices)),
        "all_initial_once": bool(a.initial_source == "all" and np.array_equal(indices, np.arange(len(initial)))),
        "inference_batch_size": a.batch_size, "model_retrained": False,
        "shapes": {k: list(v.shape) for k,v in arrays.items()}}, indent=2))
    print(a.output)


if __name__ == "__main__":
    main()
