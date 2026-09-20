#!/usr/bin/env python
"""Bounded storage-equivalence and full-input pilots; never a 20k benchmark.

All outputs are new directories. Temporary float64 distances are removed on
normal exit; input fixtures and the clean official checkout remain untouched.
The watchdog only terminates this diagnostic process, not other user jobs.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading
import time

from run_gastrulation import ROOT, TIMES, UPSTREAM_COMMIT, make_model, sha256
from benchmark_gastrulation import SPLITS, load_inputs, checkpoint_payload
import numpy as np
import torch
import psutil


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ResourceGuard:
    def __init__(self, output, max_seconds, max_rss_gib, min_available_gib):
        self.output = output
        self.max_seconds, self.max_rss_gib = max_seconds, max_rss_gib
        self.min_available_gib = min_available_gib
        self.started = time.monotonic()
        self.phase = "initialization"
        self.phase_timings = {}
        self.peak_rss_gib = 0.
        self.minimum_available_gib = float("inf")
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def snapshot(self):
        return {"pid": os.getpid(), "phase": self.phase, "elapsed_seconds": time.monotonic()-self.started,
                "sampled_peak_rss_gib": self.peak_rss_gib,
                "minimum_available_gib": self.minimum_available_gib,
                "rss_sampling_interval_seconds": .2,
                "max_seconds": self.max_seconds, "max_rss_gib": self.max_rss_gib,
                "min_available_gib": self.min_available_gib}

    def watch(self):
        proc = psutil.Process()
        last_report = 0.
        while not self.stop.is_set():
            try:
                rss = proc.memory_info().rss / 1024**3
                available = psutil.virtual_memory().available / 1024**3
                self.peak_rss_gib = max(self.peak_rss_gib, rss)
                self.minimum_available_gib = min(self.minimum_available_gib, available)
                if time.monotonic()-last_report > 2:
                    dump(self.output / "resource_live.json", self.snapshot())
                    last_report = time.monotonic()
                reasons = []
                if rss > self.max_rss_gib:
                    reasons.append("RSS budget exceeded")
                if available < self.min_available_gib:
                    reasons.append("System available-memory reserve reached")
                if time.monotonic()-self.started > self.max_seconds:
                    reasons.append("Wall-time budget reached")
                if reasons:
                    report = self.snapshot() | {"status": "guard_stopped", "reasons": reasons}
                    dump(self.output / "resource_guard.json", report)
                    print(json.dumps(report), flush=True)
                    os._exit(75)
            except (OSError, psutil.Error) as error:
                dump(self.output / "resource_guard.json", {"status": "guard_failed", "error": repr(error)})
                os._exit(76)
            self.stop.wait(.2)


def progress_type(guard, records, label):
    class Progress:
        def __init__(self, iterable, desc="", **kwargs):
            self.iterable, self.desc, self.step = iterable, desc, 0

        def __iter__(self):
            guard.phase = label + self.desc
            start = time.monotonic()
            print(f"{label} {self.desc} START", flush=True)
            for i in self.iterable:
                self.step = i + 1
                yield i
            print(f"{label} {self.desc} END {time.monotonic()-start:.3f}s", flush=True)
            guard.phase_timings[label + " " + self.desc] = time.monotonic()-start
            if "AE Pre" in self.desc:
                guard.phase = label + " QGW distances and matching"

        def set_postfix(self, values):
            values = {key: float(value) for key, value in values.items()}
            if not np.isfinite(list(values.values())).all():
                raise FloatingPointError("Nonfinite displayed loss")
            records.append({"phase": self.desc, "iteration": self.step, **values})
            if self.step == 1 or self.step % 100 == 0:
                print(label, self.desc, self.step, values, flush=True)
    return Progress


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("compare", "pilot"), required=True)
    p.add_argument("--scenario", choices=list(SPLITS), default="loo1")
    p.add_argument("--data-dir", type=Path, default=ROOT.parent / "TraInf/TraInf/Gastrulation/data")
    p.add_argument("--output-dir", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--cells-per-time", type=int)
    group.add_argument("--all-observed", action="store_true")
    p.add_argument("--ae-iters", type=int, default=200)
    p.add_argument("--fusion-iters", type=int, default=20)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=900)
    p.add_argument("--max-rss-gib", type=float, default=10)
    p.add_argument("--min-available-gib", type=float, default=4)
    p.add_argument("--row-block-size", type=int, default=128)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    if min(a.ae_iters, a.fusion_iters, a.iters, a.row_block_size) < 1 or a.iters > 100:
        raise ValueError("Diagnostic only: positive counts, at most 100 dynamics updates")
    if a.cells_per_time is not None and a.cells_per_time < 64:
        raise ValueError("Use at least 64 cells/time for the unchanged public AE/fusion batches")
    if a.mode == "compare" and (a.all_observed or a.cells_per_time > 512):
        raise ValueError("Official reference comparison is limited to <=512 cells/time")
    for value in (a.max_seconds, a.max_rss_gib, a.min_available_gib):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Invalid resource budget")
    available = psutil.virtual_memory().available / 1024**3
    if available < a.min_available_gib + 4:
        raise MemoryError("Insufficient headroom even for a bounded diagnostic")
    upstream = ROOT / "external/scMultiNODE"
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip()
    if commit != UPSTREAM_COMMIT or dirty:
        raise RuntimeError("Official source must remain clean and pinned")
    sys.path.insert(0, str(upstream))
    from optim import running
    from memory_optimized_train import make_memory_trainer
    from trajectory_export import export_trajectories
    torch.set_num_threads(1)
    data, initial, provenance, indices = load_inputs(a.data_dir, a.scenario)
    if a.cells_per_time is not None:
        for modality in data:
            for pos, stage in enumerate(SPLITS[a.scenario]):
                rng = np.random.default_rng(a.seed + stage + (1000 if modality == "atac" else 0))
                chosen = rng.choice(len(data[modality][pos]), min(a.cells_per_time, len(data[modality][pos])), replace=False)
                data[modality][pos] = data[modality][pos][chosen]
                indices[modality][f"time{stage}"] = chosen
    a.output_dir.mkdir(parents=True)
    config = json.loads((ROOT / f"model/scMultiNODE/configs/gastrulation_{a.scenario}.json").read_text())
    config.update(ae_iters=a.ae_iters, fusion_iters=a.fusion_iters, iters=a.iters, seed=a.seed)
    manifest = {"status": "running", "purpose": "bounded implementation diagnostic, not final benchmark",
                "mode": a.mode, "scenario": a.scenario, "all_observed": a.all_observed,
                "cells_per_time": a.cells_per_time, "config": config, "inputs": provenance,
                "actual_training_counts": {m: [len(x) for x in data[m]] for m in data},
                "training_times": TIMES[SPLITS[a.scenario]].tolist(),
                "initial_export_count": len(initial), "upstream_commit": commit,
                "adapter_source_sha256": {name: sha256(Path(__file__).parent / name) for name in (
                    "memory_optimized_train.py", "memory_distances.py", "memory_sparse_qgw.py", "trial_memory_optimized.py")},
                "official_source_clean": True}
    dump(a.output_dir / "manifest.json", manifest)
    np.savez_compressed(a.output_dir / "training_indices.npz", **{
        f"{m}_{stage}": ix for m in indices for stage, ix in indices[m].items()})
    guard = ResourceGuard(a.output_dir, a.max_seconds, a.max_rss_gib, a.min_available_gib)
    guard.thread.start()
    snapshots = {}
    try:
        with tempfile.TemporaryDirectory(prefix="scmultinode-distance-", dir="/private/tmp") as scratch:
            dump(a.output_dir / "scratch.json", {"path": scratch, "temporary_only": True})
            modes = ("official", "optimized") if a.mode == "compare" else ("optimized",)
            for label in modes:
                seed_all(a.seed)
                model = make_model(running, 50, 14, 10)
                losses = []
                progress = progress_type(guard, losses, label)
                old_progress = running.tqdm
                if label == "optimized":
                    trainer, distances, adaptation = make_memory_trainer(running, scratch,
                        row_block_size=a.row_block_size, progress=progress)
                else:
                    trainer, distances, adaptation = running.scMultiNODETrain, None, None
                    running.tqdm = progress
                hooks = [param.register_hook(check_gradient) for param in model.parameters()]
                start = time.monotonic()
                try:
                    model, ae, fusion, coupling, c1, c2 = trainer(
                        data["rna"], data["atac"], torch.from_numpy(TIMES[SPLITS[a.scenario]]),
                        torch.from_numpy(TIMES[SPLITS[a.scenario]]), model,
                        a.iters, 1024, .001, ae_iters=a.ae_iters, ae_lr=.001, ae_batch_size=128,
                        fusion_iters=a.fusion_iters, fusion_lr=.001, fusion_batch_size=128,
                        align_coeff=.1, dyn_reg_coeff=.1, train_all=True, n_neighbors=10,
                        qgw_sample_ratio=.1, gw_type="gw", epsilon=.01)
                finally:
                    running.tqdm = old_progress
                    for hook in hooks:
                        hook.remove()
                elapsed = time.monotonic()-start
                dump(a.output_dir / f"{label}_losses.json", losses)
                model.eval()
                torch.save(checkpoint_payload(model, config, a.scenario, "dynamics", a.iters),
                           a.output_dir / f"model_{label}.pt")
                with torch.no_grad():
                    predictions = model(data["rna"][0][:64], torch.from_numpy(TIMES[SPLITS[a.scenario]]),
                                        torch.from_numpy(TIMES[SPLITS[a.scenario]]), batch_size=None)
                snapshot = {"model": {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()},
                            "predictions": [v.detach().cpu().numpy().copy() for v in predictions],
                            "gradient": {k: v.grad.detach().numpy().copy() for k,v in model.named_parameters() if v.grad is not None}}
                if a.mode == "compare":
                    snapshot.update(c1=np.asarray(c1).copy(), c2=np.asarray(c2).copy(), coupling=coupling.toarray())
                snapshots[label] = snapshot
                manifest[label] = {"training_seconds": elapsed, "loss_records": len(losses),
                    "correspondence_nonzeros": int(coupling.count_nonzero()),
                    "correspondence_stored_entries": int(coupling.nnz),
                    "distance_records": distances.records if distances is not None else None,
                    "adaptation": adaptation}
                if label == "optimized":
                    guard.phase = "all-initial native-grid export"
                    manifest["trajectory_audit"] = export_trajectories(model, initial, np.arange(len(initial)),
                        TIMES[SPLITS[a.scenario]], TIMES, a.output_dir / "trajectories.npz", batch_size=512)
                del c1, c2, coupling, ae, fusion, model, predictions
                gc.collect()
            if a.mode == "compare":
                errors = {}
                left, right = snapshots["official"], snapshots["optimized"]
                for key in ("c1", "c2", "coupling"):
                    errors[key] = float(np.max(np.abs(left[key]-right[key])))
                for key in ("model", "gradient"):
                    errors[key] = max(float(np.max(np.abs(left[key][k]-right[key][k]))) for k in left[key])
                errors["prediction"] = max(float(np.max(np.abs(x-y))) for x,y in zip(left["predictions"],right["predictions"]))
                manifest["equivalence_max_abs_error"] = errors
                tolerances = {"c1":1e-12, "c2":1e-12, "coupling":1e-10, "model":1e-6, "gradient":1e-6, "prediction":1e-6}
                if any(errors[key] > tolerances[key] for key in errors):
                    raise AssertionError(f"Storage equivalence exceeded tolerances: {errors}")
            manifest.update(status="completed", resources=guard.snapshot(),
                            phase_timings=guard.phase_timings, scratch_removed_on_exit=True)
        dump(a.output_dir / "manifest.json", manifest)
        print(json.dumps({"status": manifest["status"], "output": str(a.output_dir),
                          "resources": manifest["resources"],
                          "equivalence": manifest.get("equivalence_max_abs_error")}), flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=repr(error), resources=guard.snapshot())
        dump(a.output_dir / "manifest.json", manifest)
        raise
    finally:
        guard.stop.set()
        guard.thread.join(timeout=2)


def check_gradient(gradient):
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite gradient")
    return gradient


if __name__ == "__main__":
    main()
