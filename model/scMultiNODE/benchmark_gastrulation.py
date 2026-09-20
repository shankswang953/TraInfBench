#!/usr/bin/env python
"""Full-observed-data scMultiNODE full/LOO adapters; native objective unchanged.

Use --check first: upstream QGW has large quadratic allocations. A smoke run is
explicitly a different, subsampled engineering test, never the full benchmark.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time
import sys

from run_gastrulation import ROOT, TIMES, STAGES, UPSTREAM_COMMIT, make_model, sha256
from dataset_protocols import DatasetProtocol, GASTRULATION
import numpy as np
import torch

SPLITS = GASTRULATION.splits


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def load_inputs(data_dir, scenario, seed=0, smoke=False, *, dataset: DatasetProtocol = GASTRULATION):
    """Only observed matrices enter training; frozen upstream representations retained."""
    selected = dataset.splits[scenario]
    training, initial, provenance, indices = {}, None, {}, {}
    for modality, matrix_name, norm_name, dimension in dataset.modalities:
        source, norm = data_dir / matrix_name, data_dir / norm_name
        scale = float(torch.load(norm, map_location="cpu", weights_only=False)["scale"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid frozen scale")
        arrays, counts, stage_indices = [], [], {}
        with np.load(source) as data:
            # Read shape/count only for withheld stages; no held-out tensor is
            # supplied to the AE, correspondence, fusion, or dynamics APIs.
            for i, key in enumerate(dataset.keys):
                counts.append(len(data[key]))
                if i not in selected:
                    continue
                normed = np.asarray(data[key], dtype=np.float32) / scale
                if normed.ndim != 2 or normed.shape[1] != dimension:
                    raise ValueError(f"Unexpected {modality} representation shape: {normed.shape}")
                if not np.isfinite(normed).all():
                    raise ValueError("Nonfinite observed input")
                if modality == "rna" and i == 0:
                    initial = torch.from_numpy(normed.copy())
                if smoke:
                    rng = np.random.default_rng(seed + i + (1000 if modality == "atac" else 0))
                    chosen = rng.choice(len(normed), min(64, len(normed)), replace=False)
                else:
                    chosen = np.arange(len(normed), dtype=np.int64)
                arrays.append(torch.from_numpy(normed[chosen]))
                stage_indices[key] = chosen
        training[modality] = arrays
        indices[modality] = stage_indices
        provenance[modality] = {"source": str(source.resolve()), "source_sha256": sha256(source),
            "normalization": str(norm.resolve()), "normalization_sha256": sha256(norm), "scale": scale,
            "stage_keys": list(dataset.keys),
            "dimension": arrays[0].shape[1], "full_counts": counts,
            "training_stage_indices": selected,
            "selected_counts": [len(x) for x in arrays],
            "training_cells": sum(len(x) for x in arrays),
            "minimum": min(float(x.min()) for x in arrays),
            "maximum": max(float(x.max()) for x in arrays)}
    if initial is None:
        raise ValueError("Initial RNA missing")
    return training, initial, provenance, indices


def memory_plan(n_rna, n_atac):
    """Conservative planning envelope, NOT a measured peak or a guarantee.

    Four float64 square matrices coexist during QGW compression. The official
    sparse <= threshold also creates dense-shaped masks/indices/explicit zeros.
    Provision 5x the four-matrix baseline for sparse assignment temporaries;
    round up to 32 GiB for large jobs. This is deliberately conservative.
    No replacement algorithm, reduced input pool or smaller dtype is used.
    """
    base = 16 * (n_rna ** 2 + n_atac ** 2) / 1024**3
    increment = 32 if base > 2 else 1
    provision = max(2., math.ceil(5 * base / increment) * increment)
    return {"four_dense_arrays_gib": base, "recommended_available_gib": provision,
            "basis": "5x four float64 quadratic arrays rounded up to 32 GiB for large jobs; not measured peak"}


def available_memory_gib():
    try:
        import psutil
        available = int(psutil.virtual_memory().available)
    except (ImportError, PermissionError, OSError):
        return None
    # Respect root and nested Linux process cgroups and each ancestor limit.
    roots = [(Path("/sys/fs/cgroup"), "memory.max", "memory.current"),
             (Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes", "memory.usage_in_bytes")]
    pairs = [(r / lim, r / use) for r, lim, use in roots]
    try:
        memberships = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        memberships = []
    for entry in memberships:
        pieces = entry.split(":", 2)
        if len(pieces) != 3:
            continue
        hierarchy, controllers, group = pieces
        if hierarchy == "0" and not controllers:
            root, lim, use = roots[0]
        elif "memory" in controllers.split(","):
            root, lim, use = roots[1]
        else:
            continue
        node = root / group.lstrip("/")
        while node == root or root in node.parents:
            pairs.append((node / lim, node / use))
            if node == root:
                break
            node = node.parent
    for max_path, current_path in pairs:
        try:
            limit, used = max_path.read_text().strip(), int(current_path.read_text())
            if limit.isdigit():
                available = min(available, max(0, int(limit) - used))
        except (OSError, ValueError):
            pass
    return available / 1024**3


def disk_memory_plan(n_rna, n_atac, scratch_root, max_rss_gib, min_available_gib):
    """Resource policy for opt-in exact disk storage, not a predicted RAM peak."""
    scratch_root = Path(scratch_root).resolve()
    if not scratch_root.is_dir():
        raise ValueError(f"Scratch root must be an existing directory: {scratch_root}")
    distance_gib = 8 * (n_rna**2 + n_atac**2) / 1024**3
    free_gib = shutil.disk_usage(scratch_root).free / 1024**3
    return {"storage": "disk", "recommended_available_gib": max_rss_gib + min_available_gib,
            "basis": "RSS watchdog budget plus system reserve; policy, not a measured peak or reservation",
            "max_rss_gib": max_rss_gib, "min_available_gib": min_available_gib,
            "distance_scratch_gib": distance_gib, "scratch_free_gib": free_gib,
            "scratch_required_gib": distance_gib + 2., "scratch_root": str(scratch_root),
            "scratch_passes": free_gib >= distance_gib + 2.}


def validate_distances_blockwise(values, row_block_size=256):
    """Never invoke implicit full-array conversion of a disk-backed distance."""
    for start in range(0, values.shape[0], row_block_size):
        if not np.isfinite(values[start:start + row_block_size, :]).all():
            raise FloatingPointError("Invalid correspondence/distance")


def summarize_qgw_audit(audit):
    """Validate basic plan feasibility, without changing returned solver plans.

    Inner iteration warnings retain upstream continuation semantics. Nonfinite,
    negative, zero-mass, or materially infeasible plans fail before fusion.
    Tests of marginal quality apply to raw probability plans, never max-normalized
    and thresholded correspondence weights.
    """
    solver = audit["qgw_solver_audit"]
    raw = audit["qgw_raw_coupling_audit"]
    if not solver or not raw:
        raise FloatingPointError("Missing QGW solver/feasibility audit")
    needs_review = []
    warnings_count = sum(x.get("warning_count", 0) for x in solver)
    if warnings_count:
        needs_review.append("solver_warnings")
    for record in solver + raw:
        if record.get("status") != "returned":
            raise FloatingPointError("QGW did not return a valid audited plan")
        coupling = record.get("coupling", {})
        fields = ("minimum", "mass", "source_mass", "target_mass",
                  "row_marginal_max_abs_error", "column_marginal_max_abs_error")
        if (coupling.get("audit_error") or coupling.get("finite") is not True
                or any(k not in coupling for k in fields)
                or not np.isfinite([coupling[k] for k in fields]).all()):
            raise FloatingPointError("Nonfinite/missing QGW feasibility audit")
        if coupling["minimum"] < -1e-10 or min(coupling[k] for k in ("mass", "source_mass", "target_mass")) <= 0:
            raise FloatingPointError("Negative/zero-mass QGW plan")
        if (abs(coupling["mass"] - coupling["source_mass"]) > 1e-6
                or abs(coupling["mass"] - coupling["target_mass"]) > 1e-6
                or max(coupling[k] for k in ("row_marginal_max_abs_error", "column_marginal_max_abs_error")) > 1e-6):
            raise FloatingPointError("QGW probability marginal/mass error exceeds validation tolerance 1e-6")
        log = record.get("log", {})
        if log.get("loss_finite") is False:
            raise FloatingPointError("Nonfinite compressed GW objective")
        for key in ("loss_initial", "loss_final", "gw_dist", "cost"):
            if log.get(key) is not None and not np.isfinite(log[key]):
                raise FloatingPointError("Nonfinite compressed GW objective")
        if log.get("warning") or log.get("result_code") not in (None, 1):
            needs_review.append("solver_status")
        if record.get("log_audit_error"):
            needs_review.append("log_audit_error")
    return {"quality_status": "needs_review" if needs_review else "no_flag_in_recorded_checks",
            "review_reasons": sorted(set(needs_review)), "solver_warning_count": warnings_count,
            "iteration_limit_warning_count": sum(x.get("iteration_limit_warning_count", 0) for x in solver),
            "feasibility_max_abs_tolerance": 1e-6, "negative_entry_tolerance": 1e-10,
            "policy": "Official solver parameters and warning continuation unchanged; no convergence guarantee"}


def checkpoint_payload(model, config, scenario, phase, iteration, *, dataset: DatasetProtocol = GASTRULATION):
    return {"model_state_dict": model.state_dict(), "rna_dim": model.n_genes,
            "atac_dim": model.n_peaks, "latent_dim": model.latent_dim,
            "upstream_commit": UPSTREAM_COMMIT, "decoder_output": "identity",
            "training_times": [dataset.times[i] for i in dataset.splits[scenario]], "scenario": scenario,
            "dataset": dataset.name, "all_times": list(dataset.times), "stage_names": list(dataset.stages),
            "phase": phase, "iteration": iteration, "config": config,
            "purpose": "inference checkpoint; optimizer resume is not implemented"}


def recorder_class(model, config, scenario, output, guard=None, *, dataset: DatasetProtocol = GASTRULATION):
    """Log official rounded loss display and save AFTER completed optimizer steps."""
    fields = ["phase", "iteration", "elapsed_seconds", "RNA Loss", "ATAC Loss", "Loss",
              "RNA recon", "ATAC recon", "RNA align", "ATAC align", "RNA Reg", "ATAC Reg"]
    handle = (output / "losses_display_precision.csv").open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    handle.flush()
    started = time.monotonic()
    class Recorder:
        def __init__(self, iterable, desc="", **kwargs):
            self.iterable, self.desc, self.step = iterable, desc, 0
        def __iter__(self):
            if guard is not None:
                guard.phase = self.desc
            print(self.desc, "START", flush=True)
            for i in self.iterable:
                self.step = i + 1
                yield i
                if "Dynamic Training" in self.desc and self.step % config["checkpoint_every"] == 0:
                    save("dynamics", self.step)
            phase = "dynamics" if "Dynamic" in self.desc else "fusion" if "Joint" in self.desc else "ae"
            save(phase, self.step)
            if guard is not None and phase == "ae":
                guard.phase = "QGW distances and matching"
        def set_postfix(self, values):
            values = {k: float(v) for k, v in values.items()}
            if not np.isfinite(list(values.values())).all():
                raise FloatingPointError(f"Nonfinite loss: {self.desc} {values}")
            writer.writerow({"phase": self.desc, "iteration": self.step,
                             "elapsed_seconds": time.monotonic()-started, **values})
            handle.flush()
            if self.step == 1 or self.step % 100 == 0:
                print(self.desc, self.step, values, flush=True)
    def save(phase, step):
        dest = output / "checkpoints" / f"{phase}_{step:06d}.pt"
        torch.save(checkpoint_payload(model, config, scenario, phase, step, dataset=dataset), dest)
    return Recorder, handle


def main(*, dataset: DatasetProtocol = GASTRULATION, argv=None, input_auditor=None):
    times = np.asarray(dataset.times, dtype=np.float32)
    end_time = float(times[-1])
    splits = dataset.splits
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=ROOT.parent / "TraInf/TraInf" / dataset.data_subdirectory)
    p.add_argument("--upstream", type=Path, default=ROOT / "external/scMultiNODE")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--expected-scenario", choices=list(splits), help="Launcher/config consistency check")
    p.add_argument("--check", action="store_true", help="Validate data/config/source and report memory; do not train")
    p.add_argument("--smoke", action="store_true", help="64 observed cells/time, 2 updates per stage, no formal result")
    p.add_argument("--memory-budget-gib", type=float,
                   help="Explicit available allocation if automatic discovery fails; also caps detected availability")
    p.add_argument("--qgw-storage", choices=("official", "disk"), default="official",
                   help="Opt-in storage-equivalent disk/sparse implementation; numerical settings unchanged")
    p.add_argument("--row-block-size", type=int, default=256)
    p.add_argument("--scratch-root", type=Path, default=Path("/private/tmp"))
    p.add_argument("--max-rss-gib", type=float, default=10.)
    p.add_argument("--min-available-gib", type=float, default=6.)
    a = p.parse_args(argv)
    if a.row_block_size < 1 or any(not np.isfinite(x) or x <= 0 for x in (a.max_rss_gib, a.min_available_gib)):
        raise ValueError("Storage block size and resource budgets must be positive")
    config = json.loads(a.config.read_text())
    scenario = config["scenario"]
    if scenario not in splits:
        raise ValueError(scenario)
    if a.expected_scenario is not None and scenario != a.expected_scenario:
        raise ValueError("Launcher scenario does not match configuration")
    if config.get("dataset", dataset.name) != dataset.name:
        raise ValueError("Configuration belongs to a different dataset")
    if config["input_space"] != dataset.input_space or config["solver"] != "official_euler":
        raise ValueError("Unexpected benchmark protocol")
    if a.smoke:
        config.update(ae_iters=2, fusion_iters=2, iters=2, batch_size=32, checkpoint_every=1)
    for key in ("ae_iters", "fusion_iters", "iters", "batch_size", "ae_batch_size", "fusion_batch_size",
                "latent_dim", "n_neighbors", "threads", "checkpoint_every", "inference_batch_size"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"Invalid positive integer: {key}")
    for key in ("lr", "trajectory_dt"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Invalid coefficient: {key}")
    for key in ("align_coeff", "dyn_reg_coeff"):
        if not np.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"Invalid nonnegative coefficient: {key}")
    commit = subprocess.check_output(["git", "-C", str(a.upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(a.upstream), "status", "--porcelain"], text=True).strip()
    if commit != UPSTREAM_COMMIT or dirty:
        raise RuntimeError("Official checkout must be clean and pinned")
    torch.set_num_threads(config["threads"])
    training, initial, provenance, indices = load_inputs(a.data_dir, scenario, config["seed"], a.smoke, dataset=dataset)
    reference_audit = (None if input_auditor is None else
                       input_auditor(a.data_dir, training, initial, provenance, indices))
    n_rna, n_atac = (sum(map(len, training[m])) for m in ("rna", "atac"))
    for n in (n_rna, n_atac):
        if min(n, int(.1*n)) < 1 or max(config["ae_batch_size"], config["fusion_batch_size"], config["n_neighbors"]+1) > n:
            raise ValueError("Too few training cells for official no-replacement stages")
    plan = (disk_memory_plan(n_rna, n_atac, a.scratch_root, a.max_rss_gib, a.min_available_gib)
            if a.qgw_storage == "disk" else memory_plan(n_rna, n_atac))
    detected = available_memory_gib()
    available = detected
    if a.memory_budget_gib is not None:
        if not np.isfinite(a.memory_budget_gib) or a.memory_budget_gib <= 0:
            raise ValueError("Memory allocation must be positive")
        available = a.memory_budget_gib if detected is None else min(a.memory_budget_gib, detected)
    plan.update(detected_available_gib=detected, effective_available_gib=available,
                passes=(available is not None and available >= plan["recommended_available_gib"]
                        and plan.get("scratch_passes", True)))
    withheld = [i for i in range(len(times)) if i not in splits[scenario]]
    report = {"dataset": dataset.name, "scenario": scenario, "engineering_smoke": a.smoke, "config": config,
        "training_stage_indices": splits[scenario], "training_times": times[splits[scenario]].tolist(),
        "heldout_stage": None if not withheld else dataset.stages[withheld[0]],
        "training_cells_per_modality": [n_rna, n_atac], "inference_initial_count": len(initial),
        "upstream_commit": commit, "upstream_unmodified": True, "memory": plan,
        "qgw_storage": a.qgw_storage, "runtime_storage_adaptation": a.qgw_storage == "disk",
        "fixed_internal_target_samples": 200, "inputs": provenance,
        "representation_policy": dataset.representation_policy}
    if reference_audit is not None:
        report["reference_input_audit"] = reference_audit
    print(json.dumps(report, indent=2), flush=True)
    if a.check:
        return
    if not plan["passes"]:
        raise MemoryError(f"Insufficient/unknown resources for {a.qgw_storage} QGW storage. Check the memory/scratch report; no automatic subsampling or algorithm changes.")
    if a.output_dir is None:
        raise ValueError("--output-dir required for training")
    if a.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {a.output_dir}")
    dt = config["trajectory_dt"]
    if not np.isclose(round(end_time/dt)*dt, end_time):
        raise ValueError(f"trajectory_dt must divide {end_time}")
    a.output_dir.mkdir(parents=True)
    (a.output_dir / "checkpoints").mkdir()
    source_names = ["benchmark_gastrulation.py", "dataset_protocols.py", "run_gastrulation.py", "trajectory_export.py"]
    if dataset.name == "moscot_pancreas":
        source_names.append("benchmark_pancreas.py")
    elif dataset.name == "human_cerebral_7time_d4_d21_no_d16":
        source_names += ["benchmark_human_cerebral.py", "human_cerebral_input_audit.py"]
    elif dataset.name == "palate":
        source_names += ["benchmark_palate.py", "palate_input_audit.py"]
    if a.qgw_storage == "disk":
        source_names += ["memory_optimized_train.py", "memory_distances.py", "memory_sparse_qgw.py", "resource_guard.py"]
    report.update(status="training", time=times.tolist(), stage_names=list(dataset.stages),
                  config_source=str(a.config.resolve()), config_sha256=sha256(a.config),
                  adapter_source_sha256={name: sha256(Path(__file__).parent / name) for name in source_names})
    write_json(a.output_dir / "manifest.json", report)
    np.savez_compressed(a.output_dir / "training_indices.npz", **{
        f"{m}_{stage}": ix for m in indices for stage, ix in indices[m].items()})
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    sys.path.insert(0, str(a.upstream.resolve()))
    from optim import running
    model = make_model(running, initial.shape[1], training["atac"][0].shape[1], config["latent_dim"])
    guard, adaptation, distances, scratch = None, None, None, None
    resources = ExitStack()
    if a.qgw_storage == "disk":
        from resource_guard import ResourceGuard
        guard = ResourceGuard(a.output_dir, max_rss_gib=a.max_rss_gib,
                              min_available_gib=a.min_available_gib,
                              available_memory=available_memory_gib)
    recorder, log_handle = recorder_class(model, config, scenario, a.output_dir, guard=guard, dataset=dataset)
    old_tqdm, running.tqdm = running.tqdm, recorder
    def finite_gradient(gradient):
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite gradient; no clipping applied")
        return gradient
    hooks = [v.register_hook(finite_gradient) for v in model.parameters()]
    started = time.monotonic()
    try:
        trainer = running.scMultiNODETrain
        if a.qgw_storage == "disk":
            from memory_optimized_train import make_memory_trainer
            scratch = resources.enter_context(tempfile.TemporaryDirectory(
                prefix="scmultinode-distance-", dir=a.scratch_root.resolve()))
            write_json(a.output_dir / "scratch.json", {"path": scratch, "temporary_only": True})
            def record_qgw(audit):
                # Preserve the evidence even when validation refuses the plan.
                write_json(a.output_dir / "qgw_audit.json", audit)
                summary = summarize_qgw_audit(audit)
                report["qgw_quality"] = summary
                write_json(a.output_dir / "qgw_audit.json", audit | {"summary": summary})
                write_json(a.output_dir / "manifest.json", report)
                print("QGW audit:", json.dumps(summary), flush=True)
                if summary["quality_status"] == "needs_review":
                    print("WARNING: QGW solver reported a warning; continuing as upstream does. Completion does not establish convergence.", flush=True)
            trainer, distances, adaptation = make_memory_trainer(
                running, scratch, row_block_size=a.row_block_size, progress=recorder,
                qgw_callback=record_qgw)
            guard.start()
        model, ae, fusion, coupling, c1, c2 = trainer(
            training["rna"], training["atac"], torch.from_numpy(times[splits[scenario]]),
            torch.from_numpy(times[splits[scenario]]), model, config["iters"], config["batch_size"], config["lr"],
            ae_iters=config["ae_iters"], ae_lr=config["lr"], ae_batch_size=config["ae_batch_size"],
            fusion_iters=config["fusion_iters"], fusion_lr=config["lr"], fusion_batch_size=config["fusion_batch_size"],
            align_coeff=config["align_coeff"], dyn_reg_coeff=config["dyn_reg_coeff"], train_all=True,
            n_neighbors=config["n_neighbors"], qgw_sample_ratio=.1, gw_type="gw", epsilon=.01)
        if guard is not None:
            guard.phase = "validating distances and exporting all initial cells"
        for values in (c1, c2):
            validate_distances_blockwise(values, a.row_block_size)
        if not np.isfinite(coupling.data).all() or coupling.count_nonzero() == 0:
            raise FloatingPointError("Invalid/empty correspondence")
        model.eval()
        torch.save(checkpoint_payload(model, config, scenario, "dynamics", config["iters"], dataset=dataset), a.output_dir / "model.pt")
        del ae, fusion, c1, c2, values
        # Only compact the output artifact after training; no algorithm change.
        coupling.eliminate_zeros()
        from scipy.sparse import save_npz
        save_npz(a.output_dir / "qgw_correspondence.npz", coupling)
        report["qgw_nonzero_entries"] = int(coupling.count_nonzero())
        del coupling
        from trajectory_export import export_trajectories, query_time_grid
        query = query_time_grid(times, dt)
        report["trajectory_audit"] = export_trajectories(model, initial, np.arange(len(initial)),
            times[splits[scenario]], query, a.output_dir / "trajectories.npz",
            batch_size=config["inference_batch_size"])
        report.update(status="smoke_completed" if a.smoke else "completed",
                      elapsed_seconds=time.monotonic()-started,
                      checkpoint_selection="Final predeclared iteration; no heldout-based selection",
                      versions={k: importlib.metadata.version(k) for k in ("torch", "torchdiffeq", "numpy", "scipy", "POT", "geomloss")},
                      adaptations=["Decoder output Identity for signed embeddings", "Native Euler dense interpolation for query-grid invariance", "Logging/checkpoint hooks only"])
        if adaptation is not None:
            report["adaptations"].append("Opt-in disk-backed distances and sparse QGW storage, unchanged numerical parameters")
        write_json(a.output_dir / "manifest.json", report)
    except BaseException as error:
        report.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-started)
        write_json(a.output_dir / "manifest.json", report)
        raise
    finally:
        running.tqdm = old_tqdm
        log_handle.close()
        for h in hooks:
            h.remove()
        try:
            resources.close()
        finally:
            if guard is not None:
                guard.close()
                report["resources"] = guard.snapshot()
            if adaptation is not None:
                report["storage_adaptation"] = adaptation
                report["distance_records"] = distances.records
                write_json(a.output_dir / "qgw_audit.json", {
                    "qgw_solver_audit": adaptation["qgw_solver_audit"],
                    "qgw_raw_coupling_audit": adaptation["qgw_raw_coupling_audit"],
                    "qgw_timing": adaptation["qgw_timing"],
                    "summary": report.get("qgw_quality", {"quality_status": "unverified"})})
            if scratch is not None:
                report["scratch_removed_on_exit"] = not Path(scratch).exists()
            write_json(a.output_dir / "manifest.json", report)
    print("Completed", a.output_dir, flush=True)


if __name__ == "__main__":
    main()
