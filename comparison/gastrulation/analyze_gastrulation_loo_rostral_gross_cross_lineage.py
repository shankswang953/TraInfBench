#!/usr/bin/env python
"""Held-out endpoint gross cross-lineage leakage from E7.5 Rostral cells."""

from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import json
import os
from pathlib import Path
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch
from torchdiffeq import odeint


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

from evaluate_gastrulation_full_cmcc import _load_references  # noqa: E402
from evaluate_terminal_push import _load_trajectorynet_model  # noqa: E402
from evaluate_gastrulation_flow_methods_normalized import (  # noqa: E402
    _load_mioflow_bundle,
    _mioflow_model_path_to_xnorm,
    _mioflow_xnorm_to_model_space,
)


OUTPUT_DIR = ROOT / "results/gastrulation_loo_rostral_gross_cross_lineage"
SOURCE_H5AD = ROOT / "data/gastrulation_rna_loo_time1_cytobridge.h5ad"
RNA_ONLY_DIR = (
    ROOT / "results/gastrulation_loo_shared_t_trajectorynet_base_rna_only_floor"
)
SCENARIOS = {
    "E8.0": {
        "stage_index": 1,
        "prediction": (
            ROOT
            / "results/gastrulation_loo_time1_coverage_validity"
            / "loo_time1_shared_t_atac_predictions.npz"
        ),
        "rna_only": RNA_ONLY_DIR / "loo_time1_rna_only_predictions.npz",
        "trajectorynet_checkpoint": (
            ROOT
            / "results/trajectorynet_gastrulation_loo_time1_20000"
            / "checkpt-20000.pth"
        ),
        "trajectorynet_clock": "physical-label LOO clock",
        "mioflow_gaga10_checkpoint": (
            ROOT
            / "results/mioflow_gastrulation_loo_time1_pca_gaga10_n1024_20000"
            / "model.pt"
        ),
        "mioflow_heldout_model_time": 0.5,
    },
    "E8.5": {
        "stage_index": 2,
        "prediction": (
            ROOT
            / "results/gastrulation_loo_time2_shared_t_atac"
            / "loo_time2_shared_t_predictions.npz"
        ),
        "rna_only": RNA_ONLY_DIR / "loo_time2_rna_only_predictions.npz",
        "trajectorynet_checkpoint": (
            ROOT
            / "results/trajectorynet_gastrulation_loo_time2_20000"
            / "checkpt-20000.pth"
        ),
        "trajectorynet_clock": "rank-encoded LOO clock",
        "mioflow_gaga10_checkpoint": (
            ROOT
            / "results/mioflow_gastrulation_loo_time2_pca_gaga10_n1024_20000"
            / "model.pt"
        ),
        "mioflow_heldout_model_time": 5.0 / 3.0,
    },
}

SOURCE_CELLTYPE = "Rostral neurectoderm"
K_NEIGHBORS = 5
MESODERM = {
    "Allantois",
    "Cardiomyocytes",
    "Caudal Mesoderm",
    "Intermediate mesoderm",
    "Mesenchyme",
    "Mixed mesoderm",
    "Nascent mesoderm",
    "Notochord",
    "Paraxial mesoderm",
    "Pharyngeal mesoderm",
    "Somitic mesoderm",
}
BLOOD_ERYTHROID = {
    "Haematoendothelial progenitors",
    "Endothelium",
    "Blood progenitors 1",
    "Blood progenitors 2",
    "Erythroid1",
    "Erythroid2",
    "Erythroid3",
}
ENDODERM = {"Def. endoderm", "Gut"}
GROSS_CROSS_LINEAGE = MESODERM | BLOOD_ERYTHROID | ENDODERM


def mioflow_gaga10_prediction(
    x0_norm: np.ndarray,
    *,
    checkpoint: Path,
    heldout_model_time: float,
    rna_scale: float,
    batch_size: int = 1024,
    steps: int = 20,
) -> tuple[np.ndarray, dict[str, object]]:
    """Push an observed E7.5 cohort through a frozen MIOFlow GAGA10 model."""

    device = torch.device("cpu")
    bundle = _load_mioflow_bundle(checkpoint, device)
    if not bundle.use_gaga or len(bundle.mean_vals) != 10:
        raise ValueError(f"Expected a GAGA10 checkpoint, found {checkpoint}")
    if not np.allclose(bundle.model_times, np.asarray([0.0, 1.0, 2.0])):
        raise ValueError(
            f"Unexpected MIOFlow model clock: {bundle.model_times.tolist()}"
        )

    z0 = _mioflow_xnorm_to_model_space(
        x0_norm, bundle, rna_scale, device, batch_size
    )
    times = torch.linspace(
        0.0,
        float(heldout_model_time),
        steps + 1,
        dtype=torch.float32,
        device=device,
    )
    paths: list[np.ndarray] = []
    for start in range(0, len(z0), batch_size):
        batch = torch.as_tensor(
            z0[start : start + batch_size], dtype=torch.float32, device=device
        )
        if hasattr(bundle.model, "reset_momentum"):
            bundle.model.reset_momentum()
        with torch.no_grad():
            path = odeint(bundle.model, batch, times)
        paths.append(path.detach().cpu().numpy())
    path_model = np.concatenate(paths, axis=1).astype(np.float32, copy=False)
    path_norm = _mioflow_model_path_to_xnorm(
        path_model, bundle, rna_scale, device, batch_size
    )
    audit = {
        "checkpoint": str(checkpoint.resolve()),
        "gaga_model": str(Path(bundle.gaga_model.model_path).resolve())
        if hasattr(bundle.gaga_model, "model_path")
        else "loaded from checkpoint metadata",
        "model_space": "GAGA 10D",
        "source": "observed E7.5 Rostral neurectoderm cells",
        "heldout_model_time": float(heldout_model_time),
        "integration_steps": int(steps),
        "decoded_evaluation_space": "shared normalized 50D RNA PCA",
    }
    return path_norm[-1].astype(np.float32, copy=False), audit


def integrate_density_direction(
    model: torch.nn.Module,
    values_raw: np.ndarray,
    *,
    start: float,
    stop: float,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Integrate one frozen TrajectoryNet density-direction segment."""

    pieces: list[np.ndarray] = []
    for batch_start in range(0, len(values_raw), batch_size):
        state = torch.as_tensor(
            values_raw[batch_start : batch_start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        zero = torch.zeros((len(state), 1), dtype=state.dtype, device=device)
        with torch.no_grad():
            result, _ = model(
                state,
                zero,
                integration_times=torch.tensor(
                    [start, stop], dtype=torch.float32, device=device
                ),
                reverse=False,
            )
        pieces.append(result.detach().cpu().numpy())
    return np.concatenate(pieces).astype(np.float32, copy=False)


def trajectorynet_terminal_rostral_prediction(
    *,
    stage: str,
    checkpoint: Path,
    rna_references: list[np.ndarray],
    labels_by_stage: list[np.ndarray],
    rna_scale: float,
    output_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Select Rostral particles at reconstructed E7.5, then read held-out state."""

    device = torch.device("cpu")
    terminal_norm = np.asarray(rna_references[3], dtype=np.float32)
    terminal_raw = terminal_norm * float(rna_scale)
    model, model_args = _load_trajectorynet_model(
        checkpoint, terminal_raw.shape[1], device, "rk4", 0.1
    )
    if not np.isclose(float(model_args.time_scale), 0.5):
        raise ValueError(
            f"Expected TrajectoryNet time_scale=0.5, got {model_args.time_scale}"
        )

    if stage == "E8.0":
        # Physical-label checkpoint trained on E7.5, E8.5 and E8.75.
        e85_raw = integrate_density_direction(
            model, terminal_raw, start=1.5, stop=2.0, device=device
        )
        heldout_raw = integrate_density_direction(
            model, e85_raw, start=1.0, stop=1.25, device=device
        )
        initial_raw = integrate_density_direction(
            model, e85_raw, start=1.0, stop=1.5, device=device
        )
        schedule = {
            "terminal_to_training_neighbor": [[1.5, 2.0]],
            "training_neighbor_to_heldout": [[1.0, 1.25]],
            "training_neighbor_to_initial": [[1.0, 1.5]],
        }
    elif stage == "E8.5":
        # Rank checkpoint trained on E7.5, E8.0 and E8.75.
        heldout_raw = integrate_density_direction(
            model, terminal_raw, start=1.0, stop=7.0 / 6.0, device=device
        )
        e80_raw = integrate_density_direction(
            model, terminal_raw, start=1.0, stop=1.5, device=device
        )
        initial_raw = integrate_density_direction(
            model, e80_raw, start=0.5, stop=1.0, device=device
        )
        schedule = {
            "terminal_to_heldout": [[1.0, 7.0 / 6.0]],
            "terminal_to_training_neighbor": [[1.0, 1.5]],
            "training_neighbor_to_initial": [[0.5, 1.0]],
        }
    else:
        raise ValueError(f"Unsupported held-out stage: {stage}")

    initial_norm = initial_raw / float(rna_scale)
    heldout_norm = heldout_raw / float(rna_scale)
    initial_labels = np.asarray(
        [clean_label(value) for value in labels_by_stage[0]], dtype=str
    )
    initial_neighbors = NearestNeighbors(
        n_neighbors=K_NEIGHBORS,
        algorithm="brute",
        metric="euclidean",
        n_jobs=-1,
    ).fit(np.asarray(rna_references[0], dtype=np.float32))
    initial_indices = initial_neighbors.kneighbors(
        initial_norm, n_neighbors=K_NEIGHBORS, return_distance=False
    )
    initial_neighbor_labels = initial_labels[initial_indices]
    rostral_vote_fraction = np.mean(
        initial_neighbor_labels == SOURCE_CELLTYPE, axis=1
    )
    rostral_mask = rostral_vote_fraction >= 0.5
    if int(rostral_mask.sum()) < 20:
        raise RuntimeError(
            f"{stage}: only {int(rostral_mask.sum())} terminal particles "
            "reconstruct as E7.5 Rostral under 5-NN majority voting"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        terminal_rna_norm=terminal_norm,
        reconstructed_initial_rna_norm=initial_norm,
        heldout_prediction_rna_norm=heldout_norm,
        initial_neighbor_labels=initial_neighbor_labels,
        rostral_vote_fraction=rostral_vote_fraction,
        rostral_mask=rostral_mask,
    )
    audit = {
        "checkpoint": str(checkpoint.resolve()),
        "terminal_particles": int(len(terminal_norm)),
        "reconstructed_initial_rostral_particles": int(rostral_mask.sum()),
        "reconstructed_initial_rostral_fraction": float(rostral_mask.mean()),
        "initial_classifier": (
            "5-NN majority vote against real E7.5 RNA reference; "
            "Rostral requires at least 3 of 5 neighbours"
        ),
        "schedule": schedule,
        "prediction_cache": str(output_path.resolve()),
    }
    return heldout_norm[rostral_mask], rostral_mask, audit


def clean_label(value: object) -> str:
    text = str(value)
    if text in {"Unknown", "unknown", "nan"}:
        return "Unannotated"
    if text == "Definitive endoderm":
        return "Def. endoderm"
    if text == "Caudal mesoderm":
        return "Caudal Mesoderm"
    return text


def weights(log_mass: np.ndarray, native: bool) -> np.ndarray:
    values = np.asarray(log_mass, dtype=np.float64).reshape(-1)
    if not native:
        return np.full(len(values), 1.0 / len(values), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Native log mass contains non-finite values")
    result = np.exp(values - values.max())
    return result / result.sum()


def summarize_prediction(
    *,
    stage: str,
    method: str,
    prediction: np.ndarray,
    log_mass: np.ndarray,
    native_mass: bool,
    source_mask: np.ndarray,
    neighbors: NearestNeighbors,
    target_labels: np.ndarray,
    provenance: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if prediction.shape[0] != len(source_mask):
        raise ValueError(
            f"{stage} {method}: prediction/source mismatch "
            f"{prediction.shape[0]} != {len(source_mask)}"
        )
    prediction = np.asarray(prediction[source_mask], dtype=np.float32)
    local_weights = weights(np.asarray(log_mass)[source_mask], native_mass)
    indices = neighbors.kneighbors(
        prediction, n_neighbors=K_NEIGHBORS, return_distance=False
    )
    labels = target_labels[indices]

    def fraction(celltypes: set[str]) -> float:
        particle_scores = np.mean(np.isin(labels, list(celltypes)), axis=1)
        return float(np.sum(local_weights * particle_scores))

    summary = {
        "stage": stage,
        "method": method,
        "source_celltype": SOURCE_CELLTYPE,
        "source_n": int(source_mask.sum()),
        "k": K_NEIGHBORS,
        "weighting": "native mass" if native_mass else "uniform",
        "gross_cross_lineage_leakage_fraction": fraction(GROSS_CROSS_LINEAGE),
        "mesoderm_leakage_fraction": fraction(MESODERM),
        "blood_erythroid_leakage_fraction": fraction(BLOOD_ERYTHROID),
        "endoderm_leakage_fraction": fraction(ENDODERM),
        "provenance": provenance,
    }

    composition_rows: list[dict[str, object]] = []
    for celltype in sorted(set(target_labels)):
        composition_rows.append(
            {
                "stage": stage,
                "method": method,
                "predicted_celltype": celltype,
                "predicted_fraction": fraction({celltype}),
            }
        )
    return summary, composition_rows


def main() -> None:
    source_adata = ad.read_h5ad(SOURCE_H5AD, backed="r")
    source_time = pd.to_numeric(
        source_adata.obs["time_point_processed"], errors="raise"
    ).to_numpy(float)
    source_obs = source_adata.obs.loc[np.isclose(source_time, 0.0)].copy()
    if len(source_obs) != 9018:
        raise ValueError(f"Expected 9018 E7.5 source cells, got {len(source_obs)}")
    source_labels = np.asarray(
        [clean_label(value) for value in source_obs["celltype"]], dtype=str
    )
    source_mask = source_labels == SOURCE_CELLTYPE
    if int(source_mask.sum()) != 1568:
        raise ValueError(
            f"Expected 1568 E7.5 Rostral cells, got {source_mask.sum()}"
        )

    rna_references, _, labels_by_stage, rna_scale, _ = _load_references()
    reference_initial_labels = np.asarray(
        [clean_label(value) for value in labels_by_stage[0]], dtype=str
    )
    if not np.array_equal(reference_initial_labels, source_labels):
        raise ValueError(
            "E7.5 RNA reference order does not match the source H5AD labels"
        )
    summary_rows: list[dict[str, object]] = []
    composition_rows: list[dict[str, object]] = []
    trajectorynet_audits: dict[str, object] = {}
    mioflow_audits: dict[str, object] = {}

    for stage, spec in SCENARIOS.items():
        stage_index = int(spec["stage_index"])
        reference = np.asarray(rna_references[stage_index], dtype=np.float32)
        target_labels = np.asarray(
            [clean_label(value) for value in labels_by_stage[stage_index]],
            dtype=str,
        )
        neighbors = NearestNeighbors(
            n_neighbors=K_NEIGHBORS,
            algorithm="brute",
            metric="euclidean",
            n_jobs=-1,
        ).fit(reference)

        cache_path = Path(spec["prediction"])
        with np.load(cache_path, allow_pickle=True) as cache:
            methods = np.asarray(cache["methods"], dtype=str)
            for index, raw_method in enumerate(methods):
                if raw_method in {"TrajectoryNet 20k", "MIOFlow 20k"}:
                    # These rows are regenerated below: TrajectoryNet from the
                    # terminal-backward protocol and MIOFlow from GAGA10D.
                    continue
                if raw_method.startswith("BSOT C_y="):
                    display = raw_method.replace("BSOT", "COATI bal.")
                elif raw_method.startswith("USOT C_y="):
                    display = raw_method.replace("USOT", "COATI unbal.")
                elif raw_method == "CytoBridge balanced 20k":
                    display = "CytoBridge bal."
                elif raw_method == "CytoBridge unbalanced 20k":
                    display = "CytoBridge unbal."
                elif raw_method == "MIOFlow 20k":
                    display = "MIOFlow"
                elif raw_method == "TIGON 20k":
                    display = "TIGON"
                else:
                    display = raw_method
                summary, composition = summarize_prediction(
                    stage=stage,
                    method=display,
                    prediction=np.asarray(cache["rna_predictions"][index]),
                    log_mass=np.asarray(cache["native_log_masses"][index]),
                    native_mass=bool(cache["has_native_mass"][index]),
                    source_mask=source_mask,
                    neighbors=neighbors,
                    target_labels=target_labels,
                    provenance=str(cache_path),
                )
                summary_rows.append(summary)
                composition_rows.extend(composition)

        terminal_cache = (
            OUTPUT_DIR
            / f"trajectorynet_terminal_backward_{stage.lower().replace('.', '')}.npz"
        )
        trajectorynet_prediction, trajectorynet_mask, trajectorynet_audit = (
            trajectorynet_terminal_rostral_prediction(
                stage=stage,
                checkpoint=Path(spec["trajectorynet_checkpoint"]),
                rna_references=rna_references,
                labels_by_stage=labels_by_stage,
                rna_scale=rna_scale,
                output_path=terminal_cache,
            )
        )
        trajectorynet_audits[stage] = trajectorynet_audit
        local_mask = np.ones(len(trajectorynet_prediction), dtype=bool)
        trajectorynet_summary, trajectorynet_composition = summarize_prediction(
            stage=stage,
            method="TrajectoryNet (terminal backward)",
            prediction=trajectorynet_prediction,
            log_mass=np.zeros(len(trajectorynet_prediction), dtype=np.float32),
            native_mass=False,
            source_mask=local_mask,
            neighbors=neighbors,
            target_labels=target_labels,
            provenance=str(terminal_cache),
        )
        trajectorynet_summary["terminal_particle_count"] = int(
            len(trajectorynet_mask)
        )
        summary_rows.append(trajectorynet_summary)
        composition_rows.extend(trajectorynet_composition)

        mioflow_prediction, mioflow_audit = mioflow_gaga10_prediction(
            np.asarray(rna_references[0], dtype=np.float32)[source_mask],
            checkpoint=Path(spec["mioflow_gaga10_checkpoint"]),
            heldout_model_time=float(spec["mioflow_heldout_model_time"]),
            rna_scale=rna_scale,
        )
        mioflow_audits[stage] = mioflow_audit
        mioflow_local_mask = np.ones(len(mioflow_prediction), dtype=bool)
        mioflow_summary, mioflow_composition = summarize_prediction(
            stage=stage,
            method="MIOFlow (GAGA10D)",
            prediction=mioflow_prediction,
            log_mass=np.zeros(len(mioflow_prediction), dtype=np.float32),
            native_mass=False,
            source_mask=mioflow_local_mask,
            neighbors=neighbors,
            target_labels=target_labels,
            provenance=str(Path(spec["mioflow_gaga10_checkpoint"])),
        )
        summary_rows.append(mioflow_summary)
        composition_rows.extend(mioflow_composition)

        rna_only_path = Path(spec["rna_only"])
        with np.load(rna_only_path, allow_pickle=True) as cache:
            for prefix, display, native in (
                ("BSOT RNA-only (no Sync)", "OT(RNA)", False),
                ("USOT RNA-only (no Sync)", "UOT(RNA)", True),
            ):
                rna_key = f"{prefix}__rna"
                if rna_key not in cache.files:
                    continue
                mass_key = f"{prefix}__log_mass"
                log_mass = (
                    np.asarray(cache[mass_key])
                    if mass_key in cache.files
                    else np.zeros(len(source_mask), dtype=np.float32)
                )
                summary, composition = summarize_prediction(
                    stage=stage,
                    method=display,
                    prediction=np.asarray(cache[rna_key]),
                    log_mass=log_mass,
                    native_mass=native,
                    source_mask=source_mask,
                    neighbors=neighbors,
                    target_labels=target_labels,
                    provenance=str(rna_only_path),
                )
                summary_rows.append(summary)
                composition_rows.extend(composition)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUTPUT_DIR / "loo_rostral_gross_cross_lineage_leakage_k5.csv",
        index=False,
    )
    pd.DataFrame(composition_rows).to_csv(
        OUTPUT_DIR / "loo_rostral_heldout_composition_k5.csv", index=False
    )
    manifest = {
        "analysis": (
            "E7.5 Rostral source-conditioned soft 5-NN gross cross-lineage "
            "leakage at held-out E8.0 and E8.5 endpoints"
        ),
        "source_n": int(source_mask.sum()),
        "reference": "real cells from the corresponding held-out RNA stage",
        "gross_cross_lineage": {
            "mesoderm": sorted(MESODERM),
            "blood_endothelium_erythroid": sorted(BLOOD_ERYTHROID),
            "endoderm": sorted(ENDODERM),
        },
        "balanced_weighting": "uniform",
        "unbalanced_weighting": "native mass normalized within Rostral cohort",
        "trajectorynet_note": (
            "Uses each LOO checkpoint in its trained density direction. Real "
            "E8.75 terminal particles are integrated backward to E7.5; the "
            "particles classified as Rostral at reconstructed E7.5 are then "
            "read at the held-out endpoint along the same backward trajectory."
        ),
        "trajectorynet": trajectorynet_audits,
        "mioflow_note": (
            "MIOFlow uses the LOO-specific 20k GAGA10D checkpoint, starts from "
            "the same 1,568 observed E7.5 Rostral cells as the other forward "
            "methods, and is decoded to shared normalized 50D RNA PCA before "
            "the held-out 5-NN readout."
        ),
        "mioflow": mioflow_audits,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    selected_methods = [
        "OT(RNA)",
        "COATI bal. C_y=0.3",
        "UOT(RNA)",
        "COATI unbal. C_y=0.3",
        "CytoBridge bal.",
        "CytoBridge unbal.",
        "MIOFlow (GAGA10D)",
        "TIGON",
        "TrajectoryNet (terminal backward)",
    ]
    selected = summary[summary["method"].isin(selected_methods)].copy()
    table = 100.0 * selected.pivot(
        index="method",
        columns="stage",
        values="gross_cross_lineage_leakage_fraction",
    )
    print("Gross cross-lineage leakage, soft 5-NN (%)")
    print(
        table.reindex(index=selected_methods, columns=["E8.0", "E8.5"])
        .to_string(float_format=lambda value: f"{value:.2f}")
    )
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
