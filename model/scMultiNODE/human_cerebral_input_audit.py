"""Read-only proof that human-cerebral inputs match the CytoBridge benchmark.

Pairing IDs are used only to audit fixture identity and row order. Neither these
IDs nor any biological labels are supplied to scMultiNODE's training objective.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from run_gastrulation import ROOT, sha256


AGES = (4, 7, 9, 11, 12, 18, 21)
KEYS = tuple(f"age_{age}" for age in AGES)
COUNTS = (501, 1309, 1630, 1560, 4100, 5842, 5721)
TIMES = (0.0, 0.3, 0.5, 0.7, 0.8, 1.4, 1.7)
SCALES = {"rna": 63.89521587795941, "atac": 34.16607592332852}
DIMENSIONS = {"rna": 30, "atac": 12}
ATAC_COMPONENTS = (2, 3, 4, 5, 6, 7, 10, 11, 12, 14, 22, 27)
EXCLUDED_KEYS = ("age_16", "age_26", "age_31", "age_61")
DEFAULT_REFERENCE_H5AD = (
    ROOT / "data/human_cerebral_7time_d4_d21_no_d16_rna_cytobridge_balanced.h5ad"
)
DEFAULT_TRAINED_H5AD = (
    ROOT / "results/cytobridge_human_cerebral_7time_d4_d21_no_d16_balanced_30000/adata.h5ad"
)
MATRIX_FILES = {"rna": "rna_pca30", "atac": "atac_lsi12"}
NORM_FILES = {
    "rna": "primal_norm_params.pt",
    "atac": "secondary_norm_params_lsi12.pt",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"human-cerebral input audit: {message}")


def _array(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _exact(actual, expected, name: str) -> dict:
    actual, expected = _array(actual), _array(expected)
    _require(actual.shape == expected.shape,
             f"{name} shape {actual.shape} differs from {expected.shape}")
    numeric = np.issubdtype(actual.dtype, np.number)
    if numeric:
        _require(np.isfinite(actual).all(), f"{name} has nonfinite values")
        difference = float(np.max(np.abs(actual.astype(np.float64)
                                        - expected.astype(np.float64)))) if actual.size else 0.0
    else:
        difference = None
    _require(np.array_equal(actual, expected),
             f"{name} is not exactly equal to the frozen reference; max_abs_difference={difference}")
    result = {"exact_equal": True, "shape": list(actual.shape)}
    if numeric:
        result["max_abs_difference"] = difference
    return result


def _npz_blocks(path: Path) -> list[np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        _require(tuple(archive.files) == KEYS, f"{path} does not have the seven keys in chronological order")
        return [np.asarray(archive[key]) for key in KEYS]


def _file_record(path: Path) -> dict:
    _require(path.is_file(), f"required file missing: {path}")
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _audit_h5ad(path: Path, normalized: np.ndarray, raw: np.ndarray,
                paired_ids: np.ndarray, row_keys: np.ndarray,
                row_ages: np.ndarray, row_times: np.ndarray) -> dict:
    record = _file_record(path)
    reference = ad.read_h5ad(path)
    _require(reference.shape == (sum(COUNTS), DIMENSIONS["rna"]),
             f"{path} is not the unreplicated 20,663-row balanced RNA fixture")
    _require("X_latent" in reference.obsm, f"{path} has no X_latent")
    _require(reference.obsm["X_latent"].dtype == np.float32,
             f"{path} X_latent is not float32")
    record["normalized_rna"] = _exact(reference.obsm["X_latent"], normalized, f"{path} X_latent")
    if "X_pca" in reference.obsm:
        record["normalized_rna_pca"] = _exact(reference.obsm["X_pca"], normalized, f"{path} X_pca")
    if "X_pca_raw" in reference.obsm:
        record["raw_rna"] = _exact(reference.obsm["X_pca_raw"], raw, f"{path} X_pca_raw")
    record["obs_names"] = _exact(np.asarray(reference.obs_names, dtype=str), paired_ids, f"{path} obs_names")
    required = {"paired_metacell_id", "time_key", "processed_age", "time_point_processed"}
    _require(required.issubset(reference.obs.columns), f"{path} lacks required identity/time columns")
    record["paired_ids"] = _exact(reference.obs["paired_metacell_id"].astype(str).to_numpy(), paired_ids, f"{path} paired IDs")
    record["time_keys"] = _exact(reference.obs["time_key"].astype(str).to_numpy(), row_keys, f"{path} row time keys")
    record["ages_days"] = _exact(reference.obs["processed_age"].to_numpy(), row_ages, f"{path} row ages")
    # The reference stores its physical clock at float32 precision.
    record["row_physical_times"] = _exact(reference.obs["time_point_processed"].to_numpy(), row_times, f"{path} row physical times")
    _require(tuple(reference.uns["time_keys"]) == KEYS, f"{path} time-key metadata differs")
    _require(tuple(reference.uns["excluded_time_keys"]) == EXCLUDED_KEYS, f"{path} excluded-day metadata differs")
    _exact(np.asarray(reference.uns["physical_time_values"], dtype=np.float32),
           np.asarray(TIMES, dtype=np.float32), f"{path} physical-time metadata")
    _require(float(reference.uns["rna_raw_to_model_scale"]) == SCALES["rna"],
             f"{path} RNA normalization scale differs")
    record["status"] = "verified"
    return record


def audit_inputs(data_dir, training, initial, provenance, indices, *,
                 reference_h5ad=DEFAULT_REFERENCE_H5AD,
                 trained_h5ad=DEFAULT_TRAINED_H5AD) -> dict:
    """Validate full/smoke ``load_inputs`` outputs before any model training.

    ``reference_h5ad`` is required. The previous training-output H5AD is checked
    when present, and its absence is explicitly reported; ``None`` disables that
    optional historical proof for isolated fixture tests. The seven-time source
    contract itself is fixed even when these reference paths are overridden.
    Every mismatch raises; no source data or existing results are modified.
    """
    data_dir = Path(data_dir).resolve()
    _require(set(training) == set(DIMENSIONS), "training must contain RNA and ATAC only")
    _require(set(provenance) == set(DIMENSIONS), "provenance must contain RNA and ATAC only")
    _require(set(indices) == set(DIMENSIONS), "indices must contain RNA and ATAC only")
    manifest_path = data_dir / "manifest.json"
    files = {"manifest": _file_record(manifest_path)}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(tuple(manifest["time_keys"]) == KEYS, "manifest time keys differ")
    _require(tuple(manifest["ages_days"]) == AGES, "manifest ages differ")
    _require(tuple(manifest["model_time_points"]) == TIMES, "manifest physical times differ")
    _require(tuple(manifest["excluded_time_keys"]) == EXCLUDED_KEYS, "manifest excluded days differ")
    _require(tuple(manifest["paired_counts"][key] for key in KEYS) == COUNTS, "manifest paired counts differ")
    _require(manifest["total_paired_metacells"] == sum(COUNTS), "manifest total count differs")
    _require(tuple(manifest["atac_original_lsi_dims_1based"]) == ATAC_COMPONENTS,
             "manifest ATAC component selection differs")

    ids_path = data_dir / "paired_metacell_ids_by_time.npz"
    files["paired_ids"] = _file_record(ids_path)
    id_blocks = _npz_blocks(ids_path)
    for key, block, count in zip(KEYS, id_blocks, COUNTS):
        _require(block.shape == (count,), f"{key} pairing ID count differs")
    paired_ids = np.concatenate(id_blocks).astype(str)
    _require(len(np.unique(paired_ids)) == sum(COUNTS), "paired metacell IDs are not unique")
    row_keys = np.repeat(np.asarray(KEYS), COUNTS)
    row_ages = np.repeat(np.asarray(AGES), COUNTS)
    row_times = np.repeat(np.asarray(TIMES, dtype=np.float32), COUNTS)
    obs_path = data_dir / "paired_obs.csv"
    files["paired_obs"] = _file_record(obs_path)
    # Deliberately do not load cell-type or biological annotation columns.
    obs = pd.read_csv(obs_path, usecols=["paired_metacell_id", "time_key", "processed_age"])
    metadata_audit = {
        "paired_ids": _exact(obs["paired_metacell_id"].astype(str).to_numpy(), paired_ids, "paired_obs IDs"),
        "time_keys": _exact(obs["time_key"].astype(str).to_numpy(), row_keys, "paired_obs time order"),
        "ages_days": _exact(obs["processed_age"].to_numpy(), row_ages, "paired_obs ages"),
    }

    full_raw, full_normalized, modalities = {}, {}, {}
    for modality, dimension in DIMENSIONS.items():
        stem = MATRIX_FILES[modality]
        source = data_dir / f"{stem}_by_time.npz"
        normalized_path = data_dir / f"{stem}_normalized_by_time.npz"
        norm_path = data_dir / NORM_FILES[modality]
        for suffix, path in (("raw", source), ("normalized", normalized_path), ("scale", norm_path)):
            files[f"{modality}_{suffix}"] = _file_record(path)
        scale = float(torch.load(norm_path, map_location="cpu", weights_only=False)["scale"])
        _require(scale == SCALES[modality], f"{modality} frozen normalization scale differs")
        _require(manifest[f"{modality}_dimensions"] == dimension, f"{modality} manifest dimension differs")
        _require(float(manifest[f"{modality}_scale"]) == scale, f"{modality} manifest scale differs")
        details = provenance[modality]
        _require(Path(details["source"]).resolve() == source.resolve(), f"{modality} provenance source differs")
        _require(Path(details["normalization"]).resolve() == norm_path.resolve(), f"{modality} provenance scale source differs")
        _require(details["source_sha256"] == files[f"{modality}_raw"]["sha256"], f"{modality} source changed since loading")
        _require(details["normalization_sha256"] == files[f"{modality}_scale"]["sha256"], f"{modality} scale changed since loading")
        _require(float(details["scale"]) == scale, f"{modality} training scale differs")
        _require(tuple(details["stage_keys"]) == KEYS, f"{modality} training stage keys differ")
        _require(tuple(details["full_counts"]) == COUNTS, f"{modality} training source counts differ")
        _require(details["dimension"] == dimension, f"{modality} training dimension differs")
        _require(list(details["training_stage_indices"]) == list(range(len(KEYS))),
                 f"{modality} must retain all seven stages; no held-out training is allowed")
        _require(tuple(indices[modality]) == KEYS, f"{modality} selected index stages/order differ")
        _require(len(training[modality]) == len(KEYS), f"{modality} must have seven training blocks")
        raw_blocks = _npz_blocks(source)
        normalized_blocks = _npz_blocks(normalized_path)
        block_audit = {}
        for index, (key, count, raw, normed) in enumerate(zip(KEYS, COUNTS, raw_blocks, normalized_blocks)):
            _require(raw.dtype == np.float32 and normed.dtype == np.float32,
                     f"{modality} {key} source arrays must preserve float32 precision")
            _require(raw.shape == (count, dimension), f"{modality} {key} raw shape differs")
            normalization_audit = _exact(normed, raw / scale, f"{modality} {key} stored normalization")
            selected = _array(indices[modality][key])
            _require(selected.ndim == 1 and np.issubdtype(selected.dtype, np.integer),
                     f"{modality} {key} indices must be a one-dimensional integer array")
            _require(len(selected) > 0 and len(np.unique(selected)) == len(selected),
                     f"{modality} {key} indices are empty or repeated")
            _require(np.all((selected >= 0) & (selected < count)), f"{modality} {key} indices are out of range")
            values = _array(training[modality][index])
            _require(values.dtype == np.float32, f"{modality} {key} training data are not float32")
            block_audit[key] = {
                "source_count": count, "training_count": len(selected),
                "stored_normalization": normalization_audit,
                "training_selection": _exact(values, normed[selected], f"{modality} {key} training selection"),
            }
            if "coordinate_sha256" in manifest:
                digest = hashlib.sha256(np.ascontiguousarray(raw).view(np.uint8)).hexdigest()
                _require(digest == manifest["coordinate_sha256"][modality][key],
                         f"{modality} {key} raw coordinate checksum differs from manifest")
        selected_counts = [item["training_count"] for item in block_audit.values()]
        _require(list(details["selected_counts"]) == selected_counts, f"{modality} provenance selected counts differ")
        _require(details["training_cells"] == sum(selected_counts), f"{modality} provenance training total differs")
        modalities[modality] = {"dimension": dimension, "scale": scale, "stages": block_audit}
        full_raw[modality] = np.concatenate(raw_blocks)
        full_normalized[modality] = np.concatenate(normalized_blocks)

    if "coordinate_sha256" in manifest:
        for key, values in zip(KEYS, id_blocks):
            # The source preparer hashes fixed-width ASCII IDs, not Unicode.
            digest = hashlib.sha256(np.ascontiguousarray(values.astype("S")).view(np.uint8)).hexdigest()
            _require(digest == manifest["coordinate_sha256"]["paired_ids"][key],
                     f"{key} paired ID checksum differs from manifest")
    _require(_array(initial).dtype == np.float32, "initial RNA is not float32")
    initial_audit = _exact(initial, full_normalized["rna"][:COUNTS[0]], "all D4 RNA initial states")
    reference = _audit_h5ad(Path(reference_h5ad), full_normalized["rna"], full_raw["rna"],
                            paired_ids, row_keys, row_ages, row_times)
    if trained_h5ad is None:
        trained = {"status": "not_requested", "path": None}
    else:
        trained_path = Path(trained_h5ad)
        if trained_path.exists():
            trained = _audit_h5ad(trained_path, full_normalized["rna"], full_raw["rna"],
                                  paired_ids, row_keys, row_ages, row_times)
        else:
            trained = {"status": "not_present", "path": str(trained_path.resolve())}
    return {
        "status": "passed", "dataset": "human_cerebral_7time_d4_d21_no_d16",
        "scenario": "full", "held_out_stages": [], "data_directory": str(data_dir),
        "stage_keys": list(KEYS), "ages_days": list(AGES), "physical_times": list(TIMES),
        "physical_time_convention": "(age_day - 4) / 10",
        "source_counts": list(COUNTS), "source_paired_metacells": sum(COUNTS),
        "source_files": files, "source_row_identity": metadata_audit,
        "modalities": modalities, "all_initial_rna": initial_audit,
        "cytobridge_prepared_input": reference, "cytobridge_trained_input": trained,
        "normalization_precision": "float32 raw coordinates divided by frozen scalar; exact equality required",
        "pairing_policy": "Computational pairing is audited only; no pairing IDs or cell-type labels are supplied to training.",
        "sampling_policy": "Source references remain full; every training block is checked against its explicit selected indices.",
    }
