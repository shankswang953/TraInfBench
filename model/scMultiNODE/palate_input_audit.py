"""Audit palate scMultiNODE inputs against frozen CytoBridge common-space data.

CytoBridge trains in raw RNA PCA40. This adapter divides that same representation
by the frozen RNA scale and uses independently scaled ATAC LSI15. Pairing IDs
serve only as audit evidence, never as supervision or training-time labels.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch

from run_gastrulation import ROOT, sha256


KEYS = ("time_0", "time_1", "time_2", "time_3")
STAGES = ("E12.5", "E13.5", "E14.0", "E14.5")
TIMES = (0.0, 1.0, 1.5, 2.0)
COUNTS = (2570, 5692, 6638, 4933)
SPLITS = {"full": (0, 1, 2, 3), "loo1": (0, 2, 3), "loo2": (0, 1, 3)}
SCALES = {"rna": 22.22992031699182, "atac": 0.10373610732172349}
DIMENSIONS = {"rna": 40, "atac": 15}
MATRICES = {"rna": "rna_pca_by_time.npz", "atac": "atac_lsi15_by_time.npz"}
NORMALIZATIONS = {"rna": "primal_norm_params.pt", "atac": "secondary_norm_params_lsi15.pt"}
BENCHMARK_NORMALIZATIONS = {
    "rna": "palate_rna_primal_norm_params.pt",
    "atac": "palate_atac_secondary_norm_params_lsi15.pt",
}
PREPARED_RNA = {
    "full": "palate_rna_cytobridge.h5ad",
    "loo1": "palate_rna_loo_time1_cytobridge.h5ad",
    "loo2": "palate_rna_loo_time2_cytobridge.h5ad",
}
TRAINED_RNA = {
    "full": "cytobridge_palate_rna_20000/adata.h5ad",
    "loo1": "cytobridge_palate_loo_time1_20000/adata.h5ad",
    "loo2": "cytobridge_palate_loo_time2_20000/adata.h5ad",
}


def _require(condition, message):
    if not condition:
        raise ValueError(f"palate input audit: {message}")


def _array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _exact(actual, expected, name):
    actual, expected = _array(actual), _array(expected)
    _require(actual.shape == expected.shape, f"{name} shape {actual.shape} differs from {expected.shape}")
    numeric = np.issubdtype(actual.dtype, np.number)
    difference = None
    if numeric:
        _require(np.isfinite(actual).all() and np.isfinite(expected).all(), f"{name} contains nonfinite values")
        difference = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))) if actual.size else 0.0
    _require(np.array_equal(actual, expected), f"{name} differs; max_abs_difference={difference}")
    audit = {"exact_equal": True, "shape": list(actual.shape)}
    if numeric:
        audit["max_abs_difference"] = difference
    return audit


def _file_record(path):
    path = Path(path)
    _require(path.is_file(), f"required file missing: {path}")
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _check_npz_headers(path, dimension):
    """Read array shapes/dtypes without decompressing held-out coordinate data."""
    with zipfile.ZipFile(path) as archive:
        _require(archive.namelist() == [f"{key}.npy" for key in KEYS], f"{path} stage keys/order differ")
        for key, count in zip(KEYS, COUNTS):
            with archive.open(f"{key}.npy") as stream:
                version = np.lib.format.read_magic(stream)
                _require(version in ((1, 0), (2, 0)), f"{path} unsupported NPY header version {version}")
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = reader(stream)
            _require(shape == (count, dimension), f"{path} {key} shape differs from frozen full counts/dimension")
            _require(dtype == np.float32, f"{path} {key} is not frozen float32")


def _column(node):
    """Read only a requested H5AD identity/time column, not biological labels."""
    if isinstance(node, h5py.Dataset):
        return node.asstr()[...] if h5py.check_string_dtype(node.dtype) is not None else node[...]
    _require(node.attrs.get("encoding-type") == "categorical", f"unsupported H5AD column encoding at {node.name}")
    categories, codes = _column(node["categories"]), np.asarray(node["codes"])
    _require(np.all((codes >= 0) & (codes < len(categories))), f"missing/invalid metadata at {node.name}")
    return categories[codes]


def _read_reference(path, modality, available_stages, observed_stages, raw_blocks, expected_ids=None):
    """Compare observed coordinates only; full-stage IDs and times are metadata."""
    path = Path(path)
    record = _file_record(path)
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        ids = np.asarray(_column(obs[obs.attrs["_index"]]), dtype=str)
        stages = np.asarray(_column(obs["stage"]), dtype=str)
        times = np.asarray(_column(obs["time_point_processed"]))
        expected_stages = np.concatenate([np.repeat(STAGES[i], COUNTS[i]) for i in available_stages])
        expected_times = np.concatenate([np.full(COUNTS[i], TIMES[i], dtype=np.float32) for i in available_stages])
        record["stages"] = _exact(stages, expected_stages, f"{path} stages/order")
        record["physical_times"] = _exact(times, expected_times, f"{path} physical times/order")
        _require(len(ids) == len(expected_stages) and len(np.unique(ids)) == len(ids), f"{path} IDs/counts are not unique/aligned")
        if expected_ids is not None:
            record["row_ids"] = _exact(ids, expected_ids, f"{path} row IDs")
        rows = np.flatnonzero(np.isin(times, [TIMES[i] for i in observed_stages]))
        field = "X_latent" if modality == "rna" else "X_lsi15"
        coordinates = handle["obsm"][field]
        _require(coordinates.shape == (len(ids), DIMENSIONS[modality]), f"{path} {field} dimension/count differs")
        _require(coordinates.dtype == np.float32, f"{path} {field} is not float32")
        # HDF5 fancy indexing reads only observed rows, including in LOO mode.
        raw_reference = coordinates[rows]
        raw_source = np.concatenate([raw_blocks[i] for i in observed_stages])
        record["observed_raw_coordinates"] = _exact(raw_reference, raw_source, f"{path} observed {field}")
        record["observed_normalized_coordinates"] = _exact(
            raw_reference / SCALES[modality], raw_source / SCALES[modality], f"{path} observed common normalization")
    record.update(status="verified", embedding=field,
                  coordinate_space="raw RNA PCA40" if modality == "rna" else "raw filtered ATAC LSI15",
                  observed_stage_indices=list(observed_stages))
    return record, ids


def audit_inputs(data_dir, training, initial, provenance, indices, *,
                 reference_dir=ROOT / "data", trained_root=ROOT / "results"):
    """Fail on any frozen-source, split, order, scale, or selected-tensor mismatch.

    Historical CytoBridge outputs are checked when present. Reference prepared
    RNA (full and selected LOO), ATAC LSI15 and scale files are required. All
    source preprocessing/scalars remain frozen from the complete dataset: LOO
    is conditional on these representations, not raw-data inductive LOO.
    """
    data_dir, reference_dir = Path(data_dir).resolve(), Path(reference_dir).resolve()
    _require(set(training) == set(DIMENSIONS), "training must contain RNA and ATAC only")
    _require(set(provenance) == set(DIMENSIONS), "provenance must contain RNA and ATAC only")
    _require(set(indices) == set(DIMENSIONS), "indices must contain RNA and ATAC only")
    observed = tuple(provenance["rna"]["training_stage_indices"])
    matches = [name for name, split in SPLITS.items() if observed == split]
    _require(len(matches) == 1, f"unsupported training stage indices: {observed}")
    scenario = matches[0]
    _require(tuple(provenance["atac"]["training_stage_indices"]) == observed, "RNA/ATAC observed stages differ")
    selected_keys = tuple(KEYS[i] for i in observed)
    files, modalities, raw_by_modality = {}, {}, {}

    for modality, dimension in DIMENSIONS.items():
        source = data_dir / MATRICES[modality]
        norm = data_dir / NORMALIZATIONS[modality]
        benchmark_norm = reference_dir / BENCHMARK_NORMALIZATIONS[modality]
        files[f"{modality}_source"] = _file_record(source)
        files[f"{modality}_normalization"] = _file_record(norm)
        files[f"{modality}_benchmark_normalization"] = _file_record(benchmark_norm)
        scale = float(torch.load(norm, map_location="cpu", weights_only=False)["scale"])
        reference_scale = float(torch.load(benchmark_norm, map_location="cpu", weights_only=False)["scale"])
        _require(scale == SCALES[modality] and reference_scale == scale, f"{modality} source/benchmark scale differs")
        _check_npz_headers(source, dimension)
        details = provenance[modality]
        _require(Path(details["source"]).resolve() == source.resolve(), f"{modality} source provenance differs")
        _require(Path(details["normalization"]).resolve() == norm.resolve(), f"{modality} normalization provenance differs")
        _require(details["source_sha256"] == files[f"{modality}_source"]["sha256"], f"{modality} source changed since loading")
        _require(details["normalization_sha256"] == files[f"{modality}_normalization"]["sha256"], f"{modality} scale changed since loading")
        _require(details["scale"] == scale and details["dimension"] == dimension, f"{modality} training scale/dimension differs")
        _require(tuple(details["stage_keys"]) == KEYS and tuple(details["full_counts"]) == COUNTS, f"{modality} stage/count provenance differs")
        _require(tuple(indices[modality]) == selected_keys, f"{modality} selected-index stage keys/order differ")
        _require(len(training[modality]) == len(observed), f"{modality} number of training stages differs")
        with np.load(source, allow_pickle=False) as archive:
            raw_blocks = {i: np.asarray(archive[KEYS[i]]) for i in observed}
        block_audit = {}
        for block_index, i in enumerate(observed):
            key, raw = KEYS[i], raw_blocks[i]
            _require(np.isfinite(raw).all(), f"{modality} {key} nonfinite observed source")
            chosen = _array(indices[modality][key])
            _require(chosen.ndim == 1 and np.issubdtype(chosen.dtype, np.integer), f"{modality} {key} indices must be one-dimensional integers")
            _require(len(chosen) > 0 and len(np.unique(chosen)) == len(chosen), f"{modality} {key} indices empty/repeated")
            _require(np.all((chosen >= 0) & (chosen < COUNTS[i])), f"{modality} {key} indices out of range")
            values = _array(training[modality][block_index])
            _require(values.dtype == np.float32, f"{modality} {key} training data are not float32")
            block_audit[key] = {
                "source_count": COUNTS[i], "training_count": len(chosen),
                "normalized_training_selection": _exact(values, (raw / scale)[chosen], f"{modality} {key} normalized training selection"),
            }
        selected_counts = [block_audit[key]["training_count"] for key in selected_keys]
        _require(list(details["selected_counts"]) == selected_counts, f"{modality} selected counts provenance differs")
        _require(details["training_cells"] == sum(selected_counts), f"{modality} training total provenance differs")
        raw_by_modality[modality] = raw_blocks
        modalities[modality] = {"dimension": dimension, "scale": scale, "stages": block_audit,
            "benchmark_scale_exact": True,
            "benchmark_scale_file_sha256_equal": files[f"{modality}_normalization"]["sha256"] == files[f"{modality}_benchmark_normalization"]["sha256"]}

    _require(_array(initial).dtype == np.float32, "initial RNA is not float32")
    initial_audit = _exact(initial, raw_by_modality["rna"][0] / SCALES["rna"], "all E12.5 initial RNA")
    full_rna, full_ids = _read_reference(reference_dir / PREPARED_RNA["full"], "rna", SPLITS["full"], observed, raw_by_modality["rna"])
    atac, _ = _read_reference(reference_dir / "palate_atac_benchmark.h5ad", "atac", SPLITS["full"], observed, raw_by_modality["atac"], full_ids)
    full_times = np.repeat(np.asarray(TIMES), COUNTS)
    observed_ids = full_ids[np.isin(full_times, [TIMES[i] for i in observed])]
    if scenario == "full":
        scenario_rna = full_rna
    else:
        scenario_rna, _ = _read_reference(reference_dir / PREPARED_RNA[scenario], "rna", observed, observed, raw_by_modality["rna"], observed_ids)
    historical = {}
    for name in dict.fromkeys(("full", scenario)):
        if trained_root is None:
            historical[name] = {"status": "not_requested", "path": None}
            continue
        path = Path(trained_root) / TRAINED_RNA[name]
        if not path.exists():
            historical[name] = {"status": "not_present", "path": str(path.resolve())}
            continue
        historical[name], _ = _read_reference(path, "rna", SPLITS[name], observed, raw_by_modality["rna"], full_ids if name == "full" else observed_ids)
    return {
        "status": "passed", "dataset": "palate", "scenario": scenario,
        "data_directory": str(data_dir), "source_stage_keys": list(KEYS),
        "source_stages": list(STAGES), "source_counts": list(COUNTS), "source_cells": sum(COUNTS),
        "physical_times": list(TIMES), "observed_stage_indices": list(observed),
        "observed_times": [TIMES[i] for i in observed],
        "held_out_stages": [STAGES[i] for i in range(4) if i not in observed],
        "source_files": files, "modalities": modalities, "all_initial_rna": initial_audit,
        "prepared_full_rna": full_rna, "prepared_scenario_rna": scenario_rna,
        "prepared_atac_lsi15": atac, "historical_cytobridge_inputs": historical,
        "cytobridge_training_space": "raw RNA PCA40; normalized comparison divides by the frozen RNA scale",
        "scmultinode_training_space": "frozen normalized RNA PCA40 and ATAC filtered LSI15",
        "pairing_policy": "Row IDs and times are audited only; paired-cell IDs and biological labels do not supervise training.",
        "loo_policy": "No held-out coordinates are read for these training checks; source headers/counts and identity/time metadata cover the full dataset. File hashes fingerprint whole frozen files. All training comparisons use observed stages only.",
        "representation_caveat": "PCA/LSI and normalization are frozen full-data representations; LOO is conditional on these inputs, not end-to-end inductive raw-data LOO. No held-out metrics or hyperparameter selection are performed.",
    }
