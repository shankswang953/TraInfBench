"""Disk-backed, float64 execution of the official scMultiNODE graph distances.

Only storage and Dijkstra source-row batching change. KNN construction, metric,
tie handling, undirected shortest paths, and the global disconnected-distance
replacement are the same as ``optim.running._mod_distance``. These routines do
not alter the official source or reduce the set of training cells.
"""
from __future__ import annotations

from contextlib import contextmanager
import mmap
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn import get_config
from sklearn.neighbors import kneighbors_graph


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def estimate_distance_resources(
    n_cells: int,
    n_features: int,
    n_neighbors: int,
    row_block_size: int = 128,
) -> dict[str, int | float]:
    """Conservative workspace/scratch estimate, not a measured peak guarantee.

    The unchanged sklearn call can hold a distance chunk plus full-shaped
    argpartition indices. Budget three configured chunks, a conservative CSR
    allowance, input copies, and Dijkstra/block masks. Mapped-file size is NOT a
    promise about RSS: page eviction is platform-dependent and advisory.
    """
    n_cells = _positive_integer(n_cells, "n_cells")
    n_features = _positive_integer(n_features, "n_features")
    n_neighbors = _positive_integer(n_neighbors, "n_neighbors")
    block_rows = min(_positive_integer(row_block_size, "row_block_size"), n_cells)
    if n_neighbors > n_cells:
        raise ValueError("n_neighbors exceeds the number of cells")
    working_memory_mib = float(get_config()["working_memory"])
    if not np.isfinite(working_memory_mib) or working_memory_mib < 0:
        raise ValueError("Unbounded/invalid sklearn working_memory setting")
    # Even working_memory=0 still permits at least one pairwise-distance row.
    knn_chunk = min(8 * n_cells**2, max(8 * n_cells, int(working_memory_mib * 1024**2)))
    mapped_bytes = 8 * n_cells**2
    workspace = 3 * knn_chunk + 64 * n_cells * n_neighbors + 16 * n_cells * n_features + 32 * block_rows * n_cells
    return {
        "n_cells": n_cells,
        "row_block_size": block_rows,
        "mapped_array_bytes": mapped_bytes,
        "scratch_file_bytes": mapped_bytes + 4096,
        "estimated_workspace_bytes": workspace,
        "sklearn_working_memory_mib": working_memory_mib,
    }


class DiskDistanceArray:
    """Array-like owner that holds no live mapping between operations.

    Slices/advanced indexing return owned NumPy copies. Each read or write maps
    the file, performs the requested operation, then closes that mapping. This
    removes mapped pages from process RSS even where MADV_DONTNEED is only a
    weak hint. Callers must still bound explicit selection sizes: requesting
    ``array[:, :]`` explicitly copies the entire matrix.

    Implicit NumPy coercion is restricted to ``max_array_bytes`` (64 MiB by
    default). Larger diagnostic copies require an explicit ``to_numpy`` byte
    budget. This is an array adapter for the indexed QGW path, not a universal
    drop-in replacement for all ndarray methods or arithmetic.
    """

    def __init__(self, path: str | Path, max_array_bytes: int = 64 * 1024**2) -> None:
        self.path = Path(path).resolve()
        self.max_array_bytes = _positive_integer(max_array_bytes, "max_array_bytes")
        self._active_mappings = 0
        with self._mapped("r") as mapping:
            self.shape = mapping.shape
            self.dtype = mapping.dtype
            self.ndim = mapping.ndim
            self.size = mapping.size
            self.nbytes = mapping.nbytes
        if self.ndim != 2 or min(self.shape) < 1 or self.dtype != np.float64:
            raise ValueError("DiskDistanceArray requires a nonempty two-dimensional float64 .npy file")

    @property
    def active_mappings(self) -> int:
        """Diagnostic count; zero outside an individual read/write operation."""
        return self._active_mappings

    @contextmanager
    def _mapped(self, mode: str):
        mapping = np.load(self.path, mmap_mode=mode, allow_pickle=False)
        if not isinstance(mapping, np.memmap):
            raise ValueError("DiskDistanceArray requires a memory-mappable .npy file")
        self._active_mappings += 1
        try:
            yield mapping
        finally:
            # No view of this map may escape: __getitem__ copies first, while
            # __setitem__ has already completed and flushed its assignment.
            mapping._mmap.close()
            self._active_mappings -= 1

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        with self._mapped("r") as mapping:
            return np.array(mapping[key], copy=True, subok=False)

    def __setitem__(self, key, value) -> None:
        with self._mapped("r+") as mapping:
            mapping[key] = value
            mapping.flush()

    def to_numpy(self, max_bytes: int | None = None) -> np.ndarray:
        """Materialize an owned diagnostic copy only within a stated budget."""
        budget = self.max_array_bytes if max_bytes is None else _positive_integer(max_bytes, "max_bytes")
        if self.nbytes > budget:
            raise MemoryError(
                f"Refusing to materialize {self.nbytes} distance bytes with a {budget}-byte budget; "
                "use bounded slices or explicitly budget to_numpy(max_bytes=...)"
            )
        return self[:, :]

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        if copy is False:
            raise ValueError("DiskDistanceArray cannot provide a zero-copy ndarray")
        if dtype is not None and self.size * np.dtype(dtype).itemsize > self.max_array_bytes:
            raise MemoryError("Requested dtype exceeds the implicit distance-array copy budget")
        values = self.to_numpy()
        return values if dtype is None else values.astype(dtype, copy=False)


def flush_and_drop_pages(array: np.ndarray) -> bool:
    """Flush writes and advise eviction when supported; never close the mmap.

    ``MADV_DONTNEED`` is only a page-cache/RSS hint after a successful flush.
    Missing platform support or rejected advice is harmless and reported as
    False. Copy-on-write maps are skipped BEFORE flush/advice: their private
    modified pages are not persisted by flush and could be lost to eviction.
    The mapping and every outstanding NumPy view remain valid.
    """
    if not isinstance(array, np.memmap):
        return False
    if array.mode == "c":
        return False
    array.flush()
    mapping = getattr(array, "_mmap", None)
    advice = getattr(mmap, "MADV_DONTNEED", None)
    if mapping is None or advice is None or not hasattr(mapping, "madvise"):
        return False
    try:
        mapping.madvise(advice)
    except (OSError, ValueError, NotImplementedError):
        return False
    return True


def _report_large_progress(
    stage: str, start: int, stop: int, total: int, target: Path, started: float,
) -> None:
    """Bounded pilot visibility without adding noise to small equivalence tests."""
    interval = 8192
    if total >= interval and (stop // interval > start // interval or stop == total):
        print(
            f"[Disk-backed distances | {stage}] {stop}/{total} rows; "
            f"elapsed={time.monotonic() - started:.1f}s; file={target}", flush=True,
        )


def blockwise_nan_to_num_and_normalize(
    C: np.ndarray | DiskDistanceArray,
    row_block_size: int = 128,
) -> np.ndarray | DiskDistanceArray:
    """Match ``nan_to_num(C, nan=0); C /= max(C)`` in-place in float64.

    NumPy's default +/-infinity replacements are preserved, as is the global
    (not per-block) normalization scalar. An all-zero maximum safely fails
    instead of producing NaNs. The same array object is returned, with any
    persistent memory map left open for downstream QGW code. A DiskDistanceArray
    instead copies/writes one block at a time and closes every temporary map.
    """
    row_block_size = _positive_integer(row_block_size, "row_block_size")
    owner = isinstance(C, DiskDistanceArray)
    if not isinstance(C, (np.ndarray, DiskDistanceArray)) or C.ndim != 2 or min(C.shape) < 1:
        raise ValueError("C must be a nonempty two-dimensional NumPy or disk-backed array")
    if C.dtype != np.float64 or (not owner and not C.flags.writeable):
        raise ValueError("C must be a writable float64 array")
    maximum = -np.inf
    for start in range(0, len(C), row_block_size):
        block = C[start:start + row_block_size]
        np.nan_to_num(block, copy=False, nan=0.0)
        maximum = max(maximum, float(np.max(block)))
        if owner:
            C[start:start + row_block_size] = block
        del block
        flush_and_drop_pages(C)
    if not np.isfinite(maximum) or maximum == 0:
        raise ValueError("Cannot normalize distances with zero or nonfinite global maximum")
    for start in range(0, len(C), row_block_size):
        block = C[start:start + row_block_size]
        block /= maximum
        if owner:
            C[start:start + row_block_size] = block
        del block
        flush_and_drop_pages(C)
    return C


class DiskBackedDistances:
    """Callable replacement for official ``_mod_distance`` with unique files.

    Each successful call returns an open writable ``np.memmap`` by default,
    or a ``DiskDistanceArray`` when bounded_mapping=True, and adds a
    JSON-compatible entry to ``records``. In bounded mode all writes and reads
    open/close maps per operation; no persistent map can accumulate RSS. The
    caller owns scratch cleanup: files are retained on success or failure.
    Existing files are rejected, never truncated or reused. Resource checks
    happen before KNN/Dijkstra allocation. The default 4 GiB workspace cap is
    a conservative planning guard, not an operating-system RSS limit.
    """

    def __init__(
        self,
        scratch_dir: str | Path,
        row_block_size: int = 128,
        max_workspace_bytes: int = 4 * 1024**3,
        min_free_bytes: int = 256 * 1024**2,
        prefix: str = "distance",
        bounded_mapping: bool = False,
    ) -> None:
        self.scratch_dir = Path(scratch_dir)
        self.row_block_size = _positive_integer(row_block_size, "row_block_size")
        self.max_workspace_bytes = _positive_integer(max_workspace_bytes, "max_workspace_bytes")
        if isinstance(min_free_bytes, bool) or not isinstance(min_free_bytes, (int, np.integer)) or min_free_bytes < 0:
            raise ValueError("min_free_bytes must be a nonnegative integer")
        self.min_free_bytes = int(min_free_bytes)
        if not isinstance(prefix, str) or not prefix or Path(prefix).name != prefix or prefix in (".", ".."):
            raise ValueError("prefix must be one safe filename component")
        self.prefix = prefix
        if not isinstance(bounded_mapping, bool):
            raise ValueError("bounded_mapping must be a bool")
        self.bounded_mapping = bounded_mapping
        self.records: list[dict[str, Any]] = []
        self._next_file = 0

    def __call__(self, feature_mat: np.ndarray, n_neighbors: int, metric: str) -> np.memmap | DiskDistanceArray:
        started = time.monotonic()
        if not isinstance(feature_mat, np.ndarray) or feature_mat.ndim != 2 or min(feature_mat.shape) < 1:
            raise ValueError("feature_mat must be a nonempty cells-by-features array")
        if metric != "correlation":
            raise ValueError("This audited adapter supports the official correlation metric only")
        plan = estimate_distance_resources(*feature_mat.shape, n_neighbors, self.row_block_size)
        if plan["estimated_workspace_bytes"] > self.max_workspace_bytes:
            raise MemoryError(
                f"Estimated distance workspace {plan['estimated_workspace_bytes']} bytes exceeds "
                f"the configured cap {self.max_workspace_bytes}; no graph was allocated"
            )
        if not np.isfinite(feature_mat).all():
            raise ValueError("Nonfinite input features")
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        target = self.scratch_dir / f"{self.prefix}_{self._next_file:04d}.npy"
        self._next_file += 1
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite distance file {target}")
        free_bytes = shutil.disk_usage(self.scratch_dir).free
        required_bytes = int(plan["scratch_file_bytes"]) + self.min_free_bytes
        if free_bytes < required_bytes:
            raise OSError(
                f"Insufficient scratch space: {free_bytes} bytes free; "
                f"need {required_bytes} including the safety reserve"
            )
        # Reserve exclusively before open_memmap's write mode. The zero-byte
        # file is newly owned by this call; no preexisting output is truncated.
        with target.open("xb"):
            pass
        record: dict[str, Any] = {
            **plan, "path": str(target.resolve()), "dtype": "float64", "metric": metric,
            "n_neighbors": int(n_neighbors), "free_scratch_bytes_before": free_bytes,
            "status": "building", "madv_dontneed_succeeded": False,
            "mapping_strategy": "open_close_per_operation" if self.bounded_mapping else "persistent_memmap",
        }
        self.records.append(record)
        distances = None
        try:
            initial_mapping = np.lib.format.open_memmap(
                target, mode="w+", dtype=np.float64, shape=(len(feature_mat), len(feature_mat))
            )
            if self.bounded_mapping:
                initial_mapping.flush()
                initial_mapping._mmap.close()
                distances = DiskDistanceArray(target)
            else:
                distances = initial_mapping
            del initial_mapping
            # Intentionally verbatim graph construction from the official code.
            # No working-memory override, alternate graph, float32 conversion,
            # smaller neighbor count, or tie-handling substitution is applied.
            knn_graph = kneighbors_graph(
                feature_mat, n_neighbors, mode="connectivity", metric=metric, include_self=True
            )
            graph = csr_matrix(knn_graph)
            maximum = -np.inf
            for start in range(0, len(feature_mat), self.row_block_size):
                stop = min(start + self.row_block_size, len(feature_mat))
                block = dijkstra(
                    csgraph=graph, directed=False, return_predecessors=False,
                    indices=np.arange(start, stop),
                )
                if block.dtype != np.float64:
                    raise AssertionError("Dijkstra did not return float64 distances")
                maximum = max(maximum, float(np.max(block, where=np.isfinite(block), initial=-np.inf)))
                distances[start:stop] = block
                del block
                record["madv_dontneed_succeeded"] |= flush_and_drop_pages(distances)
                _report_large_progress("Dijkstra", start, stop, len(feature_mat), target, started)
            del graph, knn_graph
            if not np.isfinite(maximum):
                raise ValueError("No finite graph distance was produced")
            # A second pass is essential: every disconnected pair gets the
            # same GLOBAL maximum, not an earlier/per-row/per-block maximum.
            for start in range(0, len(feature_mat), self.row_block_size):
                block = distances[start:start + self.row_block_size]
                block[block > maximum] = maximum
                if self.bounded_mapping:
                    distances[start:start + self.row_block_size] = block
                del block
                record["madv_dontneed_succeeded"] |= flush_and_drop_pages(distances)
                stop = min(start + self.row_block_size, len(feature_mat))
                _report_large_progress("global disconnected replacement", start, stop, len(feature_mat), target, started)
            record.update(status="complete", global_finite_max=maximum, elapsed_seconds=time.monotonic() - started)
            return distances
        except BaseException as error:
            if distances is not None:
                flush_and_drop_pages(distances)
            record.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic() - started)
            raise
