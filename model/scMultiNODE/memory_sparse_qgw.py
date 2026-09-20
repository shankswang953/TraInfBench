"""Storage-only adapters for the pinned official scMultiNODE QGW implementation.

The original QGW control flow, compressed GW and local EMD computations are
reused. Only the two deterministic N-by-N couplings use sparse storage; an
individual column is materialized when the original algorithm requests it.
The official source module and its global dictionary are never modified.
"""
from __future__ import annotations

from functools import wraps
import mmap
import time
from types import FunctionType
import warnings

import numpy as np
from scipy import sparse


class ColumnCoupling:
    """CSC-backed deterministic coupling exposing the upstream column access.

    Upstream QGW accesses only ``coupling[:, landmark]``. Returning a dense
    one-dimensional column preserves its indexing, summation and support order
    exactly, without ever materializing the full dense coupling.
    """

    def __init__(self, coupling):
        self._csc = coupling.tocsc()
        self.shape = self._csc.shape
        self.dtype = self._csc.dtype
        self._cached_index = None
        self._cached_column = None

    @property
    def storage_bytes(self):
        return sum(x.nbytes for x in (self._csc.data, self._csc.indices, self._csc.indptr))

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("Only upstream coupling[:, integer_column] access is supported")
        rows, column = key
        if rows != slice(None) or not isinstance(column, (int, np.integer)):
            raise TypeError("Only upstream coupling[:, integer_column] access is supported")
        column = int(column)
        if column < 0:
            column += self.shape[1]
        if not 0 <= column < self.shape[1]:
            raise IndexError(column)
        if column != self._cached_index:
            start, stop = self._csc.indptr[column:column + 2]
            vector = np.zeros(self.shape[0], dtype=self.dtype)
            vector[self._csc.indices[start:stop]] = self._csc.data[start:stop]
            self._cached_index = column
            self._cached_column = vector
        return self._cached_column

    def __array__(self, *args, **kwargs):
        raise TypeError("Full dense materialization of ColumnCoupling is intentionally disabled")


def compress_graph_sparse(Dist, p, node_subset, *, renormalize_prob, row_block_size=512,
                          release_callback=None):
    """Same deterministic coupling and compressed masses, with bounded gathers.

    ``node_subset`` is intentionally not sorted: argmin tie-breaking depends on
    its incoming order, as in the original implementation. COO mass summation
    and the upstream renormalization are also retained, including their order.
    """
    if not isinstance(row_block_size, (int, np.integer)) or row_block_size <= 0:
        raise ValueError("row_block_size must be a positive integer")
    n = Dist.shape[0]
    if len(node_subset) == 0:
        raise ValueError("QGW requires at least one landmark")
    landmarks = np.asarray(node_subset, dtype=np.intp)
    columns = np.empty(n, dtype=np.intp)
    for start in range(0, n, row_block_size):
        stop = min(start + row_block_size, n)
        nearest = np.argmin(Dist[start:stop, :][:, landmarks], axis=1)
        columns[start:stop] = landmarks[nearest]
        if release_callback is not None:
            # The advanced-indexed distance block is a temporary copy and has
            # already been consumed; only integer assignments remain here.
            release_callback(Dist)
    coupling = sparse.coo_matrix((p, (np.arange(n), columns)), shape=(n, n))
    compressed = renormalize_prob(np.squeeze(np.array(np.sum(coupling, axis=0))))
    return ColumnCoupling(coupling), compressed


class _AttributeProxy:
    def __init__(self, original, **replacements):
        self._original = original
        self._replacements = replacements

    def __getattr__(self, name):
        if name in self._replacements:
            return self._replacements[name]
        return getattr(self._original, name)


def _coupling_audit(coupling, source_mass, target_mass):
    """Read-only scalar diagnostics; never densify a sparse coupling."""
    try:
        values = coupling.data if sparse.issparse(coupling) else np.asarray(coupling)
        row = np.asarray(coupling.sum(axis=1)).reshape(-1)
        column = np.asarray(coupling.sum(axis=0)).reshape(-1)
        return {
            "shape": list(coupling.shape), "dtype": str(coupling.dtype),
            "sparse": bool(sparse.issparse(coupling)),
            "finite": bool(np.isfinite(values).all()),
            "minimum": float(coupling.min()), "mass": float(row.sum()),
            "source_mass": float(np.asarray(source_mass).sum()),
            "target_mass": float(np.asarray(target_mass).sum()),
            "row_marginal_max_abs_error": float(np.max(np.abs(row - np.asarray(source_mass)))),
            "column_marginal_max_abs_error": float(np.max(np.abs(column - np.asarray(target_mass)))),
        }
    except Exception as error:
        # Auditing should not replace a returned solver result or mask a solver
        # exception. A failed diagnostic is explicitly reported, never called a
        # successful feasibility check.
        return {"audit_error": f"{type(error).__name__}: {error}"}


def _solver_log_audit(log):
    """Bounded summaries of the existing POT log; no solver option changes."""
    summary = {"log_keys": sorted(str(key) for key in log)}
    for key in ("warning", "result_code", "gw_dist", "cost"):
        value = log.get(key)
        if value is None or isinstance(value, (str, bool, int, float)):
            summary[key] = value
        elif np.asarray(value).size == 1:
            summary[key] = np.asarray(value).item()
    losses = np.asarray(log.get("loss", []), dtype=float).reshape(-1)
    summary.update(
        loss_count=int(len(losses)),
        loss_initial=float(losses[0]) if len(losses) else None,
        loss_final=float(losses[-1]) if len(losses) else None,
        loss_finite=bool(np.isfinite(losses).all()),
        loss_head=losses[:5].tolist(), loss_tail=losses[-5:].tolist(),
        # POT generic CG overwrites its inner log each iteration. This status
        # therefore cannot prove all preceding LP calls converged.
        final_inner_status_only=True,
    )
    if len(losses) <= 256:
        summary["loss_values"] = losses.tolist()
    else:
        summary["loss_values_omitted"] = True
    return summary


def make_memory_qgw(official_module, *, row_block_size=512, max_dense_output_bytes=None,
                    release_memmap_pages=False, release_interval=128):
    """Return a drop-in QGW callable using the unmodified official function.

    An optional byte limit refuses a large *official dense fallback* before
    allocation. It does not substitute an approximate/sparse fallback or alter
    the official GW result. ``return_dense=True`` is checked before any work.

    Optional flush + MADV_DONTNEED requests eviction of clean file-backed pages
    between bounded operations. It never closes the map or changes distances.
    Copy-on-write maps are skipped because their private edits are not durable.
    This is advisory only: ``qgw.memory_advice`` records calls/errors, not an RSS
    guarantee. A caller needing a hard memory limit must still enforce one.

    Read-only ``qgw.solver_audit`` captures existing solver logs, every warning
    count and compressed marginals. ``qgw.raw_coupling_audit`` records returned
    full couplings before normalization/thresholding. Warnings are re-emitted,
    not suppressed; unchanged returned objects are passed back to upstream.
    """
    if not isinstance(row_block_size, (int, np.integer)) or row_block_size <= 0:
        raise ValueError("row_block_size must be a positive integer")
    if max_dense_output_bytes is not None and max_dense_output_bytes <= 0:
        raise ValueError("max_dense_output_bytes must be positive or None")
    if not isinstance(release_interval, (int, np.integer)) or release_interval <= 0:
        raise ValueError("release_interval must be a positive integer")
    original = official_module.compressed_gw_point_cloud
    advice = {
        "enabled": bool(release_memmap_pages),
        "supported": hasattr(mmap.mmap, "madvise") and hasattr(mmap, "MADV_DONTNEED"),
        "attempts": 0, "succeeded": 0, "failed": 0,
        "copy_on_write_skipped": 0, "first_error": None,
    }
    solver_audit = []
    raw_coupling_audit = []

    def release(array):
        if not release_memmap_pages or not isinstance(array, np.memmap):
            return
        if array.mode == "c":
            advice["copy_on_write_skipped"] += 1
            return
        if not advice["supported"]:
            return
        advice["attempts"] += 1
        try:
            # Flush all dirty file-backed pages before asking for eviction.
            # The underlying mapping stays live and all retained views remain
            # valid; later accesses fault the identical persisted bytes back in.
            array.flush()
            array._mmap.madvise(mmap.MADV_DONTNEED)
            advice["succeeded"] += 1
        except (OSError, ValueError, AttributeError) as exc:
            advice["failed"] += 1
            if advice["first_error"] is None:
                advice["first_error"] = f"{type(exc).__name__}: {exc}"

    def compression(Dist, p, node_subset):
        return compress_graph_sparse(
            Dist, p, node_subset, renormalize_prob=official_module.renormalize_prob,
            row_block_size=row_block_size, release_callback=release if release_memmap_pages else None,
        )

    @wraps(original)
    def qgw(Dist1, Dist2, p1, p2, node_subset1, node_subset2,
            verbose=False, return_dense=True, gw_type="gw", epsilon=None):
        started = time.monotonic()
        raw_record = {"call_index": len(raw_coupling_audit), "gw_type": gw_type,
                      "phase": "raw_qgw_before_max_normalization_and_threshold", "status": "running"}
        raw_coupling_audit.append(raw_record)
        output_cells = int(Dist1.shape[0]) * int(Dist2.shape[0])
        output_bytes = output_cells * np.result_type(p1, p2).itemsize
        # The sparse assembly starts with a default float64 COO, even when
        # supplied probabilities are float32. The outer-product fallback uses
        # their native dtype, and is checked separately below.
        dense_return_bytes = output_cells * np.result_type(p1, p2, np.float64).itemsize
        if return_dense and max_dense_output_bytes is not None and dense_return_bytes > max_dense_output_bytes:
            raw_record.update(status="requested_dense_output_refused", estimated_bytes=dense_return_bytes,
                              duration_seconds=time.monotonic() - started)
            raise MemoryError(f"Requested dense QGW output requires up to {dense_return_bytes} bytes; limit is {max_dense_output_bytes}")
        globals_copy = original.__globals__.copy()
        globals_copy["compress_graph_from_subset_point_cloud"] = compression
        if release_memmap_pages:
            original_compress = globals_copy["compress_graph"]
            original_local = globals_copy["find_submatching_locally_linear"]
            local_calls = 0

            def compress_and_release(Dist, p_compressed):
                result = original_compress(Dist, p_compressed)
                # The official np.ix_ indexing above owns a copied matrix.
                release(Dist)
                return result

            def local_and_release(*args, **kwargs):
                nonlocal local_calls
                result = original_local(*args, **kwargs)
                local_calls += 1
                if local_calls % release_interval == 0:
                    release(Dist1)
                    release(Dist2)
                return result

            globals_copy["compress_graph"] = compress_and_release
            globals_copy["find_submatching_locally_linear"] = local_and_release
        def guard_gw(function, name):
            @wraps(function)
            def guarded(*args, **kwargs):
                solver_started = time.monotonic()
                record = {"solver": name, "qgw_call_index": raw_record["call_index"], "status": "running"}
                solver_audit.append(record)
                caught = []
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        result = function(*args, **kwargs)
                    coupling, log = result
                    record.update(status="returned", coupling=_coupling_audit(coupling, args[2], args[3]))
                    try:
                        record["log"] = _solver_log_audit(log)
                    except Exception as error:
                        record["log_audit_error"] = f"{type(error).__name__}: {error}"
                    dense_fallback = bool(np.sum(coupling > 1e-10) > len(coupling) ** 1.5)
                    record["official_dense_fallback_condition"] = dense_fallback
                    if (dense_fallback and max_dense_output_bytes is not None
                            and output_bytes > max_dense_output_bytes):
                        record["status"] = "dense_fallback_refused"
                        raw_record["dense_fallback_refused"] = True
                        raise MemoryError(
                            "Official QGW selected its dense outer-product fallback, "
                            f"requiring {output_bytes} bytes; limit is {max_dense_output_bytes}. "
                            "No approximate replacement was made."
                        )
                    # Return the exact solver tuple, without a copied/repaired
                    # coupling, changed parameter or additional RNG call.
                    return result
                except BaseException as error:
                    if record["status"] != "dense_fallback_refused":
                        record["status"] = "failed"
                    record["error"] = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    record["duration_seconds"] = time.monotonic() - solver_started
                    record["warning_count"] = len(caught)
                    record["iteration_limit_warning_count"] = sum(
                        "numItermax reached before optimality" in str(item.message) for item in caught
                    )
                    record["warning_examples"] = [{
                        "message": str(item.message), "category": item.category.__name__,
                        "filename": item.filename, "lineno": item.lineno,
                    } for item in caught[:10]]
                    for item in caught:
                        warnings.showwarning(item.message, item.category, item.filename, item.lineno,
                                             line=item.line)
            return guarded

        original_ot = globals_copy["ot"]
        guarded_gromov = _AttributeProxy(
            original_ot.gromov,
            gromov_wasserstein=guard_gw(original_ot.gromov.gromov_wasserstein, "gromov_wasserstein"),
            entropic_gromov_wasserstein=guard_gw(
                original_ot.gromov.entropic_gromov_wasserstein, "entropic_gromov_wasserstein"),
        )
        globals_copy["ot"] = _AttributeProxy(original_ot, gromov=guarded_gromov)
        cloned = FunctionType(original.__code__, globals_copy, original.__name__,
                              original.__defaults__, original.__closure__)
        cloned.__kwdefaults__ = original.__kwdefaults__
        try:
            result = cloned(Dist1, Dist2, p1, p2, node_subset1, node_subset2,
                            verbose=verbose, return_dense=return_dense, gw_type=gw_type, epsilon=epsilon)
            raw_record.update(status="returned", coupling=_coupling_audit(result, p1, p2))
            return result
        except BaseException as error:
            raw_record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            raw_record["duration_seconds"] = time.monotonic() - started
            release(Dist1)
            release(Dist2)

    qgw.memory_advice = advice
    qgw.solver_audit = solver_audit
    qgw.raw_coupling_audit = raw_coupling_audit
    return qgw


def compact_threshold(coupling, *, threshold=0.01, normalize=True):
    """Official max normalization and absolute threshold, without storing zeros.

    Return a compact CSR matrix with unchanged dense entries. Thresholding only
    stored entries is equivalent because implicit zeros already satisfy the
    threshold and remain zero. Input objects are not modified.
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    value = coupling.copy()
    if normalize:
        scale = value.max()
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("QGW max normalization requires a finite positive maximum")
        value = value / scale
    if sparse.issparse(value):
        result = value.tocsr()
        result.data[np.abs(result.data) <= threshold] = 0.0
        result.eliminate_zeros()
        return result
    value[np.abs(value) <= threshold] = 0.0
    return sparse.csr_matrix(value)
