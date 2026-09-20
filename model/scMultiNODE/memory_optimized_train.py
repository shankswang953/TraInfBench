"""Narrow, fail-closed storage substitutions around the pinned official trainer.

The upstream file is never edited. An in-memory AST clone replaces only distance
storage/normalization and sparse thresholding. All training statements, loss
constants, RNG calls, optimizer updates and the original QGW solvers remain.
This is an implementation adaptation, not an unmodified-upstream execution.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
import time
from functools import partial
from pathlib import Path

from memory_distances import DiskBackedDistances, blockwise_nan_to_num_and_normalize
from memory_sparse_qgw import make_memory_qgw, compact_threshold


def make_memory_trainer(running, scratch_dir, *, row_block_size=128, progress=None,
                        qgw_callback=None):
    from optim import quantizedGW

    source = textwrap.dedent(inspect.getsource(running.scMultiNODETrain))
    tree = ast.parse(source)
    replacements = {
        "C1 = np.nan_to_num(C1, nan=0.0)": "C1 = _block_normalize(C1)",
        "C2 = np.nan_to_num(C2, nan=0.0)": "C2 = _block_normalize(C2)",
        "C1 /= np.max(C1)": "pass",
        "C2 /= np.max(C2)": "pass",
        "sgw = sgw / sgw.max()": "sgw = _compact_threshold(sgw)",
        "sgw[np.abs(sgw) <= 0.01] = 0.0": "pass",
    }
    counts = {key: 0 for key in replacements}

    class RewriteStorage(ast.NodeTransformer):
        def visit_Assign(self, node):
            return self.replace(node)

        def visit_AugAssign(self, node):
            return self.replace(node)

        def replace(self, node):
            key = ast.unparse(node)
            if key not in replacements:
                return node
            counts[key] += 1
            return ast.copy_location(ast.parse(replacements[key]).body[0], node)

    tree = RewriteStorage().visit(tree)
    if any(count != 1 for count in counts.values()):
        raise RuntimeError(f"Pinned trainer storage statements changed: {counts}")
    ast.fix_missing_locations(tree)
    distances = DiskBackedDistances(Path(scratch_dir), row_block_size=row_block_size,
                                    bounded_mapping=True)
    qgw = make_memory_qgw(quantizedGW, row_block_size=row_block_size,
                          max_dense_output_bytes=512 * 1024**2,
                          release_memmap_pages=True)
    qgw_timing = {}

    def timed_qgw(*args, **kwargs):
        start = time.monotonic()
        try:
            result = qgw(*args, **kwargs)
        finally:
            qgw_timing["elapsed_seconds"] = time.monotonic()-start
        if qgw_callback is not None:
            qgw_callback({"qgw_timing": qgw_timing,
                          "qgw_solver_audit": qgw.solver_audit,
                          "qgw_raw_coupling_audit": qgw.raw_coupling_audit})
        return result

    namespace = dict(running.scMultiNODETrain.__globals__)
    namespace.update(
        _mod_distance=distances,
        qGW=timed_qgw,
        _block_normalize=partial(blockwise_nan_to_num_and_normalize, row_block_size=row_block_size),
        _compact_threshold=compact_threshold,
    )
    if progress is not None:
        namespace["tqdm"] = progress
    exec(compile(tree, "<scMultiNODE-storage-adapter>", "exec"), namespace)
    return namespace["scMultiNODETrain"], distances, {
        "official_file_modified": False,
        "runtime_storage_substitutions": replacements,
        "row_block_size": row_block_size,
        "distance_mapping_strategy": "open_close_per_operation",
        "qgw_timing": qgw_timing,
        "qgw_memory_advice": qgw.memory_advice,
        "qgw_solver_audit": qgw.solver_audit,
        "qgw_raw_coupling_audit": qgw.raw_coupling_audit,
        "method_or_hyperparameter_changes": False,
        "equivalence_requires_validation": True,
    }
