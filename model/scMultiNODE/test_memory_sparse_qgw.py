"""Bounded numerical equivalence tests against the pinned official QGW code."""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "scMultiNODE"))
from optim import quantizedGW as official
from optim.running import _extractAlignIdx

from memory_sparse_qgw import ColumnCoupling, compact_threshold, compress_graph_sparse, make_memory_qgw


def as_dense(value):
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


class SparseQGWTests(unittest.TestCase):
    def test_compression_exact_including_ties_and_landmark_order(self):
        rng = np.random.default_rng(47)
        for n in (8, 21):
            distance = rng.integers(0, 4, (n, n)).astype(float)
            np.fill_diagonal(distance, 0)
            mass = rng.random(n)
            mass /= mass.sum()
            # Deliberately unsorted: equal distances must use the same first tie.
            landmarks = [n - 1, 2, 0, 4]
            expected, expected_mass = official.compress_graph_from_subset_point_cloud(distance, mass, landmarks)
            for block_size in (1, 3, 512):
                with self.subTest(n=n, block_size=block_size):
                    actual, actual_mass = compress_graph_sparse(
                        distance, mass, landmarks, renormalize_prob=official.renormalize_prob,
                        row_block_size=block_size,
                    )
                    self.assertIsInstance(actual, ColumnCoupling)
                    np.testing.assert_array_equal(actual_mass, expected_mass)
                    for column in range(n):
                        np.testing.assert_array_equal(actual[:, column], expected[:, column])
                    np.testing.assert_array_equal(actual[:, -1], expected[:, -1])
                    self.assertLess(actual.storage_bytes, n * n * 8)
                    with self.assertRaises(TypeError):
                        np.asarray(actual)

    def test_qgw_exact_unequal_sizes_and_nonuniform_masses(self):
        rng = np.random.default_rng(39)
        d1 = cdist(rng.normal(size=(24, 3)), rng.normal(size=(24, 3)))
        d1 = (d1 + d1.T) / 2
        np.fill_diagonal(d1, 0)
        d2 = cdist(rng.normal(size=(17, 2)), rng.normal(size=(17, 2)))
        d2 = (d2 + d2.T) / 2
        np.fill_diagonal(d2, 0)
        p1, p2 = rng.random(24), rng.random(17)
        p1 /= p1.sum()
        p2 /= p2.sum()
        args = (d1, d2, p1, p2, [15, 1, 22, 8, 3], [12, 0, 7, 16])
        expected = official.compressed_gw_point_cloud(*args, return_dense=False)
        module_function = official.compress_graph_from_subset_point_cloud
        for block_size in (1, 5, 512):
            with self.subTest(block_size=block_size):
                actual = make_memory_qgw(official, row_block_size=block_size)(*args, return_dense=False)
                np.testing.assert_array_equal(as_dense(actual), as_dense(expected))
                self.assertIs(official.compress_graph_from_subset_point_cloud, module_function)
                self.assertIs(official.compressed_gw_point_cloud.__globals__["compress_graph_from_subset_point_cloud"], module_function)
                expected_compact, actual_compact = compact_threshold(expected), compact_threshold(actual)
                for lhs, rhs in ((expected_compact, actual_compact), (expected_compact.T, actual_compact.T)):
                    for index in (np.arange(lhs.shape[0]), np.array([0, 2, 4])):
                        old_indices, new_indices = _extractAlignIdx(lhs, index), _extractAlignIdx(rhs, index)
                        for old, new in zip(old_indices, new_indices):
                            np.testing.assert_array_equal(old, new)

    def test_qgw_tied_disconnected_graph_distances(self):
        def graph_distance(n):
            graph = np.zeros((n, n))
            for start, stop in ((0, n // 2), (n // 2, n)):
                for index in range(start, stop - 1):
                    graph[index, index + 1] = graph[index + 1, index] = 1
            distance = dijkstra(sparse.csr_matrix(graph), directed=False)
            distance[~np.isfinite(distance)] = distance[np.isfinite(distance)].max()
            distance /= distance.max()
            return distance

        args = (graph_distance(14), graph_distance(12), np.ones(14) / 14, np.ones(12) / 12,
                [12, 1, 5, 8], [0, 4, 8, 11])
        for return_dense in (False, True):
            with self.subTest(return_dense=return_dense):
                expected = official.compressed_gw_point_cloud(*args, return_dense=return_dense)
                actual = make_memory_qgw(official, row_block_size=3)(*args, return_dense=return_dense)
                np.testing.assert_array_equal(as_dense(actual), as_dense(expected))

    def test_entropic_option_preserved(self):
        coordinate = np.arange(8)[:, None].astype(float)
        distance = cdist(coordinate, coordinate)
        args = (distance, distance, np.ones(8) / 8, np.ones(8) / 8, [0, 3, 7], [0, 3, 7])
        expected = official.compressed_gw_point_cloud(*args, return_dense=False, gw_type="egw", epsilon=.5)
        actual = make_memory_qgw(official)(*args, return_dense=False, gw_type="egw", epsilon=.5)
        np.testing.assert_array_equal(as_dense(actual), as_dense(expected))

    def test_compact_threshold_matches_official_dense_values_and_alignment(self):
        values = np.array([[2., .01, 0., 0.], [0., .009, .25, 0.], [0., 0., 0., 0.], [.2, .2, .2, 0.]])
        for constructor in (np.array, sparse.csr_matrix, sparse.csc_matrix):
            for normalize in (False, True):
                with self.subTest(constructor=constructor, normalize=normalize):
                    source = constructor(values)
                    old = source.copy()
                    if normalize:
                        old = old / old.max()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", sparse.SparseEfficiencyWarning)
                        old[np.abs(old) <= .01] = 0.
                    old = sparse.csr_matrix(old)
                    actual = compact_threshold(source, normalize=normalize)
                    np.testing.assert_array_equal(actual.toarray(), old.toarray())
                    np.testing.assert_array_equal(as_dense(source), values)
                    self.assertEqual(actual.nnz, np.count_nonzero(old.toarray()))
                    for lhs, rhs in ((old, actual), (old.T, actual.T)):
                        index = np.arange(lhs.shape[0])
                        for before, after in zip(_extractAlignIdx(lhs, index), _extractAlignIdx(rhs, index)):
                            np.testing.assert_array_equal(before, after)

    def test_dense_fallback_preserved_or_refused_before_allocation(self):
        distance = np.ones((5, 5)) - np.eye(5)
        mass = np.ones(5) / 5
        args = (distance, distance, mass, mass, list(range(5)), list(range(5)))
        fake_result = (np.ones((5, 5)) / 25, {})
        with patch.object(official.ot.gromov, "gromov_wasserstein", return_value=fake_result) as solve:
            expected = official.compressed_gw_point_cloud(*args, return_dense=False)
            reference_call = solve.call_args
            actual = make_memory_qgw(official, max_dense_output_bytes=200)(*args, return_dense=False)
            optimized_call = solve.call_args
            np.testing.assert_array_equal(expected, actual)
            self.assertEqual(reference_call.kwargs, optimized_call.kwargs)
            for left, right in zip(reference_call.args, optimized_call.args):
                np.testing.assert_array_equal(left, right)
            guarded = make_memory_qgw(official, max_dense_output_bytes=199)
            with self.assertRaisesRegex(MemoryError, "dense outer-product fallback"):
                guarded(*args, return_dense=False)
            self.assertEqual(guarded.solver_audit[-1]["status"], "dense_fallback_refused")
            self.assertTrue(guarded.raw_coupling_audit[-1]["dense_fallback_refused"])
            with self.assertRaisesRegex(MemoryError, "Requested dense QGW"):
                guarded(*args, return_dense=True)
            self.assertEqual(guarded.raw_coupling_audit[-1]["status"], "requested_dense_output_refused")

    def test_solver_telemetry_preserves_values_and_reports_all_warnings(self):
        distance = cdist(np.arange(9)[:, None], np.arange(9)[:, None])
        mass = np.ones(9) / 9
        args = (distance, distance, mass, mass, [0, 4, 8], [0, 4, 8])
        expected = official.compressed_gw_point_cloud(*args, return_dense=False)
        original_solve = official.ot.gromov.gromov_wasserstein
        solver_returns = []

        def warning_solve(*args, **kwargs):
            result = original_solve(*args, **kwargs)
            solver_returns.append(result)
            for _ in range(2):
                warnings.warn("numItermax reached before optimality. Try to increase numItermax.", UserWarning)
            return result

        qgw = make_memory_qgw(official)
        with patch.object(official.ot.gromov, "gromov_wasserstein", side_effect=warning_solve):
            with warnings.catch_warnings(record=True) as displayed:
                warnings.simplefilter("always")
                actual = qgw(*args, return_dense=False)
        np.testing.assert_array_equal(as_dense(actual), as_dense(expected))
        self.assertEqual(len(displayed), 2)  # Re-emitted, not hidden.
        self.assertEqual(len(solver_returns), 1)
        self.assertEqual(len(qgw.solver_audit), 1)
        record = qgw.solver_audit[0]
        self.assertEqual(record["status"], "returned")
        self.assertEqual(record["warning_count"], 2)
        self.assertEqual(record["iteration_limit_warning_count"], 2)
        self.assertEqual(record["log"]["loss_count"], len(solver_returns[0][1]["loss"]))
        self.assertTrue(record["log"]["final_inner_status_only"])
        self.assertTrue(record["coupling"]["finite"])
        self.assertLess(record["coupling"]["row_marginal_max_abs_error"], 1e-12)
        self.assertLess(record["coupling"]["column_marginal_max_abs_error"], 1e-12)
        raw = qgw.raw_coupling_audit[0]
        self.assertEqual(raw["phase"], "raw_qgw_before_max_normalization_and_threshold")
        self.assertEqual(raw["status"], "returned")
        self.assertTrue(raw["coupling"]["sparse"])
        self.assertAlmostEqual(raw["coupling"]["mass"], 1.)
        self.assertLess(raw["coupling"]["row_marginal_max_abs_error"], 1e-12)
        self.assertGreater(record["duration_seconds"], 0.)
        json.dumps({"solver_audit": qgw.solver_audit, "raw_coupling_audit": qgw.raw_coupling_audit})

    def test_solver_log_telemetry_is_bounded(self):
        from memory_sparse_qgw import _solver_log_audit
        source = {"loss": np.arange(10001., 0., -1.), "result_code": np.int64(1),
                  "warning": None, "gw_dist": np.float64(1.)}
        summary = _solver_log_audit(source)
        self.assertEqual(summary["loss_count"], 10001)
        self.assertEqual(summary["loss_initial"], 10001.)
        self.assertEqual(summary["loss_final"], 1.)
        self.assertNotIn("loss_values", summary)
        self.assertEqual(len(summary["loss_head"]), 5)
        self.assertEqual(len(summary["loss_tail"]), 5)
        self.assertTrue(summary["loss_values_omitted"])
        self.assertEqual(summary["result_code"], 1)
        np.testing.assert_array_equal(source["loss"], np.arange(10001., 0., -1.))
        json.dumps(summary)

    def test_no_rng_consumption_change(self):
        distance = cdist(np.arange(9)[:, None], np.arange(9)[:, None])
        args = (distance, distance, np.ones(9) / 9, np.ones(9) / 9, [0, 4, 8], [0, 4, 8])
        np.random.seed(818)
        original = official.compressed_gw_point_cloud(*args, return_dense=False)
        after_original = np.random.get_state()
        np.random.seed(818)
        actual = make_memory_qgw(official)(*args, return_dense=False)
        after_actual = np.random.get_state()
        np.testing.assert_array_equal(as_dense(original), as_dense(actual))
        for before, after in zip(after_original, after_actual):
            np.testing.assert_array_equal(before, after)

    def test_memmap_release_preserves_values_and_coupling(self):
        distance = cdist(np.arange(18)[:, None], np.arange(18)[:, None])
        mass = np.ones(18) / 18
        landmarks = [0, 5, 12, 17]
        expected = official.compressed_gw_point_cloud(
            distance, distance, mass, mass, landmarks, landmarks, return_dense=False,
        )
        with tempfile.TemporaryDirectory(prefix="scmultinode-qgw-memmap-") as directory:
            path = Path(directory) / "distance.dat"
            mapped = np.memmap(path, dtype=np.float64, mode="w+", shape=distance.shape)
            mapped[:] = distance  # Deliberately not explicitly flushed here.
            qgw = make_memory_qgw(official, row_block_size=3, release_memmap_pages=True, release_interval=1)
            actual = qgw(mapped, mapped, mass, mass, landmarks, landmarks, return_dense=False)
            np.testing.assert_array_equal(as_dense(actual), as_dense(expected))
            np.testing.assert_array_equal(mapped, distance)
            self.assertEqual(qgw.memory_advice["failed"], 0)
            if qgw.memory_advice["supported"]:
                self.assertGreater(qgw.memory_advice["succeeded"], 12)
                self.assertEqual(qgw.memory_advice["attempts"], qgw.memory_advice["succeeded"])
            # Read-only maps are safe to advise too; persisted bytes survived.
            readonly = np.memmap(path, dtype=np.float64, mode="r", shape=distance.shape)
            again = qgw(readonly, readonly, mass, mass, landmarks, landmarks, return_dense=False)
            np.testing.assert_array_equal(as_dense(again), as_dense(expected))
            np.testing.assert_array_equal(readonly, distance)

    def test_copy_on_write_mapping_is_not_discarded(self):
        distance = cdist(np.arange(12)[:, None], np.arange(12)[:, None])
        mass = np.ones(12) / 12
        landmarks = [0, 5, 11]
        with tempfile.TemporaryDirectory(prefix="scmultinode-qgw-cow-") as directory:
            path = Path(directory) / "distance.dat"
            original = np.memmap(path, dtype=np.float64, mode="w+", shape=distance.shape)
            original[:] = distance
            original.flush()
            private = np.memmap(path, dtype=np.float64, mode="c", shape=distance.shape)
            private[:] = distance * 2
            qgw = make_memory_qgw(official, row_block_size=3, release_memmap_pages=True)
            qgw(private, private, mass, mass, landmarks, landmarks, return_dense=False)
            self.assertEqual(qgw.memory_advice["attempts"], 0)
            self.assertGreater(qgw.memory_advice["copy_on_write_skipped"], 0)
            np.testing.assert_array_equal(private, distance * 2)
            np.testing.assert_array_equal(original, distance)


if __name__ == "__main__":
    unittest.main(verbosity=2)
