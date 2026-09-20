"""Small exact-equivalence tests against the clean official distance routine."""
from __future__ import annotations

import gc
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

# Shared-environment initialization must precede scientific-package imports.
from run_gastrulation import ROOT
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph

import memory_distances
from memory_distances import (
    DiskBackedDistances,
    DiskDistanceArray,
    blockwise_nan_to_num_and_normalize,
    estimate_distance_resources,
    flush_and_drop_pages,
)


class MemoryDistanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        upstream = ROOT / "external/scMultiNODE"
        sys.path.insert(0, str(upstream))
        from optim import running

        if Path(inspect.getsourcefile(running._mod_distance)).resolve() != (upstream / "optim/running.py").resolve():
            raise AssertionError("Expected the actual official repository distance function")
        cls.official_distance = staticmethod(running._mod_distance)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="scmultinode-distances-")
        self.addCleanup(self.tmp.cleanup)
        self.scratch = Path(self.tmp.name)
        self.rng = np.random.default_rng(714)

    def assert_official_equivalence(self, features, neighbors, prefix):
        expected = self.official_distance(features, neighbors, "correlation")
        adapter = DiskBackedDistances(self.scratch, row_block_size=3, prefix=prefix)
        with patch.object(memory_distances, "dijkstra", wraps=memory_distances.dijkstra) as solver:
            actual = adapter(features, neighbors, "correlation")
        self.assertIsInstance(actual, np.memmap)
        self.assertEqual(actual.dtype, np.float64)
        np.testing.assert_array_equal(actual, expected)
        for call in solver.call_args_list:
            self.assertIn("indices", call.kwargs)
            self.assertLessEqual(len(call.kwargs["indices"]), 3)
            self.assertFalse(call.kwargs["directed"])
            self.assertFalse(call.kwargs["return_predecessors"])
        self.assertEqual(adapter.records[0]["status"], "complete")
        self.assertEqual(adapter.records[0]["global_finite_max"], float(expected.max()))
        if expected.max() != 0:
            expected = np.nan_to_num(expected, nan=0)
            expected /= np.max(expected)
            self.assertIs(blockwise_nan_to_num_and_normalize(actual, 2), actual)
            np.testing.assert_array_equal(actual, expected)
        self.assertFalse(actual._mmap.closed)
        return actual

    def test_connected_graph_exact_equivalence(self):
        features = self.rng.normal(size=(24, 8))
        graph = kneighbors_graph(features, 8, mode="connectivity", metric="correlation", include_self=True)
        self.assertEqual(connected_components(graph, directed=False)[0], 1)
        self.assert_official_equivalence(features, 8, "connected")

    def test_disconnected_graph_uses_global_finite_maximum(self):
        # Two disconnected components with different sizes/diameters ensure
        # early source-row blocks cannot dictate the replacement maximum.
        base = np.arange(8, dtype=np.float64)
        features = np.vstack([
            base + .05 * self.rng.normal(size=(5, 8)),
            -base + .05 * self.rng.normal(size=(13, 8)),
        ])
        graph = kneighbors_graph(features, 3, mode="connectivity", metric="correlation", include_self=True)
        self.assertGreaterEqual(connected_components(graph, directed=False)[0], 2)
        result = self.assert_official_equivalence(features, 3, "disconnected")
        self.assertTrue(np.isfinite(result).all())

    def test_tied_neighbor_distances_exact_equivalence(self):
        features = np.tile(np.eye(5, dtype=np.float32), (3, 1))
        self.assert_official_equivalence(features, 4, "ties")

    def test_all_disconnected_nodes_match_raw_zero_output(self):
        features = self.rng.normal(size=(8, 4))
        actual = self.assert_official_equivalence(features, 1, "isolated")
        np.testing.assert_array_equal(actual, np.zeros((8, 8)))
        with self.assertRaisesRegex(ValueError, "zero"):
            blockwise_nan_to_num_and_normalize(actual, 3)

    def test_mapping_survives_factory_and_flush_advice(self):
        adapter = DiskBackedDistances(self.scratch, row_block_size=2)
        result = adapter(self.rng.normal(size=(9, 5)), 4, "correlation")
        path = Path(adapter.records[0]["path"])
        view = result[1:4]
        del adapter
        gc.collect()
        self.assertIsInstance(flush_and_drop_pages(result), bool)
        self.assertFalse(result._mmap.closed)
        view[0, 0] = 123.5
        result.flush()
        reopened = np.load(path, mmap_mode="r")
        self.assertEqual(float(reopened[1, 0]), 123.5)
        np.testing.assert_array_equal(result, reopened)

    def test_copy_on_write_pages_are_never_flushed_or_evicted(self):
        path = self.scratch / "copy_on_write.npy"
        shared = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(3, 3))
        shared[:] = np.arange(9).reshape(3, 3)
        shared.flush()
        private = np.load(path, mmap_mode="c")
        private[1, 2] = 987.5
        with patch.object(private, "flush", wraps=private.flush) as flush:
            self.assertFalse(flush_and_drop_pages(private))
            flush.assert_not_called()
        self.assertEqual(float(private[1, 2]), 987.5)
        self.assertEqual(float(shared[1, 2]), 5.)
        self.assertEqual(float(np.load(path)[1, 2]), 5.)
        self.assertFalse(private._mmap.closed)

    def test_bounded_mapping_matches_official_and_closes_every_map(self):
        base = np.arange(6, dtype=np.float64)
        examples = [
            ("bounded_connected", self.rng.normal(size=(18, 6)), 7),
            ("bounded_disconnected", np.vstack([
                base + .02 * self.rng.normal(size=(5, 6)),
                -base + .02 * self.rng.normal(size=(11, 6)),
            ]), 3),
            ("bounded_tied", np.tile(np.eye(5, dtype=np.float32), (3, 1)), 4),
        ]
        original_load = np.load
        original_create = np.lib.format.open_memmap
        mapped = []

        def record_load(*args, **kwargs):
            mapping = original_load(*args, **kwargs)
            if isinstance(mapping, np.memmap):
                mapped.append(mapping)
            return mapping

        def record_create(*args, **kwargs):
            mapping = original_create(*args, **kwargs)
            mapped.append(mapping)
            return mapping

        for prefix, features, neighbors in examples:
            with self.subTest(prefix=prefix):
                expected = self.official_distance(features, neighbors, "correlation")
                with patch.object(memory_distances.np, "load", side_effect=record_load), \
                     patch.object(memory_distances.np.lib.format, "open_memmap", side_effect=record_create):
                    adapter = DiskBackedDistances(self.scratch, row_block_size=3, prefix=prefix, bounded_mapping=True)
                    actual = adapter(features, neighbors, "correlation")
                    self.assertIsInstance(actual, DiskDistanceArray)
                    self.assertEqual(actual.active_mappings, 0)
                    self.assertTrue(all(mapping._mmap.closed for mapping in mapped))
                    np.testing.assert_array_equal(np.asarray(actual), expected)
                    expected = np.nan_to_num(expected, nan=0)
                    expected /= np.max(expected)
                    self.assertIs(blockwise_nan_to_num_and_normalize(actual, 2), actual)
                    np.testing.assert_array_equal(np.asarray(actual), expected)
                    self.assertEqual(actual.active_mappings, 0)
                    self.assertEqual(adapter.records[0]["mapping_strategy"], "open_close_per_operation")
                self.assertTrue(all(mapping._mmap.closed for mapping in mapped))

    def test_disk_owner_returns_owned_copies_and_persists_writes(self):
        features = self.rng.normal(size=(9, 5))
        actual = DiskBackedDistances(self.scratch, bounded_mapping=True)(features, 4, "correlation")
        original = actual[0:2, :]
        self.assertIs(type(original), np.ndarray)
        self.assertTrue(original.flags.owndata)
        edited = original.copy()
        edited[0, 0] = 321.5
        self.assertEqual(float(actual[0, 0]), float(original[0, 0]))
        actual[0:2, :] = edited
        np.testing.assert_array_equal(actual[0:2, :], edited)
        self.assertEqual(float(np.load(actual.path)[0, 0]), 321.5)
        self.assertEqual(actual.active_mappings, 0)
        with self.assertRaises(IndexError):
            actual[999, :]
        self.assertEqual(actual.active_mappings, 0)
        with self.assertRaises(ValueError):
            actual[0:2, :] = np.ones((6, 6))
        self.assertEqual(actual.active_mappings, 0)

    def test_disk_owner_rejects_unbudgeted_full_coercion(self):
        path = self.scratch / "copy_guard.npy"
        mapping = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(3, 3))
        mapping[:] = np.arange(9).reshape(3, 3)
        mapping.flush()
        mapping._mmap.close()
        actual = DiskDistanceArray(path, max_array_bytes=8)
        with patch.object(memory_distances.np, "load") as loader:
            with self.assertRaises(MemoryError):
                np.asarray(actual)
            with self.assertRaises(MemoryError):
                actual.to_numpy(max_bytes=71)
            loader.assert_not_called()
        np.testing.assert_array_equal(actual.to_numpy(max_bytes=72), np.arange(9).reshape(3, 3))
        self.assertEqual(actual.active_mappings, 0)

    def test_disk_owner_nan_inf_normalization_matches_official(self):
        original = self.rng.normal(size=(7, 9))
        original[0, 1], original[4, 3], original[6, 8] = np.nan, np.inf, -np.inf
        expected = np.nan_to_num(original, nan=0)
        expected /= np.max(expected)
        path = self.scratch / "owner_normalization.npy"
        mapping = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=original.shape)
        mapping[:] = original
        mapping.flush()
        mapping._mmap.close()
        actual = DiskDistanceArray(path)
        self.assertIs(blockwise_nan_to_num_and_normalize(actual, 2), actual)
        np.testing.assert_array_equal(np.asarray(actual), expected)
        self.assertEqual(actual.active_mappings, 0)

    def test_exclusive_output_files_are_not_overwritten(self):
        features = self.rng.normal(size=(8, 4))
        first = DiskBackedDistances(self.scratch, row_block_size=3)
        result = first(features, 3, "correlation")
        path = Path(first.records[0]["path"])
        contents = path.read_bytes()
        with self.assertRaises(FileExistsError):
            DiskBackedDistances(self.scratch)(features, 3, "correlation")
        self.assertEqual(path.read_bytes(), contents)
        second = first(features, 3, "correlation")
        self.assertNotEqual(first.records[0]["path"], first.records[1]["path"])
        np.testing.assert_array_equal(result, second)

    def test_blockwise_nan_inf_replacement_and_global_normalization(self):
        original = self.rng.normal(size=(7, 9))
        original[0, 1], original[4, 3], original[6, 8] = np.nan, np.inf, -np.inf
        expected = np.nan_to_num(original, nan=0)
        expected /= np.max(expected)
        path = self.scratch / "normalization.npy"
        actual = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=original.shape)
        actual[:] = original
        self.assertIs(blockwise_nan_to_num_and_normalize(actual, 2), actual)
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(actual._mmap.closed)
        self.assertTrue(np.isfinite(actual).all())
        dense = original.copy()
        self.assertIs(blockwise_nan_to_num_and_normalize(dense, 3), dense)
        np.testing.assert_array_equal(dense, expected)

    def test_preflight_budget_and_disk_checks_precede_graph_allocation(self):
        features = self.rng.normal(size=(8, 4))
        with patch.object(memory_distances, "kneighbors_graph") as graph_builder:
            with self.assertRaises(MemoryError):
                DiskBackedDistances(self.scratch, max_workspace_bytes=1)(features, 3, "correlation")
            graph_builder.assert_not_called()
            with patch.object(memory_distances.shutil, "disk_usage") as disk:
                disk.return_value.free = 1
                with self.assertRaisesRegex(OSError, "scratch"):
                    DiskBackedDistances(self.scratch, min_free_bytes=0)(features, 3, "correlation")
            graph_builder.assert_not_called()
        self.assertEqual(list(self.scratch.iterdir()), [])
        plan = estimate_distance_resources(35503, 10, 10)
        self.assertEqual(plan["mapped_array_bytes"], 8 * 35503**2)
        self.assertEqual(plan["row_block_size"], 128)
        self.assertLess(plan["estimated_workspace_bytes"], 4 * 1024**3)

    def test_dtype_and_zero_normalization_safety(self):
        with self.assertRaisesRegex(ValueError, "float64"):
            blockwise_nan_to_num_and_normalize(np.ones((3, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "zero"):
            blockwise_nan_to_num_and_normalize(np.zeros((3, 3), dtype=np.float64))


if __name__ == "__main__":
    unittest.main(verbosity=2)
