"""Structural safety checks for the storage-only trainer clone; no training."""
from __future__ import annotations

import ast
import builtins
import inspect
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "scMultiNODE"))

from optim import running
import memory_optimized_train as adapter


class StorageTrainerStructureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="scmultinode-clone-audit-")
        self.addCleanup(temporary.cleanup)
        self.scratch = Path(temporary.name)

    def test_clone_does_not_mutate_original_globals_or_signature(self):
        original = running.scMultiNODETrain
        before = original.__globals__.copy()
        unique_progress = object()
        clone, distances, metadata = adapter.make_memory_trainer(
            running, self.scratch, progress=unique_progress,
        )
        self.assertIs(running.scMultiNODETrain, original)
        self.assertEqual(set(original.__globals__), set(before))
        for name, value in before.items():
            self.assertIs(original.__globals__[name], value, name)
        self.assertIsNot(clone.__globals__, original.__globals__)
        self.assertEqual(inspect.signature(clone), inspect.signature(original))
        self.assertIs(clone.__globals__["tqdm"], unique_progress)
        self.assertIs(clone.__globals__["_mod_distance"], distances)
        allowed = {"scMultiNODETrain", "_mod_distance", "qGW", "tqdm"}
        for name, value in before.items():
            if name not in allowed:
                self.assertIs(clone.__globals__[name], value, name)
        self.assertFalse(metadata["official_file_modified"])
        self.assertFalse(metadata["method_or_hyperparameter_changes"])
        self.assertTrue(metadata["equivalence_requires_validation"])
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_exactly_six_storage_statements_change_and_all_losses_remain(self):
        captured = []

        def capture_compile(tree, filename, mode, *args, **kwargs):
            if filename == "<scMultiNODE-storage-adapter>":
                captured.append(tree)
            return builtins.compile(tree, filename, mode, *args, **kwargs)

        with patch.object(adapter, "compile", side_effect=capture_compile, create=True):
            adapter.make_memory_trainer(running, self.scratch)
        self.assertEqual(len(captured), 1)
        actual = captured[0].body[0]
        original = ast.parse(textwrap.dedent(inspect.getsource(running.scMultiNODETrain))).body[0]
        self.assertEqual(ast.dump(actual.args), ast.dump(original.args))
        substitutions = {
            "C1 = np.nan_to_num(C1, nan=0.0)": "C1 = _block_normalize(C1)",
            "C2 = np.nan_to_num(C2, nan=0.0)": "C2 = _block_normalize(C2)",
            "C1 /= np.max(C1)": "pass",
            "C2 /= np.max(C2)": "pass",
            "sgw = sgw / sgw.max()": "sgw = _compact_threshold(sgw)",
            "sgw[np.abs(sgw) <= 0.01] = 0.0": "pass",
        }
        changed = []
        self.assertEqual(len(actual.body), len(original.body))
        for before, after in zip(original.body, actual.body):
            text = ast.unparse(before)
            if text in substitutions:
                expected = ast.parse(substitutions[text]).body[0]
                self.assertEqual(ast.dump(after), ast.dump(expected))
                changed.append(text)
            else:
                self.assertEqual(ast.dump(after), ast.dump(before))
        self.assertEqual(set(changed), set(substitutions))
        self.assertEqual(len(changed), 6)
        loss_calls = [node for node in ast.walk(actual)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                      and node.func.id == "SinkhornLoss"]
        self.assertEqual(len(loss_calls), 4)
        for call in loss_calls:
            keywords = {item.arg: item.value for item in call.keywords}
            self.assertEqual(ast.literal_eval(keywords["batch_size"]), 200)
            self.assertEqual(ast.unparse(keywords["blur"]), "blur")
            self.assertEqual(ast.unparse(keywords["scaling"]), "scaling")

    def test_missing_or_duplicate_upstream_statement_fails_before_execution(self):
        source = textwrap.dedent(inspect.getsource(running.scMultiNODETrain))
        target = "C1 = np.nan_to_num(C1, nan=0.0)"
        self.assertIn(target, source)
        variants = (
            source.replace(target, "C1 = np.nan_to_num(C1, nan=1.0)", 1),
            source.replace(target, target + "\n    " + target, 1),
        )
        for changed_source in variants:
            with self.subTest(source=changed_source.count(target)):
                with patch.object(adapter.inspect, "getsource", return_value=changed_source), \
                        patch.object(adapter, "DiskBackedDistances") as distances:
                    with self.assertRaisesRegex(RuntimeError, "Pinned trainer storage statements changed"):
                        adapter.make_memory_trainer(running, self.scratch)
                    distances.assert_not_called()
        self.assertEqual(list(self.scratch.iterdir()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
