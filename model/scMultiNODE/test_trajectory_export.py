"""Small CPU checks of native Euler exports against the official forward.

Run with the shared environment's Python; no training or benchmark data needed.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from trajectory_export import export_trajectories


class TrajectoryExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        upstream = Path(__file__).resolve().parents[2] / "external/scMultiNODE"
        sys.path.insert(0, str(upstream))
        from model.diff_solver import ODE
        from model.dynamic_model import scMultiNODE
        from model.layer import LinearNet

        # Real official model and solver, small dimensions for a fast smoke.
        # Identity-final decoders match the signed-embedding adapter.
        torch.set_num_threads(1)
        torch.manual_seed(714)
        cls.model = scMultiNODE(
            n_genes=3, n_peaks=2, latent_dim=4, anchor_mod="rna",
            rna_enc=LinearNet(3, [5], 4, "relu"),
            rna_dec=LinearNet(4, [5], 3, "tanh"),
            atac_enc=LinearNet(2, [5], 4, "relu"),
            atac_dec=LinearNet(4, [5], 2, "tanh"),
            fusion_layer=LinearNet(4, [5], 4, "relu"),
            diffeq_decoder=ODE(4, LinearNet(4, [5], 4, "tanh"), "euler"),
        )
        for decoder in (cls.model.rna_dec, cls.model.atac_dec):
            decoder.net[-1] = torch.nn.Identity()
        cls.initial = torch.randn(17, 3)
        cls.indices = np.arange(17, dtype=np.int64)[::-1].copy()
        cls.query = np.linspace(0, 2.5, 101, dtype=np.float32)

    def test_full_and_strict_leave_one_out(self):
        for train in ([0, 1, 2, 2.5], [0, 2, 2.5], [0, 1, 2.5]):
            with self.subTest(train=train), tempfile.TemporaryDirectory(prefix="scmultinode-export-") as tmp:
                target = Path(tmp) / "trajectories.npz"
                before = {key: value.clone() for key, value in self.model.state_dict().items()}
                self.model.train()
                audit = export_trajectories(
                    self.model, self.initial, self.indices, train, self.query, target, batch_size=6
                )
                self.assertTrue(self.model.training)
                self.assertEqual(audit["n_initial"], 17)
                self.assertEqual(audit["inference_batches"], 3)
                self.assertTrue(audit["all_supplied_initial_once"])
                self.assertEqual(audit["training_times"], train)
                self.assertLessEqual(max(audit["native_training_grid_max_abs_error"].values()), 1e-5)
                for key, value in self.model.state_dict().items():
                    self.assertTrue(torch.equal(value, before[key]))
                with torch.no_grad():
                    native = self.model(self.initial, torch.tensor(train), torch.tensor(train))
                with np.load(target) as saved:
                    self.assertEqual(
                        set(saved.files),
                        {"rna_norm", "atac_norm", "latent", "time", "initial_rna_norm",
                         "initial_indices", "weights", "training_time"},
                    )
                    np.testing.assert_array_equal(saved["initial_indices"], self.indices)
                    np.testing.assert_array_equal(saved["initial_rna_norm"], self.initial.numpy())
                    np.testing.assert_array_equal(saved["training_time"], train)
                    np.testing.assert_allclose(saved["weights"].sum(axis=1), 1, rtol=0, atol=1e-14)
                    positions = np.searchsorted(self.query, train)
                    for key, reference in (("rna_norm", native[0]), ("atac_norm", native[1]), ("latent", native[3])):
                        self.assertEqual(saved[key].shape[:2], (101, 17))
                        self.assertTrue(np.isfinite(saved[key]).all())
                        np.testing.assert_allclose(
                            saved[key][positions], reference.numpy().transpose(1, 0, 2), rtol=0, atol=1e-5
                        )
                    # Held-out times really use the observed interval's latent
                    # dense output rather than an additional Euler update.
                    if 1 not in train:
                        np.testing.assert_allclose(saved["latent"][40], .5 * (saved["latent"][0] + saved["latent"][80]), atol=1e-7)
                    if 2 not in train:
                        expected = saved["latent"][40] + (2 / 3) * (saved["latent"][100] - saved["latent"][40])
                        np.testing.assert_allclose(saved["latent"][80], expected, atol=1e-7)
                with self.assertRaises(FileExistsError):
                    export_trajectories(self.model, self.initial, self.indices, train, self.query, target)

    def test_query_density_does_not_change_predictions(self):
        train = [0, 2, 2.5]
        with tempfile.TemporaryDirectory(prefix="scmultinode-export-") as tmp:
            dense_path, sparse_path = Path(tmp) / "dense.npz", Path(tmp) / "sparse.npz"
            export_trajectories(self.model, self.initial, self.indices, train, self.query, dense_path, 6)
            export_trajectories(self.model, self.initial, self.indices, train, [0, 1, 2, 2.5], sparse_path, 6)
            with np.load(dense_path) as dense, np.load(sparse_path) as sparse:
                for key in ("rna_norm", "atac_norm", "latent"):
                    np.testing.assert_allclose(dense[key][[0, 40, 80, 100]], sparse[key], rtol=0, atol=1e-6)

    def test_reject_duplicate_initial_cells(self):
        with tempfile.TemporaryDirectory(prefix="scmultinode-export-") as tmp:
            target = Path(tmp) / "bad.npz"
            with self.assertRaisesRegex(ValueError, "unique"):
                export_trajectories(self.model, self.initial, np.zeros(17, dtype=np.int64), [0, 2.5], self.query, target)
            self.assertFalse(target.exists())

    def test_reject_extrapolation(self):
        with tempfile.TemporaryDirectory(prefix="scmultinode-export-") as tmp:
            target = Path(tmp) / "bad.npz"
            with self.assertRaisesRegex(ValueError, "within"):
                export_trajectories(self.model, self.initial, self.indices, [0, 2.5], [0, 3], target)
            self.assertFalse(target.exists())

    def test_reject_nonfinite_initial_cells(self):
        initial = self.initial.clone()
        initial[1, 0] = float("nan")
        with tempfile.TemporaryDirectory(prefix="scmultinode-export-") as tmp:
            with self.assertRaises(FloatingPointError):
                export_trajectories(self.model, initial, self.indices, [0, 2.5], self.query, Path(tmp) / "bad.npz")


if __name__ == "__main__":
    unittest.main(verbosity=2)
