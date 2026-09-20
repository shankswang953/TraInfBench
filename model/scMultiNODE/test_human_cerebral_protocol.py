"""Synthetic-only checks for the seven-time human cerebral scMultiNODE setup.

No benchmark data is modified or used for training. The sole official-model
test performs two inference calls on synthetic data, with no optimization.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import benchmark_gastrulation as benchmark
from dataset_protocols import GASTRULATION, HUMAN_CEREBRAL, PANCREAS
import numpy as np
import torch
from test_pancreas_protocol import ExportTestModel, PadDecoder
from trajectory_export import export_trajectories, query_time_grid


class TinyHumanModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.zeros(()))
        self.n_genes, self.n_peaks, self.latent_dim = 30, 12, 10


class HumanCerebralProtocolTests(unittest.TestCase):
    COUNTS = (70, 73, 77, 81, 83, 86, 89)
    SCALES = {"rna": 2.5, "atac": .125}
    TIMES = (0., .3, .5, .7, .8, 1.4, 1.7)
    STAGES = ("D4", "D7", "D9", "D11", "D12", "D18", "D21")
    KEYS = ("age_4", "age_7", "age_9", "age_11", "age_12", "age_18", "age_21")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="scmultinode-human-protocol-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "input"
        self.data.mkdir()
        rng = np.random.default_rng(618)
        self.raw = {}
        for modality, matrix_name, norm_name, dimension in HUMAN_CEREBRAL.modalities:
            values = {
                key: (rng.normal(size=(count, dimension)) + 10 * stage).astype(np.float32)
                for stage, (key, count) in enumerate(zip(self.KEYS, self.COUNTS))
            }
            self.raw[modality] = values
            np.savez_compressed(self.data / matrix_name, **values)
            torch.save({"scale": self.SCALES[modality]}, self.data / norm_name)

    def load(self, **kwargs):
        return benchmark.load_inputs(self.data, "full", dataset=HUMAN_CEREBRAL, **kwargs)

    def config(self):
        return json.loads((Path(__file__).parent / "configs/human_cerebral_full.json").read_text())

    def test_contract_has_only_selected_seven_times_and_full_scenario(self):
        self.assertEqual(HUMAN_CEREBRAL.name, "human_cerebral_7time_d4_d21_no_d16")
        self.assertEqual(HUMAN_CEREBRAL.times, self.TIMES)
        self.assertEqual(HUMAN_CEREBRAL.stages, self.STAGES)
        self.assertEqual(HUMAN_CEREBRAL.keys, self.KEYS)
        self.assertEqual(HUMAN_CEREBRAL.splits, {"full": list(range(7))})
        self.assertEqual(HUMAN_CEREBRAL.modalities, (
            ("rna", "rna_pca30_by_time.npz", "primal_norm_params.pt", 30),
            ("atac", "atac_lsi12_by_time.npz", "secondary_norm_params_lsi12.pt", 12),
        ))
        for scenario in ("loo1", "loo2", "loo_day7"):
            with self.subTest(scenario=scenario), self.assertRaises(KeyError):
                benchmark.load_inputs(self.data, scenario, dataset=HUMAN_CEREBRAL)

    def test_all_seven_inputs_are_divided_by_their_saved_scale_once(self):
        training, initial, provenance, indices = self.load()
        np.testing.assert_array_equal(initial.numpy(), self.raw["rna"]["age_4"] / 2.5)
        for modality, matrix_name, norm_name, dimension in HUMAN_CEREBRAL.modalities:
            self.assertEqual(len(training[modality]), 7)
            self.assertEqual(list(indices[modality]), list(self.KEYS))
            self.assertEqual(provenance[modality]["full_counts"], list(self.COUNTS))
            self.assertEqual(provenance[modality]["selected_counts"], list(self.COUNTS))
            self.assertEqual(provenance[modality]["training_cells"], sum(self.COUNTS))
            self.assertEqual(provenance[modality]["stage_keys"], list(self.KEYS))
            self.assertEqual(provenance[modality]["scale"], self.SCALES[modality])
            self.assertEqual(Path(provenance[modality]["source"]).name, matrix_name)
            self.assertEqual(Path(provenance[modality]["normalization"]).name, norm_name)
            for stage, (key, values) in enumerate(zip(self.KEYS, training[modality])):
                self.assertEqual(values.shape, (self.COUNTS[stage], dimension))
                self.assertEqual(values.dtype, torch.float32)
                np.testing.assert_array_equal(values.numpy(), self.raw[modality][key] / self.SCALES[modality])
                np.testing.assert_array_equal(indices[modality][key], np.arange(self.COUNTS[stage]))

    def test_all_initial_rows_are_retained_even_for_smoke(self):
        _, expected, _, _ = self.load()
        for smoke in (False, True):
            for seed in (0, 59):
                with self.subTest(smoke=smoke, seed=seed):
                    training, initial, provenance, indices = self.load(smoke=smoke, seed=seed)
                    self.assertTrue(torch.equal(initial, expected))
                    self.assertEqual(initial.shape, (70, 30))
                    for modality in ("rna", "atac"):
                        expected_counts = [64] * 7 if smoke else list(self.COUNTS)
                        self.assertEqual([len(x) for x in training[modality]], expected_counts)
                        self.assertEqual(provenance[modality]["selected_counts"], expected_counts)
                        for chosen in indices[modality].values():
                            self.assertEqual(len(chosen), len(np.unique(chosen)))

    def test_config_changes_only_dataset_identity_and_representation(self):
        config = self.config()
        pancreas = json.loads((Path(__file__).parent / "configs/pancreas_full.json").read_text())
        self.assertEqual(config["dataset"], HUMAN_CEREBRAL.name)
        self.assertEqual(config["input_space"], HUMAN_CEREBRAL.input_space)
        self.assertEqual(config["scenario"], "full")
        for key in ("dataset", "input_space"):
            config.pop(key)
            pancreas.pop(key)
        self.assertEqual(config, pancreas)
        self.assertEqual((config["batch_size"], config["iters"], config["latent_dim"]), (1024, 20000, 10))
        self.assertEqual((config["ae_batch_size"], config["fusion_batch_size"]), (128, 128))

    def test_checkpoint_retains_physical_not_integer_rank_clock(self):
        payload = benchmark.checkpoint_payload(TinyHumanModel(), self.config(), "full", "dynamics", 20000,
                                               dataset=HUMAN_CEREBRAL)
        self.assertEqual(payload["dataset"], HUMAN_CEREBRAL.name)
        self.assertEqual(payload["training_times"], list(self.TIMES))
        self.assertEqual(payload["all_times"], list(self.TIMES))
        self.assertEqual(payload["stage_names"], list(self.STAGES))
        self.assertEqual((payload["rna_dim"], payload["atac_dim"], payload["latent_dim"]), (30, 12, 10))
        self.assertEqual(payload["iteration"], 20000)

    def test_query_grid_has_69_unique_points_and_exact_native_knots(self):
        native = np.asarray(self.TIMES, dtype=np.float32)
        for times in (self.TIMES, native):
            query = query_time_grid(times, .025)
            self.assertEqual(query.dtype, np.float32)
            self.assertEqual(len(query), 69)
            self.assertEqual(len(np.unique(query)), 69)
            self.assertTrue(np.all(np.diff(query) > 0))
            self.assertEqual(query[-1], native[-1])
            for knot in native:
                self.assertEqual(int(np.count_nonzero(query == knot)), 1)
            np.testing.assert_array_equal(query[np.searchsorted(query, native)], native)
        # A genuine off-grid observed time must not disappear during snapping.
        off_grid = np.asarray([0., .31, 1.7], dtype=np.float32)
        query = query_time_grid(off_grid, .025)
        self.assertEqual(len(query), 70)
        self.assertEqual(int(np.count_nonzero(query == off_grid[1])), 1)

    def test_old_gastrulation_and_pancreas_query_grids_are_unchanged(self):
        for dataset, expected_count in ((GASTRULATION, 101), (PANCREAS, 81)):
            with self.subTest(dataset=dataset.name):
                times = np.asarray(dataset.times, dtype=np.float32)
                end = float(times[-1])
                old = np.unique(np.concatenate([
                    np.linspace(0, end, round(end / .025) + 1, dtype=np.float32), times,
                ]))
                new = query_time_grid(times, .025)
                self.assertEqual(len(new), expected_count)
                np.testing.assert_array_equal(new, old)

    def test_export_uses_native_seven_time_solver_grid_and_all_initial_cells(self):
        initial = self.load()[1]
        model = ExportTestModel()
        model.rna_dec = PadDecoder(30)
        model.atac_dec = PadDecoder(12)
        native = np.asarray(self.TIMES, dtype=np.float32)
        query = query_time_grid(native, .025)
        output = self.root / "trajectories.npz"
        audit = export_trajectories(model, initial, np.arange(len(initial)), native, query, output, batch_size=23)
        self.assertTrue(audit["all_supplied_initial_once"])
        self.assertEqual(audit["training_times"], native.tolist())
        self.assertTrue(all(grid == native.tolist() for grid in model.diffeq_decoder.grids))
        self.assertEqual(audit["inference_batches"], 4)
        self.assertLessEqual(max(audit["native_training_grid_max_abs_error"].values()), 1e-5)
        with np.load(output) as saved:
            self.assertEqual(saved["rna_norm"].shape, (69, 70, 30))
            self.assertEqual(saved["atac_norm"].shape, (69, 70, 12))
            self.assertEqual(saved["latent"].shape, (69, 70, 10))
            np.testing.assert_array_equal(saved["time"], query)
            np.testing.assert_array_equal(saved["training_time"], native)
            np.testing.assert_array_equal(saved["initial_indices"], np.arange(70))
            np.testing.assert_array_equal(saved["initial_rna_norm"], initial.numpy())
            np.testing.assert_allclose(saved["weights"].sum(axis=1), 1., rtol=0, atol=1e-14)

    def test_input_auditor_hook_runs_before_training_and_is_saved_in_check_report(self):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config()))
        output = self.root / "must-not-exist"
        capture = []

        def auditor(directory, training, initial, provenance, indices):
            capture.append(directory)
            self.assertEqual(directory, self.data)
            self.assertEqual([len(x) for x in training["rna"]], list(self.COUNTS))
            self.assertEqual(initial.shape, (70, 30))
            self.assertEqual(provenance["atac"]["dimension"], 12)
            self.assertEqual(list(indices["rna"]), list(self.KEYS))
            return {"synthetic_audit_completed": True}

        stdout = io.StringIO()
        with patch.object(benchmark.subprocess, "check_output", side_effect=[benchmark.UPSTREAM_COMMIT, ""]), \
             patch.object(benchmark, "available_memory_gib", return_value=64.), \
             patch.object(benchmark, "make_model") as constructor, \
             contextlib.redirect_stdout(stdout):
            benchmark.main(dataset=HUMAN_CEREBRAL, input_auditor=auditor, argv=[
                "--config", str(config_path), "--data-dir", str(self.data),
                "--output-dir", str(output), "--check",
            ])
            constructor.assert_not_called()
        self.assertEqual(capture, [self.data])
        self.assertFalse(output.exists())
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["reference_input_audit"], {"synthetic_audit_completed": True})
        self.assertEqual(report["training_times"], np.asarray(self.TIMES, dtype=np.float32).tolist())
        self.assertIsNone(report["heldout_stage"])

    def test_official_forward_handles_501_initial_cells_with_batch_1024_natively(self):
        upstream = Path(__file__).resolve().parents[2] / "external/scMultiNODE"
        old_sys_path, numpy_state = list(sys.path), np.random.get_state()
        old_threads = torch.get_num_threads()
        try:
            sys.path.insert(0, str(upstream))
            from optim import running
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                torch.manual_seed(157)
                np.random.seed(157)
                model = benchmark.make_model(running, 30, 12, 10).eval()
                initial = torch.randn(501, 30)
                untouched = initial.clone()
                times = torch.tensor(self.TIMES, dtype=torch.float32)
                native_choice = np.random.choice
                with patch.object(np.random, "choice", wraps=native_choice) as choice:
                    rna, atac, chosen, rna_latent, atac_latent = model(initial, times, times, batch_size=1024)
                    choice.assert_called_once()
                    self.assertEqual(choice.call_args.kwargs, {"size": 1024, "replace": True})
                    np.testing.assert_array_equal(choice.call_args.args[0], np.arange(501))
                self.assertEqual(rna.shape, (1024, 7, 30))
                self.assertEqual(atac.shape, (1024, 7, 12))
                self.assertEqual(chosen.shape, (1024, 30))
                self.assertTrue(torch.equal(rna_latent, atac_latent))
                self.assertTrue(all(bool(torch.isfinite(x).all()) for x in (rna, atac, chosen, rna_latent)))
                self.assertTrue(torch.equal(initial, untouched))
                # Inference explicitly disables native minibatch sampling,
                # retaining 501 original rows, not 1024 repeated trajectories.
                with patch.object(np.random, "choice", wraps=native_choice) as choice:
                    exact = model(initial, times, times, batch_size=None)
                    choice.assert_not_called()
                self.assertEqual(exact[0].shape, (501, 7, 30))
                self.assertEqual(exact[1].shape, (501, 7, 12))
                self.assertTrue(torch.equal(exact[2], initial))
        finally:
            np.random.set_state(numpy_state)
            sys.path[:] = old_sys_path
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main(verbosity=2)
