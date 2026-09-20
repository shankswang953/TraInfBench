"""Synthetic-only full/LOO protocol tests for the palate scMultiNODE adapter.

Only temporary fixtures are created. No real benchmark data or model training
is used; the main-entry integration test stops at the native training boundary.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_gastrulation as benchmark
from dataset_protocols import PALATE
import numpy as np
import torch
from test_pancreas_protocol import ExportTestModel, PadDecoder
from trajectory_export import export_trajectories, query_time_grid


class TinyPalateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.zeros(()))
        self.n_genes, self.n_peaks, self.latent_dim = 40, 15, 10


class PalateProtocolTests(unittest.TestCase):
    COUNTS = (70, 73, 77, 81)
    TIMES = (0., 1., 1.5, 2.)
    STAGES = ("E12.5", "E13.5", "E14.0", "E14.5")
    KEYS = ("time_0", "time_1", "time_2", "time_3")
    SPLITS = {"full": [0, 1, 2, 3], "loo1": [0, 2, 3], "loo2": [0, 1, 3]}
    SPECS = {
        "rna": (40, 2.5, "rna_pca_by_time.npz", "primal_norm_params.pt"),
        "atac": (15, .125, "atac_lsi15_by_time.npz", "secondary_norm_params_lsi15.pt"),
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="scmultinode-palate-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(806)
        self.raw = {
            modality: {
                key: (rng.normal(size=(count, spec[0])) + 10 * stage).astype(np.float32)
                for stage, (key, count) in enumerate(zip(self.KEYS, self.COUNTS))
            }
            for modality, spec in self.SPECS.items()
        }
        self.data = self.write_fixture("original", self.raw)

    def write_fixture(self, name, raw):
        destination = self.root / name
        destination.mkdir()
        for modality, (_, scale, source, norm) in self.SPECS.items():
            np.savez_compressed(destination / source, **raw[modality])
            torch.save({"scale": scale}, destination / norm)
        return destination

    def load(self, scenario, *, data=None, **kwargs):
        return benchmark.load_inputs(data or self.data, scenario, dataset=PALATE, **kwargs)

    def config(self, scenario):
        return json.loads((Path(__file__).parent / "configs" / f"palate_{scenario}.json").read_text())

    def assert_training_equal(self, first, second):
        for modality in self.SPECS:
            self.assertEqual(len(first[modality]), len(second[modality]))
            for x, y in zip(first[modality], second[modality]):
                self.assertTrue(torch.equal(x, y))

    def test_dataset_contract_preserves_four_physical_times_and_two_loo_splits(self):
        self.assertEqual(PALATE.name, "palate")
        self.assertEqual(PALATE.times, self.TIMES)
        self.assertEqual(PALATE.stages, self.STAGES)
        self.assertEqual(PALATE.keys, self.KEYS)
        self.assertEqual(PALATE.splits, self.SPLITS)
        self.assertEqual(PALATE.modalities, (
            ("rna", "rna_pca_by_time.npz", "primal_norm_params.pt", 40),
            ("atac", "atac_lsi15_by_time.npz", "secondary_norm_params_lsi15.pt", 15),
        ))
        with self.assertRaises(KeyError):
            self.load("loo3")

    def test_only_observed_matrices_are_normalized_once_in_original_row_order(self):
        for scenario, selected in self.SPLITS.items():
            with self.subTest(scenario=scenario):
                training, initial, provenance, indices = self.load(scenario)
                np.testing.assert_array_equal(initial.numpy(), self.raw["rna"]["time_0"] / 2.5)
                for modality, (dimension, scale, source, norm) in self.SPECS.items():
                    self.assertEqual(list(indices[modality]), [self.KEYS[i] for i in selected])
                    self.assertEqual(provenance[modality]["full_counts"], list(self.COUNTS))
                    self.assertEqual(provenance[modality]["stage_keys"], list(self.KEYS))
                    self.assertEqual(provenance[modality]["training_stage_indices"], selected)
                    self.assertEqual(provenance[modality]["selected_counts"], [self.COUNTS[i] for i in selected])
                    self.assertEqual(provenance[modality]["training_cells"], sum(self.COUNTS[i] for i in selected))
                    self.assertEqual(provenance[modality]["scale"], scale)
                    self.assertEqual(provenance[modality]["dimension"], dimension)
                    self.assertEqual(Path(provenance[modality]["source"]).name, source)
                    self.assertEqual(Path(provenance[modality]["normalization"]).name, norm)
                    for tensor, stage in zip(training[modality], selected):
                        self.assertEqual(tensor.shape, (self.COUNTS[stage], dimension))
                        self.assertEqual(tensor.dtype, torch.float32)
                        np.testing.assert_array_equal(tensor.numpy(), self.raw[modality][self.KEYS[stage]] / scale)
                        np.testing.assert_array_equal(indices[modality][self.KEYS[stage]], np.arange(self.COUNTS[stage]))

    def test_hidden_nan_or_extreme_values_cannot_change_training_inputs(self):
        for scenario, withheld in (("loo1", 1), ("loo2", 2)):
            for poison in (np.nan, 1e20):
                changed = {m: {k: v.copy() for k, v in matrices.items()} for m, matrices in self.raw.items()}
                for modality in self.SPECS:
                    changed[modality][self.KEYS[withheld]].fill(poison)
                altered = self.write_fixture(f"{scenario}-{poison}", changed)
                for smoke in (False, True):
                    with self.subTest(scenario=scenario, poison=poison, smoke=smoke):
                        original = self.load(scenario, seed=7, smoke=smoke)
                        poisoned = self.load(scenario, data=altered, seed=7, smoke=smoke)
                        self.assert_training_equal(original[0], poisoned[0])
                        self.assertTrue(torch.equal(original[1], poisoned[1]))
                        for modality in self.SPECS:
                            for field in ("minimum", "maximum", "scale", "full_counts", "selected_counts", "training_cells"):
                                self.assertEqual(original[2][modality][field], poisoned[2][modality][field])
                            self.assertNotEqual(original[2][modality]["source_sha256"], poisoned[2][modality]["source_sha256"])
                            for key in original[3][modality]:
                                np.testing.assert_array_equal(original[3][modality][key], poisoned[3][modality][key])
                if np.isnan(poison):
                    with self.assertRaisesRegex(ValueError, "Nonfinite observed input"):
                        self.load("full", data=altered)

    def test_all_initial_cells_retained_even_when_smoke_training_is_subsampled(self):
        reference = self.load("full")[1]
        for scenario in self.SPLITS:
            for smoke in (False, True):
                with self.subTest(scenario=scenario, smoke=smoke):
                    training, initial, provenance, indices = self.load(scenario, seed=91, smoke=smoke)
                    self.assertTrue(torch.equal(reference, initial))
                    self.assertEqual(initial.shape, (70, 40))
                    for modality in self.SPECS:
                        expected = [64] * len(self.SPLITS[scenario]) if smoke else [self.COUNTS[i] for i in self.SPLITS[scenario]]
                        self.assertEqual([len(x) for x in training[modality]], expected)
                        self.assertEqual(provenance[modality]["selected_counts"], expected)
                        for chosen in indices[modality].values():
                            self.assertEqual(len(chosen), len(np.unique(chosen)))

    def test_formal_inputs_do_not_depend_on_random_seed(self):
        for scenario in self.SPLITS:
            first, second = self.load(scenario, seed=0), self.load(scenario, seed=611)
            self.assert_training_equal(first[0], second[0])
            self.assertTrue(torch.equal(first[1], second[1]))

    def test_configs_match_other_benchmarks_except_dataset_space_and_scenario(self):
        config_dir = Path(__file__).parent / "configs"
        common = None
        for scenario in self.SPLITS:
            config = self.config(scenario)
            self.assertEqual(config.pop("scenario"), scenario)
            self.assertEqual(config.pop("dataset"), PALATE.name)
            self.assertEqual(config.pop("input_space"), PALATE.input_space)
            if common is None:
                common = config
            else:
                self.assertEqual(config, common)
        for name in ("pancreas_full", "human_cerebral_full", "gastrulation_full"):
            other = json.loads((config_dir / f"{name}.json").read_text())
            for field in ("dataset", "input_space", "scenario"):
                other.pop(field, None)
            self.assertEqual(other, common)
        self.assertEqual((common["latent_dim"], common["batch_size"], common["iters"]), (10, 1024, 20000))
        self.assertEqual((common["ae_iters"], common["fusion_iters"]), (2000, 2000))
        self.assertEqual(common["solver"], "official_euler")

    def test_checkpoint_contains_true_physical_times_and_modalities(self):
        for scenario, selected in self.SPLITS.items():
            payload = benchmark.checkpoint_payload(TinyPalateModel(), self.config(scenario), scenario,
                                                   "dynamics", 20000, dataset=PALATE)
            self.assertEqual(payload["dataset"], "palate")
            self.assertEqual(payload["training_times"], [self.TIMES[i] for i in selected])
            self.assertEqual(payload["all_times"], list(self.TIMES))
            self.assertEqual(payload["stage_names"], list(self.STAGES))
            self.assertEqual((payload["rna_dim"], payload["atac_dim"], payload["latent_dim"]), (40, 15, 10))
            self.assertEqual(payload["iteration"], 20000)

    def test_query_grid_has_81_points_and_observed_and_heldout_times_exactly_once(self):
        native = np.asarray(self.TIMES, dtype=np.float32)
        query = query_time_grid(native, .025)
        self.assertEqual(len(query), 81)
        self.assertEqual(query.dtype, np.float32)
        self.assertTrue(np.all(np.diff(query) > 0))
        np.testing.assert_array_equal(query[np.searchsorted(query, native)], native)
        np.testing.assert_array_equal(query, np.linspace(0, 2, 81, dtype=np.float32))

    def test_exports_all_initial_on_native_grid_and_interpolates_both_heldouts(self):
        initial = self.load("full")[1]
        query = query_time_grid(self.TIMES, .025)
        for scenario, selected in self.SPLITS.items():
            with self.subTest(scenario=scenario):
                native = np.asarray([self.TIMES[i] for i in selected], dtype=np.float32)
                model = ExportTestModel()
                model.rna_dec, model.atac_dec = PadDecoder(40), PadDecoder(15)
                target = self.root / f"{scenario}.npz"
                audit = export_trajectories(model, initial, np.arange(len(initial)), native, query, target, batch_size=23)
                self.assertTrue(audit["all_supplied_initial_once"])
                self.assertEqual(audit["training_times"], native.tolist())
                self.assertTrue(all(grid == native.tolist() for grid in model.diffeq_decoder.grids))
                with np.load(target) as saved:
                    self.assertEqual(saved["rna_norm"].shape, (81, 70, 40))
                    self.assertEqual(saved["atac_norm"].shape, (81, 70, 15))
                    self.assertEqual(saved["latent"].shape, (81, 70, 10))
                    np.testing.assert_array_equal(saved["initial_indices"], np.arange(70))
                    np.testing.assert_array_equal(saved["initial_rna_norm"], initial.numpy())
                    np.testing.assert_array_equal(saved["training_time"], native)
                    if scenario == "loo1":
                        expected = saved["latent"][0] + (2 / 3) * (saved["latent"][60] - saved["latent"][0])
                        np.testing.assert_allclose(saved["latent"][40], expected, rtol=1e-6, atol=1e-6)
                        self.assertNotIn(1., native)
                    elif scenario == "loo2":
                        expected = (saved["latent"][40] + saved["latent"][80]) / 2
                        np.testing.assert_allclose(saved["latent"][60], expected, rtol=1e-6, atol=1e-6)
                        self.assertNotIn(1.5, native)

    def test_main_passes_only_observed_data_and_true_times_to_native_trainer(self):
        class StopBeforeTraining(RuntimeError):
            pass

        captured = []
        audited = []

        def fake_train(*args, **kwargs):
            captured.append((args, kwargs))
            raise StopBeforeTraining("No actual training in this test")

        def auditor(directory, training, initial, provenance, indices):
            audited.append((directory, training, initial, provenance, indices))
            return {"synthetic_audit": True}

        fake_running = SimpleNamespace(scMultiNODETrain=fake_train, tqdm=None)
        for scenario in ("loo1", "loo2"):
            for smoke in (False, True):
                with self.subTest(scenario=scenario, smoke=smoke):
                    config_path = self.root / f"{scenario}-{smoke}.json"
                    config_path.write_text(json.dumps(self.config(scenario)))
                    output = self.root / f"no-training-{scenario}-{smoke}"
                    args = ["--config", str(config_path), "--data-dir", str(self.data),
                            "--output-dir", str(output)]
                    if smoke:
                        args.append("--smoke")
                    old_sys_path = list(sys.path)
                    try:
                        with patch.dict(sys.modules, {"optim": SimpleNamespace(running=fake_running)}), \
                             patch.object(benchmark.subprocess, "check_output", side_effect=[benchmark.UPSTREAM_COMMIT, ""]), \
                             patch.object(benchmark, "available_memory_gib", return_value=64.), \
                             patch.object(benchmark, "make_model", return_value=TinyPalateModel()), \
                             contextlib.redirect_stdout(io.StringIO()):
                            with self.assertRaises(StopBeforeTraining):
                                benchmark.main(dataset=PALATE, argv=args, input_auditor=auditor)
                    finally:
                        sys.path[:] = old_sys_path
                    positional, keywords = captured[-1]
                    expected = self.load(scenario, seed=0, smoke=smoke)
                    self.assert_training_equal({"rna": positional[0], "atac": positional[1]}, expected[0])
                    self.assert_training_equal(audited[-1][1], expected[0])
                    self.assertTrue(torch.equal(audited[-1][2], expected[1]))
                    expected_times = [self.TIMES[i] for i in self.SPLITS[scenario]]
                    np.testing.assert_array_equal(positional[2].numpy(), expected_times)
                    np.testing.assert_array_equal(positional[3].numpy(), expected_times)
                    self.assertEqual(positional[5:7], (2, 32) if smoke else (20000, 1024))
                    self.assertTrue(keywords["train_all"])
                    self.assertEqual((keywords["ae_batch_size"], keywords["fusion_batch_size"]), (128, 128))
                    manifest = json.loads((output / "manifest.json").read_text())
                    self.assertEqual(manifest["reference_input_audit"], {"synthetic_audit": True})
                    self.assertEqual(manifest["heldout_stage"], "E13.5" if scenario == "loo1" else "E14.0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
