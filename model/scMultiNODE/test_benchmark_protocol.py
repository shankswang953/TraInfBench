"""Bounded benchmark protocol checks using synthetic temporary input files.

No benchmark training, real fixture mutation, or production-file edits occur.
Run directly using the shared CytoBridge Python environment.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

# Import this first: its existing adapter initializes the environment before
# NumPy/PyTorch import, including the shared environment's OpenMP workaround.
import benchmark_gastrulation as benchmark
import numpy as np
import torch


class BenchmarkProtocolTests(unittest.TestCase):
    COUNTS = (70, 73, 77, 81)
    SPECS = {
        "rna": (50, 2.5, "rna_pca_by_time.npz", "primal_norm_params.pt"),
        "atac": (14, .125, "atac_lsi_by_time_14D.npz", "secondary_norm_params.pt"),
    }
    EXPECTED_SPLITS = {"full": [0, 1, 2, 3], "loo1": [0, 2, 3], "loo2": [0, 1, 3]}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="scmultinode-protocol-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(512)
        self.raw = {
            modality: {
                f"time{i}": (rng.standard_normal((count, spec[0])) + 10 * i).astype(np.float32)
                for i, count in enumerate(self.COUNTS)
            }
            for modality, spec in self.SPECS.items()
        }
        self.data = self.write_fixture("original", self.raw)

    def write_fixture(self, name, raw):
        directory = self.root / name
        directory.mkdir()
        for modality, (_, scale, matrix_name, norm_name) in self.SPECS.items():
            np.savez_compressed(directory / matrix_name, **raw[modality])
            torch.save({"scale": scale}, directory / norm_name)
        return directory

    def assert_same_training(self, first, second):
        for modality in self.SPECS:
            self.assertEqual(len(first[modality]), len(second[modality]))
            for before, after in zip(first[modality], second[modality]):
                self.assertTrue(torch.equal(before, after))

    def test_exact_observed_stages_and_frozen_normalization(self):
        self.assertEqual(benchmark.SPLITS, self.EXPECTED_SPLITS)
        for scenario, stages in self.EXPECTED_SPLITS.items():
            with self.subTest(scenario=scenario):
                training, initial, provenance, indices = benchmark.load_inputs(self.data, scenario)
                self.assertEqual(initial.shape, (70, 50))
                np.testing.assert_array_equal(initial.numpy(), self.raw["rna"]["time0"] / 2.5)
                for modality, (dimension, scale, _, _) in self.SPECS.items():
                    self.assertEqual(set(indices[modality]), {f"time{i}" for i in stages})
                    self.assertEqual(len(training[modality]), len(stages))
                    self.assertEqual(provenance[modality]["training_stage_indices"], stages)
                    self.assertEqual(provenance[modality]["full_counts"], list(self.COUNTS))
                    self.assertEqual(provenance[modality]["selected_counts"], [self.COUNTS[i] for i in stages])
                    self.assertEqual(provenance[modality]["training_cells"], sum(self.COUNTS[i] for i in stages))
                    self.assertEqual(provenance[modality]["dimension"], dimension)
                    for tensor, stage in zip(training[modality], stages):
                        self.assertEqual(tensor.dtype, torch.float32)
                        np.testing.assert_array_equal(tensor.numpy(), self.raw[modality][f"time{stage}"] / scale)
                        np.testing.assert_array_equal(indices[modality][f"time{stage}"], np.arange(self.COUNTS[stage]))

    def test_full_initial_cohort_is_identical_across_scenarios_and_smoke(self):
        reference = benchmark.load_inputs(self.data, "full")[1]
        for scenario in self.EXPECTED_SPLITS:
            for smoke in (False, True):
                with self.subTest(scenario=scenario, smoke=smoke):
                    training, initial, _, indices = benchmark.load_inputs(self.data, scenario, seed=37, smoke=smoke)
                    self.assertTrue(torch.equal(initial, reference))
                    self.assertEqual(len(initial), self.COUNTS[0])
                    if smoke:
                        for modality in self.SPECS:
                            self.assertTrue(all(len(x) == 64 for x in training[modality]))
                            for chosen in indices[modality].values():
                                self.assertEqual(len(np.unique(chosen)), 64)

    def test_hidden_values_cannot_change_training_tensors(self):
        for scenario, hidden_stage in (("loo1", 1), ("loo2", 2)):
            for poison in (1e20, np.nan):
                with self.subTest(scenario=scenario, poison=poison):
                    changed = {
                        modality: {key: value.copy() for key, value in matrices.items()}
                        for modality, matrices in self.raw.items()
                    }
                    for modality in self.SPECS:
                        changed[modality][f"time{hidden_stage}"].fill(poison)
                    altered_dir = self.write_fixture(f"{scenario}-{poison}", changed)
                    for smoke in (False, True):
                        base = benchmark.load_inputs(self.data, scenario, seed=23, smoke=smoke)
                        altered = benchmark.load_inputs(altered_dir, scenario, seed=23, smoke=smoke)
                        self.assert_same_training(base[0], altered[0])
                        self.assertTrue(torch.equal(base[1], altered[1]))
                        for modality in self.SPECS:
                            for key in ("minimum", "maximum", "training_cells", "selected_counts"):
                                self.assertEqual(base[2][modality][key], altered[2][modality][key])
                            for stage in base[3][modality]:
                                np.testing.assert_array_equal(base[3][modality][stage], altered[3][modality][stage])
                            # Source hashes must still reflect altered fixtures;
                            # provenance is not falsely declared unchanged.
                            self.assertNotEqual(base[2][modality]["source_sha256"], altered[2][modality]["source_sha256"])

    def test_formal_inputs_are_not_seed_dependent_or_subsampled(self):
        for scenario in self.EXPECTED_SPLITS:
            first = benchmark.load_inputs(self.data, scenario, seed=0)
            second = benchmark.load_inputs(self.data, scenario, seed=9876)
            self.assert_same_training(first[0], second[0])
            for modality in self.SPECS:
                self.assertEqual(sum(len(x) for x in first[0][modality]), sum(self.COUNTS[i] for i in self.EXPECTED_SPLITS[scenario]))

    def test_configurations_differ_only_in_scenario(self):
        config_dir = Path(benchmark.__file__).resolve().parent / "configs"
        common = None
        for scenario in self.EXPECTED_SPLITS:
            config = json.loads((config_dir / f"gastrulation_{scenario}.json").read_text())
            self.assertEqual(config.pop("scenario"), scenario)
            if common is None:
                common = config
            else:
                self.assertEqual(config, common)
        self.assertEqual(common["input_space"], "frozen_normalized_rna50_atac14")
        self.assertEqual(common["solver"], "official_euler")
        self.assertEqual(common["batch_size"], 1024)
        self.assertEqual(common["iters"], 20000)
        self.assertEqual(common["ae_iters"], 2000)
        self.assertEqual(common["fusion_iters"], 2000)
        self.assertEqual(common["trajectory_dt"], .025)
        self.assertEqual(common["inference_batch_size"], 512)

    def test_real_count_memory_plans_are_192_128_96_gib(self):
        # Current real-data stage counts, not allocation of real-size fixtures.
        counts = [9018, 7796, 11470, 7219]
        expected = {"full": 192, "loo1": 128, "loo2": 96}
        for scenario, stages in self.EXPECTED_SPLITS.items():
            count = sum(counts[i] for i in stages)
            plan = benchmark.memory_plan(count, count)
            self.assertEqual(plan["recommended_available_gib"], expected[scenario])
            self.assertAlmostEqual(plan["four_dense_arrays_gib"], 32 * count**2 / 1024**3)
        self.assertEqual(benchmark.memory_plan(100, 100)["recommended_available_gib"], 2)

    def test_checkpoints_capture_completed_updates(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.zeros(()))
                self.n_genes, self.n_peaks, self.latent_dim = 50, 14, 10

        model = TinyModel()
        output = self.root / "checkpoints-test"
        (output / "checkpoints").mkdir(parents=True)
        captured = []

        def capture(payload, destination):
            captured.append((Path(destination).name, float(payload["model_state_dict"]["value"])))
            self.assertEqual(payload["training_times"], [0, 2, 2.5])
            self.assertEqual(payload["scenario"], "loo1")

        recorder, handle = benchmark.recorder_class(model, {"checkpoint_every": 2}, "loo1", output)
        try:
            with patch.object(benchmark.torch, "save", side_effect=capture):
                progress = recorder(range(3), desc="[ Dynamic Training ]")
                for step in progress:
                    progress.set_postfix({"Loss": str(3 - step)})
                    # Loss display precedes backward/optimizer.step upstream;
                    # checkpoint must not fire from set_postfix.
                    self.assertEqual(len(captured), 0 if step < 2 else 1)
                    with torch.no_grad():
                        model.value.add_(1)
                self.assertEqual(captured, [("dynamics_000002.pt", 2.), ("dynamics_000003.pt", 3.)])
        finally:
            handle.close()
        with (output / "losses_display_precision.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["iteration"]) for row in rows], [1, 2, 3])
        self.assertEqual([float(row["Loss"]) for row in rows], [3., 2., 1.])


if __name__ == "__main__":
    unittest.main(verbosity=2)
