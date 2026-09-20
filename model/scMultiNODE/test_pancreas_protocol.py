"""Synthetic-only protocol tests for the moscot pancreas scMultiNODE adapter.

Tests create only temporary fixtures, never train on or mutate benchmark data.
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

# Initializes the shared environment's OpenMP workaround before torch import.
import benchmark_gastrulation as benchmark
from dataset_protocols import PANCREAS
import numpy as np
import torch
from trajectory_export import export_trajectories


class TinyPancreasModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.zeros(()))
        self.n_genes, self.n_peaks, self.latent_dim = 50, 22, 10


class FirstTen(torch.nn.Module):
    def forward(self, values):
        return values[..., :10]


class PadDecoder(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    def forward(self, values):
        return torch.nn.functional.pad(values, (0, self.dimension - values.shape[-1]))


class RecordingEuler(torch.nn.Module):
    ode_method = "euler"

    def __init__(self):
        super().__init__()
        self.grids = []

    def forward(self, initial, times):
        self.grids.append(times.detach().cpu().tolist())
        states = [initial]
        for start, stop in zip(times[:-1], times[1:]):
            # State-dependent drift makes extra integration knots detectable.
            states.append(states[-1] + (stop - start) * (states[-1] + 1))
        return torch.stack(states, dim=1)


class ExportTestModel(torch.nn.Module):
    anchor_mod = "rna"

    def __init__(self):
        super().__init__()
        self.anchor_enc = FirstTen()
        self.fusion_layer = torch.nn.Identity()
        self.diffeq_decoder = RecordingEuler()
        self.rna_dec = PadDecoder(50)
        self.atac_dec = PadDecoder(22)

    def forward(self, initial, rna_times, atac_times, batch_size=None):
        assert batch_size is None
        assert torch.equal(rna_times, atac_times)
        latent = self.diffeq_decoder(self.fusion_layer(self.anchor_enc(initial)), rna_times)
        return self.rna_dec(latent), self.atac_dec(latent), initial, latent, latent


class PancreasProtocolTests(unittest.TestCase):
    COUNTS = (70, 73, 77)
    SPECS = {
        "rna": (50, 2.5, "rna_pca_by_time.npz", "primal_norm_params.pt"),
        "atac": (22, .125, "atac_poissonvi_time_data.npz", "secondary_norm_params_poissonvi.pt"),
    }
    SPLITS = {"full": [0, 1, 2], "loo1": [0, 2]}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="scmultinode-pancreas-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(217)
        self.raw = {
            modality: {
                f"time_{stage}": (rng.normal(size=(count, spec[0])) + 10 * stage).astype(np.float32)
                for stage, count in enumerate(self.COUNTS)
            }
            for modality, spec in self.SPECS.items()
        }
        self.data = self.write_fixture("original", self.raw)

    def write_fixture(self, name, raw):
        directory = self.root / name
        directory.mkdir()
        for modality, (_, scale, source, norm) in self.SPECS.items():
            np.savez_compressed(directory / source, **raw[modality])
            torch.save({"scale": scale}, directory / norm)
        return directory

    def load(self, scenario, *, data=None, **kwargs):
        return benchmark.load_inputs(data or self.data, scenario, dataset=PANCREAS, **kwargs)

    def config(self, scenario):
        location = Path(__file__).resolve().parent / "configs" / f"pancreas_{scenario}.json"
        return json.loads(location.read_text())

    def assert_training_equal(self, first, second):
        for modality in self.SPECS:
            self.assertEqual(len(first[modality]), len(second[modality]))
            for x, y in zip(first[modality], second[modality]):
                self.assertTrue(torch.equal(x, y))

    def test_dataset_contract_and_only_interior_loo(self):
        self.assertEqual(PANCREAS.name, "moscot_pancreas")
        self.assertEqual(PANCREAS.times, (0., 1., 2.))
        self.assertEqual(PANCREAS.stages, ("E14.5", "E15.5", "E16.5"))
        self.assertEqual(PANCREAS.keys, ("time_0", "time_1", "time_2"))
        self.assertEqual(PANCREAS.splits, self.SPLITS)
        with self.assertRaises(KeyError):
            self.load("loo2")

    def test_observed_inputs_are_same_normalized_space_without_rescaling(self):
        for scenario, stages in self.SPLITS.items():
            with self.subTest(scenario=scenario):
                training, initial, provenance, indices = self.load(scenario)
                np.testing.assert_array_equal(initial.numpy(), self.raw["rna"]["time_0"] / 2.5)
                for modality, (dimension, scale, source, norm) in self.SPECS.items():
                    self.assertEqual(set(indices[modality]), {f"time_{i}" for i in stages})
                    self.assertEqual(provenance[modality]["scale"], scale)
                    self.assertEqual(Path(provenance[modality]["source"]).name, source)
                    self.assertEqual(Path(provenance[modality]["normalization"]).name, norm)
                    self.assertEqual(provenance[modality]["dimension"], dimension)
                    self.assertEqual(provenance[modality]["full_counts"], list(self.COUNTS))
                    self.assertEqual(provenance[modality]["training_stage_indices"], stages)
                    self.assertEqual(provenance[modality]["selected_counts"], [self.COUNTS[i] for i in stages])
                    for values, stage in zip(training[modality], stages):
                        self.assertEqual(values.dtype, torch.float32)
                        np.testing.assert_array_equal(values.numpy(), self.raw[modality][f"time_{stage}"] / scale)
                        np.testing.assert_array_equal(indices[modality][f"time_{stage}"], np.arange(self.COUNTS[stage]))

    def test_all_initial_cells_retained_once_across_full_loo_and_smoke(self):
        reference = self.load("full")[1]
        for scenario in self.SPLITS:
            for smoke in (False, True):
                with self.subTest(scenario=scenario, smoke=smoke):
                    training, initial, _, indices = self.load(scenario, seed=81, smoke=smoke)
                    self.assertTrue(torch.equal(reference, initial))
                    self.assertEqual(initial.shape, (70, 50))
                    if smoke:
                        for modality in self.SPECS:
                            self.assertEqual(sum(map(len, training[modality])), 64 * len(self.SPLITS[scenario]))
                            for selected in indices[modality].values():
                                self.assertEqual(len(np.unique(selected)), 64)
                        if scenario == "loo1":
                            # Native AE/fusion sample 128 without replacement.
                            self.assertEqual(sum(map(len, training["rna"])), 128)

    def test_heldout_poison_cannot_change_any_supplied_training_tensor(self):
        for poison in (np.nan, 1e20):
            changed = {m: {k: v.copy() for k, v in matrices.items()} for m, matrices in self.raw.items()}
            for modality in self.SPECS:
                changed[modality]["time_1"].fill(poison)
            other = self.write_fixture(f"poison-{poison}", changed)
            for smoke in (False, True):
                with self.subTest(poison=poison, smoke=smoke):
                    before = self.load("loo1", seed=7, smoke=smoke)
                    after = self.load("loo1", data=other, seed=7, smoke=smoke)
                    self.assert_training_equal(before[0], after[0])
                    self.assertTrue(torch.equal(before[1], after[1]))
                    for modality in self.SPECS:
                        for key in ("scale", "minimum", "maximum", "training_cells", "selected_counts"):
                            self.assertEqual(before[2][modality][key], after[2][modality][key])
                        self.assertNotEqual(before[2][modality]["source_sha256"], after[2][modality]["source_sha256"])
                        for stage in before[3][modality]:
                            np.testing.assert_array_equal(before[3][modality][stage], after[3][modality][stage])
            if np.isnan(poison):
                with self.assertRaisesRegex(ValueError, "Nonfinite observed input"):
                    self.load("full", data=other)

    def test_formal_inputs_are_not_seed_dependent(self):
        for scenario in self.SPLITS:
            first, second = self.load(scenario, seed=0), self.load(scenario, seed=191)
            self.assert_training_equal(first[0], second[0])

    def test_configs_preserve_gastrulation_hyperparameters(self):
        full, loo = self.config("full"), self.config("loo1")
        self.assertEqual(full.pop("scenario"), "full")
        self.assertEqual(loo.pop("scenario"), "loo1")
        self.assertEqual(full, loo)
        self.assertEqual(full["dataset"], PANCREAS.name)
        self.assertEqual(full["input_space"], PANCREAS.input_space)
        expected = {"latent_dim": 10, "solver": "official_euler", "batch_size": 1024,
                    "iters": 20000, "ae_iters": 2000, "fusion_iters": 2000,
                    "ae_batch_size": 128, "fusion_batch_size": 128,
                    "trajectory_dt": .025, "inference_batch_size": 512}
        for key, value in expected.items():
            self.assertEqual(full[key], value)
        original = json.loads((Path(__file__).parent / "configs/gastrulation_full.json").read_text())
        for key, value in original.items():
            if key not in ("scenario", "dataset", "input_space"):
                self.assertEqual(full[key], value, key)

    def test_checkpoint_contains_dataset_and_uncompressed_physical_clock(self):
        for scenario, expected_times in (("full", [0., 1., 2.]), ("loo1", [0., 2.])):
            payload = benchmark.checkpoint_payload(
                TinyPancreasModel(), self.config(scenario), scenario, "dynamics", 20000,
                dataset=PANCREAS,
            )
            self.assertEqual(payload["dataset"], PANCREAS.name)
            self.assertEqual(payload["training_times"], expected_times)
            self.assertEqual(payload["all_times"], [0., 1., 2.])
            self.assertEqual(payload["stage_names"], ["E14.5", "E15.5", "E16.5"])
            self.assertEqual((payload["rna_dim"], payload["atac_dim"], payload["latent_dim"]), (50, 22, 10))
            self.assertEqual(payload["iteration"], 20000)

    def test_recorder_saves_pancreas_metadata_after_completed_update(self):
        model = TinyPancreasModel()
        output = self.root / "recorder"
        (output / "checkpoints").mkdir(parents=True)
        saved = []

        def capture(payload, destination):
            saved.append(float(payload["model_state_dict"]["value"]))
            self.assertEqual(payload["dataset"], PANCREAS.name)
            self.assertEqual(payload["training_times"], [0., 2.])
            self.assertEqual(payload["atac_dim"], 22)

        recorder, handle = benchmark.recorder_class(model, {"checkpoint_every": 2}, "loo1", output, dataset=PANCREAS)
        try:
            with patch.object(benchmark.torch, "save", side_effect=capture), contextlib.redirect_stdout(io.StringIO()):
                progress = recorder(range(3), desc="[ Dynamic Training ]")
                for step in progress:
                    progress.set_postfix({"Loss": str(3 - step)})
                    self.assertEqual(len(saved), 0 if step < 2 else 1)
                    with torch.no_grad():
                        model.value.add_(1)
                self.assertEqual(saved, [2., 3.])
        finally:
            handle.close()

    def test_export_uses_observed_grid_and_interpolates_heldout_in_latent(self):
        initial = self.load("full")[1]
        for scenario, train in (("full", [0., 1., 2.]), ("loo1", [0., 2.])):
            model = ExportTestModel()
            output = self.root / f"{scenario}-trajectory.npz"
            audit = export_trajectories(model, initial, np.arange(len(initial)), train,
                                       np.linspace(0, 2, 81, dtype=np.float32), output, batch_size=23)
            self.assertTrue(audit["all_supplied_initial_once"])
            self.assertTrue(all(grid == train for grid in model.diffeq_decoder.grids))
            with np.load(output) as archive:
                self.assertEqual(archive["rna_norm"].shape, (81, 70, 50))
                self.assertEqual(archive["atac_norm"].shape, (81, 70, 22))
                self.assertEqual(archive["latent"].shape, (81, 70, 10))
                np.testing.assert_array_equal(archive["initial_indices"], np.arange(70))
                np.testing.assert_array_equal(archive["initial_rna_norm"], initial.numpy())
                np.testing.assert_array_equal(archive["training_time"], train)
                if scenario == "loo1":
                    np.testing.assert_allclose(archive["latent"][40],
                                               (archive["latent"][0] + archive["latent"][-1]) / 2,
                                               rtol=1e-6, atol=1e-6)

    def test_main_supplies_only_observed_matrices_to_native_all_stage_trainer(self):
        class StopBeforeTraining(RuntimeError):
            pass

        captures = []

        def fake_train(*args, **kwargs):
            captures.append((args, kwargs))
            raise StopBeforeTraining("No training is run in this test")

        fake_running = SimpleNamespace(scMultiNODETrain=fake_train, tqdm=None)
        for smoke in (False, True):
            config = self.root / f"config-{smoke}.json"
            config.write_text(json.dumps(self.config("loo1")))
            args = ["--config", str(config), "--data-dir", str(self.data),
                    "--output-dir", str(self.root / f"no-training-{smoke}")]
            if smoke:
                args.append("--smoke")
            # Mock only expensive/external entry points; argument assembly and
            # all data/config validation remain the actual production code.
            old_sys_path = list(sys.path)
            try:
                with patch.dict(sys.modules, {"optim": SimpleNamespace(running=fake_running)}), \
                     patch.object(benchmark.subprocess, "check_output", side_effect=[benchmark.UPSTREAM_COMMIT, ""]), \
                     patch.object(benchmark, "available_memory_gib", return_value=64.), \
                     patch.object(benchmark, "make_model", return_value=TinyPancreasModel()), \
                     contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(StopBeforeTraining):
                        benchmark.main(dataset=PANCREAS, argv=args)
            finally:
                sys.path[:] = old_sys_path
            positional, keywords = captures[-1]
            expected = self.load("loo1", seed=0, smoke=smoke)[0]
            self.assert_training_equal({"rna": positional[0], "atac": positional[1]}, expected)
            np.testing.assert_array_equal(positional[2].numpy(), [0., 2.])
            np.testing.assert_array_equal(positional[3].numpy(), [0., 2.])
            self.assertTrue(keywords["train_all"])
            self.assertEqual(keywords["ae_batch_size"], 128)
            self.assertEqual(keywords["fusion_batch_size"], 128)
            self.assertEqual(positional[5:7], (2, 32) if smoke else (20000, 1024))


if __name__ == "__main__":
    unittest.main(verbosity=2)
