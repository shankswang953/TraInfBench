"""Formal disk launcher resource/validation tests, without large training."""
from __future__ import annotations
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import benchmark_gastrulation as b
import numpy as np


def good_audit():
    coupling = {"shape": [10, 10], "finite": True, "minimum": 0., "mass": 1.,
                "source_mass": 1., "target_mass": 1.,
                "row_marginal_max_abs_error": 1e-15, "column_marginal_max_abs_error": 1e-15}
    return {"qgw_solver_audit": [{"status": "returned", "coupling": coupling.copy(),
                "warning_count": 0, "iteration_limit_warning_count": 0,
                "log": {"loss_finite": True, "gw_dist": .1, "result_code": 1}}],
            "qgw_raw_coupling_audit": [{"status": "returned", "coupling": coupling.copy()}]}


class DiskBenchmarkTests(unittest.TestCase):
    def test_resource_policy_is_not_old_dense_gate(self):
        with tempfile.TemporaryDirectory() as d:
            fake = type("Disk", (), {"free": 100 * 1024**3})()
            with patch.object(b.shutil, "disk_usage", return_value=fake):
                plan = b.disk_memory_plan(27707, 27707, d, 10, 6)
            self.assertEqual(plan["recommended_available_gib"], 16)
            self.assertTrue(plan["scratch_passes"])
            self.assertAlmostEqual(plan["distance_scratch_gib"], 16 * 27707**2 / 1024**3)
            with patch.object(b.shutil, "disk_usage", return_value=type("Disk", (), {"free": 1})()):
                self.assertFalse(b.disk_memory_plan(27707, 27707, d, 10, 6)["scratch_passes"])

    def test_distances_never_cast_whole_array(self):
        class Bounded:
            shape = (9, 9)
            def __array__(self, *args, **kwargs):
                raise AssertionError("Whole-array cast")
            def __getitem__(self, key):
                self.assert_block(key)
                return np.zeros((min(key[0].stop, 9)-key[0].start, 9))
            def assert_block(self, key):
                assert key[0].stop - key[0].start == 4
        b.validate_distances_blockwise(Bounded(), 4)
        invalid = np.zeros((9, 9))
        invalid[7, 1] = np.nan
        with self.assertRaises(FloatingPointError):
            b.validate_distances_blockwise(invalid, 4)

    def test_warning_keeps_upstream_continuation_and_is_not_success_claim(self):
        audit = good_audit()
        audit["qgw_solver_audit"][0].update(warning_count=2, iteration_limit_warning_count=2)
        audit["qgw_solver_audit"][0]["log"].update(result_code=3, warning="iteration cap")
        before = copy.deepcopy(audit)
        result = b.summarize_qgw_audit(audit)
        self.assertEqual(result["quality_status"], "needs_review")
        self.assertEqual(result["iteration_limit_warning_count"], 2)
        self.assertEqual(audit, before)

    def test_valid_feasibility_does_not_claim_convergence(self):
        result = b.summarize_qgw_audit(good_audit())
        self.assertEqual(result["quality_status"], "no_flag_in_recorded_checks")
        self.assertIn("no convergence guarantee", result["policy"])

    def test_invalid_audits_fail_before_training(self):
        for field, value in (("finite", False), ("minimum", -.1), ("mass", 0.),
                             ("mass", .9), ("row_marginal_max_abs_error", .2),
                             ("column_marginal_max_abs_error", np.nan)):
            with self.subTest(field=field):
                audit = good_audit()
                audit["qgw_raw_coupling_audit"][0]["coupling"][field] = value
                with self.assertRaises(FloatingPointError):
                    b.summarize_qgw_audit(audit)
        for key in ("qgw_solver_audit", "qgw_raw_coupling_audit"):
            audit = good_audit()
            audit[key] = []
            with self.assertRaises(FloatingPointError):
                b.summarize_qgw_audit(audit)
        audit = good_audit()
        audit["qgw_raw_coupling_audit"][0]["coupling"] = {"audit_error": "failed"}
        with self.assertRaises(FloatingPointError):
            b.summarize_qgw_audit(audit)


if __name__ == "__main__":
    unittest.main()
