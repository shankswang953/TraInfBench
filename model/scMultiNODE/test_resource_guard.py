"""Resource-guard checks with mocked samples, clocks, threads, and process exit.

No training, real process termination, or sleeps occur. Run directly with the
shared CytoBridge interpreter, or with unittest discovery.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import resource_guard as guard_module
from resource_guard import GIB, ResourceGuard, atomic_write_json, evaluate_sample


class GuardExit(BaseException):
    """Model os._exit's non-returning behavior without terminating the test."""

    def __init__(self, code):
        self.code = code


def mock_exit(code):
    raise GuardExit(code)


class EvaluationTests(unittest.TestCase):
    def test_boundaries_and_both_violations(self):
        self.assertEqual(evaluate_sample(10, 6), [])
        self.assertEqual(evaluate_sample(0, 60), [])
        self.assertEqual(evaluate_sample(10.01, 8), ["RSS budget exceeded"])
        self.assertEqual(evaluate_sample(1, 0), ["Available-memory reserve reached"])
        self.assertEqual(evaluate_sample(11, 5), [
            "RSS budget exceeded", "Available-memory reserve reached",
        ])

    def test_invalid_readings_are_not_treated_as_safe(self):
        for value in (None, float("nan"), float("inf"), -float("inf"), -1, True, "bad"):
            for field in ("rss", "available"):
                with self.subTest(value=value, field=field), self.assertRaises(ValueError):
                    evaluate_sample(value if field == "rss" else 1, value if field == "available" else 20)

    def test_thresholds_must_be_positive_and_finite(self):
        for value in (None, 0, -1, float("nan"), float("inf"), True, "bad"):
            for field in ("max_rss_gib", "min_available_gib"):
                with self.subTest(value=value, field=field), self.assertRaises(ValueError):
                    ResourceGuard(Path("unused"), **{field: value})


class AtomicReportTests(unittest.TestCase):
    def test_replacement_is_complete_and_uses_unique_sibling_files(self):
        with tempfile.TemporaryDirectory(prefix="scmultinode-guard-atomic-") as directory:
            path = Path(directory) / "report.json"
            seen = []
            replace = os.replace

            def inspect_replace(source, destination):
                source, destination = Path(source), Path(destination)
                self.assertEqual(source.parent, path.parent)
                self.assertEqual(destination, path)
                self.assertNotEqual(source, destination)
                self.assertEqual(json.loads(source.read_text()), {"version": len(seen) + 1})
                if destination.exists():
                    self.assertEqual(json.loads(destination.read_text()), {"version": len(seen)})
                seen.append(source)
                replace(source, destination)

            with patch.object(guard_module.os, "replace", side_effect=inspect_replace):
                atomic_write_json(path, {"version": 1})
                atomic_write_json(path, {"version": 2})
            self.assertEqual(len(set(seen)), 2)
            self.assertEqual(list(path.parent.iterdir()), [path])
            self.assertEqual(json.loads(path.read_text()), {"version": 2})

    def test_failed_write_keeps_prior_report_and_removes_only_temp(self):
        with tempfile.TemporaryDirectory(prefix="scmultinode-guard-atomic-") as directory:
            path = Path(directory) / "report.json"
            atomic_write_json(path, {"valid": True})
            with self.assertRaises(ValueError):
                atomic_write_json(path, {"invalid": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"valid": True})
            self.assertEqual(list(path.parent.iterdir()), [path])


class ResourceGuardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="scmultinode-guard-test-")
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.clock = 100.0
        self.rss = 1.0
        self.available = 20.0
        self.provider = Mock(side_effect=lambda: self.available)
        self.process = Mock()
        self.process.memory_info.side_effect = lambda: SimpleNamespace(rss=self.rss * GIB)
        for target, kwargs in (
            ("time.monotonic", {"side_effect": lambda: self.clock}),
            ("psutil.Process", {"return_value": self.process}),
            ("threading.Thread", {}),
            ("os._exit", {"side_effect": mock_exit}),
            ("ResourceGuard._print_report", {}),
        ):
            patcher = patch(f"resource_guard.{target}", **kwargs)
            mocked = patcher.start()
            self.addCleanup(patcher.stop)
            if target == "threading.Thread":
                self.thread_type = mocked
            elif target == "os._exit":
                self.exit = mocked
        self.guard = ResourceGuard(self.output, available_memory=self.provider)

    def read(self, name):
        return json.loads((self.output / name).read_text())

    def assert_exit(self, code, call):
        with self.assertRaises(GuardExit) as captured:
            call()
        self.assertEqual(captured.exception.code, code)
        self.exit.assert_called_once_with(code)

    def test_start_samples_immediately_and_uses_injected_provider(self):
        with patch.object(guard_module.psutil, "virtual_memory") as host_memory:
            self.guard.start()
        host_memory.assert_not_called()
        self.process.memory_info.assert_called_once_with()
        self.provider.assert_called_once_with()
        self.thread_type.return_value.start.assert_called_once_with()
        self.assertTrue(self.thread_type.call_args.kwargs["daemon"])
        report = self.read("resource_live.json")
        self.assertEqual(report["status"], "running")
        self.assertEqual(report["pid"], os.getpid())
        self.assertEqual(report["sampled_peak_rss_gib"], 1)
        self.assertEqual(report["minimum_available_gib"], 20)
        self.assertNotIn("max_seconds", report)

    def test_default_provider_uses_psutil_and_converts_bytes(self):
        default = ResourceGuard(self.output)
        with patch.object(guard_module.psutil, "virtual_memory", return_value=SimpleNamespace(available=12 * GIB)):
            default.start()
        self.assertEqual(default.snapshot()["minimum_available_gib"], 12)

    def test_immediate_overlimit_saves_report_before_exit_and_no_thread(self):
        self.rss, self.available = 11, 5
        self.assert_exit(75, self.guard.start)
        report = self.read("resource_guard.json")
        self.assertEqual(report["status"], "guard_stopped")
        self.assertEqual(report["reasons"], ["RSS budget exceeded", "Available-memory reserve reached"])
        self.assertEqual(report["sampled_peak_rss_gib"], 11)
        self.assertEqual(report["minimum_available_gib"], 5)
        self.thread_type.assert_not_called()

    def test_unknown_and_nonfinite_availability_fail_closed(self):
        for index, value in enumerate((None, float("nan"), float("inf"))):
            with self.subTest(value=value):
                output = self.output / str(index)
                output.mkdir()
                guard = ResourceGuard(output, available_memory=lambda: value)
                self.exit.reset_mock()
                self.assert_exit(76, guard.start)
                report = json.loads((output / "resource_guard.json").read_text())
                self.assertEqual(report["status"], "guard_failed")
                self.assertIn("available_gib", report["error"])
        self.thread_type.assert_not_called()

    def test_sampling_exception_fails_closed(self):
        self.process.memory_info.side_effect = PermissionError("cannot read RSS")
        self.assert_exit(76, self.guard.start)
        report = self.read("resource_guard.json")
        self.assertEqual(report["status"], "guard_failed")
        self.assertIn("cannot read RSS", report["error"])

    def test_live_write_failure_fails_closed_and_retains_failure_report(self):
        write = atomic_write_json

        def fail_live(path, report):
            if path.name == "resource_live.json":
                raise OSError("live report unavailable")
            write(path, report)

        with patch.object(guard_module, "atomic_write_json", side_effect=fail_live):
            self.assert_exit(76, self.guard.start)
        self.assertEqual(self.read("resource_guard.json")["status"], "guard_failed")

    def test_failure_report_write_error_cannot_prevent_exit(self):
        self.available = None
        with patch.object(guard_module, "atomic_write_json", side_effect=OSError("disk unavailable")):
            self.assert_exit(76, self.guard.start)

    def test_threshold_report_write_error_uses_failure_exit_code(self):
        self.rss = 11
        with patch.object(guard_module, "atomic_write_json", side_effect=OSError("disk unavailable")):
            self.assert_exit(76, self.guard.start)

    def test_rolling_extrema_phase_and_two_second_live_cadence(self):
        with patch.object(guard_module, "atomic_write_json", wraps=atomic_write_json) as write:
            self.guard.start()
            for elapsed, rss, available in ((0.2, 3, 15), (1.9, 2, 10)):
                self.clock = 100 + elapsed
                self.rss, self.available = rss, available
                self.guard._sample()
            self.assertEqual(write.call_count, 1)
            self.clock, self.rss, self.available = 102, 1, 12
            self.guard.phase = "dynamics"
            self.guard._sample()
            self.assertEqual(write.call_count, 2)
        report = self.read("resource_live.json")
        self.assertEqual(report["phase"], "dynamics")
        self.assertEqual(report["elapsed_seconds"], 2)
        self.assertEqual(report["sampled_peak_rss_gib"], 3)
        self.assertEqual(report["minimum_available_gib"], 10)
        self.assertEqual(report["current_rss_gib"], 1)

    def test_watch_waits_point_two_seconds_without_walltime_limit(self):
        self.guard.start()
        self.clock = 100 + 1000 * 24 * 60 * 60
        with patch.object(self.guard._stop, "wait", side_effect=[False, False, True]) as wait:
            self.guard._watch()
        self.assertEqual([call.args for call in wait.call_args_list], [(0.2,)] * 3)
        self.assertEqual(self.provider.call_count, 3)
        self.exit.assert_not_called()

    def test_watch_detects_later_breach(self):
        self.guard.start()
        self.available = 5
        with patch.object(self.guard._stop, "wait", return_value=False):
            self.assert_exit(75, self.guard._watch)
        self.assertEqual(self.read("resource_guard.json")["status"], "guard_stopped")

    def test_watch_provider_error_fails_closed(self):
        self.guard.start()
        self.provider.side_effect = RuntimeError("provider failed")
        with patch.object(self.guard._stop, "wait", return_value=False):
            self.assert_exit(76, self.guard._watch)
        self.assertIn("provider failed", self.read("resource_guard.json")["error"])

    def test_close_signals_joins_with_bound_and_saves_final_snapshot(self):
        self.guard.start()
        self.clock, self.rss, self.available = 100.2, 4, 8
        self.guard._sample()
        self.guard.phase = "complete"
        self.guard.close()
        self.assertTrue(self.guard._stop.is_set())
        self.thread_type.return_value.join.assert_called_once_with(timeout=1.0)
        report = self.read("resource_live.json")
        self.assertEqual(report["status"], "closed")
        self.assertEqual(report["phase"], "complete")
        self.assertEqual(report["sampled_peak_rss_gib"], 4)
        self.assertEqual(report["minimum_available_gib"], 8)
        self.guard.close()
        self.thread_type.return_value.join.assert_called_once()
        self.exit.assert_not_called()

    def test_sample_after_close_does_not_call_providers(self):
        self.guard.start()
        self.guard.close()
        self.provider.reset_mock()
        self.process.memory_info.reset_mock()
        self.guard._sample()
        self.provider.assert_not_called()
        self.process.memory_info.assert_not_called()

    def test_existing_reports_and_repeat_start_are_refused(self):
        self.guard.start()
        with self.assertRaises(RuntimeError):
            self.guard.start()
        other = ResourceGuard(self.output, available_memory=self.provider)
        with self.assertRaises(FileExistsError):
            other.start()

    def test_prestart_snapshot_is_json_safe_and_close_is_safe(self):
        self.assertIsNone(self.guard.snapshot()["minimum_available_gib"])
        json.dumps(self.guard.snapshot(), allow_nan=False)
        self.guard.close()
        self.thread_type.assert_not_called()
        self.assertEqual(list(self.output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
