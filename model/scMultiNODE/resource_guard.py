"""Sampled own-process memory watchdog for the formal disk-backed trainer.

This is NOT an OS-enforced hard memory limit: native allocation bursts can
overshoot, and a busy interpreter may delay samples. There is deliberately no
wall-time limit. The guard samples every 0.2 seconds and publishes an atomic
live JSON report every 2 seconds. Pass ``benchmark_gastrulation.available_memory_gib``
as ``available_memory`` to respect Linux cgroup limits as well as host memory.

A threshold breach writes ``resource_guard.json`` and immediately exits this
process with code 75. Monitoring/reporting failures fail closed with code 76.
Immediate exits bypass Python cleanup, including temporary-distance cleanup;
only remove the exact recorded scratch directory after confirming exit. No
other process is signaled or terminated. Reports require a unique, already
created run directory; start refuses existing guard artifacts.
"""
from __future__ import annotations

from collections.abc import Callable
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any

import psutil


GIB = 1024**3
SAMPLE_INTERVAL_SECONDS = 0.2
LIVE_REPORT_INTERVAL_SECONDS = 2.0


def _finite_number(value: Any, name: str, *, positive: bool) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if isinstance(value, bool) or not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if number < 0 or (positive and number == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return number


def evaluate_sample(
    rss_gib: float,
    available_gib: float | None,
    max_rss_gib: float = 10.0,
    min_available_gib: float = 6.0,
) -> list[str]:
    """Return violated thresholds; invalid/unavailable samples raise ValueError.

    This helper is pure: it neither samples resources nor writes or terminates.
    Equality is permitted; only RSS above its ceiling or availability below its
    reserve is a threshold violation. Zero available memory is a valid reading.
    """
    maximum = _finite_number(max_rss_gib, "max_rss_gib", positive=True)
    minimum = _finite_number(min_available_gib, "min_available_gib", positive=True)
    rss = _finite_number(rss_gib, "rss_gib", positive=False)
    available = _finite_number(available_gib, "available_gib", positive=False)
    reasons = []
    if rss > maximum:
        reasons.append("RSS budget exceeded")
    if available < minimum:
        reasons.append("Available-memory reserve reached")
    return reasons


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Replace one report atomically using a unique same-directory temp file."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _host_available_memory_gib() -> float:
    return psutil.virtual_memory().available / GIB


class ResourceGuard:
    """Watch one training process until ``close()``, without a duration cap.

    ``start()`` samples synchronously before returning or starting its daemon
    thread. Thus training cannot begin with an already-violated budget or an
    unavailable memory reading. ``available_memory`` returns GiB, not bytes;
    supplying None selects psutil host availability (not cgroup availability).
    Call ``close()`` in the trainer's normal/error cleanup to stop sampling and
    publish the final ``resource_live.json``. An emergency exit cannot run it.
    """

    def __init__(
        self,
        output: Path,
        max_rss_gib: float = 10.0,
        min_available_gib: float = 6.0,
        available_memory: Callable[[], float | None] | None = None,
    ) -> None:
        self.output = Path(output)
        self.max_rss_gib = _finite_number(max_rss_gib, "max_rss_gib", positive=True)
        self.min_available_gib = _finite_number(min_available_gib, "min_available_gib", positive=True)
        if available_memory is not None and not callable(available_memory):
            raise TypeError("available_memory must be callable or None")
        self._available_memory = (
            _host_available_memory_gib if available_memory is None else available_memory
        )
        self._pid = os.getpid()
        self._phase = "initialization"
        self._started: float | None = None
        self._last_report: float | None = None
        self._peak_rss_gib = 0.0
        self._minimum_available_gib: float | None = None
        self._current_rss_gib: float | None = None
        self._available_gib: float | None = None
        self._status = "not_started"
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: psutil.Process | None = None

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    @phase.setter
    def phase(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("phase must be a string")
        with self._lock:
            self._phase = value

    def snapshot(self) -> dict[str, Any]:
        """Return a coherent sampled summary, not a new memory measurement."""
        with self._lock:
            return {
                "pid": self._pid,
                "phase": self._phase,
                "status": self._status,
                "elapsed_seconds": 0.0 if self._started is None else time.monotonic() - self._started,
                "sampled_peak_rss_gib": self._peak_rss_gib,
                "minimum_available_gib": self._minimum_available_gib,
                "current_rss_gib": self._current_rss_gib,
                "available_memory_gib": self._available_gib,
                "rss_sampling_interval_seconds": SAMPLE_INTERVAL_SECONDS,
                "live_report_interval_seconds": LIVE_REPORT_INTERVAL_SECONDS,
                "max_rss_gib": self.max_rss_gib,
                "min_available_gib": self.min_available_gib,
            }

    def start(self) -> None:
        if self._started is not None or self._stop.is_set():
            raise RuntimeError("ResourceGuard can only be started once")
        if not self.output.is_dir():
            raise NotADirectoryError(f"Guard output directory must already exist: {self.output}")
        for name in ("resource_live.json", "resource_guard.json"):
            if (self.output / name).exists():
                raise FileExistsError(f"Refusing to overwrite existing guard report: {self.output / name}")
        self._started = time.monotonic()
        self._status = "running"
        try:
            self._process = psutil.Process(self._pid)
            self._sample(force_report=True)
            self._thread = threading.Thread(
                target=self._watch, name="scmultinode-resource-guard", daemon=True,
            )
            self._thread.start()
        except Exception as error:
            self._fail_closed(error)

    def _write_live(self) -> None:
        with self._write_lock:
            atomic_write_json(self.output / "resource_live.json", self.snapshot())

    def _sample(self, *, force_report: bool = False) -> None:
        if self._stop.is_set():
            return
        if self._process is None:
            raise RuntimeError("ResourceGuard has no process handle")
        rss = self._process.memory_info().rss / GIB
        available = self._available_memory()
        reasons = evaluate_sample(rss, available, self.max_rss_gib, self.min_available_gib)
        # close() may have been called while a resource provider was blocked.
        if self._stop.is_set():
            return
        now = time.monotonic()
        with self._lock:
            self._current_rss_gib = float(rss)
            self._available_gib = float(available)
            self._peak_rss_gib = max(self._peak_rss_gib, rss)
            self._minimum_available_gib = (
                float(available) if self._minimum_available_gib is None
                else min(self._minimum_available_gib, float(available))
            )
        if reasons:
            self._stop_for_threshold(reasons)
        if force_report or self._last_report is None or now - self._last_report >= LIVE_REPORT_INTERVAL_SECONDS:
            self._write_live()
            self._last_report = now

    def _watch(self) -> None:
        try:
            while not self._stop.wait(SAMPLE_INTERVAL_SECONDS):
                self._sample()
        except Exception as error:
            self._fail_closed(error)

    @staticmethod
    def _print_report(report: dict[str, Any]) -> None:
        try:
            print(json.dumps(report, allow_nan=False), file=sys.stderr, flush=True)
        except Exception:
            # A closed log stream must never prevent the emergency exit.
            pass

    def _stop_for_threshold(self, reasons: list[str]) -> None:
        self._stop.set()
        with self._lock:
            self._status = "guard_stopped"
        report = self.snapshot() | {"reasons": reasons}
        try:
            with self._write_lock:
                atomic_write_json(self.output / "resource_guard.json", report)
        except Exception as error:
            self._fail_closed(error)
        self._print_report(report)
        os._exit(75)

    def _fail_closed(self, error: Exception) -> None:
        self._stop.set()
        with self._lock:
            self._status = "guard_failed"
        report = self.snapshot() | {"error": repr(error)}
        try:
            with self._write_lock:
                atomic_write_json(self.output / "resource_guard.json", report)
        except Exception as reporting_error:
            report["reporting_error"] = repr(reporting_error)
        finally:
            self._print_report(report)
            os._exit(76)

    def close(self) -> None:
        """Signal shutdown, join for at most one second, and save the final report."""
        self._stop.set()
        with self._lock:
            if self._status in {"closed", "guard_stopped", "guard_failed"}:
                return
            self._status = "closed"
        if self._started is None:
            return
        try:
            if self._thread is not None and self._thread is not threading.current_thread():
                self._thread.join(timeout=1.0)
            self._write_live()
        except Exception as error:
            self._fail_closed(error)
