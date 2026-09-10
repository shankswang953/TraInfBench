#!/usr/bin/env python
"""Generic entry point for the leave-one-timepoint-out input builder."""

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[1]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


from prepare_loo_time1_inputs import main


if __name__ == "__main__":
    main()
