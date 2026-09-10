"""Shared repository paths for the reorganized standalone benchmark scripts."""
from pathlib import Path
import os
import sys

def configure(root: Path) -> None:
    root = root.resolve()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/trainfbench-matplotlib")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/trainfbench-numba")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/trainfbench-cache")
    for base in (root / "model", root / "comparison"):
        for folder in sorted(base.iterdir()):
            if folder.is_dir() and str(folder) not in sys.path:
                sys.path.append(str(folder))
    sys.path.append(str(root / "common"))
    source = Path(os.environ.get("TRAINF_SOURCE", root / "external" / "COATI"))
    sys.path.append(str(source))
    os.chdir(root)
