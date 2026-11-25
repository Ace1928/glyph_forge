"""Regression tests for the lightweight package import contract."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_root_import_has_no_optional_or_filesystem_side_effects(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    script = """
import logging
import signal
import sys

handlers_before = tuple(logging.getLogger().handlers)
sigint_before = signal.getsignal(signal.SIGINT)
import glyph_forge
assert glyph_forge.__version__ == "1.0.0"
assert "cv2" not in sys.modules
assert "textual" not in sys.modules
assert "numpy" not in sys.modules
assert tuple(logging.getLogger().handlers) == handlers_before
assert signal.getsignal(signal.SIGINT) == sigint_before
"""
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "XDG_CACHE_HOME": str(home / "cache"),
            "XDG_CONFIG_HOME": str(home / "config"),
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(source_root), environment.get("PYTHONPATH", "")))
            ),
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert list(home.iterdir()) == []
