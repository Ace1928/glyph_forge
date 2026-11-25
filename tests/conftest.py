"""
Eidosian Test Configuration - Zero Waste, Maximum Efficiency

This module configures the test environment with surgical precision,
ensuring absolute path resolution, optimized fixture availability,
and pinpoint-accurate dependency injection.
"""

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import numpy as np
import pytest
from helpers_fakes import (
    DummyCompletedProcess,
    DummyMSS,
    DummyPopen,
    DummySyncPlaywright,
    DummyVideoCapture,
)
from PIL import Image

# ──── Path Configuration ────────────────────────────────────────────────
# Add src directory to path with absolute precision
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# ──── Logging Configuration ────────────────────────────────────────────
# Disable unnecessary logs during tests
logging.getLogger("glyph_forge").setLevel(logging.ERROR)


# ──── Atomic Test Fixtures ───────────────────────────────────────────────


@pytest.fixture(scope="session")
def test_images() -> Generator[Dict[str, Any], None, None]:
    """
    Generate pristine test images for conversion validation.

    This fixture creates the absolute minimum set of test images needed to
    comprehensively verify the Glyph conversion pipeline - a white square,
    a gradient, and an RGB test pattern.

    Returns:
        Dict containing test images and their paths
    """
    # Initialize test assets directory
    test_dir = tempfile.mkdtemp(prefix="glyph_forge_test_")

    # 1. Pure white square (100x100) - tests uniform brightness
    white_img = Image.new("L", (100, 100), 255)
    white_path = os.path.join(test_dir, "white.png")
    white_img.save(white_path)

    # 2. Precision gradient (100x100) - tests full brightness range
    gradient = np.linspace(0, 255, 100, dtype=np.uint8)
    gradient_img = np.repeat(gradient.reshape(1, 100), 100, axis=0)
    gradient_pil = Image.fromarray(gradient_img)
    gradient_path = os.path.join(test_dir, "gradient.png")
    gradient_pil.save(gradient_path)

    # 3. RGB test pattern (10x10) - tests color conversion
    rgb_data = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb_data[0:5, :, 0] = 255  # Red top half
    rgb_data[5:10, :, 1] = 255  # Green bottom half
    rgb_img = Image.fromarray(rgb_data)
    rgb_path = os.path.join(test_dir, "rgb.png")
    rgb_img.save(rgb_path)

    # Provide test assets as a cohesive dictionary
    yield {
        "dir": test_dir,
        "white": {"img": white_img, "path": white_path},
        "gradient": {"img": gradient_pil, "path": gradient_path},
        "rgb": {"img": rgb_img, "path": rgb_path},
    }

    # Annihilate test resources after use - zero waste
    shutil.rmtree(test_dir)


@pytest.fixture
def mock_alphabet_manager(monkeypatch):
    """
    Create a precision-crafted mock of the AlphabetManager.

    This fixture surgically isolates tests from alphabet dependencies
    by providing a controlled, deterministic character set interface.
    """
    from glyph_forge.utils.alphabet_manager import AlphabetManager

    # Define test alphabets with precise density gradients
    test_alphabets = {"standard": " .:-=+*#%@", "blocks": " ░▒▓█", "minimal": " ."}

    # Inject mock methods with atomic precision
    monkeypatch.setattr(
        AlphabetManager, "list_available_alphabets", lambda: list(test_alphabets.keys())
    )
    monkeypatch.setattr(
        AlphabetManager,
        "get_alphabet",
        lambda name: test_alphabets.get(name, test_alphabets["standard"]),
    )
    monkeypatch.setattr(
        AlphabetManager,
        "create_density_map",
        lambda charset: {
            i: charset[min(i * len(charset) // 256, len(charset) - 1)] for i in range(256)
        },
    )

    return AlphabetManager


@pytest.fixture(autouse=True)
def sandbox_env(monkeypatch, tmp_path_factory):
    """Force all filesystem writes into a temp sandbox."""
    base = tmp_path_factory.mktemp("glyph_forge_env")
    monkeypatch.setenv("HOME", str(base))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(base / "config"))
    monkeypatch.setenv("GLYPH_FORGE_LOG_LEVEL", "ERROR")
    monkeypatch.setenv("DISPLAY", ":0")
    return base


@pytest.fixture(autouse=True)
def patch_external_deps(monkeypatch):
    """Patch external dependencies to keep tests hermetic."""
    import subprocess

    # Subprocess stubs
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: DummyCompletedProcess(
            returncode=0,
            stdout="http://example.com/video\n",
            stderr="",
        ),
    )
    monkeypatch.setattr(subprocess, "Popen", DummyPopen)

    # cv2 stubs
    try:
        import cv2

        monkeypatch.setattr(cv2, "VideoCapture", DummyVideoCapture)
        monkeypatch.setattr(cv2, "selectROI", lambda *args, **kwargs: (0, 0, 0, 0))
        monkeypatch.setattr(cv2, "destroyAllWindows", lambda *args, **kwargs: None)
    except Exception:
        pass

    # mss stubs
    try:
        import mss

        monkeypatch.setattr(mss, "mss", DummyMSS)
    except Exception:
        pass

    # playwright stubs
    try:
        import glyph_forge.streaming.core.browser as browser_mod

        monkeypatch.setattr(browser_mod, "PLAYWRIGHT_AVAILABLE", True)
        monkeypatch.setattr(browser_mod, "sync_playwright", DummySyncPlaywright)
    except Exception:
        pass

    # Patch capture BrowserCapture import path
    try:
        import sys as _sys
        import types

        _sys.modules["playwright.sync_api"] = types.SimpleNamespace(
            sync_playwright=DummySyncPlaywright
        )
    except Exception:
        pass
