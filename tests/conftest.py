"""
Eidosian Test Configuration - Zero Waste, Maximum Efficiency

This module configures the test environment with surgical precision,
ensuring absolute path resolution, optimized fixture availability,
and pinpoint-accurate dependency injection.
"""
import os
import sys
from pathlib import Path
import pytest
import logging
from typing import Dict, Any, Generator
import tempfile
import shutil
import numpy as np
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
    comprehensively verify the ASCII conversion pipeline - a white square, 
    a gradient, and an RGB test pattern.
    
    Returns:
        Dict containing test images and their paths
    """
    # Initialize test assets directory
    test_dir = tempfile.mkdtemp(prefix="glyph_forge_test_")
    
    # 1. Pure white square (100x100) - tests uniform brightness
    white_img = Image.new('L', (100, 100), 255)
    white_path = os.path.join(test_dir, 'white.png')
    white_img.save(white_path)
    
    # 2. Precision gradient (100x100) - tests full brightness range
    gradient = np.linspace(0, 255, 100, dtype=np.uint8)
    gradient_img = np.repeat(gradient.reshape(1, 100), 100, axis=0)
    gradient_pil = Image.fromarray(gradient_img)
    gradient_path = os.path.join(test_dir, 'gradient.png')
    gradient_pil.save(gradient_path)
    
    # 3. RGB test pattern (10x10) - tests color conversion
    rgb_data = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb_data[0:5, :, 0] = 255  # Red top half
    rgb_data[5:10, :, 1] = 255  # Green bottom half
    rgb_img = Image.fromarray(rgb_data)
    rgb_path = os.path.join(test_dir, 'rgb.png')
    rgb_img.save(rgb_path)
    
    # Provide test assets as a cohesive dictionary
    yield {
        'dir': test_dir,
        'white': {'img': white_img, 'path': white_path},
        'gradient': {'img': gradient_pil, 'path': gradient_path},
        'rgb': {'img': rgb_img, 'path': rgb_path}
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
    test_alphabets = {
        "standard": " .:-=+*#%@",
        "blocks": " ░▒▓█",
        "minimal": " ."
    }
    
    # Inject mock methods with atomic precision
    monkeypatch.setattr(AlphabetManager, "list_available_alphabets", 
                         lambda: list(test_alphabets.keys()))
    monkeypatch.setattr(AlphabetManager, "get_alphabet", 
                         lambda name: test_alphabets.get(name, test_alphabets["standard"]))
    monkeypatch.setattr(AlphabetManager, "create_density_map", 
                         lambda charset: {i: charset[min(i * len(charset) // 256, len(charset)-1)] 
                                         for i in range(256)})
    
    return AlphabetManager