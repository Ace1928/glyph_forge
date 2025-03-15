# filepath: /home/lloyd/repos/glyph_forge/tests/test_image_to_Glyph.py
"""
⚡ Eidosian Test Suite: ImageGlyphConverter ⚡

A brutally efficient, surgically precise test suite that validates
every aspect of the Glyph art conversion pipeline with zero redundancy.
Each test serves a specific purpose with crystalline clarity.
"""
import pytest
from unittest import mock
import os
import numpy as np
from PIL import Image
import tempfile
import shutil

from glyph_forge.services.image_to_glyph import ImageGlyphConverter, ColorMode


@pytest.fixture(scope="class")
def test_images() -> dict:
    """Generate precision-crafted test images for conversion verification."""
    # Create testing directory
    test_dir = tempfile.mkdtemp()
    
    # 1. Uniform white square (100x100)
    white_img = Image.new('L', (100, 100), 255)
    white_path = os.path.join(test_dir, 'white.png')
    white_img.save(white_path)
    
    # 2. Horizontal gradient (100x100)
    gradient = np.linspace(0, 255, 100, dtype=np.uint8)
    gradient_img = np.repeat(gradient.reshape(1, 100), 100, axis=0)
    gradient_pil = Image.fromarray(gradient_img)
    gradient_path = os.path.join(test_dir, 'gradient.png')
    gradient_pil.save(gradient_path)
    
    # 3. RGB test pattern (10x10)
    rgb_data = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb_data[0:5, :, 0] = 255  # Red top half
    rgb_data[5:10, :, 1] = 255  # Green bottom half
    rgb_img = Image.fromarray(rgb_data)
    rgb_path = os.path.join(test_dir, 'rgb.png')
    rgb_img.save(rgb_path)
    
    # Yield test assets dictionary
    yield {
        'dir': test_dir,
        'white': {'img': white_img, 'path': white_path},
        'gradient': {'img': gradient_pil, 'path': gradient_path},
        'rgb': {'img': rgb_img, 'path': rgb_path}
    }
    
    # Destroy test resources after use
    shutil.rmtree(test_dir)


@pytest.fixture
def mock_alphabet_manager() -> mock.MagicMock:
    """Mock the AlphabetManager to provide deterministic outputs."""
    with mock.patch('glyph_forge.services.image_to_Glyph.AlphabetManager') as mock_manager:
        mock_manager.list_available_alphabets.return_value = ["standard", "blocks"]
        mock_manager.get_alphabet.return_value = "$@B%8&WM"
        yield mock_manager


class TestImageGlyphConverter:
    """
    Surgical verification of the ImageGlyphConverter class.
    
    This test suite methodically dissects every component of the Glyph conversion
    pipeline to ensure flawless performance, handling edge cases with precision
    and validating core functionality with maximum efficiency.
    """
    
    # ──── Core Initialization Tests ────────────────────────────────────────
    
    def test_init_default_parameters(self, mock_alphabet_manager: mock.MagicMock) -> None:
        """🎛️ Verify default parameters initialize correctly."""
        converter = ImageGlyphConverter()
        
        assert converter.width == 100
        assert converter.height is None
        assert converter.brightness == 1.0
        assert converter.contrast == 1.0
        assert converter.auto_scale is True
        assert converter.dithering is False
        assert converter.threads >= 1
        assert converter.charset is not None
        assert converter.density_map is not None
    
    def test_init_custom_parameters(self, mock_alphabet_manager: mock.MagicMock) -> None:
        """🎮 Verify custom parameters are precisely applied."""
        converter = ImageGlyphConverter(
            charset="@#$%",
            width=80,
            height=40,
            invert=True,
            brightness=1.2,
            contrast=0.8,
            auto_scale=False,
            dithering=True,
            threads=4
        )
        
        assert converter.width == 80
        assert converter.height == 40
        assert converter.brightness == 1.2
        assert converter.contrast == 0.8
        assert converter.auto_scale is False
        assert converter.dithering is True
        assert converter.threads == 4
        assert converter.charset == "%$#@"  # Inverted
    
    def test_parameter_bounds(self, mock_alphabet_manager: mock.MagicMock) -> None:
        """🛡️ Verify input sanitization enforces parameter boundaries."""
        converter = ImageGlyphConverter(
            width=-10,         # Should clamp to 1
            height=0,          # Should clamp to 1
            brightness=3.0,    # Should clamp to 2.0
            contrast=-1.0,     # Should clamp to 0.0
        )
        
        assert converter.width == 1
        assert converter.height == 1
        assert converter.brightness == 2.0
        assert converter.contrast == 0.0
    
    def test_invert_charset(self, mock_alphabet_manager: mock.MagicMock) -> None:
        """🔄 Verify charset inversion for brightness reversal."""
        normal = ImageGlyphConverter(charset="ABCDE", invert=False)
        inverted = ImageGlyphConverter(charset="ABCDE", invert=True)
        
        assert normal.charset == "ABCDE"
        assert inverted.charset == "EDCBA"
    
    def test_predefined_charset(self, mock_alphabet_manager: mock.MagicMock) -> None:
        """📚 Verify predefined charset loading from AlphabetManager."""
        converter = ImageGlyphConverter(charset="standard")
        
        mock_alphabet_manager.get_alphabet.assert_called_with("standard")
        assert converter.charset is not None
    
    # ──── Image Loading & Processing Tests ──────────────────────────────────
    
    def test_load_image_from_path(self, test_images: dict) -> None:
        """📂 Verify image loading from filesystem path."""
        converter = ImageGlyphConverter()
        img = converter._load_image(test_images['white']['path'])
        
        assert isinstance(img, Image.Image)
        assert img.mode == 'L'  # Grayscale
        assert img.size == (100, 100)
    
    def test_load_image_from_pil(self, test_images: dict) -> None:
        """🖼️ Verify image loading from PIL Image object."""
        converter = ImageGlyphConverter()
        img = converter._load_image(test_images['white']['img'])
        
        assert isinstance(img, Image.Image)
        assert img.mode == 'L'  # Grayscale
        assert img.size == (100, 100)
    
    @mock.patch('glyph_forge.services.image_to_Glyph.shutil.get_terminal_size')
    def test_terminal_scaling(self, mock_get_terminal_size: mock.MagicMock) -> None:
        """📏 Verify output auto-scales to terminal dimensions."""
        mock_get_terminal_size.return_value = (50, 25)  # Width, height
        
        converter = ImageGlyphConverter(width=100, auto_scale=True)
        scaled_width, scaled_height = converter._apply_terminal_scaling(100, 50)
        
        assert scaled_width == 48  # 50-2 for margin
        assert scaled_height == 24  # Proportionally scaled
    
    def test_brightness_contrast_adjustment(self, test_images: dict) -> None:
        """🔆 Verify brightness and contrast transforms pixel values."""
        converter = ImageGlyphConverter(brightness=1.5, contrast=1.2)
        
        adjusted = converter._apply_image_adjustments(test_images['gradient']['img'])
        
        # Sample points must differ after adjustment
        original_mid = np.array(test_images['gradient']['img'])[50, 50]
        adjusted_mid = np.array(adjusted)[50, 50]
        
        assert original_mid != adjusted_mid
        assert 0 <= adjusted_mid <= 255  # Values stay in valid range
    
    # ──── Conversion Pipeline Tests ────────────────────────────────────────
    
    def test_convert_basic(self, test_images: dict) -> None:
        """✨ Verify end-to-end conversion of uniform image."""
        converter = ImageGlyphConverter(width=10, height=5, auto_scale=False)
        
        # Force deterministic output with single-character map
        converter.charset = "@"
        converter.density_map = {i: "@" for i in range(256)}
        
        result = converter.convert(test_images['white']['path'])
        expected = "\n".join(["@" * 10] * 5)
        
        assert result == expected
    
    def test_image_processing_pipeline(self, test_images: dict) -> None:
        """⚙️ Verify dimension calculations in processing pipeline."""
        converter = ImageGlyphConverter(width=20, height=10, auto_scale=False)
        
        result = converter._process_image(test_images['gradient']['img'])
        lines = result.strip().split('\n')
        
        assert len(lines) == 10  # Output height matches specification
        assert len(lines[0]) == 20  # Output width matches specification
    
    @mock.patch('glyph_forge.services.image_to_Glyph.ThreadPoolExecutor')
    def test_parallel_conversion(self, mock_executor: mock.MagicMock) -> None:
        """⚡ Verify multi-threaded processing for large images."""
        mock_map = mock.MagicMock()
        mock_map.return_value = ["line1", "line2", "line3"]
        mock_executor.return_value.__enter__.return_value.map = mock_map
        
        converter = ImageGlyphConverter(threads=3)
        pixels = np.zeros((100, 50), dtype=np.uint8)
        
        result = converter._parallel_conversion(pixels)
        
        mock_executor.assert_called_once()
        assert result == "line1\nline2\nline3"
    
    # ──── File I/O Tests ───────────────────────────────────────────────────
    
    def test_save_to_file(self) -> None:
        """💾 Verify file output with proper encoding."""
        converter = ImageGlyphConverter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "output.txt")
            test_content = "Test Glyph Art with Unicode: ░▒▓█"
            
            converter._save_to_file(test_content, output_path)
            
            assert os.path.exists(output_path)
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            assert content == test_content
    
    @mock.patch('os.makedirs')
    def test_save_creates_directory(self, mock_makedirs: mock.MagicMock) -> None:
        """📁 Verify directory creation when saving to new path."""
        converter = ImageGlyphConverter()
        output_path = "/some/new/path/output.txt"
        test_content = "Test Glyph Art"
        
        converter._save_to_file(test_content, output_path)
        
        mock_makedirs.assert_called_once()
    
    # ──── Configuration Modification Tests ────────────────────────────────
    
    def test_set_charset(self) -> None:
        """🔡 Verify charset reconfiguration after initialization."""
        converter = ImageGlyphConverter(charset="ABC")
        
        # Set a new charset with inversion
        converter.set_charset("XYZ", invert=True)
        
        assert converter.charset == "ZYX"  # Inverted
    
    def test_set_image_params(self) -> None:
        """⚙️ Verify runtime parameter updates."""
        converter = ImageGlyphConverter()
        
        # Update all parameters at once
        converter.set_image_params(
            width=50,
            height=25,
            brightness=1.8,
            contrast=0.7,
            dithering=True
        )
        
        assert converter.width == 50
        assert converter.height == 25
        assert converter.brightness == 1.8
        assert converter.contrast == 0.7
        assert converter.dithering is True
        
        # Test partial updates
        converter.set_image_params(width=60)
        assert converter.width == 60
        assert converter.height == 25  # Unchanged
    
    # ──── Color Conversion Tests ──────────────────────────────────────────
    
    def test_color_conversion_ansi(self, test_images: dict) -> None:
        """🌈 Verify ANSI color code generation."""
        converter = ImageGlyphConverter(width=5, height=5, auto_scale=False)
        
        # Force predictable character output
        converter.charset = "#"
        converter.density_map = {i: "#" for i in range(256)}
        
        result = converter.convert_color(test_images['rgb']['img'], color_mode=ColorMode.ANSI)
        
        # ANSI color codes must be present
        assert "\033[38;2;" in result
        assert "\033[0m" in result  # Reset code
    
    def test_color_conversion_html(self, test_images: dict) -> None:
        """🖥️ Verify HTML color formatting."""
        converter = ImageGlyphConverter(width=5, height=5, auto_scale=False)
        
        # Force predictable character output
        converter.charset = "#"
        converter.density_map = {i: "#" for i in range(256)}
        
        result = converter.convert_color(test_images['rgb']['img'], color_mode=ColorMode.HTML)
        
        # HTML formatting must be present
        assert "<pre style='line-height:1; letter-spacing:0'>" in result
        assert "<span style='color:#" in result
        assert "</span>" in result
        assert "</pre>" in result
    
    def test_color_conversion_none(self, test_images: dict) -> None:
        """⚪ Verify grayscale fallback for 'none' color mode."""
        converter = ImageGlyphConverter(width=5, height=5, auto_scale=False)
        
        with mock.patch.object(converter, 'convert') as mock_convert:
            mock_convert.return_value = "standard conversion"
            
            result = converter.convert_color(test_images['rgb']['img'], color_mode=ColorMode.NONE)
            
            mock_convert.assert_called_once()
            assert result == "standard conversion"
    
    # ──── Error Handling Tests ──────────────────────────────────────────────
    
    @mock.patch('builtins.open', side_effect=IOError("Simulated IO error"))
    def test_save_error_handling(self, mock_open: mock.MagicMock) -> None:
        """❌ Verify appropriate error propagation on file save failure."""
        converter = ImageGlyphConverter()
        
        with pytest.raises(Exception):
            converter._save_to_file("content", "test_path")
    
    def test_error_handling_invalid_image_path(self) -> None:
        """⚠️ Verify graceful handling of nonexistent image paths."""
        converter = ImageGlyphConverter()
        result = converter.convert("nonexistent_image.png")
        
        assert result.startswith("Error converting image")
        assert "No such file or directory" in result or "cannot identify image file" in result
    
    @mock.patch('PIL.Image.open', side_effect=IOError("Simulated IO error"))
    def test_convert_error_handling(self, mock_open: mock.MagicMock) -> None:
        """🛡️ Verify conversion failures produce informative messages."""
        converter = ImageGlyphConverter()
        result = converter.convert("test.png")
        
        assert result.startswith("Error converting image")
        assert "Simulated IO error" in result
    
    # ──── Utility Tests ────────────────────────────────────────────────────
    
    def test_get_available_charsets(self) -> None:
        """📋 Verify charset list retrieval with copy protection."""
        converter = ImageGlyphConverter()
        converter._available_charsets = ["standard", "blocks", "detailed"]
        
        charsets = converter.get_available_charsets()
        
        # Result matches original list
        assert charsets == ["standard", "blocks", "detailed"]
        
        # Verify defensive copy - changes to result don't affect original
        charsets.append("new_set")
        assert "new_set" not in converter._available_charsets