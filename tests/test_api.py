"""
⚡ Eidosian Test Suite: ASCII Forge API Integration ⚡

A brutal, zero-compromise test suite that validates the complete
ASCII Forge API with maximum coverage and surgical precision.
Every feature is validated with atomic tests and crystal-clear assertions.
"""

import pytest
import os
import tempfile
from unittest import mock
from typing import Dict, Any, List

from ascii_forge.api.ascii_api import ASCIIForgeAPI, get_api


@pytest.fixture
def api():
    """Create a fresh API instance for each test."""
    # Reset singleton for test isolation
    import ascii_forge.api.ascii_api
    ascii_forge.api.ascii_api._api_instance = None
    
    # Create and return new instance
    api = ASCIIForgeAPI()
    
    # Inject test config values
    api.config.set('banner', 'default_style', 'minimal')
    api.config.set('banner', 'default_font', 'standard')
    api.config.set('banner', 'default_width', 80)
    
    return api


@pytest.fixture
def mock_banner_generator():
    """Mock the BannerGenerator for isolated testing."""
    with mock.patch('ascii_forge.core.banner_generator.BannerGenerator') as mock_generator:
        # Configure mock
        mock_instance = mock_generator.return_value
        mock_instance.generate.return_value = "MOCK BANNER"
        mock_instance.available_fonts.return_value = ["font1", "font2", "standard"]
        mock_instance.font = "standard"
        mock_instance.width = 80
        
        yield mock_instance


@pytest.fixture
def mock_image_converter():
    """Mock the ImageAsciiConverter for isolated testing."""
    with mock.patch('ascii_forge.services.image_to_ascii.ImageAsciiConverter') as mock_converter:
        # Configure mock
        mock_instance = mock_converter.return_value
        mock_instance.convert.return_value = "MOCK ASCII ART"
        mock_instance.convert_color.return_value = "MOCK COLOR ASCII ART"
        mock_instance.charset = "general"
        mock_instance.width = 100
        
        yield mock_instance


class TestASCIIForgeAPI:
    """Comprehensive test suite for the ASCII Forge API."""

    # ──── Core API Initialization Tests ───────────────────────────────
    
    def test_singleton_pattern(self):
        """🔂 Verify API singleton pattern works correctly."""
        api1 = get_api()
        api2 = get_api()
        
        assert api1 is api2  # Same instance returned
        
    def test_api_initialization(self, api):
        """🧰 Verify API initializes core components properly."""
        assert api._banner_generator is not None
        assert api._image_converter is None  # Lazy loading
    
    def test_lazy_loading_image_converter(self, api):
        """⚡ Verify image converter initializes on first use."""
        # Initially None
        assert api._image_converter is None
        
        # Access triggers initialization
        converter = api._get_image_converter()
        assert converter is not None
        assert api._image_converter is not None
    
    # ──── Banner Generation Tests ───────────────────────────────────────
    
    def test_generate_banner_basic(self, api, mock_banner_generator):
        """📝 Verify basic banner generation."""
        with mock.patch.object(api, '_banner_generator', mock_banner_generator):
            result = api.generate_banner("Test Text")
            
            # Check correct method was called
            mock_banner_generator.generate.assert_called_once()
            assert isinstance(result, str)
    
    def test_generate_banner_with_style(self, api, mock_banner_generator):
        """🎭 Verify banner generation with style parameter."""
        with mock.patch.object(api, '_banner_generator', mock_banner_generator):
            api.generate_banner("Test", style="boxed")
            
            mock_banner_generator.generate.assert_called_with(
                "Test", style="boxed", effects=None, color=False
            )
    
    def test_generate_banner_with_custom_font(self, api, mock_banner_generator):
        """🔤 Verify banner generation with custom font creates new generator."""
        with mock.patch('ascii_forge.api.ascii_api.BannerGenerator') as mock_bg_class:
            mock_bg_class.return_value = mock_banner_generator
            
            api.generate_banner("Test", font="big")
            
            # Should create a new BannerGenerator with the font
            mock_bg_class.assert_called_once()
            args, kwargs = mock_bg_class.call_args
            assert kwargs["font"] == "big"
    
    # ──── Image Conversion Tests ───────────────────────────────────────
    
    def test_image_to_ascii_basic(self, api, mock_image_converter):
        """🖼️ Verify basic image conversion."""
        with mock.patch.object(api, '_get_image_converter', return_value=mock_image_converter):
            result = api.image_to_ascii("image.jpg")
            
            mock_image_converter.convert.assert_called_once()
            assert result == "MOCK ASCII ART"
    
    def test_image_to_ascii_with_color(self, api, mock_image_converter):
        """🌈 Verify color image conversion."""
        with mock.patch.object(api, '_get_image_converter', return_value=mock_image_converter):
            result = api.image_to_ascii("image.jpg", color_mode="ansi")
            
            mock_image_converter.convert_color.assert_called_once()
            assert result == "MOCK COLOR ASCII ART"
    
    def test_image_to_ascii_with_params(self, api):
        """⚙️ Verify parameter forwarding to image converter."""
        # Setup more complex mock with parameter verification
        mock_converter = mock.MagicMock()
        mock_converter.charset = "general"
        mock_converter.width = 100
        mock_converter.brightness = 1.0
        mock_converter.contrast = 1.0
        mock_converter.convert.return_value = "MOCK ASCII ART"
        
        with mock.patch.object(api, '_get_image_converter', return_value=mock_converter):
            with mock.patch('ascii_forge.api.ascii_api.ImageAsciiConverter') as mock_constructor:
                mock_constructor.return_value = mock_converter
                
                # Call with custom parameters
                api.image_to_ascii(
                    "image.jpg",
                    charset="minimal",
                    width=80,
                    height=40,
                    invert=True
                )
                
                # Verify correct ImageAsciiConverter instantiation
                mock_constructor.assert_called_with(
                    charset="minimal",
                    width=80,
                    height=40,
                    invert=True,
                    dithering=False
                )
    
    # ──── Utility Method Tests ───────────────────────────────────────
    
    def test_get_available_fonts(self, api, mock_banner_generator):
        """📋 Verify font listing works correctly."""
        with mock.patch.object(api, '_banner_generator', mock_banner_generator):
            fonts = api.get_available_fonts()
            
            mock_banner_generator.available_fonts.assert_called_once()
            assert fonts == ["font1", "font2", "standard"]
    
    def test_get_available_styles(self, api):
        """🎨 Verify style listing works correctly."""
        with mock.patch('ascii_forge.api.ascii_api.get_available_styles') as mock_get_styles:
            mock_get_styles.return_value = {"minimal": {}, "boxed": {}}
            
            styles = api.get_available_styles()
            
            assert styles == {"minimal": {}, "boxed": {}}
    
    def test_get_available_alphabets(self, api):
        """🔡 Verify alphabet listing works correctly."""
        with mock.patch('ascii_forge.api.ascii_api.AlphabetManager') as mock_manager:
            mock_manager.list_available_alphabets.return_value = ["general", "blocks"]
            
            alphabets = api.get_available_alphabets()
            
            mock_manager.list_available_alphabets.assert_called_once()
            assert alphabets == ["general", "blocks"]
    
    # ──── File Operations Tests ───────────────────────────────────────
    
    def test_save_to_file(self, api):
        """💾 Verify file saving works correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "output.txt")
            content = "ASCII ART TEST"
            
            # Test saving
            result = api.save_to_file(content, file_path)
            
            # Verify results
            assert result is True
            assert os.path.exists(file_path)
            
            # Verify content
            with open(file_path, 'r', encoding='utf-8') as f:
                saved_content = f.read()
            assert saved_content == content
    
    def test_save_to_file_error_handling(self, api):
        """🛑 Verify file saving handles errors gracefully."""
        with mock.patch('builtins.open', side_effect=IOError("Test error")):
            result = api.save_to_file("content", "/some/path")
            
            assert result is False
    
    def test_save_to_file_creates_directories(self, api):
        """📁 Verify file saving creates necessary directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "nested", "dir", "output.txt")
            
            # Directory shouldn't exist yet
            assert not os.path.exists(os.path.dirname(nested_path))
            
            # Save should create directories
            result = api.save_to_file("content", nested_path)
            
            assert result is True
            assert os.path.exists(nested_path)
            
    # ──── Preview Methods Tests ────────────────────────────────────────
    
    def test_preview_font(self, api):
        """🔍 Verify font preview generation."""
        mock_banner = mock.MagicMock()
        mock_banner.generate.return_value = "FONT PREVIEW"
        
        with mock.patch('ascii_forge.api.ascii_api.BannerGenerator') as mock_generator:
            mock_generator.return_value = mock_banner
            
            result = api.preview_font("big")
            
            # Check BannerGenerator was created with correct font
            mock_generator.assert_called_with(font="big", width=api._banner_generator.width)
            assert result == "FONT PREVIEW"
    
    def test_preview_style(self, api, mock_banner_generator):
        """👁️ Verify style preview generation."""
        with mock.patch.object(api, '_banner_generator', mock_banner_generator):
            api.preview_style("boxed")
            
            mock_banner_generator.generate.assert_called_with("ASCII Forge", style="boxed")
    
    def test_convert_text_to_art(self, api):
        """🎯 Verify raw text conversion without styling."""
        mock_figlet = mock.MagicMock()
        mock_figlet.renderText.return_value = "RAW ASCII ART"
        
        mock_banner = mock.MagicMock()
        mock_banner.figlet = mock_figlet
        
        with mock.patch('ascii_forge.api.ascii_api.BannerGenerator') as mock_generator:
            mock_generator.return_value = mock_banner
            
            result = api.convert_text_to_art("Test")
            
            assert result == "RAW ASCII ART"
            mock_figlet.renderText.assert_called_with("Test")


# ──── Integration Tests ────────────────────────────────────────────────

class TestASCIIForgeAPIIntegration:
    """Integration tests for the API with actual components."""
    
    @pytest.mark.integration
    def test_actual_banner_generation(self):
        """📊 Verify actual banner generation with real components."""
        api = get_api()
        result = api.generate_banner("Test")
        
        assert isinstance(result, str)
        assert "Test" in result
    
    @pytest.mark.integration
    def test_style_application(self):
        """🖼️ Verify styles are actually applied to banners."""
        api = get_api()
        
        # Generate banners with different styles
        minimal = api.generate_banner("X", style="minimal")
        boxed = api.generate_banner("X", style="boxed")
        
        # Boxed should have more lines (borders)
        assert len(boxed.split('\n')) > len(minimal.split('\n'))
        
        # Boxed should contain border characters
        assert any(c in boxed for c in "┌─┐│└┘")
    
    @pytest.mark.integration
    def test_config_integration(self):
        """⚙️ Verify API uses configuration correctly."""
        api = get_api()
        
        # Set a config value
        api.config.set('banner', 'default_style', 'boxed')
        
        # Generate banner without specifying style
        banner = api.generate_banner("Test")
        
        # Should use the configured style
        assert any(c in banner for c in "┌─┐│└┘")  # Has border characters
