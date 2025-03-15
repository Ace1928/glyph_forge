import unittest
import os
import re
from unittest import mock
from ascii_forge.core.banner_generator import BannerGenerator, BannerStyle

class TestBannerGenerator(unittest.TestCase):
    """
    Eidosian test suite for the BannerGenerator component.
    
    Tests the core functionality of ASCII banner generation with
    surgical precision and zero redundancy.
    """
    
    def setUp(self):
        """Initialize test environment with precision."""
        self.generator = BannerGenerator(font="standard", width=80)
        
    def test_basic_banner_generation(self):
        """Verify fundamental banner generation works correctly."""
        banner = self.generator.generate("Eidos", style="minimal")
        self.assertIsInstance(banner, str)
        self.assertIn('Eidos', banner)
        
    def test_style_application(self):
        """Verify style presets are correctly applied to banners."""
        # Test boxed style
        boxed = self.generator.generate("Test", style="boxed")
        self.assertIn('┌', boxed)  # Has border
        self.assertIn('└', boxed)
        
        # Test eidosian style
        eidosian = self.generator.generate("Test", style=BannerStyle.EIDOSIAN.value)
        self.assertIn('┏', eidosian)  # Has heavy border
        
        # Test minimal style
        minimal = self.generator.generate("Test", style="minimal")
        self.assertNotIn('┌', minimal)  # No border
        
    def test_custom_parameters(self):
        """Verify custom style parameters override defaults."""
        # Custom border
        custom_border = self.generator.generate("Test", style="minimal", border="double")
        self.assertIn('╔', custom_border)
        
        # Custom padding
        padding = self.generator.generate("Test", style="minimal", padding=(2, 3))
        lines = padding.split('\n')
        self.assertTrue(len(lines) > 4)  # Should have padding lines
        
        # Custom alignment
        right_aligned = self.generator.generate("Test", style="minimal", alignment="right")
        lines = right_aligned.split('\n')
        non_empty = [line for line in lines if line.strip()]
        if non_empty:
            self.assertTrue(non_empty[0].startswith(' '))  # Right alignment adds spaces on left
            
    def test_effects_application(self):
        """Verify special effects are correctly applied."""
        # Shadow effect
        shadow = self.generator.generate("X", style="minimal", effects=["shadow"])
        self.assertIn('░', shadow)  # Contains shadow character
        
        # Glow effect
        glow = self.generator.generate("X", style="minimal", effects=["glow"])
        any_glow_char = any(c in glow for c in "✦✧✨⋆⭐")
        self.assertTrue(any_glow_char)  # Contains at least one glow character
        
    def test_cache_functionality(self):
        """Verify caching mechanism works correctly."""
        # First call - should not be cached
        self.generator.generate("CacheTest", style="minimal")
        
        # Second call with same parameters - should use cache
        self.generator.generate("CacheTest", style="minimal")
        
        metrics = self.generator.get_metrics()
        self.assertEqual(metrics["total_renders"], 1)  # Only one actual render
        self.assertEqual(metrics["cache_hits"], 1)  # One cache hit
        
    def test_available_fonts(self):
        """Verify font listing functionality."""
        fonts = self.generator.available_fonts()
        self.assertIsInstance(fonts, list)
        self.assertGreater(len(fonts), 0)
        self.assertIn("standard", fonts)
        
    def test_preview_fonts(self):
        """Verify font preview generation."""
        preview = self.generator.preview_fonts("X", limit=2)
        self.assertIsInstance(preview, str)
        self.assertIn("Font:", preview)
        
    def test_color_support(self):
        """Verify ANSI color application."""
        colored = self.generator.generate("Test", style="minimal", color=True)
        self.assertIn('\033[', colored)  # Contains ANSI color codes
        
    def test_unicode_detection(self):
        """Verify Unicode support detection."""
        # Mock os.devnull to simulate Unicode support
        with mock.patch('builtins.open') as mock_open:
            mock_file = mock.MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            # Recreate generator to trigger Unicode detection
            gen = BannerGenerator(font="standard")
            self.assertTrue(gen._unicode_supported)
            
        # Mock exception to simulate lack of Unicode support
        with mock.patch('builtins.open') as mock_open:
            mock_open.side_effect = UnicodeEncodeError('utf-8', b'test', 0, 1, 'test error')
            
            # Recreate generator to trigger Unicode detection
            gen = BannerGenerator(font="standard")
            self.assertFalse(gen._unicode_supported)
            
    def test_template_rendering(self):
        """Verify template variable substitution."""
        template = "Hello {name}!"
        result = self.generator.render_template(template, {"name": "Eidosian"})
        self.assertIn("Hello Eidosian", result)
        
    def test_invalid_font_handling(self):
        """Verify graceful handling of invalid fonts."""
        with self.assertRaises(ValueError):
            BannerGenerator(font="nonexistent_font")
            
    def test_reset_metrics(self):
        """Verify metrics reset functionality."""
        # Generate banner to increment metrics
        self.generator.generate("Test")
        self.assertGreater(self.generator._render_count, 0)
        
        # Reset metrics
        self.generator.reset_metrics()
        self.assertEqual(self.generator._render_count, 0)
        self.assertEqual(self.generator._cache_hits, 0)
        self.assertEqual(len(self.generator.cache), 0)

if __name__ == '__main__':
    unittest.main()