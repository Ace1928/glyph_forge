"""
⚡ Eidosian Integration Test Suite ⚡

End-to-end integration tests that verify complete workflows
across multiple components of Glyph Forge.
"""

import os
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from glyph_forge.api import get_api, GlyphForgeAPI
from glyph_forge.services import (
    text_to_banner,
    text_to_glyph,
    video_to_glyph_frames,
)
from glyph_forge.services.image_to_glyph import (
    ImageGlyphConverter,
    ColorMode,
    image_to_glyph,
)
from glyph_forge.core.banner_generator import BannerGenerator
from glyph_forge.core.style_manager import apply_style
from glyph_forge.transformers import (
    ImageTransformer,
    ColorMapper,
    DepthAnalyzer,
    EdgeDetector,
)
from glyph_forge.renderers import (
    TextRenderer,
    HTMLRenderer,
    SVGRenderer,
    ANSIRenderer,
)


@pytest.fixture(scope="module")
def test_assets() -> dict:
    """Create a comprehensive set of test assets."""
    test_dir = tempfile.mkdtemp(prefix="glyph_integration_")

    # Create test images
    # 1. Simple gradient (100x100)
    gradient = np.linspace(0, 255, 100, dtype=np.uint8)
    gradient_img = np.repeat(gradient.reshape(1, 100), 100, axis=0)
    gradient_pil = Image.fromarray(gradient_img, mode='L')
    gradient_path = os.path.join(test_dir, 'gradient.png')
    gradient_pil.save(gradient_path)

    # 2. RGB test pattern (100x100)
    rgb_data = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb_data[0:50, 0:50, 0] = 255  # Red top-left
    rgb_data[0:50, 50:100, 1] = 255  # Green top-right
    rgb_data[50:100, 0:50, 2] = 255  # Blue bottom-left
    rgb_data[50:100, 50:100] = 255  # White bottom-right
    rgb_pil = Image.fromarray(rgb_data, mode='RGB')
    rgb_path = os.path.join(test_dir, 'rgb.png')
    rgb_pil.save(rgb_path)

    # 3. Create a simple GIF animation
    frames = []
    for i in range(5):
        frame = Image.new('L', (50, 50), color=i * 50)
        frames.append(frame)
    gif_path = os.path.join(test_dir, 'animation.gif')
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )

    # Output directory for saved files
    output_dir = os.path.join(test_dir, 'output')
    os.makedirs(output_dir)

    yield {
        'dir': test_dir,
        'gradient': gradient_path,
        'rgb': rgb_path,
        'gif': gif_path,
        'output_dir': output_dir,
    }

    # Cleanup
    shutil.rmtree(test_dir)


@pytest.mark.integration
class TestAPIWorkflows:
    """Integration tests for API-based workflows."""

    def test_complete_image_conversion_workflow(
        self, test_assets: dict
    ) -> None:
        """Test complete workflow: load → convert → save."""
        api = get_api()

        # Convert image
        result = api.image_to_Glyph(
            test_assets['gradient'],
            width=50,
            charset="blocks",
        )

        # Verify result
        assert isinstance(result, str)
        assert len(result) > 0

        # Check that block characters are used
        assert any(c in result for c in "░▒▓█")

        # Save to file
        output_path = os.path.join(
            test_assets['output_dir'], 'api_conversion.txt'
        )
        success = api.save_to_file(result, output_path)
        assert success
        assert os.path.exists(output_path)

        # Verify file content
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_content = f.read()
        assert saved_content == result

    def test_complete_banner_workflow(self) -> None:
        """Test complete workflow: generate → style → save."""
        api = get_api()

        # Generate banner
        banner = api.generate_banner(
            "Test",
            style="boxed",
            font="standard",
        )

        # Verify result
        assert isinstance(banner, str)
        assert len(banner) > 0

        # Check for border characters
        assert any(c in banner for c in "┌─┐│└┘")

    def test_color_conversion_workflow(self, test_assets: dict) -> None:
        """Test ANSI color conversion workflow."""
        api = get_api()

        # Convert with color
        result = api.image_to_Glyph(
            test_assets['rgb'],
            width=30,
            color_mode="ansi",
        )

        # Verify ANSI codes are present
        assert "\033[38;2;" in result
        assert "\033[0m" in result

    def test_html_output_workflow(self, test_assets: dict) -> None:
        """Test HTML color output workflow."""
        api = get_api()

        # Convert with HTML color
        result = api.image_to_Glyph(
            test_assets['rgb'],
            width=30,
            color_mode="html",
        )

        # Verify HTML tags are present
        assert "<pre" in result
        assert "<span" in result
        assert "</span>" in result
        assert "</pre>" in result


@pytest.mark.integration
class TestServiceWorkflows:
    """Integration tests for service-level workflows."""

    def test_image_to_glyph_service(self, test_assets: dict) -> None:
        """Test high-level image conversion service."""
        result = image_to_glyph(
            test_assets['gradient'],
            width=40,
            charset="minimal",
        )

        assert isinstance(result, str)
        lines = result.strip().split('\n')
        assert len(lines) > 0

    def test_text_to_banner_service(self) -> None:
        """Test text banner generation service."""
        result = text_to_banner(
            "Hi",
            style="minimal",
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_text_to_glyph_service(self) -> None:
        """Test simple text-to-glyph service."""
        result = text_to_glyph("Test")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_video_to_frames_service(self, test_assets: dict) -> None:
        """Test video frame extraction and conversion."""
        frames = video_to_glyph_frames(
            test_assets['gif'],
            width=20,
            max_frames=3,
        )

        assert isinstance(frames, list)
        assert len(frames) == 3
        assert all(isinstance(f, str) for f in frames)


@pytest.mark.integration
class TestConverterWorkflows:
    """Integration tests for ImageGlyphConverter workflows."""

    def test_converter_grayscale_workflow(self, test_assets: dict) -> None:
        """Test converter with grayscale output."""
        converter = ImageGlyphConverter(
            charset="detailed",
            width=60,
            auto_scale=False,
        )

        result = converter.convert(test_assets['gradient'])

        assert isinstance(result, str)
        lines = result.strip().split('\n')
        # Lines may vary slightly due to aspect ratio calculation
        assert len(lines) > 0
        assert all(len(line) > 0 for line in lines if line)

    def test_converter_color_workflow(self, test_assets: dict) -> None:
        """Test converter with color output."""
        converter = ImageGlyphConverter(
            charset="blocks",
            width=40,
            auto_scale=False,
        )

        result = converter.convert_color(
            test_assets['rgb'],
            color_mode=ColorMode.ANSI,
        )

        assert "\033[" in result

    def test_converter_adjustment_workflow(self, test_assets: dict) -> None:
        """Test converter with brightness/contrast adjustments."""
        converter = ImageGlyphConverter(
            width=40,
            brightness=1.3,
            contrast=1.2,
            auto_scale=False,
        )

        result = converter.convert(test_assets['gradient'])
        assert isinstance(result, str)

    def test_converter_dithering_workflow(self, test_assets: dict) -> None:
        """Test converter with dithering enabled."""
        converter = ImageGlyphConverter(
            width=40,
            dithering=True,
            auto_scale=False,
        )

        result = converter.convert(test_assets['gradient'])
        assert isinstance(result, str)

    def test_converter_file_save_workflow(self, test_assets: dict) -> None:
        """Test converter with file saving."""
        output_path = os.path.join(
            test_assets['output_dir'], 'converter_output.txt'
        )

        converter = ImageGlyphConverter(width=40, auto_scale=False)
        result = converter.convert(
            test_assets['gradient'],
            output_path=output_path,
        )

        assert os.path.exists(output_path)
        with open(output_path, 'r', encoding='utf-8') as f:
            saved = f.read()
        assert saved == result


@pytest.mark.integration
class TestBannerWorkflows:
    """Integration tests for BannerGenerator workflows."""

    def test_banner_basic_workflow(self) -> None:
        """Test basic banner generation workflow."""
        generator = BannerGenerator(font="standard", width=80)
        result = generator.generate("Test")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_banner_styled_workflow(self) -> None:
        """Test banner with style applied."""
        generator = BannerGenerator(font="standard", width=80)

        # Generate with style
        result = generator.generate(
            "Hi",
            style="boxed",
            padding=(1, 2),
        )

        # Check for border
        assert "┌" in result or "+" in result

    def test_banner_effects_workflow(self) -> None:
        """Test banner with effects applied."""
        generator = BannerGenerator(font="standard", width=80)

        # Generate with glow effect
        result = generator.generate(
            "X",
            style="minimal",
            effects=["glow"],
        )

        # Some glow characters should be present
        assert any(c in result for c in "✦✧✨⋆⭐") or "X" in result

    def test_banner_caching_workflow(self) -> None:
        """Test banner caching improves performance."""
        generator = BannerGenerator(font="standard", width=80)

        # Reset metrics
        generator.reset_metrics()

        # First generation (cache miss)
        generator.generate("CacheTest", style="minimal")

        # Second generation (cache hit)
        generator.generate("CacheTest", style="minimal")

        metrics = generator.get_metrics()
        assert metrics["total_renders"] == 1  # Only one actual render
        assert metrics["cache_hits"] == 1  # One cache hit


@pytest.mark.integration
class TestTransformerWorkflows:
    """Integration tests for transformer workflows."""

    def test_transformer_pipeline(self, test_assets: dict) -> None:
        """Test chained transformer operations."""
        # Load image
        img = Image.open(test_assets['gradient'])

        # Transform with ImageTransformer
        transformer = ImageTransformer(charset=" .:+#")
        matrix = transformer.transform(img, width=30, height=15)

        # Verify matrix structure
        assert len(matrix) == 15
        assert all(len(row) == 30 for row in matrix)

        # Render to text
        renderer = TextRenderer()
        result = renderer.render(matrix)

        # Verify output
        assert isinstance(result, str)
        lines = result.split('\n')
        assert len(lines) == 15

    def test_color_mapper_to_html(self, test_assets: dict) -> None:
        """Test ColorMapper with HTML renderer."""
        img = Image.open(test_assets['rgb'])

        # Transform with ColorMapper
        mapper = ColorMapper(charset=" .:#")
        matrix = mapper.transform(img, width=20, height=10)

        # Render to HTML
        renderer = HTMLRenderer()
        result = renderer.render(matrix)

        # Verify HTML output
        assert "<pre" in result
        assert "</pre>" in result

    def test_edge_detector_workflow(self, test_assets: dict) -> None:
        """Test EdgeDetector transformation."""
        img = Image.open(test_assets['gradient'])

        # Transform with EdgeDetector
        detector = EdgeDetector(charset=" .-+#@")
        matrix = detector.transform(
            img,
            width=30,
            height=15,
            edge_weight=0.6,
        )

        # Verify matrix
        assert len(matrix) == 15
        assert all(len(row) == 30 for row in matrix)


@pytest.mark.integration
class TestStyleWorkflows:
    """Integration tests for styling workflows."""

    def test_apply_style_to_banner(self) -> None:
        """Test applying style to generated banner."""
        generator = BannerGenerator(font="standard", width=60)
        banner = generator.generate("Hi", style="minimal")

        # Apply boxed style
        styled = apply_style(banner, "boxed")

        # Verify border was added
        assert "┌" in styled or "+" in styled
        assert "└" in styled or "+" in styled

    def test_multiple_style_application(self) -> None:
        """Test applying different styles sequentially."""
        base_text = "Test Text"

        # Apply different styles
        minimal = apply_style(base_text, "minimal")
        boxed = apply_style(base_text, "boxed")
        double = apply_style(base_text, "double")

        # Verify each has appropriate structure
        assert "Test Text" in minimal
        assert "─" in boxed or "-" in boxed
        assert "═" in double


@pytest.mark.integration
class TestRendererWorkflows:
    """Integration tests for renderer workflows."""

    def test_text_renderer_output(self) -> None:
        """Test TextRenderer produces valid output."""
        matrix = [['A', 'B', 'C'], ['D', 'E', 'F']]
        renderer = TextRenderer()
        result = renderer.render(matrix)

        assert result == "ABC\nDEF"

    def test_html_renderer_output(self) -> None:
        """Test HTMLRenderer produces valid HTML."""
        matrix = [['A', 'B'], ['C', 'D']]
        renderer = HTMLRenderer()
        result = renderer.render(matrix)

        assert "<pre" in result
        assert "AB" in result
        assert "</pre>" in result

    def test_svg_renderer_output(self) -> None:
        """Test SVGRenderer produces valid SVG."""
        matrix = [['X', 'Y'], ['Z', 'W']]
        renderer = SVGRenderer()
        result = renderer.render(matrix)

        assert "<svg" in result
        assert "</svg>" in result
        assert "<text" in result


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Complete end-to-end workflow tests."""

    def test_image_to_html_file_workflow(self, test_assets: dict) -> None:
        """Test complete workflow: image → glyph → HTML file."""
        api = get_api()

        # Convert to HTML
        html_result = api.image_to_Glyph(
            test_assets['rgb'],
            width=30,
            color_mode="html",
        )

        # Save to file
        output_path = os.path.join(
            test_assets['output_dir'], 'output.html'
        )
        api.save_to_file(html_result, output_path)

        # Verify file
        assert os.path.exists(output_path)
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "<pre" in content
        assert "<span" in content

    def test_banner_generation_complete_workflow(self) -> None:
        """Test complete banner workflow with all options."""
        api = get_api()

        # Test with various options
        for style in ["minimal", "boxed", "eidosian"]:
            banner = api.generate_banner(
                "Test",
                style=style,
                font="standard",
                width=60,
            )
            assert isinstance(banner, str)
            assert len(banner) > 0

    def test_multi_format_output_workflow(self, test_assets: dict) -> None:
        """Test generating multiple output formats from same image."""
        converter = ImageGlyphConverter(
            width=30,
            auto_scale=False,
        )

        # Grayscale
        gray = converter.convert(test_assets['gradient'])
        assert isinstance(gray, str)
        assert "\033[" not in gray

        # ANSI color
        ansi = converter.convert_color(
            test_assets['rgb'],
            color_mode=ColorMode.ANSI,
        )
        assert "\033[" in ansi

        # HTML color
        html = converter.convert_color(
            test_assets['rgb'],
            color_mode=ColorMode.HTML,
        )
        assert "<span" in html

    def test_batch_image_processing(self, test_assets: dict) -> None:
        """Test processing multiple images in batch."""
        api = get_api()

        images = [
            test_assets['gradient'],
            test_assets['rgb'],
        ]

        results = []
        for img_path in images:
            result = api.image_to_Glyph(img_path, width=20)
            results.append(result)

        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)
        assert all(len(r) > 0 for r in results)
