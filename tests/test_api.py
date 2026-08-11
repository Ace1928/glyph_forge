"""
⚡ Eidosian Test Suite: Glyph Forge API Integration ⚡

A brutal, zero-compromise test suite that validates the complete
Glyph Forge API with maximum coverage and surgical precision.
Every feature is validated with atomic tests and crystal-clear assertions.
"""

import importlib
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from PIL import Image

from glyph_forge.api.glyph_api import GlyphForgeAPI, get_api
from glyph_forge.contracts import RenderArtifact, RenderFormat, RenderRequest
from glyph_forge.projects import RenderPreset


def test_top_level_image_helper_resolves_to_callable() -> None:
    """The lazy public export must not resolve to its service module."""

    from glyph_forge import image_to_glyph

    assert callable(image_to_glyph)


def test_video_metrics_result_is_available_from_the_public_package() -> None:
    from glyph_forge import VideoExportResult

    result = VideoExportResult(
        output=Path("output.mp4"),
        rendered_frames=30,
        fps=30.0,
        elapsed=1.0,
        width=640,
        height=360,
        columns=80,
        rows=45,
    )

    assert result.to_dict()["render_fps"] == 30.0


def test_legacy_image_service_alias_uses_canonical_module() -> None:
    """Legacy mixed-case imports must work without a case-colliding file."""

    services = importlib.import_module("glyph_forge.services")
    canonical = importlib.import_module("glyph_forge.services.image_to_glyph")
    legacy = importlib.import_module("glyph_forge.services.image_to_Glyph")

    assert services.image_to_glyph is canonical
    assert services.image_to_Glyph is canonical
    assert legacy is canonical


@pytest.fixture
def api():
    """Create a fresh API instance for each test."""
    # Reset singleton for test isolation
    import glyph_forge.api.glyph_api

    glyph_forge.api.glyph_api._api_instance = None

    # Create and return new instance
    api = GlyphForgeAPI()

    # Inject test config values
    api.config.set("banner", "default_style", "minimal")
    api.config.set("banner", "default_font", "standard")
    api.config.set("banner", "default_width", 80)

    return api


@pytest.fixture
def mock_banner_generator():
    """Mock the BannerGenerator for isolated testing."""
    with mock.patch(
        "glyph_forge.core.banner_generator.BannerGenerator"
    ) as mock_generator:
        # Configure mock
        mock_instance = mock_generator.return_value
        mock_instance.generate.return_value = "MOCK BANNER"
        mock_instance.available_fonts.return_value = ["font1", "font2", "standard"]
        mock_instance.font = "standard"
        mock_instance.width = 80

        yield mock_instance


class TestGlyphForgeAPI:
    """Comprehensive test suite for the Glyph Forge API."""

    # ──── Core API Initialization Tests ───────────────────────────────

    def test_singleton_pattern(self):
        """🔂 Verify API singleton pattern works correctly."""
        api1 = get_api()
        api2 = get_api()

        assert api1 is api2  # Same instance returned

    def test_api_initialization(self, api):
        """🧰 Verify API initializes core components properly."""
        assert api._banner_generator is not None

    def test_structured_renderer_uses_canonical_pipeline(self, api):
        """The high-level API returns the canonical structured artifact."""

        request = RenderRequest(width=2, height=1, brightness=1.0, contrast=1.0)
        artifact = api.render_image(Image.new("RGB", (2, 1), "white"), request)

        assert isinstance(artifact, RenderArtifact)
        assert artifact.request is request
        assert (artifact.columns, artifact.rows) == (2, 1)

    def test_project_preset_and_batch_workflows_share_the_public_contract(
        self,
        api,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setenv("GLYPH_FORGE_CONFIG_HOME", str(tmp_path / "config"))
        source = tmp_path / "source.png"
        Image.new("RGB", (4, 2), "white").save(source)
        project_path = tmp_path / "work.glyphforge.json"
        request = RenderRequest(width=4, height=2, output_format="svg")

        project = api.create_project(
            project_path,
            source,
            name="Work",
            request=request,
        )
        session = api.open_project(project_path, autosave_delay=None)
        artifact = api.render_project(project_path)
        preset = RenderPreset("Shared", request)
        report = api.render_batch(
            [source],
            tmp_path / "batch",
            preset,
            workers=1,
        )

        assert project.name == session.project.name == "Work"
        assert artifact.request == request
        assert report.succeeded == 1
        assert report.results[0].destination.suffix == ".svg"
        assert (tmp_path / "config" / "recent_projects.json").is_file()

    # ──── Banner Generation Tests ───────────────────────────────────────

    def test_generate_banner_basic(self, api, mock_banner_generator):
        """📝 Verify basic banner generation."""
        with mock.patch.object(api, "_banner_generator", mock_banner_generator):
            result = api.generate_banner("Test Text")

            # Check correct method was called
            mock_banner_generator.generate.assert_called_once()
            assert isinstance(result, str)

    def test_generate_banner_with_style(self, api, mock_banner_generator):
        """🎭 Verify banner generation with style parameter."""
        with mock.patch.object(api, "_banner_generator", mock_banner_generator):
            api.generate_banner("Test", style="boxed")

            mock_banner_generator.generate.assert_called_with(
                "Test", style="boxed", effects=None, color=False
            )

    def test_generate_banner_with_custom_font(self, api, mock_banner_generator):
        """🔤 Verify banner generation with custom font creates new generator."""
        with mock.patch("glyph_forge.api.glyph_api.BannerGenerator") as mock_bg_class:
            mock_bg_class.return_value = mock_banner_generator

            api.generate_banner("Test", font="big")

            # Should create a new BannerGenerator with the font
            mock_bg_class.assert_called_once()
            args, kwargs = mock_bg_class.call_args
            assert kwargs["font"] == "big"

    # ──── Image Conversion Tests ───────────────────────────────────────

    def test_image_to_glyph_basic(self, api):
        """🖼️ Verify convenient plain image conversion."""

        result = api.image_to_glyph(
            Image.new("RGB", (2, 1), "white"),
            width=2,
            height=1,
            brightness=1.0,
            contrast=1.0,
        )

        assert isinstance(result, str)
        assert len(result) == 2

    def test_image_to_glyph_with_color(self, api):
        """🌈 Verify truecolor conversion uses the same engine."""

        result = api.image_to_glyph(
            Image.new("RGB", (1, 1), (17, 34, 51)),
            width=1,
            height=1,
            brightness=1.0,
            contrast=1.0,
            color_mode="ansi",
        )

        assert "\x1b[38;2;17;34;51m" in result

    def test_mixed_case_image_alias_warns_and_delegates(self, api):
        with mock.patch.object(api, "image_to_glyph", return_value="compat") as modern:
            with pytest.warns(DeprecationWarning, match="image_to_Glyph"):
                result = api.image_to_Glyph("image.jpg", width=80)

        assert result == "compat"
        modern.assert_called_once()

    def test_image_to_Glyph_with_params(self, api):
        """⚙️ Verify parameter forwarding to image converter."""
        artifact = mock.MagicMock(spec=RenderArtifact)
        artifact.data = "MOCK Glyph ART"
        with mock.patch.object(api, "render_image", return_value=artifact) as renderer:
            api.image_to_glyph(
                "image.jpg",
                charset="minimal",
                width=80,
                height=40,
                invert=True,
                brightness=1.0,
                contrast=1.0,
            )

        request = renderer.call_args.args[1]
        assert isinstance(request, RenderRequest)
        assert request.charset == "minimal"
        assert (request.width, request.height) == (80, 40)
        assert request.invert is True
        assert request.render_format is RenderFormat.TEXT

    # ──── Utility Method Tests ───────────────────────────────────────

    def test_get_available_fonts(self, api, mock_banner_generator):
        """📋 Verify font listing works correctly."""
        with mock.patch.object(api, "_banner_generator", mock_banner_generator):
            fonts = api.get_available_fonts()

            mock_banner_generator.available_fonts.assert_called_once()
            assert fonts == ["font1", "font2", "standard"]

    def test_get_available_styles(self, api):
        """🎨 Verify style listing works correctly."""
        with mock.patch(
            "glyph_forge.api.glyph_api.get_available_styles"
        ) as mock_get_styles:
            mock_get_styles.return_value = {"minimal": {}, "boxed": {}}

            styles = api.get_available_styles()

            assert styles == {"minimal": {}, "boxed": {}}

    def test_get_available_alphabets(self, api):
        """🔡 Verify alphabet listing works correctly."""
        with mock.patch("glyph_forge.api.glyph_api.AlphabetManager") as mock_manager:
            mock_manager.list_available_alphabets.return_value = ["general", "blocks"]

            alphabets = api.get_available_alphabets()

            mock_manager.list_available_alphabets.assert_called_once()
            assert alphabets == ["general", "blocks"]

    # ──── File Operations Tests ───────────────────────────────────────

    def test_save_to_file(self, api):
        """💾 Verify file saving works correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "output.txt")
            content = "Glyph ART TEST"

            # Test saving
            result = api.save_to_file(content, file_path)

            # Verify results
            assert result is True
            assert os.path.exists(file_path)

            # Verify content
            with open(file_path, "r", encoding="utf-8") as f:
                saved_content = f.read()
            assert saved_content == content

    def test_save_to_file_error_handling(self, api):
        """🛑 Verify file saving handles errors gracefully."""
        with mock.patch(
            "glyph_forge.persistence.tempfile.NamedTemporaryFile",
            side_effect=IOError("Test error"),
        ):
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

        with mock.patch("glyph_forge.api.glyph_api.BannerGenerator") as mock_generator:
            mock_generator.return_value = mock_banner

            result = api.preview_font("big")

            # Check BannerGenerator was created with correct font
            mock_generator.assert_called_with(
                font="big", width=api._banner_generator.width
            )
            assert result == "FONT PREVIEW"

    def test_preview_style(self, api, mock_banner_generator):
        """👁️ Verify style preview generation."""
        with mock.patch.object(api, "_banner_generator", mock_banner_generator):
            api.preview_style("boxed")

            mock_banner_generator.generate.assert_called_with(
                "Glyph Forge", style="boxed"
            )

    def test_convert_text_to_art(self, api):
        """🎯 Verify raw text conversion without styling."""
        mock_figlet = mock.MagicMock()
        mock_figlet.renderText.return_value = "RAW Glyph ART"

        mock_banner = mock.MagicMock()
        mock_banner.figlet = mock_figlet

        with mock.patch("glyph_forge.api.glyph_api.BannerGenerator") as mock_generator:
            mock_generator.return_value = mock_banner

            result = api.convert_text_to_art("Test")

            assert result == "RAW Glyph ART"
            mock_figlet.renderText.assert_called_with("Test")


# ──── Integration Tests ────────────────────────────────────────────────


class TestGlyphForgeAPIIntegration:
    """Integration tests for the API with actual components."""

    @pytest.mark.integration
    def test_actual_banner_generation(self):
        """📊 Verify actual banner generation with real components."""
        api = get_api()
        result = api.generate_banner("Test")

        assert isinstance(result, str)
        assert result.strip() != ""

    @pytest.mark.integration
    def test_style_application(self):
        """🖼️ Verify styles are actually applied to banners."""
        api = get_api()

        # Generate banners with different styles
        minimal = api.generate_banner("X", style="minimal")
        boxed = api.generate_banner("X", style="boxed")

        # Boxed should have more lines (borders)
        assert len(boxed.split("\n")) > len(minimal.split("\n"))

        # Boxed should contain border characters
        assert any(c in boxed for c in "┌─┐│└┘")

    @pytest.mark.integration
    def test_config_integration(self):
        """⚙️ Verify API uses configuration correctly."""
        api = get_api()

        # Set a config value
        api.config.set("banner", "default_style", "boxed")

        # Generate banner without specifying style
        banner = api.generate_banner("Test")

        # Should use the configured style
        assert any(c in banner for c in "┌─┐│└┘")  # Has border characters
