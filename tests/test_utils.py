"""
⚡ Eidosian Test Suite: Utility Functions ⚡

Comprehensive tests for utility functions and helper classes.
"""

import pytest
from unittest import mock
import os

from glyph_forge.utils.glyph_utils import (
    sanitize_text,
    resolve_style,
    trim_margins,
    center_Glyph_art,
    measure_Glyph_art,
    detect_box_borders,
    get_terminal_size,
    detect_text_color_support,
    apply_ansi_style,
    wrap_text,
)

from glyph_forge.utils.alphabet_manager import (
    AlphabetManager,
    AlphabetCategory,
    ALPHABETS,
    SPECIAL_SETS,
    LANGUAGES,
)


class TestSanitizeText:
    """Tests for the sanitize_text function."""

    def test_empty_string(self) -> None:
        """Empty string should return empty string."""
        assert sanitize_text("") == ""

    def test_normal_text(self) -> None:
        """Normal text should pass through unchanged."""
        text = "Hello World"
        assert sanitize_text(text) == text

    def test_line_ending_normalization(self) -> None:
        """Line endings should be normalized to \n."""
        assert sanitize_text("line1\r\nline2") == "line1\nline2"
        assert sanitize_text("line1\rline2") == "line1\nline2"

    def test_control_character_removal(self) -> None:
        """Control characters should be removed."""
        text = "Hello\x00World\x1f"
        result = sanitize_text(text)
        assert "\x00" not in result
        assert "\x1f" not in result
        assert "HelloWorld" == result

    def test_multiple_newlines_collapsed(self) -> None:
        """Multiple consecutive newlines should be collapsed to two."""
        text = "line1\n\n\n\nline2"
        result = sanitize_text(text)
        assert "\n\n\n" not in result
        assert "line1\n\nline2" == result


class TestResolveStyle:
    """Tests for the resolve_style function."""

    def test_unknown_style_returns_default(self) -> None:
        """Unknown style name should return default configuration."""
        style = resolve_style("nonexistent_style")
        assert "border" in style
        assert "padding" in style
        assert "alignment" in style
        assert "effects" in style

    def test_known_style_returns_config(self) -> None:
        """Known style name should return appropriate configuration."""
        # Test several known styles
        cyberpunk = resolve_style("cyberpunk")
        assert cyberpunk["border"] == "heavy"
        assert "glow" in cyberpunk["effects"]

        minimalist = resolve_style("minimalist")
        assert minimalist["border"] is None
        assert minimalist["effects"] == []


class TestTrimMargins:
    """Tests for the trim_margins function."""

    def test_empty_string(self) -> None:
        """Empty string should return empty string."""
        assert trim_margins("") == ""

    def test_no_margins(self) -> None:
        """Text without margins should pass through."""
        text = "Hello\nWorld"
        result = trim_margins(text)
        assert "Hello" in result
        assert "World" in result

    def test_leading_empty_lines_removed(self) -> None:
        """Leading empty lines should be removed."""
        text = "\n\n\nHello"
        result = trim_margins(text)
        assert not result.startswith("\n")
        assert "Hello" in result

    def test_trailing_empty_lines_removed(self) -> None:
        """Trailing empty lines should be removed."""
        text = "Hello\n\n\n"
        result = trim_margins(text)
        assert not result.endswith("\n\n")

    def test_consistent_left_padding_removed(self) -> None:
        """Consistent left padding should be removed."""
        text = "    line1\n    line2\n    line3"
        result = trim_margins(text)
        assert not result.startswith("    ")


class TestCenterGlyphArt:
    """Tests for the center_Glyph_art function."""

    def test_center_narrow_text(self) -> None:
        """Narrow text should be centered within width."""
        art = "Hi"
        result = center_Glyph_art(art, 10)
        assert len(result) >= 2  # At least the original text
        assert "Hi" in result

    def test_center_multiple_lines(self) -> None:
        """Multiple lines should all be centered."""
        art = "Hi\nBye"
        result = center_Glyph_art(art, 10)
        lines = result.split("\n")
        assert len(lines) == 2

    def test_wide_text_unchanged(self) -> None:
        """Text wider than target should be unchanged."""
        art = "This is very long text"
        result = center_Glyph_art(art, 5)
        assert "This is very long text" in result


class TestMeasureGlyphArt:
    """Tests for the measure_Glyph_art function."""

    def test_empty_string(self) -> None:
        """Empty string should return (0, 1)."""
        width, height = measure_Glyph_art("")
        assert width == 0
        assert height == 1

    def test_single_line(self) -> None:
        """Single line should return (length, 1)."""
        width, height = measure_Glyph_art("Hello")
        assert width == 5
        assert height == 1

    def test_multiple_lines(self) -> None:
        """Multiple lines should return correct dimensions."""
        art = "Hello\nWorld!"
        width, height = measure_Glyph_art(art)
        assert width == 6  # "World!" is longest
        assert height == 2


class TestDetectBoxBorders:
    """Tests for the detect_box_borders function."""

    def test_no_border(self) -> None:
        """Text without border should return None."""
        art = "Hello\nWorld"
        assert detect_box_borders(art) is None

    def test_single_border(self) -> None:
        """Single border should be detected."""
        art = "┌───┐\n│Hi │\n└───┘"
        result = detect_box_borders(art)
        assert result == "single"

    def test_double_border(self) -> None:
        """Double border should be detected."""
        art = "╔═══╗\n║Hi ║\n╚═══╝"
        result = detect_box_borders(art)
        assert result == "double"

    def test_ascii_border(self) -> None:
        """ASCII border should be detected."""
        art = "+---+\n|Hi |\n+---+"
        result = detect_box_borders(art)
        assert result == "Glyph"

    def test_short_text(self) -> None:
        """Text with fewer than 3 lines should return None."""
        art = "Hi"
        assert detect_box_borders(art) is None


class TestGetTerminalSize:
    """Tests for the get_terminal_size function."""

    def test_returns_tuple(self) -> None:
        """Function should return tuple of two integers."""
        width, height = get_terminal_size()
        assert isinstance(width, int)
        assert isinstance(height, int)
        assert width > 0
        assert height > 0

    @mock.patch('shutil.get_terminal_size')
    def test_uses_shutil(self, mock_get_size) -> None:
        """Function should use shutil.get_terminal_size."""
        mock_get_size.return_value = (100, 50)
        width, height = get_terminal_size()
        mock_get_size.assert_called_once()

    @mock.patch('shutil.get_terminal_size', side_effect=AttributeError)
    def test_fallback_on_error(self, mock_get_size) -> None:
        """Function should return fallback values on error."""
        width, height = get_terminal_size()
        assert width == 80
        assert height == 24


class TestDetectTextColorSupport:
    """Tests for the detect_text_color_support function."""

    @mock.patch.dict(os.environ, {"NO_COLOR": "1"})
    def test_no_color_env_variable(self) -> None:
        """NO_COLOR environment variable should disable colors."""
        result = detect_text_color_support()
        assert result == 0

    @mock.patch.dict(os.environ, {"NO_COLOR": ""}, clear=True)
    @mock.patch.dict(os.environ, {"COLORTERM": "truecolor"})
    def test_truecolor_support(self) -> None:
        """COLORTERM=truecolor should return 3."""
        # Clear NO_COLOR first
        if "NO_COLOR" in os.environ:
            del os.environ["NO_COLOR"]
        result = detect_text_color_support()
        assert result == 3

    @mock.patch.dict(os.environ, {"NO_COLOR": "", "TERM": "xterm-256color", "COLORTERM": ""})
    def test_256_color_support(self) -> None:
        """xterm-256color should return 2."""
        result = detect_text_color_support()
        assert result == 2


class TestApplyAnsiStyle:
    """Tests for the apply_ansi_style function."""

    def test_single_style(self) -> None:
        """Single style should be applied correctly."""
        result = apply_ansi_style("Hello", "bold")
        assert "\033[1m" in result  # Bold code
        assert "\033[0m" in result  # Reset code
        assert "Hello" in result

    def test_multiple_styles(self) -> None:
        """Multiple styles should all be applied."""
        result = apply_ansi_style("Hello", ["bold", "red"])
        assert "\033[1m" in result  # Bold code
        assert "\033[31m" in result  # Red code
        assert "\033[0m" in result  # Reset code

    def test_unknown_style(self) -> None:
        """Unknown style should be ignored."""
        result = apply_ansi_style("Hello", "nonexistent")
        assert "Hello" in result
        assert "\033[0m" in result  # Reset should still be applied


class TestWrapText:
    """Tests for the wrap_text function."""

    def test_short_text(self) -> None:
        """Text shorter than width should be unchanged."""
        text = "Hello"
        result = wrap_text(text, 20)
        assert result == text

    def test_long_line_wrapped(self) -> None:
        """Long line should be wrapped at word boundaries."""
        text = "This is a very long line that needs wrapping"
        result = wrap_text(text, 20)
        lines = result.split("\n")
        assert len(lines) > 1
        assert all(len(line) <= 20 for line in lines)

    def test_preserves_newlines(self) -> None:
        """Existing newlines should be preserved."""
        text = "Line 1\nLine 2"
        result = wrap_text(text, 20)
        assert "Line 1" in result
        assert "Line 2" in result

    def test_zero_width(self) -> None:
        """Zero width should return unchanged text."""
        text = "Hello"
        result = wrap_text(text, 0)
        assert result == text


class TestAlphabetManager:
    """Tests for the AlphabetManager class."""

    def test_get_alphabet_default(self) -> None:
        """Default alphabet should be 'general'."""
        result = AlphabetManager.get_alphabet()
        assert result == ALPHABETS["general"]

    def test_get_alphabet_known(self) -> None:
        """Known alphabet should be returned."""
        result = AlphabetManager.get_alphabet("detailed")
        assert result == ALPHABETS["detailed"]

    def test_get_alphabet_unknown(self) -> None:
        """Unknown alphabet should return general."""
        result = AlphabetManager.get_alphabet("nonexistent")
        assert result == ALPHABETS["general"]

    def test_get_special_set(self) -> None:
        """Special set should be returned."""
        result = AlphabetManager.get_special_set("box_drawing")
        assert result == SPECIAL_SETS["box_drawing"]

    def test_list_available_alphabets(self) -> None:
        """Should return list of all alphabet names."""
        result = AlphabetManager.list_available_alphabets()
        assert isinstance(result, list)
        assert "general" in result
        assert "detailed" in result
        assert len(result) > 5

    def test_list_special_sets(self) -> None:
        """Should return list of all special set names."""
        result = AlphabetManager.list_special_sets()
        assert isinstance(result, list)
        assert "box_drawing" in result
        assert "braille" in result

    def test_create_density_map(self) -> None:
        """Density map should cover all 256 values."""
        charset = "ABC"
        result = AlphabetManager.create_density_map(charset)
        assert len(result) == 256
        assert all(v in charset for v in result.values())

    def test_create_custom_density_map(self) -> None:
        """Custom density map should respect range."""
        charset = "ABC"
        result = AlphabetManager.create_custom_density_map(
            charset, min_value=100, max_value=200
        )
        assert 100 in result
        assert 200 in result
        assert 50 not in result

    def test_invert_charset(self) -> None:
        """Charset should be reversed."""
        result = AlphabetManager.invert_charset("ABC")
        assert result == "CBA"

    def test_combine_alphabets(self) -> None:
        """Combined alphabets should include all unique characters."""
        result = AlphabetManager.combine_alphabets(["binary", "contrast"])
        assert "0" in result
        assert "1" in result
        assert " " in result
        assert "█" in result

    def test_get_charset_info(self) -> None:
        """Should return complete charset information."""
        result = AlphabetManager.get_charset_info("general")
        assert result["name"] == "general"
        assert result["charset"] == ALPHABETS["general"]
        assert result["length"] == len(ALPHABETS["general"])
        assert "category" in result

    def test_register_alphabet(self) -> None:
        """New alphabet should be registered."""
        AlphabetManager.register_alphabet("test_abc", "XYZ")
        assert "test_abc" in AlphabetManager.list_available_alphabets()
        assert AlphabetManager.get_alphabet("test_abc") == "XYZ"

    def test_filter_charset(self) -> None:
        """Characters should be filtered by pattern."""
        charset = "ABC123"
        result = AlphabetManager.filter_charset(charset, r"[A-Z]")
        assert result == "ABC"


class TestAlphabetData:
    """Tests for alphabet data integrity."""

    def test_all_alphabets_non_empty(self) -> None:
        """All alphabets should be non-empty strings."""
        for name, charset in ALPHABETS.items():
            assert isinstance(charset, str), f"{name} is not a string"
            assert len(charset) > 0, f"{name} is empty"

    def test_all_special_sets_non_empty(self) -> None:
        """All special sets should be non-empty strings."""
        for name, charset in SPECIAL_SETS.items():
            assert isinstance(charset, str), f"{name} is not a string"
            assert len(charset) > 0, f"{name} is empty"

    def test_all_languages_non_empty(self) -> None:
        """All language sets should be non-empty strings."""
        for name, charset in LANGUAGES.items():
            assert isinstance(charset, str), f"{name} is not a string"
            assert len(charset) > 0, f"{name} is empty"

    def test_density_alphabets_have_space(self) -> None:
        """Density-based alphabets should include space character."""
        density_alphabets = ["general", "detailed", "simple", "minimal", "blocks"]
        for name in density_alphabets:
            charset = ALPHABETS.get(name, "")
            assert " " in charset, f"{name} missing space character"
