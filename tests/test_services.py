"""Tests for high-level service functions."""

from pathlib import Path
from PIL import Image

from glyph_forge.services import (
    text_to_banner,
    text_to_glyph,
    video_to_glyph_frames,
)


def test_text_to_banner_basic() -> None:
    """Ensure ``text_to_banner`` returns non-empty glyph art."""
    result = text_to_banner("Forge")
    assert isinstance(result, str)
    assert len(result.strip()) > 0


def test_text_to_glyph() -> None:
    """Ensure ``text_to_glyph`` mirrors banner service output."""
    result = text_to_glyph("Forge")
    assert isinstance(result, str)
    assert len(result.strip()) > 0


def test_video_to_glyph_frames(tmp_path: Path) -> None:
    """Verify ``video_to_glyph_frames`` processes GIF frames."""
    # Create simple 2-frame GIF
    frame1 = Image.new("L", (8, 8), color=0)
    frame2 = Image.new("L", (8, 8), color=255)
    gif_path = tmp_path / "two.gif"
    frame1.save(gif_path, save_all=True, append_images=[frame2], duration=20, loop=0)

    frames = video_to_glyph_frames(str(gif_path), width=8)
    assert len(frames) == 2
    assert all(isinstance(f, str) for f in frames)
