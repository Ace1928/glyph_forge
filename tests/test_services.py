"""Tests for high-level service functions."""

from pathlib import Path

from PIL import Image

from glyph_forge.services import (
    iter_video_glyph_frames,
    iter_video_images,
    text_to_banner,
    text_to_glyph,
    video_to_glyph_frames,
)
from glyph_forge.services.display_stream_to_video import display_stream_to_video


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


def test_streaming_video_services_are_lazy_and_memory_bounded(tmp_path: Path) -> None:
    frames = [Image.new("RGB", (8, 8), color=value) for value in (0, 64, 128)]
    gif_path = tmp_path / "three.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=20,
        loop=0,
    )

    image_stream = iter_video_images(gif_path, max_frames=2)
    glyph_stream = iter_video_glyph_frames(gif_path, width=8, max_frames=2)

    assert iter(image_stream) is image_stream
    assert iter(glyph_stream) is glyph_stream
    assert len(list(image_stream)) == 2
    assert len(list(glyph_stream)) == 2


def test_display_stream_to_video_accepts_a_generator(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "animation.gif"
    frames = (Image.new("RGB", (4, 4), color=value) for value in (0, 255))

    display_stream_to_video(frames, destination, fps=20)

    with Image.open(destination) as result:
        assert result.n_frames == 2
        assert result.info["duration"] == 50
