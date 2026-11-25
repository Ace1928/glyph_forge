"""Tests for high-level service functions."""

from pathlib import Path
from unittest import mock

from PIL import Image

from glyph_forge.services import (
    stream_browser,
    stream_source,
    stream_webcam,
    text_to_banner,
    text_to_glyph,
    video_to_glyph_frames,
    video_to_images,
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


def test_video_to_images_generator(tmp_path: Path) -> None:
    """Verify ``video_to_images`` yields frames lazily."""
    frame1 = Image.new("L", (4, 4), color=0)
    frame2 = Image.new("L", (4, 4), color=255)
    gif_path = tmp_path / "tiny.gif"
    frame1.save(gif_path, save_all=True, append_images=[frame2], duration=20, loop=0)

    images_iter = video_to_images(str(gif_path))
    assert not isinstance(images_iter, list)
    first = next(iter(images_iter))
    assert isinstance(first, Image.Image)
    all_frames = list(video_to_images(str(gif_path)))
    assert len(all_frames) == 2


def test_image_output_metadata(tmp_path: Path) -> None:
    """Ensure output metadata sidecar is written for image conversions."""
    from glyph_forge.services.image_to_glyph import ImageGlyphConverter

    img = Image.new("L", (4, 4), color=0)
    out_dir = tmp_path / "output_dir"
    converter = ImageGlyphConverter(width=4)
    converter.convert(img, output_path=str(out_dir))

    outputs = list(out_dir.glob("*.txt"))
    assert outputs
    meta_path = outputs[0].with_suffix(".metadata.json")
    assert meta_path.exists()
    payload = meta_path.read_text(encoding="utf-8")
    assert "source_width" in payload


def test_stream_source_builds_args() -> None:
    """Verify streaming helper builds stream arguments."""
    with mock.patch("glyph_forge.services.streaming.run_stream") as run_stream:
        stream_source(
            "video.mp4",
            fps=12,
            mode="ascii",
            color=False,
            extra_args=["--debug"],
        )

        run_stream.assert_called_once()
        args = run_stream.call_args.args[0]
        assert args[0] == "video.mp4"
        assert "--fps" in args and "12" in args
        assert "--mode" in args and "ascii" in args
        assert "--color" in args and "none" in args
        assert "--resolution" in args
        assert "--debug" in args


def test_stream_webcam_builds_args() -> None:
    """Verify webcam streaming arguments."""
    with mock.patch("glyph_forge.services.streaming.run_stream") as run_stream:
        stream_webcam(device=2, fps=10)

        run_stream.assert_called_once()
        args = run_stream.call_args.args[0]
        # New format: --webcam 2
        assert args[:2] == ["--webcam", "2"]
        assert "--fps" in args and "10" in args


def test_stream_browser_builds_args() -> None:
    """Verify browser streaming builds launch arguments."""
    with mock.patch("glyph_forge.services.streaming.run_stream") as run_stream:
        stream_browser("https://example.com", fps=15, kiosk=True)

        run_stream.assert_called_once()
        args = run_stream.call_args.args[0]
        # Maps to --screen in new engine
        assert "--screen" in args
