"""Memory-bounded video-to-glyph conversion service."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .image_to_glyph import ImageGlyphConverter
from .video_to_images import iter_video_images


def iter_video_glyph_frames(
    video_path: str | Path,
    width: int = 80,
    max_frames: int | None = None,
    color_mode: str = "none",
) -> Iterator[str]:
    """Yield converted glyph frames as they are decoded."""

    mode = color_mode.casefold()
    if mode not in {"none", "ansi", "html"}:
        raise ValueError("color_mode must be none, ansi, or html")
    converter = ImageGlyphConverter(width=width)
    for image in iter_video_images(video_path, max_frames=max_frames):
        if mode == "none":
            yield converter.convert(image)
        else:
            yield converter.convert_color(image, color_mode=mode)


def video_to_glyph_frames(
    video_path: str | Path,
    width: int = 80,
    max_frames: int | None = None,
    color_mode: str = "none",
) -> list[str]:
    """Return glyph frames as a compatibility list.

    Use :func:`iter_video_glyph_frames` for live playback and long videos.
    """

    return list(
        iter_video_glyph_frames(
            video_path,
            width=width,
            max_frames=max_frames,
            color_mode=color_mode,
        )
    )


__all__ = ["iter_video_glyph_frames", "video_to_glyph_frames"]
