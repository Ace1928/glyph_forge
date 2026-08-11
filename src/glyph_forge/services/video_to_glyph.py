"""Memory-bounded video-to-glyph conversion service."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

from ..contracts import RenderFormat, RenderRequest
from ..rendering import render_image
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
    output_format = {
        "none": RenderFormat.TEXT,
        "ansi": RenderFormat.TRUECOLOR,
        "html": RenderFormat.HTML,
    }[mode]
    terminal = shutil.get_terminal_size(fallback=(80, 24))
    request = RenderRequest(
        width=width,
        output_format=output_format,
        max_width=max(1, terminal.columns - 2),
        max_height=max(1, terminal.lines - 3),
        cell_aspect=0.55,
        resample="lanczos",
    )
    for image in iter_video_images(video_path, max_frames=max_frames):
        artifact = render_image(image, request)
        assert isinstance(artifact.data, str)
        yield artifact.data


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
