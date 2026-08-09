"""Encode an image iterable as an animated GIF."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PIL import Image


def display_stream_to_video(
    frames: Iterable[Image.Image],
    output_path: str | Path,
    fps: float = 10,
) -> None:
    """Save an iterable of frames as an animated GIF."""

    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("At least one frame is required") from exc

    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    first.save(
        destination,
        save_all=True,
        append_images=iterator,
        duration=max(1, round(1000 / fps)),
        loop=0,
    )


__all__ = ["display_stream_to_video"]
