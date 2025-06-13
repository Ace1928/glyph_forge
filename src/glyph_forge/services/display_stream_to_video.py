"""Render a sequence of images to an animated GIF."""

from typing import List
from PIL import Image


def display_stream_to_video(frames: List[Image.Image], output_path: str, fps: int = 10) -> None:
    """Save frames as an animated GIF.

    Args:
        frames: Sequence of ``PIL.Image`` frames.
        output_path: Destination GIF file path.
        fps: Frames per second for playback speed.
    """
    if not frames:
        raise ValueError("At least one frame is required")

    duration = int(1000 / fps)
    first, *rest = frames
    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        duration=duration,
        loop=0,
    )
