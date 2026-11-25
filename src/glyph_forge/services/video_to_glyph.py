"""Video to glyph conversion service."""

from typing import Iterable, List, Optional

from PIL import Image

from .image_to_glyph import ImageGlyphConverter
from .video_to_images import video_to_images


def video_to_glyph_frames(
    video_path: str,
    width: int = 80,
    max_frames: Optional[int] = None,
    color_mode: str = "none",
) -> List[str]:
    """Convert a video or GIF into a list of glyph-art frames.

    Args:
        video_path: Path to the input video or GIF file.
        width: Target width of each glyph frame.
        max_frames: Optional limit on number of frames to convert.
        color_mode: ``"none"``, ``"ansi"``, ``"ansi16"``, ``"ansi256"``,
            ``"truecolor"``, or ``"html"`` for color output.

    Returns:
        List of glyph-art frames as strings.
    """
    converter = ImageGlyphConverter(width=width)
    images: Iterable[Image.Image] = video_to_images(video_path, max_frames=max_frames)

    frames: List[str] = []
    for img in images:
        mode = color_mode.lower()
        if mode == "rgb":
            mode = "truecolor"
        if mode == "web":
            mode = "html"
        if mode in ("ansi", "ansi16", "ansi256", "truecolor", "html"):
            frames.append(converter.convert_color(img, color_mode=mode))
        else:
            frames.append(converter.convert(img))

    return frames
