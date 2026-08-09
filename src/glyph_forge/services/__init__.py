"""Service layer: high-level workflow helpers.

The historical ``image_to_Glyph`` module name is retained as an in-memory
alias.  Keeping a second file whose name differs only by case breaks on the
case-insensitive filesystems commonly used by Windows and macOS.
"""

from __future__ import annotations

import sys

from . import image_to_glyph as image_to_Glyph
from .image_to_glyph import ImageGlyphConverter
from .text_to_banner import text_to_banner
from .text_to_glyph import text_to_glyph
from .video_to_glyph import iter_video_glyph_frames, video_to_glyph_frames
from .video_to_images import iter_video_images, video_to_images

sys.modules.setdefault(f"{__name__}.image_to_Glyph", image_to_Glyph)

__all__ = [
    "ImageGlyphConverter",
    "iter_video_glyph_frames",
    "iter_video_images",
    "text_to_banner",
    "text_to_glyph",
    "video_to_glyph_frames",
    "video_to_images",
]
