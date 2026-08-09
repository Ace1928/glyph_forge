"""Small, runnable examples for Glyph Forge's public Python API."""

from __future__ import annotations

import numpy as np
from PIL import Image

from glyph_forge import (
    FrameRenderer,
    RenderConfig,
    image_to_glyph,
    text_to_banner,
)


def banner_example() -> str:
    """Create a text banner with a bundled style."""

    return text_to_banner("GLYPH FORGE", font="small", style="minimal")


def image_example() -> str:
    """Convert an in-memory gradient without creating a temporary file."""

    source = Image.linear_gradient("L").resize((96, 48))
    return image_to_glyph(source, width=48, auto_scale=False)


def subpixel_example() -> str:
    """Render an RGB frame with Braille's 2-by-4 subpixel cells."""

    source = Image.linear_gradient("L").resize((64, 32)).convert("RGB")
    frame = np.asarray(source, dtype=np.uint8)
    renderer = FrameRenderer(RenderConfig(width=32, mode="braille", color="none"))
    return renderer.render(frame).text


def main() -> None:
    """Print each example to the terminal."""

    for title, artwork in (
        ("Text", banner_example()),
        ("Image", image_example()),
        ("Braille subpixels", subpixel_example()),
    ):
        print(f"\n{title}\n{'-' * len(title)}")
        print(artwork)


if __name__ == "__main__":
    main()
