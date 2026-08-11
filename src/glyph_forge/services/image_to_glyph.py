import logging
import os
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Iterable, List, Optional, Tuple, TypeAlias, TypeVar, Union, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..persistence import AtomicWriteError, atomic_write_text
from ..utils.alphabet_manager import AlphabetManager
from ..visual import (
    DEFAULT_BRIGHTNESS,
    DEFAULT_CONTRAST,
    apply_tone,
    normalize_tone,
)

# Type definitions for clarity and precision
PixelArray: TypeAlias = NDArray[np.uint8]  # Type for grayscale/RGB pixel arrays
Shape = Tuple[int, ...]  # Array dimensions
T = TypeVar("T")  # Generic type for flexible functions
GlyphRow = List[str]  # Type for rows of Glyph characters
GlyphArt = List[str]  # Type for complete Glyph art (list of strings)


class ColorMode(Enum):
    """Supported color output formats."""

    ANSI = "ansi"  # Terminal-compatible ANSI color sequences
    HTML = "html"  # Web-compatible HTML color styling
    NONE = "none"  # Fallback to standard grayscale


class ImageGlyphConverter:
    """Compatibility adapter for the original stateful image API.

    New applications should use :func:`glyph_forge.render_image` with an
    immutable :class:`glyph_forge.RenderRequest`.  This class retains the 0.x
    constructor, mutators, error strings, and helper methods while delegating
    production rendering to that canonical pipeline.
    """

    def __init__(
        self,
        charset: str = "general",
        width: int = 100,
        height: Optional[int] = None,
        invert: bool = False,
        brightness: float = DEFAULT_BRIGHTNESS,
        contrast: float = DEFAULT_CONTRAST,
        auto_scale: bool = True,
        dithering: bool = False,
        threads: int = 0,
    ):
        """
        Initialize the image converter with specified settings.

        Args:
            charset: Name of character set to use or custom charset string
            width: Width of output Glyph art in characters
            height: Optional height (maintains aspect ratio if None)
            invert: Whether to invert the brightness of the output
            brightness: Brightness adjustment factor (0.0-2.0)
            contrast: Contrast adjustment factor (0.0-2.0)
            auto_scale: Automatically scale output to terminal size
            dithering: Apply dithering for improved visual quality
            threads: Number of threads for parallel processing (0=auto)
        """
        warnings.warn(
            "ImageGlyphConverter is deprecated; use RenderRequest and "
            "render_image instead",
            DeprecationWarning,
            stacklevel=2,
        )
        # Get the appropriate charset
        self._available_charsets: List[str] = AlphabetManager.list_available_alphabets()
        self.charset = (
            AlphabetManager.get_alphabet(charset)
            if charset in self._available_charsets
            else charset
        )

        # Configure core attributes with bounds checking
        self.width = max(1, width)
        self.height = max(1, height) if height is not None else None
        self.brightness = normalize_tone(brightness, name="brightness")
        self.contrast = normalize_tone(contrast, name="contrast")
        self.auto_scale = auto_scale
        self.dithering = dithering
        self.threads = threads if threads > 0 else max(1, os.cpu_count() or 1)

        # Apply inversion if needed
        if invert:
            self.charset = self.charset[::-1]

        # Generate character density mapping
        self.density_map = AlphabetManager.create_density_map(self.charset)

        # Initialize logger
        self.logger = logging.getLogger(__name__)

    def convert(
        self,
        image_path: Union[str, Image.Image],
        output_path: Optional[str] = None,
        style: Optional[str] = None,
    ) -> str:
        """
        Convert an image to Glyph art with advanced processing.

        Args:
            image_path: Path to the image file or PIL Image object
            output_path: Optional path to save the Glyph art
            style: Optional style to apply to the output

        Returns:
            Glyph art as a string
        """
        try:
            # Load image (handle both file paths and PIL Image objects)
            img = self._load_image(image_path)

            # Process the image
            Glyph_art = self._process_image(img, style)

            # Save to file if requested
            if output_path:
                self._save_to_file(Glyph_art, output_path)
                self.logger.info(f"Glyph art saved to: {output_path}")

            return Glyph_art

        except Exception as e:
            self.logger.error(f"Error converting image: {str(e)}", exc_info=True)
            return f"Error converting image: {str(e)}"

    def _load_image(self, image_path: Union[str, Image.Image]) -> Image.Image:
        """Load and prepare image for processing."""
        if isinstance(image_path, str):
            img = Image.open(image_path).convert("L")
            self.logger.info(f"Image loaded: {image_path} [{img.width}x{img.height}]")
        else:
            # Already a PIL Image
            img = image_path.convert("L")
            self.logger.info(f"Using provided PIL image [{img.width}x{img.height}]")

        return img

    def _process_image(self, img: Image.Image, style: Optional[str] = None) -> str:
        """Render through the maintained vectorized pipeline."""

        from ..contracts import RenderFormat, RenderRequest
        from ..rendering import render_image

        width, height = self._output_dimensions(img)
        request = RenderRequest(
            width=width,
            height=height,
            charset=f"literal:{self._effective_charset()}",
            brightness=self.brightness,
            contrast=self.contrast,
            dither=self.dithering,
            resample="lanczos",
            style=style,
            output_format=RenderFormat.TEXT,
            # Preserve the original adapter's historical still-image geometry.
            cell_aspect=0.55,
        )
        return render_image(img, request).glyph_text

    def _output_dimensions(self, image: Image.Image) -> tuple[int, int]:
        """Resolve legacy explicit/aspect/terminal dimensions once."""

        aspect_ratio = image.height / max(1, image.width)
        width = self.width
        height = self.height or max(1, int(aspect_ratio * width * 0.55))
        if self.auto_scale:
            width, height = self._apply_terminal_scaling(width, height)
        return max(1, width), max(1, height)

    def _effective_charset(self) -> str:
        """Honor callers that historically customized ``density_map`` directly."""

        ordered: list[str] = []
        previous: str | None = None
        for index in range(256):
            character = self.density_map.get(index)
            if character is not None and character != previous:
                ordered.append(character)
                previous = character
        return "".join(ordered) or self.charset

    def _apply_terminal_scaling(
        self, new_width: int, new_height: int
    ) -> tuple[int, int]:
        """Scale dimensions to fit the terminal window."""
        try:
            # Get terminal dimensions
            term_size = shutil.get_terminal_size()
            term_width, term_height = term_size.columns, term_size.lines

            # Apply constraints based on terminal size
            term_width = max(20, min(term_width - 2, 200))  # Practical limits
            term_height = max(10, min(term_height - 3, 100))  # Leave space for prompt

            # Don't exceed terminal width
            if new_width > term_width:
                scale_factor = term_width / new_width
                new_width = term_width
                new_height = int(new_height * scale_factor)

            # Don't exceed terminal height (with higher weight)
            if new_height > term_height:
                scale_factor = term_height / new_height
                new_height = term_height
                new_width = int(new_width * scale_factor)

            self.logger.debug(f"Terminal-scaled dimensions: {new_width}x{new_height}")
            return new_width, new_height
        except Exception as e:
            self.logger.warning(f"Failed to apply terminal scaling: {e}")
            return new_width, new_height

    def _apply_image_adjustments(self, img: Image.Image) -> Image.Image:
        """Apply brightness and contrast adjustments to the image."""
        return Image.fromarray(
            apply_tone(np.asarray(img), self.brightness, self.contrast)
        )

    def _convert_pixels(self, pixels: PixelArray) -> str:
        """
        Convert pixel array to Glyph art (single-threaded implementation).

        Args:
            pixels: Numpy array of grayscale pixel values

        Returns:
            Glyph art string
        """
        Glyph_art: GlyphArt = []
        for row in cast(Iterable[NDArray[np.uint8]], pixels):
            Glyph_row = "".join(
                self.density_map[int(pixel_value)] for pixel_value in row
            )
            Glyph_art.append(Glyph_row)

        return "\n".join(Glyph_art)

    def _parallel_conversion(self, pixels: PixelArray) -> str:
        """
        Convert pixel array to Glyph art using parallel processing.

        Args:
            pixels: Numpy array of grayscale pixel values

        Returns:
            Glyph art string
        """
        chunk_size = max(1, len(pixels) // self.threads)
        chunks = [pixels[i : i + chunk_size] for i in range(0, len(pixels), chunk_size)]

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            results = list(executor.map(self._convert_pixels, chunks))

        return "\n".join(results)

    def _save_to_file(self, Glyph_art: str, output_path: str) -> None:
        """Atomically save legacy Glyph art with proper directory creation."""
        try:
            atomic_write_text(output_path, Glyph_art)
            self.logger.debug(f"Saved output to {output_path}")
        except AtomicWriteError as e:
            self.logger.error(f"Failed to save output: {e}")
            raise OSError(f"Failed to save output: {str(e)}") from e

    def set_charset(self, charset: str, invert: bool = False) -> None:
        """
        Change the character set used for conversion.

        Args:
            charset: Name of preset charset or custom string
            invert: Whether to invert the brightness
        """
        self.charset = (
            AlphabetManager.get_alphabet(charset)
            if charset in self._available_charsets
            else charset
        )
        if invert:
            self.charset = self.charset[::-1]

        self.density_map = AlphabetManager.create_density_map(self.charset)

    def set_image_params(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None,
        dithering: Optional[bool] = None,
    ) -> None:
        """
        Update image conversion parameters.

        Args:
            width: New width in characters
            height: New height in characters
            brightness: New brightness adjustment factor
            contrast: New contrast adjustment factor
            dithering: Enable/disable dithering
        """
        if width is not None:
            self.width = max(1, width)

        if height is not None:
            self.height = max(1, height) if height > 0 else None

        if brightness is not None:
            self.brightness = normalize_tone(brightness, name="brightness")

        if contrast is not None:
            self.contrast = normalize_tone(contrast, name="contrast")

        if dithering is not None:
            self.dithering = dithering

    def get_available_charsets(self) -> List[str]:
        """
        Get list of available character sets.

        Returns:
            List of available charset names
        """
        return self._available_charsets.copy()

    def convert_color(
        self,
        image_path: Union[str, Image.Image],
        output_path: Optional[str] = None,
        color_mode: Union[str, ColorMode] = "ansi",
    ) -> str:
        """
        Convert image to color Glyph art using ANSI or HTML color codes.

        Args:
            image_path: Path to image or PIL Image object
            output_path: Optional path to save the output
            color_mode: Color output format ("ansi", "html", or "none")

        Returns:
            Glyph art with color formatting
        """
        mode = color_mode.value if isinstance(color_mode, ColorMode) else color_mode
        mode = mode.casefold()
        if mode not in {"ansi", "html"}:
            return self.convert(image_path, output_path)
        try:
            from ..contracts import RenderFormat, RenderRequest
            from ..rendering import load_image, render_image

            probe_request = RenderRequest()
            image = load_image(image_path, probe_request)
            width, height = self._output_dimensions(image)
            request = RenderRequest(
                width=width,
                height=height,
                charset=f"literal:{self._effective_charset()}",
                brightness=self.brightness,
                contrast=self.contrast,
                dither=self.dithering,
                resample="lanczos",
                output_format=(
                    RenderFormat.TRUECOLOR if mode == "ansi" else RenderFormat.HTML
                ),
                cell_aspect=0.55,
            )
            artifact = render_image(image, request, destination=output_path)
            assert isinstance(artifact.data, str)
            return artifact.data
        except Exception as exc:
            self.logger.error("Color conversion error: %s", exc, exc_info=True)
            return f"Error converting color image: {exc}"

    def _generate_ansi_color(
        self, pixels_rgb: PixelArray, pixels_gray: PixelArray
    ) -> str:
        """Generate Glyph art with ANSI color codes."""
        Glyph_art: GlyphArt = []
        for y in range(len(pixels_gray)):
            row: GlyphRow = []
            for x in range(len(pixels_gray[y])):
                # Get character based on brightness
                char = self.density_map[int(pixels_gray[y][x])]
                # Get RGB color
                r, g, b = pixels_rgb[y][x]
                # Create ANSI color sequence
                color_code = f"\033[38;2;{r};{g};{b}m{char}\033[0m"
                row.append(color_code)
            Glyph_art.append("".join(row))
        return "\n".join(Glyph_art)

    def _generate_html_color(
        self, pixels_rgb: PixelArray, pixels_gray: PixelArray
    ) -> str:
        """Generate Glyph art with HTML color tags."""
        Glyph_art: GlyphArt = ["<pre style='line-height:1; letter-spacing:0'>"]
        for y in range(len(pixels_gray)):
            row_parts: List[str] = []
            for x in range(len(pixels_gray[y])):
                # Get character based on brightness
                char = self.density_map[int(pixels_gray[y][x])]
                # Get RGB color
                r, g, b = pixels_rgb[y][x]
                # Create HTML span with color
                color_hex = f"#{r:02x}{g:02x}{b:02x}"
                row_parts.append(f"<span style='color:{color_hex}'>{char}</span>")

            # Join row and add line break
            Glyph_art.append("".join(row_parts))
            Glyph_art.append("<br>")

        # Close container
        Glyph_art.append("</pre>")

        return "".join(Glyph_art)


def image_to_glyph(
    image_path: Union[str, Image.Image],
    output_path: Optional[str] = None,
    style: Optional[str] = None,
    color_mode: str = "none",
    **kwargs: Any,
) -> str:
    """High-level helper for quick image conversion.

    This stateless compatibility wrapper translates the historical keyword
    arguments into a :class:`glyph_forge.RenderRequest` and uses the canonical
    renderer directly. It supports both grayscale and color output modes.

    Args:
        image_path: Path to an image file or ``PIL.Image`` object.
        output_path: Optional destination to save the resulting glyph art.
        style: Optional style name applied to plain-text output.
        color_mode: ``"none"`` for grayscale, ``"ansi"`` or ``"html"`` for color.
        **kwargs: Historical width, height, charset, tone, and fitting options.

    Returns:
        Glyph art string.
    """

    from ..contracts import GlyphForgeRenderError, RenderRequest
    from ..rendering import format_for_path, render_image

    width = max(1, int(kwargs.get("width", 100)))
    height_value = kwargs.get("height")
    height = max(1, int(height_value)) if height_value is not None else None
    max_width: int | None = None
    max_height: int | None = None
    if bool(kwargs.get("auto_scale", True)):
        terminal = shutil.get_terminal_size(fallback=(80, 24))
        max_width = max(1, terminal.columns - 2)
        max_height = max(1, terminal.lines - 3)
    try:
        output_format = format_for_path(output_path, color=color_mode)
        requested_charset = str(kwargs.get("charset", "general"))
        if requested_charset not in AlphabetManager.list_available_alphabets():
            requested_charset = f"literal:{requested_charset}"
        request = RenderRequest(
            width=width,
            height=height,
            charset=requested_charset,
            invert=bool(kwargs.get("invert", False)),
            brightness=float(kwargs.get("brightness", DEFAULT_BRIGHTNESS)),
            contrast=float(kwargs.get("contrast", DEFAULT_CONTRAST)),
            dither=bool(kwargs.get("dithering", False)),
            output_format=output_format,
            style=style if output_format.value == "text" else None,
            max_width=max_width,
            max_height=max_height,
            resample="lanczos",
            cell_aspect=0.55,
        )
        artifact = render_image(image_path, request, destination=output_path)
        return artifact.data if isinstance(artifact.data, str) else artifact.glyph_text
    except GlyphForgeRenderError as exc:
        return f"Error converting image: {exc}"
