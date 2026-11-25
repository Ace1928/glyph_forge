import html
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple, TypeAlias, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter, ImageOps

from ..exceptions import GlyphForgeInputError, GlyphForgeRenderError
from ..utils.alphabet_manager import AlphabetManager

# Type definitions for clarity and precision
PixelArray: TypeAlias = NDArray[np.uint8]  # Type for grayscale/RGB pixel arrays
ImageSource: TypeAlias = Union[str, os.PathLike[str], Image.Image]
Shape = Tuple[int, ...]  # Array dimensions
T = TypeVar("T")  # Generic type for flexible functions
GlyphRow = List[str]  # Type for rows of Glyph characters
GlyphArt = List[str]  # Type for complete Glyph art (list of strings)


class ColorMode(Enum):
    """Supported color output formats."""

    ANSI = "ansi"  # Terminal-compatible ANSI color sequences (truecolor default)
    ANSI16 = "ansi16"  # 16-color ANSI output
    ANSI256 = "ansi256"  # 256-color ANSI output
    TRUECOLOR = "truecolor"  # 24-bit ANSI output
    HTML = "html"  # Web-compatible HTML color styling
    NONE = "none"  # Fallback to standard grayscale


class DitherAlgorithm(Enum):
    """Supported dithering algorithms."""

    NONE = "none"
    FLOYD_STEINBERG = "floyd-steinberg"
    ATKINSON = "atkinson"


class ImageGlyphConverter:
    """
    # ImageGlyphConverter
    A high-performance image-to-Glyph art converter that transforms visual data into textual representations with precision and flexibility.
    ## Overview
    `ImageGlyphConverter` provides comprehensive functionality to convert images into Glyph art using various character sets, processing techniques, and output formats. The converter supports grayscale and color output, with options for adjusting dimensions, brightness, contrast, and applying effects like dithering.
    ## Features
    - **Multiple character sets** - Use built-in or custom character sets for different artistic styles
    - **Adaptive rendering** - Maintains aspect ratio and can auto-scale to terminal dimensions
    - **Multi-threaded processing** - Parallel conversion for large images with configurable thread count
    - **Image adjustments** - Controls for brightness, contrast, and dithering
    - **Color output** - Support for ANSI (terminal) and HTML color formats
    - **Styling options** - Apply visual styles to the resulting Glyph art
    The implementation combines adaptive sizing, vectorized lookup tables,
    optional row-level concurrency, and specialized character sets.
    """

    def __init__(
        self,
        charset: str = "general",
        width: int = 100,
        height: Optional[int] = None,
        invert: bool = False,
        brightness: float = 1.0,
        contrast: float = 1.0,
        auto_scale: bool = True,
        dithering: bool = False,
        dither_algorithm: Optional[Union[str, DitherAlgorithm]] = None,
        aspect_ratio: float = 0.55,
        resample: str = "lanczos",
        gamma: float = 1.0,
        autocontrast: bool = False,
        equalize: bool = False,
        invert_image: bool = False,
        edge_enhance: bool = False,
        sharpen: bool = False,
        blur_radius: float = 0.0,
        posterize_bits: Optional[int] = None,
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
            gamma: Gamma correction factor (0.1-5.0)
            auto_scale: Automatically scale output to terminal size
            dithering: Apply dithering for improved visual quality
            dither_algorithm: Dithering algorithm ("none", "floyd-steinberg", "atkinson")
            aspect_ratio: Character aspect ratio correction factor
            resample: Resampling filter (nearest, bilinear, bicubic, lanczos)
            autocontrast: Auto-stretch contrast for better detail
            equalize: Histogram equalization for enhanced dynamic range
            invert_image: Invert image luminance before conversion
            edge_enhance: Emphasize edges for structural clarity
            sharpen: Apply sharpening filter for crisper details
            blur_radius: Gaussian blur radius for smoothing
            posterize_bits: Reduce color depth (1-8) for stylized output
            threads: Number of threads for parallel processing (0=auto)
        """
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
        self.brightness = max(0.0, min(2.0, brightness))
        self.contrast = max(0.0, min(2.0, contrast))
        self.gamma = max(0.1, min(5.0, gamma))
        self.auto_scale = auto_scale
        self.dither_algorithm = self._normalize_dither_algorithm(dither_algorithm, dithering)
        self.dithering = self.dither_algorithm != DitherAlgorithm.NONE
        self.aspect_ratio = max(0.1, min(2.0, aspect_ratio))
        self.resample = resample.lower()
        self.resample_filter = self._get_resample_filter(self.resample)
        self.autocontrast = autocontrast
        self.equalize = equalize
        self.invert_image = invert_image
        self.edge_enhance = edge_enhance
        self.sharpen = sharpen
        self.blur_radius = max(0.0, blur_radius)
        self.posterize_bits = self._normalize_posterize_bits(posterize_bits)
        self.threads = threads if threads > 0 else max(1, os.cpu_count() or 1)

        # Apply inversion if needed
        if invert:
            self.charset = self.charset[::-1]

        # Generate character density mapping
        self.density_map = AlphabetManager.create_density_map(self.charset)
        self._density_lut = self._build_density_lut()
        self._density_lut_html = self._build_density_lut_html()
        self._ansi_reset = "\033[0m"
        self._ansi256_prefixes = [f"\033[38;5;{i}m" for i in range(256)]
        self._ansi16_prefixes = [""] * 128
        for code in list(range(30, 38)) + list(range(90, 98)):
            self._ansi16_prefixes[code] = f"\033[{code}m"

        # Initialize logger
        self.logger = logging.getLogger(__name__)

    def _build_density_lut(self) -> NDArray[Any]:
        """Build a lookup table for fast grayscale-to-char mapping."""
        return np.array([self.density_map[i] for i in range(256)], dtype=object)

    def _build_density_lut_html(self) -> NDArray[Any]:
        """Build a lookup table for fast HTML-escaped char mapping."""
        return np.array([html.escape(self.density_map[i]) for i in range(256)], dtype=object)

    def convert(
        self,
        image_path: ImageSource,
        output_path: Optional[str] = None,
        style: Optional[str] = None,
        write_metadata: bool = True,
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
            src_w, src_h = img.size
            src_mode = img.mode

            # Process the image
            Glyph_art = self._process_image(img, style)

            resolved_output = self._resolve_output_path(output_path, image_path, ext="txt")
            if resolved_output:
                self._save_to_file(Glyph_art, resolved_output)
                if write_metadata:
                    self._write_metadata(
                        resolved_output,
                        image_path,
                        extra={
                            "format": "txt",
                            "style": style,
                            "color_mode": "none",
                            "source_width": src_w,
                            "source_height": src_h,
                            "source_mode": src_mode,
                        },
                    )
                self.logger.info(f"Glyph art saved to: {resolved_output}")

            return Glyph_art

        except (OSError, TypeError, ValueError) as exc:
            self.logger.debug("Image conversion failed", exc_info=True)
            raise GlyphForgeRenderError(f"could not convert image: {exc}") from exc

    def _load_image(self, image_path: ImageSource) -> Image.Image:
        """Load and prepare image for processing."""
        if isinstance(image_path, (str, os.PathLike)):
            img = Image.open(image_path)
            self.logger.info(f"Image loaded: {image_path} [{img.width}x{img.height}]")
        else:
            # Already a PIL Image
            img = image_path
            self.logger.info(f"Using provided PIL image [{img.width}x{img.height}]")

        return img.convert("L")

    def _process_image(self, img: Image.Image, style: Optional[str] = None) -> str:
        """Process image through the complete conversion pipeline."""
        # Calculate new dimensions
        orig_width, orig_height = img.size
        aspect_ratio = orig_height / orig_width

        # Set output dimensions, maintaining aspect ratio
        new_width = self.width
        # Character aspect ratio correction factor (chars are taller than wide)
        char_aspect = self.aspect_ratio
        new_height = self.height if self.height else int(aspect_ratio * new_width * char_aspect)

        # Auto-scale to terminal size if requested
        if self.auto_scale:
            new_width, new_height = self._apply_terminal_scaling(new_width, new_height)

        # Resize image with configured resampling
        img = img.resize((new_width, new_height), self.resample_filter)

        # Apply filters that operate on PIL images
        img = self._apply_pil_filters(img)

        # Apply brightness/contrast adjustments if needed
        if self.brightness != 1.0 or self.contrast != 1.0 or self.gamma != 1.0:
            img = self._apply_image_adjustments(img)

        # Apply dithering if enabled
        if self.dithering:
            img = self._apply_dithering(img)

        # Apply posterize after grayscale transforms
        if self.posterize_bits is not None:
            img = ImageOps.posterize(img, self.posterize_bits)

        # Convert to numpy array for faster processing
        pixels = np.array(img)

        # Generate Glyph art (with parallel processing for large images)
        if new_height > 100 and self.threads > 1:
            # Process rows in parallel
            Glyph_art = self._parallel_conversion(pixels)
        else:
            # Single-threaded processing
            Glyph_art = self._convert_pixels(pixels)

        # Apply style if requested
        if style:
            from ..core.style_manager import apply_style

            Glyph_art = apply_style(Glyph_art, style_name=style)

        return Glyph_art

    def _apply_terminal_scaling(self, new_width: int, new_height: int) -> tuple[int, int]:
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
        """Apply brightness, contrast, and gamma adjustments to the image."""
        pixels = np.array(img)
        adjusted = self._apply_pixel_adjustments(pixels)
        return Image.fromarray(adjusted)

    def _apply_pixel_adjustments(self, pixels: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Apply brightness, contrast, and gamma corrections to pixel data."""
        values = pixels.astype(np.float32)

        if self.contrast != 1.0:
            values = (values - 128.0) * self.contrast + 128.0

        if self.brightness != 1.0:
            values = values * self.brightness

        if self.gamma != 1.0:
            values = np.clip(values, 0, 255)
            values = 255.0 * np.power(values / 255.0, 1.0 / self.gamma)

        return np.clip(values, 0, 255).astype(np.uint8)

    def _apply_pil_filters(self, img: Image.Image) -> Image.Image:
        """Apply PIL-based filters for enhanced visual quality."""
        if self.autocontrast:
            img = ImageOps.autocontrast(img)
        if self.equalize:
            img = ImageOps.equalize(img)
        if self.invert_image:
            img = ImageOps.invert(img)
        if self.edge_enhance:
            img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        if self.sharpen:
            img = img.filter(ImageFilter.SHARPEN)
        if self.blur_radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        return img

    def _apply_dithering(self, img: Image.Image) -> Image.Image:
        """Apply configured dithering algorithm."""
        if self.dither_algorithm == DitherAlgorithm.FLOYD_STEINBERG:
            img = img.convert("1", dither=Image.Dither.FLOYDSTEINBERG)
            return img.convert("L")
        if self.dither_algorithm == DitherAlgorithm.ATKINSON:
            pixels = np.array(img.convert("L"))
            dithered = self._apply_atkinson_dither(pixels)
            return Image.fromarray(dithered)
        return img

    def _apply_atkinson_dither(self, pixels: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Apply Atkinson dithering to a grayscale pixel array."""
        array = pixels.astype(np.float32)
        height, width = array.shape
        for y in range(height):
            for x in range(width):
                old = array[y, x]
                new = 255.0 if old >= 128 else 0.0
                error = (old - new) / 8.0
                array[y, x] = new
                if x + 1 < width:
                    array[y, x + 1] += error
                if x + 2 < width:
                    array[y, x + 2] += error
                if y + 1 < height:
                    if x - 1 >= 0:
                        array[y + 1, x - 1] += error
                    array[y + 1, x] += error
                    if x + 1 < width:
                        array[y + 1, x + 1] += error
                if y + 2 < height:
                    array[y + 2, x] += error
        return np.clip(array, 0, 255).astype(np.uint8)

    def _convert_pixels(self, pixels: PixelArray) -> str:
        """
        Convert pixel array to Glyph art (single-threaded implementation).

        Args:
            pixels: Numpy array of grayscale pixel values

        Returns:
            Glyph art string
        """
        if pixels.size == 0:
            return ""
        chars = self._density_lut[pixels]
        return "\n".join("".join(row.tolist()) for row in chars)

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
        """Save Glyph art to a file with proper directory creation."""
        try:
            # Ensure directory exists
            dirname = os.path.dirname(output_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)

            # Write with UTF-8 encoding for maximum compatibility
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(Glyph_art)

            self.logger.debug(f"Saved output to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save output: {e}")
            raise IOError(f"Failed to save output: {str(e)}")

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
        self._density_lut = self._build_density_lut()
        self._density_lut_html = self._build_density_lut_html()

    def set_image_params(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None,
        gamma: Optional[float] = None,
        dithering: Optional[bool] = None,
        dither_algorithm: Optional[Union[str, DitherAlgorithm]] = None,
        aspect_ratio: Optional[float] = None,
        resample: Optional[str] = None,
        autocontrast: Optional[bool] = None,
        equalize: Optional[bool] = None,
        invert_image: Optional[bool] = None,
        edge_enhance: Optional[bool] = None,
        sharpen: Optional[bool] = None,
        blur_radius: Optional[float] = None,
        posterize_bits: Optional[int] = None,
    ) -> None:
        """
        Update image conversion parameters.

        Args:
            width: New width in characters
            height: New height in characters
            brightness: New brightness adjustment factor
            contrast: New contrast adjustment factor
            gamma: New gamma correction factor
            dithering: Enable/disable dithering
            dither_algorithm: Dithering algorithm selection
            aspect_ratio: Character aspect ratio correction factor
            resample: Resampling filter (nearest, bilinear, bicubic, lanczos)
            autocontrast: Enable/disable autocontrast
            equalize: Enable/disable histogram equalization
            invert_image: Enable/disable luminance inversion
            edge_enhance: Enable/disable edge enhancement
            sharpen: Enable/disable sharpening
            blur_radius: Gaussian blur radius
            posterize_bits: Posterize bit depth (1-8)
        """
        if width is not None:
            self.width = max(1, width)

        if height is not None:
            self.height = max(1, height) if height > 0 else None

        if brightness is not None:
            self.brightness = max(0.0, min(2.0, brightness))

        if contrast is not None:
            self.contrast = max(0.0, min(2.0, contrast))

        if gamma is not None:
            self.gamma = max(0.1, min(5.0, gamma))

        if dithering is not None:
            self.dithering = dithering

        if dither_algorithm is not None:
            self.dither_algorithm = self._normalize_dither_algorithm(
                dither_algorithm, self.dithering
            )
            self.dithering = self.dither_algorithm != DitherAlgorithm.NONE

        if aspect_ratio is not None:
            self.aspect_ratio = max(0.1, min(2.0, aspect_ratio))

        if resample is not None:
            self.resample = resample.lower()
            self.resample_filter = self._get_resample_filter(self.resample)

        if autocontrast is not None:
            self.autocontrast = autocontrast

        if equalize is not None:
            self.equalize = equalize

        if invert_image is not None:
            self.invert_image = invert_image

        if edge_enhance is not None:
            self.edge_enhance = edge_enhance

        if sharpen is not None:
            self.sharpen = sharpen

        if blur_radius is not None:
            self.blur_radius = max(0.0, blur_radius)

        if posterize_bits is not None:
            self.posterize_bits = self._normalize_posterize_bits(posterize_bits)

    def get_available_charsets(self) -> List[str]:
        """
        Get list of available character sets.

        Returns:
            List of available charset names
        """
        return self._available_charsets.copy()

    def get_supported_color_modes(self) -> List[str]:
        """Get list of supported color modes."""
        return [mode.value for mode in ColorMode]

    def get_supported_dither_algorithms(self) -> List[str]:
        """Get list of supported dithering algorithms."""
        return [alg.value for alg in DitherAlgorithm]

    def get_supported_resample_filters(self) -> List[str]:
        """Get list of supported resampling filters."""
        return ["nearest", "bilinear", "bicubic", "lanczos"]

    def convert_color(
        self,
        image_path: ImageSource,
        output_path: Optional[str] = None,
        color_mode: Union[str, ColorMode] = "ansi",
        write_metadata: bool = True,
    ) -> str:
        """
        Convert image to color Glyph art using ANSI or HTML color codes.

        Args:
            image_path: Path to image or PIL Image object
            output_path: Optional path to save the output
            color_mode: Color output format ("ansi", "ansi16", "ansi256",
                "truecolor", "html", or "none")

        Returns:
            Glyph art with color formatting
        """
        try:
            # Load image
            if isinstance(image_path, (str, os.PathLike)):
                img = Image.open(image_path)
            elif isinstance(image_path, Image.Image):
                img = image_path
            else:
                raise GlyphForgeInputError(
                    "image_path must be a filesystem path or PIL Image object"
                )
            src_w, src_h = img.size
            src_mode = img.mode

            # Calculate dimensions
            orig_width, orig_height = img.size
            aspect_ratio = orig_height / orig_width

            # Set output dimensions
            new_width = self.width
            char_aspect = self.aspect_ratio
            new_height = self.height if self.height else int(aspect_ratio * new_width * char_aspect)

            # Auto-scale if requested
            if self.auto_scale:
                new_width, new_height = self._apply_terminal_scaling(new_width, new_height)

            # Resize image
            img = img.resize((new_width, new_height), self.resample_filter)

            # Convert to grayscale for character selection
            gray_img = img.convert("L")

            gray_img = self._apply_pil_filters(gray_img)
            if self.brightness != 1.0 or self.contrast != 1.0 or self.gamma != 1.0:
                gray_img = self._apply_image_adjustments(gray_img)
            if self.dithering:
                gray_img = self._apply_dithering(gray_img)
            if self.posterize_bits is not None:
                gray_img = ImageOps.posterize(gray_img, self.posterize_bits)

            # Get both color and grayscale pixels
            img = self._apply_pil_filters(img.convert("RGB"))
            if self.brightness != 1.0 or self.contrast != 1.0 or self.gamma != 1.0:
                rgb_adjusted = self._apply_pixel_adjustments(np.array(img))
                img = Image.fromarray(rgb_adjusted, mode="RGB")
            if self.posterize_bits is not None:
                img = ImageOps.posterize(img, self.posterize_bits)
            pixels_rgb = np.array(img)
            pixels_gray = np.array(gray_img)

            # Normalize color mode input
            mode = color_mode.value if isinstance(color_mode, ColorMode) else color_mode
            mode = mode.lower()
            if mode == "rgb":
                mode = "truecolor"
            if mode == "web":
                mode = "html"

            # Generate color Glyph art based on mode
            if mode in {"ansi", "truecolor"}:
                Glyph_art = self._generate_ansi_truecolor(pixels_rgb, pixels_gray)
            elif mode == "ansi16":
                Glyph_art = self._generate_ansi16_color(pixels_rgb, pixels_gray)
            elif mode == "ansi256":
                Glyph_art = self._generate_ansi256_color(pixels_rgb, pixels_gray)
            elif mode == "html":
                Glyph_art = self._generate_html_color(pixels_rgb, pixels_gray)
            elif mode == "none":
                return self.convert(image_path, output_path, write_metadata=write_metadata)
            else:
                choices = ", ".join(item.value for item in ColorMode)
                raise GlyphForgeInputError(
                    f"unsupported color mode {mode!r}; choose one of: {choices}"
                )

            resolved_output = self._resolve_output_path(
                output_path,
                image_path,
                ext="html" if mode == "html" else "txt",
            )
            if resolved_output:
                self._save_to_file(Glyph_art, resolved_output)
                if write_metadata:
                    self._write_metadata(
                        resolved_output,
                        image_path,
                        extra={
                            "format": "html" if mode == "html" else "txt",
                            "color_mode": mode,
                            "source_width": src_w,
                            "source_height": src_h,
                            "source_mode": src_mode,
                        },
                    )

            return Glyph_art

        except GlyphForgeInputError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            self.logger.debug("Color image conversion failed", exc_info=True)
            raise GlyphForgeRenderError(f"could not convert color image: {exc}") from exc

    def _resolve_source_reference(self, image_path: ImageSource) -> str:
        if isinstance(image_path, (str, os.PathLike)):
            return os.fspath(image_path)
        filename = getattr(image_path, "filename", None)
        if filename:
            return str(filename)
        return "glyph_image"

    def _resolve_output_path(
        self,
        output_path: Optional[str],
        image_path: ImageSource,
        ext: str,
    ) -> Optional[str]:
        if not output_path:
            return None
        out_path = Path(output_path)
        if out_path.suffix or (out_path.exists() and out_path.is_file()):
            return str(out_path)
        source_ref = self._resolve_source_reference(image_path)
        try:
            from glyph_forge.streaming.naming import build_output_path
        except Exception:
            return str(out_path / f"glyph_forge.{ext}")
        resolved = build_output_path(
            source=source_ref,
            title=None,
            output_dir=out_path,
            ext=ext,
            output=None,
            overwrite=False,
        )
        return str(resolved)

    def _write_metadata(
        self,
        output_path: str,
        image_path: Union[str, Image.Image],
        extra: Optional[dict] = None,
    ) -> None:
        try:
            from glyph_forge.streaming.naming import build_metadata, write_metadata
        except Exception:
            return
        source_ref = self._resolve_source_reference(image_path)
        meta = build_metadata(
            source=source_ref,
            output_path=Path(output_path),
            title=None,
            info=None,
            extra=extra or {},
        )
        write_metadata(meta, Path(output_path))

    def _generate_ansi_truecolor(self, pixels_rgb: PixelArray, pixels_gray: PixelArray) -> str:
        """Generate Glyph art with truecolor ANSI codes."""
        Glyph_art: GlyphArt = []
        chars = self._density_lut[pixels_gray.astype(np.uint8)]
        height = pixels_gray.shape[0]
        prefix_cache: dict[tuple[int, int, int], str] = {}
        for y in range(height):
            row_chars = chars[y]
            row_rgb = pixels_rgb[y]
            row_parts: List[str] = []
            current_color: Optional[tuple[int, int, int]] = None
            buffer: List[str] = []
            for (r, g, b), ch in zip(row_rgb, row_chars, strict=True):
                color = (int(r), int(g), int(b))
                if current_color is None:
                    current_color = color
                if color != current_color:
                    prefix = prefix_cache.get(current_color)
                    if prefix is None:
                        prefix = (
                            f"\033[38;2;{current_color[0]};{current_color[1]};{current_color[2]}m"
                        )
                        prefix_cache[current_color] = prefix
                    row_parts.append(prefix + "".join(buffer) + self._ansi_reset)
                    buffer = []
                    current_color = color
                buffer.append(ch)
            if buffer and current_color is not None:
                prefix = prefix_cache.get(current_color)
                if prefix is None:
                    prefix = f"\033[38;2;{current_color[0]};{current_color[1]};{current_color[2]}m"
                    prefix_cache[current_color] = prefix
                row_parts.append(prefix + "".join(buffer) + self._ansi_reset)
            Glyph_art.append("".join(row_parts))
        return "\n".join(Glyph_art)

    def _generate_ansi16_color(self, pixels_rgb: PixelArray, pixels_gray: PixelArray) -> str:
        """Generate Glyph art with 16-color ANSI codes."""
        Glyph_art: GlyphArt = []
        chars = self._density_lut[pixels_gray.astype(np.uint8)]
        r = pixels_rgb[..., 0].astype(np.int16)
        g = pixels_rgb[..., 1].astype(np.int16)
        b = pixels_rgb[..., 2].astype(np.int16)
        brightness = (r + g + b) / 3.0
        bright_mask = brightness > 127
        index = (
            (r >= 128).astype(np.int16)
            + (g >= 128).astype(np.int16) * 2
            + (b >= 128).astype(np.int16) * 4
        )
        base = np.where(bright_mask, 90, 30).astype(np.int16)
        codes = base + index

        height = pixels_gray.shape[0]
        for y in range(height):
            row_chars = chars[y]
            row_codes = codes[y]
            row_parts: List[str] = []
            current_code: Optional[int] = None
            buffer: List[str] = []
            for code, ch in zip(row_codes, row_chars, strict=True):
                code_int = int(code)
                if current_code is None:
                    current_code = code_int
                if code_int != current_code:
                    row_parts.append(
                        f"{self._ansi16_prefixes[current_code]}{''.join(buffer)}{self._ansi_reset}"
                    )
                    buffer = []
                    current_code = code_int
                buffer.append(ch)
            if buffer and current_code is not None:
                row_parts.append(
                    f"{self._ansi16_prefixes[current_code]}{''.join(buffer)}{self._ansi_reset}"
                )
            Glyph_art.append("".join(row_parts))
        return "\n".join(Glyph_art)

    def _generate_ansi256_color(self, pixels_rgb: PixelArray, pixels_gray: PixelArray) -> str:
        """Generate Glyph art with 256-color ANSI codes."""
        Glyph_art: GlyphArt = []
        chars = self._density_lut[pixels_gray.astype(np.uint8)]
        r = pixels_rgb[..., 0].astype(np.int16)
        g = pixels_rgb[..., 1].astype(np.int16)
        b = pixels_rgb[..., 2].astype(np.int16)

        gray_mask = (r == g) & (g == b)
        codes = np.empty_like(r, dtype=np.int16)

        gray = r
        gray_codes = np.empty_like(gray, dtype=np.int16)
        gray_codes[gray < 8] = 16
        gray_codes[gray > 248] = 231
        mid_mask = (gray >= 8) & (gray <= 248)
        if np.any(mid_mask):
            scaled = np.rint(((gray[mid_mask] - 8) / 247.0) * 24).astype(np.int16)
            gray_codes[mid_mask] = scaled + 232
        codes[gray_mask] = gray_codes[gray_mask]

        r_val = np.rint(r / 255.0 * 5).astype(np.int16)
        g_val = np.rint(g / 255.0 * 5).astype(np.int16)
        b_val = np.rint(b / 255.0 * 5).astype(np.int16)
        cube_codes = 16 + 36 * r_val + 6 * g_val + b_val
        codes[~gray_mask] = cube_codes[~gray_mask]

        height = pixels_gray.shape[0]
        for y in range(height):
            row_chars = chars[y]
            row_codes = codes[y]
            row_parts: List[str] = []
            current_code: Optional[int] = None
            buffer: List[str] = []
            for code, ch in zip(row_codes, row_chars, strict=True):
                code_int = int(code)
                if current_code is None:
                    current_code = code_int
                if code_int != current_code:
                    row_parts.append(
                        f"{self._ansi256_prefixes[current_code]}{''.join(buffer)}{self._ansi_reset}"
                    )
                    buffer = []
                    current_code = code_int
                buffer.append(ch)
            if buffer and current_code is not None:
                row_parts.append(
                    f"{self._ansi256_prefixes[current_code]}{''.join(buffer)}{self._ansi_reset}"
                )
            Glyph_art.append("".join(row_parts))
        return "\n".join(Glyph_art)

    def _generate_html_color(self, pixels_rgb: PixelArray, pixels_gray: PixelArray) -> str:
        """Generate Glyph art with HTML color tags."""
        Glyph_art: GlyphArt = ["<pre style='line-height:1; letter-spacing:0'>"]
        chars = self._density_lut_html[pixels_gray.astype(np.uint8)]
        height = pixels_gray.shape[0]
        for y in range(height):
            row_chars = chars[y]
            row_rgb = pixels_rgb[y]
            row_parts: List[str] = []
            current_color: Optional[tuple[int, int, int]] = None
            buffer: List[str] = []
            for (r, g, b), ch in zip(row_rgb, row_chars, strict=True):
                color = (int(r), int(g), int(b))
                if current_color is None:
                    current_color = color
                if color != current_color:
                    color_hex = (
                        f"#{current_color[0]:02x}{current_color[1]:02x}{current_color[2]:02x}"
                    )
                    row_parts.append(
                        f"<span style='color:{color_hex}'>" + "".join(buffer) + "</span>"
                    )
                    buffer = []
                    current_color = color
                buffer.append(ch)
            if buffer and current_color is not None:
                color_hex = f"#{current_color[0]:02x}{current_color[1]:02x}{current_color[2]:02x}"
                row_parts.append(f"<span style='color:{color_hex}'>" + "".join(buffer) + "</span>")
            Glyph_art.append("".join(row_parts))
            Glyph_art.append("<br>")

        Glyph_art.append("</pre>")

        return "".join(Glyph_art)

    def _rgb_to_ansi256(self, r: int, g: int, b: int) -> int:
        """Map an RGB color to the closest 256-color palette index."""
        if r == g == b:
            if r < 8:
                return 16
            if r > 248:
                return 231
            return int(round(((r - 8) / 247) * 24)) + 232

        r_val = int(round(r / 255 * 5))
        g_val = int(round(g / 255 * 5))
        b_val = int(round(b / 255 * 5))
        return 16 + (36 * r_val) + (6 * g_val) + b_val

    def _rgb_to_ansi16(self, r: int, g: int, b: int) -> int:
        """Map an RGB color to a 16-color ANSI code."""
        brightness = (r + g + b) / 3
        bright = brightness > 127
        index = 0
        if r >= 128:
            index |= 1
        if g >= 128:
            index |= 2
        if b >= 128:
            index |= 4

        base = 90 if bright else 30
        return base + index

    def _normalize_dither_algorithm(
        self,
        dither_algorithm: Optional[Union[str, DitherAlgorithm]],
        dithering: bool,
    ) -> DitherAlgorithm:
        """Normalize dithering configuration."""
        if dither_algorithm is None:
            return DitherAlgorithm.FLOYD_STEINBERG if dithering else DitherAlgorithm.NONE
        if isinstance(dither_algorithm, DitherAlgorithm):
            return dither_algorithm
        value = dither_algorithm.lower()
        for alg in DitherAlgorithm:
            if alg.value == value:
                return alg
        return DitherAlgorithm.NONE

    def _normalize_posterize_bits(self, bits: Optional[int]) -> Optional[int]:
        """Normalize posterize bit depth to valid range."""
        if bits is None:
            return None
        return max(1, min(8, int(bits)))

    def _get_resample_filter(self, resample: str) -> int:
        """Resolve resampling filter from string."""
        if resample == "nearest":
            return Image.Resampling.NEAREST
        if resample == "bilinear":
            return Image.Resampling.BILINEAR
        if resample == "bicubic":
            return Image.Resampling.BICUBIC
        return Image.Resampling.LANCZOS


def image_to_glyph(
    image_path: ImageSource,
    output_path: Optional[str] = None,
    style: Optional[str] = None,
    color_mode: str = "none",
    **kwargs: Any,
) -> str:
    """High-level helper for quick image conversion.

    This convenience wrapper instantiates :class:`ImageGlyphConverter` with the
    provided parameters and performs a single image conversion. It mirrors the
    constructor arguments of :class:`ImageGlyphConverter` and supports both
    grayscale and color output modes.

    Args:
        image_path: Path to an image file or ``PIL.Image`` object.
        output_path: Optional destination to save the resulting glyph art.
        style: Optional style name forwarded to :meth:`ImageGlyphConverter.convert`.
        color_mode: ``"none"`` for grayscale, ``"ansi"``, ``"ansi16"``,
            ``"ansi256"``, ``"truecolor"``, or ``"html"`` for color.
        **kwargs: Additional parameters for :class:`ImageGlyphConverter`.

    Returns:
        Glyph art string.
    """

    params = {
        k: v
        for k, v in kwargs.items()
        if k
        in {
            "charset",
            "width",
            "height",
            "invert",
            "brightness",
            "contrast",
            "gamma",
            "auto_scale",
            "dithering",
            "dither_algorithm",
            "aspect_ratio",
            "resample",
            "autocontrast",
            "equalize",
            "invert_image",
            "edge_enhance",
            "sharpen",
            "blur_radius",
            "posterize_bits",
            "threads",
        }
    }
    converter = ImageGlyphConverter(**params)

    mode = color_mode.lower()
    if mode == "rgb":
        mode = "truecolor"
    if mode == "web":
        mode = "html"
    if mode in {"ansi", "ansi16", "ansi256", "truecolor", "html"}:
        return converter.convert_color(image_path, output_path=output_path, color_mode=mode)

    return converter.convert(image_path, output_path=output_path, style=style)
