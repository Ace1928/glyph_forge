import logging
import os
import warnings
from typing import Any, Dict, List, Optional, cast

from ..config.settings import get_config
from ..core.banner_generator import BannerGenerator
from ..core.style_manager import get_available_styles
from ..services.image_to_glyph import ImageGlyphConverter
from ..utils.alphabet_manager import AlphabetManager

logger = logging.getLogger(__name__)


class GlyphForgeAPI:
    """
    Public API for the Glyph Forge library.

    Provides a streamlined, unified interface to all Glyph Forge capabilities
    with intelligent caching, configuration management, and error handling.
    """

    def __init__(self) -> None:
        """Initialize the API with configuration and core components."""
        logger.debug("Initializing Glyph Forge API")
        self.config = get_config()
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize core components with optimal laziness."""
        # Default banner generator
        default_font = self.config.get("banner", "default_font", "slant")
        default_width = self.config.get("banner", "default_width", 80)
        self._banner_generator = BannerGenerator(font=default_font, width=default_width)

        # Image converter will be initialized on first use (lazy loading)
        self._image_converter: Optional[ImageGlyphConverter] = None

        logger.debug(
            f"Core components initialized with font='{default_font}', width={default_width}"
        )

    def _get_image_converter(self) -> ImageGlyphConverter:
        """Get or lazily initialize image converter."""
        if self._image_converter is None:
            # Default image converter settings
            default_charset = self.config.get("image", "default_charset", "general")
            default_width = self.config.get("image", "default_width", 100)
            default_brightness = self.config.get("image", "brightness", 1.0)
            default_contrast = self.config.get("image", "contrast", 1.0)
            default_gamma = self.config.get("image", "gamma", 1.0)
            default_dithering = self.config.get("image", "dithering", False)
            default_dither_algorithm = self.config.get(
                "image", "dither_algorithm", "floyd-steinberg"
            )
            default_aspect_ratio = self.config.get("image", "aspect_ratio", 0.55)
            default_resample = self.config.get("image", "resample", "lanczos")
            default_autocontrast = self.config.get("image", "autocontrast", False)
            default_equalize = self.config.get("image", "equalize", False)
            default_invert_image = self.config.get("image", "invert_image", False)
            default_edge_enhance = self.config.get("image", "edge_enhance", False)
            default_sharpen = self.config.get("image", "sharpen", False)
            default_blur_radius = self.config.get("image", "blur_radius", 0.0)
            default_posterize_bits = self.config.get("image", "posterize_bits", None)
            self._image_converter = ImageGlyphConverter(
                charset=default_charset,
                width=default_width,
                brightness=default_brightness,
                contrast=default_contrast,
                gamma=default_gamma,
                dithering=default_dithering,
                dither_algorithm=default_dither_algorithm,
                aspect_ratio=default_aspect_ratio,
                resample=default_resample,
                autocontrast=default_autocontrast,
                equalize=default_equalize,
                invert_image=default_invert_image,
                edge_enhance=default_edge_enhance,
                sharpen=default_sharpen,
                blur_radius=default_blur_radius,
                posterize_bits=default_posterize_bits,
            )
            logger.debug(
                "Image converter initialized with charset='%s', width=%s",
                default_charset,
                default_width,
            )

        assert self._image_converter is not None
        return self._image_converter

    def generate_banner(
        self,
        text: str,
        style: Optional[str] = None,
        font: Optional[str] = None,
        width: Optional[int] = None,
        effects: Optional[List[str]] = None,
        color: bool = False,
    ) -> str:
        """
        Generate an Glyph art banner from text with intelligent parameter handling.

        Args:
            text: Text to convert into banner
            style: Style preset to apply (default from config)
            font: Font to use (default from config)
            width: Width for the banner (default from config)
            effects: Special effects to apply (default from style)
            color: Whether to apply ANSI color to output

        Returns:
            Glyph art banner
        """
        # Use defaults from config if not specified
        if style is None:
            style = self.config.get("banner", "default_style", "minimal")

        # Regenerate banner generator if font or width changed
        if font is not None or width is not None:
            temp_font = font if font is not None else self._banner_generator.font
            temp_width = width if width is not None else self._banner_generator.width
            generator = BannerGenerator(font=temp_font, width=temp_width)
            return cast(str, generator.generate(text, style=style, effects=effects, color=color))

        # Use existing generator
        return cast(
            str,
            self._banner_generator.generate(
                text,
                style=style,
                effects=effects,
                color=color,
            ),
        )

    def image_to_glyph(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        charset: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        invert: bool = False,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None,
        dithering: bool = False,
        dither_algorithm: Optional[str] = None,
        gamma: Optional[float] = None,
        aspect_ratio: Optional[float] = None,
        resample: Optional[str] = None,
        autocontrast: Optional[bool] = None,
        equalize: Optional[bool] = None,
        invert_image: Optional[bool] = None,
        edge_enhance: Optional[bool] = None,
        sharpen: Optional[bool] = None,
        blur_radius: Optional[float] = None,
        posterize_bits: Optional[int] = None,
        color_mode: str = "none",
    ) -> str:
        """
        Convert an image to Glyph art with comprehensive parameter support.

        Args:
            image_path: Path to the image file
            output_path: Path to save the result (optional)
            charset: Character set to use (default from config)
            width: Width in characters (default from config)
            height: Height in characters (optional)
            invert: Whether to invert the brightness
            brightness: Brightness adjustment factor (0.0-2.0)
            contrast: Contrast adjustment factor (0.0-2.0)
            dithering: Whether to apply dithering
            dither_algorithm: Dithering algorithm ("none", "floyd-steinberg", "atkinson")
            gamma: Gamma correction factor (0.1-5.0)
            aspect_ratio: Character aspect ratio correction factor
            resample: Resampling filter (nearest, bilinear, bicubic, lanczos)
            autocontrast: Auto-stretch contrast for better detail
            equalize: Histogram equalization for enhanced dynamic range
            invert_image: Invert image luminance before conversion
            edge_enhance: Emphasize edges for structural clarity
            sharpen: Apply sharpening filter for crisper details
            blur_radius: Gaussian blur radius for smoothing
            posterize_bits: Reduce color depth (1-8) for stylized output
            color_mode: Color output mode ("none", "ansi", "ansi16", "ansi256", "truecolor", "html")

        Returns:
            Glyph art representation of the image
        """
        # Get or create image converter
        converter = self._get_image_converter()

        # Apply any parameter overrides
        if (
            charset is not None
            or width is not None
            or height is not None
            or invert
            or dithering
            or dither_algorithm is not None
            or gamma is not None
            or aspect_ratio is not None
            or resample is not None
            or autocontrast is not None
            or equalize is not None
            or invert_image is not None
            or edge_enhance is not None
            or sharpen is not None
            or blur_radius is not None
            or posterize_bits is not None
        ):
            # Create a new converter with specified parameters
            temp_charset = charset if charset is not None else converter.charset
            temp_width = width if width is not None else converter.width
            converter = ImageGlyphConverter(
                charset=temp_charset,
                width=temp_width,
                height=height,
                invert=invert,
                dithering=dithering,
                dither_algorithm=dither_algorithm,
                gamma=gamma if gamma is not None else converter.gamma,
                aspect_ratio=aspect_ratio if aspect_ratio is not None else converter.aspect_ratio,
                resample=resample if resample is not None else converter.resample,
                autocontrast=autocontrast if autocontrast is not None else converter.autocontrast,
                equalize=equalize if equalize is not None else converter.equalize,
                invert_image=invert_image if invert_image is not None else converter.invert_image,
                edge_enhance=edge_enhance if edge_enhance is not None else converter.edge_enhance,
                sharpen=sharpen if sharpen is not None else converter.sharpen,
                blur_radius=blur_radius if blur_radius is not None else converter.blur_radius,
                posterize_bits=posterize_bits
                if posterize_bits is not None
                else converter.posterize_bits,
            )

        # Set optional brightness, contrast, and gamma
        if brightness is not None or contrast is not None or gamma is not None:
            converter.set_image_params(
                brightness=brightness if brightness is not None else converter.brightness,
                contrast=contrast if contrast is not None else converter.contrast,
                gamma=gamma if gamma is not None else converter.gamma,
            )

        if (
            dither_algorithm is not None
            or aspect_ratio is not None
            or resample is not None
            or autocontrast is not None
            or equalize is not None
            or invert_image is not None
            or edge_enhance is not None
            or sharpen is not None
            or blur_radius is not None
            or posterize_bits is not None
        ):
            converter.set_image_params(
                dither_algorithm=dither_algorithm,
                aspect_ratio=aspect_ratio,
                resample=resample,
                autocontrast=autocontrast,
                equalize=equalize,
                invert_image=invert_image,
                edge_enhance=edge_enhance,
                sharpen=sharpen,
                blur_radius=blur_radius,
                posterize_bits=posterize_bits,
            )

        # Convert with or without color
        mode = color_mode.lower()
        if mode == "rgb":
            mode = "truecolor"
        if mode == "web":
            mode = "html"
        if mode in ("ansi", "ansi16", "ansi256", "truecolor", "html"):
            return converter.convert_color(
                image_path=image_path,
                output_path=output_path,
                color_mode=mode,
            )
        else:
            return converter.convert(image_path=image_path, output_path=output_path)

    def image_to_Glyph(self, *args: Any, **kwargs: Any) -> str:  # noqa: N802
        """Compatibility alias for :meth:`image_to_glyph`.

        The original mixed-case name remains available through the 1.x series
        so existing applications can migrate without an abrupt break.
        """

        warnings.warn(
            "image_to_Glyph() is deprecated; use image_to_glyph()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.image_to_glyph(*args, **kwargs)

    def get_available_fonts(self) -> List[str]:
        """
        Get a list of available font names.

        Returns:
            List of available font names
        """
        return cast(List[str], self._banner_generator.available_fonts())

    def get_available_styles(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a dictionary of available style presets.

        Returns:
            Dictionary mapping style names to their configurations
        """
        return cast(Dict[str, Dict[str, Any]], get_available_styles())

    def get_available_alphabets(self) -> List[str]:
        """
        Get a list of available character sets/alphabets.

        Returns:
            List of available alphabet names
        """
        return cast(List[str], AlphabetManager.list_available_alphabets())

    def save_to_file(self, Glyph_art: str, file_path: str) -> bool:
        """
        Save Glyph art to a file with proper directory creation.

        Args:
            Glyph_art: Glyph art text to save
            file_path: Path to save the file

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

            # Write file with UTF-8 encoding for maximum compatibility
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(Glyph_art)

            logger.debug(f"Saved Glyph art to {file_path}")
            return True
        except (IOError, OSError) as e:
            logger.error(f"Failed to save file: {str(e)}")
            return False

    def preview_font(self, font: str, text: str = "Glyph Forge") -> str:
        """
        Generate a preview of a specific font.

        Args:
            font: Name of the font to preview
            text: Text to use for preview

        Returns:
            Glyph art using specified font
        """
        generator = BannerGenerator(font=font, width=self._banner_generator.width)
        return cast(str, generator.generate(text))

    def preview_style(self, style: str, text: str = "Glyph Forge") -> str:
        """
        Generate a preview of a specific style.

        Args:
            style: Name of the style to preview
            text: Text to use for preview

        Returns:
            Glyph art using specified style
        """
        return cast(str, self._banner_generator.generate(text, style=style))

    def convert_text_to_art(self, text: str, font: str = "standard") -> str:
        """
        Convert plain text to Glyph art without additional styling.

        Args:
            text: Text to convert
            font: Font to use

        Returns:
            Glyph art representation
        """
        generator = BannerGenerator(font=font, width=self._banner_generator.width)
        return cast(str, generator.figlet.renderText(text))


# Singleton API instance
_api_instance = None


def get_api() -> GlyphForgeAPI:
    """
    Get the GlyphForgeAPI singleton instance with zero redundant initialization.

    Returns:
        GlyphForgeAPI instance
    """
    global _api_instance
    if _api_instance is None:
        _api_instance = GlyphForgeAPI()
    return _api_instance
