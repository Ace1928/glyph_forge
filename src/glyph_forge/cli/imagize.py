#!/usr/bin/env python3
"""Image conversion command and reusable CLI helpers."""

import argparse
import logging
import os
import platform
import signal
import sys
import time
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Callable, List, Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from glyph_forge.services.image_to_glyph import ImageGlyphConverter
from glyph_forge.utils.alphabet_manager import AlphabetManager
from glyph_forge.utils.glyph_utils import detect_text_color_support, get_terminal_size

# ─── ADVANCED STYLING ENGINE ──────────────────────────────────────────────────────


class Style:
    """Terminal styling constants with zero bloat execution paths."""

    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RESET = "\033[0m"

    __slots__ = ()  # Memory optimization

    @staticmethod
    def apply(text: str, *styles: str) -> str:
        """Apply multiple styles to text with automatic reset and null safety."""
        if not text:
            return ""
        style_codes = "".join(styles)
        return f"{style_codes}{text}{Style.RESET}"

    @staticmethod
    def supports_color() -> bool:
        """Determine if the current terminal supports color output with caching."""
        if not hasattr(Style.supports_color, "_cached_result"):
            Style.supports_color._cached_result = detect_text_color_support() > 0
        return Style.supports_color._cached_result


class OutputFormat(Enum):
    """Supported output formats with descriptions for user discovery."""

    TEXT = "plain Glyph in .txt file"
    ANSI = "terminal-colored Glyph with ANSI codes"
    HTML = "web-friendly HTML with CSS styling"

    def __str__(self) -> str:
        return self.name.lower()

    @classmethod
    def from_string(cls, value: str) -> "OutputFormat":
        """Convert string to enum value with smart case handling."""
        try:
            return cls[value.upper()]
        except KeyError:
            return {
                "none": cls.TEXT,
                "plain": cls.TEXT,
                "text": cls.TEXT,
                "ansi": cls.ANSI,
                "terminal": cls.ANSI,
                "color": cls.ANSI,
                "html": cls.HTML,
                "web": cls.HTML,
                "css": cls.HTML,
            }.get(value.lower(), cls.TEXT)


# ─── CORE UTILITIES ────────────────────────────────────────────────────────────────
def setup_logging(debug: bool = False) -> None:
    """
    Configure optimal logging with zero-waste precision.

    Creates a microsecond-precise logging configuration that balances
    information density with signal-to-noise ratio.

    Args:
        debug: Enable debug level logging for maximum verbosity
    """
    level = logging.DEBUG if debug else logging.INFO

    # Format with microsecond precision for performance analysis
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )

    # Setup handler with custom formatter
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Configure root logger with zero duplication
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to prevent redundant output
    for hdlr in root_logger.handlers:
        root_logger.removeHandler(hdlr)

    # Add our custom handler
    root_logger.addHandler(handler)

    # Set specific level for our module
    logging.getLogger("glyph_forge").setLevel(level)

    # Suppress unnecessary noise from dependencies
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logging.debug("Logging initialized with level: %s", "DEBUG" if debug else "INFO")


def print_header() -> None:
    """
    Print stylish Glyph Forge header with adaptive color support.

    Automatically adjusts to terminal capabilities and dimensions,
    ensuring optimal visual impact on any system.
    """
    term_width, _ = get_terminal_size()

    if Style.supports_color():
        banner = Style.apply("⚡ GLYPH FORGE IMAGIZER ⚡", Style.BOLD, Style.CYAN)
        separator = Style.apply("═" * (term_width - 4), Style.BLUE)
        print(f"\n{banner.center(term_width)}")
        print(f"{separator}\n")
    else:
        # Fallback for terminals without color support
        banner = "=== GLYPH FORGE IMAGIZER ==="
        print(f"\n{banner.center(term_width)}")
        print("=" * term_width + "\n")


def measure_performance(func: Callable) -> Callable:
    """
    Decorator for optional performance measurement.

    Times function execution with adaptive precision formatting:
    - Microseconds for ultra-fast operations (<1ms)
    - Milliseconds for quick operations (<1s)
    - Seconds with 3 decimal places for longer operations

    Args:
        func: The function to measure

    Returns:
        Wrapped function that reports its execution time
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Record start time with maximum precision
        start_time = time.perf_counter()

        # Execute wrapped function
        result = func(*args, **kwargs)

        # Calculate duration with microsecond precision
        duration = time.perf_counter() - start_time

        logging.debug("Performance[%s]: %.6f seconds", func.__name__, duration)

        return result

    return wrapper


def list_items(items: List[str], title: str, columns: int = 4) -> None:
    """
    Display items in multi-column format with optimal terminal utilization.

    Automatically adapts to terminal dimensions and content length for
    maximum information density with zero wasted space.

    Args:
        items: List of items to display
        title: Section title
        columns: Maximum number of columns
    """
    if Style.supports_color():
        print(f"\n{Style.apply(title, Style.BOLD, Style.YELLOW)}:")
    else:
        print(f"\n{title}:")

    if not items:
        print("  No items available")
        return

    # Get terminal width for optimal display
    term_width, _ = get_terminal_size()

    # Calculate column width based on longest item
    max_len = max(len(item) for item in items) + 2  # Add padding

    # Determine optimal number of columns
    columns = min(columns, max(1, term_width // max_len))

    # Sort items for consistent display
    sorted_items = sorted(items)

    # Generate multicolumn display with zero waste
    for i, item in enumerate(sorted_items):
        if i % columns == 0 and i > 0:
            print()

        # Highlight special items if color is supported
        if Style.supports_color() and any(
            tag in item.lower() for tag in ["eidosian", "quantum", "cosmic"]
        ):
            print(f"  {Style.apply(item, Style.CYAN):{max_len}}", end="")
        else:
            print(f"  {item:{max_len}}", end="")

    print("\n")


def signal_handler(sig: int, frame) -> None:
    """
    Handle interrupt signals gracefully.

    Ensures clean exit with informative message instead of stack trace.

    Args:
        sig: Signal number
        frame: Current stack frame
    """
    if Style.supports_color():
        print(Style.apply("\nOperation cancelled by user", Style.YELLOW))
    else:
        print("\nOperation cancelled by user")
    sys.exit(130)  # 128 + SIGINT value (2)


# ─── CORE CONVERSION FUNCTION ───────────────────────────────────────────────────────
@measure_performance
def convert_image(
    image_path: str,
    output_path: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    upscale: int = 4,
    charset: Optional[str] = None,
    invert: bool = False,
    color_mode: str = "none",
    dithering: bool = False,
    dither_algorithm: Optional[str] = None,
    brightness: Optional[float] = None,
    contrast: Optional[float] = None,
    gamma: Optional[float] = None,
    aspect_ratio: Optional[float] = None,
    resample: Optional[str] = None,
    autocontrast: bool = False,
    equalize: bool = False,
    invert_image: bool = False,
    edge_enhance: bool = False,
    sharpen: bool = False,
    blur_radius: Optional[float] = None,
    posterize_bits: Optional[int] = None,
    optimize_contrast: bool = False,
) -> str:
    """
    Convert image to Glyph with atomic precision and intelligent defaults.

    This core function handles all image-to-Glyph conversion with maximum
    parameter flexibility and zero redundant operations.

    Args:
        image_path: Source image file path
        output_path: Optional destination for output
        width: Width in characters
        height: Height in characters
        charset: Character set for conversion
        invert: Whether to invert brightness
        color_mode: Color output mode (none/ansi/ansi16/ansi256/truecolor/html)
        dithering: Whether to apply dithering
        dither_algorithm: Dithering algorithm (none/floyd-steinberg/atkinson)
        brightness: Brightness adjustment factor (0.0-2.0)
        contrast: Contrast adjustment factor (0.0-2.0)
        gamma: Gamma correction factor (0.1-5.0)
        aspect_ratio: Character aspect ratio correction factor
        resample: Resampling filter (nearest/bilinear/bicubic/lanczos)
        autocontrast: Auto-stretch contrast for better detail
        equalize: Histogram equalization for enhanced dynamic range
        invert_image: Invert image luminance before conversion
        edge_enhance: Emphasize edges for structural clarity
        sharpen: Apply sharpening filter for crisper details
        blur_radius: Gaussian blur radius for smoothing
        posterize_bits: Reduce color depth (1-8) for stylized output
        optimize_contrast: Whether to auto-optimize contrast

    Returns:
        Glyph art as string

    Raises:
        ValueError: If image cannot be loaded or processed
        FileNotFoundError: If image doesn't exist
    """
    logging.debug(
        "Converting image: %s (charset=%s, width=%s, color=%s)",
        image_path,
        charset,
        width,
        color_mode,
    )

    # Import here to avoid circular imports

    # Validate image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # If no explicit size is provided, default to native resolution * upscale factor
    if width is None and height is None:
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                src_w, src_h = img.size
            scale = max(1, int(upscale))
            width = max(1, src_w * scale)
            height = max(1, src_h * scale)
            if aspect_ratio is None:
                aspect_ratio = 1.0
        except Exception:
            # Fall back to previous default if image metadata not available
            width = 160

    # Create converter with specified parameters
    converter = ImageGlyphConverter(
        charset=charset or "general",
        width=width or 160,
        height=height,
        invert=invert,
        dithering=dithering,
        dither_algorithm=dither_algorithm,
        gamma=gamma if gamma is not None else 1.0,
        aspect_ratio=aspect_ratio if aspect_ratio is not None else 0.55,
        resample=resample if resample is not None else "lanczos",
        autocontrast=autocontrast,
        equalize=equalize,
        invert_image=invert_image,
        edge_enhance=edge_enhance,
        sharpen=sharpen,
        blur_radius=blur_radius if blur_radius is not None else 0.0,
        posterize_bits=posterize_bits,
    )

    # Set optional image parameters
    if brightness is not None or contrast is not None or gamma is not None or optimize_contrast:
        image_params = {}

        if brightness is not None:
            image_params["brightness"] = brightness

        if contrast is not None:
            image_params["contrast"] = contrast

        if gamma is not None:
            image_params["gamma"] = gamma

        if optimize_contrast:
            image_params["autocontrast"] = True

        converter.set_image_params(**image_params)

    if (
        dither_algorithm is not None
        or aspect_ratio is not None
        or resample is not None
        or autocontrast
        or equalize
        or invert_image
        or edge_enhance
        or sharpen
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

    # Convert image based on color mode
    if color_mode == "rgb":
        color_mode = "truecolor"
    if color_mode == "web":
        color_mode = "html"
    if color_mode in ("ansi", "ansi16", "ansi256", "truecolor", "html"):
        return converter.convert_color(
            image_path=image_path, output_path=output_path, color_mode=color_mode
        )
    else:
        return converter.convert(image_path=image_path, output_path=output_path)


def preview_charset(charset: str, sample_image: Optional[str] = None) -> None:
    """
    Generate and display a character set preview.

    Creates a representation of a character set either using a sample
    image if provided, or a gradient pattern if not.

    Args:
        charset: Character set name to preview
        sample_image: Optional path to sample image
    """
    term_width, _ = get_terminal_size()

    # Create a visually distinct header for the preview
    if Style.supports_color():
        header = Style.apply(f"CHARSET PREVIEW: '{charset}'", Style.BOLD, Style.YELLOW)
    else:
        header = f"CHARSET PREVIEW: '{charset}'"

    separator = "─" * term_width

    print(f"\n{header}")
    print(separator)

    try:
        # Get the alphabet manager
        alphabet_manager = AlphabetManager()
        chars = alphabet_manager.get_alphabet(charset)

        if sample_image and Path(sample_image).exists():
            # Use the provided sample image
            result = convert_image(
                image_path=sample_image,
                charset=charset,
                width=min(80, term_width - 4),
                color_mode="none",
            )
            print(result)
        else:
            # Create a gradient preview
            # Show character set in density order
            print("Character set from darkest to lightest:\n")
            print("".join(chars))

            # Show in a grid pattern too
            print("\nGrid pattern:\n")

            grid_width = min(16, len(chars))
            for i in range(0, len(chars), grid_width):
                print("".join(chars[i : i + grid_width]))

    except Exception as e:
        err_msg = f"Error previewing charset '{charset}': {str(e)}"
        if Style.supports_color():
            print(Style.apply(err_msg, Style.RED))
        else:
            print(err_msg)

    print("\n" + separator + "\n")


# ─── ARGUMENT PARSING ─────────────────────────────────────────────────────────────
def parse_arguments() -> argparse.Namespace:
    """
    Parse legacy standalone command-line arguments.

    This parser remains for direct module execution and compatibility.

    Returns:
        Parsed argument namespace with validated values
    """
    parser = argparse.ArgumentParser(
        description="Glyph Forge image converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  imagize.py image.jpg                        # Basic usage with auto-settings
  imagize.py -w 120 -i -c ansi landscape.png  # Wide, inverted, with ANSI color
  imagize.py --preview-charset detailed       # Preview the 'detailed' charset
  imagize.py -o art.html -c html -s blocks portrait.jpg  # HTML output with block charset
  imagize.py --optimize -d -b 1.2 dark.jpg    # Auto-optimize with dithering and brightness
        """,
    )

    # Image path argument
    parser.add_argument("image", type=str, nargs="?", help="Path to image file to convert")

    # Dimension options
    dim_group = parser.add_argument_group("Dimensions")
    dim_group.add_argument(
        "-w",
        "--width",
        type=int,
        default=None,
        metavar="CHARS",
        help="Width in characters (default: auto-fit to terminal)",
    )
    dim_group.add_argument(
        "-H",
        "--height",
        type=int,
        default=None,
        metavar="CHARS",
        help="Height in characters (default: auto from aspect ratio)",
    )
    dim_group.add_argument(
        "-a",
        "--aspect",
        type=float,
        default=None,
        metavar="RATIO",
        help="Force specific aspect ratio (width/height)",
    )

    # Style options
    style_group = parser.add_argument_group("Style Options")
    style_group.add_argument(
        "-s",
        "--charset",
        type=str,
        default=None,
        metavar="NAME",
        help="Character set to use (from --list-charsets)",
    )
    style_group.add_argument(
        "-i", "--invert", action="store_true", help="Invert brightness (dark ↔ light)"
    )
    style_group.add_argument(
        "-c",
        "--color",
        type=str,
        choices=["none", "ansi", "ansi16", "ansi256", "truecolor", "html"],
        default="none",
        help="Color output mode (default: none)",
    )
    style_group.add_argument(
        "-d", "--dither", action="store_true", help="Apply dithering for improved visual precision"
    )
    style_group.add_argument(
        "--dither-algorithm",
        type=str,
        choices=["none", "floyd-steinberg", "atkinson"],
        default=None,
        help="Select dithering algorithm",
    )

    # Image adjustment options
    adjust_group = parser.add_argument_group("Image Adjustments")
    adjust_group.add_argument(
        "-b",
        "--brightness",
        type=float,
        default=None,
        metavar="FACTOR",
        help="Brightness adjustment (0.5-2.0, default: 1.0)",
    )
    adjust_group.add_argument(
        "--contrast",
        type=float,
        default=None,
        metavar="FACTOR",
        help="Contrast adjustment (0.5-2.0, default: 1.0)",
    )
    adjust_group.add_argument(
        "--gamma",
        type=float,
        default=None,
        metavar="FACTOR",
        help="Gamma correction (0.1-5.0, default: 1.0)",
    )
    adjust_group.add_argument(
        "--autocontrast", action="store_true", help="Auto-stretch contrast for improved detail"
    )
    adjust_group.add_argument(
        "--equalize", action="store_true", help="Histogram equalization for enhanced dynamic range"
    )
    adjust_group.add_argument(
        "--invert-image", action="store_true", help="Invert image luminance before conversion"
    )
    adjust_group.add_argument(
        "--edge-enhance", action="store_true", help="Emphasize edges for structural clarity"
    )
    adjust_group.add_argument(
        "--sharpen", action="store_true", help="Apply sharpening filter for crisper details"
    )
    adjust_group.add_argument(
        "--blur-radius",
        type=float,
        default=None,
        metavar="RADIUS",
        help="Gaussian blur radius for smoothing",
    )
    adjust_group.add_argument(
        "--posterize-bits", type=int, default=None, metavar="BITS", help="Posterize bit depth (1-8)"
    )
    adjust_group.add_argument(
        "--resample",
        type=str,
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        default=None,
        help="Resampling filter (default: lanczos)",
    )
    adjust_group.add_argument(
        "--optimize", action="store_true", help="Auto-optimize contrast and brightness"
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Save output to specified file",
    )

    # Special modes
    special_group = parser.add_argument_group("Special Modes")
    special_group.add_argument(
        "--list-charsets", action="store_true", help="Display all available character sets"
    )
    special_group.add_argument(
        "--preview-charset", type=str, metavar="NAME", help="Preview a specific character set"
    )
    special_group.add_argument(
        "--sample", type=str, metavar="IMAGE", help="Sample image to use with --preview-charset"
    )

    # Global options
    global_group = parser.add_argument_group("Global Options")
    global_group.add_argument("--debug", action="store_true", help="Enable detailed debug logging")
    global_group.add_argument(
        "--version", action="store_true", help="Show version information and exit"
    )

    args = parser.parse_args()

    # Smart argument validation
    validation_errors = []

    # If not in a special mode, we need an image
    if not (args.list_charsets or args.preview_charset or args.version):
        if not args.image:
            validation_errors.append(
                "Image path is required unless using --list-charsets, --preview-charset, or --version"
            )

    # Sample image only makes sense with preview-charset
    if args.sample and not args.preview_charset:
        validation_errors.append("--sample can only be used with --preview-charset")

    if args.dither_algorithm and not args.dither:
        validation_errors.append("--dither-algorithm requires --dither to be enabled")

    # Apply validation errors if any
    if validation_errors:
        for error in validation_errors:
            parser.error(error)

    return args


def show_version() -> None:
    """
    Display version information with maximum precision.

    Retrieves version information from multiple potential sources
    to ensure accurate reporting even in unusual installation scenarios.
    """
    try:
        import importlib.metadata

        version = importlib.metadata.version("glyph-forge")
    except (ImportError, ModuleNotFoundError):
        try:
            from glyph_forge import __version__

            version = __version__
        except ImportError:
            version = "1.0.0-eidosian"

    # Get platform info
    python_version = sys.version.split()[0]
    platform_info = f"{platform.system()} {platform.release()}"

    if Style.supports_color():
        print(Style.apply("\nGlyph Forge Imagizer", Style.BOLD, Style.CYAN))
        print(Style.apply(f"Version {version}", Style.GREEN))
        print(f"Python {python_version} on {platform_info}")
    else:
        print(f"\nGlyph Forge Imagizer v{version}")
        print(f"Python {python_version} on {platform_info}")

    print("\nhttps://github.com/Ace1928/glyph_forge\n")


# ─── MAIN ENTRY POINT ────────────────────────────────────────────────────────────
def main() -> int:
    """
    Legacy standalone entry point.

    Implements a compatibility execution path with:
    - Signal handling
    - Smart parameter resolution
    - Terminal-aware output
    - Optimized error handling
    - Concise user feedback

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Setup signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)

    # Parse command line arguments
    args = parse_arguments()

    # Configure logging
    setup_logging(args.debug)
    logging.debug("Glyph Forge Imagizer initializing...")

    # Show version if requested
    if args.version:
        show_version()
        return 0

    # Display header
    print_header()

    # Handle special command modes
    if args.list_charsets:
        from src.glyph_forge.utils.alphabet_manager import AlphabetManager

        charsets = AlphabetManager.list_available_alphabets()
        list_items(charsets, "Available Character Sets")
        return 0

    if args.preview_charset:
        preview_charset(args.preview_charset, args.sample)
        return 0

    # Check if image exists
    if not os.path.exists(args.image):
        error_msg = f"Error: Image file '{args.image}' not found"
        if Style.supports_color():
            print(Style.apply(error_msg, Style.RED), file=sys.stderr)
        else:
            print(error_msg, file=sys.stderr)
        return 1

    # Generate Glyph art from image
    try:
        # Calculate dimensions if needed
        width = args.width
        if width is None:
            # Auto-fit to terminal with margin
            term_width, _ = get_terminal_size()
            width = max(
                160, term_width - 4
            )  # Default to high-fidelity width when no upscale is used
            logging.debug(f"Auto-sized width to {width} based on terminal width {term_width}")

        # Calculate aspect ratio if specified
        if args.aspect is not None and args.height is None:
            height = int(width / args.aspect)
            logging.debug(f"Calculated height {height} from width {width} and aspect {args.aspect}")
        else:
            height = args.height

            # Convert the image with all specified parameters
            result = convert_image(
                image_path=args.image,
                output_path=args.output,
                width=width,
                height=height,
                upscale=args.upscale,
                charset=args.charset,
                invert=args.invert,
                color_mode=args.color,
                dithering=args.dither,
                dither_algorithm=args.dither_algorithm,
                brightness=args.brightness,
                contrast=args.contrast,
                gamma=args.gamma,
                aspect_ratio=args.aspect,
                resample=args.resample,
                autocontrast=args.autocontrast,
                equalize=args.equalize,
                invert_image=args.invert_image,
                edge_enhance=args.edge_enhance,
                sharpen=args.sharpen,
                blur_radius=args.blur_radius,
                posterize_bits=args.posterize_bits,
                optimize_contrast=args.optimize,
            )

        # Display result if not saving to file
        if not args.output:
            print(result)

        return 0
    except Exception as e:
        error_msg = f"Error converting image: {str(e)}"
        if Style.supports_color():
            print(Style.apply(error_msg, Style.BOLD, Style.RED), file=sys.stderr)
        else:
            print(error_msg, file=sys.stderr)

        if args.debug:
            import traceback

            traceback.print_exc()

        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(130)

# Create the Typer app instance that's imported in __init__.py
app = typer.Typer(
    help="Transform images into glyph art",
    add_completion=True,
    rich_markup_mode=None,
)


@app.command()
def convert(
    image: str = typer.Argument(..., help="Path to image file"),
    width: int = typer.Option(
        None, "--width", "-w", help="Width in characters (default: native image width * upscale)"
    ),
    upscale: int = typer.Option(
        4, "--upscale", help="Upscale factor when width/height not provided (default 4)"
    ),
    height: int = typer.Option(None, "--height", "-h", help="Height in characters"),
    charset: str = typer.Option("general", "--charset", "-s", help="Character set to use"),
    invert: bool = typer.Option(False, "--invert", "-i", help="Invert brightness"),
    color: str = typer.Option(
        "none", "--color", "-c", help="Color mode (none, ansi, ansi16, ansi256, truecolor, html)"
    ),
    dither: bool = typer.Option(False, "--dither", "-d", help="Apply dithering"),
    dither_algorithm: str = typer.Option(
        None, "--dither-algorithm", help="Dither algorithm (none, floyd-steinberg, atkinson)"
    ),
    gamma: float = typer.Option(None, "--gamma", help="Gamma correction (0.1-5.0)"),
    autocontrast: bool = typer.Option(False, "--autocontrast", help="Auto-stretch contrast"),
    equalize: bool = typer.Option(False, "--equalize", help="Histogram equalization"),
    invert_image: bool = typer.Option(False, "--invert-image", help="Invert image luminance"),
    edge_enhance: bool = typer.Option(False, "--edge-enhance", help="Emphasize edges"),
    sharpen: bool = typer.Option(False, "--sharpen", help="Apply sharpening filter"),
    blur_radius: float = typer.Option(None, "--blur-radius", help="Gaussian blur radius"),
    posterize_bits: int = typer.Option(None, "--posterize-bits", help="Posterize bit depth (1-8)"),
    resample: str = typer.Option(
        None, "--resample", help="Resample filter (nearest, bilinear, bicubic, lanczos)"
    ),
    output: str = typer.Option(None, "--output", "-o", help="Save to file"),
    share: str = typer.Option(
        None, "--share", help="Export shareable output (txt/html/svg/png/gif)"
    ),
    open_output: bool = typer.Option(False, "--open", help="Open the exported output"),
    optimize: bool = typer.Option(False, "--optimize", help="Auto-optimize contrast"),
    preset: str = typer.Option(
        None, "--preset", help="Preset style (cinematic/noir/neon/sketch/ultra/filmgrain/vaporwave)"
    ),
):
    """Convert an image into glyph art."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("Converting image..."),
            transient=True,
        ) as progress:
            progress.add_task("convert", total=None)
            if dither_algorithm and not dither:
                raise typer.BadParameter("--dither-algorithm requires --dither to be enabled")
            if preset:
                preset = preset.lower()
                if preset == "cinematic":
                    color = "truecolor"
                    dither = True
                    dither_algorithm = "atkinson"
                    edge_enhance = True
                    sharpen = True
                elif preset == "noir":
                    color = "none"
                    edge_enhance = True
                    sharpen = False
                    charset = "blocks"
                elif preset == "neon":
                    color = "truecolor"
                    dither = True
                    dither_algorithm = "floyd-steinberg"
                    charset = "detailed"
                elif preset == "sketch":
                    color = "none"
                    edge_enhance = True
                    charset = "ascii"
                elif preset == "ultra":
                    color = "truecolor"
                    dither = True
                    dither_algorithm = "atkinson"
                    edge_enhance = True
                    sharpen = True
                    gamma = 1.1
                elif preset == "filmgrain":
                    color = "ansi256"
                    dither = True
                    dither_algorithm = "floyd-steinberg"
                    gamma = 0.9
                    charset = "blocks"
                elif preset == "vaporwave":
                    color = "truecolor"
                    dither = True
                    dither_algorithm = "atkinson"
                    gamma = 1.2
                    charset = "detailed"

            result = convert_image(
                image_path=image,
                output_path=output,
                width=width,
                height=height,
                upscale=upscale,
                charset=charset,
                invert=invert,
                color_mode=color,
                dithering=dither,
                dither_algorithm=dither_algorithm,
                gamma=gamma,
                autocontrast=autocontrast,
                equalize=equalize,
                invert_image=invert_image,
                edge_enhance=edge_enhance,
                sharpen=sharpen,
                blur_radius=blur_radius,
                posterize_bits=posterize_bits,
                resample=resample,
                optimize_contrast=optimize,
            )

        if share:
            from ..streaming.naming import build_metadata, write_metadata
            from .share_utils import build_share_path, try_open_path, write_share

            fmt = share.lower()
            share_path = build_share_path(image, fmt, output)
            write_share(result, fmt, share_path)
            source_meta = {}
            try:
                from PIL import Image

                with Image.open(image) as img:
                    source_meta = {
                        "source_width": img.width,
                        "source_height": img.height,
                        "source_mode": img.mode,
                    }
            except Exception:
                pass
            meta = build_metadata(
                source=image,
                output_path=share_path,
                title=None,
                info=None,
                extra={
                    "format": fmt,
                    "share": True,
                    **source_meta,
                },
            )
            write_metadata(meta, share_path)
            print(f"Share export saved to {share_path}")
            if open_output:
                if not try_open_path(share_path):
                    print("No opener available for this system.")
        elif not output:
            print(result)
        else:
            print(f"Glyph art saved to {output}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


@app.command()
def list_charsets():
    """Show all available character sets."""
    available_charsets = AlphabetManager.list_available_alphabets()
    list_items(available_charsets, "Available Character Sets")


@app.command()
def preview(
    charset: str = typer.Argument(..., help="Character set to preview"),
    sample: str = typer.Option(None, "--sample", help="Sample image to use"),
):
    """Preview a specific character set."""
    preview_charset(charset, sample)
