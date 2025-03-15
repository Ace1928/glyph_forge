#!/usr/bin/env python3
"""
⚡ GLYPH FORGE - EIDOSIAN IMAGIZER ⚡

Transform ordinary images into extraordinary ASCII art
with surgical precision and maximum efficiency.

This module provides a hyper-optimized command-line interface to the 
Glyph Forge image-to-ASCII conversion engine with zero compromises
on performance, features, or user experience.

Designed with Eidosian principles:
- Atomic efficiency - Every operation optimized to its quantum limit
- Zero waste - No redundant code, no computational fat
- Surgical precision - Parameter handling with absolute control
- Perfection by default - Intelligent decisions where options aren't specified
"""

import os
import sys
import time
import signal
import argparse
import logging
import platform
from enum import Enum
from pathlib import Path
from functools import wraps
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import typer

# Ensure project path is in Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Handle imports with fallback mechanisms for maximum resilience
try:
    from glyph_forge.api.ascii_api import get_api
    from glyph_forge.utils.ascii_utils import detect_text_color_support, get_terminal_size
    from glyph_forge.services.image_to_ascii import ColorMode, ImageAsciiConverter
    from glyph_forge.utils.alphabet_manager import AlphabetManager
except ImportError:
    # Direct import attempt for development scenarios
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.glyph_forge.api.ascii_api import get_api
    from src.glyph_forge.utils.ascii_utils import detect_text_color_support, get_terminal_size
    from src.glyph_forge.services.image_to_ascii import ImageAsciiConverter
    from src.glyph_forge.utils.alphabet_manager import AlphabetManager


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
            setattr(Style.supports_color, "_cached_result", detect_text_color_support() > 0)
        return getattr(Style.supports_color, "_cached_result")


class OutputFormat(Enum):
    """Supported output formats with descriptions for user discovery."""
    TEXT = "plain ASCII in .txt file"
    ANSI = "terminal-colored ASCII with ANSI codes"
    HTML = "web-friendly HTML with CSS styling"
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @classmethod
    def from_string(cls, value: str) -> 'OutputFormat':
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
                "css": cls.HTML
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
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", 
        datefmt="%H:%M:%S"
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
    logging.getLogger('glyph_forge').setLevel(level)
    
    # Suppress unnecessary noise from dependencies
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
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
    Decorator for surgical performance measurement with minimal overhead.
    
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
        
        # Format output based on execution time
        if duration < 0.001:  # < 1ms
            duration_str = f"{duration*1_000_000:.2f}µs"
        elif duration < 1.0:  # < 1s
            duration_str = f"{duration*1_000:.2f}ms"
        else:
            duration_str = f"{duration:.3f}s"
        
        # Use terminal color if supported
        if Style.supports_color():
            duration_msg = Style.apply(
                f"Conversion completed in {duration_str}", 
                Style.BOLD, Style.GREEN
            )
        else:
            duration_msg = f"Conversion completed in {duration_str}"
            
        print(duration_msg)
        
        # Log detailed timing data for analysis
        logging.debug(
            "Performance[%s]: %s seconds (%s)",
            func.__name__, duration, duration_str
        )
        
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
        if Style.supports_color() and any(tag in item.lower() for tag in ['eidosian', 'quantum', 'cosmic']):
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
    charset: Optional[str] = None,
    invert: bool = False,
    color_mode: str = "none",
    dithering: bool = False,
    brightness: Optional[float] = None,
    contrast: Optional[float] = None,
    optimize_contrast: bool = False
) -> str:
    """
    Convert image to ASCII with atomic precision and intelligent defaults.
    
    This core function handles all image-to-ASCII conversion with maximum
    parameter flexibility and zero redundant operations.
    
    Args:
        image_path: Source image file path
        output_path: Optional destination for output
        width: Width in characters
        height: Height in characters
        charset: Character set for conversion
        invert: Whether to invert brightness
        color_mode: Color output mode (none/ansi/html)
        dithering: Whether to apply dithering
        brightness: Brightness adjustment factor (0.0-2.0)
        contrast: Contrast adjustment factor (0.0-2.0)
        optimize_contrast: Whether to auto-optimize contrast
        
    Returns:
        ASCII art as string
    
    Raises:
        ValueError: If image cannot be loaded or processed
        FileNotFoundError: If image doesn't exist
    """
    logging.debug("Converting image: %s (charset=%s, width=%s, color=%s)",
                 image_path, charset, width, color_mode)
    
    # Import here to avoid circular imports
    from src.glyph_forge.services.image_to_ascii import ImageAsciiConverter
    
    # Validate image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Create converter with specified parameters
    converter = ImageAsciiConverter(
        charset=charset or "general",
        width=width or 100,
        height=height,
        invert=invert,
        dithering=dithering
    )
    
    # Set optional image parameters
    if brightness is not None or contrast is not None or optimize_contrast:
        image_params = {}
        
        if brightness is not None:
            image_params['brightness'] = brightness
            
        if contrast is not None:
            image_params['contrast'] = contrast
            
        if optimize_contrast:
            image_params['auto_optimize'] = True
            
        converter.set_image_params(**image_params)
    
    # Convert image based on color mode
    if color_mode in ("ansi", "html"):
        return converter.convert_color(
            image_path=image_path,
            output_path=output_path,
            color_mode=color_mode
        )
    else:
        return converter.convert(
            image_path=image_path,
            output_path=output_path
        )


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
                color_mode="none"
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
                print("".join(chars[i:i+grid_width]))
            
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
    Parse command line arguments with quantum precision.
    
    Creates a zero-compromise command interface with comprehensive options,
    validation, and clear documentation.
    
    Returns:
        Parsed argument namespace with validated values
    """
    parser = argparse.ArgumentParser(
        description='Glyph Forge - Eidosian Image Converter 🚀',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  imagize.py image.jpg                        # Basic usage with auto-settings
  imagize.py -w 120 -i -c ansi landscape.png  # Wide, inverted, with ANSI color
  imagize.py --preview-charset detailed       # Preview the 'detailed' charset
  imagize.py -o art.html -c html -s blocks portrait.jpg  # HTML output with block charset
  imagize.py --optimize -d -b 1.2 dark.jpg    # Auto-optimize with dithering and brightness
        """
    )

    # Image path argument
    parser.add_argument(
        'image', 
        type=str,
        nargs='?',
        help='Path to image file to convert'
    )

    # Dimension options
    dim_group = parser.add_argument_group('Dimensions')
    dim_group.add_argument(
        '-w', '--width', 
        type=int,
        default=None,
        metavar='CHARS',
        help='Width in characters (default: auto-fit to terminal)'
    )
    dim_group.add_argument(
        '-h', '--height', 
        type=int,
        default=None,
        metavar='CHARS',
        help='Height in characters (default: auto from aspect ratio)'
    )
    dim_group.add_argument(
        '-a', '--aspect',
        type=float,
        default=None,
        metavar='RATIO',
        help='Force specific aspect ratio (width/height)'
    )

    # Style options
    style_group = parser.add_argument_group('Style Options')
    style_group.add_argument(
        '-s', '--charset', 
        type=str,
        default=None,
        metavar='NAME',
        help='Character set to use (from --list-charsets)'
    )
    style_group.add_argument(
        '-i', '--invert',
        action='store_true',
        help='Invert brightness (dark ↔ light)'
    )
    style_group.add_argument(
        '-c', '--color',
        type=str,
        choices=['none', 'ansi', 'html'],
        default='none',
        help='Color output mode (default: none)'
    )
    style_group.add_argument(
        '-d', '--dither',
        action='store_true',
        help='Apply dithering for improved visual precision'
    )

    # Image adjustment options
    adjust_group = parser.add_argument_group('Image Adjustments')
    adjust_group.add_argument(
        '-b', '--brightness',
        type=float,
        default=None,
        metavar='FACTOR',
        help='Brightness adjustment (0.5-2.0, default: 1.0)'
    )
    adjust_group.add_argument(
        '--contrast',
        type=float,
        default=None,
        metavar='FACTOR',
        help='Contrast adjustment (0.5-2.0, default: 1.0)'
    )
    adjust_group.add_argument(
        '--optimize',
        action='store_true',
        help='Auto-optimize contrast and brightness'
    )

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '-o', '--output', 
        type=str,
        default=None,
        metavar='FILE',
        help='Save output to specified file'
    )

    # Special modes
    special_group = parser.add_argument_group('Special Modes')
    special_group.add_argument(
        '--list-charsets', 
        action='store_true',
        help='Display all available character sets'
    )
    special_group.add_argument(
        '--preview-charset',
        type=str,
        metavar='NAME',
        help='Preview a specific character set'
    )
    special_group.add_argument(
        '--sample',
        type=str,
        metavar='IMAGE',
        help='Sample image to use with --preview-charset'
    )

    # Global options
    global_group = parser.add_argument_group('Global Options')
    global_group.add_argument(
        '--debug',
        action='store_true',
        help='Enable detailed debug logging'
    )
    global_group.add_argument(
        '--version',
        action='store_true',
        help='Show version information and exit'
    )

    args = parser.parse_args()
    
    # Smart argument validation
    validation_errors = []
    
    # If not in a special mode, we need an image
    if not (args.list_charsets or args.preview_charset or args.version):
        if not args.image:
            validation_errors.append("Image path is required unless using --list-charsets, --preview-charset, or --version")
    
    # Sample image only makes sense with preview-charset
    if args.sample and not args.preview_charset:
        validation_errors.append("--sample can only be used with --preview-charset")
    
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
        version = importlib.metadata.version("ascii-forge")
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
    
    print("\nPart of the Eidosian Forge toolkit")
    print("⚡ https://github.com/username/glyph_forge ⚡\n")


# ─── MAIN ENTRY POINT ────────────────────────────────────────────────────────────

def main() -> int:
    """
    Main entry point with hyper-optimized execution flow.
    
    Implements a zero-compromise execution path with:
    - Perfect signal handling
    - Smart parameter resolution
    - Terminal-aware output
    - Optimized error handling
    - Maximum user feedback
    
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
    
    # Generate ASCII art from image
    try:
        # Calculate dimensions if needed
        width = args.width
        if width is None:
            # Auto-fit to terminal with margin
            term_width, _ = get_terminal_size()
            width = term_width - 4  # Leave small margin
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
            charset=args.charset,
            invert=args.invert,
            color_mode=args.color,
            dithering=args.dither,
            brightness=args.brightness,
            contrast=args.contrast,
            optimize_contrast=args.optimize
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
    help="Transform images into ASCII art with zero compromise",
    add_completion=True,
)

@app.command()
def convert(
    image: str = typer.Argument(..., help="Path to image file"),
    width: int = typer.Option(None, "--width", "-w", help="Width in characters"),
    height: int = typer.Option(None, "--height", "-h", help="Height in characters"),
    charset: str = typer.Option("general", "--charset", "-s", help="Character set to use"),
    invert: bool = typer.Option(False, "--invert", "-i", help="Invert brightness"),
    color: str = typer.Option("none", "--color", "-c", help="Color mode (none, ansi, html)"),
    dither: bool = typer.Option(False, "--dither", "-d", help="Apply dithering"),
    output: str = typer.Option(None, "--output", "-o", help="Save to file"),
    optimize: bool = typer.Option(False, "--optimize", help="Auto-optimize contrast"),
):
    """Convert an image into spectacular ASCII art."""
    try:
        result = convert_image(
            image_path=image,
            output_path=output,
            width=width,
            height=height,
            charset=charset,
            invert=invert,
            color_mode=color,
            dithering=dither,
            optimize_contrast=optimize,
        )
        
        if not output:
            print(result)
        else:
            print(f"ASCII art saved to {output}")
            
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
