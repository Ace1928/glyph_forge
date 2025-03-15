#!/usr/bin/env python3
"""
⚡ ASCII FORGE - EIDOSIAN BANNERIZER ⚡

Transform mundane text into extraordinary ASCII art banners
with maximum precision and surgical efficiency.

Entry point for the bannerize command line tool.
This script provides a convenient wrapper around the bannerize_text.py functionality
with full access to all customization options and styling presets.
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import signal
import threading
from functools import wraps

# Ensure project path is in Python path for development mode
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Core imports with path-awareness
try:
    from ascii_forge.api import get_api
    from ascii_forge.utils.ascii_utils import detect_text_color_support, get_terminal_size
except ImportError:
    # Direct import attempt for development scenarios
    from ..api import get_api
    from ..utils.ascii_utils import detect_text_color_support, get_terminal_size


# ANSI color codes for terminal output - Zero waste, maximum impact
class Style:
    """Terminal styling constants with zero bloat."""
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RESET = "\033[0m"
    
    @staticmethod
    def apply(text: str, *styles: str) -> str:
        """Apply multiple styles to text with automatic reset."""
        if not text:
            return ""
        style_codes = "".join(styles)
        return f"{style_codes}{text}{Style.RESET}"
    
    @staticmethod
    def supports_color() -> bool:
        """Determine if the current terminal supports color output."""
        return detect_text_color_support() > 0


def setup_logging(debug: bool = False) -> None:
    """Configure optimal logging with zero waste."""
    level = logging.DEBUG if debug else logging.INFO
    
    # Formatter with microsecond precision for performance analysis
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", 
        datefmt="%H:%M:%S"
    )
    
    # Setup handler with custom formatter
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to prevent duplication
    for hdlr in root_logger.handlers:
        root_logger.removeHandler(hdlr)
    
    # Add our custom handler
    root_logger.addHandler(handler)
    
    # Set specific level for our module
    logging.getLogger('ascii_forge').setLevel(level)
    
    logging.debug("Logging initialized with level: %s", "DEBUG" if debug else "INFO")


def print_header() -> None:
    """Print stylish ASCII Forge header with adaptive color support."""
    term_width, _ = get_terminal_size()
    
    if Style.supports_color():
        banner = Style.apply("⚡ ASCII FORGE BANNERIZER ⚡", Style.BOLD, Style.CYAN)
        separator = Style.apply("═" * (term_width - 4), Style.BLUE)
        print(f"\n{banner.center(term_width)}")
        print(f"{separator}\n")
    else:
        # Fallback for terminals without color support
        banner = "=== ASCII FORGE BANNERIZER ==="
        print(f"\n{banner.center(term_width)}")
        print("=" * term_width + "\n")


def measure_performance(func):
    """
    Decorator for surgical performance measurement with minimal overhead.
    
    Times function execution with microsecond precision and logs the duration
    at different granularities based on execution time.
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
                f"Operation completed in {duration_str}", 
                Style.BOLD, Style.GREEN
            )
        else:
            duration_msg = f"Operation completed in {duration_str}"
            
        print(duration_msg)
        
        # Log detailed timing data for analysis
        logging.debug(
            "Performance[%s]: %s seconds (%s)",
            func.__name__, duration, duration_str
        )
        
        return result
    return wrapper


def preview_style(api, text: str, style: str) -> None:
    """
    Generate and display a pixel-perfect style preview.
    
    Args:
        api: The ASCII Forge API instance
        text: Sample text to render
        style: Style name to preview
    """
    term_width, _ = get_terminal_size()
    
    # Create a visually distinct header for the preview
    if Style.supports_color():
        header = Style.apply(f"STYLE PREVIEW: '{style}'", Style.BOLD, Style.YELLOW)
    else:
        header = f"STYLE PREVIEW: '{style}'"
    
    separator = "─" * term_width
    
    print(f"\n{header}")
    print(separator)
    
    # Generate and display the banner with specified style
    try:
        result = api.generate_banner(text=text, style=style)
        print(result)
    except Exception as e:
        err_msg = f"Error rendering style '{style}': {str(e)}"
        if Style.supports_color():
            print(Style.apply(err_msg, Style.RED))
        else:
            print(err_msg)
    
    print(separator + "\n")


def list_items(items: List[str], title: str, columns: int = 4) -> None:
    """
    Display items in multi-column format with optimal terminal utilization.
    
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


@measure_performance
def generate_banner(api, args) -> None:
    """
    Generate banner with surgical precision and maximum efficiency.
    
    Args:
        api: ASCII Forge API instance
        args: Parsed command line arguments
    
    Uses the API to generate a banner and directs output appropriately,
    with comprehensive error handling and performance tracking.
    """
    logging.debug("Generating banner with parameters: text='%s', style='%s', font='%s', width=%s",
                 args.text, args.style, args.font, args.width)
    
    # Apply color if requested
    color_mode = args.color
    
    # Generate banner with specified parameters
    banner = api.generate_banner(
        text=args.text,
        style=args.style,
        font=args.font,
        width=args.width,
        color=color_mode
    )
    
    # Output based on destination
    if args.output:
        # Ensure directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        success = api.save_to_file(banner, str(output_path))
        if success:
            if Style.supports_color():
                path_msg = Style.apply(args.output, Style.BOLD)
                print(f"Banner saved to: {path_msg}")
            else:
                print(f"Banner saved to: {args.output}")
        else:
            error_msg = f"Error: Could not save to {args.output}"
            if Style.supports_color():
                print(Style.apply(error_msg, Style.RED), file=sys.stderr)
            else:
                print(error_msg, file=sys.stderr)
            sys.exit(1)
    else:
        # Print to console with proper line ending handling
        print(banner.rstrip('\n'))


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments with quantum precision.
    
    Returns:
        Parsed argument namespace
    
    Establishes a zero-compromise command interface with comprehensive
    options, validation, and self-documentation.
    """
    parser = argparse.ArgumentParser(
        description='⚡ ASCII Forge - Eidosian Bannerizer ⚡',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bannerize "Hello World"                 # Basic usage
  bannerize -s boxed -f big "ASCII Forge" # Custom style and font
  bannerize --list-fonts                  # List available fonts
  bannerize --preview -s eidosian "Test"  # Preview eidosian style
  bannerize "Quantum" -c -o banner.txt    # Colored output to file
        """
    )

    # Text argument
    parser.add_argument(
        'text', 
        type=str,
        nargs='?',
        help='Text to transform into ASCII art'
    )

    # Customization options
    style_group = parser.add_argument_group('Style Options')
    style_group.add_argument(
        '-f', '--font', 
        type=str,
        default=None,
        metavar='FONT',
        help='Font to use (from --list-fonts)'
    )
    style_group.add_argument(
        '-s', '--style', 
        type=str,
        default=None,
        metavar='STYLE',
        help='Visual style preset (from --list-styles)'
    )
    style_group.add_argument(
        '-w', '--width', 
        type=int,
        default=None,
        metavar='WIDTH',
        help='Character width constraint'
    )
    style_group.add_argument(
        '-c', '--color',
        action='store_true',
        help='Apply ANSI color to the output'
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

    # Information options
    info_group = parser.add_argument_group('Information')
    info_group.add_argument(
        '--list-fonts', 
        action='store_true',
        help='Display all available fonts'
    )
    info_group.add_argument(
        '--list-styles', 
        action='store_true',
        help='List all available style presets'
    )
    info_group.add_argument(
        '--preview',
        action='store_true',
        help='Preview the output with specified style'
    )
    
    # Debug options
    debug_group = parser.add_argument_group('Debugging')
    debug_group.add_argument(
        '--debug',
        action='store_true',
        help='Enable detailed debug logging'
    )
    debug_group.add_argument(
        '--version',
        action='store_true',
        help='Show version information and exit'
    )

    args = parser.parse_args()
    
    # Validate arguments
    validation_errors = []
    
    # Text required unless listing options or showing version
    if not args.text and not (args.list_fonts or args.list_styles or args.version):
        validation_errors.append("Text argument is required unless using --list-fonts, --list-styles, or --version")
    
    # Preview requires both text and style
    if args.preview and not args.style and args.text:
        validation_errors.append("Style must be specified with --style when using --preview")
    
    # Apply validation errors if any
    if validation_errors:
        for error in validation_errors:
            parser.error(error)
    
    return args


def signal_handler(sig, frame):
    """Handle interrupt signals gracefully."""
    print("\nOperation cancelled by user")
    sys.exit(130)  # 128 + SIGINT value (2)


def show_version():
    """Display version information with precision."""
    try:
        import importlib.metadata
        version = importlib.metadata.version("ascii-forge")
    except (ImportError, importlib.metadata.PackageNotFoundError):
        try:
            from ascii_forge import __version__
            version = __version__
        except ImportError:
            version = "unknown"
    
    # Get platform info
    python_version = sys.version.split()[0]
    platform_info = f"{platform.system()} {platform.release()}"
    
    if Style.supports_color():
        print(Style.apply(f"\nASCII Forge v{version}", Style.BOLD, Style.GREEN))
        print(f"Python {python_version} on {platform_info}")
    else:
        print(f"\nASCII Forge v{version}")
        print(f"Python {python_version} on {platform_info}")
    
    print("\nPart of the Eidosian Forge toolkit")
    print("⚡ https://github.com/your-username/ascii_forge ⚡\n")


def create_banner(text: str, font: str = "slant", style: str = "minimal") -> str:
    """
    Create a banner with the given text, font and style.
    
    Args:
        text: Text to display in banner
        font: Font name to use
        style: Style to apply
        
    Returns:
        Formatted banner text
    """
    api = get_api()
    return api.generate_banner(text=text, font=font, style=style)


def main() -> int:
    """
    Main entry point with hyper-optimized execution flow.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    
    This function implements a zero-compromise execution path,
    handling all command variations with surgical precision.
    """
    # Setup signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure logging
    setup_logging(args.debug)
    logging.debug("ASCII Forge Bannerizer initializing...")
    
    # Show version if requested
    if args.version:
        show_version()
        return 0
    
    # Display header
    print_header()
    
    # Instantiate API with zero-latency singleton pattern
    api = get_api()
    
    # Handle information commands
    if args.list_fonts:
        fonts = api.get_available_fonts()
        list_items(fonts, "Available ASCII Fonts", columns=3)
        return 0

    if args.list_styles:
        styles = api.get_available_styles()
        list_items(list(styles.keys()), "Available ASCII Styles", columns=3)
        return 0
    
    # Preview mode
    if args.preview and args.text and args.style:
        preview_style(api, args.text, args.style)
        return 0
    
    # Generate the banner
    try:
        generate_banner(api, args)
        return 0
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        error_message = f"Error generating banner: {str(e)}"
        if Style.supports_color():
            print(Style.apply(error_message, Style.BOLD, Style.RED), file=sys.stderr)
        else:
            print(error_message, file=sys.stderr)
        
        # Log detailed stack trace in debug mode
        logging.debug("Exception details:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
