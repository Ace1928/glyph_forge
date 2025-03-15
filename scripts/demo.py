#!/usr/bin/env python3
"""
⚡ GLYPH FORGE - EIDOSIAN DEMO ⚡

Showcase the full power of Glyph Forge with maximum impact.
This script demonstrates core API capabilities through practical,
visually stunning examples that highlight atomic precision and
hyper-optimized performance.
"""
import os
import sys
import time
import argparse
from pathlib import Path

# Add project root to Python path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.glyph_forge.api.glyph_api import get_api
from src.glyph_forge.utils.glyph_utils import detect_text_color_support, apply_ansi_style

# ANSI color styling
class Style:
    """Terminal styling with zero bloat."""
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RESET = "\033[0m"
    
    @staticmethod
    def apply(text, *styles):
        """Apply multiple styles with automatic reset."""
        style_codes = "".join(styles)
        return f"{style_codes}{text}{Style.RESET}"


def print_header(text):
    """Print a styled section header."""
    width = 70
    print("\n" + "=" * width)
    print(Style.apply(f"{text.center(width)}", Style.BOLD, Style.CYAN))
    print("=" * width + "\n")


def print_step(step_num, description):
    """Print a styled step marker."""
    print(Style.apply(f"【{step_num}】", Style.BOLD, Style.BLUE), 
          Style.apply(description, Style.BOLD))


def banner_showcase():
    """Demonstrate banner generation capabilities."""
    print_header("GLYPH FORGE BANNER SHOWCASE")
    api = get_api()
    
    # Step 1: Basic banner
    print_step(1, "Basic banner generation")
    banner = api.generate_banner("GLYPH FORGE", font="standard")
    print(banner)
    time.sleep(0.5)
    
    # Step 2: Styled banner
    print_step(2, "Styled banner (boxed)")
    banner = api.generate_banner("EIDOSIAN", style="boxed", font="slant")
    print(banner)
    time.sleep(0.5)
    
    # Step 3: With effects
    print_step(3, "Banner with effects (glow)")
    banner = api.generate_banner("POWER", style="minimal", effects=["glow"])
    print(banner)
    time.sleep(0.5)
    
    # Step 4: Custom style
    print_step(4, "Custom style (eidosian)")
    banner = api.generate_banner("FORGE", style="eidosian")
    print(banner)
    time.sleep(0.5)
    
    # Step 5: With color
    print_step(5, "With ANSI color")
    banner = api.generate_banner("HYPER", style="boxed", color=True)
    print(banner)


def image_showcase(image_path):
    """Demonstrate image conversion capabilities."""
    print_header("GLYPH FORGE IMAGE SHOWCASE")
    api = get_api()
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(Style.apply("Error: Image file not found!", Style.RED, Style.BOLD))
        return
    
    # Step 1: Basic conversion
    print_step(1, "Basic image conversion")
    result = api.image_to_Glyph(image_path, width=60)
    print(result)
    time.sleep(1)
    
    # Step 2: Different character set
    print_step(2, "Alternative character set (blocks)")
    result = api.image_to_Glyph(image_path, width=60, charset="blocks")
    print(result)
    time.sleep(1)
    
    # Step 3: Inverted
    print_step(3, "Inverted brightness")
    result = api.image_to_Glyph(image_path, width=60, invert=True)
    print(result)
    time.sleep(1)
    
    # Step 4: With color (if supported)
    if detect_text_color_support() > 0:
        print_step(4, "With ANSI color")
        result = api.image_to_Glyph(image_path, width=60, color_mode="ansi")
        print(result)
    else:
        print(Style.apply("Terminal does not support color output", Style.YELLOW))


def style_showcase():
    """Demonstrate styling capabilities."""
    print_header("GLYPH FORGE STYLE SHOWCASE")
    api = get_api()
    
    # Get available styles
    styles = list(api.get_available_styles().keys())
    
    # Generate sample text
    text = "Glyph"
    for i, style in enumerate(styles[:5]):  # Show first 5 styles
        print_step(i+1, f"Style: {style}")
        banner = api.generate_banner(text, style=style)
        print(banner)
        time.sleep(0.5)


def font_showcase():
    """Demonstrate different fonts."""
    print_header("GLYPH FORGE FONT SHOWCASE")
    api = get_api()
    
    # Some interesting fonts to showcase
    showcase_fonts = ["slant", "small", "standard", "big", "smslant"]
    
    # Generate sample text in each font
    text = "Forge"
    for i, font in enumerate(showcase_fonts):
        print_step(i+1, f"Font: {font}")
        try:
            banner = api.generate_banner(text, font=font, style="minimal")
            print(banner)
        except ValueError as e:
            print(Style.apply(f"Font not available: {e}", Style.YELLOW))
        time.sleep(0.5)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Glyph Forge Demo - Showcase capabilities',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        default=None,
        help='Path to image file for demos'
    )
    
    parser.add_argument(
        '--demo', '-d',
        type=str,
        choices=['all', 'banner', 'image', 'style', 'font'],
        default='all',
        help='Which demo to run'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Print title banner
    api = get_api()
    title = api.generate_banner("GLYPH FORGE", style="eidosian", color=True)
    print("\n" + title)
    print(Style.apply("EIDOSIAN HYPER-OPTIMIZED DEMO", Style.BOLD, Style.CYAN).center(80))
    print()
    
    # Run demos based on arguments
    if args.demo in ('all', 'banner'):
        banner_showcase()
    
    if args.demo in ('all', 'style'):
        style_showcase()
    
    if args.demo in ('all', 'font'):
        font_showcase()
    
    if args.demo in ('all', 'image'):
        # Find a default image if none provided
        image_path = args.image
        if not image_path:
            # Look for sample images in common locations
            possible_paths = [
                # Try to find a sample image in the project
                Path(__file__).parent.parent / 'examples' / 'sample.jpg',
                Path(__file__).parent.parent / 'tests' / 'test_images' / 'sample.jpg',
                # Try system wallpaper as fallback
                Path('/usr/share/backgrounds').glob('*.jpg'),
                Path('/usr/share/backgrounds').glob('*.png'),
            ]
            
            for path in possible_paths:
                if isinstance(path, Path) and path.exists():
                    image_path = str(path)
                    break
            
        if image_path and os.path.exists(image_path):
            image_showcase(image_path)
        else:
            print(Style.apply(
                "Image demo skipped - no valid image provided or found", 
                Style.YELLOW, Style.BOLD
            ))
    
    print("\n" + Style.apply("Demo completed. Forge on! ⚡", Style.BOLD, Style.CYAN))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(130)
