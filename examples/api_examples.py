#!/usr/bin/env python3
"""
⚡ ASCII Forge API Examples ⚡

This module demonstrates the proper usage of the ASCII Forge API
through clear, concise, and practical examples with Eidosian precision.
"""
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ascii_forge.api import get_api


def example_banner_generation():
    """Example: Basic banner generation."""
    print("Example: Basic banner generation")
    print("-" * 40)
    
    api = get_api()
    banner = api.generate_banner("Hello World", style="boxed")
    print(banner)


def example_custom_styling():
    """Example: Custom banner styling."""
    print("\nExample: Custom banner styling")
    print("-" * 40)
    
    api = get_api()
    banner = api.generate_banner(
        "CUSTOM",
        style="minimal",
        border="double",
        alignment="center",
        effects=["glow"]
    )
    print(banner)


def example_multiple_fonts():
    """Example: Using different fonts."""
    print("\nExample: Using different fonts")
    print("-" * 40)
    
    api = get_api()
    fonts = ["slant", "standard", "small"]
    
    for font in fonts:
        try:
            print(f"Font: {font}")
            print(api.generate_banner("Font", font=font))
            print()
        except ValueError:
            print(f"Font '{font}' not available\n")


def example_image_conversion(image_path):
    """Example: Converting an image to ASCII art."""
    print("\nExample: Image conversion")
    print("-" * 40)
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    api = get_api()
    
    # Basic conversion
    result = api.image_to_ascii(image_path, width=60)
    print(result)
    
    # With different character set
    print("\nWith 'blocks' character set:")
    result = api.image_to_ascii(image_path, width=60, charset="blocks")
    print(result)


def example_saving_output():
    """Example: Saving output to a file."""
    print("\nExample: Saving output to a file")
    print("-" * 40)
    
    api = get_api()
    banner = api.generate_banner("SAVE ME", style="boxed")
    
    output_file = "output.txt"
    if api.save_to_file(banner, output_file):
        print(f"Banner saved to {output_file}")
        print(f"File size: {os.path.getsize(output_file)} bytes")
    else:
        print(f"Failed to save to {output_file}")


def example_configuration():
    """Example: Using configuration."""
    print("\nExample: Using configuration")
    print("-" * 40)
    
    api = get_api()
    
    # Get current config value
    current_font = api.config.get('banner', 'default_font', 'standard')
    print(f"Current default font: {current_font}")
    
    # Change config
    api.config.set('banner', 'default_font', 'small')
    
    # Generate with new default
    banner = api.generate_banner("Config")
    print(banner)
    
    # Restore original
    api.config.set('banner', 'default_font', current_font)


def run_all_examples():
    """Run all API examples."""
    example_banner_generation()
    example_custom_styling()
    example_multiple_fonts()
    
    # Try to find a sample image
    sample_paths = [
        "examples/sample.jpg",
        "tests/test_images/sample.jpg"
    ]
    
    image_path = None
    for path in sample_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    if image_path:
        example_image_conversion(image_path)
    
    example_saving_output()
    example_configuration()


if __name__ == "__main__":
    print("⚡ ASCII FORGE API EXAMPLES ⚡\n")
    run_all_examples()
    print("\nExamples completed!")
