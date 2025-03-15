"""
⚡ ASCII Forge Style Manager ⚡

Core styling engine for ASCII art text with maximum precision and efficiency.
This module provides atomic styling operations for borders, alignment, padding,
and specialized visual effects with zero bloat.

Key features:
- Pre-defined style presets for common use cases
- Modular styling components for composition
- Unicode-aware border rendering
- Output-preserving text alignment
"""

import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class BorderStyle(Enum):
    """Border style identifiers for semantic clarity."""
    NONE = "none"
    SINGLE = "single"
    DOUBLE = "double"
    ROUNDED = "rounded"
    BOLD = "bold"
    ASCII = "ascii"
    CUSTOM = "custom"


class Alignment(Enum):
    """Text alignment specifiers."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


# Style presets with customizable properties
STYLE_PRESETS = {
    "minimal": {
        "border": None,
        "padding": 0,
        "alignment": "left",
    },
    "boxed": {
        "border": "single",
        "padding": 1,
        "alignment": "center",
    },
    "double": {
        "border": "double",
        "padding": 2,
        "alignment": "center",
    },
    "rounded": {
        "border": "rounded",
        "padding": 1,
        "alignment": "center",
    },
    "bold": {
        "border": "bold",
        "padding": 1,
        "alignment": "center",
    },
    "ascii": {
        "border": "ascii",
        "padding": 1,
        "alignment": "center",
    },
    "eidosian": {  # Special Eidosian themed style
        "border": "custom",
        "border_chars": "🔥✨🌌✨🔥",
        "padding": 2,
        "alignment": "center",
    }
}

# Border character sets - each defines the 6 characters needed for a border:
# top-left, top-right, bottom-right, bottom-left, vertical, horizontal
BORDERS = {
    "single": {
        "top_left": "┌", "top_right": "┐",
        "bottom_left": "└", "bottom_right": "┘",
        "horizontal": "─", "vertical": "│",
    },
    "double": {
        "top_left": "╔", "top_right": "╗",
        "bottom_left": "╚", "bottom_right": "╝",
        "horizontal": "═", "vertical": "║",
    },
    "rounded": {
        "top_left": "╭", "top_right": "╮",
        "bottom_left": "╰", "bottom_right": "╯",
        "horizontal": "─", "vertical": "│",
    },
    "bold": {
        "top_left": "┏", "top_right": "┓",
        "bottom_left": "┗", "bottom_right": "┛", 
        "horizontal": "━", "vertical": "┃",
    },
    "ascii": {
        "top_left": "+", "top_right": "+",
        "bottom_left": "+", "bottom_right": "+",
        "horizontal": "-", "vertical": "|",
    }
}

# ASCII fallback mapping for terminals without Unicode support
ASCII_FALLBACK = {
    "single": "ascii",
    "double": "ascii",
    "rounded": "ascii",
    "bold": "ascii",
}


def apply_style(ascii_art: str, style_name: str = "minimal", **kwargs: Any) -> str:
    """
    Apply visual styling to ASCII art text with atomic precision.
    
    This function transforms plain ASCII art by applying borders, padding,
    alignment adjustments and other visual enhancements according to a
    named style preset or custom parameters.
    
    Args:
        ascii_art: The ASCII art text to style
        style_name: Name of the style preset to apply
        **kwargs: Override specific style parameters:
            - border: Border style name or None
            - padding: Integer or tuple of (vertical, horizontal) padding
            - alignment: Text alignment ("left", "center", "right")
            - border_chars: Custom border characters for "custom" border type
    
    Returns:
        Styled ASCII art string with all transformations applied
    
    Examples:
        >>> art = figlet.renderText("ASCII")
        >>> # Apply boxed style
        >>> styled = apply_style(art, "boxed")
        >>> # Apply minimal style with custom padding
        >>> styled = apply_style(art, "minimal", padding=2)
    """
    # Get style preset (or default to minimal)
    style = STYLE_PRESETS.get(style_name, STYLE_PRESETS["minimal"]).copy()
    
    # Override with any provided kwargs
    for key, value in kwargs.items():
        if key in style:
            style[key] = value
    
    # Split input into lines for processing
    lines = ascii_art.split('\n')
    max_line_length = max((len(line) for line in lines), default=0)
    
    # Process padding - handle both integer and tuple forms
    padding = style.get("padding", 0)
    if isinstance(padding, int):
        v_pad = h_pad = padding
    elif isinstance(padding, tuple) and len(padding) == 2:
        v_pad, h_pad = padding
    else:
        v_pad = h_pad = 0
        
    # Apply vertical padding (add empty lines)
    if v_pad > 0:
        lines = [''] * v_pad + lines + [''] * v_pad
        
    # Apply horizontal padding (add spaces to each line)
    if h_pad > 0:
        lines = [" " * h_pad + line + " " * h_pad for line in lines]
        max_line_length += h_pad * 2
    
    # Handle alignment
    alignment = style.get("alignment", "left")
    if alignment == "center":
        lines = [line.center(max_line_length) for line in lines]
    elif alignment == "right":
        lines = [line.rjust(max_line_length) for line in lines]
    
    # Add borders if specified
    border_type = style.get("border")
    if border_type == "custom" and "border_chars" in style:
        # Custom border with specified characters
        chars = style["border_chars"]
        if len(chars) > 0:
            border_char = chars[0]
            top = border_char * (max_line_length + 4)
            bottom = border_char * (max_line_length + 4)
            lines = [top] + [f"{border_char} {line.ljust(max_line_length)} {border_char}" for line in lines] + [bottom]
    elif border_type and border_type in BORDERS:
        # Standard border from predefined types
        border = BORDERS[border_type]
        top = f"{border['top_left']}{border['horizontal'] * (max_line_length + 2)}{border['top_right']}"
        bottom = f"{border['bottom_left']}{border['horizontal'] * (max_line_length + 2)}{border['bottom_right']}"
        lines = [top] + [f"{border['vertical']} {line.ljust(max_line_length)} {border['vertical']}" for line in lines] + [bottom]
    
    return '\n'.join(lines)


def get_available_styles() -> Dict[str, Dict[str, Any]]:
    """
    Get all available style presets with their configurations.
    
    Returns:
        Dictionary of style presets with their configurations
    
    Example:
        >>> styles = get_available_styles()
        >>> print(f"Available styles: {', '.join(styles.keys())}")
    """
    return STYLE_PRESETS.copy()


def get_available_borders() -> List[str]:
    """
    Get list of available border styles.
    
    Returns:
        List of border style names
    """
    return list(BORDERS.keys()) + ["custom"]


def create_custom_style(name: str, 
                      border: Optional[str] = None,
                      padding: Union[int, Tuple[int, int]] = 0,
                      alignment: str = "left",
                      **kwargs: Any) -> Dict[str, Any]:
    """
    Create a custom style configuration.
    
    Args:
        name: Name for the style
        border: Border style or None
        padding: Padding amount as int or (vertical, horizontal) tuple
        alignment: Text alignment ("left", "center", "right")
        **kwargs: Additional style properties
    
    Returns:
        Style configuration dictionary
    """
    style = {
        "border": border,
        "padding": padding,
        "alignment": alignment
    }
    
    # Add any additional properties
    for key, value in kwargs.items():
        style[key] = value
        
    # Register in presets if name provided
    if name and name not in STYLE_PRESETS:
        STYLE_PRESETS[name] = style
        
    return style


def register_style(name: str, style_config: Dict[str, Any]) -> None:
    """
    Register a new style in the global preset collection.
    
    Args:
        name: Name for the style
        style_config: Style configuration dictionary
    """
    STYLE_PRESETS[name] = style_config.copy()
    logger.debug(f"Registered new style: '{name}'")


def register_border(name: str, border_config: Dict[str, str]) -> None:
    """
    Register a new border style in the global collection.
    
    Args:
        name: Name for the border style
        border_config: Border character configuration with keys:
                       top_left, top_right, bottom_left, bottom_right,
                       horizontal, vertical
    
    Raises:
        ValueError: If border configuration is incomplete
    """
    required_keys = ["top_left", "top_right", "bottom_right", "bottom_left", 
                    "vertical", "horizontal"]
    
    # Validate config has all required keys
    for key in required_keys:
        if key not in border_config:
            raise ValueError(f"Border configuration missing required key: '{key}'")
    
    BORDERS[name] = border_config.copy()
    logger.debug(f"Registered new border style: '{name}'")


def detect_border_style(text: str) -> Optional[str]:
    """
    Detect border style of existing ASCII art.
    
    Args:
        text: ASCII art text to analyze
        
    Returns:
        Detected border style name or None if no border detected
    """
    lines = text.split('\n')
    if len(lines) < 3:
        return None
        
    # Extract potential border characters
    try:
        top_left = lines[0][0]
        top_right = lines[0][-1]
        bottom_left = lines[-1][0]
        bottom_right = lines[-1][-1]
    except IndexError:
        return None
    
    # Match against known border styles
    for name, border in BORDERS.items():
        if (border["top_left"] == top_left and 
            border["top_right"] == top_right and
            border["bottom_left"] == bottom_left and
            border["bottom_right"] == bottom_right):
            return name
    
    return None


def remove_border(text: str) -> str:
    """
    Remove border from ASCII art text.
    
    Args:
        text: ASCII art text with border
        
    Returns:
        ASCII art with border removed
    """
    lines = text.split('\n')
    if len(lines) < 3:
        return text
        
    # Detect if this has a border
    border_style = detect_border_style(text)
    if not border_style:
        return text
    
    # Remove top and bottom lines
    interior_lines = lines[1:-1]
    
    # Remove side borders (first and last character of each line)
    result = []
    for line in interior_lines:
        if len(line) >= 2:
            # Remove leading and trailing border chars, and trim any padding
            result.append(line[2:-2].rstrip())
    
    return '\n'.join(result)