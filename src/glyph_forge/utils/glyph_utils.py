"""
⚡ Glyph Forge Utilities ⚡

Core utilities for Glyph art transformation and manipulation.
This module provides atomic operations for text processing,
style resolution, and Glyph art manipulation.
"""

import re
import shutil
from typing import Dict, Any, Optional, Union, List, Tuple
import logging
import os

logger = logging.getLogger(__name__)


def sanitize_text(text: str) -> str:
    """
    Sanitize input text for safe Glyph art conversion.
    
    Removes potentially problematic characters and normalizes
    line endings for consistent processing.
    
    Args:
        text: Raw input text
        
    Returns:
        Sanitized text safe for processing
    """
    if not text:
        return ""
        
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove control characters except newlines
    text = re.sub(r'[\x00-\x09\x0b-\x1f\x7f-\x9f]', '', text)
    
    # Limit consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    logger.debug(f"Sanitized text: {len(text)} chars")
    return text


def resolve_style(style_name: str) -> Dict[str, Any]:
    """
    Resolve a style name to its configuration dictionary.
    
    Args:
        style_name: Name of style to resolve
        
    Returns:
        Style configuration dictionary
    """
    # Default minimal style configuration
    default_style = {
        "border": None,
        "padding": (0, 1),
        "alignment": "center",
        "effects": []
    }
    
    # Add custom style resolution logic here
    custom_styles = {
        "cyberpunk": {
            "border": "heavy",
            "padding": (1, 2),
            "alignment": "center",
            "effects": ["glow"]
        },
        "minimalist": {
            "border": None,
            "padding": (0, 0),
            "alignment": "left",
            "effects": []
        },
        "retro": {
            "border": "Glyph",
            "padding": (1, 3),
            "alignment": "center", 
            "effects": []
        },
        "quantum": {
            "border": "double",
            "padding": (2, 4),
            "alignment": "center",
            "effects": ["sparkle"]
        },
        "matrix": {
            "border": None,
            "padding": (1, 1),
            "alignment": "left",
            "effects": ["digital"]
        },
        "shadow": {
            "border": "single",
            "padding": (1, 2),
            "alignment": "left",
            "effects": ["shadow"]
        },
        "cosmic": {
            "border": "rounded",
            "padding": (2, 3),
            "alignment": "center",
            "effects": ["glow", "sparkle"]
        }
    }
    
    return custom_styles.get(style_name, default_style)


def trim_margins(text: str) -> str:
    """
    Remove excessive whitespace from the margins of Glyph art.
    
    Args:
        text: Glyph art text
        
    Returns:
        Trimmed Glyph art with minimal margins
    """
    if not text:
        return ""
        
    lines = text.split('\n')
    
    # Remove empty lines from start and end
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
        
    if not lines:
        return ""
        
    # Find minimum left padding
    min_padding = min(
        (len(line) - len(line.lstrip())) 
        for line in lines if line.strip()
    )
    
    # Remove consistent left padding
    if min_padding > 0:
        lines = [line[min_padding:] if line.strip() else line for line in lines]
        
    return '\n'.join(lines)


def center_Glyph_art(art: str, width: int) -> str:
    """
    Center Glyph art within a specified width.
    
    Args:
        art: Glyph art text
        width: Target width to center within
        
    Returns:
        Centered Glyph art
    """
    lines = art.split('\n')
    centered_lines = []
    
    max_line_length = max(len(line) for line in lines) if lines else 0
    
    for line in lines:
        if len(line) < width:
            padding = (width - len(line)) // 2
            centered_lines.append(' ' * padding + line)
        else:
            centered_lines.append(line)
            
    return '\n'.join(centered_lines)


def measure_Glyph_art(art: str) -> Tuple[int, int]:
    """
    Measure the dimensions of Glyph art.
    
    Args:
        art: Glyph art text
        
    Returns:
        Tuple of (width, height) in characters
    """
    lines = art.split('\n')
    height = len(lines)
    width = max(len(line) for line in lines) if lines else 0
    return (width, height)


def detect_box_borders(art: str) -> Optional[str]:
    """
    Detect if Glyph art has box borders and identify style.
    
    Args:
        art: Glyph art text
        
    Returns:
        Name of detected border style or None
    """
    lines = art.split('\n')
    if len(lines) < 3:
        return None
        
    # Check for common border patterns
    first_line = lines[0]
    last_line = lines[-1]
    
    if '┌' in first_line and '┐' in first_line and '└' in last_line and '┘' in last_line:
        return "single"
    elif '╔' in first_line and '╗' in first_line and '╚' in last_line and '╝' in last_line:
        return "double"
    elif '+' in first_line and '+' in last_line:
        return "Glyph"
        
    return None


def get_terminal_size() -> Tuple[int, int]:
    """
    Get current terminal dimensions with better fallbacks.
    
    Returns:
        Tuple of (width, height) in characters
    """
    try:
        columns, lines = shutil.get_terminal_size()
        return (columns, lines)
    except (AttributeError, OSError):
        # Fallback for environments without terminal
        return (80, 24)


def detect_text_color_support() -> int:
    """
    Detect terminal color support level.
    
    Returns:
        0: No color support
        1: Basic ANSI color (8 colors)
        2: 256 color support
        3: True color (24-bit) support
    """
    # Check NO_COLOR environment variable (industry standard)
    if os.environ.get("NO_COLOR", "") != "":
        return 0
        
    # Check color-related environment variables
    term = os.environ.get("TERM", "").lower()
    colorterm = os.environ.get("COLORTERM", "").lower()
    
    # Check for true color support
    if "truecolor" in colorterm or "24bit" in colorterm:
        return 3
        
    # Check for 256 color support
    if "256" in term or term in ("xterm-256color", "screen-256color"):
        return 2
        
    # Check for basic color support
    if "color" in term or term in ("xterm", "screen", "vt100"):
        return 1
        
    return 0


def apply_ansi_style(text: str, style: Union[str, List[str]]) -> str:
    """
    Apply ANSI style codes to text.
    
    Args:
        text: Text to style
        style: Style name or list of style names
        
    Returns:
        ANSI-styled text
    """
    # ANSI escape codes
    styles = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "italic": "\033[3m",
        "underline": "\033[4m",
        "blink": "\033[5m",
        "reverse": "\033[7m",
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bg_black": "\033[40m",
        "bg_red": "\033[41m",
        "bg_green": "\033[42m",
        "bg_yellow": "\033[43m",
        "bg_blue": "\033[44m",
        "bg_magenta": "\033[45m",
        "bg_cyan": "\033[46m",
        "bg_white": "\033[47m",
    }
    
    if isinstance(style, str):
        style_codes = styles.get(style, "")
    else:
        style_codes = "".join(styles.get(s, "") for s in style)
    
    return f"{style_codes}{text}{styles['reset']}"


def wrap_text(text: str, width: int) -> str:
    """
    Wrap text to specified width while preserving word boundaries.
    
    Args:
        text: Text to wrap
        width: Maximum width
        
    Returns:
        Wrapped text
    """
    if width <= 0:
        return text
        
    result = []
    for line in text.split('\n'):
        if len(line) <= width:
            result.append(line)
            continue
            
        # Process line that needs wrapping
        current_line = ""
        for word in line.split(' '):
            if len(current_line) + len(word) + 1 <= width:
                # Word fits on current line
                if current_line:
                    current_line += " "
                current_line += word
            else:
                # Start a new line
                if current_line:
                    result.append(current_line)
                if len(word) <= width:
                    current_line = word
                else:
                    # Word is longer than width, hard wrap
                    result.append(word[:width])
                    current_line = word[width:]
        
        if current_line:
            result.append(current_line)
    
    return '\n'.join(result)