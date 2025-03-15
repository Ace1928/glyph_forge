"""
⚡ ASCII Forge Banner Generator ⚡

Core engine for transforming ordinary text into extraordinary ASCII art banners.
This module implements the atomic operations necessary for banner generation
with maximum efficiency and surgical precision.

Key features:
- Deterministic text-to-banner conversion
- Multiple styling paradigms
- Font system with extensive compatibility
- Border and decoration management
- Performance-optimized rendering pipeline
"""

from pyfiglet import Figlet
from typing import List, Dict, Optional, Tuple, Union
import os
import re
from enum import Enum
import logging
import time
import hashlib

logger = logging.getLogger(__name__)


class BannerStyle(Enum):
    """Predefined banner styling paradigms for consistent visual language."""
    MINIMAL = "minimal"         # Clean, distraction-free presentation
    BOXED = "boxed"             # Full border encapsulation
    SHADOWED = "shadowed"       # Dimensional depth effect
    DOUBLE = "double"           # Double-line border treatment
    METALLIC = "metallic"       # High-contrast industrial aesthetic
    CIRCUIT = "circuit"         # Digital circuit-inspired patterns
    EIDOSIAN = "eidosian"       # Maximum impact presentation
    CUSTOM = "custom"           # User-defined styling


class BannerGenerator:
    """
    Core banner generation engine for ASCII Forge.
    
    Transforms text into ASCII art banners with customizable styles through
    a hyper-optimized rendering pipeline. Each generated banner maintains
    pixel-perfect proportions while allowing for extensive customization.
    
    Attributes:
        figlet (Figlet): The FIGlet rendering engine
        font (str): Active font identifier
        width (int): Maximum render width
        height (int): Maximum render height
        cache (Dict): Performance optimization cache
    """

    # Default character sets for various border styles
    BORDER_SETS = {
        "single": "┌┐┘└│─",      # ┌───┐
        "double": "╔╗╝╚║═",      # ╔═══╗
        "rounded": "╭╮╯╰│─",     # ╭───╮
        "heavy": "┏┓┛┗┃━",       # ┏━━━┓
        "ascii": "++-+||"        # +---+
    }
    
    # Predefined style configurations
    STYLE_PRESETS = {
        BannerStyle.MINIMAL.value: {
            "border": None,
            "padding": (0, 1),
            "alignment": "center",
            "effects": []
        },
        BannerStyle.BOXED.value: {
            "border": "single",
            "padding": (1, 2),
            "alignment": "center",
            "effects": []
        },
        BannerStyle.SHADOWED.value: {
            "border": None,
            "padding": (0, 1),
            "alignment": "left",
            "effects": ["shadow"]
        },
        BannerStyle.DOUBLE.value: {
            "border": "double",
            "padding": (2, 2),
            "alignment": "center",
            "effects": []
        },
        BannerStyle.METALLIC.value: {
            "border": "heavy",
            "padding": (1, 2),
            "alignment": "center", 
            "effects": ["emboss"]
        },
        BannerStyle.CIRCUIT.value: {
            "border": "single",
            "padding": (1, 3),
            "alignment": "center",
            "effects": ["digital"]
        },
        BannerStyle.EIDOSIAN.value: {  # Special Eidosian themed style
            "border": "heavy",
            "padding": (1, 3),
            "alignment": "center",
            "effects": ["glow"]
        }
    }
    
    # Character sets for various effects
    EFFECT_CHARS = {
        "glow": "✦✧✨⋆⭐",
        "sparkle": "✦✧✨⋆⭐✫",
        "digital": "01",
        "shadow": "░▒▓█",
    }
    
    def __init__(self, 
                 font: str = 'slant', 
                 width: int = 80,
                 height: Optional[int] = None,
                 cache_enabled: bool = True):
        """
        Initialize the banner generator with specified parameters.
        
        Args:
            font: FIGlet font identifier (default: 'slant')
            width: Maximum width constraint for output (default: 80)
            height: Maximum height constraint for output (default: None)
            cache_enabled: Enable performance cache (default: True)
            
        Raises:
            ValueError: If font name is invalid or unavailable
        """
        try:
            self.figlet = Figlet(font=font, width=width)
            self.font = font
            self.width = width
            self.height = height
            self._validate_font(font)
            
            # Performance optimization cache
            self.cache = {} if cache_enabled else None
            self.cache_timestamps = {} if cache_enabled else None
            self.cache_enabled = cache_enabled
            
            # Metrics tracking
            self._render_count = 0
            self._cache_hits = 0
            self._last_cache_cleanup = time.time()
            self._cache_max_size = 100
            self._cache_ttl = 3600  # 1 hour in seconds
            
            # Unicode support detection
            self._unicode_supported = self._check_unicode_support()
            
            logger.debug(f"BannerGenerator initialized with font='{font}', width={width}")
        except Exception as e:
            logger.error(f"Failed to initialize BannerGenerator: {str(e)}")
            raise ValueError(f"Invalid font '{font}'. Error: {str(e)}")

    def _validate_font(self, font: str) -> None:
        """
        Validate font availability in current environment.
        
        Args:
            font: Font name to validate
            
        Raises:
            ValueError: If font is not available
        """
        if font not in self.available_fonts():
            available = ", ".join(self.available_fonts()[:5]) + "..."
            raise ValueError(
                f"Font '{font}' not available. Try: {available}"
            )

    def _check_unicode_support(self) -> bool:
        """Check if the terminal supports Unicode characters."""
        try:
            # Try writing a unicode character to null device
            with open(os.devnull, 'w', encoding='utf-8') as f:
                f.write("★")
            return True
        except (UnicodeEncodeError, UnicodeDecodeError):
            return False

    def generate(self, 
                text: str, 
                style: str = "minimal",
                padding: Optional[Tuple[int, int]] = None,
                border: Optional[str] = None,
                alignment: str = "center",
                effects: Optional[List[str]] = None,
                color: bool = False) -> str:
        """
        Generate an ASCII art banner from input text.
        
        This function performs the core text-to-ASCII-art transformation
        with precise style application and optimized rendering.
        
        Args:
            text: Input text to transform into ASCII art
            style: Style preset name (default: "minimal")
            padding: Custom (vertical, horizontal) padding
            border: Border style override
            alignment: Text alignment ("left", "center", "right")
            effects: List of special effects to apply
            color: Enable ANSI color output
            
        Returns:
            Fully styled and processed ASCII art banner
            
        Raises:
            ValueError: For invalid style or configuration parameters
        """
        # Input validation and sanitization
        from ..utils.ascii_utils import sanitize_text
        clean_text = sanitize_text(text)
        
        # Performance optimization via caching
        cache_key = self._generate_cache_key(clean_text, style, padding, border, alignment, effects)
        if self.cache_enabled and cache_key in self.cache:
            self._cache_hits += 1
            logger.debug(f"Cache hit for '{text[:20]}...'")
            return self.cache[cache_key]
        
        # Perform cache maintenance occasionally
        if self.cache_enabled and time.time() - self._last_cache_cleanup > 60:  # Once per minute
            self._maintain_cache()
        
        # Core rendering process
        ascii_art = self.figlet.renderText(clean_text)
        self._render_count += 1
        
        # Style application
        styled_art = self._apply_styling(
            ascii_art,
            style=style,
            custom_padding=padding,
            custom_border=border,
            custom_alignment=alignment,
            custom_effects=effects or []
        )
        
        # Apply ANSI colors if requested
        if color:
            styled_art = self._apply_color(styled_art)
        
        # Cache the result for future requests
        if self.cache_enabled:
            self.cache[cache_key] = styled_art
            self.cache_timestamps[cache_key] = time.time()
            
        return styled_art
    
    def _generate_cache_key(self, text: str, *args) -> str:
        """Generate a unique cache key for the given parameters."""
        # Combine parameters into a string and hash for compact representation
        params = [text, self.font, str(self.width)]
        params.extend([str(arg) for arg in args if arg is not None])
        combined = "|".join(params)
        
        hash_obj = hashlib.md5(combined.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def _maintain_cache(self) -> None:
        """Remove expired entries from cache and enforce size limits."""
        if not self.cache_enabled:
            return
            
        current_time = time.time()
        expired_keys = []
        
        # Find expired entries
        for key, timestamp in list(self.cache_timestamps.items()):
            if current_time - timestamp > self._cache_ttl:
                expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
            if key in self.cache_timestamps:
                del self.cache_timestamps[key]
        
        # Enforce maximum cache size
        if len(self.cache) > self._cache_max_size:
            # Remove oldest entries
            sorted_keys = sorted(
                self.cache_timestamps.keys(),
                key=lambda k: self.cache_timestamps.get(k, float('inf'))
            )
            for key in sorted_keys[:len(self.cache) - self._cache_max_size]:
                if key in self.cache:
                    del self.cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]
        
        self._last_cache_cleanup = current_time
        logger.debug(f"Cache maintenance: removed {len(expired_keys)} expired entries")
    
    def _apply_styling(self, 
                      ascii_art: str, 
                      style: str,
                      custom_padding: Optional[Tuple[int, int]] = None,
                      custom_border: Optional[str] = None,
                      custom_alignment: Optional[str] = None,
                      custom_effects: List[str] = None) -> str:
        """
        Apply comprehensive styling to raw ASCII art.
        
        Args:
            ascii_art: Raw ASCII art text
            style: Style preset name
            custom_padding: Override for padding
            custom_border: Override for border style
            custom_alignment: Override for text alignment
            custom_effects: Override for special effects
            
        Returns:
            Styled ASCII art
        """
        # Get style configuration (preset or custom)
        if style in self.STYLE_PRESETS:
            style_config = self.STYLE_PRESETS[style].copy()
        else:
            from ..utils.ascii_utils import resolve_style
            style_config = resolve_style(style)
        
        # Apply overrides if provided
        if custom_padding:
            style_config["padding"] = custom_padding
        if custom_border:
            style_config["border"] = custom_border  
        if custom_alignment:
            style_config["alignment"] = custom_alignment
        if custom_effects:
            style_config["effects"] = custom_effects
            
        # Process the art with each styling component
        processed_art = ascii_art
        
        # Apply alignment
        processed_art = self._align_text(processed_art, style_config["alignment"])
        
        # Apply special effects
        for effect in style_config["effects"]:
            processed_art = self._apply_effect(processed_art, effect)
        
        # Apply padding
        if style_config["padding"]:
            v_pad, h_pad = style_config["padding"]
            processed_art = self._apply_padding(processed_art, v_pad, h_pad)
            
        # Apply border
        if style_config["border"]:
            processed_art = self._apply_border(processed_art, style_config["border"])
            
        return processed_art
    
    def _align_text(self, text: str, alignment: str) -> str:
        """Align text to left, center, or right."""
        if alignment not in ["left", "center", "right"]:
            return text
            
        lines = text.split('\n')
        max_length = max(len(line) for line in lines)
        aligned_lines = []
        
        for line in lines:
            if alignment == "center":
                aligned_lines.append(line.center(max_length))
            elif alignment == "right":
                aligned_lines.append(line.rjust(max_length))
            else:  # left alignment
                aligned_lines.append(line)
                
        return '\n'.join(aligned_lines)
    
    def _apply_padding(self, text: str, v_pad: int, h_pad: int) -> str:
        """Apply vertical and horizontal padding to text."""
        lines = text.split('\n')
        padded_lines = []
        
        # Add vertical padding (top)
        padded_lines.extend([''] * v_pad)
        
        # Add horizontal padding to each line
        for line in lines:
            padded_lines.append(' ' * h_pad + line + ' ' * h_pad)
            
        # Add vertical padding (bottom)
        padded_lines.extend([''] * v_pad)
        
        return '\n'.join(padded_lines)
    
    def _apply_border(self, text: str, border_style: str) -> str:
        """Apply border around text with specified style."""
        if border_style not in self.BORDER_SETS:
            return text
            
        lines = text.split('\n')
        width = max(len(line) for line in lines)
        
        # Get border characters
        tl, tr, br, bl, v, h = self.BORDER_SETS[border_style]
        
        # If using Unicode and terminal doesn't support it, fall back to ASCII
        if not self._unicode_supported and any(ord(c) > 127 for c in self.BORDER_SETS[border_style]):
            tl, tr, br, bl, v, h = "++-+||"  # ASCII fallback
        
        bordered_lines = []
        # Top border
        bordered_lines.append(tl + h * width + tr)
        
        # Content with side borders
        for line in lines:
            bordered_lines.append(v + line.ljust(width) + v)
            
        # Bottom border
        bordered_lines.append(bl + h * width + br)
        
        return '\n'.join(bordered_lines)
    
    def _apply_effect(self, text: str, effect: str) -> str:
        """Apply special effects to text."""
        if effect == "shadow":
            return self._add_shadow(text)
        elif effect == "glow":
            return self._add_glow(text)
        elif effect == "emboss":
            return self._add_emboss(text)
        elif effect == "digital":
            return self._add_digital_effect(text)
        elif effect == "fade":
            return self._add_fade(text)
        return text
    
    def _add_shadow(self, text: str) -> str:
        """Add a shadow effect to text."""
        lines = text.split('\n')
        result_lines = []
        
        shadow_chars = self.EFFECT_CHARS.get("shadow", "░▒▓█")
        primary_shadow = shadow_chars[0] if shadow_chars else "░"
        
        for i, line in enumerate(lines):
            if i < len(lines) - 1 and i > 0:  # Skip first and last line for shadow
                shadow_line = lines[i+1]
                
                # Create shadow by replacing spaces with shadow chars
                new_shadow = ""
                for j, char in enumerate(line):
                    if char != ' ' and j < len(shadow_line) and shadow_line[j] == ' ':
                        new_shadow += primary_shadow 
                    else:
                        new_shadow += shadow_line[j] if j < len(shadow_line) else ' '
                
                lines[i+1] = new_shadow
                
            result_lines.append(line)
            
        return '\n'.join(result_lines)
    
    def _add_glow(self, text: str) -> str:
        """Add a glow effect around characters."""
        lines = text.split('\n')
        result = []
        
        glow_chars = self.EFFECT_CHARS.get("glow", "✦✧✨⋆⭐")
        
        for line in lines:
            new_line = ""
            for i, char in enumerate(line):
                if char != ' ':
                    # Apply glow with special characters surrounding content
                    new_line += glow_chars[i % len(glow_chars)] if i % 3 == 0 else char
                else:
                    new_line += char
            result.append(new_line)
            
        return '\n'.join(result)
    
    def _add_fade(self, text: str) -> str:
        """Add a gradient fade effect to text."""
        lines = text.split('\n')
        result = []
        
        # Skip if no lines
        if not lines:
            return text
            
        total_lines = len(lines)
        fade_chars = " ░▒▓█"
        
        for i, line in enumerate(lines):
            # Calculate fade intensity based on line position
            # More intense at top, fading toward bottom
            fade_position = i / total_lines
            fade_idx = min(int(fade_position * len(fade_chars)), len(fade_chars) - 1)
            fade_char = fade_chars[fade_idx]
            
            # For each line, add some fade characters at the beginning
            fade_prefix = fade_char * min(3, max(1, int(5 * (1 - fade_position))))
            
            # Add faded line
            result.append(fade_prefix + line)
            
        return '\n'.join(result)
    
    def _add_emboss(self, text: str) -> str:
        """Add an embossed 3D effect to text."""
        lines = text.split('\n')
        result = []
        
        for i, line in enumerate(lines):
            # Create a "highlight" by adding lighter characters on top
            if i > 0 and i < len(lines):
                # Add highlight to non-space characters
                highlighted = ""
                for j, char in enumerate(line):
                    if char != ' ' and (j == 0 or line[j-1] == ' '):
                        highlighted += '/'  # Highlight character
                    else:
                        highlighted += char
                result.append(highlighted)
            else:
                result.append(line)
        
        return '\n'.join(result)
    
    def _add_digital_effect(self, text: str) -> str:
        """Add a digital/matrix effect to text."""
        lines = text.split('\n')
        result = []
        
        digital_chars = self.EFFECT_CHARS.get("digital", "01")
        
        for line in lines:
            digital_line = ""
            for i, char in enumerate(line):
                if char != ' ':
                    # Replace with binary characters based on position
                    digital_line += digital_chars[i % len(digital_chars)]
                else:
                    digital_line += ' '
            result.append(digital_line)
            
        return '\n'.join(result)
    
    def _apply_color(self, text: str) -> str:
        """Apply ANSI color codes to text."""
        # Define ANSI color codes
        colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'red': '\033[31m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'magenta': '\033[35m',
            'cyan': '\033[36m'
        }
        
        # Apply different colors to different parts
        colored_text = f"{colors['bold']}{colors['cyan']}{text}{colors['reset']}"
        return colored_text
        
    def available_fonts(self) -> List[str]:
        """
        Return a list of all available FIGlet fonts.
        
        Returns:
            Alphabetically sorted list of font names
        """
        return sorted(self.figlet.getFonts())
    
    def preview_fonts(self, text: str = "ASCII Forge", limit: int = 5) -> str:
        """
        Generate previews of multiple fonts using sample text.
        
        Args:
            text: Sample text to render (default: "ASCII Forge")
            limit: Maximum number of fonts to preview (default: 5)
            
        Returns:
            String containing previews of fonts
        """
        previews = []
        fonts = self.available_fonts()[:limit]
        
        for font in fonts:
            try:
                temp_figlet = Figlet(font=font, width=self.width)
                preview = temp_figlet.renderText(text)
                previews.append(f"Font: {font}\n{'-' * 40}\n{preview}\n")
            except Exception as e:
                previews.append(f"Font: {font} (Error: {str(e)})\n")
                
        return "\n".join(previews)
    
    def render_template(self, template: str, variables: Dict[str, str]) -> str:
        """
        Render a banner template with variable substitution.
        
        Args:
            template: Template string with placeholders
            variables: Dictionary of variable values
            
        Returns:
            Processed template with substituted variables
        """
        result = template
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, value)
        return self.generate(result)
    
    def get_metrics(self) -> Dict[str, int]:
        """
        Return performance metrics from the banner generator.
        
        Returns:
            Dictionary containing operational metrics
        """
        cache_hit_rate = 0
        if self._render_count > 0:
            cache_hit_rate = int((self._cache_hits / self._render_count) * 100)
            
        return {
            "total_renders": self._render_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "unique_banners": len(self.cache) if self.cache else 0
        }
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics to zero."""
        self._render_count = 0
        self._cache_hits = 0
        if self.cache is not None:
            self.cache.clear()
        if self.cache_timestamps is not None:
            self.cache_timestamps.clear()


# Execute self-diagnostic when module is run directly
if __name__ == "__main__":
    print("🔥 ASCII Forge Banner Generator Self-Test 🔥")
    banner = BannerGenerator(font="slant", width=80)
    
    # Generate sample banner with default settings
    print(banner.generate("ASCII FORGE", style="boxed"))
    
    # Show available fonts
    print(f"Available fonts: {len(banner.available_fonts())}")
    print(banner.available_fonts()[:10])
    
    # Performance metrics
    print(banner.get_metrics())