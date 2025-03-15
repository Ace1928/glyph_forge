from typing import Dict, Any, List, Optional, Union, Tuple
import os
import logging

from ..core.banner_generator import BannerGenerator
from ..core.style_manager import get_available_styles
from ..services.image_to_ascii import ImageAsciiConverter
from ..utils.alphabet_manager import AlphabetManager
from ..config.settings import get_config

logger = logging.getLogger(__name__)

class ASCIIForgeAPI:
    """
    Public API for the ASCII Forge library.
    
    Provides a streamlined, unified interface to all ASCII Forge capabilities
    with intelligent caching, configuration management, and error handling.
    """
    
    def __init__(self):
        """Initialize the API with configuration and core components."""
        logger.debug("Initializing ASCII Forge API")
        self.config = get_config()
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize core components with optimal laziness."""
        # Default banner generator
        default_font = self.config.get('banner', 'default_font', 'slant')
        default_width = self.config.get('banner', 'default_width', 80)
        self._banner_generator = BannerGenerator(font=default_font, width=default_width)
        
        # Image converter will be initialized on first use (lazy loading)
        self._image_converter = None
        
        logger.debug(f"Core components initialized with font='{default_font}', width={default_width}")
    
    def _get_image_converter(self) -> ImageAsciiConverter:
        """Get or lazily initialize image converter."""
        if self._image_converter is None:
            # Default image converter settings
            default_charset = self.config.get('image', 'default_charset', 'general')
            default_width = self.config.get('image', 'default_width', 100)
            self._image_converter = ImageAsciiConverter(
                charset=default_charset,
                width=default_width
            )
            logger.debug(f"Image converter initialized with charset='{default_charset}', width={default_width}")
        
        return self._image_converter
    
    def generate_banner(self, 
                       text: str, 
                       style: Optional[str] = None, 
                       font: Optional[str] = None,
                       width: Optional[int] = None,
                       effects: Optional[List[str]] = None,
                       color: bool = False) -> str:
        """
        Generate an ASCII art banner from text with intelligent parameter handling.
        
        Args:
            text: Text to convert into banner
            style: Style preset to apply (default from config)
            font: Font to use (default from config)
            width: Width for the banner (default from config)
            effects: Special effects to apply (default from style)
            color: Whether to apply ANSI color to output
        
        Returns:
            ASCII art banner
        """
        # Use defaults from config if not specified
        if style is None:
            style = self.config.get('banner', 'default_style', 'minimal')
        
        # Regenerate banner generator if font or width changed
        if font is not None or width is not None:
            temp_font = font if font is not None else self._banner_generator.font
            temp_width = width if width is not None else self._banner_generator.width
            generator = BannerGenerator(font=temp_font, width=temp_width)
            return generator.generate(text, style=style, effects=effects, color=color)
        
        # Use existing generator
        return self._banner_generator.generate(
            text=text, 
            style=style,
            effects=effects,
            color=color
        )
    
    def image_to_ascii(self, 
                      image_path: str, 
                      output_path: Optional[str] = None,
                      charset: Optional[str] = None, 
                      width: Optional[int] = None, 
                      height: Optional[int] = None,
                      invert: bool = False,
                      brightness: Optional[float] = None,
                      contrast: Optional[float] = None,
                      dithering: bool = False,
                      color_mode: str = "none") -> str:
        """
        Convert an image to ASCII art with comprehensive parameter support.
        
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
            color_mode: Color output mode ("none", "ansi", "html")
        
        Returns:
            ASCII art representation of the image
        """
        # Get or create image converter
        converter = self._get_image_converter()
        
        # Apply any parameter overrides
        if charset is not None or width is not None or height is not None or invert:
            # Create a new converter with specified parameters
            temp_charset = charset if charset is not None else converter.charset
            temp_width = width if width is not None else converter.width
            converter = ImageAsciiConverter(
                charset=temp_charset,
                width=temp_width,
                height=height,
                invert=invert,
                dithering=dithering
            )
        
        # Set optional brightness and contrast
        if brightness is not None or contrast is not None:
            converter.set_image_params(
                brightness=brightness or converter.brightness,
                contrast=contrast or converter.contrast
            )
        
        # Convert with or without color
        if color_mode.lower() in ("ansi", "html"):
            return converter.convert_color(
                image_path=image_path, 
                output_path=output_path,
                color_mode=color_mode.lower()
            )
        else:
            return converter.convert(
                image_path=image_path, 
                output_path=output_path
            )
    
    def get_available_fonts(self) -> List[str]:
        """
        Get a list of available font names.
        
        Returns:
            List of available font names
        """
        return self._banner_generator.available_fonts()
    
    def get_available_styles(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a dictionary of available style presets.
        
        Returns:
            Dictionary mapping style names to their configurations
        """
        return get_available_styles()
    
    def get_available_alphabets(self) -> List[str]:
        """
        Get a list of available character sets/alphabets.
        
        Returns:
            List of available alphabet names
        """
        return AlphabetManager.list_available_alphabets()
    
    def save_to_file(self, ascii_art: str, file_path: str) -> bool:
        """
        Save ASCII art to a file with proper directory creation.
        
        Args:
            ascii_art: ASCII art text to save
            file_path: Path to save the file
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Write file with UTF-8 encoding for maximum compatibility
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(ascii_art)
                
            logger.debug(f"Saved ASCII art to {file_path}")
            return True
        except (IOError, OSError) as e:
            logger.error(f"Failed to save file: {str(e)}")
            return False
    
    def preview_font(self, font: str, text: str = "ASCII Forge") -> str:
        """
        Generate a preview of a specific font.
        
        Args:
            font: Name of the font to preview
            text: Text to use for preview
        
        Returns:
            ASCII art using specified font
        """
        generator = BannerGenerator(font=font, width=self._banner_generator.width)
        return generator.generate(text)
    
    def preview_style(self, style: str, text: str = "ASCII Forge") -> str:
        """
        Generate a preview of a specific style.
        
        Args:
            style: Name of the style to preview
            text: Text to use for preview
        
        Returns:
            ASCII art using specified style
        """
        return self._banner_generator.generate(text, style=style)
    
    def convert_text_to_art(self, text: str, font: str = "standard") -> str:
        """
        Convert plain text to ASCII art without additional styling.
        
        Args:
            text: Text to convert
            font: Font to use
        
        Returns:
            ASCII art representation
        """
        generator = BannerGenerator(font=font, width=self._banner_generator.width)
        return generator.figlet.renderText(text)


# Singleton API instance
_api_instance = None

def get_api() -> ASCIIForgeAPI:
    """
    Get the ASCIIForgeAPI singleton instance with zero redundant initialization.
    
    Returns:
        ASCIIForgeAPI instance
    """
    global _api_instance
    if _api_instance is None:
        _api_instance = ASCIIForgeAPI()
    return _api_instance