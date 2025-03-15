from PIL import Image
from typing import Optional, Union, List, Tuple, TypeVar
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from numpy.typing import NDArray
import os
import shutil
import logging
from enum import Enum

from ..utils.alphabet_manager import AlphabetManager

# Type definitions for clarity and precision
PixelArray = NDArray[np.uint8]  # Type for grayscale/RGB pixel arrays
Shape = Tuple[int, ...]         # Array dimensions
T = TypeVar('T')                # Generic type for flexible functions
AsciiRow = List[str]            # Type for rows of ASCII characters
AsciiArt = List[str]            # Type for complete ASCII art (list of strings)

class ColorMode(Enum):
    """Supported color output formats."""
    ANSI = "ansi"  # Terminal-compatible ANSI color sequences
    HTML = "html"  # Web-compatible HTML color styling
    NONE = "none"  # Fallback to standard grayscale


class ImageAsciiConverter:
    """
    # ImageAsciiConverter
    A high-performance image-to-ASCII art converter that transforms visual data into textual representations with precision and flexibility.
    ## Overview
    `ImageAsciiConverter` provides comprehensive functionality to convert images into ASCII art using various character sets, processing techniques, and output formats. The converter supports grayscale and color output, with options for adjusting dimensions, brightness, contrast, and applying effects like dithering.
    ## Features
    - **Multiple character sets** - Use built-in or custom character sets for different artistic styles
    - **Adaptive rendering** - Maintains aspect ratio and can auto-scale to terminal dimensions
    - **Multi-threaded processing** - Parallel conversion for large images with configurable thread count
    - **Image adjustments** - Controls for brightness, contrast, and dithering
    - **Color output** - Support for ANSI (terminal) and HTML color formats
    - **Styling options** - Apply visual styles to the resulting ASCII art
    ## Usage
    ⚡ Hyper-optimized image-to-ASCII converter with Eidosian principles ⚡
    
    Transforms visual data into textual art with surgical precision.
    Features adaptive rendering, multi-threaded processing, and specialized character sets.
    """
    
    def __init__(self, 
                 charset: str = "general", 
                 width: int = 100, 
                 height: Optional[int] = None, 
                 invert: bool = False,
                 brightness: float = 1.0,
                 contrast: float = 1.0,
                 auto_scale: bool = True,
                 dithering: bool = False,
                 threads: int = 0):
        """
        Initialize the image converter with specified settings.
        
        Args:
            charset: Name of character set to use or custom charset string
            width: Width of output ASCII art in characters
            height: Optional height (maintains aspect ratio if None)
            invert: Whether to invert the brightness of the output
            brightness: Brightness adjustment factor (0.0-2.0)
            contrast: Contrast adjustment factor (0.0-2.0)
            auto_scale: Automatically scale output to terminal size
            dithering: Apply dithering for improved visual quality
            threads: Number of threads for parallel processing (0=auto)
        """
        # Get the appropriate charset
        self._available_charsets = AlphabetManager.list_available_alphabets()
        self.charset = (AlphabetManager.get_alphabet(charset) 
                        if charset in self._available_charsets else charset)
        
        # Configure core attributes with bounds checking
        self.width = max(1, width)
        self.height = max(1, height) if height is not None else None
        self.brightness = max(0.0, min(2.0, brightness))
        self.contrast = max(0.0, min(2.0, contrast))
        self.auto_scale = auto_scale
        self.dithering = dithering
        self.threads = threads if threads > 0 else max(1, os.cpu_count() or 1)
        
        # Apply inversion if needed
        if invert:
            self.charset = self.charset[::-1]
        
        # Generate character density mapping
        self.density_map = AlphabetManager.create_density_map(self.charset)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    def convert(self, 
                image_path: Union[str, Image.Image], 
                output_path: Optional[str] = None,
                style: Optional[str] = None) -> str:
        """
        Convert an image to ASCII art with advanced processing.
        
        Args:
            image_path: Path to the image file or PIL Image object
            output_path: Optional path to save the ASCII art
            style: Optional style to apply to the output
            
        Returns:
            ASCII art as a string
        """
        try:
            # Load image (handle both file paths and PIL Image objects)
            img = self._load_image(image_path)
            
            # Process the image
            ascii_art = self._process_image(img, style)
            
            # Save to file if requested
            if output_path:
                self._save_to_file(ascii_art, output_path)
                self.logger.info(f"ASCII art saved to: {output_path}")
            
            return ascii_art
            
        except Exception as e:
            self.logger.error(f"Error converting image: {str(e)}", exc_info=True)
            return f"Error converting image: {str(e)}"
    
    def _load_image(self, image_path: Union[str, Image.Image]) -> Image.Image:
        """Load and prepare image for processing."""
        if isinstance(image_path, str):
            img = Image.open(image_path).convert('L')
            self.logger.info(f"Image loaded: {image_path} [{img.width}x{img.height}]")
        else:
            # Already a PIL Image
            img = image_path.convert('L')
            self.logger.info(f"Using provided PIL image [{img.width}x{img.height}]")
        
        return img
    
    def _process_image(self, img: Image.Image, style: Optional[str] = None) -> str:
        """Process image through the complete conversion pipeline."""
        # Calculate new dimensions
        orig_width, orig_height = img.size
        aspect_ratio = orig_height / orig_width
        
        # Set output dimensions, maintaining aspect ratio
        new_width = self.width
        # Character aspect ratio correction factor (chars are taller than wide)
        char_aspect = 0.55
        new_height = self.height if self.height else int(aspect_ratio * new_width * char_aspect)
        
        # Auto-scale to terminal size if requested
        if self.auto_scale:
            new_width, new_height = self._apply_terminal_scaling(new_width, new_height)
            
        # Resize image with high quality resampling
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Apply brightness/contrast adjustments if needed
        if self.brightness != 1.0 or self.contrast != 1.0:
            img = self._apply_image_adjustments(img)
        
        # Apply dithering if enabled
        if self.dithering:
            img = img.convert('1', dither=Image.Dither.FLOYDSTEINBERG)
            img = img.convert('L')  # Convert back to grayscale
        
        # Convert to numpy array for faster processing
        pixels = np.array(img)
        
        # Generate ASCII art (with parallel processing for large images)
        if new_height > 100 and self.threads > 1:
            # Process rows in parallel
            ascii_art = self._parallel_conversion(pixels)
        else:
            # Single-threaded processing
            ascii_art = self._convert_pixels(pixels)
        
        # Apply style if requested
        if style:
            from ..core.style_manager import apply_style
            ascii_art = apply_style(ascii_art, style_name=style)
        
        return ascii_art
    
    def _apply_terminal_scaling(self, new_width: int, new_height: int) -> tuple[int, int]:
        """Scale dimensions to fit the terminal window."""
        try:
            # Get terminal dimensions
            term_size = shutil.get_terminal_size()
            term_width, term_height = term_size.columns, term_size.lines
            
            # Apply constraints based on terminal size
            term_width = max(20, min(term_width - 2, 200))  # Practical limits
            term_height = max(10, min(term_height - 3, 100))  # Leave space for prompt
            
            # Don't exceed terminal width
            if new_width > term_width:
                scale_factor = term_width / new_width
                new_width = term_width
                new_height = int(new_height * scale_factor)
            
            # Don't exceed terminal height (with higher weight)
            if new_height > term_height:
                scale_factor = term_height / new_height
                new_height = term_height
                new_width = int(new_width * scale_factor)
                
            self.logger.debug(f"Terminal-scaled dimensions: {new_width}x{new_height}")
            return new_width, new_height
        except Exception as e:
            self.logger.warning(f"Failed to apply terminal scaling: {e}")
            return new_width, new_height
    
    def _apply_image_adjustments(self, img: Image.Image) -> Image.Image:
        """Apply brightness and contrast adjustments to the image."""
        pixels = np.array(img)
        pixels = pixels.astype(np.float32)
        
        # Apply contrast first
        if self.contrast != 1.0:
            pixels = (pixels - 128) * self.contrast + 128
        
        # Then brightness
        if self.brightness != 1.0:
            pixels = pixels * self.brightness
        
        # Clip values to valid range
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
        
        # Create new image from adjusted array
        return Image.fromarray(pixels)
        
    def _convert_pixels(self, pixels: PixelArray) -> str:
        """
        Convert pixel array to ASCII art (single-threaded implementation).
        
        Args:
            pixels: Numpy array of grayscale pixel values
            
        Returns:
            ASCII art string
        """
        ascii_art: AsciiArt = []
        for row in pixels:
            ascii_row = "".join(self.density_map[int(pixel_value)] for pixel_value in row)
            ascii_art.append(ascii_row)
        
        return "\n".join(ascii_art)
    
    def _parallel_conversion(self, pixels: PixelArray) -> str:
        """
        Convert pixel array to ASCII art using parallel processing.
        
        Args:
            pixels: Numpy array of grayscale pixel values
            
        Returns:
            ASCII art string
        """
        chunk_size = max(1, len(pixels) // self.threads)
        chunks = [pixels[i:i+chunk_size] for i in range(0, len(pixels), chunk_size)]
        
        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            results = list(executor.map(self._convert_pixels, chunks))
            
        return "\n".join(results)
    
    def _save_to_file(self, ascii_art: str, output_path: str) -> None:
        """Save ASCII art to a file with proper directory creation."""
        try:
            # Ensure directory exists
            dirname = os.path.dirname(output_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
                
            # Write with UTF-8 encoding for maximum compatibility
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(ascii_art)
                
            self.logger.debug(f"Saved output to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save output: {e}")
            raise IOError(f"Failed to save output: {str(e)}")
    
    def set_charset(self, charset: str, invert: bool = False) -> None:
        """
        Change the character set used for conversion.
        
        Args:
            charset: Name of preset charset or custom string
            invert: Whether to invert the brightness
        """
        self.charset = AlphabetManager.get_alphabet(charset) if charset in self._available_charsets else charset
        if invert:
            self.charset = self.charset[::-1]
            
        self.density_map = AlphabetManager.create_density_map(self.charset)
        
    def set_image_params(self, 
                         width: Optional[int] = None,
                         height: Optional[int] = None,
                         brightness: Optional[float] = None,
                         contrast: Optional[float] = None,
                         dithering: Optional[bool] = None) -> None:
        """
        Update image conversion parameters.
        
        Args:
            width: New width in characters
            height: New height in characters
            brightness: New brightness adjustment factor
            contrast: New contrast adjustment factor
            dithering: Enable/disable dithering
        """
        if width is not None:
            self.width = max(1, width)
            
        if height is not None:
            self.height = max(1, height) if height > 0 else None
            
        if brightness is not None:
            self.brightness = max(0.0, min(2.0, brightness))
            
        if contrast is not None:
            self.contrast = max(0.0, min(2.0, contrast))
            
        if dithering is not None:
            self.dithering = dithering
    
    def get_available_charsets(self) -> List[str]:
        """
        Get list of available character sets.
        
        Returns:
            List of available charset names
        """
        return self._available_charsets.copy()
    
    def convert_color(self, 
                      image_path: Union[str, Image.Image], 
                      output_path: Optional[str] = None,
                      color_mode: str = "ansi") -> str:
        """
        Convert image to color ASCII art using ANSI or HTML color codes.
        
        Args:
            image_path: Path to image or PIL Image object
            output_path: Optional path to save the output
            color_mode: Color output format ("ansi", "html", or "none")
            
        Returns:
            ASCII art with color formatting
        """
        try:
            # Load image
            if isinstance(image_path, str):
                img = Image.open(image_path).convert('RGB')
            elif hasattr(image_path, 'convert') and callable(getattr(image_path, 'convert')):
                img = image_path.convert('RGB')
            else:
                return f"Error: image_path must be a string path or PIL Image object"
            
            # Calculate dimensions
            orig_width, orig_height = img.size
            aspect_ratio = orig_height / orig_width
            
            # Set output dimensions
            new_width = self.width
            char_aspect = 0.55
            new_height = self.height if self.height else int(aspect_ratio * new_width * char_aspect)
            
            # Auto-scale if requested
            if self.auto_scale:
                new_width, new_height = self._apply_terminal_scaling(new_width, new_height)
            
            # Resize image
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to grayscale for character selection
            gray_img = img.convert('L')
            
            # Get both color and grayscale pixels
            pixels_rgb = np.array(img)
            pixels_gray = np.array(gray_img)
            
            # Generate color ASCII art based on mode
            if color_mode.lower() == "ansi":
                ascii_art = self._generate_ansi_color(pixels_rgb, pixels_gray)
            elif color_mode.lower() == "html":
                ascii_art = self._generate_html_color(pixels_rgb, pixels_gray)
            else:
                # Fallback to standard grayscale conversion
                return self.convert(gray_img, output_path)
            
            # Save to file if requested
            if output_path:
                self._save_to_file(ascii_art, output_path)
            
            return ascii_art
            
        except Exception as e:
            self.logger.error(f"Color conversion error: {e}", exc_info=True)
            return f"Error converting color image: {str(e)}"
    
    def _generate_ansi_color(self, pixels_rgb: PixelArray, pixels_gray: PixelArray) -> str:
        """Generate ASCII art with ANSI color codes."""
        ascii_art: AsciiArt = []
        for y in range(len(pixels_gray)):
            row: AsciiRow = []
            for x in range(len(pixels_gray[y])):
                # Get character based on brightness
                char = self.density_map[int(pixels_gray[y][x])]
                # Get RGB color
                r, g, b = pixels_rgb[y][x]
                # Create ANSI color sequence
                color_code = f"\033[38;2;{r};{g};{b}m{char}\033[0m"
                row.append(color_code)
            ascii_art.append("".join(row))
        return "\n".join(ascii_art)
    
    def _generate_html_color(self, pixels_rgb: PixelArray, pixels_gray: PixelArray) -> str:
        """Generate ASCII art with HTML color tags."""
        ascii_art: AsciiArt = ["<pre style='line-height:1; letter-spacing:0'>"]
        for y in range(len(pixels_gray)):
            row_parts: List[str] = []
            for x in range(len(pixels_gray[y])):
                # Get character based on brightness
                char = self.density_map[int(pixels_gray[y][x])]
                # Get RGB color
                r, g, b = pixels_rgb[y][x]
                # Create HTML span with color
                color_hex = f"#{r:02x}{g:02x}{b:02x}"
                row_parts.append(f"<span style='color:{color_hex}'>{char}</span>")
            
            # Join row and add line break
            ascii_art.append("".join(row_parts))
            ascii_art.append("<br>")
        
        # Close container
        ascii_art.append("</pre>")
        
        return "".join(ascii_art)