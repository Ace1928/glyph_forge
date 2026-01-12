"""
⚡ Glyph Forge Transformer Classes ⚡

Transformer classes for converting various input formats into GlyphMatrix
representations. These transformers implement the core conversion logic
for the glyph art generation pipeline.

Key transformers:
- ImageTransformer: Convert images to glyph matrices
- ColorMapper: Map colors to glyph characters
- DepthAnalyzer: Analyze image depth for enhanced detail
- EdgeDetector: Detect edges for structural emphasis
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# Type definitions
GlyphMatrix = List[List[str]]
PixelArray = NDArray[np.uint8]


class ImageTransformer:
    """
    Transform images into glyph matrices with intelligent character mapping.

    This transformer converts image data into a 2D matrix of characters,
    where each character represents the luminance/density of the corresponding
    pixel region.

    Attributes:
        charset: Characters ordered from lightest to darkest
        density_map: Mapping from brightness values to characters
    """

    # Default character set (light to dark)
    DEFAULT_CHARSET = " .:-=+*#%@"

    def __init__(
        self,
        charset: str = DEFAULT_CHARSET,
        invert: bool = False,
    ) -> None:
        """
        Initialize the ImageTransformer.

        Args:
            charset: String of characters from light to dark density
            invert: If True, reverse the density mapping
        """
        self.charset = charset[::-1] if invert else charset
        self.density_map = self._create_density_map(self.charset)
        logger.debug(
            f"ImageTransformer initialized with {len(charset)} characters"
        )

    def _create_density_map(self, charset: str) -> Dict[int, str]:
        """Create mapping from brightness (0-255) to character."""
        mapping: Dict[int, str] = {}
        length = len(charset)
        for i in range(256):
            idx = min(int(i * length / 256), length - 1)
            mapping[i] = charset[idx]
        return mapping

    def transform(
        self,
        source: Union[str, Image.Image, PixelArray],
        width: int = 80,
        height: Optional[int] = None,
        **options: Any,
    ) -> GlyphMatrix:
        """
        Transform an image into a glyph matrix.

        Args:
            source: Image path, PIL Image, or numpy array
            width: Target width in characters
            height: Target height in characters (auto-calculated if None)
            **options: Additional options:
                - brightness: Brightness multiplier (default: 1.0)
                - contrast: Contrast multiplier (default: 1.0)
                - aspect_ratio: Character aspect ratio (default: 0.55)

        Returns:
            2D matrix of characters representing the image
        """
        # Load and prepare image
        img = self._load_image(source)

        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')

        # Calculate dimensions
        orig_width, orig_height = img.size
        aspect_ratio = options.get('aspect_ratio', 0.55)

        if height is None:
            height = int((orig_height / orig_width) * width * aspect_ratio)

        # Resize image
        img = img.resize((width, height), Image.Resampling.LANCZOS)

        # Apply adjustments
        brightness = options.get('brightness', 1.0)
        contrast = options.get('contrast', 1.0)
        if brightness != 1.0 or contrast != 1.0:
            img = self._adjust_image(img, brightness, contrast)

        # Convert to numpy array
        pixels = np.array(img)

        # Map pixels to characters
        matrix: GlyphMatrix = []
        for row in pixels:
            char_row = [self.density_map[int(pixel)] for pixel in row]
            matrix.append(char_row)

        logger.debug(f"Transformed image to {width}x{height} matrix")
        return matrix

    def _load_image(
        self, source: Union[str, Image.Image, PixelArray]
    ) -> Image.Image:
        """Load image from various sources."""
        if isinstance(source, str):
            return Image.open(source)
        elif isinstance(source, np.ndarray):
            return Image.fromarray(source)
        elif isinstance(source, Image.Image):
            return source
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

    def _adjust_image(
        self, img: Image.Image, brightness: float, contrast: float
    ) -> Image.Image:
        """Apply brightness and contrast adjustments."""
        pixels = np.array(img, dtype=np.float32)

        # Apply contrast
        if contrast != 1.0:
            pixels = (pixels - 128) * contrast + 128

        # Apply brightness
        if brightness != 1.0:
            pixels = pixels * brightness

        # Clip to valid range
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
        return Image.fromarray(pixels)


class ColorMapper:
    """
    Map colors to glyph characters with color-aware density mapping.

    This transformer extends basic grayscale mapping by considering
    color information when selecting characters.

    Attributes:
        charset: Characters for density mapping
        color_weights: RGB weights for luminance calculation
    """

    # ITU-R BT.601 standard weights
    DEFAULT_WEIGHTS = (0.299, 0.587, 0.114)

    def __init__(
        self,
        charset: str = " .:-=+*#%@",
        color_weights: Tuple[float, float, float] = DEFAULT_WEIGHTS,
    ) -> None:
        """
        Initialize the ColorMapper.

        Args:
            charset: Characters from light to dark
            color_weights: RGB weights for luminance calculation
        """
        self.charset = charset
        self.color_weights = color_weights
        self.density_map = self._create_density_map(charset)
        logger.debug("ColorMapper initialized")

    def _create_density_map(self, charset: str) -> Dict[int, str]:
        """Create mapping from luminance to character."""
        mapping: Dict[int, str] = {}
        length = len(charset)
        for i in range(256):
            idx = min(int(i * length / 256), length - 1)
            mapping[i] = charset[idx]
        return mapping

    def transform(
        self,
        source: Union[str, Image.Image, PixelArray],
        width: int = 80,
        height: Optional[int] = None,
        **options: Any,
    ) -> GlyphMatrix:
        """
        Transform a color image into a glyph matrix using weighted luminance.

        Args:
            source: Image path, PIL Image, or numpy array
            width: Target width in characters
            height: Target height in characters
            **options: Additional options:
                - preserve_saturation: Weight highly saturated colors
                - aspect_ratio: Character aspect ratio

        Returns:
            2D matrix of characters
        """
        # Load image
        img = self._load_image(source)

        # Ensure RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Calculate dimensions
        orig_width, orig_height = img.size
        aspect_ratio = options.get('aspect_ratio', 0.55)

        if height is None:
            height = int((orig_height / orig_width) * width * aspect_ratio)

        # Resize image
        img = img.resize((width, height), Image.Resampling.LANCZOS)

        # Convert to array
        pixels = np.array(img)

        # Calculate weighted luminance
        r, g, b = self.color_weights
        luminance = (
            pixels[:, :, 0] * r +
            pixels[:, :, 1] * g +
            pixels[:, :, 2] * b
        ).astype(np.uint8)

        # Optionally weight by saturation
        if options.get('preserve_saturation', False):
            luminance = self._apply_saturation_weighting(pixels, luminance)

        # Map to characters
        matrix: GlyphMatrix = []
        for row in luminance:
            char_row = [self.density_map[int(pixel)] for pixel in row]
            matrix.append(char_row)

        return matrix

    def _load_image(
        self, source: Union[str, Image.Image, PixelArray]
    ) -> Image.Image:
        """Load image from various sources."""
        if isinstance(source, str):
            return Image.open(source)
        elif isinstance(source, np.ndarray):
            return Image.fromarray(source)
        elif isinstance(source, Image.Image):
            return source
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

    def _apply_saturation_weighting(
        self,
        rgb_pixels: PixelArray,
        luminance: PixelArray,
    ) -> PixelArray:
        """Adjust luminance based on color saturation."""
        # Calculate saturation (simplified HSV-style)
        rgb_max = np.max(rgb_pixels, axis=2)
        rgb_min = np.min(rgb_pixels, axis=2)

        # Saturation = (max - min) / max (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            saturation = np.where(
                rgb_max > 0,
                (rgb_max - rgb_min) / rgb_max,
                0
            )

        # Boost luminance for highly saturated colors
        adjustment = 1.0 + saturation * 0.3
        adjusted = (luminance * adjustment).astype(np.uint8)
        return np.clip(adjusted, 0, 255).astype(np.uint8)


class DepthAnalyzer:
    """
    Analyze image depth for enhanced glyph detail selection.

    Uses local contrast and texture analysis to determine depth
    perception, allowing for more nuanced character selection.

    This is particularly useful for images with foreground/background
    separation where depth cues can improve the visual result.
    """

    def __init__(self, charset: str = " .:-=+*#%@") -> None:
        """
        Initialize the DepthAnalyzer.

        Args:
            charset: Characters from light to dark
        """
        self.charset = charset
        self.density_map = self._create_density_map(charset)
        logger.debug("DepthAnalyzer initialized")

    def _create_density_map(self, charset: str) -> Dict[int, str]:
        """Create mapping from depth value to character."""
        mapping: Dict[int, str] = {}
        length = len(charset)
        for i in range(256):
            idx = min(int(i * length / 256), length - 1)
            mapping[i] = charset[idx]
        return mapping

    def transform(
        self,
        source: Union[str, Image.Image, PixelArray],
        width: int = 80,
        height: Optional[int] = None,
        **options: Any,
    ) -> GlyphMatrix:
        """
        Transform an image using depth analysis for enhanced detail.

        Args:
            source: Image path, PIL Image, or numpy array
            width: Target width in characters
            height: Target height in characters
            **options: Additional options:
                - depth_weight: Weight for depth influence (0.0-1.0)
                - blur_radius: Blur radius for depth estimation

        Returns:
            2D matrix of characters with depth-enhanced selection
        """
        # Load image
        img = self._load_image(source)

        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')

        # Calculate dimensions
        orig_width, orig_height = img.size
        aspect_ratio = options.get('aspect_ratio', 0.55)

        if height is None:
            height = int((orig_height / orig_width) * width * aspect_ratio)

        # Resize image
        img = img.resize((width, height), Image.Resampling.LANCZOS)

        # Calculate depth map using local contrast
        depth_weight = options.get('depth_weight', 0.3)
        blur_radius = options.get('blur_radius', 2)

        depth_map = self._estimate_depth(img, blur_radius)

        # Convert to array
        pixels = np.array(img)
        depth = np.array(depth_map)

        # Combine luminance with depth
        combined = (
            pixels * (1 - depth_weight) +
            depth * depth_weight
        ).astype(np.uint8)

        # Map to characters
        matrix: GlyphMatrix = []
        for row in combined:
            char_row = [self.density_map[int(pixel)] for pixel in row]
            matrix.append(char_row)

        return matrix

    def _load_image(
        self, source: Union[str, Image.Image, PixelArray]
    ) -> Image.Image:
        """Load image from various sources."""
        if isinstance(source, str):
            return Image.open(source)
        elif isinstance(source, np.ndarray):
            return Image.fromarray(source)
        elif isinstance(source, Image.Image):
            return source
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

    def _estimate_depth(
        self, img: Image.Image, blur_radius: int
    ) -> Image.Image:
        """
        Estimate depth using focus-based heuristics.

        Areas in focus (high local contrast) are considered closer,
        while blurred areas (low local contrast) are considered farther.
        """
        # Apply Gaussian blur
        blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Calculate local contrast (difference from blur)
        original = np.array(img, dtype=np.float32)
        blur_array = np.array(blurred, dtype=np.float32)

        # Absolute difference gives local contrast
        contrast = np.abs(original - blur_array)

        # Normalize to 0-255
        if contrast.max() > 0:
            contrast = (contrast / contrast.max() * 255).astype(np.uint8)
        else:
            contrast = contrast.astype(np.uint8)

        return Image.fromarray(contrast)


class EdgeDetector:
    """
    Detect and emphasize edges in images for structural glyph art.

    Uses Sobel or Laplacian operators to find edges, then combines
    edge information with original luminance for character selection.

    This produces glyph art that emphasizes structural boundaries,
    making the subject more recognizable at small sizes.
    """

    def __init__(
        self,
        charset: str = " .:-=+*#%@",
        edge_charset: str = "/\\|-+",
    ) -> None:
        """
        Initialize the EdgeDetector.

        Args:
            charset: Characters for fill areas
            edge_charset: Characters for edges (optional directional)
        """
        self.charset = charset
        self.edge_charset = edge_charset
        self.density_map = self._create_density_map(charset)
        logger.debug("EdgeDetector initialized")

    def _create_density_map(self, charset: str) -> Dict[int, str]:
        """Create mapping from brightness to character."""
        mapping: Dict[int, str] = {}
        length = len(charset)
        for i in range(256):
            idx = min(int(i * length / 256), length - 1)
            mapping[i] = charset[idx]
        return mapping

    def transform(
        self,
        source: Union[str, Image.Image, PixelArray],
        width: int = 80,
        height: Optional[int] = None,
        **options: Any,
    ) -> GlyphMatrix:
        """
        Transform an image with edge detection enhancement.

        Args:
            source: Image path, PIL Image, or numpy array
            width: Target width in characters
            height: Target height in characters
            **options: Additional options:
                - edge_weight: Weight for edge influence (0.0-1.0)
                - edge_threshold: Minimum edge strength to consider
                - method: Edge detection method ('sobel', 'laplacian')

        Returns:
            2D matrix of characters with edge enhancement
        """
        # Load image
        img = self._load_image(source)

        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')

        # Calculate dimensions
        orig_width, orig_height = img.size
        aspect_ratio = options.get('aspect_ratio', 0.55)

        if height is None:
            height = int((orig_height / orig_width) * width * aspect_ratio)

        # Resize image
        img = img.resize((width, height), Image.Resampling.LANCZOS)

        # Detect edges
        edge_weight = options.get('edge_weight', 0.5)
        method = options.get('method', 'sobel')

        edges = self._detect_edges(img, method)

        # Convert to arrays
        pixels = np.array(img, dtype=np.float32)
        edge_array = np.array(edges, dtype=np.float32)

        # Combine original with edges
        combined = (
            pixels * (1 - edge_weight) +
            edge_array * edge_weight
        )
        combined = np.clip(combined, 0, 255).astype(np.uint8)

        # Map to characters
        matrix: GlyphMatrix = []
        for row in combined:
            char_row = [self.density_map[int(pixel)] for pixel in row]
            matrix.append(char_row)

        return matrix

    def _load_image(
        self, source: Union[str, Image.Image, PixelArray]
    ) -> Image.Image:
        """Load image from various sources."""
        if isinstance(source, str):
            return Image.open(source)
        elif isinstance(source, np.ndarray):
            return Image.fromarray(source)
        elif isinstance(source, Image.Image):
            return source
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

    def _detect_edges(self, img: Image.Image, method: str) -> Image.Image:
        """
        Detect edges using the specified method.

        Args:
            img: Grayscale image
            method: Detection method ('sobel', 'laplacian')

        Returns:
            Edge-detected image
        """
        if method == 'laplacian':
            # Use PIL's FIND_EDGES filter (Laplacian-based)
            return img.filter(ImageFilter.FIND_EDGES)

        else:  # Default to Sobel
            # Apply Sobel filters for X and Y gradients
            pixels = np.array(img, dtype=np.float32)

            # Sobel kernels
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

            # Apply convolution (simplified using PIL)
            # Note: For production, scipy.ndimage.convolve would be better
            edges_x = img.filter(
                ImageFilter.Kernel((3, 3), sobel_x.flatten(), scale=1)
            )
            edges_y = img.filter(
                ImageFilter.Kernel((3, 3), sobel_y.flatten(), scale=1)
            )

            # Combine gradients
            gx = np.array(edges_x, dtype=np.float32)
            gy = np.array(edges_y, dtype=np.float32)
            magnitude = np.sqrt(gx**2 + gy**2)

            # Normalize
            if magnitude.max() > 0:
                magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
            else:
                magnitude = magnitude.astype(np.uint8)

            return Image.fromarray(magnitude)


__all__ = ["ImageTransformer", "ColorMapper", "DepthAnalyzer", "EdgeDetector"]
