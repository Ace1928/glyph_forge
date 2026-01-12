"""
⚡ Eidosian Test Suite: Transformer Classes ⚡

Comprehensive tests for the transformer classes that convert various
input formats into glyph matrices.
"""

import os
import tempfile
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from glyph_forge.transformers import (
    ImageTransformer,
    ColorMapper,
    DepthAnalyzer,
    EdgeDetector,
)


@pytest.fixture(scope="module")
def test_images() -> dict:
    """Generate test images for transformer verification."""
    test_dir = tempfile.mkdtemp(prefix="glyph_transformer_test_")

    # 1. Gradient image (50x50)
    gradient = np.linspace(0, 255, 50, dtype=np.uint8)
    gradient_img = np.repeat(gradient.reshape(1, 50), 50, axis=0)
    gradient_pil = Image.fromarray(gradient_img, mode='L')
    gradient_path = os.path.join(test_dir, 'gradient.png')
    gradient_pil.save(gradient_path)

    # 2. RGB color test image (50x50)
    rgb_data = np.zeros((50, 50, 3), dtype=np.uint8)
    rgb_data[0:25, :, 0] = 255  # Red top half
    rgb_data[25:50, :, 1] = 255  # Green bottom half
    rgb_img = Image.fromarray(rgb_data, mode='RGB')
    rgb_path = os.path.join(test_dir, 'rgb.png')
    rgb_img.save(rgb_path)

    # 3. Edge test image (50x50) - black with white cross
    edge_data = np.zeros((50, 50), dtype=np.uint8)
    edge_data[20:30, :] = 255  # Horizontal line
    edge_data[:, 20:30] = 255  # Vertical line
    edge_img = Image.fromarray(edge_data, mode='L')
    edge_path = os.path.join(test_dir, 'edges.png')
    edge_img.save(edge_path)

    # 4. Depth simulation image (50x50) - center bright, edges dark
    y, x = np.ogrid[0:50, 0:50]
    center_y, center_x = 25, 25
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    depth_data = 255 - (distance * 5).astype(np.uint8)
    depth_data = np.clip(depth_data, 0, 255)
    depth_img = Image.fromarray(depth_data, mode='L')
    depth_path = os.path.join(test_dir, 'depth.png')
    depth_img.save(depth_path)

    yield {
        'dir': test_dir,
        'gradient': {'img': gradient_pil, 'path': gradient_path},
        'rgb': {'img': rgb_img, 'path': rgb_path},
        'edges': {'img': edge_img, 'path': edge_path},
        'depth': {'img': depth_img, 'path': depth_path},
    }

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)


class TestImageTransformer:
    """Tests for the ImageTransformer class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        transformer = ImageTransformer()
        assert transformer.charset == ImageTransformer.DEFAULT_CHARSET
        assert len(transformer.density_map) == 256

    def test_init_custom_charset(self) -> None:
        """Test initialization with custom character set."""
        charset = "ABC"
        transformer = ImageTransformer(charset=charset)
        assert transformer.charset == charset

    def test_init_inverted(self) -> None:
        """Test initialization with inverted character set."""
        charset = "ABC"
        transformer = ImageTransformer(charset=charset, invert=True)
        assert transformer.charset == "CBA"

    def test_transform_from_path(self, test_images: dict) -> None:
        """Test transformation from file path."""
        transformer = ImageTransformer()
        matrix = transformer.transform(
            test_images['gradient']['path'],
            width=20,
            height=10,
        )

        assert len(matrix) == 10  # Height
        assert len(matrix[0]) == 20  # Width
        assert all(isinstance(row, list) for row in matrix)
        assert all(isinstance(char, str) for row in matrix for char in row)

    def test_transform_from_pil_image(self, test_images: dict) -> None:
        """Test transformation from PIL Image object."""
        transformer = ImageTransformer()
        matrix = transformer.transform(
            test_images['gradient']['img'],
            width=20,
            height=10,
        )

        assert len(matrix) == 10
        assert len(matrix[0]) == 20

    def test_transform_from_numpy_array(self) -> None:
        """Test transformation from numpy array."""
        transformer = ImageTransformer()
        data = np.zeros((50, 50), dtype=np.uint8)
        matrix = transformer.transform(data, width=20, height=10)

        assert len(matrix) == 10
        assert len(matrix[0]) == 20

    def test_transform_auto_height(self, test_images: dict) -> None:
        """Test transformation with automatic height calculation."""
        transformer = ImageTransformer()
        matrix = transformer.transform(
            test_images['gradient']['path'],
            width=40,
        )

        # Height should be calculated based on aspect ratio
        assert len(matrix) > 0
        assert len(matrix[0]) == 40

    def test_transform_with_adjustments(self, test_images: dict) -> None:
        """Test transformation with brightness and contrast adjustments."""
        transformer = ImageTransformer()
        matrix_normal = transformer.transform(
            test_images['gradient']['path'],
            width=20,
            height=10,
        )

        matrix_bright = transformer.transform(
            test_images['gradient']['path'],
            width=20,
            height=10,
            brightness=1.5,
            contrast=1.2,
        )

        # Results should differ with adjustments
        assert matrix_normal != matrix_bright

    def test_density_map_coverage(self) -> None:
        """Test that density map covers all 256 values."""
        transformer = ImageTransformer()
        for i in range(256):
            assert i in transformer.density_map
            assert isinstance(transformer.density_map[i], str)

    def test_invalid_source_type(self) -> None:
        """Test that invalid source types raise TypeError."""
        transformer = ImageTransformer()
        with pytest.raises(TypeError):
            transformer.transform(12345, width=20)


class TestColorMapper:
    """Tests for the ColorMapper class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        mapper = ColorMapper()
        assert mapper.charset is not None
        assert mapper.color_weights == ColorMapper.DEFAULT_WEIGHTS

    def test_init_custom_weights(self) -> None:
        """Test initialization with custom color weights."""
        weights = (0.5, 0.3, 0.2)
        mapper = ColorMapper(color_weights=weights)
        assert mapper.color_weights == weights

    def test_transform_rgb_image(self, test_images: dict) -> None:
        """Test transformation of RGB image."""
        mapper = ColorMapper()
        matrix = mapper.transform(
            test_images['rgb']['path'],
            width=20,
            height=10,
        )

        assert len(matrix) == 10
        assert len(matrix[0]) == 20

    def test_transform_with_saturation(self, test_images: dict) -> None:
        """Test transformation with saturation preservation."""
        mapper = ColorMapper()
        matrix_normal = mapper.transform(
            test_images['rgb']['path'],
            width=20,
            height=10,
        )

        matrix_saturated = mapper.transform(
            test_images['rgb']['path'],
            width=20,
            height=10,
            preserve_saturation=True,
        )

        # Results may differ when saturation is preserved
        # (depends on actual image content)
        assert len(matrix_saturated) == 10

    def test_grayscale_conversion(self, test_images: dict) -> None:
        """Test that grayscale images are handled correctly."""
        mapper = ColorMapper()
        # Grayscale image should be converted to RGB internally
        matrix = mapper.transform(
            test_images['gradient']['path'],
            width=20,
            height=10,
        )

        assert len(matrix) == 10


class TestDepthAnalyzer:
    """Tests for the DepthAnalyzer class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        analyzer = DepthAnalyzer()
        assert analyzer.charset is not None

    def test_transform_basic(self, test_images: dict) -> None:
        """Test basic depth analysis transformation."""
        analyzer = DepthAnalyzer()
        matrix = analyzer.transform(
            test_images['depth']['path'],
            width=20,
            height=10,
        )

        assert len(matrix) == 10
        assert len(matrix[0]) == 20

    def test_transform_with_depth_weight(self, test_images: dict) -> None:
        """Test transformation with different depth weights."""
        analyzer = DepthAnalyzer()
        matrix_low = analyzer.transform(
            test_images['depth']['path'],
            width=20,
            height=10,
            depth_weight=0.1,
        )

        matrix_high = analyzer.transform(
            test_images['depth']['path'],
            width=20,
            height=10,
            depth_weight=0.9,
        )

        # Different weights should produce different results
        assert matrix_low != matrix_high

    def test_transform_with_blur_radius(self, test_images: dict) -> None:
        """Test transformation with different blur radii."""
        analyzer = DepthAnalyzer()
        matrix = analyzer.transform(
            test_images['depth']['path'],
            width=20,
            height=10,
            blur_radius=5,
        )

        assert len(matrix) == 10


class TestEdgeDetector:
    """Tests for the EdgeDetector class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        detector = EdgeDetector()
        assert detector.charset is not None
        assert detector.edge_charset is not None

    def test_transform_basic(self, test_images: dict) -> None:
        """Test basic edge detection transformation."""
        detector = EdgeDetector()
        matrix = detector.transform(
            test_images['edges']['path'],
            width=20,
            height=10,
        )

        assert len(matrix) == 10
        assert len(matrix[0]) == 20

    def test_transform_sobel_method(self, test_images: dict) -> None:
        """Test edge detection with Sobel method."""
        detector = EdgeDetector()
        matrix = detector.transform(
            test_images['edges']['path'],
            width=20,
            height=10,
            method='sobel',
        )

        assert len(matrix) == 10

    def test_transform_laplacian_method(self, test_images: dict) -> None:
        """Test edge detection with Laplacian method."""
        detector = EdgeDetector()
        matrix = detector.transform(
            test_images['edges']['path'],
            width=20,
            height=10,
            method='laplacian',
        )

        assert len(matrix) == 10

    def test_transform_with_edge_weight(self, test_images: dict) -> None:
        """Test transformation with different edge weights."""
        detector = EdgeDetector()
        matrix_low = detector.transform(
            test_images['edges']['path'],
            width=20,
            height=10,
            edge_weight=0.1,
        )

        matrix_high = detector.transform(
            test_images['edges']['path'],
            width=20,
            height=10,
            edge_weight=0.9,
        )

        # Different weights should produce different results
        # (in most cases)
        assert len(matrix_low) == 10
        assert len(matrix_high) == 10


class TestTransformerIntegration:
    """Integration tests for transformer classes."""

    def test_all_transformers_produce_valid_output(
        self, test_images: dict
    ) -> None:
        """Verify all transformers produce valid glyph matrices."""
        transformers = [
            ImageTransformer(),
            ColorMapper(),
            DepthAnalyzer(),
            EdgeDetector(),
        ]

        for transformer in transformers:
            matrix = transformer.transform(
                test_images['gradient']['path'],
                width=20,
                height=10,
            )

            # Validate structure
            assert isinstance(matrix, list)
            assert len(matrix) == 10
            assert all(len(row) == 20 for row in matrix)
            assert all(
                isinstance(char, str) and len(char) == 1
                for row in matrix
                for char in row
            )

    def test_transformers_with_different_charsets(
        self, test_images: dict
    ) -> None:
        """Test all transformers work with different character sets."""
        charsets = [" .", " .:-=+*#%@", "░▒▓█"]

        for charset in charsets:
            transformer = ImageTransformer(charset=charset)
            matrix = transformer.transform(
                test_images['gradient']['path'],
                width=20,
                height=10,
            )

            # All characters in output should be from charset
            all_chars = set(char for row in matrix for char in row)
            assert all_chars.issubset(set(charset))
