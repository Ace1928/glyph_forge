"""Vectorized directional edge analysis for glyph rendering.

The implementation intentionally uses only NumPy, which keeps edge-aware
rendering available in the base installation.  It works on the reduced output
grid, so its cost scales with the number of terminal cells rather than source
image resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


class EdgeAlgorithm(str, Enum):
    """Supported edge detectors."""

    SOBEL = "sobel"
    PREWITT = "prewitt"
    SCHARR = "scharr"
    LAPLACIAN = "laplacian"
    CANNY = "canny"


@dataclass(frozen=True, slots=True)
class EdgeMap:
    """Normalized edge strength and raw directional gradients."""

    magnitude: NDArray[np.uint8]
    gradient_x: NDArray[np.float32]
    gradient_y: NDArray[np.float32]


_KERNELS: dict[str, NDArray[np.float32]] = {
    "sobel_x": np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "sobel_y": np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
    "prewitt_x": np.asarray([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32),
    "prewitt_y": np.asarray([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32),
    "scharr_x": np.asarray([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32),
    "scharr_y": np.asarray([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32),
    "laplacian": np.asarray([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
    "gaussian": np.asarray([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
    / np.float32(16.0),
}


def normalize_algorithm(value: EdgeAlgorithm | str) -> EdgeAlgorithm:
    """Normalize an algorithm name and report valid choices on failure."""

    if isinstance(value, EdgeAlgorithm):
        return value
    try:
        return EdgeAlgorithm(value.casefold())
    except ValueError as exc:
        choices = ", ".join(item.value for item in EdgeAlgorithm)
        raise ValueError(f"Unknown edge algorithm {value!r}; choose {choices}") from exc


def _convolve(
    values: NDArray[np.float32], kernel: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Apply a 3×3 kernel with replicated boundaries and bounded memory."""

    height, width = values.shape
    padded = np.pad(values, 1, mode="edge")
    result = np.zeros((height, width), dtype=np.float32)
    for y in range(3):
        for x in range(3):
            weight = float(kernel[y, x])
            if weight:
                result += padded[y : y + height, x : x + width] * weight
    return result


def _normalize_magnitude(values: NDArray[np.float32]) -> NDArray[np.uint8]:
    peak = float(np.max(values, initial=0.0))
    if peak <= 0:
        return np.zeros(values.shape, dtype=np.uint8)
    return cast(
        NDArray[np.uint8],
        np.clip(values * (255.0 / peak), 0, 255).astype(np.uint8),
    )


def _directional_gradients(
    values: NDArray[np.float32], algorithm: EdgeAlgorithm
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    prefix = algorithm.value
    return (
        _convolve(values, _KERNELS[f"{prefix}_x"]),
        _convolve(values, _KERNELS[f"{prefix}_y"]),
    )


def _non_maximum_suppression(
    magnitude: NDArray[np.float32],
    gradient_x: NDArray[np.float32],
    gradient_y: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Thin Sobel gradients along four quantized directions."""

    angle = (np.rad2deg(np.arctan2(gradient_y, gradient_x)) + 180) % 180
    left = np.roll(magnitude, 1, axis=1)
    right = np.roll(magnitude, -1, axis=1)
    up = np.roll(magnitude, 1, axis=0)
    down = np.roll(magnitude, -1, axis=0)
    up_left = np.roll(up, 1, axis=1)
    down_right = np.roll(down, -1, axis=1)
    up_right = np.roll(up, -1, axis=1)
    down_left = np.roll(down, 1, axis=1)

    horizontal = (angle < 22.5) | (angle >= 157.5)
    diagonal_up = (angle >= 22.5) & (angle < 67.5)
    vertical = (angle >= 67.5) & (angle < 112.5)
    diagonal_down = (angle >= 112.5) & (angle < 157.5)
    keep = (
        (horizontal & (magnitude >= left) & (magnitude >= right))
        | (diagonal_up & (magnitude >= up_right) & (magnitude >= down_left))
        | (vertical & (magnitude >= up) & (magnitude >= down))
        | (diagonal_down & (magnitude >= up_left) & (magnitude >= down_right))
    )
    keep[[0, -1], :] = False
    keep[:, [0, -1]] = False
    return np.where(keep, magnitude, 0).astype(np.float32)


def _connected_canny(
    magnitude: NDArray[np.float32], threshold: int
) -> NDArray[np.uint8]:
    """Apply dual thresholds and connect weak pixels to strong neighbours."""

    normalized = _normalize_magnitude(magnitude)
    high = max(1, min(255, threshold))
    low = max(1, high // 2)
    strong = normalized >= high
    weak = normalized >= low
    connected = strong.copy()
    for _ in range(16):
        neighbours = np.zeros_like(connected)
        for y in (-1, 0, 1):
            for x in (-1, 0, 1):
                if x or y:
                    neighbours |= np.roll(np.roll(connected, y, axis=0), x, axis=1)
        expanded = strong | (weak & neighbours)
        expanded[[0, -1], :] = strong[[0, -1], :]
        expanded[:, [0, -1]] = strong[:, [0, -1]]
        if np.array_equal(expanded, connected):
            break
        connected = expanded
    return np.where(connected, 255, 0).astype(np.uint8)


def detect_edges(
    grayscale: NDArray[Any],
    algorithm: EdgeAlgorithm | str = EdgeAlgorithm.SOBEL,
    *,
    threshold: int = 48,
) -> EdgeMap:
    """Detect directional edges in a two-dimensional grayscale array."""

    values = np.asarray(grayscale)
    if values.ndim != 2:
        raise ValueError("Edge detection requires a two-dimensional grayscale array")
    if not 0 <= threshold <= 255:
        raise ValueError("edge threshold must be between 0 and 255")
    values = np.clip(values, 0, 255).astype(np.float32, copy=False)
    selected = normalize_algorithm(algorithm)

    if selected is EdgeAlgorithm.LAPLACIAN:
        smoothed = _convolve(values, _KERNELS["gaussian"])
        gradient_x, gradient_y = _directional_gradients(smoothed, EdgeAlgorithm.SOBEL)
        magnitude = _normalize_magnitude(
            np.abs(_convolve(smoothed, _KERNELS["laplacian"]))
        )
    else:
        base = EdgeAlgorithm.SOBEL if selected is EdgeAlgorithm.CANNY else selected
        gradient_x, gradient_y = _directional_gradients(values, base)
        raw_magnitude = np.hypot(gradient_x, gradient_y).astype(np.float32)
        if selected is EdgeAlgorithm.CANNY:
            magnitude = _connected_canny(
                _non_maximum_suppression(
                    raw_magnitude,
                    gradient_x,
                    gradient_y,
                ),
                threshold,
            )
        else:
            magnitude = _normalize_magnitude(raw_magnitude)

    if selected is not EdgeAlgorithm.CANNY and threshold:
        magnitude = np.where(magnitude >= threshold, magnitude, 0).astype(np.uint8)
    return EdgeMap(
        magnitude=magnitude,
        gradient_x=gradient_x.astype(np.float32, copy=False),
        gradient_y=gradient_y.astype(np.float32, copy=False),
    )


def directional_glyphs(edges: EdgeMap) -> NDArray[np.str_]:
    """Map gradient tangents to light, standard, or heavy line glyphs."""

    tangent = (np.rad2deg(np.arctan2(edges.gradient_y, edges.gradient_x)) + 90) % 180
    horizontal = (tangent < 22.5) | (tangent >= 157.5)
    diagonal_up = (tangent >= 22.5) & (tangent < 67.5)
    vertical = (tangent >= 67.5) & (tangent < 112.5)

    glyphs = np.full(edges.magnitude.shape, "╲", dtype="<U1")
    glyphs[diagonal_up] = "╱"
    glyphs[vertical] = "│"
    glyphs[horizontal] = "─"
    strong = edges.magnitude >= 176
    glyphs[strong & vertical] = "┃"
    glyphs[strong & horizontal] = "━"
    return glyphs


__all__ = [
    "EdgeAlgorithm",
    "EdgeMap",
    "detect_edges",
    "directional_glyphs",
    "normalize_algorithm",
]
