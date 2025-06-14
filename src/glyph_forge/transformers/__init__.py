"""Placeholder transformer classes for Glyph Forge."""
from __future__ import annotations
from typing import Any, List

GlyphMatrix = List[List[str]]

class ImageTransformer:
    def transform(self, source: Any, **options: Any) -> GlyphMatrix:
        raise NotImplementedError

class ColorMapper:
    def transform(self, source: Any, **options: Any) -> GlyphMatrix:
        raise NotImplementedError

class DepthAnalyzer:
    def transform(self, source: Any, **options: Any) -> GlyphMatrix:
        raise NotImplementedError

class EdgeDetector:
    def transform(self, source: Any, **options: Any) -> GlyphMatrix:
        raise NotImplementedError

__all__ = ["ImageTransformer", "ColorMapper", "DepthAnalyzer", "EdgeDetector"]
