"""Optional extraction, buffering, and terminal-stream rendering APIs."""

from __future__ import annotations

from .engine import (
    UnifiedStreamConfig,
    UnifiedStreamEngine,
)
from .extractors import (
    ExtractionResult,
    StreamExtractionError,
    VideoSourceExtractor,
    YouTubeExtractor,
)

# Aliases for convenience
StreamEngine = UnifiedStreamEngine
StreamConfig = UnifiedStreamConfig

__all__ = [
    "UnifiedStreamEngine",
    "UnifiedStreamConfig",
    "StreamEngine",
    "StreamConfig",
    "VideoSourceExtractor",
    "YouTubeExtractor",
    "ExtractionResult",
    "StreamExtractionError",
]
