"""
Glyph Forge Streaming Core - Modular High-Performance Architecture

This package provides the core streaming infrastructure:
- config.py: Configuration management
- buffer.py: Smart adaptive buffering system
- capture.py: Video capture from URLs, files, webcams, browsers
- renderer.py: High-fidelity glyph rendering engine
- recorder.py: Terminal-to-video recording
- sync.py: Audio playback and synchronization

Unified streaming engine lives in :mod:`glyph_forge.streaming.engine`.
"""

from .buffer import AdaptiveBuffer, BufferedFrame, BufferMetrics
from .capture import CaptureSource, VideoCapture, VideoInfo
from .config import BufferStrategy, ColorMode, RenderMode, StreamConfig
from .recorder import GlyphRecorder, RecorderConfig
from .renderer import GlyphRenderer, LookupTables, RenderConfig
from .sync import AudioConfig, AudioDownloader, AudioSync

__all__ = [
    # Config
    "StreamConfig",
    "RenderMode",
    "ColorMode",
    "BufferStrategy",
    # Buffer
    "AdaptiveBuffer",
    "BufferedFrame",
    "BufferMetrics",
    # Capture
    "VideoCapture",
    "VideoInfo",
    "CaptureSource",
    # Renderer
    "GlyphRenderer",
    "RenderConfig",
    "LookupTables",
    # Recorder
    "GlyphRecorder",
    "RecorderConfig",
    # Audio
    "AudioSync",
    "AudioConfig",
    "AudioDownloader",
]
