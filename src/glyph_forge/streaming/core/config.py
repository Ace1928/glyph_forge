"""
Configuration management for Glyph Forge streaming.

Centralized configuration with sensible defaults and validation.
"""

import os
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Tuple


class RenderMode(Enum):
    """Available rendering modes."""

    GRADIENT = auto()  # Block characters (█▓▒░·)
    BRAILLE = auto()  # Braille patterns (2x4 sub-pixel)
    ASCII = auto()  # ASCII characters only
    BLOCKS = auto()  # Unicode block elements
    HYBRID = auto()  # Combined modes


class ColorMode(Enum):
    """Color output modes."""

    NONE = auto()  # No color (fastest)
    ANSI16 = auto()  # 16 basic colors
    ANSI256 = auto()  # 256 color palette (balanced)
    TRUECOLOR = auto()  # 24-bit RGB (highest quality)


class BufferStrategy(Enum):
    """Buffering strategies."""

    FIXED = auto()  # Fixed buffer size
    ADAPTIVE = auto()  # Adjust based on performance
    FULL = auto()  # Buffer entire source before playback


@dataclass
class StreamConfig:
    """Comprehensive streaming configuration.

    Attributes:
        resolution: Target resolution as (width, height) or 'auto'
        max_resolution: Maximum resolution cap (e.g., 720 for 720p)
        render_mode: Glyph rendering mode
        color_mode: Color output mode
        target_fps: Target frames per second (0 = source fps)

        buffer_strategy: How to handle buffering
        min_buffer_seconds: Minimum buffer before playback starts
        target_buffer_seconds: Target buffer for smooth playback

        record_output: Whether to record glyph output
        output_path: Path for recorded output (auto-generated if None)
        output_resolution: Resolution for recording (None = same as display)
        mux_audio: Whether to mux audio into output

        show_metrics: Display performance metrics
        clear_screen: Clear screen before playback
        fit_terminal: Scale to fit terminal window
    """

    # Resolution settings
    resolution: Tuple[int, int] | str = "auto"
    max_resolution: int = 720
    fit_terminal: bool = True
    aspect_ratio: Optional[float] = None

    # Rendering
    render_mode: RenderMode = RenderMode.GRADIENT
    color_mode: ColorMode = ColorMode.ANSI256
    dithering: bool = False

    # Timing
    target_fps: float = 0  # 0 = use source fps

    # Buffering
    buffer_strategy: BufferStrategy = BufferStrategy.ADAPTIVE
    min_buffer_seconds: float = 5.0
    target_buffer_seconds: float = 30.0
    max_buffer_frames: int = 3600  # Cap at 1 minute @ 60fps

    # Recording
    record_output: bool = True
    output_path: Optional[Path] = None
    output_resolution: Optional[Tuple[int, int]] = None
    mux_audio: bool = True
    force_rerender: bool = False

    # Display
    show_metrics: bool = True
    clear_screen: bool = True

    # Audio
    audio_enabled: bool = True
    audio_backend: str = "ffplay"  # ffplay, pygame

    # Performance
    num_workers: int = 0  # 0 = auto-detect
    use_cache: bool = True
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "glyph_forge")

    def __post_init__(self):
        """Validate and normalize configuration."""
        if self.num_workers == 0:
            self.num_workers = max(1, (os.cpu_count() or 1) - 1)
        elif self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")

        if self.resolution != "auto":
            if not isinstance(self.resolution, tuple) or len(self.resolution) != 2:
                raise ValueError("resolution must be 'auto' or a (width, height) tuple")
            if any(dimension < 1 for dimension in self.resolution):
                raise ValueError("resolution dimensions must be positive")

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

    def get_terminal_size(self) -> Tuple[int, int]:
        """Get current terminal dimensions."""
        try:
            size = os.get_terminal_size()
            return size.columns, size.lines - 2  # Reserve lines for metrics
        except OSError:
            return 120, 40

    def calculate_display_size(self, source_width: int, source_height: int) -> Tuple[int, int]:
        """Calculate optimal display size for terminal.

        Args:
            source_width: Source video width
            source_height: Source video height

        Returns:
            (width, height) in characters
        """
        if isinstance(self.resolution, tuple):
            return self.resolution

        term_w, term_h = self.get_terminal_size()

        # Calculate aspect ratio (terminal chars are ~2:1)
        aspect = source_width / source_height
        char_aspect = 2.0  # Terminal characters are roughly twice as tall as wide

        # Fit to terminal while preserving aspect ratio
        w1 = term_w
        h1 = int(w1 / aspect / char_aspect)

        h2 = term_h
        w2 = int(h2 * aspect * char_aspect)

        if h1 <= term_h:
            return w1, h1
        else:
            return w2, h2

    def calculate_record_size(self, source_width: int, source_height: int) -> Tuple[int, int]:
        """Calculate recording resolution (higher than display).

        Args:
            source_width: Source video width
            source_height: Source video height

        Returns:
            (width, height) in characters for HD recording
        """
        if self.output_resolution:
            return self.output_resolution

        # Target HD recording based on source resolution
        aspect = source_width / source_height
        char_aspect = 2.0

        # Use max_resolution as a guide
        if source_height >= 1080:
            # 1080p source -> ~320x90 chars
            target_h = 90
        elif source_height >= 720:
            # 720p source -> ~213x60 chars
            target_h = 60
        else:
            # Lower res -> ~160x45 chars
            target_h = 45

        target_w = int(target_h * aspect * char_aspect)
        return target_w, target_h

    def generate_output_path(self, source_name: str) -> Path:
        """Generate output path based on source name.

        Args:
            source_name: Name of source (URL, file, etc.)

        Returns:
            Path for output file
        """
        from ..naming import build_output_path

        return build_output_path(
            source=source_name,
            title=None,
            output_dir=None,
            ext="mp4",
            output=self.output_path,
            overwrite=self.force_rerender,
        )
