"""
Smart Adaptive Buffer for Glyph Forge streaming.

Implements intelligent buffering that ensures smooth playback:
- Calculates required buffer based on render performance
- Adapts to network conditions and processing speed
- Never starts playback until sufficient buffer exists
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class BufferMetrics:
    """Metrics for buffer performance analysis."""

    render_times: deque = field(default_factory=lambda: deque(maxlen=100))
    frame_sizes: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def avg_render_time(self) -> float:
        """Average render time in seconds."""
        if not self.render_times:
            return 0.1  # Default assumption
        return sum(self.render_times) / len(self.render_times)

    @property
    def max_render_time(self) -> float:
        """Maximum render time (for worst-case planning)."""
        if not self.render_times:
            return 0.2
        return max(self.render_times)

    @property
    def render_fps(self) -> float:
        """Estimated sustainable render FPS."""
        avg = self.avg_render_time
        if avg <= 0:
            return 60.0
        return 1.0 / avg

    def add_sample(self, render_time: float, frame_size: int = 0):
        """Add a timing sample."""
        self.render_times.append(render_time)
        if frame_size > 0:
            self.frame_sizes.append(frame_size)


@dataclass
class BufferedFrame:
    """A single buffered frame with metadata."""

    display_data: str  # Rendered glyph data for display
    record_data: Optional[str]  # HD rendered data for recording (may be same)
    timestamp: float  # Frame timestamp in video
    frame_idx: int  # Frame index
    raw_frame: Optional[np.ndarray] = None  # Original frame (for re-rendering)


class AdaptiveBuffer:
    """Smart adaptive buffer for smooth glyph streaming.

    Key features:
    - Calculates minimum required buffer based on render performance
    - Pre-buffers enough frames to guarantee smooth playback
    - Background thread continues buffering during playback
    - Never allows playback to start if buffer is insufficient

    Usage:
        buffer = AdaptiveBuffer(config, renderer)
        buffer.start_buffering(video_capture, total_frames)

        # Wait until safe to start
        while not buffer.ready_for_playback:
            time.sleep(0.1)

        # Playback loop
        while buffer.has_frames:
            frame = buffer.get_next_frame()
            display(frame.display_data)
    """

    def __init__(
        self,
        target_fps: float,
        min_buffer_seconds: float = 5.0,
        target_buffer_seconds: float = 30.0,
        max_buffer_frames: int = 3600,
    ):
        """Initialize adaptive buffer.

        Args:
            target_fps: Target playback FPS
            min_buffer_seconds: Minimum buffer before playback allowed
            target_buffer_seconds: Target buffer for comfort zone
            max_buffer_frames: Maximum frames to buffer (memory limit)
        """
        self.target_fps = target_fps
        self.min_buffer_seconds = min_buffer_seconds
        self.target_buffer_seconds = target_buffer_seconds
        self.max_buffer_frames = max_buffer_frames

        # Buffer storage
        self._frames: deque[BufferedFrame] = deque()
        self._lock = threading.RLock()

        # Metrics
        self.metrics = BufferMetrics()

        # State
        self._total_frames = 0
        self._frames_buffered = 0
        self._frames_consumed = 0
        self._buffering_complete = False
        self._buffering_thread: Optional[threading.Thread] = None
        self._stop_buffering = threading.Event()

    @property
    def buffer_size(self) -> int:
        """Current number of frames in buffer."""
        with self._lock:
            return len(self._frames)

    @property
    def buffer_seconds(self) -> float:
        """Current buffer duration in seconds."""
        return self.buffer_size / self.target_fps if self.target_fps > 0 else 0

    @property
    def min_required_frames(self) -> int:
        """Minimum frames required before playback can start."""
        render_fps = self.metrics.render_fps

        # If we can render faster than playback, we need less buffer
        if render_fps >= self.target_fps * 1.5:
            # Fast renderer - minimal buffer needed
            return int(self.min_buffer_seconds * self.target_fps)

        # If render is slow, we need more buffer
        # Calculate how much we need to stay ahead
        render_ratio = self.target_fps / render_fps if render_fps > 0 else 2.0

        if render_ratio > 1.0:
            # We render slower than playback - need proportionally more buffer
            needed_seconds = self.target_buffer_seconds * render_ratio
            needed_seconds = min(needed_seconds, 60.0)  # Cap at 60 seconds
        else:
            needed_seconds = self.min_buffer_seconds

        frames = int(needed_seconds * self.target_fps)
        return min(frames, self.max_buffer_frames, self._total_frames)

    @property
    def ready_for_playback(self) -> bool:
        """Check if buffer has enough frames to start playback safely."""
        if self._buffering_complete:
            return self.buffer_size > 0

        return self.buffer_size >= self.min_required_frames

    @property
    def has_frames(self) -> bool:
        """Check if there are frames available."""
        with self._lock:
            return len(self._frames) > 0 or not self._buffering_complete

    @property
    def progress(self) -> float:
        """Buffer fill progress (0.0 to 1.0)."""
        if self._total_frames == 0:
            return 0.0
        return self._frames_buffered / self._total_frames

    @property
    def playback_progress(self) -> float:
        """Playback progress (0.0 to 1.0)."""
        if self._total_frames == 0:
            return 0.0
        return self._frames_consumed / self._total_frames

    def start_buffering(
        self,
        frame_generator: Callable[[], Tuple[bool, Optional[np.ndarray]]],
        render_func: Callable[[np.ndarray, int, int], str],
        display_size: Tuple[int, int],
        record_size: Optional[Tuple[int, int]],
        total_frames: int,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ):
        """Start background buffering.

        Args:
            frame_generator: Function that returns (success, frame) tuple
            render_func: Function to render frame to glyph string
            display_size: (width, height) for display rendering
            record_size: (width, height) for recording (None = same as display)
            total_frames: Total frames in source (0 if unknown)
            on_progress: Callback for progress updates (frames_buffered, total)
        """
        self._total_frames = total_frames if total_frames > 0 else self.max_buffer_frames
        self._stop_buffering.clear()

        def buffer_worker():
            frame_idx = 0
            display_w, display_h = display_size
            dual_render = record_size is not None and record_size != display_size
            record_w, record_h = record_size if record_size else display_size

            while not self._stop_buffering.is_set():
                # Check if buffer is full
                if self.buffer_size >= self.max_buffer_frames:
                    time.sleep(0.01)
                    continue

                # Get next frame
                try:
                    success, frame = frame_generator()
                except StopIteration:
                    break
                if not success or frame is None:
                    break

                # Render for display
                start = time.perf_counter()
                display_data = render_func(frame, display_w, display_h)
                render_time = time.perf_counter() - start

                # Render for recording if different resolution
                if dual_render:
                    record_data = render_func(frame, record_w, record_h)
                else:
                    record_data = display_data

                self.metrics.add_sample(render_time, len(display_data))

                # Create buffered frame
                buffered = BufferedFrame(
                    display_data=display_data,
                    record_data=record_data,
                    timestamp=frame_idx / self.target_fps,
                    frame_idx=frame_idx,
                )

                # Add to buffer
                with self._lock:
                    self._frames.append(buffered)
                    self._frames_buffered += 1

                frame_idx += 1

                if on_progress:
                    on_progress(frame_idx, self._total_frames)

            self._buffering_complete = True

        self._buffering_thread = threading.Thread(target=buffer_worker, daemon=True)
        self._buffering_thread.start()

    def get_next_frame(self, timeout: float = 1.0) -> Optional[BufferedFrame]:
        """Get next frame from buffer.

        Args:
            timeout: Maximum time to wait for frame

        Returns:
            BufferedFrame or None if buffer empty and complete
        """
        start = time.perf_counter()

        while time.perf_counter() - start < timeout:
            with self._lock:
                if self._frames:
                    self._frames_consumed += 1
                    return self._frames.popleft()

            if self._buffering_complete:
                return None

            time.sleep(0.001)

        return None

    def peek_next_frame(self) -> Optional[BufferedFrame]:
        """Peek at next frame without consuming it."""
        with self._lock:
            if self._frames:
                return self._frames[0]
        return None

    def stop(self):
        """Stop buffering."""
        self._stop_buffering.set()
        if self._buffering_thread:
            self._buffering_thread.join(timeout=5.0)

    def clear(self):
        """Clear buffer and reset state."""
        self.stop()
        with self._lock:
            self._frames.clear()
        self._frames_buffered = 0
        self._frames_consumed = 0
        self._buffering_complete = False

    def get_status(self) -> dict:
        """Get buffer status for display."""
        return {
            "buffer_size": self.buffer_size,
            "buffer_seconds": self.buffer_seconds,
            "min_required": self.min_required_frames,
            "ready": self.ready_for_playback,
            "render_fps": self.metrics.render_fps,
            "progress": self.progress,
            "playback_progress": self.playback_progress,
            "complete": self._buffering_complete,
        }
