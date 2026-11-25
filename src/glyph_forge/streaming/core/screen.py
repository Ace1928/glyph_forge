"""
Screen Capture Module for Glyph Forge.

Captures video directly from the screen - works with any application:
- Netflix in your logged-in browser
- Any streaming service
- Games
- Any window

Much simpler than browser automation - just capture what's on screen.

Requirements:
    pip install mss
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np

try:
    import mss
    import mss.tools

    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class ScreenConfig:
    """Screen capture configuration."""

    monitor: int = 0  # 0 = all monitors, 1 = first, 2 = second, etc.
    region: Optional[Tuple[int, int, int, int]] = None  # (left, top, width, height)
    fps: float = 30.0
    resize: Optional[Tuple[int, int]] = None  # Resize output to (width, height)


class ScreenCapture:
    """High-performance screen capture.

    Captures frames directly from the screen at high FPS.
    Works with any application - Netflix, games, anything visible.

    Usage:
        # Capture entire screen
        cap = ScreenCapture()
        cap.start()

        for frame in cap.frames():
            process(frame)

        # Capture specific region
        cap = ScreenCapture(ScreenConfig(
            region=(100, 100, 1920, 1080)  # x, y, width, height
        ))

        # Capture specific monitor
        cap = ScreenCapture(ScreenConfig(monitor=1))
    """

    def __init__(self, config: Optional[ScreenConfig] = None):
        """Initialize screen capture.

        Args:
            config: Screen capture configuration
        """
        if not MSS_AVAILABLE:
            raise RuntimeError("Screen capture requires mss. Install with: pip install mss")

        self.config = config or ScreenConfig()
        self._sct: Optional[mss.mss] = None
        self._monitor: Dict[str, Any] = {}
        self._running = False
        self._frame_interval = 1.0 / self.config.fps

    def start(self):
        """Start screen capture."""
        self._sct = mss.mss()

        # Get monitor info
        if self.config.region:
            # Custom region
            left, top, width, height = self.config.region
            self._monitor = {
                "left": left,
                "top": top,
                "width": width,
                "height": height,
            }
            print(f"🖥️ Capturing region: {width}x{height} at ({left}, {top})")
        else:
            # Use specified monitor
            monitors = self._sct.monitors
            if self.config.monitor < len(monitors):
                self._monitor = monitors[self.config.monitor]
            else:
                self._monitor = monitors[0]  # Fallback to all monitors

            print(
                f"🖥️ Capturing monitor {self.config.monitor}: {self._monitor['width']}x{self._monitor['height']}"
            )

        self._running = True
        print("✓ Screen capture ready")

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the screen.

        Returns:
            BGR frame as numpy array, or None on error
        """
        if not self._running or not self._sct:
            return None

        try:
            # Capture screenshot
            screenshot = self._sct.grab(self._monitor)

            # Convert to numpy array (BGRA format)
            frame = np.array(screenshot)

            # Convert BGRA to BGR
            frame = frame[:, :, :3]

            # Resize if requested
            if self.config.resize and CV2_AVAILABLE:
                frame = cv2.resize(frame, self.config.resize, interpolation=cv2.INTER_AREA)

            return frame

        except Exception as e:
            print(f"Capture error: {e}")
            return None

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Generator yielding captured frames.

        Yields:
            BGR frames as numpy arrays
        """
        last_capture = 0

        while self._running:
            now = time.perf_counter()

            # Rate limiting
            elapsed = now - last_capture
            if elapsed < self._frame_interval:
                time.sleep(self._frame_interval - elapsed)

            frame = self.capture_frame()
            if frame is not None:
                yield frame

            last_capture = time.perf_counter()

    def stop(self):
        """Stop screen capture."""
        self._running = False
        if self._sct:
            self._sct.close()
            self._sct = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._running

    @staticmethod
    def list_monitors() -> list:
        """List available monitors.

        Returns:
            List of monitor dictionaries with dimensions
        """
        with mss.mss() as sct:
            monitors = []
            for i, mon in enumerate(sct.monitors):
                monitors.append(
                    {
                        "index": i,
                        "left": mon["left"],
                        "top": mon["top"],
                        "width": mon["width"],
                        "height": mon["height"],
                        "name": f"Monitor {i}" if i > 0 else "All Monitors",
                    }
                )
            return monitors

    @staticmethod
    def select_region_interactive() -> Optional[Tuple[int, int, int, int]]:
        """Interactively select a screen region.

        Opens a window to select the capture region.

        Returns:
            (left, top, width, height) tuple or None if cancelled
        """
        if not CV2_AVAILABLE:
            print("Interactive region selection requires opencv-python")
            return None

        with mss.mss() as sct:
            # Capture full screen
            screenshot = sct.grab(sct.monitors[0])
            img = np.array(screenshot)[:, :, :3]

            # Resize for display
            scale = 0.5
            display = cv2.resize(img, None, fx=scale, fy=scale)

            print("Select region with mouse, press ENTER to confirm, ESC to cancel")

            # Use OpenCV's ROI selector
            roi = cv2.selectROI("Select Capture Region", display, fromCenter=False)
            cv2.destroyAllWindows()

            if roi[2] > 0 and roi[3] > 0:
                # Scale back to original coordinates
                left = int(roi[0] / scale)
                top = int(roi[1] / scale)
                width = int(roi[2] / scale)
                height = int(roi[3] / scale)
                return (left, top, width, height)

            return None


def create_screen_capture(
    monitor: int = 0,
    region: Optional[Tuple[int, int, int, int]] = None,
    fps: float = 30.0,
) -> ScreenCapture:
    """Factory function to create screen capture.

    Args:
        monitor: Monitor index (0 = all, 1 = first, etc.)
        region: Optional (left, top, width, height) region
        fps: Target capture FPS

    Returns:
        Configured ScreenCapture instance
    """
    config = ScreenConfig(
        monitor=monitor,
        region=region,
        fps=fps,
    )
    return ScreenCapture(config)
