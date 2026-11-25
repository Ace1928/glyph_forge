"""
Video Capture Module for Glyph Forge.

Unified interface for capturing video from:
- Local files (MP4, AVI, etc.)
- YouTube URLs (via yt-dlp)
- Direct video URLs
- Webcams
- Browser/screen capture (Netflix, etc.)
"""

import re
import subprocess
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Generator, Optional, Tuple

try:  # Optional dependency (streaming extra)
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
import numpy as np


class CaptureSource(Enum):
    """Types of video sources."""

    FILE = auto()
    YOUTUBE = auto()
    URL = auto()
    WEBCAM = auto()
    BROWSER = auto()
    SCREEN = auto()


@dataclass
class VideoInfo:
    """Metadata about a video source."""

    width: int
    height: int
    fps: float
    total_frames: int
    duration: float  # seconds
    source_type: CaptureSource
    source_path: str
    audio_url: Optional[str] = None
    title: Optional[str] = None


class VideoCapture:
    """Unified video capture interface.

    Supports multiple source types with automatic detection:
    - Local files: /path/to/video.mp4
    - YouTube: https://youtube.com/watch?v=...
    - Direct URLs: https://example.com/video.mp4
    - Webcam: "webcam", "webcam:0", "webcam:1"
    - Screen capture: "screen", "screen:0"

    Usage:
        with VideoCapture("https://youtube.com/watch?v=...") as cap:
            info = cap.info
            for success, frame in cap.frames():
                if success:
                    process(frame)
    """

    def __init__(self, source: str, max_resolution: int = 720):
        """Initialize capture from source.

        Args:
            source: Video source (file path, URL, "webcam", etc.)
            max_resolution: Maximum height to request (for streams)
        """
        self.source = source
        self.max_resolution = max_resolution

        self._cap: Optional[cv2.VideoCapture] = None
        self._info: Optional[VideoInfo] = None
        self._source_type: Optional[CaptureSource] = None
        self._video_url: Optional[str] = None
        self._audio_url: Optional[str] = None
        self._temp_files: list = []

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def info(self) -> VideoInfo:
        """Get video information."""
        if self._info is None:
            raise RuntimeError("Capture not opened. Call open() first.")
        return self._info

    def open(self):
        """Open the video source."""
        self._source_type = self._detect_source_type()

        if self._source_type == CaptureSource.YOUTUBE:
            self._open_youtube()
        elif self._source_type == CaptureSource.WEBCAM:
            self._open_webcam()
        elif self._source_type == CaptureSource.URL:
            self._open_url()
        elif self._source_type == CaptureSource.FILE:
            self._open_file()
        elif self._source_type == CaptureSource.BROWSER:
            self._open_browser()
        elif self._source_type == CaptureSource.SCREEN:
            self._open_screen()
        else:
            raise ValueError(f"Unknown source type: {self.source}")

        self._gather_info()

    def close(self):
        """Close the capture and cleanup."""
        # Close screen capture if active
        if hasattr(self, "_screen_capture") and self._screen_capture:
            self._screen_capture.stop()
            self._screen_capture = None

        # Close browser capture if active
        if hasattr(self, "_browser_capture") and self._browser_capture:
            self._browser_capture.stop()
            self._browser_capture = None

        if self._cap:
            self._cap.release()
            self._cap = None

        # Cleanup temp files
        for f in self._temp_files:
            try:
                Path(f).unlink()
            except OSError:
                pass
        self._temp_files.clear()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame.

        Returns:
            (success, frame) tuple
        """
        # Handle screen capture
        if hasattr(self, "_screen_capture") and self._screen_capture:
            frame = self._screen_capture.capture_frame()
            return (frame is not None, frame)

        # Handle browser capture
        if hasattr(self, "_browser_capture") and self._browser_capture:
            frame = self._browser_capture.capture_frame()
            return (frame is not None, frame)

        if not self.is_open:
            return False, None
        return self._cap.read()

    @property
    def is_open(self) -> bool:
        """Check if capture is open."""
        if hasattr(self, "_screen_capture") and self._screen_capture:
            return self._screen_capture.is_running
        if hasattr(self, "_browser_capture") and self._browser_capture:
            return self._browser_capture._running
        return self._cap is not None and self._cap.isOpened()

    def frames(self) -> Generator[Tuple[bool, Optional[np.ndarray]], None, None]:
        """Generator yielding all frames.

        Yields:
            (success, frame) tuples
        """
        while True:
            success, frame = self.read()
            if not success:
                break
            yield success, frame

    def seek(self, frame_idx: int) -> bool:
        """Seek to specific frame.

        Args:
            frame_idx: Frame index to seek to

        Returns:
            True if successful
        """
        if not self.is_open:
            return False
        return self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def _detect_source_type(self) -> CaptureSource:
        """Detect the type of video source."""
        source = self.source.lower()

        if source.startswith("webcam"):
            return CaptureSource.WEBCAM
        elif source.startswith("screen"):
            return CaptureSource.SCREEN
        elif "youtube.com" in source or "youtu.be" in source:
            return CaptureSource.YOUTUBE
        elif "netflix.com" in source or "hulu.com" in source or "disneyplus.com" in source:
            return CaptureSource.BROWSER
        elif source.startswith(("http://", "https://")):
            return CaptureSource.URL
        elif Path(self.source).exists():
            return CaptureSource.FILE
        else:
            # Try as file path anyway
            return CaptureSource.FILE

    def _open_youtube(self):
        """Open YouTube video using yt-dlp."""
        print("🎬 Extracting YouTube URL...")
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for YouTube streaming.")

        try:
            # Get video URL - prefer h264 codec (most compatible with OpenCV)
            # Try h264 first, then fall back to any codec
            formats_to_try = [
                f"bestvideo[height<={self.max_resolution}][vcodec^=avc1]",  # h264
                f"bestvideo[height<={self.max_resolution}][vcodec^=h264]",
                f"bestvideo[height<={self.max_resolution}]",
                f"best[height<={self.max_resolution}]",
                "best",
            ]

            for fmt in formats_to_try:
                result = subprocess.run(
                    ["yt-dlp", "-f", fmt, "-g", self.source],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0 and result.stdout.strip():
                    self._video_url = result.stdout.strip().split("\n")[0]
                    break

            if not self._video_url:
                raise RuntimeError("Could not extract video URL")

            # Get audio URL
            result = subprocess.run(
                ["yt-dlp", "-f", "bestaudio", "-g", self.source],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                self._audio_url = result.stdout.strip().split("\n")[0]

            # Get title
            result = subprocess.run(
                ["yt-dlp", "--get-title", self.source], capture_output=True, text=True, timeout=30
            )
            title = result.stdout.strip() if result.returncode == 0 else None

            print(f"📺 {title or 'YouTube video'}")

            self._cap = cv2.VideoCapture(self._video_url)
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open YouTube stream: {self._video_url[:50]}...")

        except subprocess.TimeoutExpired:
            raise RuntimeError("yt-dlp timed out")
        except FileNotFoundError:
            raise RuntimeError("yt-dlp not found. Install with: pip install yt-dlp")

    def _open_webcam(self):
        """Open webcam capture."""
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for webcam capture.")
        # Parse webcam index from "webcam:N" format
        match = re.match(r"webcam:?(\d*)", self.source.lower())
        idx = int(match.group(1)) if match and match.group(1) else 0

        print(f"📷 Opening webcam {idx}...")
        self._cap = cv2.VideoCapture(idx)

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open webcam {idx}")

        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def _open_url(self):
        """Open direct URL."""
        print("🌐 Opening URL stream...")
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for URL streaming.")
        self._video_url = self.source
        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open URL: {self.source}")

    def _open_file(self):
        """Open local file."""
        path = Path(self.source)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {self.source}")

        print(f"📁 Opening: {path.name}")
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for file playback.")
        self._cap = cv2.VideoCapture(str(path))

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open file: {self.source}")

        # Extract audio for local files
        self._audio_url = str(path)  # ffplay can handle the video file directly for audio

    def _open_browser(self):
        """Open browser-based streaming (Netflix, etc.).

        This requires playwright for browser automation and screen capture.
        """
        print(f"🌐 Browser capture: {self.source}")

        try:
            from .browser import BrowserConfig, create_browser_capture

            config = BrowserConfig(
                headless=False,  # Netflix blocks headless
                width=1920,
                height=1080,
                capture_fps=30.0,
            )

            self._browser_capture = create_browser_capture(self.source, config)
            self._browser_capture.start()

            # Create a fake VideoCapture interface
            # Store the browser capture for frame reading
            self._source_type = CaptureSource.BROWSER

        except ImportError as e:
            raise RuntimeError(
                f"Browser capture requires playwright: {e}. Install with: pip install playwright && playwright install"
            )

    def _open_screen(self):
        """Open screen capture."""
        print("🖥️ Screen capture mode")

        try:
            from .screen import ScreenCapture, ScreenConfig

            # Parse screen source (e.g., "screen", "screen:1", "screen:region:100,100,1920,1080")
            parts = self.source.lower().split(":")
            monitor = 0
            region = None

            if len(parts) >= 2:
                if parts[1] == "region" and len(parts) >= 3:
                    # Parse region: screen:region:left,top,width,height
                    coords = parts[2].split(",")
                    if len(coords) == 4:
                        region = tuple(int(x) for x in coords)
                else:
                    # Parse monitor index: screen:1
                    try:
                        monitor = int(parts[1])
                    except ValueError:
                        pass

            config = ScreenConfig(
                monitor=monitor,
                region=region,
                fps=30.0,
            )

            self._screen_capture = ScreenCapture(config)
            self._screen_capture.start()

        except ImportError:
            raise RuntimeError("Screen capture requires mss. Install with: pip install mss")

    def _gather_info(self):
        """Gather video information from capture."""
        # Handle screen capture
        if hasattr(self, "_screen_capture") and self._screen_capture:
            from .screen import ScreenConfig

            config = getattr(self._screen_capture, "config", ScreenConfig())
            mon = getattr(self._screen_capture, "_monitor", {})

            self._info = VideoInfo(
                width=mon.get("width", 1920),
                height=mon.get("height", 1080),
                fps=config.fps,
                total_frames=0,  # Unknown for live capture
                duration=0,
                source_type=CaptureSource.SCREEN,
                source_path=self.source,
                audio_url=None,  # No audio for screen capture (use system audio)
                title="Screen Capture",
            )
            return

        # Handle browser capture
        if hasattr(self, "_browser_capture") and self._browser_capture:
            from .browser import BrowserConfig

            config = getattr(self._browser_capture, "config", BrowserConfig())

            self._info = VideoInfo(
                width=config.width,
                height=config.height,
                fps=config.capture_fps,
                total_frames=0,  # Unknown for live capture
                duration=0,
                source_type=CaptureSource.BROWSER,
                source_path=self.source,
                audio_url=None,  # No separate audio for browser
                title=f"Browser: {self.source[:50]}",
            )
            return

        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required to gather video info.")
        if not self._cap:
            raise RuntimeError("Capture not opened")

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Extract title from YouTube or filename
        title = None
        if self._source_type == CaptureSource.FILE:
            title = Path(self.source).stem

        self._info = VideoInfo(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            source_type=self._source_type,
            source_path=self.source,
            audio_url=self._audio_url,
            title=title,
        )


class BrowserCapture:
    """Browser-based video capture for streaming services.

    Uses Playwright for browser automation and captures video frames
    directly from the browser rendering.

    Note: This is for personal/educational use only.
    Respect content licensing and terms of service.
    """

    def __init__(self, url: str, headless: bool = True):
        """Initialize browser capture.

        Args:
            url: URL to capture (Netflix, etc.)
            headless: Run browser in headless mode
        """
        self.url = url
        self.headless = headless
        self._browser = None
        self._page = None
        self._running = False

    def start(self):
        """Start browser and navigate to URL."""
        try:
            from playwright.sync_api import sync_playwright

            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self.headless)
            self._page = self._browser.new_page()

            # Navigate to URL
            self._page.goto(self.url)
            self._running = True

            print(f"🌐 Browser opened: {self.url}")

        except Exception as e:
            raise RuntimeError(f"Failed to start browser: {e}")

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture current frame from browser.

        Returns:
            Frame as numpy array or None
        """
        if not self._running or not self._page:
            return None

        try:
            # Take screenshot
            screenshot = self._page.screenshot()

            # Convert to numpy array
            import io

            from PIL import Image

            img = Image.open(io.BytesIO(screenshot))
            frame = np.array(img)

            # Convert RGBA to BGR for OpenCV
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            return frame

        except Exception as e:
            print(f"Capture error: {e}")
            return None

    def stop(self):
        """Stop browser capture."""
        self._running = False

        if self._browser:
            self._browser.close()
        if hasattr(self, "_playwright"):
            self._playwright.stop()
