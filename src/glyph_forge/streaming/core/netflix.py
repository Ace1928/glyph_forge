"""
Netflix Streaming Capture for Glyph Forge.

A robust system that:
1. Launches system Firefox with Netflix URL
2. Waits for page load and starts playback
3. Goes fullscreen
4. Captures screen + audio with FFmpeg
5. Feeds frames to glyph renderer

Uses system Firefox (already logged in) and FFmpeg for reliable capture.
"""

import os
import re
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import numpy as np

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class NetflixConfig:
    """Netflix capture configuration."""

    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    audio_device: str = "default"  # PulseAudio device
    output_dir: Path = Path(tempfile.gettempdir()) / "glyph_netflix"
    fullscreen: bool = True
    wait_for_load: float = 10.0  # Seconds to wait for Netflix to load


class FFmpegCapture:
    """Captures screen and audio using FFmpeg.

    FFmpeg is the most reliable way to capture screen + audio on Linux.
    This class manages the FFmpeg process and provides frame access.
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
        audio_device: str = "default",
        output_path: Optional[Path] = None,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.audio_device = audio_device
        # Ensure output_path is a Path object
        if output_path is None:
            self.output_path = Path(tempfile.gettempdir()) / "glyph_capture.mp4"
        elif isinstance(output_path, str):
            self.output_path = Path(output_path)
        else:
            self.output_path = output_path

        self._process: Optional[subprocess.Popen] = None
        self._running = False

    def start_recording(self, duration: Optional[float] = None) -> bool:
        """Start recording screen + audio to file.

        Args:
            duration: Optional duration limit in seconds

        Returns:
            True if started successfully
        """
        cmd = [
            "ffmpeg",
            "-y",
            # Video input (X11 screen grab)
            "-f",
            "x11grab",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(int(self.fps)),
            "-i",
            os.environ.get("DISPLAY", ":0"),
            # Audio input (PulseAudio)
            "-f",
            "pulse",
            "-i",
            self.audio_device,
            # Video encoding
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            # Audio encoding
            "-c:a",
            "aac",
            "-b:a",
            "192k",
        ]

        if duration:
            cmd.extend(["-t", str(duration)])

        cmd.append(str(self.output_path))

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._running = True
            print(f"📹 Recording started: {self.output_path}")
            return True
        except Exception as e:
            print(f"Failed to start recording: {e}")
            return False

    def stop_recording(self) -> Optional[Path]:
        """Stop recording and return output path.

        Returns:
            Path to recorded file, or None on error
        """
        if not self._process:
            return None

        self._running = False

        try:
            # Send 'q' to ffmpeg to gracefully stop
            self._process.stdin.write(b"q")
            self._process.stdin.flush()
            self._process.wait(timeout=10)
        except Exception:
            # Force kill if graceful stop fails
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

        self._process = None

        if self.output_path.exists():
            print(f"✓ Recording saved: {self.output_path}")
            return self.output_path
        return None

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._running and self._process is not None


class FirefoxController:
    """Controls system Firefox for Netflix playback.

    Uses xdotool/ydotool for automation since we need the real Firefox
    with existing Netflix login.
    """

    def __init__(self):
        self._firefox_pid: Optional[int] = None
        self._window_id: Optional[str] = None

    def launch(self) -> bool:
        """Launch Firefox (no URL).

        Returns:
            True if launched successfully
        """
        return self.launch_firefox("about:blank")

    def navigate(self, url: str) -> bool:
        """Navigate to URL using xdotool.

        Args:
            url: URL to navigate to

        Returns:
            True if successful
        """
        if not self._window_id:
            if not self.wait_for_window(timeout=10):
                return False

        if not self.focus_window():
            return False

        try:
            # Ctrl+L to focus address bar, then type URL
            subprocess.run(["xdotool", "key", "ctrl+l"], capture_output=True, timeout=5)
            time.sleep(0.2)
            # Clear any existing URL
            subprocess.run(["xdotool", "key", "ctrl+a"], capture_output=True, timeout=5)
            time.sleep(0.1)
            # Type the URL
            subprocess.run(
                ["xdotool", "type", "--delay", "10", url], capture_output=True, timeout=30
            )
            time.sleep(0.2)
            # Press Enter to go
            subprocess.run(["xdotool", "key", "Return"], capture_output=True, timeout=5)
            print(f"📍 Navigated to: {url[:60]}...")
            return True
        except Exception as e:
            print(f"Navigation failed: {e}")
            return False

    def launch_firefox(self, url: str) -> bool:
        """Launch Firefox with URL.

        Args:
            url: URL to open

        Returns:
            True if launched successfully
        """
        try:
            # Launch Firefox
            process = subprocess.Popen(
                ["firefox", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._firefox_pid = process.pid
            print(f"🦊 Launched Firefox (PID: {self._firefox_pid})")
            return True
        except Exception as e:
            print(f"Failed to launch Firefox: {e}")
            return False

    def wait_for_window(self, timeout: float = 30.0) -> bool:
        """Wait for Firefox window to appear.

        Args:
            timeout: Maximum time to wait

        Returns:
            True if window found
        """
        start = time.time()
        while time.time() - start < timeout:
            try:
                result = subprocess.run(
                    ["xdotool", "search", "--name", "Mozilla Firefox"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    self._window_id = result.stdout.strip().split("\n")[0]
                    print(f"✓ Found Firefox window: {self._window_id}")
                    return True
            except Exception:
                pass
            time.sleep(0.5)

        return False

    def focus_window(self) -> bool:
        """Focus the Firefox window."""
        if not self._window_id:
            return False

        try:
            subprocess.run(
                ["xdotool", "windowactivate", self._window_id], capture_output=True, timeout=5
            )
            time.sleep(0.3)
            return True
        except Exception as e:
            print(f"Failed to focus window: {e}")
            return False

    def fullscreen(self) -> bool:
        """Toggle fullscreen (F11 or F key for Netflix)."""
        if not self.focus_window():
            return False

        try:
            # Press F for Netflix fullscreen
            subprocess.run(["xdotool", "key", "f"], capture_output=True, timeout=5)
            print("⬜ Toggled fullscreen")
            time.sleep(1)
            return True
        except Exception as e:
            print(f"Fullscreen failed: {e}")
            return False

    def press_space(self) -> bool:
        """Press space to play/pause."""
        if not self.focus_window():
            return False

        try:
            subprocess.run(["xdotool", "key", "space"], capture_output=True, timeout=5)
            print("▶ Toggled play/pause")
            return True
        except Exception as e:
            print(f"Space press failed: {e}")
            return False

    def close(self):
        """Close Firefox."""
        if self._firefox_pid:
            try:
                os.kill(self._firefox_pid, signal.SIGTERM)
            except OSError:
                pass


class NetflixCapture:
    """Complete Netflix capture system.

    Orchestrates Firefox control and FFmpeg capture to record
    Netflix streams for glyph rendering.

    Usage:
        cap = NetflixCapture()

        # Start Netflix and recording
        cap.start('https://netflix.com/watch/...')

        # Get frames for rendering (from the recording)
        for frame in cap.frames():
            glyph_output = renderer.render(frame)

        cap.stop()
    """

    def __init__(self, config: Optional[NetflixConfig] = None):
        self.config = config or NetflixConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self._firefox = FirefoxController()
        self._ffmpeg = FFmpegCapture(
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
            audio_device=self.config.audio_device,
        )
        self._running = False
        self._recording_path: Optional[Path] = None

    def start(self, url: str, duration: Optional[float] = None) -> bool:
        """Start Netflix playback and recording.

        Args:
            url: Netflix watch URL
            duration: Optional recording duration in seconds

        Returns:
            True if started successfully
        """
        print(f"🎬 Starting Netflix capture: {url[:60]}...")

        # Generate output path
        match = re.search(r"/watch/(\d+)", url)
        video_id = match.group(1) if match else "unknown"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._recording_path = self.config.output_dir / f"netflix_{video_id}_{timestamp}.mp4"
        self._ffmpeg.output_path = self._recording_path

        # Launch Firefox
        if not self._firefox.launch_firefox(url):
            return False

        # Wait for window
        print("⏳ Waiting for Firefox to load...")
        if not self._firefox.wait_for_window(timeout=30):
            print("❌ Firefox window not found")
            return False

        # Wait for Netflix to load
        time.sleep(self.config.wait_for_load)

        # Go fullscreen
        if self.config.fullscreen:
            self._firefox.fullscreen()
            time.sleep(2)

        # Ensure playing
        self._firefox.press_space()
        time.sleep(1)

        # Start recording
        if not self._ffmpeg.start_recording(duration):
            return False

        self._running = True
        print("✓ Capture running!")
        return True

    def stop(self) -> Optional[Path]:
        """Stop capture and return recording path.

        Returns:
            Path to recorded video with audio
        """
        self._running = False

        # Stop recording
        output = self._ffmpeg.stop_recording()

        # Close Firefox (optional - user may want to keep it)
        # self._firefox.close()

        return output

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Generate frames from the recording.

        Note: This reads from the recorded file, so it's best used
        after recording is complete, or for live preview during
        recording (with some latency).

        Yields:
            BGR frames as numpy arrays
        """
        if not self._recording_path or not self._recording_path.exists():
            return

        if not CV2_AVAILABLE:
            print("OpenCV required for frame extraction")
            return

        cap = cv2.VideoCapture(str(self._recording_path))
        if not cap.isOpened():
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()

    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._running

    @property
    def recording_path(self) -> Optional[Path]:
        """Get path to recording."""
        return self._recording_path


def capture_netflix(
    url: str,
    duration: float = 60.0,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Quick function to capture Netflix video.

    Args:
        url: Netflix watch URL
        duration: Recording duration in seconds
        output_dir: Output directory

    Returns:
        Path to recorded video, or None on error
    """
    config = NetflixConfig()
    if output_dir:
        config.output_dir = output_dir

    cap = NetflixCapture(config)

    if not cap.start(url, duration):
        return None

    try:
        # Wait for recording to complete
        print(f"⏱️ Recording for {duration}s...")
        time.sleep(duration + 2)
    except KeyboardInterrupt:
        print("\n⏹ Interrupted")

    return cap.stop()
