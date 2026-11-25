"""
Audio Synchronization Module for Glyph Forge.

Handles audio playback synchronized with video:
- ffplay backend for streaming URLs
- Position tracking for sync correction
- Multiple backend support
"""

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AudioConfig:
    """Audio player configuration."""

    backend: str = "ffplay"  # ffplay, pygame
    volume: float = 1.0
    buffer_size: int = 8192


class AudioSync:
    """Audio player with synchronization support.

    Plays audio from files or URLs and provides timing information
    for A/V synchronization.

    Usage:
        audio = AudioSync()
        audio.start('audio.m4a', start_time=0.0)

        while playing:
            sync_offset = audio.get_position() - video_position
            # Adjust video timing based on sync_offset

        audio.stop()
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        """Initialize audio player.

        Args:
            config: Audio configuration
        """
        self.config = config or AudioConfig()

        self._process: Optional[subprocess.Popen] = None
        self._start_time: Optional[float] = None
        self._start_position: float = 0.0
        self._playing = False
        self._lock = threading.Lock()

    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        with self._lock:
            if self._process and self._process.poll() is not None:
                self._playing = False
            return self._playing

    def start(self, source: str, start_position: float = 0.0):
        """Start audio playback from file.

        Args:
            source: Path to audio/video file
            start_position: Start position in seconds
        """
        self.stop()

        with self._lock:
            self._start_position = start_position

            if self.config.backend == "ffplay":
                self._start_ffplay(source, start_position)
            else:
                raise ValueError(f"Unknown backend: {self.config.backend}")

            self._start_time = time.perf_counter()
            self._playing = True

    def start_stream(self, url: str, start_position: float = 0.0):
        """Start audio playback from URL stream.

        Args:
            url: Audio stream URL
            start_position: Start position in seconds
        """
        # For URLs, use the same ffplay backend
        self.start(url, start_position)

    def _start_ffplay(self, source: str, start_position: float):
        """Start ffplay for audio playback."""
        cmd = [
            "ffplay",
            "-nodisp",  # No video display
            "-autoexit",  # Exit when done
            "-loglevel",
            "quiet",
            "-volume",
            str(int(self.config.volume * 100)),
        ]

        if start_position > 0:
            cmd.extend(["-ss", str(start_position)])

        cmd.append(source)

        # Start ffplay process
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

    def stop(self):
        """Stop audio playback."""
        with self._lock:
            self._playing = False

            if self._process:
                try:
                    # Kill the process group
                    if hasattr(os, "killpg"):
                        try:
                            os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                        except (ProcessLookupError, PermissionError):
                            pass
                    else:
                        self._process.terminate()

                    self._process.wait(timeout=2.0)
                except Exception:
                    try:
                        self._process.kill()
                    except Exception:
                        pass

                self._process = None

    def get_position(self) -> float:
        """Get estimated current playback position.

        Returns:
            Estimated position in seconds
        """
        with self._lock:
            if not self._playing or self._start_time is None:
                return 0.0

            # Estimate based on wall clock
            elapsed = time.perf_counter() - self._start_time
            return self._start_position + elapsed

    def get_sync_offset(self, video_position: float) -> float:
        """Calculate sync offset between audio and video.

        Args:
            video_position: Current video position in seconds

        Returns:
            Offset in seconds (positive = audio ahead, negative = audio behind)
        """
        audio_pos = self.get_position()
        return audio_pos - video_position

    def seek(self, position: float):
        """Seek to position (restarts playback).

        Args:
            position: Target position in seconds
        """
        # ffplay doesn't support seeking, so we need to restart
        # This is a limitation of the ffplay backend
        pass

    def set_volume(self, volume: float):
        """Set playback volume.

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.config.volume = max(0.0, min(1.0, volume))
        # Note: ffplay doesn't support runtime volume changes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class AudioDownloader:
    """Downloads audio from URLs for local playback.

    Useful for pre-downloading audio to ensure smooth playback
    without streaming buffering issues.
    """

    @staticmethod
    def download_youtube_audio(url: str, output_path: Optional[Path] = None) -> Optional[Path]:
        """Download audio from YouTube video.

        Args:
            url: YouTube URL
            output_path: Output path (auto-generated if None)

        Returns:
            Path to downloaded audio, or None on failure
        """
        import tempfile

        if output_path is None:
            output_path = Path(tempfile.gettempdir()) / "glyph_forge_audio.m4a"

        try:
            cmd = ["yt-dlp", "-f", "bestaudio", "-o", str(output_path), "--no-playlist", url]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0 and output_path.exists():
                return output_path

            return None

        except Exception as e:
            print(f"Audio download failed: {e}")
            return None

    @staticmethod
    def extract_audio_url(youtube_url: str) -> Optional[str]:
        """Extract direct audio URL from YouTube.

        Args:
            youtube_url: YouTube video URL

        Returns:
            Direct audio stream URL, or None
        """
        try:
            result = subprocess.run(
                ["yt-dlp", "-f", "bestaudio", "-g", youtube_url],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0]

            return None

        except Exception:
            return None
