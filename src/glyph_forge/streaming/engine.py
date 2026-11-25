"""
Unified Glyph Streaming Engine.

This module consolidates the best features of all previous engines:
- Ultra-high performance rendering (Vectorized, Delta Compression)
- Smart Buffering (Adaptive, Pre-buffering)
- Recording support (HD output)
- Audio synchronization
- Modular configuration

"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    import cv2

    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from .core.buffer import AdaptiveBuffer
from .core.recorder import GlyphRecorder, RecorderConfig
from .core.renderer import GlyphRenderer, RenderConfig
from .core.sync import AudioDownloader, AudioSync
from .extractors import VideoSourceExtractor, YouTubeExtractor
from .naming import build_metadata, build_output_path, write_metadata

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════


@dataclass
class UnifiedStreamConfig:
    """Comprehensive streaming configuration."""

    # Source
    source: Union[str, int]

    # Resolution & Quality
    resolution: str = "720p"  # auto, 1080p, 720p, 480p
    target_fps: int = 30
    render_mode: str = "gradient"  # gradient, braille, hybrid
    color_mode: str = "ansi256"  # truecolor, ansi256, none

    # Performance
    use_delta_compression: bool = True
    use_vectorized_render: bool = True
    frame_backend: str = "auto"  # auto, cv2, ffmpeg

    # Render tuning
    render_dithering: bool = True
    render_gamma: float = 1.15
    render_contrast: float = 0.98
    render_brightness: float = 0.02
    render_auto_contrast: bool = True

    # Buffering
    buffer_seconds: float = 30.0
    prebuffer_seconds: float = 5.0

    # Audio
    audio_enabled: bool = True
    audio_sync: bool = True
    mux_audio: bool = True

    # Playlist handling (YouTube)
    playlist_max_items: Optional[int] = None
    playlist_start: Optional[int] = None
    playlist_end: Optional[int] = None

    # YouTube/yt-dlp options
    yt_cookies: Optional[str] = None
    yt_cookies_from_browser: Optional[str] = None
    yt_user_agent: Optional[str] = None
    yt_proxy: Optional[str] = None
    yt_skip_authcheck: bool = False
    yt_player_client: Optional[str] = None

    # Recording
    record_enabled: bool = False
    record_path: Optional[str] = None
    output_dir: Optional[str] = None
    overwrite_output: bool = False
    write_metadata: bool = True
    render_then_play: bool = False
    play_after_render: bool = True

    # Display
    show_metrics: bool = True
    show_border: bool = False
    terminal_control: bool = True

    # Limits
    max_duration_seconds: Optional[float] = None

    # Terminal override (None = auto)
    terminal_width: Optional[int] = None
    terminal_height: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate stream settings before opening files, devices, or networks."""

        if isinstance(self.source, str) and not self.source.strip():
            raise ValueError("source cannot be empty")
        if self.resolution not in {"auto", "1080p", "720p", "480p", "360p"}:
            raise ValueError(f"unsupported resolution: {self.resolution}")
        if not 1 <= self.target_fps <= 240:
            raise ValueError("target_fps must be between 1 and 240")
        if self.buffer_seconds < 0 or self.prebuffer_seconds < 0:
            raise ValueError("buffer durations cannot be negative")
        if self.max_duration_seconds is not None and self.max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be greater than zero")
        if self.frame_backend not in {"auto", "cv2", "ffmpeg"}:
            raise ValueError(f"unsupported frame backend: {self.frame_backend}")
        for name, value in (
            ("terminal_width", self.terminal_width),
            ("terminal_height", self.terminal_height),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive")

    def get_pixel_resolution(self) -> Tuple[int, int]:
        """Get target pixel resolution."""
        resolutions = {
            "1080p": (1920, 1080),
            "720p": (1280, 720),
            "480p": (854, 480),
            "360p": (640, 360),
        }
        if self.resolution in resolutions:
            return resolutions[self.resolution]
        return resolutions["720p"]


# ═══════════════════════════════════════════════════════════════
# Delta Encoder (Optimized)
# ═══════════════════════════════════════════════════════════════


class DeltaEncoder:
    """Optimizes terminal output by only printing changed lines."""

    def __init__(self):
        self._prev_lines: List[str] = []

    def encode(self, lines: List[str]) -> List[str]:
        """Return full frame with optimization where possible.

        Note: Terminal delta updates are tricky without ncurses.
        For now, we simply return the full frame, but this hook
        allows for future cursor-movement based optimizations.
        """
        # In a raw ANSI stream, clearing screen and reprinting is often
        # cleaner than jumping around, unless we use tput/cup.
        # We'll stick to full frame rewrite with Home cursor for now
        # to avoid artifacting, but we can detect static frames.

        if self._prev_lines == lines:
            return []  # No update needed

        self._prev_lines = list(lines)
        return lines


# ═══════════════════════════════════════════════════════════════
# Unified Engine
# ═══════════════════════════════════════════════════════════════


class UnifiedStreamEngine:
    """The converged streaming engine."""

    def __init__(self, config: UnifiedStreamConfig):
        self.config = config

        # Components
        render_cfg = RenderConfig(
            mode=config.render_mode,
            color=config.color_mode,
            dithering=config.render_dithering,
            gamma=config.render_gamma,
            contrast=config.render_contrast,
            brightness=config.render_brightness,
            auto_contrast=config.render_auto_contrast,
        )
        self.renderer = GlyphRenderer(render_cfg)

        self.buffer = AdaptiveBuffer(
            target_fps=float(config.target_fps),
            min_buffer_seconds=config.prebuffer_seconds,
            target_buffer_seconds=config.buffer_seconds,
        )

        self.recorder: Optional[GlyphRecorder] = None
        self.audio: Optional[AudioSync] = None
        self.delta_encoder = DeltaEncoder() if config.use_delta_compression else None
        self.audio_path: Optional[Path] = None
        self.audio_source_for_mux: Optional[str] = None
        self.total_frames: Optional[int] = None
        self.last_recording_path: Optional[Path] = None
        self.last_record_text: Optional[str] = None
        self.last_metadata_path: Optional[Path] = None
        self.extraction_info = None
        self._ffmpeg_process: Optional[subprocess.Popen] = None

        # State
        self._running = False
        self._stop_event = threading.Event()

    def run(self) -> Optional[Path]:
        """Main execution entry point."""
        if not HAS_OPENCV and self.config.frame_backend != "ffmpeg":
            print("Error: OpenCV (cv2) is required.")
            return None

        try:
            self._setup_pipeline()
            self._stream_loop()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self._cleanup()
            if self.recorder and self.config.play_after_render:
                self._play_recording()
        return self.last_recording_path

    def _setup_pipeline(self):
        """Initialize all pipeline components."""
        # 1. Source Extraction
        print("Initializing source...")
        resolution_hint: Optional[int] = None
        if isinstance(self.config.resolution, str) and self.config.resolution.endswith("p"):
            try:
                resolution_hint = int(self.config.resolution.rstrip("p"))
            except ValueError:
                resolution_hint = None
        extractor = VideoSourceExtractor()
        info = extractor.extract(
            self.config.source,
            resolution=resolution_hint,
            include_audio=self.config.audio_enabled,
            playlist_max_items=self.config.playlist_max_items,
            playlist_start=self.config.playlist_start,
            playlist_end=self.config.playlist_end,
            yt_cookies=self.config.yt_cookies,
            yt_cookies_from_browser=self.config.yt_cookies_from_browser,
            yt_user_agent=self.config.yt_user_agent,
            yt_proxy=self.config.yt_proxy,
            yt_skip_authcheck=self.config.yt_skip_authcheck,
            yt_player_client=self.config.yt_player_client,
        )
        self.extraction_info = info

        self.source_fps = info.fps or 30.0
        self.source_path = info.video_url
        if info.format == "webcam" and isinstance(info.video_url, str) and info.video_url.isdigit():
            self.source_path = int(info.video_url)
        elif isinstance(self.config.source, int):
            self.source_path = self.config.source
        if not self.source_path:
            raise RuntimeError("No video source URL available.")
        if info.duration and self.config.target_fps:
            self.total_frames = int(info.duration * self.config.target_fps)

        # 2. Audio Setup
        if (
            self.config.audio_enabled
            and self.config.record_enabled
            and self.config.mux_audio
            and not info.is_live
        ):
            self.audio_source_for_mux = self._select_audio_source_for_mux(info)

        if self.config.audio_enabled and info.audio_url and not self.config.render_then_play:
            self.audio = AudioSync()
            # If it's a URL, we might stream or download.
            # AudioSync supports local files and URLs via ffplay backend
            if info.audio_url.startswith(("http://", "https://")):
                self.audio.start_stream(info.audio_url)
            else:
                self.audio.start(info.audio_url)
        elif self.config.audio_enabled and info.audio_url and self.config.render_then_play:
            audio_source = self.audio_source_for_mux or info.audio_url
            self.audio_path = self._prepare_audio_for_mux(audio_source)

        # 3. Recorder Setup
        if self.config.record_enabled:
            out_dir = Path(self.config.output_dir) if self.config.output_dir else None
            path = build_output_path(
                source=str(self.config.source),
                title=info.title,
                output_dir=out_dir,
                ext="mp4",
                output=self.config.record_path,
                overwrite=self.config.overwrite_output,
            )
            rec_cfg = RecorderConfig(output_path=path, fps=self.config.target_fps, font_size=14)
            self.recorder = GlyphRecorder(rec_cfg)
            self.last_recording_path = path

        # 4. Buffering
        # We start a background thread to feed the buffer
        self._start_buffering_thread()

    def _ffmpeg_available(self) -> bool:
        return shutil.which("ffmpeg") is not None

    def _ffprobe_stream_info(
        self, source: str | int
    ) -> tuple[Optional[int], Optional[int], Optional[float]]:
        if isinstance(source, int):
            return None, None, None
        if shutil.which("ffprobe") is None:
            return None, None, None
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-of",
            "json",
            str(source),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None, None, None
        try:
            payload = json.loads(result.stdout)
            streams = payload.get("streams") or []
            if not streams:
                return None, None, None
            stream = streams[0]
            width = int(stream.get("width")) if stream.get("width") else None
            height = int(stream.get("height")) if stream.get("height") else None
            fps = None
            rate = stream.get("avg_frame_rate")
            if rate and isinstance(rate, str) and "/" in rate:
                num, den = rate.split("/", 1)
                if float(den) != 0:
                    fps = float(num) / float(den)
            return width, height, fps
        except Exception:
            return None, None, None

    def _select_frame_backend(self) -> str:
        backend = (self.config.frame_backend or "auto").lower()
        if backend not in {"auto", "cv2", "ffmpeg"}:
            raise RuntimeError(f"Unsupported frame backend: {backend}")
        is_webcam = isinstance(self.config.source, int) or (
            getattr(self.extraction_info, "format", "") == "webcam"
        )
        if backend == "auto":
            if self._ffmpeg_available() and not is_webcam:
                return "ffmpeg"
            return "cv2"
        if backend == "ffmpeg":
            if is_webcam:
                return "cv2"
            if not self._ffmpeg_available():
                raise RuntimeError("ffmpeg not available for frame backend")
        return backend

    def _start_buffering_thread(self):
        """Start the producer thread."""

        def frame_producer():
            backend = self._select_frame_backend()
            if backend == "ffmpeg":
                width = getattr(self.extraction_info, "width", None)
                height = getattr(self.extraction_info, "height", None)
                if width is None or height is None:
                    probe_w, probe_h, probe_fps = self._ffprobe_stream_info(self.source_path)
                    width = width or probe_w
                    height = height or probe_h
                    if probe_fps and not getattr(self.extraction_info, "fps", None):
                        self.source_fps = probe_fps
                if not width or not height:
                    raise RuntimeError("Unable to determine stream dimensions for ffmpeg backend.")
                for success, frame in self._ffmpeg_frames(self.source_path, width, height):
                    yield success, frame
                yield False, None
                return

            cap = cv2.VideoCapture(self.source_path)
            if self.total_frames is None:
                total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if total and total > 0:
                    self.total_frames = int(total)
            start_time = time.time()
            while not self._stop_event.is_set():
                if self.config.max_duration_seconds:
                    if time.time() - start_time >= self.config.max_duration_seconds:
                        break
                ret, frame = cap.read()
                if not ret:
                    break
                yield True, frame
            cap.release()
            yield False, None

        def render_wrapper(frame, w, h):
            # This is called by the buffer thread
            return self.renderer.render(frame, width=w, height=h)

        # Calculate dimensions
        w, h = self._get_terminal_dims()

        # Determine recording resolution (often higher than terminal)
        rec_w, rec_h = (w, h)
        if self.recorder:
            # Basic logic: 2x terminal resolution for recording?
            # Or match config resolution.
            rw, rh = self.config.get_pixel_resolution()
            # Approximate char dimensions: 8x16 pixels
            rec_w = rw // 8
            rec_h = rh // 16

        self.buffer.start_buffering(
            frame_generator=frame_producer().__next__,
            render_func=render_wrapper,
            display_size=(w, h),
            record_size=(rec_w, rec_h) if self.recorder else None,
            total_frames=0,  # Unknown for stream
        )

    def _ffmpeg_frames(
        self,
        source: str | int,
        width: int,
        height: int,
    ):
        if isinstance(source, int):
            raise RuntimeError("ffmpeg backend does not support webcam sources.")
        frame_size = width * height * 3
        cmd = ["ffmpeg", "-loglevel", "error"]
        if getattr(self.extraction_info, "is_live", False):
            cmd.append("-re")
        cmd.extend(["-i", str(source)])
        if self.config.target_fps:
            cmd.extend(["-vf", f"fps={self.config.target_fps}"])
        cmd.extend(["-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=max(1024 * 1024, frame_size * 2),
        )
        self._ffmpeg_process = proc
        try:
            while not self._stop_event.is_set():
                if not proc.stdout:
                    break
                raw = proc.stdout.read(frame_size)
                if not raw or len(raw) < frame_size:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                yield True, frame
        finally:
            self._terminate_ffmpeg(proc)

    def _get_terminal_dims(self) -> Tuple[int, int]:
        """Get effective terminal dimensions in characters."""
        if self.config.terminal_width and self.config.terminal_height:
            return self.config.terminal_width, self.config.terminal_height

        try:
            cols, lines = os.get_terminal_size()
            return cols, lines - 2  # Reserve space for status
        except OSError:
            return 120, 40

    def _stream_loop(self):
        """Consumer loop."""
        self._running = True

        # Wait for prebuffer
        print("Pre-buffering...")
        while not self.buffer.ready_for_playback and not self._stop_event.is_set():
            time.sleep(0.1)

        print("Starting playback!")
        if not self.config.render_then_play and self.config.terminal_control:
            sys.stdout.write("\033[2J")  # Clear screen

        time.time()
        frame_interval = 1.0 / self.config.target_fps
        stream_start = time.time()

        progress = None
        task_id = None
        if self.total_frames and self.config.terminal_control:
            from rich.progress import BarColumn, Progress, TimeElapsedColumn, TimeRemainingColumn

            progress = Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            progress.start()
            task_id = progress.add_task("Rendering", total=self.total_frames)

        while self._running and not self._stop_event.is_set():
            now = time.time()
            if self.config.max_duration_seconds:
                if now - stream_start >= self.config.max_duration_seconds:
                    break

            # 1. Sync check
            if self.audio:
                # Logic to skip frames if behind audio
                pass

            # 2. Get Frame
            frame_obj = self.buffer.get_next_frame(timeout=0.5)
            if not frame_obj:
                if not self.buffer.has_frames:
                    break  # End of stream
                continue

            # 3. Display
            if not self.config.render_then_play:
                self._display_frame(frame_obj.display_data)

            # 4. Record
            if self.recorder:
                record_payload = frame_obj.record_data or frame_obj.display_data
                if record_payload:
                    self.recorder.write_frame(record_payload)
                    self.last_record_text = record_payload
                if progress and task_id is not None:
                    progress.update(task_id, advance=1)

            # 5. Timing
            elapsed = time.time() - now
            sleep_time = max(0, frame_interval - elapsed)
            time.sleep(sleep_time)

        if progress:
            progress.stop()

    def _display_frame(self, data: str):
        """Output frame to terminal."""
        if self.config.terminal_control:
            sys.stdout.write("\033[H")  # Move home
            sys.stdout.write(data)
        else:
            sys.stdout.write(data)
            sys.stdout.write("\n")
        sys.stdout.flush()

    def _cleanup(self):
        self._stop_event.set()
        self.buffer.stop()
        if self.recorder:
            self.recorder.close()
        if self.audio:
            self.audio.stop()
        if self._ffmpeg_process:
            self._terminate_ffmpeg(self._ffmpeg_process)
        if not self.config.render_then_play and self.config.terminal_control:
            sys.stdout.write("\033[0m\033[2J")
        print("Stream ended.")

        if (
            self.recorder
            and self.config.record_enabled
            and self.config.audio_enabled
            and self.config.mux_audio
            and self.audio_source_for_mux
        ):
            if not self.audio_path:
                self.audio_path = self._prepare_audio_for_mux(self.audio_source_for_mux)
            if self.audio_path:
                muxed = self.recorder.mux_audio(str(self.audio_path))
                if muxed:
                    self.last_recording_path = muxed

        if (
            self.config.write_metadata
            and self.last_recording_path
            and self.last_recording_path.exists()
        ):
            metadata = build_metadata(
                source=str(self.config.source),
                output_path=self.last_recording_path,
                title=getattr(self.extraction_info, "title", None),
                info=self.extraction_info,
            )
            self.last_metadata_path = write_metadata(metadata, self.last_recording_path)

    def _terminate_ffmpeg(self, proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _prepare_audio_for_mux(self, source: str) -> Optional[Path]:
        """Prepare local audio file for muxing."""
        if source.startswith(("http://", "https://")):
            return AudioDownloader.download_youtube_audio(source)
        src_path = Path(source)
        if src_path.exists():
            if not shutil.which("ffmpeg"):
                print("ffmpeg not available for audio extraction.")
                return None
            temp_audio = Path("/tmp") / "glyph_forge_audio.m4a"
            import subprocess

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(src_path),
                "-vn",
                "-acodec",
                "aac",
                "-loglevel",
                "error",
                str(temp_audio),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and temp_audio.exists():
                return temp_audio
        return None

    def _select_audio_source_for_mux(self, info) -> Optional[str]:
        if info and info.audio_url:
            audio_url = str(info.audio_url)
            if not audio_url.startswith(("http://", "https://")) and Path(audio_url).exists():
                return audio_url
        source = self.config.source
        if isinstance(source, str) and YouTubeExtractor.is_youtube_url(source):
            return source
        if info and info.audio_url:
            return str(info.audio_url)
        return None

    def _play_recording(self) -> None:
        """Play rendered recording with ffplay."""
        if not self.recorder:
            return
        output = self.recorder.config.output_path
        if not output.exists():
            print("Recording not found for playback.")
            return
        if not shutil.which("ffplay"):
            print("ffplay not available for playback.")
            return
        import subprocess

        cmd = ["ffplay", "-autoexit", "-loglevel", "quiet", str(output)]
        try:
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"Playback failed: {e}")


def stream(source: str | int, **kwargs) -> Optional[Path]:
    """Convenience wrapper for UnifiedStreamEngine."""
    config = UnifiedStreamConfig(source=source, **kwargs)
    engine = UnifiedStreamEngine(config)
    return engine.run()
