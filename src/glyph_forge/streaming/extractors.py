"""Video/Audio source extraction for streaming.

This module handles extraction of video and audio streams from various sources
including YouTube, local files, webcams, and network streams.

Classes:
    YouTubeExtractor: Extract streams from YouTube URLs
    VideoSourceExtractor: Unified source extraction interface
    StreamExtractionError: Extraction failure exception
    DependencyError: Missing dependency exception
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse

try:
    import yt_dlp

    HAS_YT_DLP = True
except ImportError:
    yt_dlp = None
    HAS_YT_DLP = False

try:
    import cv2

    HAS_OPENCV = True
except ImportError:
    cv2 = None
    HAS_OPENCV = False


def _parse_cookies_from_browser_spec(
    spec: Optional[str],
) -> Optional[Tuple[str, Optional[str], Optional[str], Optional[str]]]:
    if not spec or not isinstance(spec, str):
        return None
    keyring: Optional[str] = None
    browser_part = spec
    if "+" in spec:
        browser_part, keyring = spec.split("+", 1)
    parts = browser_part.split(":", 2)
    browser = parts[0]
    profile = parts[1] if len(parts) > 1 and parts[1] else None
    container = parts[2] if len(parts) > 2 and parts[2] else None
    return (browser, profile, keyring, container)


def _parse_player_client_spec(spec: Optional[str]) -> Optional[list[str]]:
    if not spec or not isinstance(spec, str):
        return None
    clients = [item.strip() for item in spec.split(",") if item.strip()]
    return clients or None


# ═══════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════


class StreamExtractionError(Exception):
    """Stream extraction error with rich diagnostic context.

    Provides categorized error information with original exception tracking
    for intelligent recovery strategies and detailed error reporting.
    """

    def __init__(
        self,
        message: str,
        original: Optional[Exception] = None,
        category: str = "general",
    ) -> None:
        super().__init__(message)
        self.original = original
        self.category = category
        self.timestamp = datetime.now()

    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic information."""
        return {
            "message": str(self),
            "category": self.category,
            "timestamp": self.timestamp.isoformat(),
            "original_type": type(self.original).__name__ if self.original else None,
            "original_message": str(self.original) if self.original else None,
        }


class DependencyError(Exception):
    """Missing dependency error with actionable installation guidance."""

    def __init__(
        self,
        package: str,
        install_cmd: str,
        required_for: str = "",
    ) -> None:
        feature_info = f" (required for {required_for})" if required_for else ""
        super().__init__(f"Missing dependency: {package}{feature_info}")
        self.package = package
        self.install_cmd = install_cmd
        self.required_for = required_for

    def get_installation_instructions(self) -> str:
        """Get user-friendly installation instructions."""
        return f"Install {self.package}: {self.install_cmd}"


# ═══════════════════════════════════════════════════════════════
# Extraction Result Types
# ═══════════════════════════════════════════════════════════════


@dataclass
class ExtractionResult:
    """Result of stream extraction with metadata."""

    video_url: Optional[str] = None
    audio_url: Optional[str] = None
    title: str = "Unknown"
    duration: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: str = "unknown"
    is_live: bool = False

    @property
    def has_video(self) -> bool:
        return self.video_url is not None

    @property
    def has_audio(self) -> bool:
        return self.audio_url is not None


# ═══════════════════════════════════════════════════════════════
# YouTube Extractor
# ═══════════════════════════════════════════════════════════════


class YouTubeExtractor:
    """High-performance YouTube stream extractor with caching.

    Extracts video and audio streams from YouTube URLs with intelligent
    caching, format selection, and error recovery.
    """

    _cache: Dict[str, Tuple[ExtractionResult, float]] = {}
    _cache_ttl: int = 3600  # 1 hour
    _cache_max_size: int = 100

    # YouTube URL patterns
    YOUTUBE_PATTERNS = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
    ]

    @classmethod
    def is_youtube_url(cls, url: str) -> bool:
        """Check if URL is a valid YouTube URL."""
        return any(re.match(pattern, url) for pattern in cls.YOUTUBE_PATTERNS)

    @classmethod
    def extract_video_id(cls, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        for pattern in cls.YOUTUBE_PATTERNS:
            match = re.match(pattern, url)
            if match:
                return match.group(1)
        return None

    @classmethod
    def extract(
        cls,
        url: str,
        resolution: Optional[int] = None,
        include_audio: bool = True,
        yt_cookies: Optional[str] = None,
        yt_cookies_from_browser: Optional[str] = None,
        yt_user_agent: Optional[str] = None,
        yt_proxy: Optional[str] = None,
        yt_skip_authcheck: bool = False,
        yt_player_client: Optional[str] = None,
    ) -> ExtractionResult:
        """Extract video and audio streams from YouTube URL.

        Args:
            url: YouTube URL or video ID
            resolution: Preferred vertical resolution (None=auto)
            include_audio: Whether to extract audio stream

        Returns:
            ExtractionResult: Extracted stream information

        Raises:
            DependencyError: If yt-dlp is not available
            StreamExtractionError: If extraction fails
        """
        if not HAS_YT_DLP:
            raise DependencyError(
                "yt-dlp",
                "pip install yt-dlp",
                "YouTube streaming",
            )

        # Check cache
        cache_key = f"{url}:{resolution}:{include_audio}"
        current_time = time.time()
        cls._prune_cache(current_time)

        if cache_key in cls._cache:
            result, timestamp = cls._cache[cache_key]
            if current_time - timestamp < cls._cache_ttl:
                return result

        # Determine optimal resolution
        actual_resolution = resolution or cls._determine_optimal_resolution()

        # Video format: prefer direct https MP4 for OpenCV compatibility
        video_format = (
            f"best[height<={actual_resolution}][protocol=https][ext=mp4]/"
            f"best[height<={actual_resolution}][protocol=https]/"
            f"best[protocol=https]"
        )

        # Audio format: prefer best audio quality
        audio_format = "bestaudio[ext=m4a]/bestaudio"

        result = ExtractionResult()

        # Extract video stream
        cookies_from_browser = _parse_cookies_from_browser_spec(yt_cookies_from_browser)
        player_clients = _parse_player_client_spec(yt_player_client)
        video_opts: Dict[str, Any] = {
            "format": video_format,
            "quiet": True,
            "skip_download": True,
            "no_warnings": True,
            "socket_timeout": 15,
        }
        if yt_cookies:
            video_opts["cookiefile"] = yt_cookies
        if cookies_from_browser:
            video_opts["cookiesfrombrowser"] = cookies_from_browser
        if player_clients:
            video_opts["extractor_args"] = {"youtube": {"player_client": player_clients}}
        if yt_user_agent:
            video_opts["user_agent"] = yt_user_agent
        if yt_proxy:
            video_opts["proxy"] = yt_proxy

        for retry in range(3):
            try:
                with yt_dlp.YoutubeDL(video_opts) as ydl:
                    info = ydl.extract_info(url, download=False)

                    if info:
                        result.video_url = info.get("url")
                        result.title = info.get("title", "Unknown")
                        result.duration = info.get("duration")
                        result.fps = info.get("fps")
                        result.width = info.get("width")
                        result.height = info.get("height")
                        result.format = info.get("format", "unknown")
                        result.is_live = info.get("is_live", False)
                        break

            except Exception as e:
                if retry < 2:
                    time.sleep(1 * (2**retry))
                    # Fallback to lower quality
                    video_opts["format"] = (
                        "best[height<=360][protocol=https]/worst[protocol=https]/worst"
                    )
                else:
                    raise StreamExtractionError(
                        f"Failed to extract video stream: {e}",
                        original=e,
                        category=cls._categorize_error(e),
                    )

        # Extract audio stream separately if requested
        if include_audio and not result.is_live:
            audio_opts: Dict[str, Any] = {
                "format": audio_format,
                "quiet": True,
                "skip_download": True,
                "no_warnings": True,
                "socket_timeout": 15,
            }
            if yt_cookies:
                audio_opts["cookiefile"] = yt_cookies
            if cookies_from_browser:
                audio_opts["cookiesfrombrowser"] = cookies_from_browser
            if player_clients:
                audio_opts["extractor_args"] = {"youtube": {"player_client": player_clients}}
            if yt_user_agent:
                audio_opts["user_agent"] = yt_user_agent
            if yt_proxy:
                audio_opts["proxy"] = yt_proxy

            try:
                with yt_dlp.YoutubeDL(audio_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    if info:
                        result.audio_url = info.get("url")
            except Exception:
                # Audio extraction failure is non-fatal
                pass

        # Cache successful result
        cls._cache[cache_key] = (result, current_time)

        return result

    @classmethod
    def _prune_cache(cls, current_time: float) -> None:
        """Remove expired and excess cache entries."""
        expired = [k for k, (_, ts) in cls._cache.items() if current_time - ts > cls._cache_ttl]
        for key in expired:
            cls._cache.pop(key, None)

        if len(cls._cache) > cls._cache_max_size:
            sorted_items = sorted(
                cls._cache.items(),
                key=lambda item: item[1][1],
                reverse=True,
            )
            cls._cache = dict(sorted_items[: cls._cache_max_size])

    @staticmethod
    def _determine_optimal_resolution() -> int:
        """Determine optimal resolution based on terminal size."""
        try:
            term_size = os.get_terminal_size()
            if term_size.lines > 60:
                return 720
            elif term_size.lines > 40:
                return 480
            else:
                return 360
        except Exception:
            return 480

    @staticmethod
    def _categorize_error(error: Exception) -> str:
        """Categorize extraction errors for retry strategies."""
        error_str = str(error).lower()

        if "429" in error_str:
            return "rate_limited"
        elif "404" in error_str or "not found" in error_str:
            return "not_found"
        elif "private" in error_str:
            return "private"
        elif "age" in error_str or "restricted" in error_str:
            return "age_restricted"
        elif "timeout" in error_str or "timed out" in error_str:
            return "timeout"
        elif "network" in error_str or "connection" in error_str:
            return "network"
        else:
            return "general"


# ═══════════════════════════════════════════════════════════════
# Unified Video Source Extractor
# ═══════════════════════════════════════════════════════════════


class VideoSourceExtractor:
    """Unified interface for extracting video from any source.

    Handles YouTube URLs, local files, webcams, and network streams
    with automatic source type detection and appropriate extraction.
    """

    @classmethod
    def extract(
        cls,
        source: Union[str, int, Path],
        resolution: Optional[int] = None,
        include_audio: bool = True,
        playlist_max_items: Optional[int] = None,
        playlist_start: Optional[int] = None,
        playlist_end: Optional[int] = None,
        yt_cookies: Optional[str] = None,
        yt_cookies_from_browser: Optional[str] = None,
        yt_user_agent: Optional[str] = None,
        yt_proxy: Optional[str] = None,
        yt_skip_authcheck: bool = False,
        yt_player_client: Optional[str] = None,
    ) -> ExtractionResult:
        """Extract video/audio from any supported source.

        Args:
            source: Video source (URL, file path, webcam index)
            resolution: Preferred resolution for streaming sources
            include_audio: Whether to extract audio

        Returns:
            ExtractionResult: Extracted stream information
        """
        # Handle webcam
        if isinstance(source, int):
            return cls._extract_webcam(source)

        source_str = str(source)

        # Handle YouTube
        if YouTubeExtractor.is_youtube_url(source_str):
            if "list=" in source_str:
                return cls._extract_youtube_playlist(
                    source_str,
                    resolution,
                    include_audio,
                    playlist_max_items=playlist_max_items,
                    playlist_start=playlist_start,
                    playlist_end=playlist_end,
                    yt_cookies=yt_cookies,
                    yt_cookies_from_browser=yt_cookies_from_browser,
                    yt_user_agent=yt_user_agent,
                    yt_proxy=yt_proxy,
                    yt_skip_authcheck=yt_skip_authcheck,
                    yt_player_client=yt_player_client,
                )
            return YouTubeExtractor.extract(
                source_str,
                resolution=resolution,
                include_audio=include_audio,
                yt_cookies=yt_cookies,
                yt_cookies_from_browser=yt_cookies_from_browser,
                yt_user_agent=yt_user_agent,
                yt_proxy=yt_proxy,
                yt_skip_authcheck=yt_skip_authcheck,
                yt_player_client=yt_player_client,
            )

        # Handle local file
        if os.path.isfile(source_str):
            return cls._extract_local_file(source_str, include_audio)

        # Handle network stream (RTSP, HTTP, etc.)
        if source_str.startswith(("rtsp://", "http://", "https://")):
            return ExtractionResult(
                video_url=source_str,
                title=os.path.basename(source_str),
                format="network_stream",
            )

        raise StreamExtractionError(
            f"Unsupported source type: {source}",
            category="unsupported",
        )

    @classmethod
    def _extract_webcam(cls, index: int) -> ExtractionResult:
        """Extract webcam stream info."""
        if not HAS_OPENCV:
            raise DependencyError(
                "opencv-python",
                "pip install opencv-python",
                "webcam capture",
            )

        # Probe webcam for capabilities
        cap = cv2.VideoCapture(index)
        try:
            if not cap.isOpened():
                raise StreamExtractionError(
                    f"Cannot open webcam {index}",
                    category="device_error",
                )

            return ExtractionResult(
                video_url=str(index),
                title=f"Webcam {index}",
                fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None,
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None,
                format="webcam",
                is_live=True,
            )
        finally:
            cap.release()

    @classmethod
    def _extract_local_file(
        cls,
        path: str,
        include_audio: bool = True,
    ) -> ExtractionResult:
        """Extract local video file info."""
        if not HAS_OPENCV:
            raise DependencyError(
                "opencv-python",
                "pip install opencv-python",
                "video playback",
            )

        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                raise StreamExtractionError(
                    f"Cannot open video file: {path}",
                    category="file_error",
                )

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else None

            result = ExtractionResult(
                video_url=path,
                title=os.path.basename(path),
                duration=duration,
                fps=fps if fps > 0 else None,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None,
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None,
                format="local_file",
            )

            # Audio URL is same as video for local files (ffmpeg will extract)
            if include_audio:
                result.audio_url = path

            return result
        finally:
            cap.release()

    @classmethod
    def _extract_youtube_playlist(
        cls,
        url: str,
        resolution: Optional[int],
        include_audio: bool,
        playlist_max_items: Optional[int] = None,
        playlist_start: Optional[int] = None,
        playlist_end: Optional[int] = None,
        yt_cookies: Optional[str] = None,
        yt_cookies_from_browser: Optional[str] = None,
        yt_user_agent: Optional[str] = None,
        yt_proxy: Optional[str] = None,
        yt_skip_authcheck: bool = False,
        yt_player_client: Optional[str] = None,
    ) -> ExtractionResult:
        """Download and stitch a YouTube playlist into a single file."""
        if not HAS_YT_DLP:
            raise DependencyError(
                "yt-dlp",
                "pip install yt-dlp",
                "YouTube playlist streaming",
            )
        if not shutil.which("ffmpeg"):
            raise DependencyError(
                "ffmpeg",
                "Install ffmpeg (required for playlist stitching)",
                "YouTube playlist streaming",
            )
        playlist_title = None
        try:
            cookies_from_browser = _parse_cookies_from_browser_spec(yt_cookies_from_browser)
            player_clients = _parse_player_client_spec(yt_player_client)
            info_opts: Dict[str, Any] = {
                "quiet": True,
                "skip_download": True,
                "extract_flat": True,
            }
            if yt_cookies:
                info_opts["cookiefile"] = yt_cookies
            if cookies_from_browser:
                info_opts["cookiesfrombrowser"] = cookies_from_browser
            extractor_args: Dict[str, Any] = {}
            if player_clients:
                extractor_args["youtube"] = {"player_client": player_clients}
            if yt_user_agent:
                info_opts["user_agent"] = yt_user_agent
            if yt_proxy:
                info_opts["proxy"] = yt_proxy
            if yt_skip_authcheck:
                extractor_args.setdefault("youtubetab", {})["skip"] = ["authcheck"]
            if extractor_args:
                info_opts["extractor_args"] = extractor_args
            with yt_dlp.YoutubeDL(info_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info:
                    playlist_title = info.get("title")
        except Exception:
            playlist_title = None

        playlist_id = None
        try:
            parsed = urlparse(url)
            playlist_id = parse_qs(parsed.query).get("list", [None])[0]
        except Exception:
            playlist_id = None
        if not playlist_id:
            playlist_id = "playlist"
        playlist_id = re.sub(r"[^a-zA-Z0-9_-]+", "", playlist_id) or "playlist"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = Path(tempfile.gettempdir()) / f"glyph_forge_playlist_{playlist_id}_{stamp}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_pattern = str(temp_dir / "%(playlist_index)03d_%(id)s.%(ext)s")

        cmd = [
            "yt-dlp",
            "-f",
            "bestvideo*+bestaudio/best",
            "--merge-output-format",
            "mp4",
            "--no-part",
            "-o",
            output_pattern,
            "--yes-playlist",
            "--no-playlist-reverse",
            url,
        ]
        if yt_cookies:
            cmd.extend(["--cookies", yt_cookies])
        if yt_cookies_from_browser:
            cmd.extend(["--cookies-from-browser", yt_cookies_from_browser])
        if yt_user_agent:
            cmd.extend(["--user-agent", yt_user_agent])
        if yt_proxy:
            cmd.extend(["--proxy", yt_proxy])
        extractor_args: list[str] = []
        if yt_skip_authcheck:
            extractor_args.append("youtubetab:skip=authcheck")
        if yt_player_client:
            extractor_args.append(f"youtube:player_client={yt_player_client}")
        for arg in extractor_args:
            cmd.extend(["--extractor-args", arg])
        items = None
        if playlist_start or playlist_end:
            start = playlist_start or 1
            end = playlist_end or ""
            items = f"{start}-{end}"
        elif playlist_max_items:
            items = f"1-{playlist_max_items}"
        if items:
            cmd.extend(["--playlist-items", items])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 and "HTTP Error 413" in (result.stderr or ""):
            if "youtubetab:skip=authcheck" not in extractor_args:
                cmd.extend(["--extractor-args", "youtubetab:skip=authcheck"])
            result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise StreamExtractionError(
                f"Playlist download failed: {result.stderr.strip()}",
                category="download",
            )

        files = sorted(temp_dir.glob("*.mp4"))
        if not files:
            raise StreamExtractionError("Playlist download failed", category="download")

        def _run_ffmpeg(cmd: list[str], category: str) -> None:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise StreamExtractionError(
                    proc.stderr.strip() or "ffmpeg failed", category=category
                )

        audio_files: list[Path] = []
        for video in files:
            audio_path = video.with_suffix(".m4a")
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(video),
                "-vn",
                "-c:a",
                "copy",
                "-loglevel",
                "error",
                str(audio_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(video),
                    "-vn",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-loglevel",
                    "error",
                    str(audio_path),
                ]
                _run_ffmpeg(cmd, "audio_extract")
            if not audio_path.exists():
                raise StreamExtractionError("Audio extraction failed", category="audio_extract")
            audio_files.append(audio_path)

        video_concat = temp_dir / "video_concat.txt"
        with video_concat.open("w", encoding="utf-8") as f:
            for file in files:
                f.write(f"file '{file.as_posix()}'\n")

        stitched_video = temp_dir / "playlist_video.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(video_concat),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-loglevel",
            "error",
            str(stitched_video),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(video_concat),
                "-map",
                "0:v:0",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-loglevel",
                "error",
                str(stitched_video),
            ]
            _run_ffmpeg(cmd, "video_concat")

        audio_concat = temp_dir / "playlist_audio.m4a"
        audio_concat_list = temp_dir / "audio_concat.txt"
        with audio_concat_list.open("w", encoding="utf-8") as f:
            for audio in audio_files:
                f.write(f"file '{audio.as_posix()}'\n")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(audio_concat_list),
            "-c",
            "copy",
            "-loglevel",
            "error",
            str(audio_concat),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(audio_concat_list),
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-loglevel",
                "error",
                str(audio_concat),
            ]
            _run_ffmpeg(cmd, "audio_concat")

        stitched = temp_dir / "playlist_merged.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(stitched_video),
            "-i",
            str(audio_concat),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-loglevel",
            "error",
            str(stitched),
        ]
        _run_ffmpeg(cmd, "mux")

        result = cls._extract_local_file(str(stitched), include_audio)
        if playlist_title:
            result.title = playlist_title
        if include_audio and audio_concat.exists():
            result.audio_url = str(audio_concat)
            result.format = "playlist"
        return result


# ═══════════════════════════════════════════════════════════════
# Audio Extractor
# ═══════════════════════════════════════════════════════════════


class AudioExtractor:
    """Extract and download audio for synchronized playback.

    Uses ffmpeg to extract audio streams to temporary files for
    synchronized playback with video rendering.
    """

    @staticmethod
    def has_ffmpeg() -> bool:
        """Check if ffmpeg is available."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
            )
            return True
        except Exception:
            return False

    @classmethod
    def extract_to_file(
        cls,
        source: str,
        output_path: Optional[str] = None,
        start_time: float = 0,
    ) -> Optional[str]:
        """Extract audio to a temporary file.

        Args:
            source: Audio source URL or path
            output_path: Output path (None=auto temp file)
            start_time: Start time offset in seconds

        Returns:
            str: Path to extracted audio file, or None on failure
        """
        if not cls.has_ffmpeg():
            return None

        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        try:
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite
                "-ss",
                str(start_time),
                "-i",
                source,
                "-vn",  # No video
                "-acodec",
                "pcm_s16le",
                "-ar",
                "44100",
                "-ac",
                "2",
                output_path,
            ]

            subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                timeout=60,
            )

            return output_path
        except Exception:
            if os.path.exists(output_path):
                os.unlink(output_path)
            return None
