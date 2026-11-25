"""Audio muxing utilities for Glyph Forge."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .core.sync import AudioDownloader


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def download_audio(source: str, output_path: Optional[Path] = None) -> Optional[Path]:
    """Download audio from a URL (YouTube supported via yt-dlp)."""
    return AudioDownloader.download_youtube_audio(source, output_path=output_path)


def mux_audio(
    video_path: Path,
    audio_source: str,
    output_path: Optional[Path] = None,
    offset_seconds: float = 0.0,
    overwrite: bool = False,
) -> Optional[Path]:
    """Mux an audio track into a video file.

    Args:
        video_path: Path to video file.
        audio_source: Audio file path or URL (YouTube supported).
        output_path: Optional output path.
        offset_seconds: Offset audio by seconds (positive delays audio).
        overwrite: Overwrite output if it exists.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not has_ffmpeg():
        raise RuntimeError("ffmpeg not available")

    temp_audio: Optional[Path] = None
    audio_path = Path(audio_source)
    if audio_source.startswith(("http://", "https://")):
        temp_audio = download_audio(audio_source)
        if temp_audio is None:
            raise RuntimeError("Failed to download audio from URL")
        audio_path = temp_audio
    elif not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    if output_path is None:
        output_path = video_path.with_stem(video_path.stem + "_with_audio")

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}")

    cmd = ["ffmpeg"]
    if overwrite:
        cmd.append("-y")
    else:
        cmd.append("-n")
    cmd.extend(["-i", str(video_path)])
    if offset_seconds:
        cmd.extend(["-itsoffset", str(offset_seconds)])
    cmd.extend(["-i", str(audio_path)])
    cmd.extend(
        [
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-loglevel",
            "error",
            str(output_path),
        ]
    )

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg mux failed")

    if temp_audio and temp_audio.exists():
        temp_audio.unlink()

    if output_path.exists():
        return output_path
    return None
