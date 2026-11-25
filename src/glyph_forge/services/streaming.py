"""Glyph stream integration helpers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union


class StreamCommandError(RuntimeError):
    """Raised when stream command execution fails."""


def _get_stream_command() -> list[str]:
    """Get the command to invoke the glyph stream CLI."""
    return [sys.executable, "-m", "glyph_forge.cli", "stream"]


def _run_subprocess(command: Sequence[str], check: bool = False) -> int:
    """Run glyph stream command and return the exit code."""
    cmd = [str(c) for c in command if c is not None]
    try:
        # Filter out None values just in case
        result = subprocess.run(cmd, check=check)
    except Exception as exc:
        raise StreamCommandError(f"Failed to run: {' '.join(cmd)}") from exc
    return result.returncode


def _build_common_args(
    *,
    fps: Optional[int] = None,
    mode: Optional[str] = None,
    color: bool = True,
    audio: bool = True,
    record: bool = False,
    stats: bool = True,
    resolution: str = "720p",
) -> list[str]:
    args: list[str] = []

    if fps is not None:
        args += ["--fps", str(fps)]

    args += ["--mode", mode or "gradient"]

    # Map color boolean to string
    args += ["--color", "ansi256" if color else "none"]

    if not audio:
        args.append("--no-audio")

    if record:
        args += ["--record", "auto"]

    if not stats:
        args.append("--no-stats")

    args += ["--resolution", resolution]

    return args


def run_stream(args: Optional[Sequence[str]] = None, check: bool = False) -> int:
    """Execute the stream CLI with the provided arguments."""
    command = _get_stream_command()
    if args:
        command.extend(args)
    return _run_subprocess(command, check=check)


def stream_source(
    source: Union[str, Path],
    *,
    fps: Optional[int] = None,
    mode: Optional[str] = None,
    color: bool = True,
    audio: bool = True,
    record: bool = False,
    resolution: str = "720p",
    extra_args: Optional[Iterable[str]] = None,
) -> int:
    """Stream a video file or URL."""
    args = [str(source)]
    args += _build_common_args(
        fps=fps,
        mode=mode,
        color=color,
        audio=audio,
        record=record,
        resolution=resolution,
    )
    if extra_args:
        args.extend(list(extra_args))
    return run_stream(args)


def stream_webcam(
    device: int = 0,
    *,
    fps: Optional[int] = None,
    mode: Optional[str] = None,
    color: bool = True,
    audio: bool = False,
    record: bool = False,
    resolution: str = "720p",
    extra_args: Optional[Iterable[str]] = None,
) -> int:
    """Stream a webcam feed."""
    args = ["--webcam", str(device)]
    args += _build_common_args(
        fps=fps,
        mode=mode,
        color=color,
        audio=audio,
        record=record,
        resolution=resolution,
    )
    if extra_args:
        args.extend(list(extra_args))
    return run_stream(args)


def stream_youtube(
    url: str,
    *,
    fps: Optional[int] = None,
    mode: Optional[str] = None,
    color: bool = True,
    audio: bool = True,
    record: bool = False,
    resolution: str = "720p",
    extra_args: Optional[Iterable[str]] = None,
) -> int:
    """Stream a YouTube URL."""
    return stream_source(
        url,
        fps=fps,
        mode=mode,
        color=color,
        audio=audio,
        record=record,
        resolution=resolution,
        extra_args=extra_args,
    )


def stream_virtual_display(
    *,
    display_size: Optional[Union[str, tuple[int, int]]] = None,
    launch_command: Optional[Union[str, Sequence[str]]] = None,
    fps: Optional[int] = None,
    mode: Optional[str] = None,
    color: bool = True,
    resolution: str = "720p",
    extra_args: Optional[Iterable[str]] = None,
) -> int:
    """Stream a virtual display (placeholder - not fully supported in new engine yet)."""
    # Note: New engine supports --screen, but launch_command logic is complex.
    # We map this to --screen for now.
    args: list[str] = ["--screen"]

    args += _build_common_args(
        fps=fps,
        mode=mode,
        color=color,
        resolution=resolution,
    )
    if extra_args:
        args.extend(list(extra_args))
    return run_stream(args)


def stream_browser(
    url: str,
    *,
    browser: str = "chromium",
    kiosk: bool = True,
    display_size: Optional[Union[str, tuple[int, int]]] = (1280, 720),
    fps: Optional[int] = None,
    color: bool = True,
    browser_args: Optional[Sequence[str]] = None,
    extra_args: Optional[Iterable[str]] = None,
) -> int:
    """Launch a browser in a virtual display and stream it."""
    # This functionality is specific to the old script's virtual display capabilities.
    # The new engine handles --screen, but launching a browser is outside its scope currently.
    # We will invoke the NEW engine with --screen, assuming user launches browser manually or we add a helper.
    print("Warning: Auto-launching browser is deprecated. Please launch browser manually.")
    return stream_virtual_display(
        display_size=display_size,
        fps=fps,
        color=color,
        extra_args=extra_args,
    )


__all__ = [
    "StreamCommandError",
    "run_stream",
    "stream_source",
    "stream_webcam",
    "stream_youtube",
    "stream_virtual_display",
    "stream_browser",
]
