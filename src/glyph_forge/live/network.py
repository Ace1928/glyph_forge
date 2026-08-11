"""Optional network-media resolution for live glyph playback."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlparse

from ..runtime import python_install_hint


class NetworkSourceError(RuntimeError):
    """Raised when a remote media page cannot be resolved safely."""


@dataclass(frozen=True, slots=True)
class ResolvedNetworkSource:
    """A direct video stream plus useful display metadata."""

    url: str
    title: str
    webpage_url: str
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None


def is_network_url(value: str) -> bool:
    """Return whether ``value`` is an HTTP(S) URL with a host."""

    parsed = urlparse(value)
    return parsed.scheme.casefold() in {"http", "https"} and bool(parsed.netloc)


def _video_format(info: dict[str, Any]) -> dict[str, Any]:
    if info.get("url") and info.get("vcodec") != "none":
        return info
    for item in info.get("requested_formats") or ():
        if item.get("url") and item.get("vcodec") != "none":
            return cast(dict[str, Any], item)
    for item in reversed(info.get("formats") or ()):
        if item.get("url") and item.get("vcodec") != "none":
            return cast(dict[str, Any], item)
    raise NetworkSourceError("The page did not expose a playable video stream")


def resolve_network_source(
    url: str,
    *,
    max_height: int = 720,
) -> ResolvedNetworkSource:
    """Resolve a supported page URL to one direct, video-bearing stream.

    ``yt-dlp`` is imported only for this operation, keeping local workflows and
    command discovery lightweight.  No media is downloaded to disk.
    """

    if not is_network_url(url):
        raise ValueError("Network sources must use an http:// or https:// URL")
    if max_height < 1:
        raise ValueError("max_height must be positive")
    try:
        yt_dlp = importlib.import_module("yt_dlp")
    except (ImportError, OSError) as exc:
        raise NetworkSourceError(
            f"Network video pages require yt-dlp; {python_install_hint('network')}"
        ) from exc

    options = {
        "format": (
            f"best[height<=?{max_height}][vcodec!=none]/"
            f"bestvideo[height<=?{max_height}]/best"
        ),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            extracted = downloader.extract_info(url, download=False)
    except Exception as exc:
        raise NetworkSourceError(f"Could not resolve network media: {exc}") from exc
    if not isinstance(extracted, dict):
        raise NetworkSourceError("The network extractor returned no media information")
    if extracted.get("_type") == "playlist":
        entries = [item for item in extracted.get("entries") or () if item]
        if not entries:
            raise NetworkSourceError("The playlist contains no playable entries")
        entry = entries[0]
        if not isinstance(entry, dict):
            raise NetworkSourceError("The playlist entry contains no media information")
        extracted = entry

    selected = _video_format(extracted)
    return ResolvedNetworkSource(
        url=str(selected["url"]),
        title=str(extracted.get("title") or selected.get("format_note") or url),
        webpage_url=str(extracted.get("webpage_url") or url),
        duration=_optional_float(extracted.get("duration")),
        width=_optional_int(selected.get("width") or extracted.get("width")),
        height=_optional_int(selected.get("height") or extracted.get("height")),
        fps=_optional_float(selected.get("fps") or extracted.get("fps")),
    )


def _optional_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


__all__ = [
    "NetworkSourceError",
    "ResolvedNetworkSource",
    "is_network_url",
    "resolve_network_source",
]
