"""Bounded, ephemeral assets for local Glyph Forge sharing.

The store deliberately knows nothing about HTTP. Browser snapshots are retained
in bounded memory while large media remains file-backed, so publishing a video
does not duplicate it or scale memory usage with its size.
"""

from __future__ import annotations

import mimetypes
import os
import re
import secrets
import stat
import threading
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{20,128}$")
_MEDIA_TYPE_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*$"
)


class ShareError(RuntimeError):
    """Base error for ephemeral sharing."""


class ShareLimitError(ShareError):
    """Raised when an asset exceeds a configured safety limit."""


class ShareUnavailableError(ShareError):
    """Raised when a shared asset has expired or changed on disk."""


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    """Properties used to reject a replaced or modified shared file."""

    device: int
    inode: int
    size: int
    modified_ns: int

    @classmethod
    def from_path(cls, path: Path) -> "FileFingerprint":
        details = path.stat()
        if not stat.S_ISREG(details.st_mode):
            raise ShareError(f"Only regular files can be shared: {path}")
        return cls(
            device=details.st_dev,
            inode=details.st_ino,
            size=details.st_size,
            modified_ns=details.st_mtime_ns,
        )

    @classmethod
    def from_stream(cls, stream: BinaryIO) -> "FileFingerprint":
        details = os.fstat(stream.fileno())
        if not stat.S_ISREG(details.st_mode):
            raise ShareError("Only regular files can be shared")
        return cls(
            device=details.st_dev,
            inode=details.st_ino,
            size=details.st_size,
            modified_ns=details.st_mtime_ns,
        )


@dataclass(frozen=True, slots=True)
class ShareAsset:
    """One immutable capability-addressed memory or file asset."""

    token: str
    filename: str
    media_type: str
    size: int
    created_at: float
    expires_at: float
    deadline: float
    data: bytes | None = None
    path: Path | None = None
    fingerprint: FileFingerprint | None = None

    @property
    def is_file(self) -> bool:
        """Return whether this asset streams from a file."""

        return self.path is not None

    def open_file(self) -> BinaryIO:
        """Open a shared file only if it still matches its publication snapshot."""

        if self.path is None or self.fingerprint is None:
            raise ShareUnavailableError("The shared asset is not file-backed")
        descriptor: int | None = None
        stream: BinaryIO | None = None
        try:
            flags = (
                os.O_RDONLY
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            descriptor = os.open(self.path, flags)
            stream = os.fdopen(descriptor, "rb")
            descriptor = None
            current = FileFingerprint.from_stream(stream)
        except (OSError, ShareError) as exc:
            if descriptor is not None:
                os.close(descriptor)
            if stream is not None:
                stream.close()
            raise ShareUnavailableError(
                "The shared file is no longer available"
            ) from exc
        if current != self.fingerprint:
            stream.close()
            raise ShareUnavailableError(
                "The shared file changed after its link was created"
            )
        return stream


def safe_filename(value: str) -> str:
    """Return a portable, header-safe leaf filename."""

    leaf = value.replace("\\", "/").rsplit("/", 1)[-1]
    normalized = unicodedata.normalize("NFKC", leaf)
    cleaned = "".join(
        character
        if character.isalnum() or character in {" ", ".", "_", "-", "(", ")"}
        else "_"
        for character in normalized
        if character >= " " and character != "\x7f"
    )
    cleaned = " ".join(cleaned.split()).strip(" .")
    return cleaned[:160] or "glyph-forge-output"


def media_type_for(filename: str, supplied: str | None = None) -> str:
    """Choose a safe MIME type from an explicit value or filename."""

    media_type = (supplied or mimetypes.guess_type(filename)[0] or "").strip()
    if not _MEDIA_TYPE_PATTERN.fullmatch(media_type):
        return "application/octet-stream"
    return media_type.casefold()


class EphemeralShareStore:
    """Thread-safe, capability-addressed store with hard resource bounds."""

    def __init__(
        self,
        *,
        default_ttl: float = 3600.0,
        max_ttl: float = 86400.0,
        max_items: int = 32,
        max_memory_bytes: int = 64 * 1024 * 1024,
        max_upload_bytes: int = 16 * 1024 * 1024,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        if default_ttl <= 0 or max_ttl <= 0 or default_ttl > max_ttl:
            raise ValueError("Share TTLs must be positive and default_ttl <= max_ttl")
        if max_items < 1:
            raise ValueError("max_items must be at least one")
        if max_memory_bytes < 1 or max_upload_bytes < 1:
            raise ValueError("Share byte limits must be positive")
        if max_upload_bytes > max_memory_bytes:
            raise ValueError("max_upload_bytes cannot exceed max_memory_bytes")
        self.default_ttl = float(default_ttl)
        self.max_ttl = float(max_ttl)
        self.max_items = max_items
        self.max_memory_bytes = max_memory_bytes
        self.max_upload_bytes = max_upload_bytes
        self._clock = clock
        self._wall_clock = wall_clock
        self._token_factory = token_factory or (lambda: secrets.token_urlsafe(24))
        self._assets: OrderedDict[str, ShareAsset] = OrderedDict()
        self._memory_bytes = 0
        self._lock = threading.RLock()

    @property
    def memory_bytes(self) -> int:
        """Return retained in-memory bytes after pruning expired entries."""

        with self._lock:
            self._prune_locked(self._clock())
            return self._memory_bytes

    def __len__(self) -> int:
        with self._lock:
            self._prune_locked(self._clock())
            return len(self._assets)

    def _duration(self, ttl: float | None) -> float:
        duration = self.default_ttl if ttl is None else float(ttl)
        if duration <= 0 or duration > self.max_ttl:
            raise ValueError(
                f"Share TTL must be between 0 and {self.max_ttl:g} seconds"
            )
        return duration

    def _new_token_locked(self) -> str:
        for _attempt in range(32):
            token = self._token_factory()
            if not _TOKEN_PATTERN.fullmatch(token):
                raise ShareError("Token factory returned an unsafe capability token")
            if token not in self._assets:
                return token
        raise ShareError("Could not allocate a unique share token")

    def _drop_locked(self, token: str) -> None:
        asset = self._assets.pop(token, None)
        if asset is not None and asset.data is not None:
            self._memory_bytes -= asset.size

    def _prune_locked(self, now: float) -> None:
        for token, asset in tuple(self._assets.items()):
            if asset.deadline <= now:
                self._drop_locked(token)

    def _make_room_locked(self, incoming_memory: int) -> None:
        while len(self._assets) >= self.max_items:
            self._drop_locked(next(iter(self._assets)))
        while self._memory_bytes + incoming_memory > self.max_memory_bytes:
            memory_token = next(
                (
                    token
                    for token, asset in self._assets.items()
                    if asset.data is not None
                ),
                None,
            )
            if memory_token is None:
                raise ShareLimitError("The in-memory share limit has been reached")
            self._drop_locked(memory_token)

    def publish_bytes(
        self,
        data: bytes,
        filename: str,
        *,
        media_type: str | None = None,
        ttl: float | None = None,
    ) -> ShareAsset:
        """Publish immutable bytes, evicting the oldest bounded entries if needed."""

        payload = bytes(data)
        if len(payload) > self.max_upload_bytes:
            raise ShareLimitError(
                f"Browser share exceeds the {self.max_upload_bytes}-byte upload limit"
            )
        now = self._clock()
        wall_now = self._wall_clock()
        duration = self._duration(ttl)
        with self._lock:
            self._prune_locked(now)
            token = self._new_token_locked()
            self._make_room_locked(len(payload))
            asset = ShareAsset(
                token=token,
                filename=safe_filename(filename),
                media_type=media_type_for(filename, media_type),
                size=len(payload),
                created_at=wall_now,
                expires_at=wall_now + duration,
                deadline=now + duration,
                data=payload,
            )
            self._assets[token] = asset
            self._memory_bytes += asset.size
            return asset

    def publish_file(
        self,
        path: str | Path,
        *,
        filename: str | None = None,
        media_type: str | None = None,
        ttl: float | None = None,
    ) -> ShareAsset:
        """Publish one regular file by reference without loading it into memory."""

        try:
            resolved = Path(path).expanduser().resolve(strict=True)
            fingerprint = FileFingerprint.from_path(resolved)
        except OSError as exc:
            raise ShareError(f"Could not inspect shared file: {path}") from exc
        now = self._clock()
        wall_now = self._wall_clock()
        duration = self._duration(ttl)
        display_name = safe_filename(filename or resolved.name)
        with self._lock:
            self._prune_locked(now)
            token = self._new_token_locked()
            self._make_room_locked(0)
            asset = ShareAsset(
                token=token,
                filename=display_name,
                media_type=media_type_for(display_name, media_type),
                size=fingerprint.size,
                created_at=wall_now,
                expires_at=wall_now + duration,
                deadline=now + duration,
                path=resolved,
                fingerprint=fingerprint,
            )
            self._assets[token] = asset
            return asset

    def get(self, token: str) -> ShareAsset:
        """Resolve an active token or raise without revealing other assets."""

        if not _TOKEN_PATTERN.fullmatch(token):
            raise ShareUnavailableError("Unknown or expired share link")
        with self._lock:
            self._prune_locked(self._clock())
            asset = self._assets.get(token)
            if asset is None:
                raise ShareUnavailableError("Unknown or expired share link")
            if asset.path is not None and asset.fingerprint is not None:
                try:
                    current = FileFingerprint.from_path(asset.path)
                except (OSError, ShareError):
                    self._drop_locked(token)
                    raise ShareUnavailableError(
                        "The shared file is no longer available"
                    ) from None
                if current != asset.fingerprint:
                    self._drop_locked(token)
                    raise ShareUnavailableError(
                        "The shared file changed after its link was created"
                    )
            return asset

    def discard(self, token: str) -> None:
        """Revoke one share link if it exists."""

        with self._lock:
            self._drop_locked(token)

    def clear(self) -> None:
        """Revoke every link and release all retained browser snapshots."""

        with self._lock:
            self._assets.clear()
            self._memory_bytes = 0


__all__ = [
    "EphemeralShareStore",
    "FileFingerprint",
    "ShareAsset",
    "ShareError",
    "ShareLimitError",
    "ShareUnavailableError",
    "media_type_for",
    "safe_filename",
]
