"""Small crash-safe persistence primitives shared by every interface."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


class AtomicWriteError(OSError):
    """A destination could not be replaced with a complete new payload."""


def atomic_write_bytes(
    destination: str | os.PathLike[str],
    payload: bytes,
    *,
    permissions: int | None = None,
) -> Path:
    """Durably replace a file without exposing a partially written payload."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    target = Path(destination).expanduser()
    temporary: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if permissions is not None and os.name == "posix":
            os.chmod(temporary, permissions)
        os.replace(temporary, target)
        temporary = None
        if os.name == "posix":
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except OSError as exc:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise AtomicWriteError(f"Could not save {target}: {exc}") from exc
    return target


def atomic_write_text(
    destination: str | os.PathLike[str],
    content: str,
    *,
    encoding: str = "utf-8",
    permissions: int | None = None,
) -> Path:
    """Encode and atomically persist text."""

    if not isinstance(content, str):
        raise TypeError("content must be a string")
    return atomic_write_bytes(
        destination,
        content.encode(encoding),
        permissions=permissions,
    )


__all__ = ["AtomicWriteError", "atomic_write_bytes", "atomic_write_text"]
