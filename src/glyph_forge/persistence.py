"""Small crash-safe persistence primitives shared by every interface."""

from __future__ import annotations

import os
import shutil
import stat
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


def atomic_copy_file(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> Path:
    """Stream a regular file into place without exposing a partial copy."""

    original = Path(source).expanduser()
    target = Path(destination).expanduser()
    temporary: Path | None = None
    try:
        if not original.is_file():
            raise OSError(f"source is not a regular file: {original}")
        if original.resolve() == target.resolve():
            raise OSError("source and destination must be different")
        source_mode = stat.S_IMODE(original.stat().st_mode)
        target.parent.mkdir(parents=True, exist_ok=True)
        with original.open("rb") as input_stream, tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as output_stream:
            temporary = Path(output_stream.name)
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if os.name == "posix":
            os.chmod(temporary, source_mode)
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
        raise AtomicWriteError(f"Could not copy {original} to {target}: {exc}") from exc
    return target


__all__ = [
    "AtomicWriteError",
    "atomic_copy_file",
    "atomic_write_bytes",
    "atomic_write_text",
]
