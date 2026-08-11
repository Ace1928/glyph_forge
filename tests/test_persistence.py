"""Crash-safe shared persistence behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from glyph_forge import persistence
from glyph_forge.persistence import (
    AtomicWriteError,
    atomic_copy_file,
    atomic_write_bytes,
    atomic_write_text,
)


def test_atomic_text_replaces_existing_content_without_temp_files(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "nested" / "result.txt"
    destination.parent.mkdir()
    destination.write_text("old", encoding="utf-8")

    result = atomic_write_text(destination, "new ✓")

    assert result == destination
    assert destination.read_text(encoding="utf-8") == "new ✓"
    assert not list(destination.parent.glob(".*.tmp"))


def test_atomic_binary_write_preserves_destination_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "result.bin"
    destination.write_bytes(b"old")
    monkeypatch.setattr(
        persistence.os,
        "replace",
        lambda *_: (_ for _ in ()).throw(OSError("interrupted")),
    )

    with pytest.raises(AtomicWriteError, match="interrupted"):
        atomic_write_bytes(destination, b"new")

    assert destination.read_bytes() == b"old"
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_writers_reject_the_wrong_payload_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="bytes"):
        atomic_write_bytes(tmp_path / "bad", "text")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="string"):
        atomic_write_text(tmp_path / "bad", b"bytes")  # type: ignore[arg-type]


def test_atomic_copy_streams_files_and_preserves_existing_data_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "nested" / "copy.bin"
    source.write_bytes(b"source" * 1000)

    assert atomic_copy_file(source, destination) == destination
    assert destination.read_bytes() == source.read_bytes()
    monkeypatch.setattr(
        persistence.os,
        "replace",
        lambda *_: (_ for _ in ()).throw(OSError("interrupted")),
    )
    source.write_bytes(b"changed")

    with pytest.raises(AtomicWriteError, match="interrupted"):
        atomic_copy_file(source, destination)

    assert destination.read_bytes() == b"source" * 1000
    assert not list(destination.parent.glob(".*.tmp"))


def test_atomic_copy_rejects_missing_or_identical_sources(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"data")
    with pytest.raises(AtomicWriteError, match="must be different"):
        atomic_copy_file(source, source)
    with pytest.raises(AtomicWriteError, match="regular file"):
        atomic_copy_file(tmp_path / "missing", tmp_path / "target")
